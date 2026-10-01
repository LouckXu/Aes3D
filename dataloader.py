from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SH_C0 = 0.28209479177387814          # RGB = 0.5 + SH_C0 * f_dc (zeroth-order spherical harmonics)
SCENE_FILE = "gaussians_open3d_fps_2048.npz"
CAMERA_FILE = "cameras.pt"
TARGETS = {"8-attr": "8attr_mean_score", "total": "total_score"}


@dataclass
class DataConfig:
    label_csv: str = "data/aesthetic3d_scene_scores.csv"
    target: str = TARGETS["8-attr"]   # scene-level score normalized to [0, 1]
    # Aesthetic3D = DL3DV + Bilarf; dataset name -> directory with one sub-directory per scene
    reconstruction_roots: Dict[str, str] = field(default_factory=lambda: {
        "bilarf": "dataset/bilarf_data/reconstructions",
        "dl3dv": "dataset/dl3dv/reconstructions",
    })
    test_ratio: float = 0.2
    num_views: int = 32               # V candidate cameras, for training and evaluation
    radius_quantile: float = 0.95
    # E_v = [R_v | t_v] is world-to-camera (OpenCV axes); use "c2w" when cameras.pt stores camera-to-world.
    extrinsic_convention: str = "w2c"
    batch_size: int = 4
    num_workers: int = 0


@dataclass
class SceneRecord:
    key: str            # "<dataset>:<scene_name>"
    dataset: str
    scene_name: str
    scene_path: Path
    camera_path: Path
    label: float


# ---------------------------------------------------------------------------
# Scene list and 80 / 20 split
# ---------------------------------------------------------------------------


def read_scene_records(cfg: DataConfig) -> List[SceneRecord]:
    """Label rows, in CSV order, whose Gaussians and cameras exist; other rows are skipped."""
    records = []
    with open(cfg.label_csv, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            dataset, scene_name = row["dataset"], row["scene_name"]
            if dataset not in cfg.reconstruction_roots:
                continue
            scene_dir = Path(cfg.reconstruction_roots[dataset]) / scene_name
            if not (scene_dir / SCENE_FILE).exists() or not (scene_dir / CAMERA_FILE).exists():
                continue
            records.append(SceneRecord(f"{dataset}:{scene_name}", dataset, scene_name, scene_dir / SCENE_FILE,
                                       scene_dir / CAMERA_FILE, float(row[cfg.target])))
    return records


def split_scenes(records: Sequence[SceneRecord], test_ratio: float,
                 seed: int) -> Tuple[List[SceneRecord], List[SceneRecord]]:
    """Seed-controlled split drawn separately inside each source dataset; both parts keep the CSV order."""
    keys_by_dataset: Dict[str, List[str]] = defaultdict(list)
    for record in records:
        keys_by_dataset[record.dataset].append(record.key)
    test_keys = set()
    for dataset in sorted(keys_by_dataset):
        keys = sorted(set(keys_by_dataset[dataset]))
        random.Random(f"{seed}:{dataset}").shuffle(keys)
        num_test = max(1, int(round(len(keys) * test_ratio)))
        if num_test >= len(keys):
            raise ValueError(f"The test split would leave no training scene for {dataset!r} ({len(keys)} scenes).")
        test_keys.update(keys[:num_test])
    return [r for r in records if r.key not in test_keys], [r for r in records if r.key in test_keys]


# ---------------------------------------------------------------------------
# Gaussians and cameras (App. I.1)
# ---------------------------------------------------------------------------


def stable_scene_hash(text: str) -> int:
    value = 0
    for char in text:
        value = (value * 131 + ord(char)) % (2**31 - 1)
    return value


def load_gaussians(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Centers and RGB colors of the FPS-sampled Gaussians (the first three feature columns are the SH DC)."""
    with np.load(path) as payload:
        coords = np.asarray(payload["coords"], dtype=np.float32)
        sh_dc = np.asarray(payload["feats"], dtype=np.float32)[:, :3]
    return coords, np.clip(0.5 + SH_C0 * sh_dc, 0.0, 1.0).astype(np.float32)


def normalize_scene(coords: np.ndarray, radius_quantile: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Subtract the centroid and divide by the 95th-percentile radius."""
    center = coords.mean(axis=0)
    scale = max(float(np.quantile(np.linalg.norm(coords - center, axis=1), radius_quantile)), 1.0e-4)
    return ((coords - center) / scale).astype(np.float32), center, scale


def _to_numpy(value) -> np.ndarray:
    array = np.asarray(value.detach().cpu().numpy() if torch.is_tensor(value) else value, dtype=np.float64)
    return array[0] if array.ndim == 4 and array.shape[0] == 1 else array


def load_cameras(path: Path, center: np.ndarray, scale: float,
                 convention: str = "w2c") -> Tuple[np.ndarray, np.ndarray]:
    """World-to-camera extrinsics [C, 4, 4] in the normalized scene frame and normalized intrinsics [C, 3, 3]."""
    blob = torch.load(path, map_location="cpu")
    extrinsics = _to_numpy(blob["extrinsic"])
    intrinsics = _to_numpy(blob["intrinsic"]).astype(np.float32)
    if convention == "c2w":
        extrinsics = np.linalg.inv(extrinsics)
    elif convention != "w2c":
        raise ValueError(f"extrinsic_convention must be 'w2c' or 'c2w', got {convention!r}")
    # The Gaussian normalization p' = (p - center) / scale moves the camera centers the same way:
    # R p' + t' = (R p + t) / scale  with  t' = (t + R center) / scale.
    extrinsics = extrinsics.copy()
    extrinsics[:, :3, 3] = (extrinsics[:, :3, 3] + extrinsics[:, :3, :3] @ center) / scale
    return extrinsics.astype(np.float32), intrinsics


def sample_candidate_cameras(extrinsics: np.ndarray, num_views: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform azimuth-elevation binning of the camera directions and uniform sampling across the bins.

    Returns ``num_views`` indices and a validity mask; scenes with fewer cameras keep all of them.
    """
    num_cameras = len(extrinsics)
    if num_cameras <= num_views:
        chosen = np.arange(num_cameras)
    else:
        centers = -np.einsum("cji,cj->ci", extrinsics[:, :3, :3], extrinsics[:, :3, 3])   # o_v = -R^T t
        direction = centers / np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1.0e-6, None)
        azimuth = np.mod(np.arctan2(direction[:, 1], direction[:, 0]), 2.0 * np.pi)
        elevation = np.arcsin(np.clip(direction[:, 2], -1.0, 1.0))
        num_az = int(np.ceil(np.sqrt(2.0 * num_views)))
        num_el = int(np.ceil(num_views / num_az))
        az_bin = np.minimum((azimuth / (2.0 * np.pi) * num_az).astype(np.int64), num_az - 1)
        el_bin = np.minimum(((elevation + 0.5 * np.pi) / np.pi * num_el).astype(np.int64), num_el - 1)
        bins = az_bin * num_el + el_bin
        rng = np.random.default_rng(seed)
        queues = [list(rng.permutation(np.flatnonzero(bins == b))) for b in rng.permutation(np.unique(bins))]
        picked: List[int] = []
        while len(picked) < num_views:  # one random camera per non-empty bin and round
            for queue in queues:
                if queue and len(picked) < num_views:
                    picked.append(int(queue.pop()))
        chosen = np.sort(np.asarray(picked, dtype=np.int64))
    indices = np.zeros((num_views,), dtype=np.int64)
    indices[: len(chosen)] = chosen
    valid = np.zeros((num_views,), dtype=bool)
    valid[: len(chosen)] = True
    return indices, valid


def _gather(array: np.ndarray, indices: np.ndarray, valid: np.ndarray) -> torch.Tensor:
    out = np.zeros((len(indices),) + array.shape[1:], dtype=np.float32)
    out[valid] = array[indices[valid]]
    return torch.from_numpy(out)


# ---------------------------------------------------------------------------
# Dataset, collate and loaders
# ---------------------------------------------------------------------------


class Aesthetic3DDataset(Dataset):
    def __init__(self, records: Sequence[SceneRecord], cfg: DataConfig, train: bool) -> None:
        self.records = list(records)
        self.cfg = cfg
        self.train = train
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Training re-initializes the camera-sampling seed every epoch; evaluation keeps a fixed seed."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record, cfg = self.records[index], self.cfg
        seed = stable_scene_hash(record.key)
        if self.train:
            seed = (seed + 104729 * self.epoch) % (2**31 - 1)

        coords, colors = load_gaussians(record.scene_path)
        coords, center, scale = normalize_scene(coords, cfg.radius_quantile)
        extrinsics, intrinsics = load_cameras(record.camera_path, center, scale, cfg.extrinsic_convention)
        indices, valid = sample_candidate_cameras(extrinsics, cfg.num_views, seed)
        return {
            "scene_id": record.scene_name,
            "dataset": record.dataset,
            "coords": torch.from_numpy(coords),
            "colors": torch.from_numpy(colors),
            "extrinsics": _gather(extrinsics, indices, valid),
            "intrinsics": _gather(intrinsics, indices, valid),
            "view_mask": torch.from_numpy(valid),
            "label": torch.tensor([record.label], dtype=torch.float32),
        }


def _pad(tensors: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack along a new batch dim with zero (False) padding of dim 0; also returns the validity mask."""
    max_len = max(tensor.shape[0] for tensor in tensors)
    padded = tensors[0].new_zeros((len(tensors), max_len, *tensors[0].shape[1:]))
    mask = torch.zeros((len(tensors), max_len), dtype=torch.bool)
    for i, tensor in enumerate(tensors):
        padded[i, : tensor.shape[0]] = tensor
        mask[i, : tensor.shape[0]] = True
    return padded, mask


def collate_scenes(items: Sequence[Dict[str, object]]) -> Dict[str, object]:
    def stack(key: str) -> torch.Tensor:
        return _pad([item[key] for item in items])[0]

    coords, point_mask = _pad([item["coords"] for item in items])
    return {
        "coords": coords,                                   # [B, N, 3]
        "colors": stack("colors"),                          # [B, N, 3]
        "point_mask": point_mask,                           # [B, N]
        "extrinsics": stack("extrinsics"),                  # [B, V, 4, 4]
        "intrinsics": stack("intrinsics"),                  # [B, V, 3, 3]
        "view_mask": stack("view_mask"),                    # [B, V], padded views are False
        "label": torch.stack([item["label"] for item in items]),  # [B, 1]
        "scene_ids": [item["scene_id"] for item in items],
        "datasets": [item["dataset"] for item in items],
    }


def build_dataloaders(cfg: DataConfig, seed: int) -> Tuple[DataLoader, DataLoader]:
    """Shuffled training loader and ordered test loader for one seed-controlled 80 / 20 split."""
    train_records, test_records = split_scenes(read_scene_records(cfg), cfg.test_ratio, seed)
    loader_kwargs = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, collate_fn=collate_scenes)
    train_loader = DataLoader(Aesthetic3DDataset(train_records, cfg, train=True), shuffle=True, **loader_kwargs)
    test_loader = DataLoader(Aesthetic3DDataset(test_records, cfg, train=False), shuffle=False, **loader_kwargs)
    return train_loader, test_loader
