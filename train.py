from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from dataloader import TARGETS, DataConfig, build_dataloaders
from loss import AestheticsAwareLoss
from model import Aes3DGSNet

METRICS = ("plcc", "srcc", "krcc", "mae", "rmse")


def set_seed(seed: int) -> None:
    """Seed every RNG and use deterministic kernels (math attention, no TF32)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)


# ---------------------------------------------------------------------------
# Metrics: PLCC / SRCC / KRCC, and MAE / RMSE on the [0, 1] scene score
# ---------------------------------------------------------------------------


def _average_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return 0.0
    a, b = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1.0e-12 else 0.0


def _kendall(a: np.ndarray, b: np.ndarray) -> float:
    """(concordant - discordant) / (concordant + discordant), pairs with a tie on either side are skipped."""
    i, j = np.triu_indices(a.size, k=1)
    diff_a, diff_b = a[i] - a[j], b[i] - b[j]
    untied = (diff_a != 0) & (diff_b != 0)
    concordant = int(np.sum(untied & (diff_a * diff_b > 0)))
    discordant = int(np.sum(untied)) - concordant
    total = concordant + discordant
    return float((concordant - discordant) / total) if total else 0.0


def regression_metrics(pred, target) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    return {
        "plcc": _pearson(pred, target),
        "srcc": _pearson(_average_rank(pred), _average_rank(target)),
        "krcc": _kendall(pred, target),
        "mae": float(np.mean(np.abs(pred - target))),
        "rmse": float(np.sqrt(np.mean(np.square(pred - target)))),
    }


# ---------------------------------------------------------------------------
# Training (App. J.3): AdamW 5e-5 / 1e-4, batch 4, 16 epochs, cosine decay, bf16, gradient clipping 1.0
# ---------------------------------------------------------------------------


def _autocast(enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if enabled else contextlib.nullcontext()


def _to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.no_grad()
def predict(model: Aes3DGSNet, loader, device: torch.device, amp: bool) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        with _autocast(amp):
            preds.append(model(batch)["pred"].detach().float().cpu())
        labels.append(batch["label"].detach().float().cpu())
    return torch.cat(preds).view(-1).numpy(), torch.cat(labels).view(-1).numpy()


def train_one_seed(cfg: DataConfig, seed: int, epochs: int, device: torch.device, out_dir: Path) -> Dict[str, float]:
    """Train for ``epochs`` epochs, then score the final model once on the test split.

    Test labels are used neither during training nor for choosing a checkpoint.
    """
    set_seed(seed)
    train_loader, test_loader = build_dataloaders(cfg, seed)
    if not len(train_loader.dataset) or not len(test_loader.dataset):
        raise RuntimeError(f"No scenes found for {cfg.label_csv} under {cfg.reconstruction_roots}.")
    model = Aes3DGSNet().to(device)
    criterion = AestheticsAwareLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-5, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp = device.type == "cuda"
    print(f"seed {seed}: train {len(train_loader.dataset)} / test {len(test_loader.dataset)} scenes on {device}")

    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        train_loader.dataset.set_epoch(epoch)
        loss_sum, num_scenes = 0.0, 0
        for batch in train_loader:
            batch = _to_device(batch, device)
            with _autocast(amp):
                loss, stats = criterion(model(batch)["pred"], batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            loss_sum += stats["loss"] * len(batch["scene_ids"])
            num_scenes += len(batch["scene_ids"])
        scheduler.step()
        print(f"  epoch {epoch:2d}  loss {loss_sum / num_scenes:.4f}  {time.time() - start:.0f}s")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "seed": seed, "epochs": epochs, "target": cfg.target},
               out_dir / "model.pt")
    metrics = regression_metrics(*predict(model, test_loader, device, amp))
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump({"seed": seed, **metrics}, handle, indent=2)
    print("  test   " + "  ".join(f"{key.upper()} {metrics[key]:.4f}" for key in METRICS))
    return {"seed": seed, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Aes3DGSNet and report test scores (mean +- std over seeds).")
    parser.add_argument("--target", choices=sorted(TARGETS), default="8-attr")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 42])
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent,
                        help="release directory containing data/aesthetic3d_scene_scores.csv")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="directory containing dataset/<name>/reconstructions (default: --repo-root)")
    parser.add_argument("--extrinsic-convention", choices=["w2c", "c2w"], default="w2c",
                        help="how cameras.pt stores the extrinsic matrices")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    cfg = DataConfig(target=TARGETS[args.target], extrinsic_convention=args.extrinsic_convention,
                     num_workers=args.num_workers)
    data_root = args.data_root or args.repo_root
    cfg = replace(cfg, label_csv=str(args.repo_root / cfg.label_csv),
                  reconstruction_roots={name: str(data_root / root) for name, root in cfg.reconstruction_roots.items()})
    out_root = args.out or Path("runs") / args.target
    device = torch.device(args.device)

    results = [train_one_seed(cfg, seed, args.epochs, device, out_root / f"seed{seed}") for seed in args.seeds]
    summary = {"target": args.target, "epochs": args.epochs, "runs": results}
    for key in METRICS:
        values = [r[key] for r in results]
        summary[key] = {"mean": statistics.fmean(values), "std": statistics.pstdev(values) if len(values) > 1 else 0.0}
    with open(out_root / "results.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nAes3DGSNet ({args.target}): test split, final epoch")
    print("seed      " + "  ".join(f"{key.upper():>6s}" for key in METRICS))
    for r in results:
        print(f"{r['seed']:>4d}      " + "  ".join(f"{r[key]:6.3f}" for key in METRICS))
    print("mean+-std " + "  ".join(f"{summary[key]['mean']:.3f}+-{summary[key]['std']:.3f}" for key in METRICS))


if __name__ == "__main__":
    main()
