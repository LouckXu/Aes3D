from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, dropout: float) -> nn.Sequential:
    """(Linear -> GELU -> Dropout) x (num_layers - 1) -> Linear."""
    layers = []
    dim = in_dim
    for _ in range(num_layers - 1):
        layers += [nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = torch.softmax(logits.masked_fill(~mask, torch.finfo(logits.dtype).min), dim=dim)
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    denom = weights.sum(dim=dim, keepdim=True)
    return torch.where(mask.any(dim=dim, keepdim=True), weights / denom.clamp_min(1.0e-6), torch.zeros_like(weights))


def _keep_one_key(valid: torch.Tensor) -> torch.Tensor:
    """Rows without any valid key attend to all keys instead of producing NaNs."""
    return torch.where(valid.any(dim=1, keepdim=True), valid, torch.ones_like(valid))


class TransformerBlock(nn.Module):
    """Pre-LN multi-head self-attention + MLP."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = _mlp(dim, int(dim * mlp_ratio), dim, num_layers=2, dropout=dropout)

    def forward(self, tokens: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        key_padding_mask = None if valid is None else ~_keep_one_key(valid)
        h = self.norm1(tokens)
        tokens = tokens + self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return tokens + self.ffn(self.norm2(tokens))


# ---------------------------------------------------------------------------
# Sec. 4.1  Gaussian Scene Encoding
# ---------------------------------------------------------------------------


class SceneEncoder(nn.Module):
    """Enc_3D: h_i^0 = phi_in(x_i) + phi_pos(p_i) with x_i = [p_i, c_i, p_i / |p_i|], then a point Transformer."""

    def __init__(self, dim: int = 192, depth: int = 4, num_heads: int = 4, mlp_ratio: float = 2.0,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.phi_in = nn.Linear(9, dim)
        self.phi_pos = _mlp(3, dim, dim, num_layers=2, dropout=dropout)
        self.blocks = nn.ModuleList([TransformerBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, coords: torch.Tensor, colors: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
        direction = coords / torch.linalg.norm(coords, dim=-1, keepdim=True).clamp_min(1.0e-6)
        h = self.phi_in(torch.cat([coords, colors, direction], dim=-1)) + self.phi_pos(coords)
        for block in self.blocks:
            h = block(h, point_mask)
        scene_tokens = self.norm(h)
        return torch.where(point_mask.unsqueeze(-1), scene_tokens, torch.zeros_like(scene_tokens))  # T: [B, N, D]


class SceneGlobalPooling(nn.Module):
    """Single-query attention pooling of the scene tokens T into the scene-global token s."""

    def __init__(self, dim: int = 192, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, scene_tokens: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
        query = self.q_norm(self.query.expand(scene_tokens.shape[0], -1, -1))
        kv = self.kv_norm(scene_tokens)
        pooled = self.attn(query, kv, kv, key_padding_mask=~_keep_one_key(point_mask), need_weights=False)[0]
        return self.norm(pooled.squeeze(1))  # s: [B, D]


# ---------------------------------------------------------------------------
# Sec. 4.2 / App. I  Projection-based View Tokenization
# ---------------------------------------------------------------------------


def view_geometry(extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """[R^T e_x; R^T e_y; R^T e_z; o_v; log(1 + f_x); log(1 + f_y); c_x; c_y] -> [..., 16], o_v = -R^T t."""
    rotation_t = extrinsics[..., :3, :3].transpose(-1, -2)
    camera_center = -(rotation_t @ extrinsics[..., :3, 3:4]).squeeze(-1)
    focal = torch.log1p(torch.stack([intrinsics[..., 0, 0], intrinsics[..., 1, 1]], dim=-1).clamp_min(1.0e-6))
    principal = torch.stack([intrinsics[..., 0, 2], intrinsics[..., 1, 2]], dim=-1)
    return torch.cat([rotation_t[..., :, 0], rotation_t[..., :, 1], rotation_t[..., :, 2], camera_center,
                      focal, principal], dim=-1)


class ProjectionViewTokenizer(nn.Module):
    """u_iv = K_v Pi(E_v p_i) on a G x G patch grid; View Pooling b_{v,m} = Mean + Max (e_empty for empty
    cells); r_{v,m} = phi_cell(b_{v,m}) + a_v with a_v = phi_geom(view geometry); a 2-block view Transformer
    (empty cells masked) and a learnable-query summary give one view descriptor v_v per candidate view."""

    def __init__(self, dim: int = 192, grid_size: int = 14, depth: int = 2, num_heads: int = 4,
                 mlp_ratio: float = 2.0, dropout: float = 0.1, min_depth: float = 1.0e-4) -> None:
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        self.min_depth = min_depth
        self.empty_cell_token = nn.Parameter(torch.zeros(dim))                           # e_empty
        self.phi_cell = _mlp(dim, dim, dim, num_layers=2, dropout=dropout)
        self.phi_geom = _mlp(16, dim, dim, num_layers=2, dropout=dropout)
        # Learned position embedding of the G x G patch grid, part of the view Transformer.
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, 1, self.num_patches, dim))
        self.view_transformer = nn.ModuleList(
            [TransformerBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.summary_query = nn.Parameter(torch.zeros(1, 1, dim))
        self.summary_q_norm = nn.LayerNorm(dim)
        self.summary_kv_norm = nn.LayerNorm(dim)
        self.summary_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.summary_norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.empty_cell_token, std=0.02)
        nn.init.trunc_normal_(self.patch_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.summary_query, std=0.02)

    def view_pooling(self, scene_tokens: torch.Tensor, coords: torch.Tensor, point_mask: torch.Tensor,
                     view_mask: torch.Tensor, extrinsics: torch.Tensor,
                     intrinsics: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """b_{v,m} for every patch cell: [B, V, G*G, D], and the occupancy mask [B, V, G*G]."""
        batch_size, _, dim = scene_tokens.shape
        num_views = view_mask.shape[1]
        grid, num_patches = self.grid_size, self.num_patches

        camera_coords = (torch.einsum("bvij,bnj->bvni", extrinsics[:, :, :3, :3], coords)
                         + extrinsics[:, :, :3, 3][:, :, None, :])                          # E_v p~_i
        depth = camera_coords[..., 2]
        xy = camera_coords[..., :2] / depth.unsqueeze(-1).clamp_min(self.min_depth)        # Pi(.)
        u = intrinsics[:, :, None, 0, 0] * xy[..., 0] + intrinsics[:, :, None, 0, 2]        # K_v Pi(.)
        v = intrinsics[:, :, None, 1, 1] * xy[..., 1] + intrinsics[:, :, None, 1, 2]
        col = torch.floor(u * grid).long()
        row = torch.floor(v * grid).long()
        valid = (point_mask[:, None, :] & view_mask[:, :, None] & (depth > self.min_depth)
                 & (col >= 0) & (col < grid) & (row >= 0) & (row < grid)).reshape(-1)
        cell = row.clamp(0, grid - 1) * grid + col.clamp(0, grid - 1)
        offset = torch.arange(batch_size * num_views, device=cell.device).view(batch_size, num_views, 1) * num_patches
        slot = (cell + offset).reshape(-1)

        total = batch_size * num_views * num_patches
        cell_sum = scene_tokens.new_zeros((total, dim))
        cell_count = scene_tokens.new_zeros((total, 1))
        cell_max = torch.full((total, dim), torch.finfo(scene_tokens.dtype).min, device=scene_tokens.device,
                              dtype=scene_tokens.dtype)
        if valid.any():
            source = scene_tokens[:, None].expand(-1, num_views, -1, -1).reshape(-1, dim)[valid]
            index = slot[valid]
            cell_sum.index_add_(0, index, source)
            cell_count.index_add_(0, index, torch.ones_like(source[:, :1]))
            cell_max.scatter_reduce_(0, index[:, None].expand(-1, dim), source, reduce="amax", include_self=True)
        occupied = cell_count.squeeze(-1) > 0
        b = cell_sum / cell_count.clamp_min(1.0) + cell_max                                 # Mean + Max
        b = torch.where(occupied[:, None], b, self.empty_cell_token.to(b.dtype).expand(total, -1))
        return b.view(batch_size, num_views, num_patches, dim), occupied.view(batch_size, num_views, num_patches)

    def forward(self, scene_tokens: torch.Tensor, coords: torch.Tensor, point_mask: torch.Tensor,
                view_mask: torch.Tensor, extrinsics: torch.Tensor,
                intrinsics: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_views = view_mask.shape
        view_context = self.phi_geom(view_geometry(extrinsics, intrinsics))                 # a_v: [B, V, D]
        b, occupied = self.view_pooling(scene_tokens, coords, point_mask, view_mask, extrinsics, intrinsics)
        patch_tokens = self.phi_cell(b) + view_context.unsqueeze(2)                         # r_{v,m}
        patch_tokens = (patch_tokens + self.patch_pos_embed).reshape(batch_size * num_views, self.num_patches, self.dim)
        cell_valid = occupied.reshape(batch_size * num_views, self.num_patches)
        for block in self.view_transformer:
            patch_tokens = block(patch_tokens, cell_valid)

        query = self.summary_q_norm(self.summary_query.expand(batch_size * num_views, -1, -1))
        kv = self.summary_kv_norm(patch_tokens)
        pooled = self.summary_attn(query, kv, kv, key_padding_mask=~_keep_one_key(cell_valid), need_weights=False)[0]
        view_descriptors = self.summary_norm(pooled.squeeze(1)).view(batch_size, num_views, self.dim)
        view_descriptors = torch.where(view_mask.unsqueeze(-1), view_descriptors, torch.zeros_like(view_descriptors))
        return view_descriptors, occupied  # v_v: [B, V, D], [B, V, G*G]


# ---------------------------------------------------------------------------
# Sec. 4.3  Candidate View Selection
# ---------------------------------------------------------------------------


class ViewSelector(nn.Module):
    """Z = [phi_s(s), c_1..c_M, z_1..z_V] with z_v = phi_sel(v_v); u_v = psi(z'_v); alpha_v from Eq. (3).

    Hard top-K runs on the detached utilities (straight-through) and the softmax of the selected views is
    renormalized.
    """

    def __init__(self, dim: int = 192, depth: int = 2, num_heads: int = 4, mlp_ratio: float = 2.0,
                 dropout: float = 0.1, num_control_tokens: int = 2, top_k: int = 8, tau: float = 1.0) -> None:
        super().__init__()
        self.top_k = top_k
        self.tau = tau
        self.phi_sel = _mlp(dim, dim, dim, num_layers=2, dropout=dropout)
        self.phi_s = _mlp(dim, dim, dim, num_layers=2, dropout=dropout)
        self.control_tokens = nn.Parameter(torch.zeros(1, num_control_tokens, dim))       # c_1..c_M
        nn.init.trunc_normal_(self.control_tokens, std=0.02)
        self.transformer = nn.ModuleList([TransformerBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.psi = _mlp(dim, dim, 1, num_layers=2, dropout=dropout)                       # utility head

    def forward(self, view_descriptors: torch.Tensor, scene_global_token: torch.Tensor,
                view_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, num_views, _ = view_descriptors.shape
        prefix = torch.cat([self.phi_s(scene_global_token).unsqueeze(1),
                            self.control_tokens.expand(batch_size, -1, -1)], dim=1)
        z = torch.cat([prefix, self.phi_sel(view_descriptors)], dim=1)                     # Z
        valid = torch.cat([torch.ones(batch_size, prefix.shape[1], dtype=torch.bool, device=view_mask.device),
                           view_mask], dim=1)
        for block in self.transformer:
            z = block(z, valid)

        utility = self.psi(z[:, prefix.shape[1]:]).squeeze(-1)                             # u_v
        utility = torch.where(view_mask, utility, torch.full_like(utility, -1.0e4))
        soft = _masked_softmax(utility / self.tau, view_mask, dim=1)
        topk = utility.detach().topk(min(self.top_k, num_views), dim=1).indices
        topk_mask = torch.zeros_like(view_mask).scatter(1, topk, torch.ones_like(topk, dtype=torch.bool)) & view_mask
        selected = soft * topk_mask.to(soft.dtype)
        alpha = selected / selected.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        alpha = torch.where(topk_mask.any(dim=1, keepdim=True), alpha, soft)
        topk = torch.where(view_mask.gather(1, topk), topk, torch.full_like(topk, -1))      # -1 marks padding
        return {"utility": utility, "alpha": alpha, "topk_indices": topk}


# ---------------------------------------------------------------------------
# Sec. 4.4  Multi-view Fusion & Scene-level Regression
# ---------------------------------------------------------------------------


class MultiViewFusionRegressor(nn.Module):
    """h = sum_v alpha_v v_v,  y_hat = phi_reg(LN(h + phi_fuse(h))) with a 3-layer phi_reg."""

    def __init__(self, dim: int = 192, dropout: float = 0.1) -> None:
        super().__init__()
        self.phi_fuse = _mlp(dim, dim, dim, num_layers=2, dropout=dropout)
        self.ln = nn.LayerNorm(dim)
        self.phi_reg = _mlp(dim, dim, 1, num_layers=3, dropout=dropout)

    def forward(self, view_descriptors: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        h = (view_descriptors * alpha.unsqueeze(-1)).sum(dim=1)
        return self.phi_reg(self.ln(h + self.phi_fuse(h)))  # y_hat: [B, 1]


class Aes3DGSNet(nn.Module):
    def __init__(self, dim: int = 192, encoder_depth: int = 4, num_heads: int = 4, mlp_ratio: float = 2.0,
                 dropout: float = 0.1, grid_size: int = 14, view_transformer_depth: int = 2, selector_depth: int = 2,
                 num_control_tokens: int = 2, top_k: int = 8, tau: float = 1.0) -> None:
        super().__init__()
        self.scene_encoder = SceneEncoder(dim, encoder_depth, num_heads, mlp_ratio, dropout)
        self.scene_global_pooling = SceneGlobalPooling(dim, num_heads, dropout)
        self.view_tokenizer = ProjectionViewTokenizer(dim, grid_size, view_transformer_depth, num_heads, mlp_ratio,
                                                      dropout)
        self.view_selector = ViewSelector(dim, selector_depth, num_heads, mlp_ratio, dropout, num_control_tokens,
                                          top_k, tau)
        self.fusion_regressor = MultiViewFusionRegressor(dim, dropout)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """batch: coords [B,N,3] (normalized p_i), colors [B,N,3] (RGB c_i), point_mask [B,N],
        extrinsics [B,V,4,4] (world-to-camera E_v, normalized frame), intrinsics [B,V,3,3] (normalized K_v),
        view_mask [B,V]."""
        point_mask = batch["point_mask"].bool()
        view_mask = batch["view_mask"].bool()
        scene_tokens = self.scene_encoder(batch["coords"], batch["colors"], point_mask)
        scene_global_token = self.scene_global_pooling(scene_tokens, point_mask)
        view_descriptors, occupied = self.view_tokenizer(scene_tokens, batch["coords"], point_mask, view_mask,
                                                         batch["extrinsics"], batch["intrinsics"])
        selection = self.view_selector(view_descriptors, scene_global_token, view_mask)
        return {
            "pred": self.fusion_regressor(view_descriptors, selection["alpha"]),   # y_hat: [B, 1]
            "alpha": selection["alpha"],                   # sparse top-K selection weights: [B, V]
            "utility": selection["utility"],               # u: [B, V]
            "topk_indices": selection["topk_indices"],     # [B, K], -1 for padding
            "patch_mask": occupied,                        # occupied patch cells: [B, V, G*G]
        }
