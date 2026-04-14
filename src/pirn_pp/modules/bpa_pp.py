from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from pirn_pp.config import PIRNPPConfig
from pirn_pp.utils import l2_normalize


class SinkhornRouter(nn.Module):
    """
    Partial/unbalanced OT-style router with a dustbin channel.
    Ambiguous tokens can assign mass to the dustbin instead of normal prototypes.
    """

    def __init__(self, tau: float, iters: int, mass_floor: float) -> None:
        super().__init__()
        self.tau = tau
        self.iters = iters
        self.mass_floor = mass_floor

    def forward(self, token_proto_sim: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_proto_sim: [B, N, K]
        Returns:
            assign: [B, N, K]
            confidence: [B, N]
        """
        b, n, k = token_proto_sim.shape
        dustbin = token_proto_sim.new_zeros((b, n, 1))
        scores = torch.cat([token_proto_sim, dustbin], dim=-1) / self.tau
        q = scores.exp()

        row_target = token_proto_sim.new_full((b, n), 1.0 / n)
        col_main = (1.0 - self.mass_floor) / max(k, 1)
        col_target = token_proto_sim.new_full((b, k + 1), col_main)
        col_target[:, -1] = self.mass_floor

        for _ in range(self.iters):
            q = q / (q.sum(dim=-1, keepdim=True) + 1e-8)
            q = q * row_target.unsqueeze(-1)
            q = q / (q.sum(dim=-2, keepdim=True) + 1e-8)
            q = q * col_target.unsqueeze(-2)

        q = q / (q.sum(dim=-1, keepdim=True) + 1e-8)
        assign = q[..., :k]
        confidence = 1.0 - q[..., -1]
        return assign, confidence


class BPAPlusPlus(nn.Module):
    """Confidence-aware bidirectional prototype assignment for RGB/XYZ features."""

    def __init__(self, cfg: PIRNPPConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.router = SinkhornRouter(cfg.sinkhorn_tau, cfg.sinkhorn_iters, cfg.assign_mass_floor)
        self.proto_rgb = nn.Parameter(torch.randn(cfg.num_proto_rgb, cfg.dim))
        self.proto_xyz = nn.Parameter(torch.randn(cfg.num_proto_xyz, cfg.dim))
        nn.init.normal_(self.proto_rgb, std=0.02)
        nn.init.normal_(self.proto_xyz, std=0.02)

    def forward(self, f_rgb: torch.Tensor, f_xyz: torch.Tensor) -> Dict[str, torch.Tensor]:
        p_rgb = l2_normalize(self.proto_rgb, dim=-1)
        p_xyz = l2_normalize(self.proto_xyz, dim=-1)
        x_rgb = l2_normalize(f_rgb, dim=-1)
        x_xyz = l2_normalize(f_xyz, dim=-1)

        sim_rgb = torch.einsum("bnd,kd->bnk", x_rgb, p_rgb)
        sim_xyz = torch.einsum("bnd,kd->bnk", x_xyz, p_xyz)

        assign_rgb, conf_rgb = self.router(sim_rgb)
        assign_xyz, conf_xyz = self.router(sim_xyz)

        z_rgb = torch.einsum("bnk,kd->bnd", assign_rgb, p_rgb)
        z_xyz = torch.einsum("bnk,kd->bnd", assign_xyz, p_xyz)

        # Lightweight cross-modal harmonization proxy.
        h_rgb = torch.einsum("bnk,bmk->bnm", assign_rgb, assign_rgb).mean()
        h_xyz = torch.einsum("bnk,bmk->bnm", assign_xyz, assign_xyz).mean()
        harmonization = 0.5 * (h_rgb + h_xyz)

        return {
            "z_rgb": z_rgb,
            "z_xyz": z_xyz,
            "assign_rgb": assign_rgb,
            "assign_xyz": assign_xyz,
            "conf_rgb": conf_rgb,
            "conf_xyz": conf_xyz,
            "harmonization": harmonization,
            "sim_rgb": sim_rgb,
            "sim_xyz": sim_xyz,
        }

