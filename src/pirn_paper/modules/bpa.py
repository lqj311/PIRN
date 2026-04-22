from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pirn_paper.config import PIRNConfig


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


class BPA(nn.Module):
    """
    Bidirectional Prototypical Assignment (BPA)
    - modality-wise balanced assignment via Sinkhorn
    - outputs reconstructed token features and assignment maps
    """

    def __init__(self, cfg: PIRNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proto_rgb = nn.Parameter(torch.randn(cfg.num_proto_rgb, cfg.dim) * 0.02)
        self.proto_sn = nn.Parameter(torch.randn(cfg.num_proto_sn, cfg.dim) * 0.02)

    def _sinkhorn_balanced(self, logits: torch.Tensor, iters: int, tau: float) -> torch.Tensor:
        """
        Args:
            logits: [B, N, K]
        Returns:
            assignment matrix [B, N, K] with approximately balanced marginals.
        """
        b, n, k = logits.shape
        q = torch.exp(logits / tau)
        q = q / (q.sum(dim=(1, 2), keepdim=True) + 1e-8)

        target_row = 1.0 / float(n)
        target_col = 1.0 / float(k)
        for _ in range(iters):
            q = q / (q.sum(dim=2, keepdim=True) + 1e-8)
            q = q * target_row
            q = q / (q.sum(dim=1, keepdim=True) + 1e-8)
            q = q * target_col

        q = q / (q.sum(dim=2, keepdim=True) + 1e-8)
        return q

    def _assign(
        self, x: torch.Tensor, proto: torch.Tensor, sinkhorn_iters: int, sinkhorn_tau: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_n = l2_normalize(x, dim=-1)
        p_n = l2_normalize(proto, dim=-1)

        sim = torch.einsum("bnd,kd->bnk", x_n, p_n)
        assign = self._sinkhorn_balanced(sim, sinkhorn_iters, sinkhorn_tau)
        z_rec = torch.einsum("bnk,kd->bnd", assign, p_n)
        return z_rec, assign, sim

    def forward(
        self,
        f_rgb: torch.Tensor,
        f_sn: torch.Tensor,
        proto_rgb: Optional[torch.Tensor] = None,
        proto_sn: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        proto_rgb = self.proto_rgb if proto_rgb is None else proto_rgb
        proto_sn = self.proto_sn if proto_sn is None else proto_sn
        z_rgb, assign_rgb, sim_rgb = self._assign(
            f_rgb, proto_rgb, self.cfg.sinkhorn_iters, self.cfg.sinkhorn_tau
        )
        z_sn, assign_sn, sim_sn = self._assign(
            f_sn, proto_sn, self.cfg.sinkhorn_iters, self.cfg.sinkhorn_tau
        )

        # Cross-modal assignment consistency (paper-style semantic alignment proxy).
        rgb_to_sn = torch.matmul(assign_rgb, assign_sn.transpose(-1, -2))
        sn_to_rgb = torch.matmul(assign_sn, assign_rgb.transpose(-1, -2))
        sem_consistency = 0.5 * (
            1.0 - rgb_to_sn.mean().clamp(min=0.0, max=1.0)
            + 1.0 - sn_to_rgb.mean().clamp(min=0.0, max=1.0)
        )

        return {
            "z_rgb": z_rgb,
            "z_sn": z_sn,
            "assign_rgb": assign_rgb,
            "assign_sn": assign_sn,
            "sim_rgb": sim_rgb,
            "sim_sn": sim_sn,
            "sem_consistency": sem_consistency,
        }
