from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pirn_pp.config import PIRNPPConfig


class SequentialIBBlock(nn.Module):
    """Sequential cross-modal exchange with information bottleneck regularization proxy."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q_a = nn.Linear(dim, dim)
        self.k_b = nn.Linear(dim, dim)
        self.v_b = nn.Linear(dim, dim)

        self.q_b = nn.Linear(dim, dim)
        self.k_a = nn.Linear(dim, dim)
        self.v_a = nn.Linear(dim, dim)

        self.gate_a = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.gate_b = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.proj_a = nn.Linear(dim, dim)
        self.proj_b = nn.Linear(dim, dim)

    def _attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        score = torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
        attn = F.softmax(score, dim=-1)
        return torch.matmul(attn, v)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a_to_b = self._attn(self.q_a(a), self.k_b(b), self.v_b(b))
        gate_a = self.gate_a(a)
        a_out = self.proj_a(gate_a * a_to_b + (1.0 - gate_a) * a)

        b_to_a = self._attn(self.q_b(b), self.k_a(a_out), self.v_a(a_out))
        gate_b = self.gate_b(b)
        b_out = self.proj_b(gate_b * b_to_a + (1.0 - gate_b) * b)

        ib_loss = 0.5 * ((a_to_b - a).pow(2).mean() + (b_to_a - b).pow(2).mean())
        return a_out, b_out, ib_loss


class MNCPlusPlus(nn.Module):
    """Sequential information-bottleneck multimodal normality communication."""

    def __init__(self, cfg: PIRNPPConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([SequentialIBBlock(cfg.dim) for _ in range(cfg.mnc_steps)])
        self.fuse = nn.Linear(cfg.dim * 2, cfg.dim)

    def forward(self, z_rgb: torch.Tensor, z_xyz: torch.Tensor) -> Dict[str, torch.Tensor]:
        a, b = z_rgb, z_xyz
        ib_total = z_rgb.new_tensor(0.0)
        for block in self.blocks:
            a, b, ib = block(a, b)
            ib_total = ib_total + ib
        fused = self.fuse(torch.cat([a, b], dim=-1))
        return {"fused": fused, "ib_loss": ib_total / max(len(self.blocks), 1)}

