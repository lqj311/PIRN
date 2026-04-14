from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from pirn_paper.config import PIRNConfig


class CrossModalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.rgb_q = nn.Linear(dim, dim)
        self.rgb_k = nn.Linear(dim, dim)
        self.rgb_v = nn.Linear(dim, dim)

        self.sn_q = nn.Linear(dim, dim)
        self.sn_k = nn.Linear(dim, dim)
        self.sn_v = nn.Linear(dim, dim)

        self.rgb_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.sn_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

        self.rgb_norm = nn.LayerNorm(dim)
        self.sn_norm = nn.LayerNorm(dim)
        self.rgb_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.sn_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, z_rgb: torch.Tensor, z_sn: torch.Tensor) -> Dict[str, torch.Tensor]:
        # RGB receives message from SN.
        rgb_q = self.rgb_q(z_rgb)
        sn_k = self.rgb_k(z_sn)
        sn_v = self.rgb_v(z_sn)
        rgb_msg, _ = self.rgb_attn(rgb_q, sn_k, sn_v, need_weights=False)
        z_rgb = self.rgb_norm(z_rgb + rgb_msg)
        z_rgb = self.rgb_norm(z_rgb + self.rgb_ffn(z_rgb))

        # SN receives message from updated RGB.
        sn_q = self.sn_q(z_sn)
        rgb_k = self.sn_k(z_rgb)
        rgb_v = self.sn_v(z_rgb)
        sn_msg, _ = self.sn_attn(sn_q, rgb_k, rgb_v, need_weights=False)
        z_sn = self.sn_norm(z_sn + sn_msg)
        z_sn = self.sn_norm(z_sn + self.sn_ffn(z_sn))

        return {"z_rgb": z_rgb, "z_sn": z_sn}


class MNC(nn.Module):
    """
    Multimodal Normality Communication (MNC)
    """

    def __init__(self, cfg: PIRNConfig) -> None:
        super().__init__()
        self.block = CrossModalBlock(cfg.dim, cfg.mnc_heads, cfg.mnc_dropout)

    def forward(self, z_rgb: torch.Tensor, z_sn: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.block(z_rgb, z_sn)

