from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from pirn_paper.config import PIRNConfig
from pirn_paper.modules.bpa import l2_normalize


class APRUnit(nn.Module):
    """
    Adaptive Prototype Refinement:
    u_k = sigmoid(Wz[p_k;c_k]+bz)
    r_k = sigmoid(Wr[p_k;c_k]+br)
    p_tilde = tanh(W[r_k ⊙ p_k; c_k]+b)
    p'_k = u_k ⊙ p_k + (1-u_k) ⊙ p_tilde
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        in_dim = dim * 2
        self.w_z = nn.Linear(in_dim, dim)
        self.w_r = nn.Linear(in_dim, dim)
        self.w_h = nn.Linear(in_dim, dim)

    def forward(self, proto: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        cat_pc = torch.cat([proto, context], dim=-1)
        u = torch.sigmoid(self.w_z(cat_pc))
        r = torch.sigmoid(self.w_r(cat_pc))
        cat_rc = torch.cat([r * proto, context], dim=-1)
        p_tilde = torch.tanh(self.w_h(cat_rc))
        return u * proto + (1.0 - u) * p_tilde


class APR(nn.Module):
    def __init__(self, cfg: PIRNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.unit_rgb = APRUnit(cfg.dim)
        self.unit_sn = APRUnit(cfg.dim)

    def _prototype_context(
        self, assign: torch.Tensor, token_feat: torch.Tensor, num_proto: int, eps: float
    ) -> torch.Tensor:
        """
        Args:
            assign: [B, N, K]
            token_feat: [B, N, D]
        Returns:
            context per prototype: [K, D]
        """
        b, n, k = assign.shape
        assert k == num_proto
        w = assign.reshape(b * n, k)
        x = token_feat.reshape(b * n, token_feat.shape[-1])
        denom = w.sum(dim=0, keepdim=True).transpose(0, 1) + eps  # [K,1]
        context = torch.matmul(w.transpose(0, 1), x) / denom  # [K,D]
        return context

    def forward(
        self,
        proto_rgb: torch.Tensor,
        proto_sn: torch.Tensor,
        assign_rgb: torch.Tensor,
        assign_sn: torch.Tensor,
        token_rgb: torch.Tensor,
        token_sn: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        c_rgb = self._prototype_context(
            assign_rgb, token_rgb, num_proto=proto_rgb.shape[0], eps=self.cfg.apr_eps
        )
        c_sn = self._prototype_context(
            assign_sn, token_sn, num_proto=proto_sn.shape[0], eps=self.cfg.apr_eps
        )

        p_rgb_new = l2_normalize(self.unit_rgb(proto_rgb, c_rgb), dim=-1)
        p_sn_new = l2_normalize(self.unit_sn(proto_sn, c_sn), dim=-1)

        return {"proto_rgb_refined": p_rgb_new, "proto_sn_refined": p_sn_new}

