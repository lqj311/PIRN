from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from pirn_paper.config import PIRNConfig
from pirn_paper.modules.bpa import l2_normalize


def normalized_entropy(probs: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    probs = probs.clamp_min(eps)
    entropy = -(probs * probs.log()).sum(dim=dim)
    num_bins = probs.shape[dim]
    if num_bins <= 1:
        return entropy.new_zeros(entropy.shape)
    return entropy / math.log(float(num_bins))


class APRUnit(nn.Module):
    """
    Adaptive Prototype Refinement:
    u_k = sigmoid(Wz[p_k;c_k]+bz)
    r_k = sigmoid(Wr[p_k;c_k]+br)
    p_tilde = tanh(W[r_k * p_k; c_k]+b)
    p'_k = u_k * p_k + (1-u_k) * p_tilde
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
    """
    Dual-memory adaptive refinement:
    - base prototypes stay in BPA and are optimized by gradient descent
    - residual memories are updated online with reliability gating
    """

    def __init__(self, cfg: PIRNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.unit_rgb = APRUnit(cfg.dim)
        self.unit_sn = APRUnit(cfg.dim)

        self.register_buffer("residual_rgb", torch.zeros(cfg.num_proto_rgb, cfg.dim))
        self.register_buffer("residual_sn", torch.zeros(cfg.num_proto_sn, cfg.dim))

    @torch.no_grad()
    def reset_memory(self) -> None:
        self.residual_rgb.zero_()
        self.residual_sn.zero_()

    def current_prototypes(
        self, base_rgb: torch.Tensor, base_sn: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        base_rgb = l2_normalize(base_rgb, dim=-1)
        base_sn = l2_normalize(base_sn, dim=-1)
        proto_rgb = l2_normalize(base_rgb + self.cfg.apr_residual_scale * self.residual_rgb, dim=-1)
        proto_sn = l2_normalize(base_sn + self.cfg.apr_residual_scale * self.residual_sn, dim=-1)
        return {"proto_rgb": proto_rgb, "proto_sn": proto_sn}

    def _token_reliability(
        self, assign: torch.Tensor, token_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        confidence = assign.max(dim=-1).values
        entropy = normalized_entropy(assign, dim=-1)
        reliable = (confidence > self.cfg.apr_confidence_threshold) & (
            entropy < self.cfg.apr_entropy_threshold
        )
        if token_mask is not None:
            reliable = reliable & token_mask
        return confidence, entropy, reliable

    def _prototype_context(
        self,
        assign: torch.Tensor,
        token_feat: torch.Tensor,
        reliable: torch.Tensor,
        num_proto: int,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            assign: [B, N, K]
            token_feat: [B, N, D]
            reliable: [B, N]
        Returns:
            context per prototype: [K, D]
            prototype support mass: [K]
        """
        b, n, k = assign.shape
        assert k == num_proto
        weights = assign * reliable.unsqueeze(-1).to(assign.dtype)
        w = weights.reshape(b * n, k)
        x = token_feat.reshape(b * n, token_feat.shape[-1])
        proto_mass = w.sum(dim=0)
        denom = proto_mass.unsqueeze(-1) + eps
        context = torch.matmul(w.transpose(0, 1), x) / denom
        return context, proto_mass

    def _refine_residual(
        self,
        unit: APRUnit,
        base_proto: torch.Tensor,
        residual_proto: torch.Tensor,
        context: torch.Tensor,
        proto_mass: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        base_proto = l2_normalize(base_proto, dim=-1)
        context_delta = context - base_proto
        residual_candidate = unit(residual_proto, context_delta)
        mean_mass = proto_mass.mean().clamp_min(self.cfg.apr_eps)
        support_gate = (proto_mass / mean_mass).clamp(min=0.0, max=1.0).unsqueeze(-1)
        refined_residual = support_gate * residual_candidate + (1.0 - support_gate) * residual_proto
        return refined_residual, support_gate.squeeze(-1)

    @torch.no_grad()
    def _update_memory(self, refined_rgb: torch.Tensor, refined_sn: torch.Tensor) -> None:
        momentum = self.cfg.apr_memory_momentum
        self.residual_rgb.mul_(momentum).add_((1.0 - momentum) * refined_rgb.detach())
        self.residual_sn.mul_(momentum).add_((1.0 - momentum) * refined_sn.detach())

    def forward(
        self,
        base_proto_rgb: torch.Tensor,
        base_proto_sn: torch.Tensor,
        assign_rgb: torch.Tensor,
        assign_sn: torch.Tensor,
        token_rgb: torch.Tensor,
        token_sn: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True,
    ) -> Dict[str, torch.Tensor]:
        _, _, reliable_rgb = self._token_reliability(assign_rgb, token_mask=token_mask)
        _, _, reliable_sn = self._token_reliability(assign_sn, token_mask=token_mask)

        context_rgb, mass_rgb = self._prototype_context(
            assign_rgb, token_rgb, reliable_rgb, num_proto=base_proto_rgb.shape[0], eps=self.cfg.apr_eps
        )
        context_sn, mass_sn = self._prototype_context(
            assign_sn, token_sn, reliable_sn, num_proto=base_proto_sn.shape[0], eps=self.cfg.apr_eps
        )

        refined_residual_rgb, gate_rgb = self._refine_residual(
            self.unit_rgb, base_proto_rgb, self.residual_rgb, context_rgb, mass_rgb
        )
        refined_residual_sn, gate_sn = self._refine_residual(
            self.unit_sn, base_proto_sn, self.residual_sn, context_sn, mass_sn
        )

        base_proto_rgb = l2_normalize(base_proto_rgb, dim=-1)
        base_proto_sn = l2_normalize(base_proto_sn, dim=-1)
        proto_rgb_new = l2_normalize(base_proto_rgb + self.cfg.apr_residual_scale * refined_residual_rgb, dim=-1)
        proto_sn_new = l2_normalize(base_proto_sn + self.cfg.apr_residual_scale * refined_residual_sn, dim=-1)

        if update_memory:
            self._update_memory(refined_residual_rgb, refined_residual_sn)

        residual_reg = 0.5 * (
            refined_residual_rgb.pow(2).mean() + refined_residual_sn.pow(2).mean()
        )
        reliable_ratio = 0.5 * (
            reliable_rgb.float().mean() + reliable_sn.float().mean()
        )
        support_ratio = 0.5 * (gate_rgb.mean() + gate_sn.mean())

        return {
            "proto_rgb_refined": proto_rgb_new,
            "proto_sn_refined": proto_sn_new,
            "residual_rgb_refined": refined_residual_rgb,
            "residual_sn_refined": refined_residual_sn,
            "residual_reg": residual_reg,
            "reliable_ratio": reliable_ratio,
            "support_ratio": support_ratio,
        }
