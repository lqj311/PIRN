from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pirn_pp.config import PIRNPPConfig
from pirn_pp.utils import l2_normalize, normalized_entropy_from_logits


class APRPlusPlus(nn.Module):
    """
    Adaptive prototype refinement:
    - uncertainty-gated updates
    - EMA teacher consistency
    """

    def __init__(self, cfg: PIRNPPConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gru_rgb = nn.GRUCell(cfg.dim, cfg.dim)
        self.gru_xyz = nn.GRUCell(cfg.dim, cfg.dim)

        self.register_buffer("teacher_rgb", torch.empty(0))
        self.register_buffer("teacher_xyz", torch.empty(0))
        self.teacher_initialized = False

    @torch.no_grad()
    def _update_teacher(self, proto_rgb: torch.Tensor, proto_xyz: torch.Tensor) -> None:
        m = self.cfg.ema_momentum
        if not self.teacher_initialized:
            self.teacher_rgb = proto_rgb.detach().clone()
            self.teacher_xyz = proto_xyz.detach().clone()
            self.teacher_initialized = True
            return
        self.teacher_rgb.mul_(m).add_((1.0 - m) * proto_rgb.detach())
        self.teacher_xyz.mul_(m).add_((1.0 - m) * proto_xyz.detach())

    def forward(
        self,
        proto_rgb: torch.Tensor,
        proto_xyz: torch.Tensor,
        z_rgb: torch.Tensor,
        z_xyz: torch.Tensor,
        conf_rgb: torch.Tensor,
        conf_xyz: torch.Tensor,
        sim_rgb: torch.Tensor,
        sim_xyz: torch.Tensor,
        update_teacher: bool = True,
    ) -> Dict[str, torch.Tensor]:
        _, _, dim = z_rgb.shape
        k_rgb = proto_rgb.shape[0]
        k_xyz = proto_xyz.shape[0]

        ent_rgb = normalized_entropy_from_logits(sim_rgb, num_bins=k_rgb, dim=-1)
        ent_xyz = normalized_entropy_from_logits(sim_xyz, num_bins=k_xyz, dim=-1)

        reliable_rgb = (conf_rgb > 0.5) & (ent_rgb < self.cfg.update_uncertainty_threshold)
        reliable_xyz = (conf_xyz > 0.5) & (ent_xyz < self.cfg.update_uncertainty_threshold)

        if reliable_rgb.any():
            pooled_rgb = z_rgb[reliable_rgb].mean(dim=0, keepdim=True)
        else:
            pooled_rgb = proto_rgb.mean(dim=0, keepdim=True)
        if reliable_xyz.any():
            pooled_xyz = z_xyz[reliable_xyz].mean(dim=0, keepdim=True)
        else:
            pooled_xyz = proto_xyz.mean(dim=0, keepdim=True)

        pooled_rgb = pooled_rgb.expand_as(proto_rgb)
        pooled_xyz = pooled_xyz.expand_as(proto_xyz)

        refined_rgb = self.gru_rgb(pooled_rgb.reshape(-1, dim), proto_rgb.reshape(-1, dim)).view_as(proto_rgb)
        refined_xyz = self.gru_xyz(pooled_xyz.reshape(-1, dim), proto_xyz.reshape(-1, dim)).view_as(proto_xyz)
        refined_rgb = l2_normalize(refined_rgb, dim=-1)
        refined_xyz = l2_normalize(refined_xyz, dim=-1)

        if update_teacher:
            self._update_teacher(refined_rgb, refined_xyz)

        consistency = proto_rgb.new_tensor(0.0)
        if self.teacher_initialized:
            consistency = 0.5 * (
                F.mse_loss(refined_rgb, self.teacher_rgb.detach())
                + F.mse_loss(refined_xyz, self.teacher_xyz.detach())
            )

        entropy_reg = 0.5 * (ent_rgb.mean() + ent_xyz.mean())

        return {
            "proto_rgb_refined": refined_rgb,
            "proto_xyz_refined": refined_xyz,
            "consistency_loss": consistency,
            "entropy_reg": entropy_reg,
        }

