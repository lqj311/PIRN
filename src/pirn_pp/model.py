from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pirn_pp.config import PIRNPPConfig
from pirn_pp.modules.apr_pp import APRPlusPlus
from pirn_pp.modules.bpa_pp import BPAPlusPlus
from pirn_pp.modules.mnc_pp import MNCPlusPlus


class PIRNPlusPlus(nn.Module):
    """End-to-end PIRN++ model wrapper."""

    def __init__(self, cfg: Optional[PIRNPPConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or PIRNPPConfig()
        self.bpa = BPAPlusPlus(self.cfg)
        self.apr = APRPlusPlus(self.cfg)
        self.mnc = MNCPlusPlus(self.cfg)
        self.head_anomaly = nn.Sequential(
            nn.LayerNorm(self.cfg.dim),
            nn.Linear(self.cfg.dim, self.cfg.dim),
            nn.GELU(),
            nn.Linear(self.cfg.dim, 1),
        )

    def forward(
        self, f_rgb: torch.Tensor, f_xyz: torch.Tensor, update_teacher: bool = True
    ) -> Dict[str, torch.Tensor]:
        bpa_out = self.bpa(f_rgb, f_xyz)
        apr_out = self.apr(
            proto_rgb=self.bpa.proto_rgb,
            proto_xyz=self.bpa.proto_xyz,
            z_rgb=bpa_out["z_rgb"],
            z_xyz=bpa_out["z_xyz"],
            conf_rgb=bpa_out["conf_rgb"],
            conf_xyz=bpa_out["conf_xyz"],
            sim_rgb=bpa_out["sim_rgb"],
            sim_xyz=bpa_out["sim_xyz"],
            update_teacher=update_teacher,
        )

        with torch.no_grad():
            self.bpa.proto_rgb.copy_(apr_out["proto_rgb_refined"])
            self.bpa.proto_xyz.copy_(apr_out["proto_xyz_refined"])

        mnc_out = self.mnc(bpa_out["z_rgb"], bpa_out["z_xyz"])
        anomaly_logit = self.head_anomaly(mnc_out["fused"]).squeeze(-1)

        return {
            "anomaly_logit": anomaly_logit,
            "harmonization": bpa_out["harmonization"],
            "consistency_loss": apr_out["consistency_loss"],
            "entropy_reg": apr_out["entropy_reg"],
            "ib_loss": mnc_out["ib_loss"],
            "conf_rgb": bpa_out["conf_rgb"],
            "conf_xyz": bpa_out["conf_xyz"],
        }

    def compute_loss(
        self, out: Dict[str, torch.Tensor], target: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        if target is not None:
            losses["bce"] = F.binary_cross_entropy_with_logits(out["anomaly_logit"], target.float())
        else:
            losses["bce"] = out["anomaly_logit"].mean() * 0.0

        losses["proto_align"] = (1.0 - out["harmonization"]) * self.cfg.proto_align_weight
        losses["consistency"] = out["consistency_loss"] * self.cfg.consistency_weight
        losses["entropy_reg"] = out["entropy_reg"] * self.cfg.entropy_reg_weight
        losses["ib"] = out["ib_loss"] * self.cfg.ib_beta
        losses["total"] = (
            losses["bce"]
            + losses["proto_align"]
            + losses["consistency"]
            + losses["entropy_reg"]
            + losses["ib"]
        )
        return losses

