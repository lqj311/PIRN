from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pirn_paper.config import PIRNConfig
from pirn_paper.modules import APR, BPA, MNC
from pirn_paper.modules.bpa import l2_normalize


class PIRNModel(nn.Module):
    """
    PIRN pipeline:
    1) BPA: assignment + reconstruction
    2) APR: adaptive prototype refinement
    3) MNC: multimodal communication
    4) anomaly score from cosine discrepancy
    """

    def __init__(self, cfg: Optional[PIRNConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or PIRNConfig()
        self.bpa = BPA(self.cfg)
        self.apr = APR(self.cfg)
        self.mnc = MNC(self.cfg)

    @torch.no_grad()
    def _refresh_prototypes(self, p_rgb: torch.Tensor, p_sn: torch.Tensor) -> None:
        self.bpa.proto_rgb.copy_(p_rgb)
        self.bpa.proto_sn.copy_(p_sn)

    def forward(
        self,
        f_rgb: torch.Tensor,
        f_sn: torch.Tensor,
        update_proto: bool = True,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        bpa_out = self.bpa(f_rgb, f_sn)

        apr_out = self.apr(
            proto_rgb=self.bpa.proto_rgb,
            proto_sn=self.bpa.proto_sn,
            assign_rgb=bpa_out["assign_rgb"],
            assign_sn=bpa_out["assign_sn"],
            token_rgb=bpa_out["z_rgb"],
            token_sn=bpa_out["z_sn"],
        )

        if update_proto:
            self._refresh_prototypes(apr_out["proto_rgb_refined"], apr_out["proto_sn_refined"])

        # Reconstruct with refined prototypes.
        rec_rgb = torch.einsum("bnk,kd->bnd", bpa_out["assign_rgb"], apr_out["proto_rgb_refined"])
        rec_sn = torch.einsum("bnk,kd->bnd", bpa_out["assign_sn"], apr_out["proto_sn_refined"])

        # Cross-modal normality communication.
        mnc_out = self.mnc(rec_rgb, rec_sn)
        rec_rgb_comm = mnc_out["z_rgb"]
        rec_sn_comm = mnc_out["z_sn"]

        # Token-level anomaly map.
        rgb_dist = 1.0 - F.cosine_similarity(l2_normalize(f_rgb), l2_normalize(rec_rgb_comm), dim=-1)
        sn_dist = 1.0 - F.cosine_similarity(l2_normalize(f_sn), l2_normalize(rec_sn_comm), dim=-1)
        token_anomaly = rgb_dist + sn_dist

        if token_mask is not None:
            neg_inf = torch.finfo(token_anomaly.dtype).min
            token_anomaly_for_pool = torch.where(token_mask, token_anomaly, token_anomaly.new_full((), neg_inf))
            image_anomaly = token_anomaly_for_pool.max(dim=1).values
        else:
            image_anomaly = token_anomaly.max(dim=1).values

        return {
            "rec_rgb": rec_rgb_comm,
            "rec_sn": rec_sn_comm,
            "token_anomaly": token_anomaly,
            "image_anomaly": image_anomaly,
            "sem_consistency": bpa_out["sem_consistency"],
        }

    def compute_loss(
        self,
        out: Dict[str, torch.Tensor],
        f_rgb: torch.Tensor,
        f_sn: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # One-class training: enforce high-fidelity reconstruction for normal samples.
        rgb_target = l2_normalize(f_rgb)
        sn_target = l2_normalize(f_sn)
        if token_mask is None:
            rec_loss = F.smooth_l1_loss(out["rec_rgb"], rgb_target) + F.smooth_l1_loss(out["rec_sn"], sn_target)
        else:
            valid = token_mask.unsqueeze(-1)
            rec_rgb = torch.masked_select(out["rec_rgb"], valid).view(-1, out["rec_rgb"].shape[-1])
            tgt_rgb = torch.masked_select(rgb_target, valid).view(-1, rgb_target.shape[-1])
            rec_sn = torch.masked_select(out["rec_sn"], valid).view(-1, out["rec_sn"].shape[-1])
            tgt_sn = torch.masked_select(sn_target, valid).view(-1, sn_target.shape[-1])
            rec_loss = F.smooth_l1_loss(rec_rgb, tgt_rgb) + F.smooth_l1_loss(rec_sn, tgt_sn)

        # Prototype diversity to reduce collapse.
        p_rgb = l2_normalize(self.bpa.proto_rgb, dim=-1)
        p_sn = l2_normalize(self.bpa.proto_sn, dim=-1)
        eye_rgb = torch.eye(p_rgb.shape[0], device=p_rgb.device)
        eye_sn = torch.eye(p_sn.shape[0], device=p_sn.device)
        div_loss = (torch.matmul(p_rgb, p_rgb.t()) - eye_rgb).pow(2).mean() + (
            torch.matmul(p_sn, p_sn.t()) - eye_sn
        ).pow(2).mean()

        sem_loss = out["sem_consistency"]

        total = self.cfg.rec_weight * rec_loss + self.cfg.sem_weight * sem_loss + self.cfg.div_weight * div_loss
        return {"total": total, "rec": rec_loss, "sem": sem_loss, "div": div_loss}
