from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize tensor along a given dimension."""
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def entropy_from_logits(logits: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Compute entropy from logits."""
    probs = F.softmax(logits, dim=dim)
    return -(probs * probs.clamp_min(eps).log()).sum(dim=dim)


def normalized_entropy_from_logits(logits: torch.Tensor, num_bins: int, dim: int = -1) -> torch.Tensor:
    """Entropy normalized to [0, 1] for approximately num_bins classes."""
    ent = entropy_from_logits(logits, dim=dim)
    denom = math.log(max(float(num_bins), 2.0))
    return ent / denom

