from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    """
    Average rank for ties, equivalent to scipy.stats.rankdata(method="average").
    """
    sorted_idx = torch.argsort(values)
    sorted_vals = values[sorted_idx]
    ranks = torch.zeros_like(values, dtype=torch.float32)

    i = 0
    n = sorted_vals.numel()
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1].item() == sorted_vals[i].item():
            j += 1
        avg_rank = (i + j) * 0.5 + 1.0
        ranks[sorted_idx[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    labels: 1 anomaly, 0 normal
    """
    scores = scores.detach().flatten().float().cpu()
    labels = labels.detach().flatten().long().cpu()
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum().item())
    n_neg = int(neg.sum().item())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    ranks = _rankdata(scores)
    rank_sum_pos = ranks[pos].sum().item()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def binary_average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.detach().flatten().float().cpu()
    labels = labels.detach().flatten().long().cpu()
    idx = torch.argsort(scores, descending=True)
    labels = labels[idx]

    tp = 0.0
    fp = 0.0
    precision_sum = 0.0
    pos_total = float((labels == 1).sum().item())
    if pos_total == 0:
        return 0.0

    for y in labels:
        if int(y.item()) == 1:
            tp += 1.0
            precision_sum += tp / (tp + fp)
        else:
            fp += 1.0
    return precision_sum / pos_total


def summarize_loss(loss_items: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    agg: Dict[str, List[float]] = {}
    for item in loss_items:
        for k, v in item.items():
            agg.setdefault(k, []).append(float(v.detach().cpu().item()))
    return {k: sum(vs) / max(len(vs), 1) for k, vs in agg.items()}


def summarize_scores(scores: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    return {
        "auroc": binary_auroc(scores, labels),
        "ap": binary_average_precision(scores, labels),
        "normal_mean": float(scores[labels == 0].mean().item()) if (labels == 0).any() else 0.0,
        "anomaly_mean": float(scores[labels == 1].mean().item()) if (labels == 1).any() else 0.0,
    }

