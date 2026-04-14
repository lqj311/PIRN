"""
PIRN-style baseline that runs without installing extra packages.

This script is a lightweight, pure-stdlib approximation of the paper pipeline:
1) Train only on normal samples (unsupervised one-class setting).
2) Learn modality-specific prototypes for RGB and SN features.
3) Reconstruct token features via prototype assignment.
4) Use reconstruction discrepancy as anomaly score.

It is intended as a runnable starter for project setup on restricted environments.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


Vector = List[float]
TokenSet = List[Vector]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def l2_norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a)) + 1e-8


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return 1.0 - dot(a, b) / (l2_norm(a) * l2_norm(b))


def add(a: Sequence[float], b: Sequence[float]) -> Vector:
    return [x + y for x, y in zip(a, b)]


def sub(a: Sequence[float], b: Sequence[float]) -> Vector:
    return [x - y for x, y in zip(a, b)]


def mul_scalar(a: Sequence[float], s: float) -> Vector:
    return [x * s for x in a]


def mean_vec(vectors: Iterable[Sequence[float]], dim: int) -> Vector:
    vectors = list(vectors)
    if not vectors:
        return [0.0] * dim
    out = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            out[i] += x
    n = float(len(vectors))
    return [x / n for x in out]


def argmin_index(values: Sequence[float]) -> int:
    best_i = 0
    best_v = values[0]
    for i in range(1, len(values)):
        if values[i] < best_v:
            best_v = values[i]
            best_i = i
    return best_i


def auc_roc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Mann-Whitney U based AUROC. labels: 1 anomaly, 0 normal."""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    n_pos = 0
    n_neg = 0
    rank_sum_pos = 0.0
    for r, (_, y) in zip(ranks, pairs):
        if y == 1:
            n_pos += 1
            rank_sum_pos += r
        else:
            n_neg += 1
    if n_pos == 0 or n_neg == 0:
        return 0.5
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


@dataclass
class PIRNStdConfig:
    dim: int = 32
    num_tokens: int = 64
    num_proto_rgb: int = 16
    num_proto_sn: int = 16
    epochs: int = 25
    lr: float = 0.25
    comm_alpha: float = 0.20
    seed: int = 42


class PIRNStdBaseline:
    def __init__(self, cfg: PIRNStdConfig) -> None:
        self.cfg = cfg
        rnd = random.Random(cfg.seed)
        self.proto_rgb = [
            [rnd.uniform(-0.5, 0.5) for _ in range(cfg.dim)] for _ in range(cfg.num_proto_rgb)
        ]
        self.proto_sn = [
            [rnd.uniform(-0.5, 0.5) for _ in range(cfg.dim)] for _ in range(cfg.num_proto_sn)
        ]

    def _assign_and_reconstruct(self, tokens: TokenSet, prototypes: TokenSet) -> Tuple[TokenSet, List[int]]:
        recon: TokenSet = []
        assign: List[int] = []
        for t in tokens:
            dists = [cosine_distance(t, p) for p in prototypes]
            idx = argmin_index(dists)
            assign.append(idx)
            recon.append(prototypes[idx][:])
        return recon, assign

    def _update_prototypes(self, tokens: TokenSet, assign: List[int], prototypes: TokenSet) -> None:
        grouped: List[TokenSet] = [[] for _ in range(len(prototypes))]
        for t, k in zip(tokens, assign):
            grouped[k].append(t)
        for k, group in enumerate(grouped):
            if not group:
                continue
            center = mean_vec(group, self.cfg.dim)
            prototypes[k] = add(mul_scalar(prototypes[k], 1.0 - self.cfg.lr), mul_scalar(center, self.cfg.lr))

    def fit(self, train_rgb: List[TokenSet], train_sn: List[TokenSet]) -> None:
        for _ in range(self.cfg.epochs):
            for x_rgb, x_sn in zip(train_rgb, train_sn):
                # BPA-like per-modality assignment.
                z_rgb, a_rgb = self._assign_and_reconstruct(x_rgb, self.proto_rgb)
                z_sn, a_sn = self._assign_and_reconstruct(x_sn, self.proto_sn)

                # MNC-like simple bidirectional communication.
                comm_rgb = [add(mul_scalar(r, 1.0 - self.cfg.comm_alpha), mul_scalar(s, self.cfg.comm_alpha)) for r, s in zip(z_rgb, z_sn)]
                comm_sn = [add(mul_scalar(s, 1.0 - self.cfg.comm_alpha), mul_scalar(r, self.cfg.comm_alpha)) for r, s in zip(z_rgb, z_sn)]

                # APR-like prototype updates.
                self._update_prototypes(comm_rgb, a_rgb, self.proto_rgb)
                self._update_prototypes(comm_sn, a_sn, self.proto_sn)

    def score(self, x_rgb: TokenSet, x_sn: TokenSet) -> float:
        z_rgb, _ = self._assign_and_reconstruct(x_rgb, self.proto_rgb)
        z_sn, _ = self._assign_and_reconstruct(x_sn, self.proto_sn)
        token_scores = []
        for t_rgb, t_sn, r_rgb, r_sn in zip(x_rgb, x_sn, z_rgb, z_sn):
            d = cosine_distance(t_rgb, r_rgb) + cosine_distance(t_sn, r_sn)
            token_scores.append(d)
        return max(token_scores) if token_scores else 0.0


def random_vec(rnd: random.Random, dim: int, mean_shift: float, noise: float) -> Vector:
    return [rnd.gauss(mean_shift, noise) for _ in range(dim)]


def make_synthetic_dataset(
    cfg: PIRNStdConfig,
    n_train: int = 180,
    n_test_normal: int = 60,
    n_test_anomaly: int = 60,
) -> Tuple[List[TokenSet], List[TokenSet], List[TokenSet], List[TokenSet], List[int]]:
    rnd = random.Random(cfg.seed + 7)

    train_rgb: List[TokenSet] = []
    train_sn: List[TokenSet] = []
    for _ in range(n_train):
        x_rgb = [random_vec(rnd, cfg.dim, mean_shift=0.0, noise=0.30) for _ in range(cfg.num_tokens)]
        x_sn = [random_vec(rnd, cfg.dim, mean_shift=0.0, noise=0.30) for _ in range(cfg.num_tokens)]
        train_rgb.append(x_rgb)
        train_sn.append(x_sn)

    test_rgb: List[TokenSet] = []
    test_sn: List[TokenSet] = []
    labels: List[int] = []

    for _ in range(n_test_normal):
        x_rgb = [random_vec(rnd, cfg.dim, mean_shift=0.0, noise=0.30) for _ in range(cfg.num_tokens)]
        x_sn = [random_vec(rnd, cfg.dim, mean_shift=0.0, noise=0.30) for _ in range(cfg.num_tokens)]
        test_rgb.append(x_rgb)
        test_sn.append(x_sn)
        labels.append(0)

    for _ in range(n_test_anomaly):
        x_rgb = [random_vec(rnd, cfg.dim, mean_shift=0.0, noise=0.30) for _ in range(cfg.num_tokens)]
        x_sn = [random_vec(rnd, cfg.dim, mean_shift=0.0, noise=0.30) for _ in range(cfg.num_tokens)]
        # Inject local anomalies on a subset of tokens.
        idxs = rnd.sample(range(cfg.num_tokens), k=max(1, cfg.num_tokens // 8))
        for j in idxs:
            x_rgb[j] = random_vec(rnd, cfg.dim, mean_shift=1.2, noise=0.35)
            x_sn[j] = random_vec(rnd, cfg.dim, mean_shift=-1.0, noise=0.35)
        test_rgb.append(x_rgb)
        test_sn.append(x_sn)
        labels.append(1)

    return train_rgb, train_sn, test_rgb, test_sn, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a pure-stdlib PIRN-style baseline.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = PIRNStdConfig(dim=args.dim, num_tokens=args.tokens, epochs=args.epochs, seed=args.seed)
    model = PIRNStdBaseline(cfg)
    train_rgb, train_sn, test_rgb, test_sn, labels = make_synthetic_dataset(cfg)

    model.fit(train_rgb, train_sn)
    scores = [model.score(xr, xs) for xr, xs in zip(test_rgb, test_sn)]
    auroc = auc_roc(scores, labels)

    normal_scores = [s for s, y in zip(scores, labels) if y == 0]
    anomaly_scores = [s for s, y in zip(scores, labels) if y == 1]

    print("PIRN-style baseline run complete")
    print(f"seed={cfg.seed}, epochs={cfg.epochs}, dim={cfg.dim}, tokens={cfg.num_tokens}")
    print(f"test_auroc={auroc:.4f}")
    print(f"normal_mean={sum(normal_scores)/len(normal_scores):.4f}")
    print(f"anomaly_mean={sum(anomaly_scores)/len(anomaly_scores):.4f}")


if __name__ == "__main__":
    main()

