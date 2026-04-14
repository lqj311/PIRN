from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _make_sample(tokens: int, dim: int, anomaly: bool) -> dict:
    rgb = torch.randn(tokens, dim) * 0.3
    sn = torch.randn(tokens, dim) * 0.3
    if anomaly:
        n = max(1, tokens // 8)
        idx = torch.randperm(tokens)[:n]
        rgb[idx] = rgb[idx] + 1.2
        sn[idx] = sn[idx] - 1.0
    return {"rgb": rgb, "sn": sn}


def _dump_split(root: Path, split: str, count: int, tokens: int, dim: int, anomaly: bool) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        sample = _make_sample(tokens=tokens, dim=dim, anomaly=anomaly)
        torch.save(sample, split_dir / f"{i:05d}.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create toy PIRN feature dataset.")
    parser.add_argument("--out", type=str, default="toy_features")
    parser.add_argument("--tokens", type=int, default=196)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--train", type=int, default=120)
    parser.add_argument("--test-normal", type=int, default=40)
    parser.add_argument("--test-anomaly", type=int, default=40)
    args = parser.parse_args()

    root = Path(args.out)
    _dump_split(root / "train", "normal", args.train, args.tokens, args.dim, anomaly=False)
    _dump_split(root / "test", "normal", args.test_normal, args.tokens, args.dim, anomaly=False)
    _dump_split(root / "test", "anomaly", args.test_anomaly, args.tokens, args.dim, anomaly=True)
    print(f"Toy dataset generated at: {root}")


if __name__ == "__main__":
    main()

