from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pirn_paper.config import PIRNConfig
from pirn_paper.data import build_dataloaders
from pirn_paper.train import evaluate
from pirn_paper.model import PIRNModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PIRN paper model checkpoint.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--num-tokens", type=int, default=196)
    parser.add_argument("--num-proto-rgb", type=int, default=24)
    parser.add_argument("--num-proto-sn", type=int, default=24)
    parser.add_argument("--sinkhorn-tau", type=float, default=0.07)
    parser.add_argument("--sinkhorn-iters", type=int, default=7)
    parser.add_argument("--apr-eps", type=float, default=1e-6)
    parser.add_argument("--mnc-heads", type=int, default=8)
    parser.add_argument("--mnc-dropout", type=float, default=0.1)
    parser.add_argument("--rec-weight", type=float, default=1.0)
    parser.add_argument("--sem-weight", type=float, default=0.1)
    parser.add_argument("--div-weight", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    cfg = PIRNConfig(
        dim=args.dim,
        num_tokens=args.num_tokens,
        num_proto_rgb=args.num_proto_rgb,
        num_proto_sn=args.num_proto_sn,
        sinkhorn_tau=args.sinkhorn_tau,
        sinkhorn_iters=args.sinkhorn_iters,
        apr_eps=args.apr_eps,
        mnc_heads=args.mnc_heads,
        mnc_dropout=args.mnc_dropout,
        rec_weight=args.rec_weight,
        sem_weight=args.sem_weight,
        div_weight=args.div_weight,
    )
    model = PIRNModel(cfg).to(device)

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    _, test_loader = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    metrics = evaluate(model, test_loader, device)
    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()

