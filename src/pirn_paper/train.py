from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.optim import AdamW

from pirn_paper.config import PIRNConfig
from pirn_paper.data import build_dataloaders
from pirn_paper.metrics import summarize_loss, summarize_scores
from pirn_paper.model import PIRNModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: PIRNModel,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    loss_items: List[Dict[str, torch.Tensor]] = []
    for batch in loader:
        f_rgb = batch["rgb"].to(device)
        f_sn = batch["sn"].to(device)
        token_mask = batch["mask"].to(device)

        out = model(f_rgb, f_sn, update_proto=True, token_mask=token_mask)
        losses = model.compute_loss(out, f_rgb, f_sn, token_mask=token_mask)

        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        loss_items.append(losses)

    return summarize_loss(loss_items)


@torch.no_grad()
def evaluate(model: PIRNModel, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_scores = []
    all_labels = []
    for batch in loader:
        f_rgb = batch["rgb"].to(device)
        f_sn = batch["sn"].to(device)
        token_mask = batch["mask"].to(device)
        labels = batch["label"].to(device)

        out = model(f_rgb, f_sn, update_proto=False, token_mask=token_mask)
        all_scores.append(out["image_anomaly"].detach().cpu())
        all_labels.append(labels.detach().cpu())

    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return summarize_scores(scores, labels)


def save_checkpoint(path: Path, model: PIRNModel, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def append_csv(path: Path, row: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PIRN paper model on pre-extracted features.")
    parser.add_argument("--data-root", type=str, required=True, help="Dataset root with train/test .pt files.")
    parser.add_argument("--output-dir", type=str, default="runs/pirn_paper")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--save-every", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    train_loader, test_loader = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = PIRNModel(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    csv_path = out_dir / "metrics.csv"
    best_ckpt = out_dir / "best.pt"
    best_auroc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args.grad_clip)
        eval_metrics = evaluate(model, test_loader, device)

        row: Dict[str, float] = {"epoch": float(epoch)}
        for k, v in train_metrics.items():
            row[f"train_{k}"] = float(v)
        for k, v in eval_metrics.items():
            row[f"eval_{k}"] = float(v)
        append_csv(csv_path, row)

        if eval_metrics["auroc"] > best_auroc:
            best_auroc = eval_metrics["auroc"]
            save_checkpoint(best_ckpt, model, optimizer, epoch)

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(out_dir / f"epoch_{epoch}.pt", model, optimizer, epoch)

        summary = " ".join([f"{k}={v:.4f}" for k, v in eval_metrics.items()])
        print(f"[Epoch {epoch:03d}] {summary}")

    print(f"Training finished. Best AUROC={best_auroc:.4f}. Artifacts: {out_dir}")


if __name__ == "__main__":
    main()

