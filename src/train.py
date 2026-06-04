"""
Training script for GATsig shear-jamming classifier.

Usage:
    python train.py --data_dir /path/to/data --epochs 100 --lr 0.0001

Matches Mathematica NetTrain:
  - LearningRate: 0.0001
  - ValidationSet: Scaled[0.2]
  - Binary cross-entropy (boolean labels → sigmoid output)
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from model import GATsig
from data import make_dataloaders


def parse_args():
    p = argparse.ArgumentParser(description="Train GATsig on particle packing data")
    p.add_argument("--data_dir", type=str, required=True, help="Directory with .dat files")
    p.add_argument("--output_dir", type=str, default="./outputs", help="Checkpoints and logs")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default matches Mathematica)")
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--n_nodes", type=int, default=2000)
    p.add_argument("--fdim", type=int, default=5)
    p.add_argument("--hidden_dim", type=int, default=10, help="Hidden dim per GAT layer")
    p.add_argument("--n_heads", type=int, default=1, help="Attention heads per layer")
    p.add_argument("--n_layers", type=int, default=1, help="Number of stacked GAT layers")
    p.add_argument("--mconst", type=float, default=-10.0)
    p.add_argument("--alpha", type=float, default=0.2, help="LeakyReLU slope")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--non_dst_file", type=str, default="data1.csv")
    p.add_argument("--dst_file", type=str, default="data3.csv")
    p.add_argument("--skip", type=int, default=100, help="Burn-in packings to discard")
    p.add_argument("--stride", type=int, default=10, help="Keep every n-th packing after skip")
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for feats, pos, rad, labels in loader:
        # Squeeze batch dim (batch_size=1)
        feats  = feats.squeeze(0).to(device)
        pos    = pos.squeeze(0).to(device)
        rad    = rad.squeeze(0).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logit = model(feats, pos, rad)
        loss = criterion(logit.unsqueeze(0), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = (torch.sigmoid(logit) > 0.5).float()
        correct += (pred == labels).sum().item()
        total += labels.numel()

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for feats, pos, rad, labels in loader:
        feats  = feats.squeeze(0).to(device)
        pos    = pos.squeeze(0).to(device)
        rad    = rad.squeeze(0).to(device)
        labels = labels.to(device)

        logit = model(feats, pos, rad)
        loss = criterion(logit.unsqueeze(0), labels)

        total_loss += loss.item()
        pred = (torch.sigmoid(logit) > 0.5).float()
        correct += (pred == labels).sum().item()
        total += labels.numel()

    return total_loss / len(loader), correct / total


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Data
    files = [(args.non_dst_file, 0), (args.dst_file, 1)]
    print("Building dataset...")
    train_loader, val_loader = make_dataloaders(
        data_dir=args.data_dir,
        files=files,
        val_fraction=args.val_fraction,
        batch_size=1,
        seed=args.seed,
        skip=args.skip,
        stride=args.stride,
    )

    # Model
    model = GATsig(
        n_nodes=args.n_nodes,
        fdim=args.fdim,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mconst=args.mconst,
        alpha=args.alpha,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer & loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # Optionally resume
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    log = []
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        row = dict(
            epoch=epoch,
            train_loss=round(train_loss, 6),
            train_acc=round(train_acc, 4),
            val_loss=round(val_loss, 6),
            val_acc=round(val_acc, 4),
            time_s=round(elapsed, 1),
        )
        log.append(row)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f} | {elapsed:.1f}s")

        # Save checkpoint every epoch
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        }
        torch.save(ckpt, output_dir / "last.pt")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, output_dir / "best.pt")
            print(f"  → New best val_loss: {best_val_loss:.4f}")

        # Flush log
        with open(output_dir / "log.json", "w") as f:
            json.dump(log, f, indent=2)

    print("Training complete.")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
