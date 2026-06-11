"""
Training script for GATsig shear-jamming classifier.

Usage:
    python src/train.py --data_dir /path/to/data --epochs 100

GPU (Colab):
    Automatically uses CUDA if available.  Mixed-precision (AMP) is enabled
    by default on CUDA for ~2x speedup on T4/A100; disable with --no_amp.

    In Colab, make sure the runtime is set to GPU:
        Runtime -> Change runtime type -> T4 GPU
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from model import GATsig
from data import make_dataloaders


def parse_args():
    p = argparse.ArgumentParser(description="Train GATsig on particle packing data")
    p.add_argument("--data_dir",     type=str, required=True, help="Directory with data files")
    p.add_argument("--output_dir",   type=str, default="./outputs")
    p.add_argument("--epochs",       type=int, default=100)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--n_nodes",      type=int, default=2000)
    p.add_argument("--fdim",         type=int, default=5)
    p.add_argument("--hidden_dim",   type=int, default=10)
    p.add_argument("--n_heads",      type=int, default=1)
    p.add_argument("--n_layers",     type=int, default=1)
    p.add_argument("--mconst",       type=float, default=-50.0)
    p.add_argument("--alpha",        type=float, default=0.2)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--resume",       type=str, default=None)
    p.add_argument("--non_dst_file", type=str, default="data1.csv")
    p.add_argument("--dst_file",     type=str, default="data3.csv")
    p.add_argument("--skip",         type=int, default=100, help="Burn-in packings to discard before selection")
    p.add_argument("--stride",       type=int, default=10,  help="Take every n-th packing")
    p.add_argument("--n_packings",   type=int, default=100, help="Number of packings per file")
    p.add_argument("--batch_size",   type=int, default=1,  help="Batch size (default 1; increase only if all packings have the same N)")
    p.add_argument("--no_amp",       action="store_true", help="Disable mixed-precision training")
    p.add_argument("--no_checkpoints", action="store_true", help="Skip saving best.pt and last.pt (weights CSV still saved)")
    p.add_argument("--patience",     type=int, default=10,
                   help="Early stopping: stop after this many epochs with no val_loss improvement. 0 = disabled.")
    return p.parse_args()


def _asrc_atarg_grad_norms(model: GATsig):
    """Return mean gradient norms for a_src and a_tgt across all layers/heads."""
    src_norms, tgt_norms = [], []
    for layer in model.layers:
        for k in range(layer.n_heads):
            if layer.a_src[k].grad is not None:
                src_norms.append(layer.a_src[k].grad.norm().item())
            if layer.a_tgt[k].grad is not None:
                tgt_norms.append(layer.a_tgt[k].grad.norm().item())
    return (
        sum(src_norms) / len(src_norms) if src_norms else float("nan"),
        sum(tgt_norms) / len(tgt_norms) if tgt_norms else float("nan"),
    )


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = scaler is not None

    for feats, pos, rad, labels in loader:
        # feats/pos/rad: (B, N, *); iterate over batch dimension
        feats  = feats.to(device, non_blocking=True)
        pos    = pos.to(device, non_blocking=True)
        rad    = rad.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = []
        with autocast(enabled=use_amp):
            for b in range(feats.shape[0]):
                logits.append(model(feats[b], pos[b], rad[b]))
            logits = torch.stack(logits)
            loss   = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pred = (logits.detach().float() > 0.5).float()
        correct += (pred == labels).sum().item()
        total += labels.numel()

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for feats, pos, rad, labels in loader:
        feats  = feats.to(device, non_blocking=True)
        pos    = pos.to(device, non_blocking=True)
        rad    = rad.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = torch.stack([model(feats[b], pos[b], rad[b]) for b in range(feats.shape[0])])
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        pred = (logits.float() > 0.5).float()
        correct += (pred == labels).sum().item()
        total += labels.numel()

    return total_loss / len(loader), correct / total


def _save_weights_csv(model: GATsig, output_dir: Path):
    """Save W, asrc, atarg for the first GAT layer/head to CSV (Mathematica-compatible)."""
    layer = model.layers[0]
    W     = layer.W[0].weight.detach().cpu().numpy()       # (hidden_dim, in_dim) -> transpose to (in_dim, hidden_dim)
    asrc  = layer.a_src[0].detach().cpu().numpy()          # (hidden_dim,)
    atarg = layer.a_tgt[0].detach().cpu().numpy()          # (hidden_dim,)
    np.savetxt(output_dir / "out_W.csv",     W.T,   delimiter=",", fmt="%.10f")   # (fdim, hidden_dim)
    np.savetxt(output_dir / "out_asrc.csv",  asrc,  delimiter=",", fmt="%.10f")
    np.savetxt(output_dir / "out_atarg.csv", atarg, delimiter=",", fmt="%.10f")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"Device: {device}  |  AMP: {use_amp}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Data
    files = [(args.non_dst_file, 0), (args.dst_file, 1)]
    print("Building dataset...")
    train_loader, val_loader = make_dataloaders(
        data_dir=args.data_dir,
        files=files,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        seed=args.seed,
        skip=args.skip,
        stride=args.stride,
        n_packings=args.n_packings,
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

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    scaler    = GradScaler() if use_amp else None

    start_epoch   = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    log = []
    epochs_no_improve = 0
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        row = dict(
            epoch=epoch,
            train_loss=round(train_loss, 6),
            train_acc=round(train_acc, 4),
            val_loss=round(val_loss, 6),
            val_acc=round(val_acc, 4),
            time_s=round(elapsed, 1),
        )
        asrc_gnorm, atarg_gnorm = _asrc_atarg_grad_norms(model)
        row["asrc_grad_norm"]  = round(asrc_gnorm,  8)
        row["atarg_grad_norm"] = round(atarg_gnorm, 8)
        log.append(row)
        print(f"Epoch {epoch:03d} | train={train_loss:.4f}/{train_acc:.3f} "
              f"val={val_loss:.4f}/{val_acc:.3f} | "
              f"∇asrc={asrc_gnorm:.2e}  ∇atarg={atarg_gnorm:.2e} | {elapsed:.1f}s")

        if not args.no_checkpoints:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }
            if scaler:
                ckpt["scaler"] = scaler.state_dict()
            torch.save(ckpt, output_dir / "last.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            if not args.no_checkpoints:
                ckpt = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                }
                if scaler:
                    ckpt["scaler"] = scaler.state_dict()
                torch.save(ckpt, output_dir / "best.pt")
            print(f"  → New best val_loss: {best_val_loss:.4f}")
            _save_weights_csv(model, output_dir)
        else:
            epochs_no_improve += 1

        with open(output_dir / "log.json", "w") as f:
            json.dump(log, f, indent=2)

        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"Early stopping: val_loss did not improve for {args.patience} epochs.")
            break

    print("Training complete.")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
