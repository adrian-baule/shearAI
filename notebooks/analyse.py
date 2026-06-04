"""
Evaluate or run inference with a trained GATsig checkpoint.

Two modes:

  1. evaluate  — labelled data (two files, known DST / non-DST)
                 reports ROC-AUC, classification report, probability histogram

  2. predict   — unlabelled data (one file, unknown label)
                 outputs per-packing DST probability to a CSV

Examples:
    # Evaluate on labelled data
    python notebooks/analyse.py evaluate \\
        --checkpoint outputs/best.pt \\
        --data_dir   /content/drive/MyDrive/data \\
        --non_dst_file data1.csv --dst_file data3.csv

    # Predict on a new unlabelled file
    python notebooks/analyse.py predict \\
        --checkpoint outputs/best.pt \\
        --data_dir   /content/drive/MyDrive/new_data \\
        --input_file new_experiment.csv \\
        --output_csv predictions.csv
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

from model import GATsig
from data import PackingDataset, load_dat_file, extract_packings, packing_to_tensors, SKIP, STRIDE


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device) -> GATsig:
    """Load a GATsig checkpoint; reads config.json from the same directory."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
    model = GATsig(
        n_nodes=cfg.get("n_nodes", 2000),
        fdim=cfg.get("fdim", 5),
        hidden_dim=cfg.get("hidden_dim", 10),
        n_heads=cfg.get("n_heads", 1),
        n_layers=cfg.get("n_layers", 1),
        mconst=cfg.get("mconst", -10.0),
        alpha=cfg.get("alpha", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, packings, device):
    """Return DST probability for each packing (list of np arrays)."""
    probs = []
    for packing in packings:
        feats, pos, rad = packing_to_tensors(packing)
        out = model(feats.to(device), pos.to(device), rad.to(device))
        probs.append(torch.sigmoid(out).item())
    return np.array(probs)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_training_log(log_path: str, out_path: str):
    with open(log_path) as f:
        log = json.load(f)
    epochs     = [r["epoch"]      for r in log]
    train_loss = [r["train_loss"] for r in log]
    val_loss   = [r["val_loss"]   for r in log]
    train_acc  = [r["train_acc"]  for r in log]
    val_acc    = [r["val_acc"]    for r in log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, train_loss, label="train")
    ax1.plot(epochs, val_loss,   label="val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE Loss")
    ax1.legend(); ax1.set_title("Loss")

    ax2.plot(epochs, train_acc, label="train")
    ax2.plot(epochs, val_acc,   label="val")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.legend(); ax2.set_title("Accuracy")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# evaluate mode
# ---------------------------------------------------------------------------

def cmd_evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.checkpoint, device)
    print(f"Loaded model from {args.checkpoint}")

    log_path = os.path.join(os.path.dirname(args.checkpoint), "log.json")
    if os.path.exists(log_path):
        plot_training_log(log_path, os.path.join(args.output_dir, "training_curves.png"))

    files = [(args.non_dst_file, 0), (args.dst_file, 1)]
    print("Loading labelled dataset...")
    dataset = PackingDataset(args.data_dir, files, skip=args.skip, stride=args.stride)

    probs, labels = [], []
    for feats, pos, rad, label in dataset:
        out  = model(feats.to(device), pos.to(device), rad.to(device))
        probs.append(torch.sigmoid(out).item())
        labels.append(int(label.item()))
    probs  = np.array(probs)
    labels = np.array(labels)
    preds  = (probs > 0.5).astype(int)

    auc = roc_auc_score(labels, probs)
    print(f"\nROC-AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["non-DST", "DST"]))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))

    os.makedirs(args.output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(probs[labels == 0], bins=20, alpha=0.6, label="non-DST")
    ax.hist(probs[labels == 1], bins=20, alpha=0.6, label="DST")
    ax.axvline(0.5, color="k", linestyle="--", label="threshold")
    ax.set_xlabel("Predicted DST probability"); ax.set_ylabel("Count")
    ax.legend(); ax.set_title(f"GATsig predictions  (AUC={auc:.3f})")
    out = os.path.join(args.output_dir, "probability_dist.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.show()


# ---------------------------------------------------------------------------
# predict mode
# ---------------------------------------------------------------------------

def cmd_predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.checkpoint, device)
    print(f"Loaded model from {args.checkpoint}")

    path = os.path.join(args.data_dir, args.input_file)
    print(f"Loading {path} ...")
    raw      = load_dat_file(path)
    packings = extract_packings(raw, skip=args.skip, stride=args.stride)
    print(f"  → {len(packings)} packings extracted")

    probs = run_inference(model, packings, device)

    # Packing indices in the original file (before striding)
    packing_ids = list(range(args.skip, args.skip + len(packings) * args.stride, args.stride))

    df = pd.DataFrame({
        "packing_index": packing_ids,
        "dst_probability": probs,
        "predicted_dst": (probs > 0.5).astype(int),
    })
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")
    print(df.describe())

    # Probability trace over time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(packing_ids, probs, lw=1)
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("Packing index (strain step)"); ax.set_ylabel("DST probability")
    ax.set_title(f"GATsig inference — {args.input_file}")
    out = os.path.join(args.output_dir, "prediction_trace.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.show()


# ---------------------------------------------------------------------------
# attention mode
# ---------------------------------------------------------------------------

@torch.no_grad()
def cmd_attention(args):
    """
    Extract non-zero attention weights for a single selected packing and
    write them to alphaweights.dat.

    Output format (whitespace-separated):
        node_i  node_j  alpha_L0H0  alpha_L0H1 ...  alpha_L1H0 ...

    One row per directed contact edge (i -> j) where at least one alpha > threshold.
    node_i is the source (aggregating) node, node_j is the neighbour it attends to.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.checkpoint, device)
    print(f"Loaded model from {args.checkpoint}")

    path = os.path.join(args.data_dir, args.input_file)
    print(f"Loading {path} ...")
    raw      = load_dat_file(path)
    packings = extract_packings(raw, skip=args.skip, stride=args.stride)
    total    = len(packings)
    print(f"  → {total} packings available (indices 0 .. {total - 1})")

    idx = args.packing_index
    if idx < 0 or idx >= total:
        raise ValueError(f"--packing_index {idx} out of range [0, {total - 1}]")

    packing = packings[idx]
    feats, pos, rad = packing_to_tensors(packing)
    print(f"Running inference on packing {idx} ...")

    _, all_attns = model(
        feats.to(device), pos.to(device), rad.to(device),
        return_attention=True,
    )
    # all_attns: list[layer] of list[head] of (N, N) tensors

    N = feats.shape[0]
    n_layers = len(all_attns)
    n_heads  = len(all_attns[0])

    # Build column header names: L{l}H{k} for each layer/head combination
    head_names = [f"L{l}H{k}" for l in range(n_layers) for k in range(n_heads)]

    # Stack all alpha values into (N, N, n_layers*n_heads) numpy array
    alpha_stack = np.stack(
        [all_attns[l][k].cpu().numpy() for l in range(n_layers) for k in range(n_heads)],
        axis=-1,
    )  # (N, N, n_cols)

    # Keep only edges where max alpha across heads/layers exceeds threshold
    max_alpha = alpha_stack.max(axis=-1)                        # (N, N)
    i_idx, j_idx = np.where(max_alpha > args.threshold)

    if len(i_idx) == 0:
        print(f"No edges exceed threshold {args.threshold}. Try lowering --threshold.")
        return

    rows = np.column_stack([i_idx, j_idx, alpha_stack[i_idx, j_idx]])  # (E, 2 + n_cols)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.output_file)

    header = "node_i  node_j  " + "  ".join(head_names)
    np.savetxt(out_path, rows, fmt="%d %d" + " %.6f" * len(head_names), header=header)
    print(f"Saved {len(i_idx)} contact edges to {out_path}")
    print(f"  Columns: node_i, node_j, {', '.join(head_names)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint",  required=True, help="Path to best.pt or last.pt")
    common.add_argument("--data_dir",    required=True, help="Directory containing data files")
    common.add_argument("--output_dir",  default="outputs")
    common.add_argument("--skip",        type=int, default=SKIP,   help="Burn-in packings to discard")
    common.add_argument("--stride",      type=int, default=STRIDE, help="Keep every n-th packing")

    ev = sub.add_parser("evaluate", parents=[common])
    ev.add_argument("--non_dst_file", default="data1.csv")
    ev.add_argument("--dst_file",     default="data3.csv")

    pr = sub.add_parser("predict", parents=[common])
    pr.add_argument("--input_file",  required=True, help="Unlabelled .csv or .dat file to analyse")
    pr.add_argument("--output_csv",  default="predictions.csv")

    at = sub.add_parser("attention", parents=[common])
    at.add_argument("--input_file",     required=True, help=".csv or .dat file to inspect")
    at.add_argument("--packing_index",  type=int, default=0, help="Which packing to inspect (0-based after skip/stride)")
    at.add_argument("--threshold",      type=float, default=1e-3, help="Min alpha to include an edge")
    at.add_argument("--output_file",    default="alphaweights.dat")

    args = p.parse_args()
    if args.mode == "evaluate":
        cmd_evaluate(args)
    elif args.mode == "predict":
        cmd_predict(args)
    else:
        cmd_attention(args)


if __name__ == "__main__":
    main()
