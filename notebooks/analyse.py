"""
Evaluate a trained GATsig checkpoint and visualise results.

Run as a script:
    python notebooks/analyse.py --checkpoint outputs/best.pt --data_dir data/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

from model import GATsig
from data import PackingDataset


def load_model(checkpoint_path: str, device) -> GATsig:
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Try to load config
    config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
    else:
        cfg = {}
    model = GATsig(
        n_nodes=cfg.get("n_nodes", 2000),
        fdim=cfg.get("fdim", 5),
        newfdim=cfg.get("newfdim", 10),
        mconst=cfg.get("mconst", -10.0),
        alpha=cfg.get("alpha", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def plot_training_log(log_path: str, out_path: str = "training_curves.png"):
    with open(log_path) as f:
        log = json.load(f)
    epochs = [r["epoch"] for r in log]
    train_loss = [r["train_loss"] for r in log]
    val_loss   = [r["val_loss"]   for r in log]
    train_acc  = [r["train_acc"]  for r in log]
    val_acc    = [r["val_acc"]    for r in log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, train_loss, label="train")
    ax1.plot(epochs, val_loss,   label="val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE Loss"); ax1.legend(); ax1.set_title("Loss")

    ax2.plot(epochs, train_acc, label="train")
    ax2.plot(epochs, val_acc,   label="val")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend(); ax2.set_title("Accuracy")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()


@torch.no_grad()
def evaluate_dataset(model, dataset, device):
    probs, labels = [], []
    for feats, pos, rad, label in dataset:
        out = model(feats.to(device), pos.to(device), rad.to(device))
        prob = torch.sigmoid(out).item()
        probs.append(prob)
        labels.append(int(label.item()))
    return np.array(probs), np.array(labels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--non_dst_file", default="phi0p752.dat")
    p.add_argument("--dst_file", default="phi0p764.dat")
    p.add_argument("--output_dir", default="outputs")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)
    print(f"Loaded model from {args.checkpoint}")

    # Training curves
    log_path = os.path.join(args.output_dir, "log.json")
    if os.path.exists(log_path):
        plot_training_log(log_path, os.path.join(args.output_dir, "training_curves.png"))

    # Full evaluation
    files = [(args.non_dst_file, 0), (args.dst_file, 1)]
    print("Loading full dataset for evaluation...")
    dataset = PackingDataset(args.data_dir, files)

    probs, labels = evaluate_dataset(model, dataset, device)
    preds = (probs > 0.5).astype(int)

    auc = roc_auc_score(labels, probs)
    print(f"\nROC-AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["non-DST", "DST"]))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))

    # Probability distribution plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(probs[labels == 0], bins=20, alpha=0.6, label="non-DST (phi=0.752)")
    ax.hist(probs[labels == 1], bins=20, alpha=0.6, label="DST (phi=0.764)")
    ax.axvline(0.5, color="k", linestyle="--", label="threshold")
    ax.set_xlabel("Predicted DST probability"); ax.set_ylabel("Count")
    ax.legend(); ax.set_title(f"GATsig predictions  (AUC={auc:.3f})")
    out = os.path.join(args.output_dir, "probability_dist.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
