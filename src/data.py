"""
Data pipeline for shear-jamming particle packing data.

Raw .dat files have rows of particle data.  Each row is one particle:
  col 0  : (ignored)
  col 1  : particle index / type
  col 2  : radius          → radii (also used as feature)
  col 3  : x position      → positions[:,0]
  col 4  : y position      → positions[:,1]
  col 5  : feature         → features[:,1]
  col 6  : (ignored)
  col 7  : feature         → features[:,2]
  col 8  : (ignored)
  col 9  : feature         → features[:,3]
  col 10 : (ignored)
  col 11 : feature         → features[:,4]

One "packing" = 2000 consecutive rows.
Mathematica selects every 10th packing (i=0..99 → 100 packings per file).

Label:
  phi=0.752  (data1) → False / 0   (not DST)
  phi=0.758  (data2) → False / 0   (not DST) — omitted in original
  phi=0.764  (data3) → True  / 1   (DST)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Optional, Tuple, List


FEATURE_COLS = [1, 4, 6, 8, 10]   # 0-indexed: cols 2,5,7,9,11 in 1-indexed Mathematica
POSITION_COLS = [2, 3]              # 0-indexed: cols 3,4
RADIUS_COL = 1                      # 0-indexed: col 2
N_NODES = 2000
STRIDE = 10                         # every 10th packing
N_PACKINGS = 100                    # 0..99


def load_dat_file(path: str) -> np.ndarray:
    """Load a whitespace-separated .dat file into a numpy array."""
    return np.loadtxt(path)


def extract_packings(data: np.ndarray, n_nodes: int = N_NODES, stride: int = STRIDE, n_packings: int = N_PACKINGS) -> List[np.ndarray]:
    """
    Extract every `stride`-th packing from raw data matrix.
    Packing i occupies rows [i*n_nodes : (i+1)*n_nodes].
    Returns list of (n_nodes, n_cols) arrays.
    """
    packings = []
    for i in range(n_packings):
        start = stride * i * n_nodes
        end = start + n_nodes
        if end > data.shape[0]:
            break
        packings.append(data[start:end])
    return packings


def packing_to_tensors(packing: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a single packing array to (features, positions, radii) tensors.

    features  : (N, 5)  float32
    positions : (N, 2)  float32
    radii     : (N,)    float32
    """
    features = torch.tensor(packing[:, FEATURE_COLS], dtype=torch.float32)
    positions = torch.tensor(packing[:, POSITION_COLS], dtype=torch.float32)
    radii = torch.tensor(packing[:, RADIUS_COL], dtype=torch.float32)
    return features, positions, radii


class PackingDataset(Dataset):
    """
    Dataset of particle packings with binary DST labels.

    Args:
        data_dir : directory containing .dat files
        files    : list of (filename, label) tuples
                   e.g. [("phi0p752.dat", 0), ("phi0p764.dat", 1)]
        n_nodes  : particles per packing
        stride   : select every n-th packing
        n_packings : number of packings to take per file
    """

    def __init__(
        self,
        data_dir: str,
        files: List[Tuple[str, int]],
        n_nodes: int = N_NODES,
        stride: int = STRIDE,
        n_packings: int = N_PACKINGS,
    ):
        self.samples = []  # list of (features, positions, radii, label)

        for filename, label in files:
            path = os.path.join(data_dir, filename)
            print(f"  Loading {filename} (label={label}) ...", flush=True)
            raw = load_dat_file(path)
            packings = extract_packings(raw, n_nodes, stride, n_packings)
            print(f"    → {len(packings)} packings extracted")
            for packing in packings:
                feats, pos, rad = packing_to_tensors(packing)
                self.samples.append((feats, pos, rad, torch.tensor(float(label))))

        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_dataloaders(
    data_dir: str,
    files: List[Tuple[str, int]],
    val_fraction: float = 0.2,
    batch_size: int = 1,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders matching Mathematica's ValidationSet->Scaled[0.2].

    Note: batch_size=1 because each packing is a full graph (variable N,
    and the model uses N-sized weight matrices). Increase if you pad/batch graphs.
    """
    dataset = PackingDataset(data_dir, files)
    total = len(dataset)
    val_size = int(total * val_fraction)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    return train_loader, val_loader


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    files = [("phi0p752.dat", 0), ("phi0p764.dat", 1)]
    train_loader, val_loader = make_dataloaders(data_dir, files)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    feats, pos, rad, label = next(iter(train_loader))
    print(f"  features: {feats.shape}, positions: {pos.shape}, radii: {rad.shape}, label: {label}")
