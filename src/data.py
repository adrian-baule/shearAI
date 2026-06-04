"""
Data pipeline for shear-jamming particle packing data.

Raw .dat files interleave '#' comment lines (global header + per-packing
metadata) with particle data rows.  np.loadtxt skips '#' lines by default,
yielding a flat (P*N, 11) array where P is the number of packings.

Each particle row (11 columns, 0-indexed):
  col 0  : particle index   (ignored)
  col 1  : radius           → radii, features[:,0]
  col 2  : position x       → positions[:,0]
  col 3  : position z       → positions[:,1]
  col 4  : velocity x       → features[:,1]
  col 5  : velocity y       (ignored)
  col 6  : velocity z       → features[:,2]
  col 7  : angular vel x    (ignored)
  col 8  : angular vel y    → features[:,3]
  col 9  : angular vel z    (ignored)
  col 10 : angle            → features[:,4]

Packing selection (matching Mathematica GATsig_layer1head1.nb):
  - i = 0..99, packing index = stride*i  → packings 0, 10, 20, …, 990
  - No burn-in skip; `skip=0` reproduces the notebook exactly
  - Use skip > 0 only if you want to discard early transient packings

Label:
  phi=0.752  (data1) → 0   (not DST)
  phi=0.764  (data3) → 1   (DST)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Optional, Tuple, List


FEATURE_COLS = [1, 4, 6, 8, 10]   # 0-indexed columns kept as node features
POSITION_COLS = [2, 3]              # 0-indexed: pos_x, pos_z
RADIUS_COL = 1                      # 0-indexed
N_NODES = 2000
SKIP = 0                            # no burn-in skip (matches Mathematica notebook)
STRIDE = 10                         # keep every n-th packing
N_PACKINGS = 100                    # number of packings to take (i=0..99)


def load_dat_file(path: str) -> np.ndarray:
    """
    Load a particle data file (.dat or .csv) into a float32 numpy array.

    Handles:
    - Whitespace-separated .dat files with '#' comment lines (LF-DEM output)
    - Comma-separated .csv files with optional '#' comment lines or a text header
    """
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path, comment="#", header=None)
        # Drop any non-numeric header row that snuck through
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        return df.values.astype(np.float32)
    else:
        return np.loadtxt(path).astype(np.float32)


def extract_packings(
    data: np.ndarray,
    n_nodes: int = N_NODES,
    skip: int = SKIP,
    stride: int = STRIDE,
    n_packings: int = N_PACKINGS,
) -> List[np.ndarray]:
    """
    Extract packings matching the Mathematica notebook selection:
        for i in 0..n_packings-1: packing index = skip + stride*i

    skip=0, stride=10, n_packings=100 reproduces GATsig_layer1head1.nb exactly
    (packings 0, 10, 20, …, 990).  Increase skip to discard early transients.
    """
    packings = []
    for i in range(n_packings):
        packing_idx = skip + stride * i
        start = packing_idx * n_nodes
        end   = start + n_nodes
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
        skip: int = SKIP,
        stride: int = STRIDE,
        n_packings: int = N_PACKINGS,
    ):
        self.samples = []  # list of (features, positions, radii, label)

        for filename, label in files:
            path = os.path.join(data_dir, filename)
            print(f"  Loading {filename} (label={label}) ...", flush=True)
            raw = load_dat_file(path)
            packings = extract_packings(raw, n_nodes, skip, stride, n_packings)
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
    skip: int = SKIP,
    stride: int = STRIDE,
    n_packings: int = N_PACKINGS,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders matching Mathematica's ValidationSet->Scaled[0.2].

    Note: batch_size=1 because each packing is a full graph (variable N,
    and the model uses N-sized weight matrices). Increase if you pad/batch graphs.
    """
    dataset = PackingDataset(data_dir, files, skip=skip, stride=stride, n_packings=n_packings)
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
