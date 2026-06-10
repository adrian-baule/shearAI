"""
siggat_layer.py — standalone siggat computation for validation.

Computes the output of the siggat layer (attention weight matrix) given
a packing and explicit weight arrays, without needing a trained GATsig model.

Matches the Mathematica computation exactly:
    e[i,j]   = LeakyReLU( a_src·H[j] + a_tgt·H[i] )   (outersum convention)
    masked   = A * e + (1 - A) * mconst
    siggat   = sigmoid(masked)

Usage as a script
-----------------
    python src/siggat_layer.py \\
        --packing   data/phi0p752.dat \\
        --packing_idx 0 \\
        --W         weights/W.csv \\
        --asrc      weights/asrc.csv \\
        --atarg     weights/atarg.csv

Weight files must be CSV exported from Mathematica via Export["file.csv", array]:
  W     : (fdim, hidden_dim) = (5, 10)  — 5 rows x 10 cols, comma-separated
  asrc  : (hidden_dim,)      = (10,)    — 10 rows, one value per line
  atarg : (hidden_dim,)      = (10,)    — 10 rows, one value per line

The script prints summary statistics and saves the full (N, N) attention matrix
to siggat_output.npy (numpy) and siggat_output.csv (CSV, for Mathematica Import).

Usage as a library
------------------
    from siggat_layer import compute_siggat
    attn = compute_siggat(features, positions, radii, W, asrc, atarg)
"""

import argparse
import sys
import os
import numpy as np

# ── constants matching the Mathematica notebook ──────────────────────────────
FEATURE_COLS  = [1, 4, 6, 8, 10]   # 0-indexed columns used as node features
POSITION_COLS = [2, 3]              # 0-indexed: x, z
RADIUS_COL    = 1                   # 0-indexed
N_NODES       = 2000
MCONST        = -10.0
ALPHA         = 0.2                 # LeakyReLU negative slope


# ── core computation ──────────────────────────────────────────────────────────

def build_contact_matrix(
    positions: np.ndarray,
    radii:     np.ndarray,
) -> np.ndarray:
    """
    Binary contact matrix A[i,j] = 1 iff ||r_i - r_j|| < R_i + R_j, diagonal 0.

    positions : (N, 2)
    radii     : (N,)
    returns   : (N, N) float64
    """
    diff  = positions[:, None, :].astype(np.float64) - positions[None, :, :].astype(np.float64)
    dist  = np.sqrt((diff ** 2).sum(axis=-1))
    r_sum = radii[:, None].astype(np.float64) + radii[None, :].astype(np.float64)
    A     = (dist < r_sum).astype(np.float64)
    np.fill_diagonal(A, 0.0)
    return A


def compute_siggat(
    features:  np.ndarray,   # (N, fdim)
    positions: np.ndarray,   # (N, 2)
    radii:     np.ndarray,   # (N,)
    W:         np.ndarray,   # (fdim, hidden_dim)  — NOTE: row = input feature
    asrc:      np.ndarray,   # (hidden_dim,)
    atarg:     np.ndarray,   # (hidden_dim,)
    mconst:    float = MCONST,
    alpha:     float = ALPHA,
    verbose:   bool  = False,
) -> np.ndarray:
    """
    Returns the siggat attention weight matrix (N, N).

    Steps (matching Mathematica NetGraph):
      1. H        = features @ W                    (N, hidden_dim)
      2. src[i]   = H[i] · asrc                     (N,)
      3. tgt[i]   = H[i] · atarg                    (N,)
      4. e[i,j]   = src[j] + tgt[i]                 (N, N)  outer sum
      5. e        = LeakyReLU(e, alpha)
      6. A        = contact matrix, diagonal 0
      7. masked   = A * e + (1 - A) * mconst
      8. siggat   = sigmoid(masked)
    """
    def _stats(name, arr):
        a = arr.flatten()
        print(f"  {name}: mean={a.mean():.6f}  std={a.std():.6f}  "
              f"min={a.min():.6f}  max={a.max():.6f}  first3={a[:3]}")

    # 1. linear projection (no bias, matching Mathematica NetArrayLayer)
    H = features.astype(np.float64) @ W.astype(np.float64)    # (N, hidden_dim)
    if verbose: _stats("H        ", H)

    # 2–3. per-node attention scalars
    src_scores = H @ asrc.astype(np.float64)                   # (N,)
    tgt_scores = H @ atarg.astype(np.float64)                  # (N,)
    if verbose: _stats("src_scores", src_scores)
    if verbose: _stats("tgt_scores", tgt_scores)

    # 4. outer sum: e[i,j] = src[j] + tgt[i]  (Mathematica left=(1,N), right=(N,1))
    e = src_scores[np.newaxis, :] + tgt_scores[:, np.newaxis]  # (N, N)
    if verbose: _stats("e (pre-LReLU)", e)

    # 5. LeakyReLU
    e = np.where(e >= 0, e, alpha * e)
    if verbose: _stats("e (post-LReLU)", e)

    # 6. contact matrix
    A = build_contact_matrix(positions, radii)
    if verbose:
        n_contacts = int(A.sum())
        print(f"  contact pairs: {n_contacts}  (mean degree {n_contacts/A.shape[0]:.2f})")

    # 7. masking
    masked = A * e + (1.0 - A) * mconst

    if verbose:
        contact_masked = masked[masked > mconst + 1.0]
        _stats("masked (contacts)", contact_masked)

    # 8. sigmoid
    attn = 1.0 / (1.0 + np.exp(-masked))

    if verbose:
        contact_w = attn[A == 1]
        _stats("siggat (contacts)", contact_w)

    return attn.astype(np.float32), e, masked


# ── data loading helpers ──────────────────────────────────────────────────────

def load_packing(dat_path: str, packing_idx: int, n_nodes: int = N_NODES):
    """Extract one packing from a .dat file and return (features, positions, radii)."""
    data  = np.loadtxt(dat_path, dtype=np.float32)
    start = packing_idx * n_nodes
    end   = start + n_nodes
    if end > data.shape[0]:
        raise ValueError(
            f"packing_idx={packing_idx} out of range "
            f"(file has {data.shape[0] // n_nodes} packings)"
        )
    packing   = data[start:end]
    features  = packing[:, FEATURE_COLS]
    positions = packing[:, POSITION_COLS]
    radii     = packing[:, RADIUS_COL]
    return features, positions, radii


def load_weight(path: str, shape: tuple) -> np.ndarray:
    """
    Load a weight array from a CSV file exported by Mathematica.

    Mathematica Export["file.csv", matrix] convention:
      - 2-D array (e.g. W):   rows are comma-separated, one row per line -> (rows, cols)
      - 1-D list (e.g. asrc): one value per line, no commas            -> (n,)

    Both cases are handled by loadtxt with delimiter=",".
    """
    data = np.loadtxt(path, delimiter=",", dtype=np.float64).flatten()
    if data.size != np.prod(shape):
        raise ValueError(
            f"{path}: expected {np.prod(shape)} values for shape {shape}, "
            f"got {data.size}"
        )
    return data.reshape(shape).astype(np.float32)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute siggat attention weights for a single packing."
    )
    p.add_argument("--packing",      required=True, help="Path to .dat packing file")
    p.add_argument("--packing_idx",  type=int, default=0,
                   help="Which packing to use (0-indexed, default 0)")
    p.add_argument("--W",            required=True,
                   help="CSV file for W, shape (fdim, hidden_dim) = (5, 10); "
                        "export from Mathematica with Export[\"W.csv\", Normal@NetExtract[net, {\"W\", \"Array\"}]]")
    p.add_argument("--asrc",         required=True,
                   help="CSV file for asrc, shape (hidden_dim,) = (10,); "
                        "export with Export[\"asrc.csv\", Normal@NetExtract[net, {\"asrc\", \"Array\"}]]")
    p.add_argument("--atarg",        required=True,
                   help="CSV file for atarg, shape (hidden_dim,) = (10,); "
                        "export with Export[\"atarg.csv\", Normal@NetExtract[net, {\"atarg\", \"Array\"}]]")
    p.add_argument("--fdim",         type=int, default=5)
    p.add_argument("--hidden_dim",   type=int, default=10)
    p.add_argument("--n_nodes",      type=int, default=N_NODES)
    p.add_argument("--mconst",       type=float, default=MCONST)
    p.add_argument("--alpha",        type=float, default=ALPHA)
    p.add_argument("--out_csv",      default="siggat_output.csv",
                   help="Output CSV: i,j,alpha for each contact edge")
    p.add_argument("--verbose", action="store_true",
                   help="Print intermediate statistics (H, scores, e, contacts) "
                        "for comparison with Mathematica")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading packing {args.packing_idx} from {args.packing} ...")
    features, positions, radii = load_packing(
        args.packing, args.packing_idx, args.n_nodes
    )
    print(f"  features:  {features.shape}")
    print(f"  positions: {positions.shape}")
    print(f"  radii:     {radii.shape}")

    print("Loading weights ...")
    W     = load_weight(args.W,     (args.fdim, args.hidden_dim))
    asrc  = load_weight(args.asrc,  (args.hidden_dim,))
    atarg = load_weight(args.atarg, (args.hidden_dim,))
    print(f"  W:     {W.shape}  norm={np.linalg.norm(W):.4f}")
    print(f"  asrc:  {asrc}")
    print(f"  atarg: {atarg}")

    print("Computing siggat ...")
    attn, e_mat, masked_mat = compute_siggat(
        features, positions, radii, W, asrc, atarg,
        mconst=args.mconst, alpha=args.alpha, verbose=args.verbose,
    )
    print(f"  attn shape: {attn.shape}")

    # contact pairs only (non-contacts are sigmoid(-10) ≈ 0)
    A = build_contact_matrix(positions, radii)
    rows, cols = np.where(A == 1)

    contact_weights = attn[A == 1]
    print(f"\nContact-pair attention weights ({len(contact_weights)} pairs):")
    print(f"  mean  = {contact_weights.mean():.4f}")
    print(f"  std   = {contact_weights.std():.4f}")
    print(f"  min   = {contact_weights.min():.4f}")
    print(f"  max   = {contact_weights.max():.4f}")

    # output: i, j (1-indexed), alpha, x_i, z_i, x_j, z_j, r_i, r_j
    pos_i = positions[rows]          # (E, 2)
    pos_j = positions[cols]          # (E, 2)
    r_i   = radii[rows]              # (E,)
    r_j   = radii[cols]              # (E,)
    out = np.column_stack([
        rows + 1, cols + 1,
        contact_weights,
        pos_i, pos_j,
        r_i, r_j,
    ])
    np.savetxt(args.out_csv, out, delimiter=",",
               fmt="%d,%d,%.8f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
               header="i,j,alpha,x_i,z_i,x_j,z_j,r_i,r_j", comments="")
    print(f"\nSaved: {args.out_csv}  ({len(contact_weights)} contact edges, columns: i,j,alpha,x_i,z_i,x_j,z_j,r_i,r_j)")


if __name__ == "__main__":
    main()
