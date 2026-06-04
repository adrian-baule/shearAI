"""
GATsig - Graph Attention Network for shear jamming classification
Translated from Mathematica/Wolfram NetGraph (GATsig_layer1head1.nb)

Architecture summary (single-head, single-layer GAT with sigma-gated attention):
  Inputs:
    features  : (N, 5)   node features from columns [2,5,7,9,11] of raw data
    positions : (N, 2)   (x, y) particle positions from columns [3,4]
    radii     : (N,)     particle radii from column [2] — used for contact detection

  Forward pass:
    1. Build contact adjacency: A_ij = 1 if |r_i - r_j| < R_i + R_j, 0 on diagonal
    2. Linear projection: H = X @ W   (N, fdim=5) -> (N, newfdim=10)
    3. Attention logits: e_ij = LeakyReLU(a_src · h_i + a_tgt · h_j)
    4. Mask non-contacts with large negative constant, then sigmoid (not softmax)
    5. Weighted aggregation: h'_i = sigma(A_ij) @ H
    6. MLP readout: Linear(N,N) -> sigmoid -> Linear(N,N) -> softmax -> dot -> scalar
  Output:
    scalar ∈ (0,1)  — probability of DST (discontinuous shear thickening)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATsig(nn.Module):
    """
    Single-head GAT with sigmoid attention gating for particle packings.

    Args:
        n_nodes  : number of particles (fixed per dataset, default 2000)
        fdim     : input feature dimension (default 5)
        newfdim  : attention embedding dimension (default 10)
        mconst   : large negative mask constant for non-contacts (default -10)
        alpha    : LeakyReLU negative slope (default 0.2)
    """

    def __init__(
        self,
        n_nodes: int = 2000,
        fdim: int = 5,
        newfdim: int = 10,
        mconst: float = -10.0,
        alpha: float = 0.2,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.fdim = fdim
        self.newfdim = newfdim
        self.mconst = mconst
        self.alpha = alpha

        # --- Layer definitions (matching Mathematica NetGraph) ---

        # W: linear projection of node features  (fdim -> newfdim)
        self.W = nn.Linear(fdim, newfdim, bias=False)

        # Attention vectors (a_src, a_tgt each of size newfdim)
        self.a_src = nn.Parameter(torch.empty(newfdim))
        self.a_tgt = nn.Parameter(torch.empty(newfdim))

        # Readout MLP
        # W2: (n_nodes, newfdim) -> (n_nodes,)   applied per row via Linear
        self.W2 = nn.Linear(newfdim, 1, bias=True)   # equivalent to W2 in Mathematica
        # W3: (n_nodes,) -> (n_nodes,)
        self.W3 = nn.Linear(n_nodes, n_nodes, bias=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.xavier_uniform_(self.W3.weight)
        nn.init.zeros_(self.W2.bias)
        nn.init.zeros_(self.W3.bias)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_tgt.unsqueeze(0))

    # ------------------------------------------------------------------
    # Contact adjacency (replaces ThreadingLayer[If[sqrt(dx²+dy²) < ri+rj]])
    # ------------------------------------------------------------------
    @staticmethod
    def build_contact_matrix(positions: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
        """
        Returns binary contact matrix A ∈ {0,1}^(N×N).
        A_ij = 1 if particle i and j are in contact (centres closer than sum of radii)
        Diagonal is zeroed (no self-contact).

        positions : (N, 2)
        radii     : (N,)
        returns   : (N, N) float
        """
        N = positions.shape[0]
        # Pairwise displacement
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)      # (N, N, 2)
        dist = torch.norm(diff, dim=-1)                              # (N, N)
        # Sum of radii for each pair
        r_sum = radii.unsqueeze(1) + radii.unsqueeze(0)             # (N, N)
        contact = (dist < r_sum).float()
        # Zero diagonal (no self-attention)
        contact = contact * (1 - torch.eye(N, device=positions.device))
        return contact

    # ------------------------------------------------------------------
    def forward(
        self,
        features: torch.Tensor,   # (N, fdim)
        positions: torch.Tensor,  # (N, 2)
        radii: torch.Tensor,      # (N,)
    ) -> torch.Tensor:
        """Returns scalar logit (pre-sigmoid) for DST probability."""
        N = features.shape[0]

        # 1. Build contact adjacency
        A = self.build_contact_matrix(positions, radii)              # (N, N)

        # 2. Linear projection of node features
        H = self.W(features)                                         # (N, newfdim)

        # 3. Attention logits via outer sum of attention scalars
        #    e_ij = a_src · h_i + a_tgt · h_j
        src_scores = (H * self.a_src).sum(dim=-1)                    # (N,)
        tgt_scores = (H * self.a_tgt).sum(dim=-1)                    # (N,)
        e = src_scores.unsqueeze(1) + tgt_scores.unsqueeze(0)       # (N, N)  outer sum

        # 4. LeakyReLU (alpha=0.2, matching Mathematica 0.2*# branch)
        e = F.leaky_relu(e, negative_slope=self.alpha)

        # 5. Mask non-contacts with mconst then apply sigmoid
        #    Mathematica: masking = contact*e + (1 - contact)*mconst
        masked = A * e + (1 - A) * self.mconst                      # (N, N)
        attn = torch.sigmoid(masked)                                 # (N, N)  siggat

        # 6. Aggregate: H' = attn @ H   (N, N) x (N, newfdim) -> (N, newfdim)
        H_agg = attn @ H                                             # (N, newfdim)

        # 7. Readout MLP
        #    W2: (N, newfdim) -> (N, 1) -> squeeze -> (N,)
        out = torch.sigmoid(self.W2(H_agg).squeeze(-1))             # (N,)
        #    W3: (N,) -> (N,), then softmax
        out = F.softmax(self.W3(out), dim=0)                        # (N,)
        #    dot with H_agg aggregated output -> scalar
        # In Mathematica: dotprob = dot(softmax2_out, nlin2_out [per node])
        # Equivalent to weighted mean of per-node outputs
        scalar = (out * torch.sigmoid(self.W2(H_agg).squeeze(-1))).sum()  # scalar

        return scalar   # logit-like scalar; wrap in sigmoid for probability
