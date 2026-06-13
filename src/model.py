"""
GATsig - Graph Attention Network for shear jamming classification

Architecture (multi-head, multi-layer GAT with sigma-gated attention):
  Inputs:
    features  : (N, fdim)   node features
    positions : (N, 2)      (x, y) particle positions
    radii     : (N,)        particle radii — used for contact detection

  Per GAT layer (GATsigLayer):
    1. K parallel attention heads, each with its own W_k, a_src_k, a_tgt_k
    2. Attention logit:  e_ij^k = LeakyReLU(a_src_k · h_i^k + a_tgt_k · h_j^k)
    3. Mask non-contacts then sigmoid gating
    4. Aggregation:      H'^k_i = sigmoid(A) @ H^k
    5. Concatenate heads -> linear projection back to hidden_dim

  Readout (after final layer):
    Linear(hidden_dim, 1) applied per node -> sigmoid -> node_scores (N,)
    Linear(N, N) -> softmax -> weighted sum -> scalar output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATsigLayer(nn.Module):
    """
    Single GAT layer with K attention heads and sigma gating.

    Each head independently projects, computes attention, and aggregates.
    Head outputs are concatenated then projected back to hidden_dim.

    Args:
        in_dim     : input feature dimension
        hidden_dim : output feature dimension (after head projection)
        n_heads    : number of attention heads
        mconst     : large negative mask constant for non-contacts
        alpha      : LeakyReLU negative slope
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_heads: int = 1,
        mconst: float = -50.0,
        alpha: float = 0.2,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.mconst = mconst
        self.alpha = alpha
        # dimension per head before concatenation
        self.head_dim = hidden_dim

        # Per-head projection and attention parameters
        self.W = nn.ModuleList([
            nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(n_heads)
        ])
        self.a_src = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_dim)) for _ in range(n_heads)
        ])
        self.a_tgt = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_dim)) for _ in range(n_heads)
        ])

        # Multi-head projection: only added when n_heads > 1.
        # For n_heads=1, adoth goes directly to the caller (matching Mathematica exactly).
        self.proj = nn.Linear(n_heads * hidden_dim, hidden_dim, bias=False) if n_heads > 1 else None

        self._init_weights()

    def _init_weights(self):
        for k in range(self.n_heads):
            nn.init.xavier_uniform_(self.W[k].weight)
            # Small normal init for a_src/a_tgt: zero init gives identical node features
            # (uniform GAT attention) → gate_nn sees same input for every node → uniform softmax.
            nn.init.normal_(self.a_src[k], std=0.01)
            nn.init.normal_(self.a_tgt[k], std=0.01)
        if self.proj is not None:
            nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor, A: torch.Tensor, return_attention: bool = False):
        """
        x : (N, in_dim)
        A : (N, N) binary contact matrix
        return_attention : if True, also return list of (N, N) attention matrices per head

        returns: (N, hidden_dim)  or  ((N, hidden_dim), list[(N, N)])
        """
        head_outputs = []
        attn_matrices = []
        for k in range(self.n_heads):
            H = self.W[k](x)                                             # (N, head_dim)
            src_scores = (H * self.a_src[k]).sum(dim=-1)                 # (N,) H[i]·a_src
            tgt_scores = (H * self.a_tgt[k]).sum(dim=-1)                 # (N,) H[i]·a_tgt
            # outersum[i,j] = H[j]·a_src + H[i]·a_tgt  (matches Mathematica left/right reshape)
            e = src_scores.unsqueeze(0) + tgt_scores.unsqueeze(1)        # (N, N)
            e = F.leaky_relu(e, negative_slope=self.alpha)
            masked = A * e + (1.0 - A) * self.mconst                     # (N, N)
            attn = torch.sigmoid(masked)                                  # (N, N)
            head_outputs.append(attn @ H)                                 # (N, head_dim)
            if return_attention:
                attn_matrices.append(attn)

        if self.n_heads == 1:
            out = head_outputs[0]                                         # (N, hidden_dim) — no proj, matches Mathematica
        else:
            out = self.proj(torch.cat(head_outputs, dim=-1))              # (N, hidden_dim)

        if return_attention:
            return out, attn_matrices
        return out


class GlobalAttentionPooling(nn.Module):
    """
    GlobalAttention graph pooling (Li et al., 2016).

    For each node i with hidden vector h_i:
        gate_i  = sigmoid( W_gate · h_i + b_gate )        scalar gate
        feat_i  = W_feat · h_i + b_feat                   feature projection (pool_dim,)
        r       = sum_i  gate_i * feat_i                   graph-level vector (pool_dim,)
        output  = sigmoid( W_out · r + b_out )             scalar probability

    Args:
        hidden_dim : dimension of incoming node features
        pool_dim   : intermediate graph-level representation size (default 1)
    """

    def __init__(self, hidden_dim: int, pool_dim: int = 1):
        super().__init__()
        self.gate_nn = nn.Linear(hidden_dim, 1,        bias=True)
        self.feat_nn = nn.Linear(hidden_dim, pool_dim, bias=True)
        self.out_nn  = nn.Linear(pool_dim,   1,        bias=True)

        nn.init.xavier_uniform_(self.gate_nn.weight); nn.init.zeros_(self.gate_nn.bias)
        nn.init.xavier_uniform_(self.feat_nn.weight); nn.init.zeros_(self.feat_nn.bias)
        nn.init.xavier_uniform_(self.out_nn.weight);  nn.init.zeros_(self.out_nn.bias)

    def forward(self, h: torch.Tensor):
        """h : (N, hidden_dim)  →  scalar probability in (0, 1)"""
        gates  = F.softmax(self.gate_nn(h), dim=0)        # (N, 1) sums to 1 across nodes
        feats  = self.feat_nn(h)                          # (N, pool_dim)
        r      = (gates * feats).sum(dim=0, keepdim=True) # (1, pool_dim) — softmax already normalises
        return torch.sigmoid(self.out_nn(r)).squeeze()    # scalar


class GATsig(nn.Module):
    """
    Multi-head, multi-layer GAT with sigmoid attention gating and
    GlobalAttention graph pooling readout.

    Args:
        n_nodes    : number of particles (fixed per dataset, default 2000)
        fdim       : input feature dimension (default 5)
        hidden_dim : hidden dimension per layer (default 10)
        n_heads    : number of attention heads per layer (default 1)
        n_layers   : number of stacked GAT layers (default 1)
        mconst     : large negative mask constant for non-contacts (default -50)
        alpha      : LeakyReLU negative slope (default 0.2)
        pool_dim   : GlobalAttention intermediate dimension (default 1)
    """

    def __init__(
        self,
        n_nodes: int = 2000,
        fdim: int = 5,
        hidden_dim: int = 10,
        n_heads: int = 1,
        n_layers: int = 1,
        mconst: float = -50.0,
        alpha: float = 0.2,
        pool_dim: int = 1,
    ):
        super().__init__()
        self.n_nodes = n_nodes

        # Stack of GAT layers; first layer takes fdim, rest take hidden_dim
        dims = [fdim] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList([
            GATsigLayer(
                in_dim=dims[i],
                hidden_dim=dims[i + 1],
                n_heads=n_heads,
                mconst=mconst,
                alpha=alpha,
            )
            for i in range(n_layers)
        ])

        # GlobalAttention readout
        self.pooling = GlobalAttentionPooling(hidden_dim, pool_dim)

    # ------------------------------------------------------------------
    @staticmethod
    def build_contact_matrix(positions: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
        """
        Returns binary contact matrix A ∈ {0,1}^(N×N).
        A_ij = 1 if particle i and j are in contact (distance < sum of radii).
        Diagonal is zeroed (no self-contact).

        positions : (N, 2)
        radii     : (N,)
        returns   : (N, N) float
        """
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)           # (N, N, 2)
        dist = torch.norm(diff, dim=-1)                                   # (N, N)
        r_sum = radii.unsqueeze(1) + radii.unsqueeze(0)                  # (N, N)
        contact = (dist < r_sum).float()
        contact = contact * (1 - torch.eye(contact.shape[0], device=positions.device))
        return contact

    # ------------------------------------------------------------------
    def forward(
        self,
        features: torch.Tensor,   # (N, fdim)
        positions: torch.Tensor,  # (N, 2)
        radii: torch.Tensor,      # (N,)
        return_attention: bool = False,
    ):
        """
        Returns scalar logit (pre-sigmoid) for DST probability.

        If return_attention=True, also returns a list of attention matrices,
        one per layer, each being a list of (N, N) tensors (one per head).
        Shape: [ layer0_heads[(N,N), ...], layer1_heads[(N,N), ...], ... ]
        """
        A = self.build_contact_matrix(positions, radii)                  # (N, N)

        h = features
        all_attns = []
        for layer in self.layers:
            if return_attention:
                h, attn_matrices = layer(h, A, return_attention=True)
                all_attns.append(attn_matrices)
            else:
                h = layer(h, A)                                          # (N, hidden_dim)
            h = torch.tanh(h)                                            # nlin: avoids sigmoid saturation to 1

        # GlobalAttention readout: gated sum over nodes → scalar probability
        scalar = self.pooling(h)                                         # scalar in (0, 1)

        if return_attention:
            return scalar, all_attns
        return scalar   # probability in (0, 1); use BCELoss
