"""
GATsig - Graph Attention Network for shear jamming classification (sparse PyG version)

Architecture (multi-head, multi-layer GAT with sigma-gated attention):
  Inputs:
    features  : (N, fdim)   node features
    positions : (N, 2)      (x, y) particle positions
    radii     : (N,)        particle radii — used for contact detection

  Per GAT layer (GATsigLayer):
    1. K parallel attention heads, each with its own W_k, a_src_k, a_tgt_k
    2. For each contact edge (i,j): e_ij^k = LeakyReLU(a_src_k·H_j + a_tgt_k·H_i)
    3. Sigmoid gating: alpha_ij = sigmoid(e_ij)
       Non-contact edges never created → no masking needed
    4. Aggregation: H'^k_i = sum_{j in N(i)} alpha_ij * H^k_j  (sparse matmul)
    5. Concatenate heads -> linear projection back to hidden_dim

  Readout (after final layer):
    per-node: sigmoid(W_node @ h_i)  -> node_scores (N,)
    W3: Linear(N, N) -> softmax -> weighted sum -> scalar output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import MessagePassing
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


def build_contact_edges(positions: torch.Tensor, radii: torch.Tensor):
    """
    Build sparse edge index for contact graph (excluding self-loops).

    positions : (N, 2)
    radii     : (N,)
    returns   : edge_index (2, E)  LongTensor — row 0 = target i, row 1 = source j
    """
    diff  = positions.unsqueeze(1) - positions.unsqueeze(0)   # (N, N, 2)
    dist  = torch.norm(diff, dim=-1)                           # (N, N)
    r_sum = radii.unsqueeze(1) + radii.unsqueeze(0)           # (N, N)
    contact = (dist < r_sum) & ~torch.eye(dist.shape[0], dtype=torch.bool, device=positions.device)
    return contact.nonzero(as_tuple=False).t().contiguous()    # (2, E)


if _HAS_PYG:
    class GATsigLayer(MessagePassing):
        """
        Single GAT layer with K attention heads and sigma gating (PyG sparse version).

        Uses MessagePassing with aggr='add'.  For each contact edge (i->j):
            e_ij = LeakyReLU(a_src · H_j  +  a_tgt · H_i)
            alpha_ij = sigmoid(e_ij)
            msg_ij = alpha_ij * H_j

        Node update: h'_i = sum_{j} msg_ij
        """

        def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            n_heads: int = 1,
            mconst: float = -10.0,
            alpha: float = 0.2,
        ):
            super().__init__(aggr="add")
            self.n_heads   = n_heads
            self.mconst    = mconst      # kept for checkpoint compatibility; unused in sparse path
            self.alpha     = alpha
            self.head_dim  = hidden_dim

            self.W = nn.ModuleList([
                nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(n_heads)
            ])
            self.a_src = nn.ParameterList([
                nn.Parameter(torch.empty(hidden_dim)) for _ in range(n_heads)
            ])
            self.a_tgt = nn.ParameterList([
                nn.Parameter(torch.empty(hidden_dim)) for _ in range(n_heads)
            ])

            self.proj = nn.Linear(n_heads * hidden_dim, hidden_dim, bias=False) if n_heads > 1 else None
            self._init_weights()

        def _init_weights(self):
            for k in range(self.n_heads):
                nn.init.xavier_uniform_(self.W[k].weight)
                nn.init.xavier_uniform_(self.a_src[k].unsqueeze(0))
                nn.init.xavier_uniform_(self.a_tgt[k].unsqueeze(0))
            if self.proj is not None:
                nn.init.xavier_uniform_(self.proj.weight)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor, return_attention: bool = False):
            """
            x          : (N, in_dim)
            edge_index : (2, E)  — row 0 = target i, row 1 = source j
            returns    : (N, hidden_dim)  or  ((N, hidden_dim), list[(E,) per head])
            """
            head_outputs = []
            attn_per_head = []

            for k in range(self.n_heads):
                H = self.W[k](x)                                          # (N, head_dim)

                # per-node attention scalars
                src_scores = (H * self.a_src[k]).sum(dim=-1)              # (N,)  H·a_src
                tgt_scores = (H * self.a_tgt[k]).sum(dim=-1)              # (N,)  H·a_tgt

                # e_ij = LeakyReLU( src_scores[j] + tgt_scores[i] )
                i_nodes = edge_index[0]   # target
                j_nodes = edge_index[1]   # source
                e = src_scores[j_nodes] + tgt_scores[i_nodes]             # (E,)
                e = F.leaky_relu(e, negative_slope=self.alpha)
                attn = torch.sigmoid(e)                                    # (E,)  alpha_ij

                # aggregate: h'_i = sum_j  alpha_ij * H_j
                agg = self.propagate(edge_index, x=H, attn=attn)          # (N, head_dim)
                head_outputs.append(agg)
                if return_attention:
                    attn_per_head.append(attn)

            if self.n_heads == 1:
                out = head_outputs[0]
            else:
                out = self.proj(torch.cat(head_outputs, dim=-1))

            if return_attention:
                return out, attn_per_head
            return out

        def message(self, x_j: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
            # x_j : (E, head_dim)  — feature of source node j for each edge
            # attn: (E,)
            return attn.unsqueeze(-1) * x_j                               # (E, head_dim)

else:
    # Fallback dense implementation when PyG is not installed
    class GATsigLayer(nn.Module):
        """Dense fallback (identical to original) when torch_geometric is unavailable."""

        def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            n_heads: int = 1,
            mconst: float = -10.0,
            alpha: float = 0.2,
        ):
            super().__init__()
            self.n_heads  = n_heads
            self.mconst   = mconst
            self.alpha    = alpha
            self.head_dim = hidden_dim

            self.W = nn.ModuleList([
                nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(n_heads)
            ])
            self.a_src = nn.ParameterList([
                nn.Parameter(torch.empty(hidden_dim)) for _ in range(n_heads)
            ])
            self.a_tgt = nn.ParameterList([
                nn.Parameter(torch.empty(hidden_dim)) for _ in range(n_heads)
            ])
            self.proj = nn.Linear(n_heads * hidden_dim, hidden_dim, bias=False) if n_heads > 1 else None
            self._init_weights()

        def _init_weights(self):
            for k in range(self.n_heads):
                nn.init.xavier_uniform_(self.W[k].weight)
                nn.init.xavier_uniform_(self.a_src[k].unsqueeze(0))
                nn.init.xavier_uniform_(self.a_tgt[k].unsqueeze(0))
            if self.proj is not None:
                nn.init.xavier_uniform_(self.proj.weight)

        def forward(self, x, edge_index_or_A, return_attention=False):
            # edge_index_or_A accepted as dense (N,N) matrix in fallback mode
            A = edge_index_or_A
            head_outputs, attn_matrices = [], []
            for k in range(self.n_heads):
                H = self.W[k](x)
                src_scores = (H * self.a_src[k]).sum(dim=-1)
                tgt_scores = (H * self.a_tgt[k]).sum(dim=-1)
                e = src_scores.unsqueeze(0) + tgt_scores.unsqueeze(1)
                e = F.leaky_relu(e, negative_slope=self.alpha)
                masked = A * e + (1.0 - A) * self.mconst
                attn = torch.sigmoid(masked)
                head_outputs.append(attn @ H)
                if return_attention:
                    attn_matrices.append(attn)
            if self.n_heads == 1:
                out = head_outputs[0]
            else:
                out = self.proj(torch.cat(head_outputs, dim=-1))
            if return_attention:
                return out, attn_matrices
            return out


class GATsig(nn.Module):
    """
    Multi-head, multi-layer GATsig with sparse PyG attention.

    When PyG is available the contact graph is stored as edge_index (2×E)
    and attention is computed only on contact edges — O(E) instead of O(N²).

    W2 is replaced by a per-node linear (hidden_dim → 1) so the readout is
    also independent of n_nodes and works for variable-size graphs.

    Args:
        n_nodes    : number of particles (used only for W3 dense readout)
        fdim       : input feature dimension (default 5)
        hidden_dim : hidden dimension per layer (default 10)
        n_heads    : number of attention heads per layer (default 1)
        n_layers   : number of stacked GAT layers (default 1)
        mconst     : large negative mask constant (kept for compat, unused in sparse path)
        alpha      : LeakyReLU negative slope (default 0.2)
    """

    def __init__(
        self,
        n_nodes: int = 2000,
        fdim: int = 5,
        hidden_dim: int = 10,
        n_heads: int = 1,
        n_layers: int = 1,
        mconst: float = -10.0,
        alpha: float = 0.2,
    ):
        super().__init__()
        self.n_nodes    = n_nodes
        self.hidden_dim = hidden_dim

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

        # Readout
        # W_node: per-node projection  (hidden_dim → 1),  replaces the dense W2 flatten
        # W3: global dense re-weighting over N node scores (same as original)
        self.W_node = nn.Linear(hidden_dim, 1, bias=True)
        self.W3     = nn.Linear(n_nodes, n_nodes, bias=True)

        nn.init.xavier_uniform_(self.W_node.weight)
        nn.init.zeros_(self.W_node.bias)
        nn.init.xavier_uniform_(self.W3.weight)
        nn.init.zeros_(self.W3.bias)

    # ------------------------------------------------------------------
    @staticmethod
    def build_contact_matrix(positions: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
        """Dense contact matrix — kept for compatibility with analysis scripts."""
        diff    = positions.unsqueeze(1) - positions.unsqueeze(0)
        dist    = torch.norm(diff, dim=-1)
        r_sum   = radii.unsqueeze(1) + radii.unsqueeze(0)
        contact = (dist < r_sum).float()
        contact = contact * (1 - torch.eye(contact.shape[0], device=positions.device))
        return contact

    # ------------------------------------------------------------------
    def forward(
        self,
        features:  torch.Tensor,   # (N, fdim)
        positions: torch.Tensor,   # (N, 2)
        radii:     torch.Tensor,   # (N,)
        return_attention: bool = False,
    ):
        """
        Returns scalar logit (pre-sigmoid) for DST probability.

        If return_attention=True also returns attention data:
          - PyG path  : list[layer] of list[head] of (E,) edge-weight tensors
          - dense path: list[layer] of list[head] of (N,N) matrices
        """
        if _HAS_PYG:
            edge_index = build_contact_edges(positions, radii)  # (2, E)
            graph_repr = edge_index
        else:
            graph_repr = self.build_contact_matrix(positions, radii)

        h = features
        all_attns = []
        for layer in self.layers:
            if return_attention:
                h, attn = layer(h, graph_repr, return_attention=True)
                all_attns.append(attn)
            else:
                h = layer(h, graph_repr)
            h = torch.sigmoid(h)                                 # nlin: matches Mathematica adoth->nlin

        # Readout: per-node score then global weighted sum
        node_scores = torch.sigmoid(self.W_node(h).squeeze(-1))  # (N,)  replaces W2 + nlin2
        weights     = F.softmax(self.W3(node_scores), dim=0)     # (N,)  softmax2
        scalar      = (weights * node_scores).sum()              # scalar dotprob

        if return_attention:
            return scalar, all_attns
        return scalar
