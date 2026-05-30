"""
HGT (Heterogeneous Graph Transformer) layer for MIRROR drug graph.

Simplified HGT for 1 node type, K edge types. Each edge type gets its own
K/Q/V linear projections and message weighting.

Reference: Hu et al. 2020 — Heterogeneous Graph Transformer.
"""

import torch
import torch.nn as nn

from ..registry import GRAPH_LAYERS


@GRAPH_LAYERS.register("hgt")
class HGTLayer(nn.Module):
    """A single Heterogeneous Graph Transformer layer.

    Simplified HGT for 1 node type, K edge types. Each edge type gets its own
    K/Q/V linear projections and message weighting.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_edge_types: int = 4,  # FIX (BUG-ATC-PHANTOM): was 2, silently dropped ATC edges (type 3)
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_edge_types = num_edge_types

        # Per-relation-type Q, K, V projections
        self.q_linears = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_edge_types)
        ])
        self.k_linears = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_edge_types)
        ])
        self.v_linears = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_edge_types)
        ])

        # Per-relation attention prior (learnable scalar importance)
        self.relation_pri = nn.Parameter(torch.ones(num_edge_types, num_heads))
        self.relation_att = nn.Parameter(torch.ones(num_edge_types, num_heads))

        # M6 fix: attention dropout — standard Transformer practice.
        # Dropout on attention weights after softmax, before value aggregation.
        # Prevents over-reliance on specific drug-drug attention patterns.
        self.attn_dropout = nn.Dropout(dropout)

        # Output projection + skip connection
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,            # (num_nodes, hidden_dim)
        edge_index: torch.Tensor,    # (2, num_edges) — [source, target]
        edge_type: torch.Tensor,     # (num_edges,) — type in [0, K-1]
        edge_weight: torch.Tensor | None = None,  # (num_edges,) — Phase 2 directed weights
    ) -> torch.Tensor:
        """
        Returns:
            x_out: (num_nodes, hidden_dim) — updated node features
        """
        N = x.size(0)
        H = self.num_heads
        D = self.head_dim

        # Initialize message accumulator
        msg_acc = torch.zeros(N, H, D, device=x.device)
        att_acc = torch.zeros(N, H, 1, device=x.device)

        for etype in range(self.num_edge_types):
            # Get edges of this type
            mask = edge_type == etype
            if not mask.any():
                continue

            src_idx = edge_index[0, mask]
            tgt_idx = edge_index[1, mask]

            # Compute Q, K, V with type-specific projections
            q = self.q_linears[etype](x[tgt_idx]).view(-1, H, D)
            k = self.k_linears[etype](x[src_idx]).view(-1, H, D)
            v = self.v_linears[etype](x[src_idx]).view(-1, H, D)

            # Attention scores
            att = (q * k).sum(dim=-1, keepdim=True) / (D ** 0.5)  # (E, H, 1)
            att = att * self.relation_att[etype].unsqueeze(-1)      # relation weight
            att = att + self.relation_pri[etype].unsqueeze(-1)      # relation prior

            # Phase 2: Directed edge weight bias (L1 — HI-DR).
            # w(i→j) = P(j|i) for co-occurrence edges; 1.0 for DDI/ATC/self-loop.
            # BUG FIX: must use log(w/mean_w), NOT log(w).
            # Since P(j|i) ≤ 1, log(P) ≤ 0 for ALL co-occur edges, which crushes
            # their entire type vs DDI/ATC/self-loop (log(1.0)=0 for those).
            # Centering: high-P edges get positive bias, low-P get negative bias,
            # mean-P edge is neutral — preserves total co-occur attention share.
            if edge_weight is not None:
                w = edge_weight[mask].clamp(min=1e-8)
                # Only center if this edge type has variable weights
                # (type 1 = co-occur; others are all 1.0 so log=0 already)
                if etype == 1:
                    log_w = torch.log(w)
                    log_w = log_w - log_w.mean()  # center: mean co-occur edge gets 0 bias
                else:
                    log_w = torch.log(w)  # 1.0 → 0.0, no change for DDI/ATC/self-loop
                att = att + log_w.unsqueeze(1).unsqueeze(2)  # (E, 1, 1)

            att = att.clamp(-10, 10)  # prevent exp() overflow → inf → NaN

            # Softmax per target node (scatter-based)
            att_exp = att.exp()
            # Accumulate for normalization (no dropout here — dropout
            # before normalization corrupts the softmax distribution)
            att_acc.index_add_(0, tgt_idx, att_exp)
            msg_acc.index_add_(0, tgt_idx, v * att_exp)

        # Normalize first, then apply attention dropout
        att_acc = att_acc.clamp(min=1e-8)
        msg = (msg_acc / att_acc).view(N, -1)  # (N, hidden_dim)
        msg = self.attn_dropout(msg)  # M6: dropout after normalized attention

        # Output projection + residual + layer norm
        out = self.out_linear(msg)
        out = self.dropout(out)
        x_out = self.layer_norm(x + out)
        return x_out
