"""
GAT (Graph Attention Network) layer for MIRROR drug graph.

Standard GAT with edge-type awareness. Unlike HGT, GAT uses SHARED Q/K/V
projections across all edge types. Edge type is injected via a learned
embedding added to attention scores (GAMENet-era approach).

Run 26D ablation: tests whether HGT's per-relation parameters help or
just add noise for a single-node-type (drug-only) graph.
"""

import torch
import torch.nn as nn

from ..registry import GRAPH_LAYERS


@GRAPH_LAYERS.register("gat")
class GATLayer(nn.Module):
    """Standard GAT layer with edge-type awareness (simpler than HGT).

    Unlike HGT, GAT uses SHARED Q/K/V projections across all edge types.
    Edge type is injected via a learned embedding added to attention scores.
    This is the approach SafeDrug-era papers (GAMENet) use implicitly.

    Run 26D ablation: tests whether HGT's per-relation parameters help or
    just add noise for a single-node-type (drug-only) graph.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_edge_types: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_edge_types = num_edge_types

        # Shared Q, K, V projections (key difference from HGT)
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)

        # Edge type embedding: adds to attention scores per head
        self.edge_type_embed = nn.Embedding(num_edge_types, num_heads)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,            # (num_nodes, hidden_dim)
        edge_index: torch.Tensor,    # (2, num_edges)
        edge_type: torch.Tensor,     # (num_edges,)
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N = x.size(0)
        H = self.num_heads
        D = self.head_dim

        src_idx = edge_index[0]  # (E,)
        tgt_idx = edge_index[1]  # (E,)

        # Shared projections for all edges
        q = self.q_linear(x[tgt_idx]).view(-1, H, D)  # (E, H, D)
        k = self.k_linear(x[src_idx]).view(-1, H, D)
        v = self.v_linear(x[src_idx]).view(-1, H, D)

        # Attention scores + edge-type bias
        att = (q * k).sum(dim=-1, keepdim=True) / (D ** 0.5)  # (E, H, 1)
        type_bias = self.edge_type_embed(edge_type).unsqueeze(-1)  # (E, H, 1)
        att = att + type_bias

        if edge_weight is not None:
            w = edge_weight.clamp(min=1e-8)
            log_w = torch.log(w)
            # FIX: center co-occurrence weights so mean EHR edge gets 0 bias.
            # Raw log(P)<=0 for all P<=1 would crush all EHR edges relative to
            # DDI/self-loop (log(1)=0). Centering: high-P edges get +bias, low-P get -bias.
            # Only center EHR edges (type 1); DDI/self/ATC stay at 0.
            ehr_mask = (edge_type == 1)
            if ehr_mask.any():
                log_w = log_w.clone()
                ehr_log = log_w[ehr_mask]
                log_w[ehr_mask] = ehr_log - ehr_log.mean()
            att = att + log_w.unsqueeze(1).unsqueeze(2)

        att = att.clamp(-10, 10)

        # Softmax per target node (scatter-based)
        att_exp = att.exp()
        msg_acc = torch.zeros(N, H, D, device=x.device)
        att_acc = torch.zeros(N, H, 1, device=x.device)
        att_acc.index_add_(0, tgt_idx, att_exp)
        msg_acc.index_add_(0, tgt_idx, v * att_exp)

        att_acc = att_acc.clamp(min=1e-8)
        msg = (msg_acc / att_acc).view(N, -1)
        msg = self.attn_dropout(msg)

        out = self.out_linear(msg)
        out = self.dropout(out)
        return self.layer_norm(x + out)
