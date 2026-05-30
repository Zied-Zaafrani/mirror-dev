"""
HAN (Heterogeneous Attention Network) layer for MIRROR drug graph.

DRecHGR-adapted for MIRROR's drug-only graph (Run 29 P1-B).
Replaces HGT's degenerate single-node-type per-edge-type projections with
two semantically meaningful channels:
  Channel 1 (pharmacological): DDI (type 0) + EHR co-occurrence (type 1) + self-loop (type 2)
  Channel 2 (semantic):        ATC-class (type 3) + LEDG cosine (type 4) + self-loop (type 2)

Each channel runs an independent GAT with shared Q/K/V projections per channel.
SemanticAttention then learns per-node weights over the two channel outputs.

Requires --add_semantic_edges for Channel 2 to be non-trivial (type 4 edges must exist).
Falls back gracefully when type 4 edges are absent (Channel 2 = ATC + self-loop only).

Reference: Yang et al. 2019 — Heterogeneous Graph Attention Network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import GRAPH_LAYERS


class SemanticAttention(nn.Module):
    """Learned per-channel attention over HANLayer meta-path channels (HAN paper style).

    Uses a global learned attention vector as the query — NOT the channel mean.
    Avoids the self-referential bias where a dominant channel (e.g. Channel 1 with more edges)
    reinforces its own weight via the mean query. Reference: Yang et al. 2019 HAN, Eq. 5-6.
    """

    def __init__(self, hidden_dim: int, num_channels: int = 2):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        # Global attention vector — learned independently of channel content
        self.attn_vec = nn.Parameter(torch.empty(hidden_dim))
        nn.init.xavier_uniform_(self.attn_vec.unsqueeze(0))

    def forward(self, channels: list) -> torch.Tensor:
        """
        Args:
            channels: list of C tensors, each (N, hidden_dim)
        Returns:
            out: (N, hidden_dim) — weighted combination of channels
        """
        stacked = torch.stack(channels, dim=1)              # (N, C, D)
        e = torch.tanh(self.proj(stacked))                  # (N, C, D)
        scores = (e * self.attn_vec).sum(dim=-1)            # (N, C)
        attn = F.softmax(scores, dim=-1)                    # (N, C)
        self._last_attn = attn.detach()  # expose for logging: model.drug_gnn.hgt_layers[0].semantic_attn._last_attn.mean(0)
        return (stacked * attn.unsqueeze(-1)).sum(dim=1)    # (N, D)


@GRAPH_LAYERS.register("han")
class HANLayer(nn.Module):
    """Heterogeneous Attention Network layer — decoupled 2-channel GAT + semantic attention.

    DRecHGR-adapted for MIRROR's drug-only graph (Run 29 P1-B).
    Replaces HGT's degenerate single-node-type per-edge-type projections with
    two semantically meaningful channels:
      Channel 1 (pharmacological): DDI (type 0) + EHR co-occurrence (type 1) + self-loop (type 2)
      Channel 2 (semantic):        ATC-class (type 3) + LEDG cosine (type 4) + self-loop (type 2)

    Each channel runs an independent GAT with shared Q/K/V projections per channel.
    SemanticAttention then learns per-node weights over the two channel outputs.

    Requires --add_semantic_edges for Channel 2 to be non-trivial (type 4 edges must exist).
    Falls back gracefully when type 4 edges are absent (Channel 2 = ATC + self-loop only).
    """

    # Edge type assignments (kept in sync with drug_gnn.py module docstring)
    CH1_TYPES = (0, 1, 2)  # pharmacological channel
    CH2_TYPES = (2, 3, 4)  # semantic channel

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_edge_types: int = 4,  # kept for API compatibility with HGTLayer/GATLayer
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Independent Q/K/V per channel — each channel learns its own attention
        self.q_ch1 = nn.Linear(hidden_dim, hidden_dim)
        self.k_ch1 = nn.Linear(hidden_dim, hidden_dim)
        self.v_ch1 = nn.Linear(hidden_dim, hidden_dim)
        self.q_ch2 = nn.Linear(hidden_dim, hidden_dim)
        self.k_ch2 = nn.Linear(hidden_dim, hidden_dim)
        self.v_ch2 = nn.Linear(hidden_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.semantic_attn = SemanticAttention(hidden_dim, num_channels=2)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _channel_forward(
        self,
        x: torch.Tensor,                    # (N, hidden_dim)
        edge_index: torch.Tensor,           # (2, E_ch) — edges for this channel
        edge_weight_ch: torch.Tensor | None,  # (E_ch,) — co-occurrence weights for this channel
        q_lin, k_lin, v_lin,               # channel-specific Q/K/V projections
    ) -> torch.Tensor:
        """Run single-channel GAT over the given edge subset.

        Returns zero message if no edges (identity via residual in forward).
        Applies centered log(edge_weight) bias matching GATLayer / HGTLayer treatment.
        """
        N = x.size(0)
        H = self.num_heads
        D = self.head_dim

        if edge_index.size(1) == 0:
            return torch.zeros(N, H * D, device=x.device)

        src_idx = edge_index[0]
        tgt_idx = edge_index[1]

        q = q_lin(x[tgt_idx]).view(-1, H, D)  # (E, H, D)
        k = k_lin(x[src_idx]).view(-1, H, D)
        v = v_lin(x[src_idx]).view(-1, H, D)

        att = (q * k).sum(dim=-1, keepdim=True) * self.scale  # (E, H, 1)

        # Apply centered log(edge_weight) bias — matching GATLayer / HGTLayer exactly.
        # Only EHR co-occurrence edges (weight = P(j|i) < 1.0) get centered bias.
        # DDI / ATC / self-loop / LEDG edges all have weight=1.0 → log=0 → zero bias.
        # BUG FIXED: prior code used log_w.std() to gate centering, which centered ALL
        # edge types together. When EHR edges dominate with negative log(P), DDI and
        # self-loop edges (log=0) were shifted to large positive bias — inflating DDI
        # attention relative to GATLayer where DDI stays at exactly 0 bias.
        # Fix: center only within the non-unity weight edges (co-occurrence only).
        if edge_weight_ch is not None:
            w = edge_weight_ch.clamp(min=1e-8)
            log_w = torch.log(w)
            ehr_mask = log_w.abs() > 1e-4   # weight != 1.0 → EHR co-occurrence
            if ehr_mask.any():
                log_w = log_w.clone()
                log_w[ehr_mask] = log_w[ehr_mask] - log_w[ehr_mask].mean()
            att = att + log_w.view(-1, 1, 1)

        att = att.clamp(-10, 10)

        att_exp = att.exp()
        msg_acc = torch.zeros(N, H, D, device=x.device)
        att_acc = torch.zeros(N, H, 1, device=x.device)
        att_acc.index_add_(0, tgt_idx, att_exp)
        msg_acc.index_add_(0, tgt_idx, v * att_exp)

        att_acc = att_acc.clamp(min=1e-8)
        msg = (msg_acc / att_acc).view(N, -1)  # (N, hidden_dim)
        return self.attn_dropout(msg)

    def forward(
        self,
        x: torch.Tensor,            # (num_nodes, hidden_dim)
        edge_index: torch.Tensor,   # (2, num_edges)
        edge_type: torch.Tensor,    # (num_edges,)
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = x.device
        ch1_types = torch.tensor(self.CH1_TYPES, device=device)
        ch2_types = torch.tensor(self.CH2_TYPES, device=device)

        ch1_mask = torch.isin(edge_type, ch1_types)
        ch2_mask = torch.isin(edge_type, ch2_types)

        # Pass edge_weight subset per channel so co-occurrence direction is not silently dropped
        ew_ch1 = edge_weight[ch1_mask] if edge_weight is not None else None
        ew_ch2 = edge_weight[ch2_mask] if edge_weight is not None else None

        h1 = F.elu(self._channel_forward(x, edge_index[:, ch1_mask], ew_ch1, self.q_ch1, self.k_ch1, self.v_ch1))
        h2 = F.elu(self._channel_forward(x, edge_index[:, ch2_mask], ew_ch2, self.q_ch2, self.k_ch2, self.v_ch2))

        combined = self.semantic_attn([h1, h2])  # (N, hidden_dim)
        out = self.dropout(self.out_linear(combined))
        return self.layer_norm(x + out)
