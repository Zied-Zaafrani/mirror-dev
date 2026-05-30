import torch
import torch.nn as nn
from ..registry import GRAPH_LAYERS

@GRAPH_LAYERS.register("gcn")
class GCNLayer(nn.Module):
    """R-GCN style GCN layer with per-relation degree normalization.

    Each edge type is aggregated separately with its own D^{-1/2} A D^{-1/2}
    normalization, then contributions are summed. This prevents high-degree
    edge types (e.g. DDI with hundreds of edges) from diluting the signal
    from low-degree types (e.g. EHR co-occurrence with a handful of edges).

    Shares one linear projection across all edge types (simpler than full
    R-GCN which uses per-type W_r). Sufficient for the drug graph where
    edge-type semantics are partially captured by the weight values.
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.3, **kwargs):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N = x.size(0)
        src = edge_index[0]
        tgt = edge_index[1]
        w = torch.ones(src.size(0), device=x.device) if edge_weight is None else edge_weight

        # Per-relation D^{-1/2} A D^{-1/2} normalization then sum.
        # Without this, one dominant type (e.g. DDI with 800+ edges) absorbs
        # most of the degree budget and crushes signal from EHR/ATC types.
        out = torch.zeros(N, x.size(1), device=x.device)
        for t in edge_type.unique():
            mask = edge_type == t
            t_src = src[mask]
            t_tgt = tgt[mask]
            t_w   = w[mask]

            deg_s = torch.zeros(N, device=x.device).index_add_(0, t_src, t_w)
            deg_t = torch.zeros(N, device=x.device).index_add_(0, t_tgt, t_w)
            ds_inv = deg_s.pow(-0.5).clamp(max=1e4); ds_inv[deg_s == 0] = 0
            dt_inv = deg_t.pow(-0.5).clamp(max=1e4); dt_inv[deg_t == 0] = 0

            norm_w = t_w * ds_inv[t_src] * dt_inv[t_tgt]
            out.index_add_(0, t_tgt, x[t_src] * norm_w.unsqueeze(-1))

        out = self.linear(out)
        out = self.act(out)
        out = self.dropout(out)
        return self.layer_norm(x + out)
