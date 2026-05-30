import torch
import torch.nn as nn
from ..registry import GRAPH_LAYERS

@GRAPH_LAYERS.register("sage")
class SAGELayer(nn.Module):
    """GraphSAGE layer with per-relation mean aggregation.

    Each edge type is mean-aggregated independently, then contributions are
    summed before the concat-and-project. Mixing all types into one mean is
    wrong: a drug with 50 DDI neighbours and 2 EHR neighbours would give the
    EHR signal weight 2/52 ≈ 4%, destroying the co-prescription information.
    Per-type mean gives every relation equal standing before the projection.
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.3, **kwargs):
        super().__init__()
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
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

        # Per-relation mean aggregation, then sum across relations.
        agg = torch.zeros(N, x.size(1), device=x.device)
        for t in edge_type.unique():
            mask = edge_type == t
            t_src = src[mask]
            t_tgt = tgt[mask]
            t_w   = w[mask]

            t_msg = x[t_src] * t_w.unsqueeze(-1)
            t_agg = torch.zeros(N, x.size(1), device=x.device)
            t_agg.index_add_(0, t_tgt, t_msg)

            deg = torch.zeros(N, device=x.device).index_add_(0, t_tgt, t_w).clamp(min=1e-5)
            agg = agg + t_agg / deg.unsqueeze(-1)

        out = torch.cat([x, agg], dim=-1)
        out = self.linear(out)
        out = self.act(out)
        out = self.dropout(out)
        return self.layer_norm(x + out)
