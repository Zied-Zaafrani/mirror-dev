import torch
import torch.nn as nn
from ..registry import GRAPH_LAYERS

@GRAPH_LAYERS.register("digcn")
class DiGCNLayer(nn.Module):
    """
    HI-DR style directed GCN layer.
    Directly uses the asymmetric edge probabilities P(j|i) without symmetric normalization.
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
        
        # In DiGCN, the weights are already normalized/asymmetric probabilities.
        # So we just multiply and sum, no degree normalization.
        msg = x[src] * w.unsqueeze(-1)
        
        out = torch.zeros(N, x.size(1), device=x.device)
        out.index_add_(0, tgt, msg)
        
        out = self.linear(out)
        out = self.act(out)
        out = self.dropout(out)
        
        return self.layer_norm(x + out)
