import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from ..registry import AGGREGATORS

logger = logging.getLogger(__name__)

@AGGREGATORS.register("attention")
@AGGREGATORS.register("attention_residual")
class AttentionAggregator(nn.Module):
    """
    Query-Key Attention Aggregator.
    Uses the last hidden state as a query over all sequence states.
    Optional last-visit residual shortcut.
    """
    def __init__(self, hidden_dim, use_residual=True, alpha=0.3, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual
        self.last_visit_alpha = nn.Parameter(torch.tensor(alpha)) if use_residual else None
        
        self.visit_attn_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, lengths=None, **kwargs):
        """
        Args:
            x: (B, T, H)
            lengths: (B,)
        Returns:
            patient_repr: (B, H)
        """
        if not hasattr(self, "_logged_flow"):
            res_str = f" (alpha={self.last_visit_alpha.item():.2f})" if self.use_residual else ""
            logger.info(f"  [Aggregator] Attention-Pooling Active{res_str}")
            self._logged_flow = True

        B, T, H = x.shape
        device = x.device
        
        # 1) Extract query: the last non-padded state
        if lengths is not None:
            last_idx = (lengths - 1).clamp(min=0)
            query = x[torch.arange(B, device=device), last_idx] # (B, H)
        else:
            query = x[:, -1, :]
            
        # 2) Compute attention weights over the sequence
        keys = self.visit_attn_proj(x)  # (B, T, H)
        # Dot product attention
        attn_scores = (query.unsqueeze(1) * keys).sum(dim=-1) / (H ** 0.5) # (B, T)
        
        # 3) Mask padding
        if lengths is not None:
            visit_mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(~visit_mask, float("-inf"))
            
        attn_weights = F.softmax(attn_scores, dim=-1) # (B, T)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        # 4) Weighted sum (context vector)
        context = (attn_weights.unsqueeze(-1) * x).sum(dim=1) # (B, H)
        
        # 5) Optional residual from the last visit
        if self.use_residual:
            patient_repr = context + self.last_visit_alpha * query
        else:
            patient_repr = context
            
        return patient_repr
