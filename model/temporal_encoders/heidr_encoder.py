import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from ..registry import TEMPORAL_ENCODERS
from .transformer_encoder import TransformerTemporalEncoder

logger = logging.getLogger(__name__)

@TEMPORAL_ENCODERS.register("heidr_gumbel")
class HEIDRGumbelEncoder(nn.Module):
    """HI-DR style Gumbel-Softmax visit selection before Transformer encoding."""
    def __init__(self, hidden_dim, tau=0.6, num_layers=2, **kwargs):
        super().__init__()
        self.tau = tau
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gumbel_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )
        self.transformer = TransformerTemporalEncoder(hidden_dim, num_layers=num_layers)

    def forward(self, x, lengths, **kwargs):
        """
        Args:
            x: (B, T, H)
            lengths: (B,) actual visits
        Returns:
            output: (B, T, H)
        """
        if not hasattr(self, "_logged_flow"):
            logger.info(f"  [Temporal] HEIDR Gumbel Active | tau={self.tau} | backbone='Transformer' ({self.num_layers}L)")
            self._logged_flow = True

        B, T, H = x.shape
        device = x.device
        
        if T > 1:
            logits = self.gumbel_gate(x) # (B, T, 2)
            
            if self.training:
                gumbel_weights = F.gumbel_softmax(logits, tau=self.tau, hard=True)
            else:
                # In eval, we manually do argmax to avoid noise from gumbel_softmax
                gumbel_weights = F.one_hot(logits.argmax(dim=-1), num_classes=2).float()
                
            # Index 1 is "keep"
            gumbel_mask = gumbel_weights[..., 1:2] # (B, T, 1)
            
            # Force current visit to be kept.
            if lengths is not None:
                last_indices = (lengths - 1).clamp(min=0)
                current_mask = F.one_hot(last_indices, num_classes=T).float().unsqueeze(-1) # (B, T, 1)
                gumbel_mask = torch.max(gumbel_mask, current_mask)
            else:
                gumbel_mask[:, -1, 0] = 1.0
                
            x_masked = x * gumbel_mask
        else:
            x_masked = x
            
        return self.transformer(x_masked, lengths)
