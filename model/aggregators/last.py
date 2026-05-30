import torch
import torch.nn as nn
import logging
from ..registry import AGGREGATORS

logger = logging.getLogger(__name__)

@AGGREGATORS.register("last")
class LastAggregator(nn.Module):
    """Simple aggregator that extracts the last non-padded visit state."""
    def __init__(self, **kwargs):
        super().__init__()
        
    def forward(self, x, lengths=None, **kwargs):
        """
        Args:
            x: (B, T, H) sequence of hidden states
            lengths: (B,) number of valid visits
        Returns:
            final: (B, H)
        """
        if not hasattr(self, "_logged_flow"):
            logger.info("  [Aggregator] Last-State Pooling Active")
            self._logged_flow = True

        B, T, H = x.shape
        device = x.device
        
        if lengths is not None:
            last_idx = (lengths - 1).clamp(min=0)
            final = x[torch.arange(B, device=device), last_idx]
        else:
            final = x[:, -1, :]
            
        return final
