import torch
import torch.nn as nn
import logging
from ..registry import LOSS_FUNCTIONS

logger = logging.getLogger(__name__)

@LOSS_FUNCTIONS.register("soft_jaccard")
class MIRROR_SoftJaccardLoss(nn.Module):
    """Differentiable proxy for Jaccard similarity.
    
    Formula: 1 - intersection / union
    Softening via sigmoid probabilities.
    """
    def __init__(self, eps: float = 1e-8, **kwargs):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        if not hasattr(self, "_logged_flow"):
            logger.info(f"  [SoftJaccardLoss] Active")
            self._logged_flow = True
            
        y_pred = torch.sigmoid(logits)
        intersection = (y_pred * target).sum(dim=1)
        union = y_pred.sum(dim=1) + target.sum(dim=1) - intersection
        return (1.0 - intersection / (union + self.eps)).mean()

