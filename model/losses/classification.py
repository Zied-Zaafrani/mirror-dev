import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from ..registry import LOSS_FUNCTIONS

logger = logging.getLogger(__name__)

@LOSS_FUNCTIONS.register("bce")
class MIRROR_BCELoss(nn.Module):
    """Standard Binary Cross Entropy loss for MIRROR.
    
    Supports:
        - Label smoothing
        - Pos weight capping/scaling
    """
    def __init__(self, label_smoothing: float = 0.0, pos_weight: torch.Tensor | None = None, **kwargs):
        super().__init__()
        self.label_smoothing = label_smoothing
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def smooth_labels(self, y: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing <= 0:
            return y
        eps = self.label_smoothing
        return y * (1 - eps) + 0.5 * eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor, reduction: str = "mean", **kwargs) -> torch.Tensor:
        if not hasattr(self, "_logged_flow"):
            logger.info(f"  [BCELoss] Active | smoothing={self.label_smoothing} | pos_weight={self.pos_weight is not None}")
            self._logged_flow = True
            
        y_smooth = self.smooth_labels(target)
        if self.pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(logits, y_smooth, pos_weight=self.pos_weight, reduction=reduction)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, y_smooth, reduction=reduction)
            
        if reduction == "none":
            return loss.mean(dim=-1)
        return loss

@LOSS_FUNCTIONS.register("focal")
class MIRROR_FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in medication recommendation.
    
    References:
        - Lin et al. 2017: Focal Loss for Dense Object Detection
        - Run 30 finding: focal_gamma_neg=2.0, focal_gamma_pos=0.0 stabilizes Jaccard.
    """
    def __init__(self, gamma_neg: float = 2.0, gamma_pos: float = 0.0, pos_weight: torch.Tensor | None = None, **kwargs):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor, reduction: str = "mean", **kwargs) -> torch.Tensor:
        if not hasattr(self, "_logged_flow"):
            logger.info(f"  [FocalLoss] Active | gamma_neg={self.gamma_neg} | gamma_pos={self.gamma_pos}")
            self._logged_flow = True
            
        bce_raw = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        gamma = self.gamma_pos * target + self.gamma_neg * (1 - target)
        focal_weight = (1 - p_t) ** gamma
        
        if self.pos_weight is not None:
            class_weight = self.pos_weight * target + (1 - target)
            focal_weight = focal_weight * class_weight
            
        loss = (focal_weight * bce_raw)
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            # reduction='none' usually returns per-element, but MIRROR expects per-patient (B, D) -> (B,)
            return loss.mean(dim=-1)

@LOSS_FUNCTIONS.register("soft_margin")
class MIRROR_SoftMarginLoss(nn.Module):
    """Multi-label soft margin loss."""
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, logits: torch.Tensor, target: torch.Tensor, reduction: str = "mean", **kwargs) -> torch.Tensor:
        if not hasattr(self, "_logged_flow"):
            logger.info(f"  [SoftMarginLoss] Active")
            self._logged_flow = True
        loss = F.multilabel_soft_margin_loss(logits, target, reduction=reduction)
        if reduction == "none":
            return loss.mean(dim=-1)
        return loss
