import torch
import torch.nn as nn
import logging
from ..registry import LOSS_FUNCTIONS

logger = logging.getLogger(__name__)

@LOSS_FUNCTIONS.register("ddi")
class MIRROR_DDILoss(nn.Module):
    """Differentiable DDI penalty loss.

    Concept from SafeDrug (Zheng et al., 2021, AAAI) and GAMENet
    (Shang et al., 2019). Encourages the model to avoid co-prescribing
    interacting drug pairs.

    Formula (MIRROR): loss = mean_over_patients( sum_j( (p @ DDI)_j * p_j ) / D )
      where p = sigmoid(logits) — soft probabilities, D = num_drugs.
    This is equivalent to COGNet's neg_pred_prob1 * neg_pred_prob2 * ddi_adj
    (COGNet_model.py lines ~244–250) but uses sigmoid probs instead of binary.

    KEY DIFFERENCE from SafeDrug:
      SafeDrug uses a dynamic beta coefficient:
        beta = min(0, 1 + (target_ddi - current_ddi_rate) / kp)
      that turns OFF the DDI loss when prescriptions are already safe enough.
      MIRROR intentionally removes this mechanism — the DDI weight ddi_alpha
      is controlled globally via a sweep (0.0, 0.2, 0.5) instead.
    """
    def __init__(self, ddi_adj: torch.Tensor, num_drugs: int = 131, **kwargs):
        super().__init__()
        self.register_buffer("ddi_adj", ddi_adj)
        self.num_drugs = num_drugs

    def forward(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        if not hasattr(self, "_logged_flow"):
            logger.info(f"  [DDILoss] Active | num_drugs={self.num_drugs}")
            self._logged_flow = True
            
        y_pred = torch.sigmoid(logits)
        if self.ddi_adj is None:
            return torch.tensor(0.0, device=y_pred.device)
            
        ddi_scores = y_pred @ self.ddi_adj
        pair_sum = (ddi_scores * y_pred).sum(dim=1)
        return (pair_sum / self.num_drugs).mean()
