import torch
import torch.nn as nn
import logging

from ..registry import SCORERS

logger = logging.getLogger(__name__)

@SCORERS.register("dot_product")
class DotProductScorer(nn.Module):
    """
    Standard dot-product scoring engine with learnable patient projection.
    gen_proj(fused_patient) @ drug_reprs.T

    Option B Fix: Restores the eye-initialized gen_proj from the old
    MultiHeadCopyPredictor. This gives the model a learnable rotation/scaling
    of the patient representation before scoring against drug embeddings.
    """
    is_pointer_generator = False
    covered_modalities = []  # DotProduct uses patient_repr only

    def __init__(self, hidden_dim: int = 256, **kwargs):
        super().__init__()
        # Head 1: Learnable patient-drug projection (identity init = safe start)
        self.gen_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.eye_(self.gen_proj.weight)

    def forward(self, fused_patient, drug_reprs, gru_out=None, drug_history=None, **kwargs) -> torch.Tensor:
        """
        Args:
            fused_patient: (B, H)
            drug_reprs: (D, H)
            gru_out: (B, T, H)
            drug_history: (B, D)
        Returns:
            scores: (B, D)
        """
        if not hasattr(self, "_logged_flow"):
            logger.info(f"[DotProductScorer] Active | gen_proj=eye_init | fused_patient={fused_patient.shape} | drug_reprs={drug_reprs.shape}")
            self._logged_flow = True

        # Score = gen_proj(fused_patient) @ drug_reprs.T
        # (B, H) -> (B, H) @ (H, D) -> (B, D)
        scores = self.gen_proj(fused_patient) @ drug_reprs.T
        return scores
