"""
Modular Pretraining Tasks for MIRROR.

Constitutional Rule 1: Each task owns its prediction head and loss.
Constitutional Rule 2: All tasks are registry-driven.
Constitutional Rule 3: Talkative logging on first forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from .registry import PRETRAIN_TASKS

logger = logging.getLogger(__name__)

class PretrainTask(nn.Module):
    """Base class for MIRROR pretraining tasks."""
    def __init__(self, **kwargs):
        super().__init__()
        self._logged_flow = False

    def forward(self, model_out, target, **kwargs):
        raise NotImplementedError

@PRETRAIN_TASKS.register("masked_meds")
class MaskedMedicationTask(PretrainTask):
    """
    Self-supervised Masked Medication Prediction.
    Learns to predict held-out medications from the patient's visit history.
    """
    def __init__(self, hidden_dim: int, num_drugs: int, **kwargs):
        super().__init__()
        self.head = nn.Linear(hidden_dim, num_drugs)
        logger.info(f"  [MaskedMedsTask] Initialized | in={hidden_dim} -> out={num_drugs}")

    def forward(self, patient_repr: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        if not self._logged_flow:
            logger.info(f"  [MaskedMedsTask] Forward Flow | patient_repr={patient_repr.shape}")
            self._logged_flow = True
        
        logits = self.head(patient_repr)
        return F.binary_cross_entropy_with_logits(logits, target.float())

@PRETRAIN_TASKS.register("phi_alignment")
class PhiAlignmentTask(PretrainTask):
    """
    Multimodal Patient-Drug Alignment (Phase 1 Baseline).
    Learns to align the fused patient representation with the drug manifold.
    This replicates the original pretraining logic used to build the retrieval index.
    """
    def __init__(self, hidden_dim: int, num_drugs: int, **kwargs):
        super().__init__()
        # In Phi, the "head" is effectively the dot-product with drug_reprs.
        # We don't need a separate weight here, we use the drug_reprs from the model.
        logger.info(f"  [PhiAlignmentTask] Initialized")

    def forward(self, fused_repr: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            fused_repr: (batch, hidden_dim)
            target: (batch, num_drugs)
            drug_reprs: (num_drugs, hidden_dim) - passed via kwargs
        """
        drug_reprs = kwargs.get("drug_reprs")
        if drug_reprs is None:
            raise ValueError("PhiAlignmentTask requires 'drug_reprs' in forward pass.")

        if not self._logged_flow:
            logger.info(f"  [PhiAlignmentTask] Forward Flow | fused={fused_repr.shape}, drugs={drug_reprs.shape}")
            self._logged_flow = True

        logits = fused_repr @ drug_reprs.T
        return F.binary_cross_entropy_with_logits(logits, target.float())
