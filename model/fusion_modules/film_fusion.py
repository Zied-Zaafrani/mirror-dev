"""
FiLMFusion — Gated multimodal fusion with Feature-wise Linear Modulation.

Based on FiLM: Feature-wise Linear Modulation (Perez et al., 2018,
arXiv:1709.07871). Novel MIRROR adaptation for EHR multimodal fusion —
FiLM has not been used in any drug recommendation SOTA baseline (HI-DR,
VITA, COGNet, SafeDrug, GAMENet) examined for this project.

MIRROR modification vs original FiLM:
  gamma = 1 + tanh(W_γ · x) * 0.5  (range 0.5–1.5, dampened)
  Original FiLM: gamma = W_γ · x    (unbounded)
  The * 0.5 dampening stabilizes early training by preventing gamma from
  collapsing or exploding the patient representation before other modalities
  have warmed up.
"""

import torch
import torch.nn as nn
from .base import BaseGatedFusion
from ..registry import FUSION_MODULES

@FUSION_MODULES.register("film")
class FiLMFusion(BaseGatedFusion):
    """
    Stage 2: FiLM modulation.
    gamma = 1 + tanh(W_γ · [patient, gated_note, gated_lab]) * 0.5
    beta  = W_β · [patient, gated_note, gated_lab]
    fused = gamma * patient_repr + beta
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # FiLM head
        film_input = self.hidden_dim + self.note_proj_dim + self.lab_proj_dim
        self.film_gamma = nn.Linear(film_input, self.hidden_dim)
        self.film_beta = nn.Linear(film_input, self.hidden_dim)
        
        # Initialize to identity
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.xavier_normal_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

    def forward(
        self,
        patient_repr: torch.Tensor,
        note_embed: torch.Tensor,
        lab_vector: torch.Tensor,
        has_note: torch.Tensor,
        has_lab: torch.Tensor,
    ) -> torch.Tensor:
        self.log_identity(patient_repr.shape)
        
        gated_note, gated_lab = self._get_gated_modalities(
            patient_repr, note_embed, lab_vector, has_note, has_lab
        )
        
        combined = torch.cat([patient_repr, gated_note, gated_lab], dim=1)
        gamma = 1.0 + torch.tanh(self.film_gamma(combined)) * 0.5
        beta = self.film_beta(combined)
        
        return gamma * patient_repr + beta
