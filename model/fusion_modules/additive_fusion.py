"""
AdditiveFusion — Gated multimodal fusion with additive residuals.
"""

import torch
import torch.nn as nn
from .base import BaseGatedFusion
from ..registry import FUSION_MODULES

@FUSION_MODULES.register("additive")
class AdditiveFusion(BaseGatedFusion):
    """
    Stage 2: Additive residual mapping.
    fused = patient_repr + Project(gated_note) + Project(gated_lab)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Additive mapping projections
        self.note_add_proj = nn.Linear(self.note_proj_dim, self.hidden_dim, bias=False)
        self.lab_add_proj = nn.Linear(self.lab_proj_dim, self.hidden_dim, bias=False)
        nn.init.xavier_normal_(self.note_add_proj.weight)
        nn.init.xavier_normal_(self.lab_add_proj.weight)

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
        
        note_add = self.note_add_proj(gated_note)
        lab_add = self.lab_add_proj(gated_lab)
        
        return patient_repr + note_add + lab_add

# Alias for backward compatibility with old config keys
FUSION_MODULES._registry["additive_no_film"] = AdditiveFusion
