"""
Base class for gated fusion modules handling modality projections and gates.
Phase 2.3 Hardening: Enforcing Component Isolation (Rule 1).
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class BaseGatedFusion(nn.Module):
    """
    Base class for gated fusion modules handling modality projections and gates.
    
    Stage 1: Content-aware per-dim gates.
    Child classes implement Stage 2 (Fusion math).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        note_input_dim: int = 768,
        lab_input_dim: int = 36,
        note_proj_dim: int | None = None,
        lab_proj_dim: int | None = None,
        dropout: float = 0.3,
        **kwargs
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Scale projection dims proportionally with hidden_dim
        note_proj_dim = note_proj_dim if note_proj_dim is not None else max(32, hidden_dim // 2)
        lab_proj_dim = lab_proj_dim if lab_proj_dim is not None else max(16, hidden_dim // 4)
        self.note_proj_dim = note_proj_dim
        self.lab_proj_dim = lab_proj_dim

        # Project modalities to compact spaces
        self.note_proj = nn.Sequential(
            nn.Linear(note_input_dim, note_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lab_proj = nn.Sequential(
            nn.Linear(lab_input_dim, lab_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Content-aware per-dim gates
        self.note_gate = nn.Linear(hidden_dim + note_proj_dim, note_proj_dim)
        self.lab_gate  = nn.Linear(hidden_dim + lab_proj_dim,  lab_proj_dim)
        nn.init.constant_(self.note_gate.bias, 0.0)
        nn.init.constant_(self.lab_gate.bias,  0.0)

        # Note mean-centering buffer
        self.register_buffer("note_global_mean", torch.zeros(note_input_dim))
        
        # Identity logging flag
        self._logged_flow = False

    def _get_gated_modalities(
        self,
        patient_repr: torch.Tensor,
        note_embed: torch.Tensor,
        lab_vector: torch.Tensor,
        has_note: torch.Tensor,
        has_lab: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs Stage 1 gating logic."""
        
        # Mean-center notes
        note_embed_c = note_embed - self.note_global_mean

        # Project
        if has_note.any():
            note_proj = self.note_proj(note_embed_c)
        else:
            note_proj = torch.zeros(note_embed_c.size(0), self.note_proj_dim, device=note_embed_c.device)
            
        if has_lab.any():
            lab_proj = self.lab_proj(lab_vector)
        else:
            lab_proj = torch.zeros(lab_vector.size(0), self.lab_proj_dim, device=lab_vector.device)

        # Gate computation
        n_gate = torch.sigmoid(self.note_gate(torch.cat([patient_repr, note_proj], dim=1)))
        l_gate = torch.sigmoid(self.lab_gate(torch.cat([patient_repr, lab_proj], dim=1)))

        # Masking
        gated_note = n_gate * note_proj * has_note.unsqueeze(1)
        gated_lab  = l_gate * lab_proj  * has_lab.unsqueeze(1)
        
        return gated_note, gated_lab

    def log_identity(self, patient_repr_shape):
        """Talkative Logging Implementation (Rule 3)."""
        if not self._logged_flow:
            logger.info(f"  [Fusion] Identity: {self.__class__.__name__}")
            logger.info(f"    - Input Shape: {patient_repr_shape}")
            logger.info(f"    - Note Proj:  {self.note_proj_dim} (Input: {self.note_proj[0].in_features})")
            logger.info(f"    - Lab Proj:   {self.lab_proj_dim} (Input: {self.lab_proj[0].in_features})")
            self._logged_flow = True
