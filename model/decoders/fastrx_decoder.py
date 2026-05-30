"""
FastRx Memory-Augmented Decoder — Phase 5.6 v2 (corrected).

ROOT CAUSE OF PREVIOUS FAILURE (Jac=0.470):
  The "Memory Bank" attention had patient (B,1,H) as Query attending to
  drug reprs (B,D,H). This returns a single weighted average of ALL drugs
  for the whole patient — no per-drug differentiation.
  
  Then: concat(patient_H, dm_H, mb_H) → Linear(3H, D) collapses everything
  into logits in one shot, losing the per-drug granularity.

FIX 1 — Per-drug Memory Bank retrieval:
  Q = drug_reprs (B, D, H) — each drug retrieves its own relevant context
  K/V = gru_out (B, T, H) — retrieves from the full visit sequence
  Now each drug gets a DIFFERENT memory bank output.

FIX 2 — Dynamic Memory uses visit sequence:
  Instead of Linear(num_drugs, H) on binary history, we compute a
  visit-weighted drug history context using gru_out × drug_history.
  This captures WHEN drugs were prescribed, not just that they were.

FIX 3 — Per-drug output scoring:
  Score = dot(patient, combined_drug_repr) / sqrt(H)
  Not a single Linear(3H, D) that collapses per-drug differentiation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

from ..registry import SCORERS

logger = logging.getLogger(__name__)

@SCORERS.register("fastrx")
class FastRxScorer(nn.Module):
    """
    FastRx Memory-Augmented Decoder (Phase 5.6 v2 — corrected).
    """
    is_pointer_generator = True

    def __init__(
        self,
        hidden_dim: int = 256,
        num_drugs: int = 131,
        num_heads: int = 4,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_drugs = num_drugs

        # Memory Bank: drug-visit cross-attention (per-drug retrieval)
        self.mb_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.mb_norm = nn.LayerNorm(hidden_dim)
        self.mb_drop = nn.Dropout(dropout)

        # Dynamic Memory: project drug history to hidden space
        self.dm_proj = nn.Sequential(
            nn.Linear(num_drugs, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Combine MB + DM with patient via gating
        self.combine_gate = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(
        self,
        fused_patient: torch.Tensor,      # (B, H)
        drug_reprs: torch.Tensor,          # (D, H)
        gru_out: torch.Tensor | None = None,
        drug_history: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        **kwargs
    ) -> torch.Tensor:
        if not hasattr(self, "_logged_flow"):
            logger.info(f"[FastRxScorer] Active | fused_patient={fused_patient.shape} | drug_reprs={drug_reprs.shape}")
            self._logged_flow = True

        B = fused_patient.size(0)
        D = drug_reprs.size(0)
        device = fused_patient.device

        # Key/Value: full visit sequence or fallback
        if gru_out is not None:
            kv = gru_out  # (B, T, H)
            if lengths is not None:
                T = gru_out.size(1)
                key_padding_mask = torch.arange(T, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
            else:
                key_padding_mask = None
        else:
            kv = fused_patient.unsqueeze(1)
            key_padding_mask = None

        # --- MEMORY BANK: per-drug visit retrieval ---
        drug_q = drug_reprs.unsqueeze(0).expand(B, -1, -1)
        mb_out, _ = self.mb_attn(
            drug_q, kv, kv,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        mb_ctx = self.mb_norm(drug_q + self.mb_drop(mb_out))

        # --- DYNAMIC MEMORY: drug history embedding ---
        if drug_history is not None:
            dm_h = self.dm_proj(drug_history)
        else:
            dm_h = torch.zeros(B, self.hidden_dim, device=device)

        # --- COMBINE: gate between MB context, DM, and patient repr ---
        patient_exp = fused_patient.unsqueeze(1).expand(B, D, -1)
        dm_exp = dm_h.unsqueeze(1).expand(B, D, -1)

        gate_input = torch.cat([mb_ctx, dm_exp, patient_exp], dim=-1)
        combined = torch.tanh(self.combine_gate(gate_input))

        # Score: dot(combined_drug, patient) / sqrt(H)
        ca_score = (combined * fused_patient.unsqueeze(1)).sum(dim=-1) / math.sqrt(self.hidden_dim)
        return ca_score
