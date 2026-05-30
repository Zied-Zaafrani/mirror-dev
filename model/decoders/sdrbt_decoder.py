"""
SDRBT Substructure-Conditioned Decoder — Phase 5.6 v2 (corrected).

ROOT CAUSE OF PREVIOUS FAILURE (Jac=0.474):
  The "substructure conditioning" was:
    c_i = sigmoid(FF(patient))  — (B, H)
    conditioned_drugs = drug_reprs * c_i  — (B, D, H) elementwise scale
    score = (patient · conditioned_drug).sum(-1)
  
  This is mathematically equivalent to:
    score = patient @ diag(c_i) @ drug_repr.T
  Which is just a re-parameterized dot product. The condition vector c_i
  only re-weights dimensions, not patient-drug interactions.

FIX 1 — Use gru_out for per-drug condition:
  Instead of one global c_i from pooled patient, we compute a per-drug
  condition by attending each drug to the visit sequence:
    ctx_d = cross_attn(drug_d, gru_out)
    c_d = sigmoid(FF(ctx_d))
  This gives each drug its own condition based on which visits are relevant.

FIX 2 — Additive contribution:
  Like other decoders, SDRBT score adds to the baseline, not replaces it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

from ..registry import SCORERS

logger = logging.getLogger(__name__)

@SCORERS.register("sdrbt")
class SDRBTScorer(nn.Module):
    """
    SDRBT Substructure-Conditioned Decoder (Phase 5.6 v2 — corrected).
    """
    is_pointer_generator = False

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

        # Per-drug visit attention (each drug gets its own condition)
        self.cond_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cond_norm = nn.LayerNorm(hidden_dim)
        self.cond_drop = nn.Dropout(dropout)

        # Condition network: ctx → condition vector
        self.condition_ff = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # concat(drug_repr, visit_ctx)
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

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
            logger.info(f"[SDRBTScorer] Active | fused_patient={fused_patient.shape} | drug_reprs={drug_reprs.shape}")
            self._logged_flow = True

        B = fused_patient.size(0)
        D = drug_reprs.size(0)
        device = fused_patient.device

        # Key/Value: full visit sequence or fallback
        if gru_out is not None:
            kv = gru_out
            if lengths is not None:
                T = gru_out.size(1)
                key_padding_mask = torch.arange(T, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
            else:
                key_padding_mask = None
        else:
            kv = fused_patient.unsqueeze(1)
            key_padding_mask = None

        # --- PER-DRUG VISIT ATTENTION ---
        drug_q = drug_reprs.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.cond_attn(
            drug_q, kv, kv,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        visit_ctx = self.cond_norm(drug_q + self.cond_drop(attn_out))

        # --- PER-DRUG CONDITION VECTOR ---
        cond_input = torch.cat([visit_ctx, drug_q], dim=-1)
        c_d = torch.sigmoid(self.condition_ff(cond_input))

        # --- CONDITIONED DRUG SCORING ---
        conditioned = drug_q * c_d
        ca_score = (conditioned * fused_patient.unsqueeze(1)).sum(dim=-1) / math.sqrt(self.hidden_dim)
        return ca_score
