"""
DrugDoctor CA-MHSA Decoder — Phase 5.6 v2 (corrected).

ROOT CAUSE OF PREVIOUS FAILURE (Jac=0.10, PRAUC=0.1512 FIXED for all 21 epochs):
  The out_proj = Linear(H, 1) mapped every drug to a SCALAR via an IDENTICAL
  input (all drugs got the same attention output from attending to 1 KV pair).
  Result: all 131 drugs got the same logit → random ranking → 0.15 PRAUC.
  
  Additionally, key_padding_mask was applied WRONG — completely blocking all info.

FIX 1 — gru_out as Key/Value:
  Cross-attention Q=drugs (B,D,H) attends to K/V=gru_out (B,T,H).
  Now each drug gets a DIFFERENT context based on which visits it attends to.

FIX 2 — Output projection:
  Instead of Linear(H,1) producing a scalar, we do dot-product between the
  contextualized drug repr and the patient repr: score_d = (ctx_d · patient).
  This is differentiable and unique per drug.

FIX 3 — Additive design:
  CA-MHSA score is added ON TOP of the baseline dot-product. We do NOT replace
  the baseline — we augment it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

from ..registry import SCORERS

logger = logging.getLogger(__name__)

@SCORERS.register("drugdoctor")
class DrugDoctorScorer(nn.Module):
    """
    DrugDoctor CA-MHSA Decoder (Phase 5.6 v2 — corrected).
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

        # Step 1: Drug-visit cross-attention (drugs query visit sequence)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.drop1 = nn.Dropout(dropout)

        # Step 2: Drug-drug self-attention (inter-drug context routing)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop2 = nn.Dropout(dropout)

        # Step 3: Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.drop3 = nn.Dropout(dropout)

    def forward(
        self,
        fused_patient: torch.Tensor,     # (B, H)
        drug_reprs: torch.Tensor,         # (D, H)
        gru_out: torch.Tensor | None = None,
        drug_history: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        **kwargs
    ) -> torch.Tensor:
        if not hasattr(self, "_logged_flow"):
            logger.info(f"[DrugDoctorScorer] Active | fused_patient={fused_patient.shape} | drug_reprs={drug_reprs.shape}")
            self._logged_flow = True

        B = fused_patient.size(0)
        D = drug_reprs.size(0)
        device = fused_patient.device

        # Key/Value: full visit sequence if available, else pooled patient
        if gru_out is not None:
            kv = gru_out  # (B, T, H)
            if lengths is not None:
                T = gru_out.size(1)
                key_padding_mask = torch.arange(T, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
            else:
                key_padding_mask = None
        else:
            kv = fused_patient.unsqueeze(1)  # (B, 1, H) fallback
            key_padding_mask = None

        # --- 1. Drug-visit cross-attention ---
        drug_q = drug_reprs.unsqueeze(0).expand(B, -1, -1)
        ca_out, _ = self.cross_attn(
            drug_q, kv, kv,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = self.norm1(drug_q + self.drop1(ca_out))

        # --- 2. Drug-drug self-attention ---
        sa_out, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm2(x + self.drop2(sa_out))

        # --- 3. Feed-forward ---
        ff_out = self.ff(x)
        ctx_drugs = self.norm3(x + self.drop3(ff_out))

        # --- 4. Score: dot product between each contextualized drug and patient ---
        ca_score = (ctx_drugs * fused_patient.unsqueeze(1)).sum(dim=-1) / math.sqrt(self.hidden_dim)
        return ca_score
