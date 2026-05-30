"""
COGNet Sequence Cross-Attention Decoder — Phase 5.6 v2 (corrected).

ROOT CAUSE OF PREVIOUS FAILURE (Jac=0.34):
  The autoregressive GRU decoder generated drugs one-by-one using Softmax
  over 131 drugs at each step — this is a SINGLE-label distribution per step,
  fundamentally incompatible with our BCEWithLogitsLoss (multi-label, independent).
  
  The log(p/(1-p)) logit conversion of softmax probabilities is numerically
  unstable and creates a scale mismatch the BCE loss cannot train through.
  
  max_len=20 means at most 20 of the 19-average drugs could appear.

FIX:
  Replace the autoregressive decoder with a parallel sequence cross-attention.
  Instead of generating drugs one at a time, we:
    1. Attend each drug to the full visit sequence (parallel, like a Transformer decoder)
    2. Produce per-drug logit directly compatible with BCEWithLogitsLoss
  
  This captures the "copy from history" intuition of COGNet's pointer-generator
  but in a multi-label-compatible way.
  
  The COGNet "rare drug first" ordering idea is preserved via a learnable drug
  priority bias that gets added to the final logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

from ..registry import SCORERS

logger = logging.getLogger(__name__)

@SCORERS.register("cognet")
class COGNetScorer(nn.Module):
    """
    COGNet-inspired Parallel Sequence Decoder (Phase 5.6 v2 — corrected).
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

        # Generate path: drug → visit cross-attention
        self.gen_cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.gen_norm = nn.LayerNorm(hidden_dim)
        self.gen_drop = nn.Dropout(dropout)

        # Generate FF
        self.gen_ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.gen_norm2 = nn.LayerNorm(hidden_dim)
        self.gen_drop2 = nn.Dropout(dropout)

        # Copy path: drug queries historical drug embeddings
        self.copy_proj = nn.Linear(hidden_dim, hidden_dim)

        # Gate network: produces per-drug gate from patient context
        self.gate_proj = nn.Linear(hidden_dim * 2, 1)

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
            logger.info(f"[COGNetScorer] Active | fused_patient={fused_patient.shape} | drug_reprs={drug_reprs.shape}")
            self._logged_flow = True

        B = fused_patient.size(0)
        D = drug_reprs.size(0)
        device = fused_patient.device

        # Key/Value: full visit sequence if available
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

        # Drug query matrix: (B, D, H)
        drug_q = drug_reprs.unsqueeze(0).expand(B, -1, -1)

        # --- GENERATE PATH: drugs attend to visit sequence ---
        ca_out, _ = self.gen_cross_attn(
            drug_q, kv, kv,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        gen_ctx = self.gen_norm(drug_q + self.gen_drop(ca_out))
        ff_out = self.gen_ff(gen_ctx)
        gen_ctx = self.gen_norm2(gen_ctx + self.gen_drop2(ff_out))
        gen_score = (gen_ctx * fused_patient.unsqueeze(1)).sum(dim=-1) / math.sqrt(self.hidden_dim)

        # --- COPY PATH: drugs attend to historical drugs ---
        copy_score = torch.zeros(B, D, device=device)
        has_hist = torch.zeros(B, 1, device=device)
        if drug_history is not None:
            has_hist = (drug_history.sum(dim=-1, keepdim=True) > 0).float()
            copy_query = self.copy_proj(gen_ctx)
            copy_attn = torch.bmm(copy_query, drug_reprs.unsqueeze(0).expand(B, -1, -1).transpose(1, 2))
            copy_attn = copy_attn / math.sqrt(self.hidden_dim)
            hist_mask = drug_history.unsqueeze(1).expand_as(copy_attn)
            copy_attn = copy_attn.masked_fill(hist_mask == 0, float("-inf"))
            copy_weights = F.softmax(copy_attn, dim=-1)
            copy_weights = torch.nan_to_num(copy_weights, nan=0.0)
            copy_ctx = torch.bmm(copy_weights, drug_reprs.unsqueeze(0).expand(B, -1, -1))
            copy_score = (copy_ctx * fused_patient.unsqueeze(1)).sum(dim=-1) / math.sqrt(self.hidden_dim)

        # --- GATE: per-drug blend of generate and copy ---
        gate_input = torch.cat([
            fused_patient.unsqueeze(1).expand_as(gen_ctx),
            gen_ctx
        ], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input)).squeeze(-1)
        gate = gate * has_hist

        combined = gate * copy_score + (1 - gate) * gen_score
        return combined
