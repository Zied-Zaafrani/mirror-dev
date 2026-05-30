"""
HEIDR Scorer — MIRROR's non-autoregressive multi-label drug scorer.

Architecturally INSPIRED by HI-DR's MedTransformerDecoder (HI-DR repo:
HEIDR/HEIDR_model.py, class MedTransformerDecoder, lines ~367–446) and
VITA's equivalent (VITA/codes/VITA_model.py, lines ~306–360).

KEY DIFFERENCE FROM THE ORIGINAL:
  HI-DR/VITA decoders are AUTOREGRESSIVE — they generate one drug token at a
  time using a causal mask (sequence-to-sequence). HEIDRScorer is NOT
  autoregressive. It scores all drugs simultaneously as a multi-label
  classifier, matching MIRROR's training objective (binary cross-entropy over
  all D drugs at once).

NOVEL MIRROR CONTRIBUTIONS (not in HI-DR/VITA):
  1. Non-autoregressive transformer block: drug self-attn + drug-visit
     cross-attn without causal masking → all drugs attend freely.
  2. gru_out (B, T, H) as K/V for cross-attention instead of a single
     pooled patient vector — each drug queries DIFFERENT past visits.
  3. Additive copy branch with sigmoid gate scaled by visit history presence.
  4. Compound score: ca_score * (1-gate) + copy_score * gate (novel).

WHY THE KV FIX MATTERS:
  Cross-attention with a single (B, 1, H) patient vector is a mathematical
  no-op — all D drug queries produce identical attention weights (trivially
  1.0). Using gru_out (B, T, H) as K/V gives each drug a distinct context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

from ..registry import SCORERS

logger = logging.getLogger(__name__)

@SCORERS.register("heidr")
class HEIDRScorer(nn.Module):
    """Multi-label drug scorer.

    Inspired by HI-DR/VITA's MedTransformerDecoder but rewritten as a
    non-autoregressive multi-label scorer (see module docstring for details).
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

        # Drug self-attention block
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.drop1 = nn.Dropout(dropout)

        # Drug-visit cross-attention (drugs query the full visit sequence)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop2 = nn.Dropout(dropout)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.drop3 = nn.Dropout(dropout)

        # Copy mechanism: project contextualized drug reprs to query history
        self.copy_proj = nn.Linear(hidden_dim, hidden_dim)
        self.copy_gate = nn.Linear(hidden_dim, 1)  # per-drug gate: generate vs copy

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
            logger.info(f"[HEIDRScorer] Active | fused_patient={fused_patient.shape} | drug_reprs={drug_reprs.shape}")
            self._logged_flow = True

        B = fused_patient.size(0)
        D = drug_reprs.size(0)
        device = fused_patient.device

        # Determine the Key/Value for cross-attention.
        if gru_out is not None:
            kv = gru_out  # (B, T, H)
            if lengths is not None:
                T = gru_out.size(1)
                key_padding_mask = torch.arange(T, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
            else:
                key_padding_mask = None
        else:
            kv = fused_patient.unsqueeze(1)  # (B, 1, H)
            key_padding_mask = None

        # --- 1. Drug self-attention ---
        drug_q = drug_reprs.unsqueeze(0).expand(B, -1, -1)
        sa_out, _ = self.self_attn(drug_q, drug_q, drug_q, need_weights=False)
        x = self.norm1(drug_q + self.drop1(sa_out))

        # --- 2. Drug-visit cross-attention ---
        ca_out, _ = self.cross_attn(
            x, kv, kv,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = self.norm2(x + self.drop2(ca_out))

        # --- 3. Feed-forward ---
        ff_out = self.ff(x)
        context_drugs = self.norm3(x + self.drop3(ff_out))

        # --- 4. Cross-attention generate score ---
        ca_score = (context_drugs * fused_patient.unsqueeze(1)).sum(dim=-1)
        ca_score = ca_score / math.sqrt(self.hidden_dim)

        # --- 5. Additive copy branch ---
        if drug_history is not None:
            has_hist = (drug_history.sum(dim=-1, keepdim=True) > 0).float()
            copy_query = self.copy_proj(context_drugs)
            copy_attn = torch.matmul(copy_query, drug_reprs.unsqueeze(0).transpose(1, 2))
            copy_attn = copy_attn / math.sqrt(self.hidden_dim)
            hist_mask = drug_history.unsqueeze(1).expand_as(copy_attn)
            copy_attn = copy_attn.masked_fill(hist_mask == 0, float("-inf"))
            copy_weights = F.softmax(copy_attn, dim=-1)
            copy_weights = torch.nan_to_num(copy_weights, nan=0.0)
            copy_ctx = torch.bmm(copy_weights, drug_reprs.unsqueeze(0).expand(B, -1, -1))
            copy_score = (copy_ctx * fused_patient.unsqueeze(1)).sum(dim=-1)
            copy_score = copy_score / math.sqrt(self.hidden_dim)
            gate = torch.sigmoid(self.copy_gate(context_drugs)).squeeze(-1)
            gate = gate * has_hist
            ca_score = ca_score * (1 - gate) + copy_score * gate

        return ca_score
