"""
HI-DR-style within-patient historical visit attention (Phase 3 rewrite).

Per-patient attention to their own past visits, not cross-patient k-NN retrieval.
This is what HI-DR actually does, not the cross-patient retrieval MIRROR incorrectly
implemented.

Key insight: For each visit t, compute attention to visits 0...t-1 (and current).
Use gumbel-softmax to select which past visits are relevant. This creates an adaptive
query representation that focuses on clinically similar episodes in the patient's
own history.

Architecture mirrors HI-DR's make_query() → calc_cross_visit_scores():
  1. Aggregate diagnosis/procedure embeddings per visit
  2. Compute attention from current visit to all past visits
  3. Gumbel-softmax to select top-k relevant visits (learnable selection)
  4. Weighted aggregate of selected visits to enhance current representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HistoricalVisitAttention(nn.Module):
    """Per-patient attention to own past visits (HI-DR-style).

    Unlike cross-patient k-NN retrieval (which failed in MIRROR E2),
    this learns which of each patient's OWN history is most relevant to predict
    medications for their CURRENT visit.

    Args:
        hidden_dim: GRU output dimension for patient representation
        dropout: applied to attention weights
        att_tau: temperature for softmax (higher = softer attention, lower = harder)
        gumbel_tau: temperature for gumbel-softmax hard selection
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        att_tau: float = 20.0,
        gumbel_tau: float = 0.6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.att_tau = att_tau
        self.gumbel_tau = gumbel_tau

        # Learned projection for attention computation
        self.attention_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        # P9 FIX: LayerNorm after residual to equalize single-visit vs multi-visit magnitudes.
        # Without this, enhanced = current + aggregated has 2× the norm of single-visit
        # enhanced = current. LayerNorm normalizes both to unit-ish scale regardless of
        # visit count, ensuring fair comparison downstream.
        self.post_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        patient_repr_seq: torch.Tensor,  # (batch, max_visits, hidden_dim) — GRU outputs per visit
        lengths: torch.Tensor,  # (batch,) — actual visit count per patient
    ) -> torch.Tensor:
        """
        Returns:
            enhanced_repr: (batch, hidden_dim) — current visit repr + weighted past visits
        """
        B, T, H = patient_repr_seq.shape
        device = patient_repr_seq.device

        # Extract current (last) visit representation
        if lengths is not None:
            batch_idx = torch.arange(B, device=device)
            last_visit_idx = (lengths - 1).clamp(min=0)
            current_repr = patient_repr_seq[batch_idx, last_visit_idx, :]  # (B, H)
        else:
            last_visit_idx = torch.full((B,), T - 1, device=device, dtype=torch.long)
            current_repr = patient_repr_seq[:, -1, :]  # (B, H)

        # FIX-B13 (BUG-T1): the comment promised a skip path for single-visit
        # patients but the dead variables were never used. We now actually skip
        # the attention computation when ALL patients in the batch have len==1.
        single_visit_all = (lengths is not None) and bool((lengths == 1).all().item())
        if single_visit_all:
            return self.post_norm(current_repr)

        # FIX (BUG-VITA-GUMBEL): Use hard Gumbel-softmax for binary visit selection.
        # VITA reference (VITA_model.py line 119):
        #   pre_gumbel = F.gumbel_softmax(gumbel_input, tau=self.gumbel_tau, hard=True)[:, 0]
        #   gumbel = torch.cat([pre_gumbel[:-1], torch.ones(1)])  # current visit always=1
        # We implement the equivalent for batched processing:
        query = self.attention_proj(current_repr).unsqueeze(1)  # (B, 1, H)
        keys = self.attention_proj(patient_repr_seq)            # (B, T, H)

        # Scaled dot-product attention scores
        scores = torch.bmm(query, keys.transpose(1, 2)) / math.sqrt(H)  # (B, 1, T)
        scores = scores.squeeze(1)  # (B, T)

        # Mask padding positions BEFORE selection (VITA never sees padding)
        if lengths is not None:
            pad_mask = torch.arange(T, device=device).unsqueeze(0) >= lengths.unsqueeze(1)  # (B, T)
            scores = scores.masked_fill(pad_mask, -1e9)

        # Binary visit selection via hard Gumbel-softmax (VITA: hard=True)
        # Stack as 2-class problem: [select, reject] per visit
        selection_input = torch.stack([scores, torch.zeros_like(scores)], dim=-1)  # (B, T, 2)
        if self.training:
            # Hard Gumbel: straight-through estimator — forward=one-hot, backward=soft
            binary_out = F.gumbel_softmax(selection_input, tau=self.gumbel_tau, hard=True)
        else:
            binary_out = F.one_hot(selection_input.argmax(dim=-1), num_classes=2).float()
        binary_mask = binary_out[..., 0]  # (B, T) — 1=selected, 0=excluded

        # FIX (BUG-VITA-SELF-INCLUSION): Force current visit always selected=1.0.
        # VITA line 120: gumbel = torch.cat([pre_gumbel[:-1], torch.ones(1)])
        if lengths is not None:
            binary_mask[batch_idx, last_visit_idx] = 1.0
        else:
            binary_mask[:, -1] = 1.0

        # FIX (double-softmax): Mask NON-SELECTED visits BEFORE softmax (VITA line 277).
        # Old code: applied softmax first, then multiplied by gumbel_weights — wrong order.
        masked_scores = scores.masked_fill(binary_mask == 0, -1e9)
        attn_weights = F.softmax(masked_scores / self.att_tau, dim=-1)  # (B, T)
        attn_weights = self.dropout_layer(attn_weights)

        # Aggregate selected visits: weighted sum
        aggregated = torch.bmm(attn_weights.unsqueeze(1), patient_repr_seq).squeeze(1)  # (B, H)

        # P9 FIX: The 'aggregated' vector already includes the current visit
        # because binary_mask[last_idx] is forced to 1.0 at line 118.
        # Adding current_repr again was redundant and caused a magnitude jump
        # (~2x norm) for multi-visit patients.
        # Removing the residual ensures a consistent magnitude (weighted average)
        # across all patients before the final LayerNorm.
        enhanced = aggregated
        enhanced = self.post_norm(enhanced)

        return enhanced
