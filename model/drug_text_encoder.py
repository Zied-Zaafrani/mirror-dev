"""
G1 (Run 23) — NLA-MMR-style drug-text encoder + cross-modal alignment loss.

MIRROR's default heads score drugs through structural (h1), note (h2), lab (h3),
and retrieval (h4) views but the drug side is always a single embedding (HGT
output over drug IDs). NLA-MMR shows that surfacing a **drug-text channel** —
where each drug also has an NLP representation of its pharmacological
description — and aligning patient-text to drug-text via a margin loss adds
complementary signal.

This module:
  1. `DrugTextEncoder` — frozen ClinicalBERT embeddings for 130 ATC-3 codes
     are fed through a small projection into the model's hidden_dim space
     so they can be scored against drug_reprs and fused_patient.
  2. `cross_modal_alignment_loss(note_repr, drug_text_repr, target_multihot)`
     — margin-ranking loss that pulls the patient-text representation toward
     the text of prescribed drugs and pushes it away from the text of
     unprescribed drugs. L_align defaults to m=0.3 as the plan specifies.

The encoder itself is parameter-light (a single linear + GELU + dropout).
Drug-text embeddings are registered as a non-trainable buffer so checkpoints
are portable and training cost is dominated by the projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DrugTextEncoder(nn.Module):
    """Projection head over frozen ClinicalBERT drug-text embeddings.

    Args:
        drug_text_embeddings: (num_drugs, 768) frozen ClinicalBERT CLS vectors
            aligned to the med_voc.idx2word order. Mean-centered at init for
            parity with MIRROR's note/diag/proc embedding treatment.
        hidden_dim: target hidden dimension (model.hidden_dim).
        dropout: dropout on the projection output.
    """

    def __init__(
        self,
        drug_text_embeddings: torch.Tensor,  # (num_drugs, 768)
        hidden_dim: int = 256,
        embed_dim: int = 768,
        dropout: float = 0.3,
    ):
        super().__init__()
        num_drugs = int(drug_text_embeddings.size(0))
        assert drug_text_embeddings.dim() == 2, (
            f"drug_text_embeddings must be 2D, got {tuple(drug_text_embeddings.shape)}"
        )
        # Mean-center — same trick as diag/proc/drug/note embeddings.
        centered = drug_text_embeddings - drug_text_embeddings.mean(dim=0, keepdim=True)
        self.register_buffer("drug_text_embed", centered)

        self.proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.num_drugs = num_drugs
        self.hidden_dim = hidden_dim

    def forward(self) -> torch.Tensor:
        """Return projected drug-text reprs, shape (num_drugs, hidden_dim)."""
        return self.proj(self.drug_text_embed)


def cross_modal_alignment_loss(
    patient_text_repr: torch.Tensor,   # (B, H)
    drug_text_repr: torch.Tensor,       # (num_drugs, H)
    target_multihot: torch.Tensor,      # (B, num_drugs) 0/1
    margin: float = 0.3,
    reduction: str = "mean",
) -> torch.Tensor:
    """Margin-ranking alignment loss.

    For each sample b:
        pos_sim_b = mean_{d : target[b,d]=1} cos(patient_text[b], drug_text[d])
        neg_sim_b = mean_{d : target[b,d]=0} cos(patient_text[b], drug_text[d])
        L_b = max(0, margin - pos_sim_b + neg_sim_b)

    Samples with no positives or no negatives contribute zero loss. Both
    patient and drug reprs are L2-normalized before cosine to keep magnitudes
    in [-1, 1] regardless of the projection's scale.
    """
    if patient_text_repr.numel() == 0 or drug_text_repr.numel() == 0:
        return torch.zeros((), device=patient_text_repr.device)

    pt = F.normalize(patient_text_repr, dim=-1)
    dt = F.normalize(drug_text_repr, dim=-1)
    sims = pt @ dt.T  # (B, num_drugs), in [-1, 1]

    mask_pos = target_multihot.to(sims.dtype)
    mask_neg = 1.0 - mask_pos

    n_pos = mask_pos.sum(dim=1).clamp(min=1.0)
    n_neg = mask_neg.sum(dim=1).clamp(min=1.0)

    pos_sim = (sims * mask_pos).sum(dim=1) / n_pos
    neg_sim = (sims * mask_neg).sum(dim=1) / n_neg

    # Zero out samples with no positives (can't anchor) OR no negatives (degenerate).
    valid = ((mask_pos.sum(dim=1) > 0) & (mask_neg.sum(dim=1) > 0)).to(sims.dtype)

    per_sample = F.relu(margin - pos_sim + neg_sim) * valid

    if reduction == "mean":
        denom = valid.sum().clamp(min=1.0)
        return per_sample.sum() / denom
    if reduction == "sum":
        return per_sample.sum()
    return per_sample
