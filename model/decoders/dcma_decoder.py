import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

from ..registry import SCORERS

logger = logging.getLogger(__name__)

class DrugModalityAttention(nn.Module):
    """Drug-Conditioned Modality Attention (DCMA) core logic."""
    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = hidden_dim ** -0.5
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        drug_reprs: torch.Tensor,
        modality_tokens: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, M, H = modality_tokens.shape
        D = drug_reprs.shape[0]

        Q = self.query_proj(drug_reprs)
        K = self.key_proj(modality_tokens)

        attn = torch.einsum("dh,bmh->bdm", Q, K) * self.scale
        mask = modality_mask.unsqueeze(1).bool()
        attn = attn.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        V = modality_tokens
        drug_ctx = torch.einsum("bdm,bmh->bdh", attn_weights, V)
        scores = (drug_ctx * drug_reprs.unsqueeze(0)).sum(dim=-1)
        return scores, attn_weights

@SCORERS.register("dcma")
class DCMAScorer(nn.Module):
    """
    Universal Scorer wrapper for DCMA.
    Replaces scalar head weights with per-drug attention over modality tokens.
    """
    is_pointer_generator = False
    covered_modalities = ["notes", "labs"]

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.3,
                 note_repr_dim: int = 768, lab_repr_dim: int = 400, **kwargs):
        super().__init__()
        self.attn = DrugModalityAttention(hidden_dim, dropout)
        self.hidden_dim = hidden_dim
        # notes_repr and labs_repr arrive in their own dims (PubMedBERT=768, flat-lab=400).
        # Project both to hidden_dim before cross-attention so cat(dim=1) works.
        # Regular Linear (not LazyLinear) so parameter counting at init works fine.
        self.note_proj = nn.Linear(note_repr_dim, hidden_dim)
        self.lab_proj  = nn.Linear(lab_repr_dim,  hidden_dim)

    def forward(
        self,
        fused_patient: torch.Tensor,
        drug_reprs: torch.Tensor,
        gru_out: torch.Tensor | None = None,
        drug_history: torch.Tensor | None = None,
        notes_repr: torch.Tensor | None = None,
        labs_repr: torch.Tensor | None = None,
        **kwargs
    ) -> torch.Tensor:
        if not hasattr(self, "_logged_flow"):
            logger.info(f"[DCMAScorer] Active | fused_patient={fused_patient.shape} | drug_reprs={drug_reprs.shape}")
            self._logged_flow = True

        B = fused_patient.size(0)
        device = fused_patient.device
        H = fused_patient.size(1)

        has_note = kwargs.get("has_note", None)
        has_lab  = kwargs.get("has_lab",  None)

        # Build modality tokens (note, lab)
        tokens = []
        mask_bits = []

        # Note token — use per-sample has_note so patients without a note
        # are correctly masked out in the cross-attention (FIX-DCMA-001).
        if notes_repr is not None:
            tokens.append(self.note_proj(F.normalize(notes_repr, dim=-1)).unsqueeze(1))
            note_mask = has_note.unsqueeze(1).float() if has_note is not None else torch.ones(B, 1, device=device)
            mask_bits.append(note_mask)
        else:
            tokens.append(torch.zeros(B, 1, H, device=device))
            mask_bits.append(torch.zeros(B, 1, device=device))

        # Lab token — use per-sample has_lab so patients without labs
        # are correctly masked out (FIX-DCMA-001).
        if labs_repr is not None:
            tokens.append(self.lab_proj(F.normalize(labs_repr, dim=-1)).unsqueeze(1))
            lab_mask = has_lab.unsqueeze(1).float() if has_lab is not None else torch.ones(B, 1, device=device)
            mask_bits.append(lab_mask)
        else:
            tokens.append(torch.zeros(B, 1, H, device=device))
            mask_bits.append(torch.zeros(B, 1, device=device))

        mod_tokens = torch.cat(tokens, dim=1)  # (B, 2, H)
        mod_mask = torch.cat(mask_bits, dim=1)   # (B, 2)

        scores, _ = self.attn(drug_reprs, mod_tokens, mod_mask)
        
        # Zero out if no modalities are present
        any_modality = mod_mask.any(dim=1, keepdim=True).float()
        return scores * any_modality
