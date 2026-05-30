"""
Visit Encoder: attention-pooled code embeddings + temporal encoder over visit sequence.

Knowledge-Grounded Drug Recommendation via GNNs and LLMs (MIRROR).

Concept lineage:
  - Per-visit code pooling: adapted from HI-DR (HEIDR_model.py encode(),
    lines ~154–217) and VITA (VITA_model.py encode(), lines ~139–192). Key
    difference: SOTA uses integer code IDs in a learned embedding table;
    MIRROR uses pre-computed PubMedBERT embeddings (768d) via nn.Embedding.from_pretrained.
  - AttentionPool: simplified SelfAttend from HI-DR/VITA layers.py (see class docstring).

Novel MIRROR additions (not in any SOTA repo):
  - Learnable position embeddings over the visit sequence
  - Medication-aware visit encoding: previous prescriptions summarized via drug embeddings
  - Decoupled encoder/aggregator registry (encoder_type / aggregator_type)
  - IMDRInfusedEncoder: drug-knowledge cross-attention before temporal modeling

Per visit:
  - Lookup PubMedBERT embeddings for diagnosis + procedure codes
  - AttentionPool → single visit representation (768d)
  - Dropout + Linear(768 → hidden_dim) + ReLU + Dropout

Across visits:
  - Add learnable position embeddings + medication summary
  - Temporal encoder (imdr_infused) over enriched visit sequence [LOCKED: Sweep 14a]
  - Aggregator (last) selects last valid visit state [LOCKED: Sweep 13a]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

logger = logging.getLogger(__name__)

# Phase 2.3 Hardening: Ensure all swappable components are registered
from .registry import TEMPORAL_ENCODERS, AGGREGATORS
from . import temporal_encoders  # Triggers @register decorators
from . import aggregators       # Triggers @register decorators

logger = logging.getLogger(__name__)


class AttentionPool(nn.Module):
    """Attention pooling over a variable-length set of code embeddings.

    Simplified variant of the `SelfAttend` module used in HI-DR and VITA
    (HI-DR/HEIDR/layers.py, VITA/codes/layers.py, class SelfAttend).
    SelfAttend uses a 2-layer MLP (Linear→Tanh→Linear→1) for attention scores;
    this implementation uses a single Linear layer — functionally equivalent
    but with fewer parameters.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.attn = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, max_codes, embed_dim) or (max_codes, embed_dim)
            mask: (batch, max_codes) bool, True = valid, False = padding

        Returns:
            pooled: (batch, embed_dim) or (embed_dim,)
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
            squeeze = True

        scores = self.attn(x).squeeze(-1)  # (batch, max_codes)
        if mask is not None:
            scores = scores.masked_fill(~mask.to(torch.bool), float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (batch, max_codes)
        # When all positions are masked (empty visit), softmax(-inf...) = NaN.
        # Replace NaN with 0 → zero pooled output, safe for GRU and backprop.
        weights = torch.nan_to_num(weights, nan=0.0)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch, embed_dim)

        if squeeze:
            pooled = pooled.squeeze(0)
        return pooled


class VisitEncoder(nn.Module):
    """Encode patient visit sequences into a fixed-size representation.

    Architecture:
        1. Per visit: lookup diag/proc LLM embeddings → attention pool → visit_repr
        2. (Run 14) Add position embeddings + medication summary to visit representations
        3. GRU over visit sequence → all hidden states
        4. (Run 14) Visit-level attention over GRU outputs + last-visit residual
    """

    def __init__(
        self,
        diag_embeddings: torch.Tensor,  # (num_diag, 768)
        proc_embeddings: torch.Tensor,  # (num_proc, 768)
        embed_dim: int = 768,
        hidden_dim: int = 256,
        encoder_layers: int = 2,
        dropout: float = 0.3,
        finetune_embeddings: bool = False,
        max_visits: int = 30,
        # Temporal encoder  [LOCKED: Sweep 14a — imdr_infused]
        encoder_type: str = "imdr_infused",
        # Aggregator  [LOCKED: Sweep 13a — last]
        aggregator_type: str = "last",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Pre-trained embeddings from PubMedBERT.
        freeze = not finetune_embeddings
        self.diag_embed = nn.Embedding.from_pretrained(diag_embeddings, freeze=freeze)
        self.proc_embed = nn.Embedding.from_pretrained(proc_embeddings, freeze=freeze)

        # --- Medication-aware buffer (set externally by model.py) ---
        self.register_buffer("drug_embeds_centered", None)

        # ── Single-stream encoding ──
        # Attention pooling over codes within a visit
        self.attn_pool = AttentionPool(embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.visit_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pos_embed = nn.Embedding(max_visits, hidden_dim)

        self.med_input_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(),
        )

        # --- Phase 11.3: Decoupled Longitudinal Architecture ---
        self.encoder_type = encoder_type
        self.aggregator_type = aggregator_type

        # 1) Sequence Encoder (Backbone)
        
        # Build backbone
        self.temporal_encoder = TEMPORAL_ENCODERS.build(
            encoder_type,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            dropout=dropout,
            drug_embed_dim=embed_dim,
        )
        
        # 2) Aggregator (Pooling)
        self.aggregator = AGGREGATORS.build(
            aggregator_type,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

    def encode_visit(
        self,
        diag_indices: torch.Tensor,
        proc_indices: torch.Tensor,
        diag_mask: torch.Tensor | None = None,
        proc_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a single visit's codes into a fixed representation.

        Args:
            diag_indices: (batch, max_diag) long tensor
            proc_indices: (batch, max_proc) long tensor
            diag_mask: (batch, max_diag) bool, True=valid
            proc_mask: (batch, max_proc) bool, True=valid

        Returns:
            visit_repr: (batch, hidden_dim)
        """
        # Safety clamp to prevent vectorized_gather_kernel crash
        diag_indices = diag_indices.clamp(min=0, max=self.diag_embed.num_embeddings - 1)
        proc_indices = proc_indices.clamp(min=0, max=self.proc_embed.num_embeddings - 1)

        diag_embeds = self.diag_embed(diag_indices)   # (batch, max_diag, 768)
        proc_embeds = self.proc_embed(proc_indices)    # (batch, max_proc, 768)

        # Concat all code embeddings
        all_embeds = torch.cat([diag_embeds, proc_embeds], dim=1)  # (batch, total_codes, 768)
        if diag_mask is not None and proc_mask is not None:
            all_mask = torch.cat([diag_mask, proc_mask], dim=1)
        else:
            all_mask = None

        # Attention pool + dropout (M5)
        visit_repr = self.attn_pool(all_embeds, all_mask)  # (batch, 768)
        visit_repr = self.attn_dropout(visit_repr)          # M5: regularize attention output
        visit_repr = self.visit_proj(visit_repr)            # (batch, hidden_dim)
        return visit_repr

    def forward(
        self,
        diag_seq: list[torch.Tensor],
        proc_seq: list[torch.Tensor],
        diag_mask_seq: list[torch.Tensor] | None = None,
        proc_mask_seq: list[torch.Tensor] | None = None,
        lengths: torch.Tensor | None = None,
        med_per_visit: torch.Tensor | None = None,
        return_sequence: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode a patient's full visit history.

        Args:
            diag_seq: list of T tensors, each (batch, max_diag_t)
            proc_seq: list of T tensors, each (batch, max_proc_t)
            diag_mask_seq: list of T tensors, each (batch, max_diag_t)
            proc_mask_seq: list of T tensors, each (batch, max_proc_t)
            lengths: (batch,) actual number of visits per patient
            med_per_visit: (batch, T, num_drugs) per-visit medication vectors
            return_sequence: if True, return (patient_repr, gru_out) for historical attention

        Returns:
            patient_repr: (batch, hidden_dim)
            OR if return_sequence=True:
            (patient_repr, gru_out): (batch, hidden_dim), (batch, T, hidden_dim)
        """
        T = len(diag_seq)
        batch_size = diag_seq[0].size(0)
        device = diag_seq[0].device

        # ── Single-stream encoding ──
        visit_reprs = []
        for t in range(T):
            d_mask = diag_mask_seq[t] if diag_mask_seq is not None else None
            p_mask = proc_mask_seq[t] if proc_mask_seq is not None else None
            vr = self.encode_visit(diag_seq[t], proc_seq[t], d_mask, p_mask)
            visit_reprs.append(vr)

        visit_sequence = torch.stack(visit_reprs, dim=1)  # (B, T, hidden_dim)
        # Clamp positions to avoid out-of-bounds indexing if T > max_visits
        positions = torch.arange(T, device=device).unsqueeze(0).clamp(max=self.pos_embed.num_embeddings - 1)
        visit_sequence = visit_sequence + self.pos_embed(positions)

        if med_per_visit is not None and self.drug_embeds_centered is not None:
            # Phase 11.4: Talkative Shape Guard (Resolves mat1/mat2 mismatch)
            if med_per_visit.shape[-1] != self.drug_embeds_centered.shape[0]:
                logger.warning(f"  [VisitEncoder] Emergency Shape Alignment: "
                             f"med_per_visit={med_per_visit.shape[-1]} drugs, "
                             f"but drug_embeds_centered={self.drug_embeds_centered.shape[0]}. "
                             f"Aligning buffer...")
                
                if self.drug_embeds_centered.shape[0] < med_per_visit.shape[-1]:
                    # Pad missing drug embeddings with zeros
                    diff = med_per_visit.shape[-1] - self.drug_embeds_centered.shape[0]
                    padding = torch.zeros(diff, self.drug_embeds_centered.shape[1], device=self.drug_embeds_centered.device)
                    self.drug_embeds_centered = torch.cat([self.drug_embeds_centered, padding], dim=0)
                else:
                    # Slice if buffer is too large
                    self.drug_embeds_centered = self.drug_embeds_centered[:med_per_visit.shape[-1]]

            med_summary = med_per_visit[:, :T, :] @ self.drug_embeds_centered
            med_summary = self.med_input_proj(med_summary)
            visit_sequence = visit_sequence + med_summary

        # ── Longitudinal Encoding ──
        # Phase 11.3: Decoupled Sequence + Aggregation
        # Backbone: (B, T, H) -> (B, T, H)
        seq_out = self.temporal_encoder(
            visit_sequence, 
            lengths, 
            drug_embeddings=getattr(self, "drug_embeds_centered", None),
            **kwargs
        )
        
        # Aggregator: (B, T, H) -> (B, H)
        patient_repr = self.aggregator(seq_out, lengths=lengths)

        patient_repr = self.dropout(patient_repr)
        
        # Logging for architectural transparency (Talkative Logging)
        if not hasattr(self, "_logged_flow"):
            logger.info(f"  [VisitEncoder] Decoupled Flow Active:")
            logger.info(f"    - Backbone:   '{self.encoder_type}' -> {seq_out.shape}")
            logger.info(f"    - Aggregator: '{self.aggregator_type}' -> {patient_repr.shape}")
            self._logged_flow = True

        if return_sequence:
            return patient_repr, seq_out
        return patient_repr
