"""
PreTrain Model Φ: Simplified MIRROR for extracting patient representations.

Used in Phase 1 to:
1. Train basic encoder on training set
2. Extract high-quality patient embeddings q^t for all training visits
3. These embeddings are then used for offline retrieval index computation

Architecture: VisitEncoder → (optional FiLM fusion with notes+labs) → Predictor
Ablation: No historical attention, simplified HGT

Thesis contribution (April 17, 2026):
    `extract_fused_repr=True` makes the extracted embedding a MULTIMODAL patient
    representation (structural GRU + clinical notes + lab values) instead of a
    purely structural one. HI-DR retrieves by structural similarity only; MIRROR
    retrieves by clinical similarity. This is the differentiating thesis delta
    on the retrieval side and pairs with the E5 ablation.
"""

import torch
import torch.nn as nn

from .visit_encoder import VisitEncoder
from .graph_encoders.drug_gnn import DrugGNN
from .registry import LAB_ENCODERS, GRAPH_ENCODERS, FUSION_MODULES, SCORERS
from .predictor import MIRRORPredictor
from . import fusion_modules     # noqa: F401
from .lab_encoders import FlatLabEncoder, PerLabAttentionEncoder


class PretrainMIRROR(nn.Module):
    """Simplified MIRROR for Phase 1 PreTrain. Identical to MIRROR but without historical attention."""

    def __init__(
        self,
        diag_embeddings: torch.Tensor,
        proc_embeddings: torch.Tensor,
        drug_embeddings: torch.Tensor,
        morgan_fingerprints: torch.Tensor,
        ddi_adj: torch.Tensor,
        hidden_dim: int = 256,
        embed_dim: int = 768,
        note_proj_dim: int | None = None,
        lab_proj_dim: int | None = None,
        lab_input_dim: int = 36,
        encoder_layers: int = 2,
        hgt_layers: int = 2,
        hgt_heads: int = 4,
        num_edge_types: int = 4,  # FIX: DDI=0, EHR=1, self-loop=2, ATC=3
        dropout: float = 0.3,
        use_notes: bool = True,
        use_labs: bool = True,
        use_copy: bool = True,
        use_hist_notes: bool = False,
        finetune_embeddings: bool = False,
        per_visit_copy: bool = True,
        max_visits: int = 30,
        # F1 (Run 23) — default flipped from False → True. Run 22 shipped every cap
        # sweep with extract_fused_repr=False, so the retrieval index lived in the
        # structural 256-d space while the training-time query used the fused repr.
        # Neighbours were effectively random w.r.t. the query manifold. Run 23 keeps
        # the index and query in the same fused space. The False path stays available
        # for the "structural only" diagnostic ablation.
        extract_fused_repr: bool = True,
        use_projection_head: bool = False,
        projection_dropout: float | None = None,
        note_proj_dim_fusion: int | None = None,
        lab_proj_dim_fusion: int | None = None,
        lab_encoder_type: str = "flat",  # "flat" | "per_lab_attn"
        # G2: CAMO cross-attention on current visit's diag tokens (notes K/V).
        use_camo: bool = False,
        camo_heads: int = 4,
        # G3: multi-view drug scoring.
        use_multi_view: bool = False,
        multi_view_weight_init: float = 0.25,
        **kwargs,
    ):
        super().__init__()
        self.use_notes = use_notes
        self.use_labs = use_labs
        self.use_copy = use_copy
        self.use_hist_notes = use_hist_notes
        self.use_camo = use_camo
        self.use_multi_view = use_multi_view
        self.hidden_dim = hidden_dim
        self.num_drugs = drug_embeddings.size(0)
        self.extract_fused_repr = extract_fused_repr
        self.use_projection_head = use_projection_head

        # Mean-center embeddings
        diag_embeddings_c = diag_embeddings - diag_embeddings.mean(dim=0, keepdim=True)
        proc_embeddings_c = proc_embeddings - proc_embeddings.mean(dim=0, keepdim=True)
        drug_embeds_centered = drug_embeddings - drug_embeddings.mean(dim=0, keepdim=True)

        # 1) Visit Encoder
        self.visit_encoder = VisitEncoder(
            diag_embeddings=diag_embeddings_c,
            proc_embeddings=proc_embeddings_c,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            encoder_layers=encoder_layers,
            dropout=dropout,
            finetune_embeddings=finetune_embeddings,
            max_visits=max_visits,
            use_camo=use_camo,
            camo_heads=camo_heads,
        )

        self.visit_encoder.register_buffer("drug_embeds_centered", drug_embeds_centered)

        # 2) Drug GNN — Use registry for modular baseline testing
        graph_encoder_type = kwargs.get("graph_encoder_type", "drug_gnn")
        graph_layer_type = kwargs.get("graph_layer_type", "hgt")
        
        encoder_cls = GRAPH_ENCODERS.get(graph_encoder_type)
        self.drug_gnn = encoder_cls(
            drug_embeddings=drug_embeddings,
            morgan_fingerprints=morgan_fingerprints,
            hidden_dim=hidden_dim,
            hgt_layers=hgt_layers,
            hgt_heads=hgt_heads,
            num_edge_types=num_edge_types,
            dropout=dropout,
            gnn_type=graph_layer_type,
        )

        # 3) Optional multimodal fusion — only when extract_fused_repr=True so
        # the downstream retrieval index uses a clinically-aware embedding.
        if extract_fused_repr and (use_notes or use_labs):
            self.fusion = FUSION_MODULES.build(
                "film", # PreTrain Φ always uses film for multimodal extract
                hidden_dim=hidden_dim,
                note_input_dim=768,
                lab_input_dim=max(lab_input_dim, 1),
                note_proj_dim=note_proj_dim_fusion,
                lab_proj_dim=lab_proj_dim_fusion,
                dropout=dropout,
            )
        else:
            self.fusion = None

        # 3b) Optional projection head for retrieval-space shaping.
        if use_projection_head:
            pdrop = float(dropout if projection_dropout is None else projection_dropout)
            self.projection_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(pdrop),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.projection_head = None

        # 4) Lab Encoder (Phase 5.0 Hardening: use registry)
        if self.use_labs:
            lab_encoder = LAB_ENCODERS.build(
                lab_encoder_type,
                lab_input_dim=lab_input_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                **kwargs
            )
        else:
            lab_encoder = None

        # 5) Predictor (Phase 5: registry-based orchestrator)
        self.predictor = MIRRORPredictor(
            primary_engine="dot_product", # PreTrain Φ always uses dot_product
            hidden_dim=hidden_dim,
            num_drugs=self.num_drugs,
            use_notes=use_notes,
            use_labs=use_labs,
            use_copy=use_copy,
            max_visits=max_visits,
            lab_encoder=lab_encoder,
            use_multi_view=use_multi_view,
            multi_view_weight_init=multi_view_weight_init,
            **kwargs
        )

        # NOTE: No historical attention in PreTrain Φ
        self.use_historical_attention = False

    def forward(
        self,
        diag_seq: list[torch.Tensor],
        proc_seq: list[torch.Tensor],
        diag_mask_seq: list[torch.Tensor] | None,
        proc_mask_seq: list[torch.Tensor] | None,
        lengths: torch.Tensor | None,
        note_embed: torch.Tensor,
        lab_vector: torch.Tensor,
        has_note: torch.Tensor,
        has_lab: torch.Tensor,
        drug_history: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        med_per_visit: torch.Tensor | None = None,
        hist_note_embed: torch.Tensor | None = None,
        has_hist_note: torch.Tensor | None = None,
        similar_reprs: torch.Tensor | None = None,
        similar_multihots: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (batch, num_drugs)
            copy_gate: (batch, 1)
            patient_repr: available via self._aux_patient_repr (batch, hidden_dim).
                When `extract_fused_repr=True`, this is a MULTIMODAL fused repr
                (structural + notes + labs). Otherwise it is the raw GRU output.

        Notes:
            Pretrain Phi does not use cross-patient retrieval inputs. N1 fix: we
            now assert they are None so accidental callers fail loudly.
        """
        # N1 fix: loudly reject retrieval kwargs at pretrain time.
        assert similar_reprs is None and similar_multihots is None, (
            "PretrainMIRROR does not accept cross-patient retrieval inputs. "
            "Build the retrieval index AFTER Phase 1 completes."
        )

        # 1) Visit Encoder — no historical attention. G2 CAMO fires only on
        # the current visit and is silently a no-op when use_camo=False.
        patient_repr = self.visit_encoder(
            diag_seq, proc_seq, diag_mask_seq, proc_mask_seq, lengths,
            med_per_visit=med_per_visit,
            return_sequence=False,
            current_note_embed=note_embed if self.use_camo and self.use_notes else None,
            has_note=has_note if self.use_camo and self.use_notes else None,
        )  # (batch, hidden_dim)

        # 2) Optional multimodal fusion for the extracted embedding.
        # The predictor itself always sees notes+labs through its h2/h3 heads,
        # so this fusion only reshapes the patient_repr used for (a) the h1
        # head and (b) downstream retrieval index extraction.
        if self.fusion is not None:
            eff_has_note = has_note if self.use_notes else torch.zeros_like(has_note)
            eff_has_lab = has_lab if self.use_labs else torch.zeros_like(has_lab)
            fused_repr = self.fusion(patient_repr, note_embed, lab_vector, eff_has_note, eff_has_lab)
        else:
            fused_repr = patient_repr

        if self.projection_head is not None:
            fused_repr = self.projection_head(fused_repr)

        # 3) Drug Graph
        drug_reprs = self.drug_gnn(edge_index, edge_type, edge_weight=edge_weight)

        # 4) Prediction
        eff_note = note_embed
        if self.use_hist_notes and hist_note_embed is not None and has_hist_note is not None:
            hist_mask = has_hist_note.unsqueeze(1)
            eff_note = note_embed + hist_mask * hist_note_embed

        # G3: surface token-level view (cached by VisitEncoder) + seq-level
        # view (here == patient_repr since pretrain has no historical attn).
        seq_repr_g3 = patient_repr if self.use_multi_view else None
        token_repr_g3 = (
            getattr(self.visit_encoder, "_aux_last_visit_token", None)
            if self.use_multi_view else None
        )

        logits, copy_gate = self.predictor(
            fused_repr, drug_reprs, drug_history,
            med_per_visit=med_per_visit, lengths=lengths,
            note_embed=eff_note, lab_vector=lab_vector,
            has_note=has_note, has_lab=has_lab,
            seq_repr=seq_repr_g3, token_repr=token_repr_g3,
        )

        # Store the repr that will be used for retrieval. When extract_fused_repr
        # is True, this is the multimodal fused embedding (thesis contribution).
        self._aux_patient_repr = fused_repr
        return logits, copy_gate

    def count_parameters(self) -> dict[str, int]:
        """Count parameters per component."""
        counts = {}
        for name, module in [
            ("visit_encoder", self.visit_encoder),
            ("drug_gnn", self.drug_gnn),
            ("predictor", self.predictor),
        ]:
            if module is not None:
                total = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                counts[name] = {"total": total, "trainable": trainable}
        counts["model_total"] = {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
        return counts
