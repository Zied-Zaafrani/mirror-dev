"""
MIRROR: Knowledge-Grounded Drug Recommendation via GNNs and LLMs.

A multimodal framework that combines diagnosis/procedure codes, clinical notes,
and lab values with a drug knowledge graph for medication recommendation.

Components:
  1. VisitEncoder  — PubMedBERT embed lookup + IMDR encoder → patient_repr
  2. FiLMFusion    — FiLM: patient_repr + notes + labs → fused_patient
  3. DrugGNN       — GCN over drug graph (DDI + co-occurrence + self-loops + ATC) → drug_reprs
  4. HEIDRScorer   — drug self-attn + cross-attn + per-visit copy → logits
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

from .visit_encoder import VisitEncoder
from .predictor import MIRRORPredictor
from .registry import LAB_ENCODERS, GRAPH_ENCODERS, FUSION_MODULES

# ── Registry Population ──
# Import modular packages to trigger @register decorators at system load time.
from . import lab_encoders        # noqa: F401 — populates LAB_ENCODERS
from . import graph_encoders      # noqa: F401 — populates GRAPH_ENCODERS
from . import temporal_encoders   # noqa: F401 — populates TEMPORAL_ENCODERS
from . import aggregators         # noqa: F401 — populates AGGREGATORS
from . import fusion_modules      # noqa: F401 — populates FUSION_MODULES
from . import decoders            # noqa: F401 — populates SCORERS
from .historical_attention import HistoricalVisitAttention


class MIRROR(nn.Module):
    """Knowledge-Grounded Drug Recommendation via GNNs and LLMs."""

    def __init__(
        self,
        # Pre-computed embeddings (tensors)
        diag_embeddings: torch.Tensor,       # (num_diag, 768)
        proc_embeddings: torch.Tensor,       # (num_proc, 768)
        drug_embeddings: torch.Tensor,       # (num_drugs, 768)
        morgan_fingerprints: torch.Tensor,   # (num_drugs, 256)
        ddi_adj: torch.Tensor,               # (num_drugs, num_drugs)
        # Architecture config
        hidden_dim: int = 256,
        embed_dim: int = 768,
        note_proj_dim: int | None = None,
        lab_proj_dim: int | None = None,
        lab_input_dim: int = 400,
        encoder_layers: int = 2,
        hgt_layers: int = 2,
        hgt_heads: int = 4,
        num_edge_types: int = 4,             # DDI=0, EHR=1, self-loop=2, ATC=3
        ehr_adj: "torch.Tensor | None" = None,
        graph_encoder_type: str = "drug_gnn",
        graph_layer_type: str = "gcn",
        dropout: float = 0.3,
        # Modality toggles
        use_notes: bool = True,
        use_labs: bool = True,
        use_copy: bool = True,
        finetune_embeddings: bool = False,
        per_visit_copy: bool = True,
        max_visits: int = 30,
        note_weight_init: float = 0.3,
        lab_weight_init: float = 0.2,
        fusion_strategy: str = "film",
        # Historical visit attention  [LOCKED: Sweep 11a]
        use_historical_attention: bool = True,
        att_tau: float = 20.0,
        gumbel_tau: float = 0.6,
        # Lab encoder  [LOCKED: Sweep 11b — flat]
        lab_encoder_type: str = "flat",
        num_labs: int = 200,
        # Architecture selections  [LOCKED: Sweeps 13a, 14a, 14c]
        encoder_type: str = "imdr_infused",
        predictor_type: str = "heidr",
        aggregator_type: str = "last",
    ):
        super().__init__()

        logger.info("\n" + "=" * 50)
        logger.info("  MIRROR Architecture Initializing")
        logger.info(f"  Encoder: {encoder_type} | Aggregator: {aggregator_type} | Predictor: {predictor_type}")
        logger.info(f"  Graph: {graph_encoder_type} (Layers: {graph_layer_type})")
        logger.info(f"  Modalities: Notes={use_notes}, Labs={use_labs}, Copy={use_copy}")
        logger.info("=" * 50 + "\n")

        self.use_notes = use_notes
        self.use_labs = use_labs
        self.use_copy = use_copy
        self.hidden_dim = hidden_dim
        self.num_drugs = drug_embeddings.size(0)
        self.fusion_strategy = fusion_strategy
        self.num_labs = num_labs
        self.use_historical_attention = use_historical_attention

        # Run 15: Mean-center ALL LLM embeddings before use.
        # PubMedBERT/ClinicalBERT embed everything on a high-dimensional cone —
        # subtracting the global mean makes individual representations discriminative.
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
            encoder_type=encoder_type,
            aggregator_type=aggregator_type,
        )

        # Register mean-centered drug embeddings for med-aware visit encoding
        self.visit_encoder.register_buffer("drug_embeds_centered", drug_embeds_centered)

        # Drug projection for contrastive alignment loss
        self.drug_proj = nn.Linear(embed_dim, hidden_dim)
        self.register_buffer("drug_llm_embeddings", drug_embeds_centered.clone())

        # 2) Multimodal Fusion — FiLM  [LOCKED: Sweep 12b]
        if use_notes or use_labs:
            self.fusion = FUSION_MODULES.build(
                fusion_strategy,
                hidden_dim=hidden_dim,
                note_input_dim=768,
                lab_input_dim=max(lab_input_dim, 1),
                note_proj_dim=note_proj_dim,
                lab_proj_dim=lab_proj_dim,
                dropout=dropout,
            )
        else:
            self.fusion = None

        # 3) Drug GNN — DrugGNN + GCN layers  [LOCKED: Sweep 14b + 15b]
        encoder_cls = GRAPH_ENCODERS.get(graph_encoder_type)
        self.drug_gnn = encoder_cls(
            drug_embeddings=drug_embeddings,
            morgan_fingerprints=morgan_fingerprints,
            ehr_adj=ehr_adj,
            hidden_dim=hidden_dim,
            hgt_layers=hgt_layers,
            hgt_heads=hgt_heads,
            num_edge_types=num_edge_types,
            dropout=dropout,
            gnn_type=graph_layer_type,
        )

        # 4a) Lab Encoder — flat MLP over 400d lab vector  [LOCKED: Sweep 11b]
        if use_labs:
            lab_encoder_cls = LAB_ENCODERS.get(lab_encoder_type)
            lab_encoder = lab_encoder_cls(
                hidden_dim=hidden_dim,
                dropout=dropout,
                num_labs=num_labs,
                lab_input_dim=lab_input_dim,
            )
        else:
            lab_encoder = None

        # 4b) MIRROR Predictor — HEIDR  [LOCKED: Sweep 14c]
        self.predictor = MIRRORPredictor(
            primary_engine=predictor_type,
            hidden_dim=hidden_dim,
            num_drugs=self.num_drugs,
            note_input_dim=768,
            lab_input_dim=lab_input_dim,
            dropout=dropout,
            per_visit_copy=per_visit_copy,
            max_visits=max_visits,
            note_weight_init=note_weight_init,
            lab_weight_init=lab_weight_init,
            lab_encoder=lab_encoder,
            use_notes=use_notes,
            use_labs=use_labs,
            use_copy=use_copy,
        )

        # 5) Historical visit attention  [LOCKED: Sweep 11a]
        if use_historical_attention:
            self.historical_attention = HistoricalVisitAttention(
                hidden_dim=hidden_dim,
                dropout=dropout,
                att_tau=att_tau,
                gumbel_tau=gumbel_tau,
            )
        else:
            self.historical_attention = None

    def forward(
        self,
        diag_seq: list[torch.Tensor],
        proc_seq: list[torch.Tensor],
        diag_mask_seq: list[torch.Tensor] | None,
        proc_mask_seq: list[torch.Tensor] | None,
        lengths: torch.Tensor | None,
        note_embed: torch.Tensor,     # (batch, 768) — zeros if no notes
        lab_vector: torch.Tensor,     # (batch, lab_dim) — zeros if no labs
        has_note: torch.Tensor,       # (batch,) binary
        has_lab: torch.Tensor,        # (batch,) binary
        drug_history: torch.Tensor,   # (batch, num_drugs) decayed or binary
        edge_index: torch.Tensor,     # (2, num_edges)
        edge_type: torch.Tensor,      # (num_edges,)
        edge_weight: torch.Tensor | None = None,
        med_per_visit: torch.Tensor | None = None,   # (batch, T, num_drugs)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits:    (batch, num_drugs) raw prediction scores
            copy_gate: (batch, 1) copy gate values
        """
        # 1) Encode visit history → patient representation
        patient_repr, gru_out = self.visit_encoder(
            diag_seq, proc_seq, diag_mask_seq, proc_mask_seq, lengths,
            med_per_visit=med_per_visit,
            return_sequence=True,
        )  # patient_repr: (batch, hidden_dim), gru_out: (batch, T, hidden_dim)

        # Historical visit attention — within-patient attention to past visits  [LOCKED]
        if self.use_historical_attention:
            patient_repr = self.historical_attention(gru_out, lengths)

        # Hard-gate modality availability for ablation support
        eff_has_note = has_note if self.use_notes else torch.zeros_like(has_note)
        eff_has_lab = has_lab if self.use_labs else torch.zeros_like(has_lab)

        # 2) Multimodal fusion — FiLM
        if self.fusion is not None:
            fused = self.fusion(patient_repr, note_embed, lab_vector, eff_has_note, eff_has_lab)
        else:
            fused = patient_repr

        self._aux_patient_repr = fused
        self._aux_fused = fused

        # 3) Drug graph encoding
        drug_reprs = self.drug_gnn(edge_index, edge_type, edge_weight=edge_weight)
        self._aux_drug_reprs = drug_reprs

        # Hard-gate modality tensors before predictor
        eff_note_for_pred = note_embed if self.use_notes else torch.zeros_like(note_embed)
        eff_lab_for_pred = lab_vector if self.use_labs else torch.zeros_like(lab_vector)

        # 4) Prediction — HEIDR scorer
        logits, copy_gate = self.predictor(
            fused, drug_reprs, drug_history,
            gru_out=gru_out,
            med_per_visit=med_per_visit, lengths=lengths,
            notes_repr=eff_note_for_pred, labs_repr=eff_lab_for_pred,
            has_note=eff_has_note, has_lab=eff_has_lab,
        )

        return logits, copy_gate

    def count_parameters(self) -> dict[str, int]:
        """Count parameters per component."""
        counts = {}
        for name, module in [
            ("visit_encoder", self.visit_encoder),
            ("fusion", self.fusion),
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

    def enable_diagnostics(self, enabled: bool = True):
        """Toggle diagnostic logging."""
        if hasattr(self, "predictor") and self.predictor is not None:
            self.predictor._enable_diagnostics = enabled
        print(f"  [!] MIRROR diagnostics {'ENABLED' if enabled else 'DISABLED'}")
