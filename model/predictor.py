"""
MIRROR Predictor Orchestrator.

Tier 1: HEIDRScorer — drug self-attn + drug-visit cross-attn + copy  [LOCKED: Sweep 14c]
Tier 2: Note grounding head (H2)
Tier 3: Lab grounding head  (H3, registry-driven)
Tier 4: Copy mechanism      (H4)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
import inspect

from .registry import SCORERS, LOSS_FUNCTIONS
from . import losses  # noqa: F401 — populates LOSS_FUNCTIONS registry

logger = logging.getLogger(__name__)


class MIRRORPredictor(nn.Module):
    """
    Tiered Predictor Orchestrator.

    Tier 1 (Primary): HEIDR scorer (registry-selected).
    Tier 2 (Notes):   Note alignment head.
    Tier 3 (Labs):    Registry-driven lab encoder → drug logits.
    Tier 4 (Copy):    Per-visit copy mechanism.
    """

    def __init__(
        self,
        primary_engine: str = "heidr",
        hidden_dim: int = 256,
        num_drugs: int = 131,
        use_notes: bool = True,
        use_labs: bool = True,
        use_copy: bool = True,
        max_visits: int = 30,
        **kwargs
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_drugs = num_drugs
        self.use_notes = use_notes
        self.use_labs = use_labs
        self.use_copy = use_copy

        # Tier 1: Primary Engine (HEIDR)
        self.engine = SCORERS.build(primary_engine, hidden_dim=hidden_dim, num_drugs=num_drugs, **kwargs)

        # Tier 2: Note grounding head (H2)
        # note_proj always created — needed for drug-text alignment even when use_notes=False.
        _note_dim = kwargs.get("note_input_dim", 768)
        _dropout = kwargs.get("dropout", 0.3)
        self.note_proj = nn.Sequential(
            nn.Linear(_note_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(_dropout),
        )
        if self.use_notes:
            # Two-stage note projection matching old MultiHeadCopyPredictor.
            # Stage 2: Linear(H→H) with eye init provides a learnable alignment rotation.
            self.note_score_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            nn.init.eye_(self.note_score_proj.weight)

        # Note centering buffer (fixes ClinicalBERT anisotropy: cos_sim ≈ 0.95 without centering).
        # Set externally after construction via model.predictor.note_global_mean = train_mean.to(device).
        self.register_buffer("note_global_mean", torch.zeros(_note_dim))

        # Tier 3: Lab grounding head (H3)
        if self.use_labs:
            self.lab_encoder = kwargs.get("lab_encoder", None)
            if self.lab_encoder:
                sig = inspect.signature(self.lab_encoder.forward)
                self._lab_encoder_params = set(sig.parameters.keys())
            else:
                self._lab_encoder_params = set()
        else:
            self.lab_encoder = None
            self._lab_encoder_params = set()

        # Tier 4: Copy mechanism (H4)
        if self.use_copy:
            self.copy_scale = nn.Parameter(torch.tensor(2.0))
            self.copy_gate_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.Tanh(),
                nn.Linear(hidden_dim // 4, 1),
            )
            self.copy_visit_proj = nn.Linear(num_drugs, hidden_dim)
            self.copy_query = nn.Linear(hidden_dim, hidden_dim)
            self.copy_key = nn.Linear(hidden_dim, hidden_dim)
            self.copy_pos_embed = nn.Embedding(max_visits, hidden_dim)

        # Mixing weights (sigmoid-gated)
        # w_engine: sigmoid(10) ≈ 1.0, w_note: sigmoid(0.3) ≈ 0.57, w_lab: sigmoid(0.2) ≈ 0.55
        self.w_engine = nn.Parameter(torch.tensor(10.0))
        self.w_note = nn.Parameter(torch.tensor(0.3))
        self.w_lab = nn.Parameter(torch.tensor(0.2))

        # Freeze inactive modality weights
        if not self.use_notes:
            self.w_note.requires_grad_(False)
        if not self.use_labs:
            self.w_lab.requires_grad_(False)
            if hasattr(self, "lab_encoder") and isinstance(self.lab_encoder, nn.Module):
                for p in self.lab_encoder.parameters():
                    p.requires_grad_(False)

        # Shared temperature for dot-product heads
        self.raw_temperature = nn.Parameter(torch.tensor(-1.5))

    def get_head_weights(self) -> dict[str, float]:
        """Returns active head weights for diagnostic reporting."""
        weights = {"w_engine": torch.sigmoid(self.w_engine).item()}
        if self.use_notes:
            weights["w_note"] = torch.sigmoid(self.w_note).item()
        if self.use_labs:
            weights["w_lab"] = torch.sigmoid(self.w_lab).item()
        return weights

    def forward(
        self,
        fused_patient: torch.Tensor,
        drug_reprs: torch.Tensor,
        drug_history: torch.Tensor,
        gru_out: torch.Tensor | None = None,
        med_per_visit: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        notes_repr: torch.Tensor | None = None,
        labs_repr: torch.Tensor | None = None,
        has_note: torch.Tensor | None = None,
        has_lab: torch.Tensor | None = None,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, "_logged_flow"):
            logger.info(
                f"[MIRRORPredictor] Active | Engine='{type(self.engine).__name__}' | "
                f"Notes={'Active' if self.use_notes else 'Disabled'} | "
                f"Labs={'Active' if self.use_labs else 'Disabled'} | "
                f"Copy={'Active' if self.use_copy else 'Disabled'}"
            )
            self._logged_flow = True

        temp = F.softplus(self.raw_temperature) + 0.2
        B = fused_patient.size(0)
        device = fused_patient.device

        # Tier 1: Primary Engine (HEIDR)
        logits = torch.sigmoid(self.w_engine) * (self.engine(
            fused_patient, drug_reprs, gru_out=gru_out, drug_history=drug_history,
            lengths=lengths, notes_repr=notes_repr, labs_repr=labs_repr, **kwargs
        ) / temp)

        # Tier 2: Note alignment head (H2)
        if self.use_notes and notes_repr is not None:
            note_centered = notes_repr - self.note_global_mean
            note_h = self.note_proj(note_centered)
            h_note = (self.note_score_proj(note_h) @ drug_reprs.T) / temp
            if has_note is not None:
                h_note = h_note * has_note.unsqueeze(1)
            logits = logits + torch.sigmoid(self.w_note) * h_note

        # Tier 3: Lab grounding head (H3)
        if self.use_labs and labs_repr is not None:
            if self.lab_encoder is not None:
                enc_kwargs = {k: v for k, v in kwargs.items() if k in self._lab_encoder_params}
                h_lab = self.lab_encoder(labs_repr, drug_reprs=drug_reprs, has_lab=has_lab, temperature=temp, **enc_kwargs)
                if h_lab.size(-1) == self.hidden_dim:
                    h_lab = (h_lab @ drug_reprs.T) / temp
            else:
                if labs_repr.size(-1) == self.hidden_dim:
                    h_lab = (labs_repr @ drug_reprs.T) / temp
                else:
                    h_lab = labs_repr
                if has_lab is not None:
                    h_lab = h_lab * has_lab.unsqueeze(1)
            logits = logits + torch.sigmoid(self.w_lab) * h_lab

        # Tier 4: Copy mechanism (H4)
        copy_gate = torch.zeros(B, 1, device=device)
        if self.use_copy:
            copy_gate = torch.sigmoid(self.copy_gate_proj(fused_patient))
            if drug_history is not None:
                has_hist = (drug_history.sum(dim=-1, keepdim=True) > 0).float()
                copy_gate = copy_gate * has_hist

            if med_per_visit is not None:
                T = med_per_visit.size(1)
                visit_med_embeds = self.copy_visit_proj(med_per_visit)
                positions = torch.arange(T, device=device).unsqueeze(0).clamp(max=self.copy_pos_embed.num_embeddings - 1)
                visit_med_embeds = visit_med_embeds + self.copy_pos_embed(positions)

                query = self.copy_query(fused_patient).unsqueeze(1)
                keys = self.copy_key(visit_med_embeds)
                attn = (query * keys).sum(dim=-1) / math.sqrt(self.hidden_dim)

                if lengths is not None:
                    mask = torch.arange(T, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
                    attn = attn.masked_fill(mask, float("-inf"))

                weights = F.softmax(attn, dim=-1)
                weights = torch.nan_to_num(weights, nan=0.0)
                copy_dist = (weights.unsqueeze(-1) * med_per_visit).sum(dim=1)
                copy_scores = (copy_dist * self.copy_scale.clamp(min=0.5, max=5.0)) / temp
                logits = logits + copy_gate * copy_scores

        return logits, copy_gate


class MIRRORLoss(nn.Module):
    """Registry-driven Loss Orchestrator for MIRROR.

    Champion loss (L4_jac_heavy): bce=0.3, soft_jaccard=1.5, margin=0.05  [LOCKED: Sweep 15a]
    """
    def __init__(
        self,
        ddi_adj: torch.Tensor | None = None,
        bce_weight: float = 0.3,
        margin_weight: float = 0.05,
        label_smoothing: float = 0.0,
        pos_weight: torch.Tensor | None = None,
        ddi_weight: float = 0.0,
        num_drugs: int = 131,
        use_focal: bool = False,
        focal_gamma_neg: float = 2.0,
        focal_gamma_pos: float = 0.0,
        soft_jaccard_weight: float = 1.5,
        **kwargs
    ):
        super().__init__()
        self.num_drugs = max(1, int(num_drugs))

        self.loss_configs = []

        # Classification (BCE or Focal)
        if use_focal:
            self.loss_configs.append({
                "key": "focal",
                "weight": bce_weight,
                "params": {"gamma_neg": focal_gamma_neg, "gamma_pos": focal_gamma_pos, "pos_weight": pos_weight}
            })
        else:
            self.loss_configs.append({
                "key": "bce",
                "weight": bce_weight,
                "params": {"label_smoothing": label_smoothing, "pos_weight": pos_weight}
            })

        # Soft Margin  [LOCKED: Sweep 15a — champion L4_jac_heavy]
        if margin_weight > 0:
            self.loss_configs.append({
                "key": "soft_margin",
                "weight": margin_weight,
                "params": {}
            })

        # DDI (swept: 0.0, 0.2, 0.5)
        if ddi_weight > 0:
            self.loss_configs.append({
                "key": "ddi",
                "weight": ddi_weight,
                "params": {"ddi_adj": ddi_adj, "num_drugs": num_drugs}
            })

        # Soft Jaccard  [LOCKED: Sweep 15a — champion L4_jac_heavy]
        if soft_jaccard_weight > 0:
            self.loss_configs.append({
                "key": "soft_jaccard",
                "weight": soft_jaccard_weight,
                "params": {}
            })

        # Build loss modules via registry
        self.loss_modules = nn.ModuleDict()
        self.weights = {}

        for cfg in self.loss_configs:
            key = cfg["key"]
            self.loss_modules[key] = LOSS_FUNCTIONS.build(key, **cfg["params"])
            self.weights[key] = cfg["weight"]

    def forward(self, logits: torch.Tensor, target: torch.Tensor, reduction: str = "mean", **kwargs) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the total weighted loss and return breakdown."""
        if not hasattr(self, "_logged_flow"):
            logger.info(f"[LossOrchestrator] Active Flow | Building MIRROR Objective (reduction={reduction}):")
            for key in self.weights:
                logger.info(f"  - '{key}' (weight={self.weights[key]}) ACTIVE")
            self._logged_flow = True

        total_loss = torch.zeros(logits.size(0) if reduction == "none" else 1, device=logits.device)
        loss_dict = {}

        for key, module in self.loss_modules.items():
            l_val = module(logits=logits, target=target, reduction=reduction, **kwargs)
            weighted_l = self.weights[key] * l_val
            total_loss = total_loss + weighted_l

            if reduction == "none":
                loss_dict[key] = l_val.mean().item()
            else:
                loss_dict[key] = l_val.item()

        if reduction == "none":
            loss_dict["total"] = total_loss.mean().item()
        else:
            loss_dict["total"] = total_loss.item()

        return total_loss, loss_dict
