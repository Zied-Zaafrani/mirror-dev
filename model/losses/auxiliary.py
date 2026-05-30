"""
Auxiliary Loss Functions for MIRROR.

Constitutional Rule 1: Each loss component owns its prediction head.
Constitutional Rule 2: All components are registry-driven.
Constitutional Rule 3: Talkative logging on first forward pass.

LabImputationHead and ATCCoarseHead were previously grafted onto model.py.
They now live here, owned by their respective loss classes. The optimizer
in train.py must include loss_fn.parameters() to train these heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from ..registry import LOSS_FUNCTIONS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction Heads (Component Isolation — Rule 1)
# These are small projection modules that belong to their loss class,
# not to the core MIRROR model.
# ─────────────────────────────────────────────────────────────────────────────

class LabImputationHead(nn.Module):
    """Projects the lab hidden state to z-score predictions.

    Input:  (B, lab_h_dim)  — the encoder's internal hidden representation
    Output: (B, num_labs)   — predicted z-scores for each lab
    """
    def __init__(self, lab_h_dim: int, num_labs: int = 200):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(lab_h_dim, lab_h_dim),
            nn.GELU(),
            nn.Linear(lab_h_dim, num_labs),
        )
        logger.info(
            f"  [LabImputationHead] Initialized | in={lab_h_dim} → out={num_labs}"
        )

    def forward(self, lab_h: torch.Tensor) -> torch.Tensor:
        return self.head(lab_h)


class ATCCoarseHead(nn.Module):
    """Projects the fused patient representation to ATC class logits.

    Input:  (B, hidden_dim)     — fused patient representation (_aux_fused)
    Output: (B, num_atc_classes) — raw logits for BCEWithLogitsLoss
    """
    def __init__(self, hidden_dim: int, num_atc_classes: int):
        super().__init__()
        self.head = nn.Linear(hidden_dim, num_atc_classes)
        logger.info(
            f"  [ATCCoarseHead] Initialized | in={hidden_dim} → out={num_atc_classes}"
        )

    def forward(self, patient_repr: torch.Tensor) -> torch.Tensor:
        return self.head(patient_repr)


# ─────────────────────────────────────────────────────────────────────────────
# Loss Classes (Registry-Driven — Rule 2)
# ─────────────────────────────────────────────────────────────────────────────

@LOSS_FUNCTIONS.register("atc_coarse")
class MIRROR_ATCLoss(nn.Module):
    """ATC Coarse Classification loss.

    Auxiliary task: predict high-level ATC classes from the fused patient
    representation. The ATCCoarseHead lives here and is trained via the
    optimizer receiving loss_fn.parameters().

    Args:
        hidden_dim:      Dimension of the model's fused patient representation.
        num_atc_classes: Number of high-level ATC classes.
        atc_projection:  Optional (num_drugs, num_atc_classes) mapping tensor.
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        num_atc_classes: int = 14,
        atc_projection: torch.Tensor | None = None,
        **kwargs,
    ):
        super().__init__()
        self.atc_head = ATCCoarseHead(hidden_dim, num_atc_classes)
        if atc_projection is not None:
            self.register_buffer("atc_projection", atc_projection)
        else:
            self.atc_projection = None

    def forward(
        self,
        model: nn.Module = None,
        target: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        # Fallback for model/target if passed in kwargs
        if model is None:
            model = kwargs.get("model")
        if target is None:
            target = kwargs.get("target")
        if not hasattr(self, "_logged_flow"):
            logger.info(
                f"  [ATCLoss] Forward | "
                f"has_fused={hasattr(model, '_aux_fused')}, "
                f"has_projection={self.atc_projection is not None}"
            )
            self._logged_flow = True

        if self.atc_projection is None:
            logger.debug("  [ATCLoss] Skipped — no atc_projection tensor provided.")
            return torch.tensor(0.0, device=target.device)

        if not hasattr(model, "_aux_fused") or model._aux_fused is None:
            logger.debug("  [ATCLoss] Skipped — model._aux_fused not populated.")
            return torch.tensor(0.0, device=target.device)

        atc_target = (target.float() @ self.atc_projection) > 0
        atc_logits = self.atc_head(model._aux_fused)

        logger.debug(
            f"  [ATCLoss] atc_logits={atc_logits.shape}, atc_target={atc_target.shape}"
        )
        return F.binary_cross_entropy_with_logits(atc_logits, atc_target.float())


@LOSS_FUNCTIONS.register("lab_impute")
class MIRROR_LabImputeLoss(nn.Module):
    """Auxiliary Lab Imputation loss.

    Predicts normalized lab values (z-scores) from the lab encoder's
    internal hidden state (_lab_h). The LabImputationHead lives here and is
    trained via the optimizer receiving loss_fn.parameters().

    Requirements:
        - lab encoder MUST expose self.lab_h_dim (the 'ID Badge')
        - lab encoder MUST set self._lab_h during forward()

    Active when: lab_encoder_type ∈ {flat, per_lab_attn, isab, traj_lstm,
                                      lab_as_text, clinical_bin}
                 AND lambda_lab > 0.0

    Args:
        lab_h_dim: Dimension of _lab_h from the lab encoder (read from
                   model.predictor.lab_encoder.lab_h_dim in train.py).
        num_labs:  Number of labs to reconstruct (should match dataset).
    """
    def __init__(
        self,
        lab_h_dim: int = 64,
        num_labs: int = 200,
        **kwargs,
    ):
        super().__init__()
        self.num_labs = num_labs
        self.lab_head = LabImputationHead(lab_h_dim, num_labs)

    def forward(
        self,
        model: nn.Module = None,
        lab_vector: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # Fallback for model if passed in kwargs
        if model is None:
            model = kwargs.get("model")
        if not hasattr(self, "_logged_flow"):
            has_lab_h = hasattr(getattr(getattr(model, "predictor", None), "lab_encoder", None), "_lab_h")
            lab_h_is_set = (
                has_lab_h
                and getattr(model.predictor.lab_encoder, "_lab_h", None) is not None
            )
            logger.info(
                f"  [LabImputeLoss] Forward | "
                f"has_lab_vector={lab_vector is not None}, "
                f"has_lab_h={has_lab_h}, "
                f"lab_h_populated={lab_h_is_set}"
            )
            self._logged_flow = True

        # ── Guard: missing inputs ──────────────────────────────────────────
        if lab_vector is None:
            logger.debug("  [LabImputeLoss] Skipped — lab_vector is None.")
            return torch.tensor(0.0, requires_grad=True)

        device = lab_vector.device

        lab_encoder = getattr(getattr(model, "predictor", None), "lab_encoder", None)
        if lab_encoder is None:
            logger.debug("  [LabImputeLoss] Skipped — no lab_encoder on model.predictor.")
            return torch.tensor(0.0, device=device, requires_grad=True)

        if not hasattr(lab_encoder, "_lab_h") or lab_encoder._lab_h is None:
            logger.warning(
                f"  [LabImputeLoss] ⚠️  Lab encoder '{type(lab_encoder).__name__}' "
                f"did not populate _lab_h during forward(). Loss will be 0. "
                f"Ensure the encoder sets self._lab_h in its forward() method."
            )
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Lab vector format: [z-scores (n) | present flags (n) | ...]
        # Flag convention: 0 = present, 1 = missing (MIMIC-III style)
        n = self.num_labs
        lab_values = lab_vector[:, :n]           # (B, n) z-scores
        lab_flags  = lab_vector[:, n : 2 * n]    # (B, n) 0=present, 1=missing
        present_mask = (lab_flags < 0.5)         # True where lab is present

        # ── Predict via head ───────────────────────────────────────────────
        lab_h  = lab_encoder._lab_h              # (B, lab_h_dim)
        pred_z = self.lab_head(lab_h)            # (B, num_labs)

        logger.debug(
            f"  [LabImputeLoss] lab_h={lab_h.shape}, pred_z={pred_z.shape}, "
            f"lab_values={lab_values.shape}, present_count={present_mask.sum().item()}"
        )

        # ── Masked MSE on the intersection of predicted and present labs ───
        m = min(pred_z.size(1), lab_values.size(1))
        mask = present_mask[:, :m]

        if not mask.any():
            logger.debug("  [LabImputeLoss] No present labs in batch — returning 0.")
            return torch.tensor(0.0, device=device, requires_grad=True)

        diff        = pred_z[:, :m] - lab_values[:, :m]  # (B, m)
        masked_diff = diff[mask]                          # (num_present,)
        return F.mse_loss(masked_diff, torch.zeros_like(masked_diff))
