"""
FlatLabEncoder — baseline flat MLP lab encoder for MIRROR.

Collapses all 18 labs into a single hidden_dim vector, then scores drugs
via dot product. The semantic gap is not addressed — this is the baseline.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

from ..registry import LAB_ENCODERS


@LAB_ENCODERS.register("flat")
class FlatLabEncoder(nn.Module):
    """Flat MLP lab encoder.

    Collapses all 18 labs into a single hidden_dim vector, then scores drugs
    via dot product. The semantic gap is not addressed — this is the baseline.
    """

    def __init__(
        self,
        lab_input_dim: int = 400,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        lab_values_zeroed: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.lab_values_zeroed = lab_values_zeroed
        self.num_labs = lab_input_dim // 2  # first half = z-scores, second half = flags
        lab_proj_dim = max(16, hidden_dim // 4)
        self.lab_proj = nn.Sequential(
            nn.Linear(lab_input_dim, lab_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lab_score_proj = nn.Linear(lab_proj_dim, hidden_dim, bias=False)
        self.lab_h_dim = lab_proj_dim

    def forward(
        self,
        lab_vector: torch.Tensor,    # (batch, 36)
        drug_reprs: torch.Tensor,    # (num_drugs, hidden_dim)
        has_lab: torch.Tensor,       # (batch,)
        temperature: "torch.Tensor | float" = 1.0,
    ) -> torch.Tensor:
        """Returns (batch, num_drugs) lab drug scores."""
        # Talkative Logging
        if not hasattr(self, "_logged_flow"):
            logger.info(f"  [FlatLabEncoder] Active Flow:")
            logger.info(f"    - Input:  {lab_vector.shape}")
            logger.info(f"    - Weight: {drug_reprs.shape}")
            if self.lab_values_zeroed:
                logger.info(f"    - Mode: presence-only (z-scores zeroed)")
            self._logged_flow = True

        # Presence-only mode: zero out z-score columns, keep binary flags only
        if self.lab_values_zeroed:
            lab_vector = lab_vector.clone()
            lab_vector[:, : self.num_labs] = 0.0

        lab_h = self.lab_proj(lab_vector)
        self._lab_h = lab_h  # (batch, hidden_dim) exposed for auxiliary heads
        if isinstance(temperature, torch.Tensor):
            temp = temperature.clamp(min=0.1)
        else:
            temp = max(temperature, 0.1)
        scores = (self.lab_score_proj(lab_h) @ drug_reprs.T) / temp
        return scores * has_lab.unsqueeze(1)
