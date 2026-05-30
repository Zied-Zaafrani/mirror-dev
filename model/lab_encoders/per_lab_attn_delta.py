"""
PerLabAttentionEncoderWithDelta — lab encoder incorporating temporal delta (Sheetrit 2023).

Uses the delta between current and previous lab Z-scores as an additional feature
to modulate the lab embeddings.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

from ..registry import LAB_ENCODERS
from .common import _split_lab_vec


@LAB_ENCODERS.register("per_lab_attn_delta")
class PerLabAttentionEncoderWithDelta(nn.Module):
    """Lab-specific encoder incorporating temporal delta (Sheetrit 2023).
    
    Instead of multiplying the base embedding by just `effective_val`, we concatenate
    `[effective_val, lab_delta]` and project it through a small MLP to a scalar weight,
    which then modulates the base lab embedding.
    """

    def __init__(
        self,
        num_labs: int = 200,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        precomputed_embeddings: torch.Tensor | None = None,  # (N, 768)
        **kwargs,
    ):
        super().__init__()
        self.num_labs = num_labs
        self.hidden_dim = hidden_dim

        if precomputed_embeddings is not None:
            if precomputed_embeddings.shape[0] != num_labs:
                 if precomputed_embeddings.shape[0] > num_labs:
                     precomputed_embeddings = precomputed_embeddings[:num_labs]
                 else:
                     pad_size = num_labs - precomputed_embeddings.shape[0]
                     pad = torch.zeros(pad_size, precomputed_embeddings.shape[1], device=precomputed_embeddings.device)
                     precomputed_embeddings = torch.cat([precomputed_embeddings, pad], dim=0)

            emb_f = precomputed_embeddings.float()
            _, _, Vh = torch.linalg.svd(emb_f, full_matrices=False)
            k = min(hidden_dim, Vh.shape[0])
            projected = emb_f @ Vh[:k].T
            if k < hidden_dim:
                pad = torch.zeros(num_labs, hidden_dim - k, device=projected.device)
                projected = torch.cat([projected, pad], dim=-1)
            projected = projected / projected.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self.lab_embed = nn.Parameter(projected)
        else:
            self.lab_embed = nn.Parameter(torch.empty(num_labs, hidden_dim))
            nn.init.xavier_uniform_(self.lab_embed)

        # Map [value, delta] -> scalar multiplier
        self.feature_proj = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

        self.lab_token_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lab_h_dim = hidden_dim

    def forward(
        self,
        lab_vector: torch.Tensor,    # (batch, lab_input_dim)
        drug_reprs: torch.Tensor,    # (num_drugs, hidden_dim)
        has_lab: torch.Tensor,       # (batch,)
        temperature: "torch.Tensor | float" = 1.0,
        lab_delta: torch.Tensor | None = None, # (batch, num_labs)
        **kwargs,
    ) -> torch.Tensor:
        # Talkative Logging
        if not hasattr(self, "_logged_flow"):
            logger.info(f"  [PerLabAttentionEncoderWithDelta] Active Flow:")
            logger.info(f"    - Input:    {lab_vector.shape}")
            logger.info(f"    - Delta:    {lab_delta.shape if lab_delta is not None else 'None'}")
            logger.info(f"    - Num Labs: {self.num_labs}")
            self._logged_flow = True

        lab_values, lab_present = _split_lab_vec(lab_vector, num_labs=self.num_labs)

        effective_z = lab_present + lab_values
        
        if lab_delta is None:
            lab_delta = torch.zeros_like(lab_values)
            
        # (B, num_labs, 2)
        features = torch.stack([effective_z, lab_delta], dim=-1)
        
        # (B, num_labs, 1)
        weights = self.feature_proj(features)
        
        # (B, num_labs, H)
        lab_tokens = self.lab_embed.unsqueeze(0) * weights

        lab_tokens = self.lab_token_proj(lab_tokens)
        lab_tokens = lab_tokens * lab_present.unsqueeze(-1)

        present_mask = lab_present.unsqueeze(-1)
        self._lab_h = (lab_tokens * present_mask).sum(dim=1) / present_mask.sum(dim=1).clamp(min=1)

        if isinstance(temperature, torch.Tensor):
            temp = temperature.clamp(min=0.1)
        else:
            temp = max(temperature, 0.1)

        # Sum independent per-lab votes across lab types.
        lab_drug_scores = (lab_tokens @ drug_reprs.T) / temp  # (B, num_labs, D)
        scores = lab_drug_scores.sum(dim=1)                   # (B, D)
        return scores * has_lab.view(-1, 1)
