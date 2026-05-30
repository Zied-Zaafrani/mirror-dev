"""
PerLabAttentionEncoder — lab-specific encoder with one embedding vector per lab test.

Each of 200 labs contributes an independent vote in drug space. Lab values
modulate each lab token as (present + zscore), so a present-normal lab still
contributes signal while missing labs remain silent.
"""

import torch
import torch.nn as nn

from ..registry import LAB_ENCODERS
from .common import _split_lab_vec


import logging

logger = logging.getLogger(__name__)


@LAB_ENCODERS.register("per_lab_attn")
class PerLabAttentionEncoder(nn.Module):
    """Lab-specific encoder with one embedding vector per lab test.

    Each of 200 labs contributes an independent vote in drug space. Lab values
    modulate each lab token as (present + zscore), so a present-normal lab still
    contributes signal while missing labs remain silent.
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
        self.hidden_dim = hidden_dim
        self.num_labs = num_labs

        if precomputed_embeddings is not None:
            if precomputed_embeddings.shape[0] != num_labs:
                 # Fallback: if we have more labs than embeddings, take top labs
                 if precomputed_embeddings.shape[0] > num_labs:
                     precomputed_embeddings = precomputed_embeddings[:num_labs]
                 else:
                     # If we have more labs than embeddings, zero-pad the rest
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

        self.lab_token_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lab_h_dim = hidden_dim

    def forward(
        self,
        lab_vector: torch.Tensor,    # (batch, 36)
        drug_reprs: torch.Tensor,    # (num_drugs, hidden_dim)
        has_lab: torch.Tensor,       # (batch,)
        temperature: "torch.Tensor | float" = 1.0,
    ) -> torch.Tensor:
        # Talkative Logging
        if not hasattr(self, "_logged_flow"):
            logger.info(f"  [PerLabAttentionEncoder] Active Flow:")
            logger.info(f"    - Input:    {lab_vector.shape}")
            logger.info(f"    - Num Labs: {self.num_labs}")
            self._logged_flow = True

        lab_values, lab_present = _split_lab_vec(lab_vector, num_labs=self.num_labs)

        # Missing lab: 0 + 0 = 0 (silent)
        # Present normal: 1 + 0 = 1 (keeps baseline lab identity signal)
        effective_z = lab_present + lab_values
        lab_tokens = self.lab_embed.unsqueeze(0) * effective_z.unsqueeze(-1)  # (B, N_labs, H)

        lab_tokens = self.lab_token_proj(lab_tokens)
        lab_tokens = lab_tokens * lab_present.unsqueeze(-1)

        present_mask = lab_present.unsqueeze(-1)
        self._lab_h = (lab_tokens * present_mask).sum(dim=1) / present_mask.sum(dim=1).clamp(min=1)

        if isinstance(temperature, torch.Tensor):
            temp = temperature.clamp(min=0.1)
        else:
            temp = max(temperature, 0.1)

        # Sum independent per-lab votes across lab types.
        lab_drug_scores = (lab_tokens @ drug_reprs.T) / temp  # (B, 18, D)
        scores = lab_drug_scores.sum(dim=1)                   # (B, D)
        return scores * has_lab.unsqueeze(1)
