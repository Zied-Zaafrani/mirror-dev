"""
ISABLabEncoder — Set Transformer Inspired Lab Encoder.

Treats present lab values as a set of elements and uses Induced Set Attention Blocks (ISAB)
to encode them, capturing complex inter-lab relationships.
"""

import torch
import torch.nn as nn

from ..registry import LAB_ENCODERS
from .common import _split_lab_vec

import logging
logger = logging.getLogger(__name__)

# Global cache to avoid redundant SVD on scaling to 200+ labs.
# Content-aware key: (shape, dtype, hidden_dim, mean) to prevent id-reused collisions.
_PROJECTION_CACHE = {}
_MAX_CACHE_SIZE = 10  # Prevent unbounded memory growth

class MAB(nn.Module):
    """Multihead Attention Block for Set Transformer."""
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
    def forward(self, X, Y, key_padding_mask=None):
        # X: (B, N_x, D), Y: (B, N_y, D)
        # key_padding_mask: (B, N_y) bool where True means IGNORE
        out, _ = self.mha(X, Y, Y, key_padding_mask=key_padding_mask)
        out = self.ln1(X + out)
        out = self.ln2(out + self.ffn(out))
        return out

class ISAB(nn.Module):
    """Induced Set Attention Block."""
    def __init__(self, dim: int, num_heads: int, num_inds: int):
        super().__init__()
        self.I = nn.Parameter(torch.empty(1, num_inds, dim))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim, num_heads)
        self.mab1 = MAB(dim, num_heads)
        
    def forward(self, X, key_padding_mask=None):
        # X: (B, N_x, D)
        # I: (B, num_inds, D)
        I_batch = self.I.repeat(X.size(0), 1, 1)
        # H: (B, num_inds, D)
        H = self.mab0(I_batch, X, key_padding_mask=key_padding_mask)
        # Out: (B, N_x, D)
        return self.mab1(X, H)


@LAB_ENCODERS.register("isab")
class ISABLabEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        num_inds: int = 4,
        num_heads: int = 4,
        num_labs: int = 200,
        precomputed_embeddings: torch.Tensor | None = None,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_labs = num_labs

        if precomputed_embeddings is not None:
            if precomputed_embeddings.shape[0] != num_labs:
                if precomputed_embeddings.shape[0] > num_labs:
                    precomputed_embeddings = precomputed_embeddings[:num_labs]
                else:
                    pad_size = num_labs - precomputed_embeddings.shape[0]
                    pad = torch.zeros(pad_size, precomputed_embeddings.shape[1], device=precomputed_embeddings.device)
                    precomputed_embeddings = torch.cat([precomputed_embeddings, pad], dim=0)

            # Phase 1.6 Hardening: Use a unique ID based on tensor content hash
            # for the cache key to prevent collisions.
            import hashlib
            data_hash = hashlib.sha256(precomputed_embeddings.cpu().numpy().tobytes()).hexdigest()
            cache_key = (data_hash, hidden_dim)
            
            if cache_key in _PROJECTION_CACHE:
                logger.info(f"  [ISAB] Using cached lab projection (hash={data_hash[:8]})")
                projected = _PROJECTION_CACHE[cache_key]
            else:
                # Cleanup if cache grows too large (Rule: no memory leaks)
                if len(_PROJECTION_CACHE) >= _MAX_CACHE_SIZE:
                    _PROJECTION_CACHE.clear()
                    
                logger.info(f"  [ISAB] Computing SVD projection for {num_labs} labs...")
                emb_f = precomputed_embeddings.float()
                _, _, Vh = torch.linalg.svd(emb_f, full_matrices=False)
                k = min(hidden_dim, Vh.shape[0])
                projected = emb_f @ Vh[:k].T
                if k < hidden_dim:
                    pad = torch.zeros(num_labs, hidden_dim - k, device=projected.device)
                    projected = torch.cat([projected, pad], dim=-1)
                projected = projected / projected.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                _PROJECTION_CACHE[cache_key] = projected
                
            self.lab_embed = nn.Parameter(projected.clone())
        else:
            self.lab_embed = nn.Parameter(torch.empty(num_labs, hidden_dim))
            nn.init.xavier_uniform_(self.lab_embed)

        self.isab = ISAB(hidden_dim, num_heads, num_inds)

        self.lab_token_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self._lab_h = None
        self.lab_h_dim = hidden_dim

    def forward(
        self,
        lab_vector: torch.Tensor,    # (batch, 36)
        drug_reprs: torch.Tensor,    # (num_drugs, hidden_dim)
        has_lab: torch.Tensor,       # (batch,)
        temperature: "torch.Tensor | float" = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        # Talkative Logging
        if not hasattr(self, "_logged_flow"):
            logger.info(f"  [ISABLabEncoder] Active Flow:")
            logger.info(f"    - Input:       {lab_vector.shape}")
            logger.info(f"    - Inducement:  {self.isab.I.shape}")
            logger.info(f"    - Num Labs:    {self.num_labs}")
            self._logged_flow = True

        lab_values, lab_present = _split_lab_vec(lab_vector, num_labs=self.num_labs)

        effective_z = lab_present + lab_values
        lab_tokens = self.lab_embed.unsqueeze(0) * effective_z.unsqueeze(-1)  # (B, 18, H)

        # key_padding_mask for MHA: True = ignore (missing lab)
        key_padding_mask = (lab_present < 0.5) # (B, 18)
        
        # Prevent NaN gradients: If a patient has NO labs, key_padding_mask is all True.
        # MHA softmax over -inf = NaN. Unmask these sequences (they are zeroed out later anyway).
        all_missing = key_padding_mask.all(dim=1, keepdim=True)
        key_padding_mask = key_padding_mask.masked_fill(all_missing, False)

        # Apply ISAB
        lab_tokens = self.isab(lab_tokens, key_padding_mask=key_padding_mask)

        lab_tokens = self.lab_token_proj(lab_tokens)
        lab_tokens = lab_tokens * lab_present.unsqueeze(-1)

        present_mask = lab_present.unsqueeze(-1)
        self._lab_h = (lab_tokens * present_mask).sum(dim=1) / present_mask.sum(dim=1).clamp(min=1)

        if isinstance(temperature, torch.Tensor):
            temp = temperature.clamp(min=0.1)
        else:
            temp = max(temperature, 0.1)

        lab_drug_scores = (lab_tokens @ drug_reprs.T) / temp  # (B, 18, D)
        scores = lab_drug_scores.sum(dim=1)                   # (B, D)
        return scores * has_lab.unsqueeze(1)
