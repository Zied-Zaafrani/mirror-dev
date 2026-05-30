import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

from ..registry import LAB_ENCODERS


@LAB_ENCODERS.register("clinical_bin")
class ClinicalBinLabEncoder(nn.Module):
    """
    Phase 7 Lab Encoder inspired by HSGNN.
    
    Instead of using raw continuous values, we map each lab into discrete clinical bins:
    0 (missing), 1 (low), 2 (normal), 3 (high).
    
    Each of the labs gets its own embedding table, mapping its bin to a continuous representation.
    These are then aggregated to form the final lab representation.
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.2, num_labs: int = 200, **kwargs):
        super().__init__()
        self.num_labs = num_labs
        self.embed_dim = hidden_dim // 2 # 32 if hidden_dim=64
        
        # num_labs separate embedding layers (vocab_size=4: missing, low, normal, high)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=4, embedding_dim=self.embed_dim)
            for _ in range(self.num_labs)
        ])
        
        # Projection after concatenation/pooling
        self.proj = nn.Sequential(
            nn.Linear(self.embed_dim * self.num_labs, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU() # Standard MIRROR lab projection
        )
        
        # Auxiliary property for MedGCN ablation
        self._lab_h = None
        self.lab_h_dim = self.embed_dim * self.num_labs

    def forward(self, lab_vector, lab_bins=None, has_lab=None, **kwargs):
        """
        Args:
            lab_vector: (B, lab_input_dim)
            lab_bins: (B, num_labs) — Discrete bin indices [0, 1, 2, 3].
            has_lab: (B, 1) — Binary flag if patient has ANY labs.
            
        Returns:
            lab_embed: (B, proj_dim)
        """
        # Talkative Logging
        if not hasattr(self, "_logged_flow"):
            logger.info(f"  [ClinicalBinLabEncoder] Active Flow:")
            logger.info(f"    - Input:    {lab_vector.shape}")
            logger.info(f"    - Bins:     {lab_bins.shape if lab_bins is not None else 'None'}")
            logger.info(f"    - Num Labs: {self.num_labs}")
            self._logged_flow = True
        """
        Args:
            lab_vector: (B, lab_input_dim) — Not used for feature extraction here, only for shape fallback if needed.
            lab_bins: (B, num_labs) — Discrete bin indices [0, 1, 2, 3].
            has_lab: (B, 1) — Binary flag if patient has ANY labs.
            
        Returns:
            lab_embed: (B, proj_dim)
        """
        B = lab_vector.size(0)
        
        if lab_bins is None:
            # Fallback if lab_bins is not provided (e.g. earlier pipeline stages)
            # Just return zeros
            device = lab_vector.device
            out = torch.zeros(B, self.proj[-2].out_features, device=device)
            self._lab_h = torch.zeros(B, self.embed_dim * self.num_labs, device=device)
            return out
            
        embedded_labs = []
        for i in range(self.num_labs):
            # Extract column i: (B,)
            bins_i = lab_bins[:, i]
            # Embed: (B, embed_dim)
            emb_i = self.embeddings[i](bins_i)
            embedded_labs.append(emb_i)
            
        # Concatenate all lab embeddings: (B, num_labs * embed_dim)
        concat_emb = torch.cat(embedded_labs, dim=1)
        
        # Store for auxiliary loss
        self._lab_h = concat_emb
        
        # Project
        out = self.proj(concat_emb)
        
        # Mask out patients with absolutely no labs
        if has_lab is not None:
            out = out * has_lab.view(-1, 1)
            
        return out
