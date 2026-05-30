import torch
import torch.nn as nn
import logging
from ..registry import TEMPORAL_ENCODERS
from .transformer_encoder import TransformerTemporalEncoder

logger = logging.getLogger(__name__)

@TEMPORAL_ENCODERS.register("imdr_infused")
class IMDRInfusedEncoder(nn.Module):
    """Drug-infused temporal encoder — novel MIRROR contribution.

    "IMDR-infused" = MIRROR's Drug-Infused temporal encoder (internal codename).
    No direct SOTA counterpart. Conceptually related to VITA's medication-aware
    visit encoding (VITA_model.py, self.medication_encoder), but the mechanism
    here is a cross-attention step where visit states (queries) attend to the
    full drug embedding matrix (keys/values), injecting drug knowledge BEFORE
    the Transformer rather than after.

    Architecture:
        1. Cross-attention: visit_states (Q) × drug_embeddings (K, V)
        2. Residual: visit_states = visit_states + attn_output
        3. Standard TransformerTemporalEncoder over enriched states
    """
    def __init__(self, hidden_dim, num_layers=2, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.drug_embed_dim = kwargs.get("drug_embed_dim", hidden_dim)
        
        self.infusion_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=4, 
            batch_first=True,
            # Allow graph encoder output (hidden_dim) or raw LLM (768)
            kdim=self.drug_embed_dim,
            vdim=self.drug_embed_dim
        )
        self.transformer = TransformerTemporalEncoder(hidden_dim, num_layers=num_layers)

    def forward(self, x, lengths, drug_embeddings=None, **kwargs):
        """
        Args:
            x: (B, T, H)
            lengths: (B,)
            drug_embeddings: Optional (num_drugs, H)
        Returns:
            output: (B, T, H)
        """
        if not hasattr(self, "_logged_flow"):
            logger.info(f"  [Temporal] IMDR-Infused Active | backbone='Transformer' ({self.num_layers}L)")
            self._logged_flow = True

        B, T, H = x.shape
        
        if drug_embeddings is not None:
            # (num_drugs, H) -> (B, num_drugs, H)
            drugs = drug_embeddings.unsqueeze(0).expand(B, -1, -1)
            
            # Query = sequence, Key = Value = drug knowledge
            attn_output, _ = self.infusion_attn(query=x, key=drugs, value=drugs)
            
            # Residual connection to fuse global drug knowledge into the patient's visit states
            x = x + attn_output
            
        return self.transformer(x, lengths)
