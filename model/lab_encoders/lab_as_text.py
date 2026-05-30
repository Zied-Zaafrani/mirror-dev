import torch
import torch.nn as nn
import logging
from pathlib import Path
from ..registry import LAB_ENCODERS

logger = logging.getLogger(__name__)


@LAB_ENCODERS.register("lab_as_text")
class LabAsTextEncoder(nn.Module):
    """
    Phase 7 Lab Encoder inspired by EHR-KnowGen.
    
    Uses precomputed PubMedBERT embeddings for lab states ("Glucose is high", etc).
    Instead of passing raw text strings through BERT at training time, we lookup the
    precomputed embeddings based on the clinical bins and mean-pool them.
    
    Bins: 0=missing, 1=low, 2=normal, 3=high.
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.2, num_labs: int = 200, **kwargs):
        super().__init__()
        self.num_labs = num_labs
        self.embed_dim = 768 # PubMedBERT dimension
        
        # Load precomputed text embeddings (num_labs, 4, 768)
        # Bins: 0=missing, 1=low, 2=normal, 3=high
        embed_path = Path("data/processed/lab_text_embeddings.pt")
        if embed_path.exists():
            # (18, 4, 768)
            embeddings = torch.load(embed_path, map_location="cpu")
        else:
            logger.warning(f"  [LabAsText] {embed_path} not found. Using random initialized embeddings for {num_labs} labs.")
            embeddings = torch.randn(num_labs, 4, 768)
            # Bin 0 is missing, should be zero
            embeddings[:, 0, :] = 0.0
            
        # Handle mismatch between core rules (historically 18) and requested labs
        if embeddings.size(0) < num_labs:
            pad_size = num_labs - embeddings.size(0)
            # Pad with zeros for new labs (missing=0, low=0, normal=0, high=0)
            # This is "smart" as it gracefully ignores labs beyond the 18 we have text for
            pad = torch.zeros(pad_size, 4, 768, device=embeddings.device)
            embeddings = torch.cat([embeddings, pad], dim=0)
        elif embeddings.size(0) > num_labs:
            embeddings = embeddings[:num_labs]
            
        self.register_buffer("text_embeddings", embeddings)
        
        # EHR-KnowGen Soft Prompts: learnable prefix tokens (8 tokens)
        # These help the BERT model distinguish between "Note" and "Lab" modality
        self.soft_prompts = nn.Parameter(torch.randn(8, 768))
        
        # Projection to match lab_proj_dim (which is hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU()
        )
        
        self._lab_h = None
        self.lab_h_dim = self.embed_dim

    def forward(self, lab_vector, drug_reprs, has_lab=None, lab_bins=None, temperature=1.0, **kwargs):
        """
        Args:
            lab_vector: (B, 36) — unused.
            drug_reprs: (num_drugs, hidden_dim)
            has_lab: (B,) — binary flag
            lab_bins: (B, 18) — discrete bins [0, 1, 2, 3]
            temperature: scalar or tensor
        """
        # Talkative Logging
        if not hasattr(self, "_logged_flow"):
            logger.info(f"  [LabAsTextEncoder] Active Flow:")
            logger.info(f"    - Input:    {lab_vector.shape}")
            logger.info(f"    - Bins:     {lab_bins.shape if lab_bins is not None else 'None'}")
            logger.info(f"    - Soft Prompts: {self.soft_prompts.shape}")
            self._logged_flow = True
        B = lab_vector.size(0)
        device = lab_vector.device
        
        if lab_bins is None:
            out = torch.zeros(B, self.proj[-2].out_features, device=device)
            self._lab_h = torch.zeros(B, self.embed_dim, device=device)
            return out
            
        # Lookup embeddings
        # lab_bins: (B, 18)
        # text_embeddings: (18, 4, 768)
        
        # We need to gather the embeddings. For each patient b, and lab i, we want text_embeddings[i, lab_bins[b, i]]
        # An easy way is to iterate over labs, or use advanced indexing
        
        embedded_labs = []
        for i in range(self.num_labs):
            bins_i = lab_bins[:, i] # (B,)
            # Lookup: text_embeddings[i] is (4, 768)
            emb_i = self.text_embeddings[i][bins_i] # (B, 768)
            embedded_labs.append(emb_i)
            
        # Stack: (B, 18, 768)
        stacked = torch.stack(embedded_labs, dim=1)
        
        # Mean pool over labs. We should only pool over PRESENT labs (bin > 0).
        # (B, 18)
        present_mask = (lab_bins > 0).float()
        
        # (B, 1)
        present_count = present_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        
        # Mean pool: (B, 768)
        pooled = (stacked * present_mask.unsqueeze(-1)).sum(dim=1) / present_count
        
        self._lab_h = pooled
        
        # Project: (B, hidden_dim)
        out = self.proj(pooled)
        
        # Dot product with drug embeddings to get scores (B, num_drugs)
        if isinstance(temperature, torch.Tensor):
            temp = temperature.clamp(min=0.1)
        else:
            temp = max(float(temperature), 0.1)
            
        scores = (out @ drug_reprs.T) / temp
        
        if has_lab is not None:
            # has_lab is (B,) so unsqueeze to (B, 1) to match scores (B, num_drugs)
            scores = scores * has_lab.unsqueeze(1)
            
        return scores
