"""
No-Graph Baseline Encoder.

Knowledge-Grounded Drug Recommendation via GNNs and LLMs (MIRROR).

Provides a mathematically pure "EHR-only" baseline by projecting drug features
(LLM + Morgan FP) into the hidden space but skipping any graph message passing.
This ensures the model capacity is identical for the node features, but the 
graph connectivity (DDI/ATC/Co-occurrence) is completely disabled.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

from ..registry import GRAPH_ENCODERS


@GRAPH_ENCODERS.register("none")
class NoGraphEncoder(nn.Module):
    """A baseline encoder that applies projection but skips all message passing.

    Research Note (HI-DR 2024): Disabling the drug graph (-G variant) was found 
    to increase Jaccard (0.6312 vs 0.6281) while slightly worsening DDI safety.
    This baseline allows MIRROR to replicate that "naked feature" performance peak.
    """

    def __init__(
        self,
        drug_embeddings: torch.Tensor,       # (num_drugs, 768)
        morgan_fingerprints: torch.Tensor,   # (num_drugs, 256)
        hidden_dim: int = 256,
        dropout: float = 0.3,
        **kwargs,  # absorb other kwargs (like hgt_layers, use_tripartite) safely
    ):
        super().__init__()
        logger.info("  [NoGraphEncoder] Active: Graph connectivity DISABLED. Using pure feature projection baseline.")
        self.num_drugs = drug_embeddings.size(0)
        drug_input_dim = drug_embeddings.size(1) + morgan_fingerprints.size(1)  # 768+256=1024

        self.register_buffer("drug_llm_centered", drug_embeddings - drug_embeddings.mean(dim=0, keepdim=True))
        self.register_buffer("morgan_fp", morgan_fingerprints)

        # learnable_proj: if True, we use a Linear layer. In HI-DR "0-layer" mode, 
        # this corresponds to a learned embedding lookup.
        self.input_proj = nn.Sequential(
            nn.Linear(drug_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Talkative Logging
        if not hasattr(self, "_logged_flow"):
            logger.info(f"  [NoGraphEncoder] Active Flow:")
            logger.info(f"    - Nodes: {self.num_drugs} drugs")
            logger.info(f"    - Edges: Skipped (Identity Projector)")
            self._logged_flow = True

        """
        Returns:
            drug_reprs: (num_drugs, hidden_dim) — Baseline representations (no graph)
        """
        # Standardized Centering
        x_drug = torch.cat([self.drug_llm_centered, self.morgan_fp], dim=1)  # (num_drugs, 1024)
        x_drug = self.input_proj(x_drug)  # (num_drugs, hidden_dim)
        
        # Skip message passing entirely
        return x_drug
