"""
Drug Knowledge Graph: GCN encoder over drug nodes.

Knowledge-Grounded Drug Recommendation via GNNs and LLMs (MIRROR).

Drug node features: LLM embedding (768d) + Morgan fingerprint (256d) = 1024d → Linear → hidden_dim
Edge types:
  - Type 0: DDI edges (TWOSIDES) — "dangerous together"
  - Type 1: Co-occurrence edges (training EHR) — "frequently co-prescribed"
  - Type 2: Self-loop edges — identity / residual signal
  - Type 3: ATC edges — same ATC-3 class (therapeutic similarity)

Champion: 2-layer GCN (graph_layer_type=gcn).  [LOCKED: Sweep 14b + 15b]
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

from ..registry import GRAPH_ENCODERS, GRAPH_LAYERS

@GRAPH_ENCODERS.register("drug_gnn")
class DrugGNN(nn.Module):
    """Drug Knowledge Graph encoder using registry-based GNN layers.

    Takes drug node features (LLM embed + Morgan FP) and graph structure,
    produces DDI-aware drug representations via multi-layer message passing.
    """

    def __init__(
        self,
        drug_embeddings: torch.Tensor,       # (num_drugs, 768)
        morgan_fingerprints: torch.Tensor,   # (num_drugs, 256)
        hidden_dim: int = 256,
        hgt_layers: int = 2,
        hgt_heads: int = 4,
        num_edge_types: int = 4,
        dropout: float = 0.3,
        gnn_type: str = "gcn",               # Champion layer  [LOCKED: Sweep 15b]
        # Absorbed but unused — model.py passes ehr_adj to all graph encoders
        ehr_adj: "torch.Tensor | None" = None,
        **kwargs,
    ):
        super().__init__()
        self.num_drugs = drug_embeddings.size(0)
        drug_input_dim = drug_embeddings.size(1) + morgan_fingerprints.size(1)  # 768+256=1024

        # Register pre-computed features as buffers (not parameters, move with .to())
        self.register_buffer("drug_llm_embed", drug_embeddings)
        self.register_buffer("drug_llm_centered", drug_embeddings - drug_embeddings.mean(dim=0, keepdim=True))
        self.register_buffer("morgan_fp", morgan_fingerprints)

        # Project concatenated drug features to hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(drug_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # GNN layers — selected via GRAPH_LAYERS registry
        layer_cls = GRAPH_LAYERS.get(gnn_type)
        self.gnn_type = gnn_type
        self.num_layers = hgt_layers
        self.num_edge_types = num_edge_types

        self.hgt_layers = nn.ModuleList([
            layer_cls(
                hidden_dim=hidden_dim,
                num_heads=hgt_heads,
                num_edge_types=num_edge_types,
                dropout=dropout,
            )
            for _ in range(hgt_layers)
        ])

        logger.info(f"  [DrugGNN] Initialized | layer='{gnn_type}' × {hgt_layers} | edge_types={num_edge_types}")

    def forward(
        self,
        edge_index: torch.Tensor,                      # (2, num_edges)
        edge_type: torch.Tensor,                       # (num_edges,)
        edge_weight: torch.Tensor | None = None,       # (num_edges,) — directed EHR weights
    ) -> torch.Tensor:
        """
        Returns:
            drug_reprs: (num_drugs, hidden_dim) — DDI-aware drug representations
        """
        if not hasattr(self, "_logged_flow"):
            logger.info(f"  [DrugGNN] Forward | {self.num_drugs} drugs | {edge_index.shape[1]} edges")
            self._logged_flow = True

        # Mean-centered drug features (removes ~93% shared PubMedBERT variance)
        x = torch.cat([self.drug_llm_centered, self.morgan_fp], dim=1)  # (num_drugs, 1024)
        x = self.input_proj(x)                                           # (num_drugs, hidden_dim)

        for layer in self.hgt_layers:
            x = layer(x, edge_index, edge_type, edge_weight=edge_weight)

        return x  # (num_drugs, hidden_dim)
