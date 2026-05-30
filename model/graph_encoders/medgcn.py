"""
MedGCN Tripartite Encoder (Phase 5.3)

Extends DrugGNN to simultaneously handle 3 modalities:
- Drug Nodes (0..num_drugs-1)
- Diagnosis Nodes (num_drugs..num_drugs+num_diag-1)
- Lab Nodes (num_drugs+num_diag..num_drugs+num_diag+num_labs-1)

Features are generated from LLM embeddings (PubMedBERT/ClinicalBERT) and 
projected into a shared hidden dimension space.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

from ..registry import GRAPH_ENCODERS, GRAPH_LAYERS
@GRAPH_ENCODERS.register("medgcn")
class MedGCN(nn.Module):
    def __init__(
        self,
        drug_embeddings: torch.Tensor,       # (num_drugs, 768)
        morgan_fingerprints: torch.Tensor,   # (num_drugs, 256)
        hidden_dim: int = 256,
        hgt_layers: int = 2,
        hgt_heads: int = 4,
        num_edge_types: int = 7,  # DDI=0, EHR=1, self=2, ATC=3, Sem=4, DrugDiag=5, DrugLab=6
        dropout: float = 0.3,
        gnn_type: str = "hgt",    # Under the hood we use hgt_layer, gat_layer, etc.
        use_tripartite: bool = True,
        diag_embeddings: torch.Tensor | None = None,  # (num_diag, 768)
        lab_embeddings: torch.Tensor | None = None,   # (num_labs, 768)
        use_lab_nodes: bool = False,                  # Required for DrugGNN parity
        **kwargs,                                     # Catch-all for extra params
    ):
        super().__init__()
        self.num_drugs = drug_embeddings.size(0)
        self.num_diag = diag_embeddings.size(0) if diag_embeddings is not None else 1958
        self.num_labs = lab_embeddings.size(0) if lab_embeddings is not None else 18
        
        self.use_tripartite = use_tripartite
        self.use_lab_nodes = lab_embeddings is not None
        self.num_edge_types = num_edge_types

        drug_input_dim = drug_embeddings.size(1) + morgan_fingerprints.size(1)

        self.register_buffer("drug_llm_embed", drug_embeddings)
        self.register_buffer("drug_llm_centered", drug_embeddings - drug_embeddings.mean(dim=0, keepdim=True))
        self.register_buffer("morgan_fp", morgan_fingerprints)

        self.drug_proj = nn.Sequential(
            nn.Linear(drug_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if use_tripartite and diag_embeddings is not None:
            self.register_buffer("diag_embed", diag_embeddings)
            self.register_buffer("diag_centered", diag_embeddings - diag_embeddings.mean(dim=0, keepdim=True))
            self.diag_proj = nn.Sequential(
                nn.Linear(diag_embeddings.size(1), hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.diag_proj = None

        if self.use_lab_nodes and lab_embeddings is not None:
            self.register_buffer("lab_embed", lab_embeddings)
            self.register_buffer("lab_centered", lab_embeddings - lab_embeddings.mean(dim=0, keepdim=True))
            self.lab_proj = nn.Sequential(
                nn.Linear(lab_embeddings.size(1), hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.lab_proj = None

        layer_cls = GRAPH_LAYERS.get(gnn_type)
        self.layers = nn.ModuleList([
            layer_cls(
                hidden_dim=hidden_dim,
                num_heads=hgt_heads,
                num_edge_types=num_edge_types,
                dropout=dropout
            )
            for _ in range(hgt_layers)
        ])
        
        logger.info(f"  [MedGCN] Tripartite System Active: Drugs={self.num_drugs}, Diags={self.num_diag}, Labs={self.num_labs}")

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Talkative Logging
        if not hasattr(self, "_logged_flow"):
            logger.info(f"  [MedGCN] Active Flow:")
            logger.info(f"    - Nodes: Drug={self.num_drugs}, Diag={self.num_diag}, Lab={self.num_labs}")
            logger.info(f"    - Edges: {edge_index.shape[1]} (Types: {self.num_edge_types})")
            logger.info(f"    - Type Stats: {torch.bincount(edge_type).tolist()}")
            self._logged_flow = True
        
        # 1. Process Drugs
        x_drug = torch.cat([self.drug_llm_centered, self.morgan_fp], dim=1)
        x_drug = self.drug_proj(x_drug)  # (num_drugs, hidden_dim)

        node_features = [x_drug]

        # 2. Process Diags
        if self.use_tripartite and self.diag_proj is not None:
            x_diag = self.diag_proj(self.diag_centered)
            node_features.append(x_diag)

        # 3. Process Labs
        if self.use_lab_nodes and self.lab_proj is not None:
            x_lab = self.lab_proj(self.lab_centered)
            node_features.append(x_lab)

        # Concatenate all nodes
        x = torch.cat(node_features, dim=0)

        # Message Passing
        for layer in self.layers:
            x = layer(x, edge_index, edge_type, edge_weight=edge_weight)

        # Return only drug nodes
        return x[:self.num_drugs]
