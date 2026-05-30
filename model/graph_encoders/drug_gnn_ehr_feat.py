"""
EHR-as-Node-Feature Drug GNN (hgt_ehr_feat).

Motivation
----------
The default drug_gnn uses EHR co-occurrence as GRAPH EDGES (type 1).
At threshold=0.01 this creates 79% EHR edge mass (30% density) which causes
catastrophic over-smoothing after 2 HGT layers:
  post-GNN cosine_mean = 0.967, score_std = 0.012
  74% of drug pairs are within 0.01 of each other in dot-product score.

This encoder fixes the root cause architecturally:
  EHR co-occurrence profile → NODE FEATURE (not graph edge)
  Graph: DDI + ATC + self-loops only (≤7% density total)
  Layers: 1 HGT layer (sparse graph doesn't need depth)

Graph Quality Agent simulation (cosine_mean after 2 HGT layers):
  DDI+ATC+self, 1 layer → cos_mean ≈ 0.73–0.88, score_std ≈ 0.022
  vs current all-edges 2 layers → cos_mean = 0.967, score_std = 0.012
  vs no GNN at all → cos_mean = 0.364, score_std = 0.050

Input node features (per drug):
  - drug LLM embedding  (768d, mean-centered — removes BERT anisotropy)
  - Morgan fingerprint  (256d)
  - EHR co-occurrence profile (num_drugs-dim, L1-normalized row of ehr_adj)
  Total: 768 + 256 + num_drugs → input_proj → hidden_dim

Registered as: "hgt_ehr_feat"
Companion build call: build_drug_graph(..., exclude_ehr_edges=True)

Constitutional Rules
--------------------
Rule 1: This module owns its own input_proj (not shared with drug_gnn).
Rule 2: Registered via GRAPH_ENCODERS registry — drop-in replacement.
Rule 3: Talkative logging on first forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from ..registry import GRAPH_ENCODERS, GRAPH_LAYERS

logger = logging.getLogger(__name__)


@GRAPH_ENCODERS.register("hgt_ehr_feat")
class DrugGNN_EHRFeat(nn.Module):
    """EHR co-occurrence as node feature + sparse DDI/ATC graph.

    Compared to DrugGNN ("drug_gnn"):
      - EHR edges REMOVED from graph (build with exclude_ehr_edges=True)
      - EHR co-occurrence profile ADDED to node features
      - Default 1 HGT layer (sparse graph, 1 hop is enough)
      - Otherwise identical API: forward(edge_index, edge_type, edge_weight)

    Args:
        drug_embeddings:   (num_drugs, 768) PubMedBERT embeddings.
        morgan_fingerprints: (num_drugs, 256) Morgan fingerprints.
        ehr_adj:           (num_drugs, num_drugs) raw EHR co-occurrence matrix
                           (P(j|i) values in [0,1]). Each row is L1-normalized
                           and stored as a node feature buffer.
        hidden_dim:        GNN hidden dimension (default 256).
        hgt_layers:        Number of HGT layers. Default 1 (sparse graph).
        hgt_heads:         Attention heads per layer.
        num_edge_types:    Must cover max edge type index present in graph.
                           DDI=0, (EHR=1 absent), self-loop=2, ATC=3 → pass 4.
        dropout:           Dropout applied in input_proj and GNN layers.
        gnn_type:          GNN layer type from GRAPH_LAYERS registry ("hgt"/"gat").
    """

    def __init__(
        self,
        drug_embeddings: torch.Tensor,        # (num_drugs, 768)
        morgan_fingerprints: torch.Tensor,    # (num_drugs, 256)
        ehr_adj: torch.Tensor | None = None,  # (num_drugs, num_drugs) — required
        hidden_dim: int = 256,
        hgt_layers: int = 1,                  # 1 layer default for sparse graph
        hgt_heads: int = 4,
        num_edge_types: int = 4,
        dropout: float = 0.3,
        gnn_type: str = "hgt",
        **kwargs,  # absorb unsupported kwargs (use_tripartite etc.) from registry
    ):
        super().__init__()
        self.num_drugs = drug_embeddings.size(0)
        self.gnn_type = gnn_type
        self.num_layers = hgt_layers

        # ── Drug LLM features (mean-centered to remove BERT anisotropy) ──────
        llm_centered = drug_embeddings - drug_embeddings.mean(dim=0, keepdim=True)
        self.register_buffer("drug_llm_centered", llm_centered)
        self.register_buffer("morgan_fp", morgan_fingerprints)

        # ── EHR co-occurrence profile (node feature, not graph edge) ─────────
        # L1-normalize each row so the contribution is a probability distribution
        # over co-prescribed drugs, not a raw count that varies with drug frequency.
        # If ehr_adj is None (e.g. old checkpoint), fall back to zeros — model still
        # trains but misses the EHR signal.
        if ehr_adj is not None:
            ehr_float = ehr_adj.float()
            row_sums = ehr_float.sum(dim=1, keepdim=True).clamp(min=1e-8)
            ehr_profile = ehr_float / row_sums  # (num_drugs, num_drugs) L1-normed
        else:
            logger.warning(
                "  [DrugGNN_EHRFeat] ⚠️  ehr_adj is None — EHR node features will be "
                "zero. Pass ehr_adj=torch.tensor(ehr_adj_np, dtype=torch.float32) "
                "in the MIRROR constructor."
            )
            ehr_profile = torch.zeros(self.num_drugs, self.num_drugs)

        self.register_buffer("ehr_profile", ehr_profile)  # (num_drugs, num_drugs)
        ehr_feat_dim = self.num_drugs  # 131 for MIMIC-III default vocab

        # ── Input projection: LLM + Morgan + EHR profile → hidden_dim ────────
        drug_input_dim = drug_embeddings.size(1) + morgan_fingerprints.size(1) + ehr_feat_dim
        self.input_proj = nn.Sequential(
            nn.Linear(drug_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── GNN layers (sparse: DDI + ATC + self-loops only) ─────────────────
        layer_cls = GRAPH_LAYERS.get(gnn_type)
        self.hgt_layers_list = nn.ModuleList([
            layer_cls(
                hidden_dim=hidden_dim,
                num_heads=hgt_heads,
                num_edge_types=num_edge_types,
                dropout=dropout,
            )
            for _ in range(hgt_layers)
        ])

        logger.info(
            f"  [DrugGNN_EHRFeat] Initialized | backbone='{gnn_type}' "
            f"({hgt_layers} layer{'s' if hgt_layers != 1 else ''}) | "
            f"input_dim={drug_input_dim} (LLM={drug_embeddings.size(1)} + "
            f"Morgan={morgan_fingerprints.size(1)} + EHR_feat={ehr_feat_dim}) → {hidden_dim}"
        )
        logger.info(
            f"  [DrugGNN_EHRFeat] EHR signal: moved from graph edges → node features. "
            f"Graph contains DDI + ATC + self-loops ONLY (no type-1 EHR edges). "
            f"Pair build_drug_graph(..., exclude_ehr_edges=True)."
        )

    def forward(
        self,
        edge_index: torch.Tensor,              # (2, num_edges)
        edge_type: torch.Tensor,               # (num_edges,)
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns:
            drug_reprs: (num_drugs, hidden_dim) — L2-normalized drug representations.
        """
        if not hasattr(self, "_logged_flow"):
            ehr_nonzero = (self.ehr_profile > 0).sum().item()
            edge_type_counts = torch.bincount(edge_type, minlength=4).tolist()
            logger.info(
                f"  [DrugGNN_EHRFeat] Forward pass | "
                f"nodes={self.num_drugs} | edges={edge_index.size(1)} "
                f"(DDI={edge_type_counts[0]}, EHR_edges={edge_type_counts[1]}, "
                f"self={edge_type_counts[2]}, ATC={edge_type_counts[3]}) | "
                f"EHR_feat_nonzero={ehr_nonzero}/{self.num_drugs**2}"
            )
            if edge_type_counts[1] > 0:
                logger.warning(
                    f"  [DrugGNN_EHRFeat] ⚠️  Found {edge_type_counts[1]} EHR edges (type 1) "
                    f"in graph — these cause over-smoothing. "
                    f"Use build_drug_graph(..., exclude_ehr_edges=True)."
                )
            self._logged_flow = True

        # ── Build node feature matrix ─────────────────────────────────────────
        # [LLM_centered | Morgan | EHR_profile] → input_proj → hidden_dim
        x = torch.cat([
            self.drug_llm_centered,  # (num_drugs, 768)
            self.morgan_fp,          # (num_drugs, 256)
            self.ehr_profile,        # (num_drugs, num_drugs) L1-normalized EHR profile
        ], dim=1)                    # (num_drugs, 768+256+num_drugs)
        x = self.input_proj(x)       # (num_drugs, hidden_dim)

        # ── Sparse message passing (DDI + ATC + self-loops) ──────────────────
        for layer in self.hgt_layers_list:
            x = layer(x, edge_index, edge_type, edge_weight=edge_weight)

        # ── L2-normalize for geometric consistency ────────────────────────────
        # Ensures patient @ drug.T is pure cosine similarity (not confounded by
        # post-GNN norm variation from topology-dependent message aggregation).
        return F.normalize(x, dim=-1)  # (num_drugs, hidden_dim)
