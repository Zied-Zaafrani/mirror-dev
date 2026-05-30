"""
MIRROR Graph Encoders — champion configuration.

Champion: drug_gnn (graph encoder) + gcn (graph layer).
Confirmed best by Sweep 15b across all 5 random seeds.
"""

from .drug_gnn import DrugGNN
from .gcn import GCNLayer

__all__ = ["DrugGNN", "GCNLayer"]
