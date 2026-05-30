"""
Abstract base class for all graph encoder layers in MIRROR.

All graph layers must implement the same forward signature so DrugGNN
can swap them via the registry without any conditional logic.
"""

import torch
import torch.nn as nn


class BaseGraphLayer(nn.Module):
    """Abstract interface for all graph layers.

    Subclasses must implement forward() with this exact signature.
    """

    def forward(
        self,
        x: torch.Tensor,            # (num_nodes, hidden_dim)
        edge_index: torch.Tensor,    # (2, num_edges) — [source, target]
        edge_type: torch.Tensor,     # (num_edges,) — type in [0, K-1]
        edge_weight: torch.Tensor | None = None,  # (num_edges,) — directed weights
    ) -> torch.Tensor:
        """Returns x_out: (num_nodes, hidden_dim) — updated node features."""
        raise NotImplementedError
