"""
MIRROR Decoders / Scorers — champion configuration.

Champion: heidr (HEIDRScorer — drug self-attention + drug-visit cross-attention + copy).
Confirmed best by Sweep 14c across all 5 random seeds and all modality contexts.
"""

from .heidr_decoder import HEIDRScorer

__all__ = ["HEIDRScorer"]
