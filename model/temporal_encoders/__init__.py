"""
MIRROR Temporal Encoders — champion configuration.

Champion: imdr_infused (IMDR-style early drug-knowledge infusion before Transformer).
Confirmed best by Sweep 14a across all 5 random seeds.

transformer_encoder is kept as a sub-component (IMDRInfusedEncoder depends on it)
but is not registered as a standalone option.
"""

from . import imdr_encoder
