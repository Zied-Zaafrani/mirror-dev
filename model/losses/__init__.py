"""
MIRROR Loss Functions — champion configuration.

Champion loss (L4_jac_heavy): bce_weight=0.3, soft_jaccard_weight=1.5, margin_weight=0.05
Confirmed best by Sweep 15a across all 5 random seeds.
"""

from . import classification
from . import regularization
from . import clinical
