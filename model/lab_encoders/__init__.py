"""
MIRROR Lab Encoders — champion configuration.

Champion: flat (FlatLabEncoder — 200-lab × 2 = 400d input, projects to hidden_dim).
Confirmed best by Sweep 11b (N_labs=200) and used throughout all subsequent sweeps.
"""

from . import flat
from .flat import FlatLabEncoder
