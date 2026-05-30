"""
MIRROR Aggregators — champion configuration.

Champion: last (LastAggregator — extracts final non-padded visit state).
Confirmed best by Sweep 13a. ar_max_seq_len=20 is the optimal sequence length.
"""

from . import last
