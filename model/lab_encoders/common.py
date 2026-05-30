"""
Shared helpers for MIRROR lab encoders.
"""

import torch

def _split_lab_vec(lab_vector: torch.Tensor, num_labs: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Split lab vector → (z-scores, presence mask).
    Supports 36d (18 labs), 100d (50 labs), or any 2*N dimension.

    Returns:
        lab_values:  (batch, N) z-scored values (0 where missing)
        lab_present: (batch, N) float, 1=lab present, 0=missing
    """
    if num_labs is not None:
        n = num_labs
    else:
        dim = lab_vector.shape[1]
        # Heuristic: Only assume trends if dim is 72 or 144
        if dim in [72, 144]:
            n = dim // 4
        else:
            n = dim // 2

    # FIX-B16: fail loudly on width mismatch instead of producing an empty
    # missing_flags slice that then broadcasts incorrectly with z-scores.
    if lab_vector.shape[1] < 2 * n:
        raise ValueError(
            f"_split_lab_vec: lab_vector width {lab_vector.shape[1]} < 2*num_labs={2*n}. "
            f"Check that lab_dim in config matches the lab encoder's num_labs."
        )

    lab_values = lab_vector[:, :n]           # z-scores
    missing_flags = lab_vector[:, n : 2*n]   # 1=missing
    lab_present = 1.0 - missing_flags        # 1=present
    return lab_values, lab_present
