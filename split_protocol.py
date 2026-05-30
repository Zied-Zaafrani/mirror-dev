"""Shared split protocol helpers for fair experiment comparability.

Provides a single implementation for train/val/test slicing so pretrain,
retrieval-index building, and training can be locked to the same protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitResult:
    train_idx: list[int]
    val_idx: list[int]
    test_idx: list[int]
    split_source: str
    split_seed_used: int | str


def _default_sequential_split(n: int) -> tuple[list[int], list[int], list[int]]:
    """HI-DR/VITA-compatible slicing: 2/3 train, 1/6 test, 1/6 val."""
    train_end = int(n * 2 / 3)
    test_end = train_end + int((n - train_end) / 2)
    train_idx = list(range(0, train_end))
    test_idx = list(range(train_end, test_end))
    val_idx = list(range(test_end, n))
    return train_idx, val_idx, test_idx


def _default_permutation_split(n: int, seed: int) -> tuple[list[int], list[int], list[int]]:
    """Deterministic random split with the same 2/3, 1/6, 1/6 proportions."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    train_end = int(n * 2 / 3)
    val_end = int(n * 5 / 6)
    train_idx = sorted(indices[:train_end].tolist())
    val_idx = sorted(indices[train_end:val_end].tolist())
    test_idx = sorted(indices[val_end:].tolist())
    return train_idx, val_idx, test_idx


def _is_valid_split_indices(train_idx: list[int], val_idx: list[int], test_idx: list[int], n: int) -> bool:
    merged = train_idx + val_idx + test_idx
    # FIX-B44: allow splits that legitimately omit some records (e.g., patients
    # with only 1 visit were filtered upstream). The split is valid as long as
    # indices are disjoint, in-range, and the union is a subset of [0, n).
    # Old check `len(merged) != n` rejected these and triggered a random
    # permutation fallback, destroying cross-model parity.
    if len(merged) > n:
        return False
    if len(set(merged)) != len(merged):
        return False
    if n == 0:
        return True
    if not merged:
        return False
    return min(merged) >= 0 and max(merged) < n


def compute_split_indices(
    num_records: int,
    cohort: dict,
    split_mode: str,
    seed: int,
    require_cohort_indices: bool = False,
) -> SplitResult:
    """Compute train/val/test indices from a unified protocol.

    split_mode values:
      - "cohort": use cohort split_indices when valid; fallback to permutation
      - "permutation": deterministic permutation split
      - "sequential": deterministic in-order split
      - "hidr_vita": alias of sequential for explicit benchmark parity runs
    """
    split_mode = (split_mode or "cohort").strip().lower()

    if split_mode in ("sequential", "hidr_vita"):
        train_idx, val_idx, test_idx = _default_sequential_split(num_records)
        source = "hidr_vita_sequential" if split_mode == "hidr_vita" else "sequential"
        return SplitResult(train_idx, val_idx, test_idx, source, "n/a")

    if split_mode == "permutation":
        train_idx, val_idx, test_idx = _default_permutation_split(num_records, seed)
        return SplitResult(train_idx, val_idx, test_idx, "runtime_permutation", seed)

    if split_mode != "cohort":
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    split_idx = cohort.get("split_indices")
    if split_idx and all(k in split_idx for k in ("train", "val", "test")):
        train_idx = sorted(int(i) for i in split_idx["train"])
        val_idx = sorted(int(i) for i in split_idx["val"])
        test_idx = sorted(int(i) for i in split_idx["test"])
        if _is_valid_split_indices(train_idx, val_idx, test_idx, num_records):
            return SplitResult(
                train_idx,
                val_idx,
                test_idx,
                "cohort_metadata",
                cohort.get("split_seed", "unknown"),
            )

    if require_cohort_indices:
        raise ValueError(
            "split_mode=cohort requires valid cohort split_indices, but none were found or they were invalid."
        )

    train_idx, val_idx, test_idx = _default_permutation_split(num_records, seed)
    return SplitResult(train_idx, val_idx, test_idx, "runtime_permutation", seed)
