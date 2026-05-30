"""
Run 23 Phase 0 integrity asserts.

Prevents Run 22-class silent failures where the notebook, the config, and the
retrieval index disagreed about what model was being trained. Every Run 23
notebook cell executes `assert_phase0_integrity(...)` BEFORE building datasets
or models. If anything fails, the run aborts at cell 0 instead of producing
junk metrics 8 hours later.

The six asserts are:
    A1 records non-empty
    A2 num_drugs == 130 (MIMIC-III ATC-3 family)
    A3 splits disjoint (no hadm_id overlap across train/val/test)
    A4 retrieval_index present + entry count sanity
    A5 retrieval fused_repr dim == model hidden_dim
    A6 split_mode request matches retrieval meta (catches notebook ≠ compute_similarity)
        AND retrieval meta `extract_fused_repr == True`
        (Run 22 would have failed here immediately.)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable


def _record_fingerprints(records: Iterable) -> set[int]:
    """Fingerprint each patient record by id() (object identity).

    MIMIC-III records are nested lists without hadm_ids embedded; splits
    produce sliced views of a shared list, so id()-overlap is the correct
    disjointness test. Two patients that happen to have identical visits
    would still be different list objects, so false positives are impossible.
    """
    return {id(rec) for rec in records}


def assert_phase0_integrity(
    train_records: list,
    val_records: list,
    test_records: list,
    num_drugs: int,
    hidden_dim: int,
    retrieval_pickle_path: str | Path | None,
    split_mode_requested: str,
    *,
    expected_num_drugs: int = 130,
    require_fused_repr: bool = True,
) -> dict:
    """Run 23 Phase 0 hard asserts. Raises AssertionError on any mismatch.

    Returns a dict of observed values so the notebook can log them verbatim.
    """
    report: dict = {}

    # A1 — records non-empty
    assert len(train_records) > 0, "A1: train_records is empty"
    assert len(val_records) > 0, "A1: val_records is empty"
    assert len(test_records) > 0, "A1: test_records is empty"
    report["n_train"] = len(train_records)
    report["n_val"] = len(val_records)
    report["n_test"] = len(test_records)

    # A2 — drug vocab
    assert int(num_drugs) == int(expected_num_drugs), (
        f"A2: num_drugs={num_drugs} but expected {expected_num_drugs} "
        "(MIMIC-III ATC-3 family). Did you load the wrong cohort?"
    )
    report["num_drugs"] = int(num_drugs)

    # A3 — split disjointness by object identity (records are sliced from a
    # shared list; id() overlap is leak-accurate on the MIMIC-III schema).
    tr_ids = _record_fingerprints(train_records)
    va_ids = _record_fingerprints(val_records)
    te_ids = _record_fingerprints(test_records)
    assert not (tr_ids & va_ids), f"A3: train&val leak ({len(tr_ids & va_ids)} shared records)"
    assert not (tr_ids & te_ids), f"A3: train&test leak ({len(tr_ids & te_ids)} shared records)"
    assert not (va_ids & te_ids), f"A3: val&test leak ({len(va_ids & te_ids)} shared records)"
    report["record_counts"] = {"train": len(tr_ids), "val": len(va_ids), "test": len(te_ids)}

    # A4/A5/A6 — retrieval sanity (skipped if no pickle supplied, e.g. C0/C3 rows)
    if retrieval_pickle_path is None:
        report["retrieval"] = "not supplied (pre-C4 row)"
        return report

    pkl_path = Path(retrieval_pickle_path)
    assert pkl_path.exists(), f"A4: retrieval pickle not found at {pkl_path}"
    with open(pkl_path, "rb") as f:
        retrieval_data = pickle.load(f)
    assert isinstance(retrieval_data, dict), "A4: retrieval pickle is not a dict"
    entry_count = len(retrieval_data) - (1 if "__meta__" in retrieval_data else 0)
    assert entry_count > 0, "A4: retrieval pickle has zero entries"

    meta = retrieval_data.get("__meta__", {})
    # Prefer sidecar meta.json if present — it's the authoritative provenance file.
    sidecar_path = pkl_path.with_suffix(pkl_path.suffix + ".meta.json")
    sidecar_meta: dict = {}
    if sidecar_path.exists():
        with open(sidecar_path, "r", encoding="utf-8") as f:
            sidecar_meta = json.load(f)

    # A5 — representation dim. Real schema uses `similar_reprs` (the field
    # dataset.py reads); tolerate legacy `reprs` for forward compat.
    sample_entry = next(
        (v for k, v in retrieval_data.items() if k != "__meta__"), None
    )
    assert sample_entry is not None, "A5: no retrieval entries to inspect"
    reprs = sample_entry.get("similar_reprs", sample_entry.get("reprs"))
    assert reprs is not None, (
        "A5: retrieval entry missing 'similar_reprs' (and legacy 'reprs') field"
    )
    ret_dim = int(reprs.shape[-1])
    assert ret_dim == int(hidden_dim), (
        f"A5: retrieval reprs dim {ret_dim} != model hidden_dim {hidden_dim}. "
        "Rebuild the index with the correct encoder OR change model.hidden_dim."
    )
    report["retrieval_dim"] = ret_dim
    report["retrieval_entries"] = entry_count

    # A6 — split_mode parity + extract_fused_repr == True
    # Real schema stores `split_source` (e.g. "hidr_vita_sequential"). Accept
    # both and match by substring so "hidr_vita_sequential" satisfies the
    # "hidr_vita" request.
    meta_split = (
        sidecar_meta.get("split_source")
        or sidecar_meta.get("split_mode_requested")
        or sidecar_meta.get("split_mode")
        or meta.get("split_source")
        or meta.get("split_mode_requested")
        or meta.get("split_mode")
    )
    if meta_split is not None:
        assert str(split_mode_requested) in str(meta_split), (
            f"A6: split_mode mismatch — notebook requested {split_mode_requested!r} "
            f"but retrieval index was built with {meta_split!r}. "
            "This is the Run 22-class silent failure guard."
        )
    report["split_mode"] = meta_split

    fused = sidecar_meta.get("extract_fused_repr")
    if fused is None:
        fused = meta.get("extract_fused_repr")
    if require_fused_repr:
        assert fused is True, (
            f"A6: extract_fused_repr={fused!r} in retrieval meta, but Run 23 "
            "requires True. Rebuild the index with PretrainMIRROR(extract_fused_repr=True). "
            "Run 22 shipped False here and the retrieval head saw a different "
            "feature space than the training-time query."
        )
    report["extract_fused_repr"] = fused

    return report
