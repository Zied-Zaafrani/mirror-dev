"""
MIRROR System Integrity Suite.

Consolidates 'Rule of Many' guardrails to prevent configuration drift 
between training, retrieval indexing, and pretraining.
"""
import logging
import pickle
import json
from pathlib import Path
from typing import Iterable, Any

logger = logging.getLogger(__name__)

def _record_fingerprints(records: Iterable) -> set[int]:
    """Fingerprint each patient record by id() (object identity)."""
    return {id(rec) for rec in records}

def verify_system_integrity(
    train_records: list,
    val_records: list,
    test_records: list,
    num_drugs: int,
    hidden_dim: int,
    retrieval_pickle_path: str | Path | None,
    split_mode_requested: str,
    **kwargs
) -> dict[str, Any]:
    """
    MIRROR Integrity Assertion Suite (The 'Rule of Many').
    Raises AssertionError on any mismatch.
    """
    report = {}
    logger.info("[INTEGRITY] Executing System Integrity Checks...")

    # A1: Records non-empty
    assert len(train_records) > 0, "A1: train_records is empty"
    assert len(val_records) > 0, "A1: val_records is empty"
    assert len(test_records) > 0, "A1: test_records is empty"
    report["n_train"] = len(train_records)
    report["n_val"] = len(val_records)
    report["n_test"] = len(test_records)

    # A2: Drug Vocabulary Parity
    expected_drugs = kwargs.get("expected_num_drugs", 131) # Standard for MIMIC-III ATC-3
    assert int(num_drugs) == int(expected_drugs), \
        f"A2: num_drugs={num_drugs} but expected {expected_drugs}. Check cohort loading."
    report["num_drugs"] = int(num_drugs)

    # A3: Split Disjointness
    tr_ids = _record_fingerprints(train_records)
    va_ids = _record_fingerprints(val_records)
    te_ids = _record_fingerprints(test_records)
    assert not (tr_ids & va_ids), f"A3: train & val leak detected!"
    assert not (tr_ids & te_ids), f"A3: train & test leak detected!"
    assert not (va_ids & te_ids), f"A3: val & test leak detected!"
    report["disjointness"] = "Verified"

    # A4/A5/A6: Retrieval Sanity
    if retrieval_pickle_path:
        _verify_retrieval_integrity(retrieval_pickle_path, hidden_dim, split_mode_requested, report)
    else:
        report["retrieval"] = "Bypassed (Not in config)"

    logger.info(f"[INTEGRITY] Success: {report}")
    return report

def _verify_retrieval_integrity(pkl_path: str | Path, hidden_dim: int, split_mode: str, report: dict):
    pkl_path = Path(pkl_path)
    assert pkl_path.exists(), f"A4: Retrieval pickle not found at {pkl_path}"
    
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    entry_count = len(data) - (1 if "__meta__" in data else 0)
    assert entry_count > 0, "A4: Retrieval pickle is empty"
    
    # A5: Hidden Dim Alignment
    sample = next((v for k, v in data.items() if k != "__meta__"), None)
    assert sample is not None, "A5: No retrieval entries to inspect"
    reprs = sample.get("similar_reprs", sample.get("reprs"))
    assert reprs is not None, "A5: Missing 'similar_reprs' in index"
    
    ret_dim = int(reprs.shape[-1])
    assert ret_dim == int(hidden_dim), \
        f"A5: Retrieval dim {ret_dim} != Model dim {hidden_dim}. Rebuild index."
    
    # A6: Split Mode Parity
    meta = data.get("__meta__", {})
    # Check for sidecar JSON as well
    sidecar = pkl_path.with_suffix(pkl_path.suffix + ".meta.json")
    if sidecar.exists():
        with open(sidecar, "r", encoding="utf-8") as f:
            meta.update(json.load(f))
            
    meta_split = meta.get("split_source") or meta.get("split_mode")
    if meta_split:
        assert str(split_mode) in str(meta_split), \
            f"A6: Split mismatch. Config: {split_mode}, Index: {meta_split}."
    
    report["retrieval_status"] = "Verified"
    report["retrieval_entries"] = entry_count
    report["retrieval_dim"] = ret_dim
