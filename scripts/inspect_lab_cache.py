"""
Inspect lab_cache.pkl and count all distinct lab types in the cohort,
then generate all Phase 9 pkl files locally.
"""
import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

MIMIC_DIR = Path("datasets/mimic-iii-clinical-database-1.4")
COHORT_PATH = Path("data/processed/cohort_mimic3.pkl")

# ── Step 1: Check lab_cache.pkl ──
cache_path = MIMIC_DIR / "lab_cache.pkl"
print(f"Checking lab_cache.pkl ({cache_path.stat().st_size/1e6:.0f} MB)...")
with open(cache_path, "rb") as f:
    cache = pickle.load(f)

print(f"Type: {type(cache)}")
if isinstance(cache, pd.DataFrame):
    print(f"DataFrame shape: {cache.shape}")
    print(f"Columns: {cache.columns.tolist()}")
    print(cache.head(3))
elif isinstance(cache, dict):
    print(f"Dict keys: {list(cache.keys())[:10]}")
    for k, v in list(cache.items())[:3]:
        vlen = len(v) if hasattr(v, "__len__") else "scalar"
        print(f"  {repr(k)[:60]} -> type={type(v).__name__} len={vlen}")
else:
    print(f"Unknown type: {type(cache)}")
    if hasattr(cache, "__len__"):
        print(f"  len={len(cache)}")
