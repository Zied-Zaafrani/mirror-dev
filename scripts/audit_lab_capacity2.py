"""
Deep audit: how many lab types exist in the raw MIMIC-3 data before our 18-lab selection?
Checks the preprocessing script to find the original lab item ID list and what
the full MIMIC-3 lab universe looks like in terms of coverage.
"""
import pickle
import numpy as np
from pathlib import Path

# The current lab_data_mimic3.pkl was built with 18 pre-selected labs
# Let's find the preprocessing script to understand where these came from
print("="*60)
print("CURRENT lab_data_mimic3.pkl STATS")
print("="*60)

lab_path = Path("data/processed/lab_data_mimic3.pkl")
with open(lab_path, "rb") as f:
    lab_data = pickle.load(f)

lab_names = lab_data["lab_names"]
lab_itemids = lab_data["lab_itemids"]
has_lab = lab_data["has_lab"]  # (N_admissions,) bool array
lab_vectors = lab_data["lab_vectors"]  # (N, 36) = 18 values + 18 flags

print(f"\nCurrent lab panel: {len(lab_names)} labs")
for i, (name, iid) in enumerate(zip(lab_names, lab_itemids)):
    # Count coverage for this lab
    flags = lab_vectors[:, 18 + i]  # presence flags start at index 18
    coverage = flags.sum()
    pct = 100 * coverage / len(flags)
    print(f"  {i+1:2d}. {name:<15} (itemid={iid}) — coverage: {coverage:,}/{len(flags):,} ({pct:.1f}%)")

# Check if lab_vectors_50labs or lab_vectors_100labs exist
print("\n" + "="*60)
print("HIGH-DENSITY LAB FILES IN data/processed/")
print("="*60)
for f in sorted(Path("data/processed").glob("lab_vectors*")):
    size_mb = f.stat().st_size / 1e6
    try:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        if isinstance(d, dict):
            names = d.get("lab_names", [])
            n = len(names)
            shape = d.get("lab_vectors", np.zeros((0,0))).shape
            print(f"  {f.name} ({size_mb:.1f} MB): {n} labs, vectors shape={shape}")
            if names:
                print(f"    Labs: {names[:5]}...{names[-5:] if len(names)>5 else ''}")
        elif isinstance(d, np.ndarray):
            print(f"  {f.name} ({size_mb:.1f} MB): ndarray shape={d.shape}")
    except Exception as e:
        print(f"  {f.name} ({size_mb:.1f} MB): error reading — {e}")
