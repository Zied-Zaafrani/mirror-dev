"""
Counts how many unique lab test types exist in the MIMIC-3 processed lab data,
and breaks down coverage by patient to help decide the optimal lab count ceiling.
"""
import pickle
import numpy as np
from pathlib import Path

lab_path = Path("data/processed/lab_data_mimic3.pkl")
if not lab_path.exists():
    print(f"ERROR: {lab_path} not found. Run from the MIRROR root directory.")
    exit(1)

with open(lab_path, "rb") as f:
    lab_data = pickle.load(f)

print(f"Type of lab_data: {type(lab_data)}")
if isinstance(lab_data, dict):
    print(f"Top-level keys: {list(lab_data.keys())[:10]}")

# Inspect structure
if "lab_names" in lab_data:
    lab_names = lab_data["lab_names"]
    print(f"\nTotal named lab types: {len(lab_names)}")
    print(f"Lab names: {lab_names}")
elif "itemid_to_idx" in lab_data:
    print(f"\nTotal lab item IDs: {len(lab_data['itemid_to_idx'])}")
elif "lab_vectors" in lab_data:
    vectors = lab_data["lab_vectors"]
    print(f"\nLab vectors shape: {np.array(list(vectors.values())[0]).shape if vectors else 'empty'}")
    print(f"Number of patients with lab data: {len(vectors)}")

# Try to find the actual number of lab dimensions
for key in lab_data.keys():
    val = lab_data[key]
    if hasattr(val, '__len__') and not isinstance(val, str):
        print(f"  key='{key}' type={type(val).__name__} len={len(val)}")
    elif isinstance(val, np.ndarray):
        print(f"  key='{key}' shape={val.shape}")
