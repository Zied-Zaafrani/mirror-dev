"""
Audit script to clarify the 204 labs vs 131 drugs distinction.
Runs on the cohort and the lab cache to explain the saturation point.
"""
import pickle
import numpy as np
from pathlib import Path
import pandas as pd

COHORT_PATH = Path("data/processed/cohort_mimic3.pkl")
CACHE_PATH = Path("datasets/mimic-iii-clinical-database-1.4/lab_cache.pkl")

# 1. Load Cohort
with open(COHORT_PATH, "rb") as f:
    cohort = pickle.load(f)

num_drugs = cohort.get("num_drugs", "N/A")
num_diag = cohort.get("num_diag", "N/A")
num_proc = cohort.get("num_proc", "N/A")
hadm_ids = cohort["hadm_ids"]
hadm_ids_set = set(map(int, hadm_ids))

print("="*60)
print("COHORT STATISTICS (Medication Recommendation Task)")
print("="*60)
print(f"Total Admissions: {len(hadm_ids):,}")
print(f"Unique Drugs (Output space): {num_drugs}")
print(f"Unique Diagnoses (Input space): {num_diag}")
print(f"Unique Procedures (Input space): {num_proc}")

# 2. Load Lab Cache and filter by Cohort
print("\n" + "="*60)
print("LAB UNIVERSE AUDIT")
print("="*60)
with open(CACHE_PATH, "rb") as f:
    lab_df = pickle.load(f)

# Filter to cohort
lab_df = lab_df.dropna(subset=["HADM_ID", "VALUENUM"])
lab_df["HADM_ID"] = lab_df["HADM_ID"].astype(int)
cohort_labs = lab_df[lab_df["HADM_ID"].isin(hadm_ids_set)]

# Group by Lab ID (ITEMID) to see coverage
lab_counts = cohort_labs.groupby("ITEMID")["HADM_ID"].nunique().sort_values(ascending=False)

print(f"Total distinct lab tests found in cohort: {len(lab_counts)}")

# Breakdown by coverage threshold
thresholds = [0.9, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
print(f"\n{'Coverage %':<12} | {'Admissions':>10} | {'Num Labs':>10}")
print("-" * 40)
for t in thresholds:
    count = (lab_counts >= len(hadm_ids) * t).sum()
    print(f"{t*100:>10.1f}% | {int(len(hadm_ids)*t):>10,} | {count:>10}")

print("\nTop 5 Most Frequent Labs:")
# Get names from D_LABITEMS if possible
d_lab_path = Path("datasets/mimic-iii-clinical-database-1.4/D_LABITEMS.csv.gz")
if d_lab_path.exists():
    d_lab = pd.read_csv(d_lab_path, compression="gzip")
    d_lab_map = dict(zip(d_lab["ITEMID"], d_lab["LABEL"]))
else:
    d_lab_map = {}

for i, (itemid, cnt) in enumerate(lab_counts.head(5).items()):
    name = d_lab_map.get(itemid, f"ID {itemid}")
    print(f"  {i+1}. {name:<20} : {cnt:,} admissions ({100*cnt/len(hadm_ids):.1f}%)")

print("\nBottom 5 Labs (tail of the 446):")
for i, (itemid, cnt) in enumerate(lab_counts.tail(5).items()):
    name = d_lab_map.get(itemid, f"ID {itemid}")
    print(f"  {len(lab_counts)-4+i}. {name:<20} : {cnt:,} admissions ({100*cnt/len(hadm_ids):.4f}%)")
