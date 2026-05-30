"""
Check for violations between ground truth prescriptions and contraindication rules.
Fixed records structure.
"""
import pickle
import numpy as np
import sys
from pathlib import Path
import json

# Add src to path for dataset imports
sys.path.append('src')

COHORT_PATH = Path("data/processed/cohort_mimic3.pkl")
RECORDS_PATH = Path("data/processed/records_final.pkl")
LAB_PKL = Path("data/processed/lab_data_mimic3.pkl")
RULES_PATH = Path("src/data/contraindication_matrix.json")

print("Loading data...")
with open(COHORT_PATH, "rb") as f:
    cohort = pickle.load(f)
with open(RECORDS_PATH, "rb") as f:
    records = pickle.load(f) # List of patients, each patient is a list of visits
with open(LAB_PKL, "rb") as f:
    labs = pickle.load(f)
with open(RULES_PATH, "rb") as f:
    rules = json.load(f)

# Build contraindication mask per lab_bin
num_drugs = cohort["num_drugs"]
contra_mask = np.zeros((18, 4, num_drugs))
for key, drugs in rules.items():
    l_idx, b_val = map(int, key.split('_'))
    if l_idx < 18:
        for d in drugs:
            contra_mask[l_idx, b_val, d] = 1.0

# Pre-map all hadm_ids to their last visit ground truth
print("Mapping admissions to last visits...")
hadm_to_gt = {}
for patient in records:
    for visit in patient:
        # visit structure: [diags, procs, drugs, hadm_id]
        if len(visit) >= 4:
            h_id = int(visit[3])
            hadm_to_gt[h_id] = visit[2] # drugs list

# 3. Check violations in test set
from dataset import compute_lab_bins
lab_means = labs["zscore_means"]
lab_stds = labs["zscore_stds"]
lab_names = labs["lab_names"]
all_lab_vectors = labs["lab_vectors"]
hadm_ids = cohort["hadm_ids"]
hadm_to_idx = {int(h): i for i, h in enumerate(hadm_ids)}

test_mask = np.array(cohort["split"]) == "test"
test_hadm_ids = [int(h) for i, h in enumerate(hadm_ids) if test_mask[i]]

print(f"Checking {len(test_hadm_ids)} test admissions...")
total_violations = 0
admissions_with_violations = 0
total_prescriptions = 0
violations_per_drug = {}

for h_id in test_hadm_ids:
    if h_id not in hadm_to_gt: continue
    idx = hadm_to_idx[h_id]
    
    last_visit_drugs = hadm_to_gt[h_id]
    lab_vec = all_lab_vectors[idx]
    bins = compute_lab_bins(lab_vec, lab_means, lab_stds, lab_names)
    
    has_violation = False
    for d_idx in last_visit_drugs:
        total_prescriptions += 1
        for l_idx in range(min(18, len(bins))):
            b_val = bins[l_idx]
            if b_val > 0 and b_val < 4:
                if contra_mask[l_idx, b_val, d_idx] == 1.0:
                    total_violations += 1
                    has_violation = True
                    violations_per_drug[d_idx] = violations_per_drug.get(d_idx, 0) + 1
    
    if has_violation:
        admissions_with_violations += 1

print("\nVIOLATION SUMMARY")
print("="*40)
print(f"Total Prescriptions Checked: {total_prescriptions}")
print(f"Total Rule Violations Found: {total_violations}")
print(f"Admissions with at least 1 violation: {admissions_with_violations} / {len(test_hadm_ids)} ({100*admissions_with_violations/len(test_hadm_ids):.1f}%)")

if violations_per_drug:
    print("\nTop 5 Violating Drugs:")
    sorted_v = sorted(violations_per_drug.items(), key=lambda x: x[1], reverse=True)
    for d_idx, count in sorted_v[:5]:
        print(f"  Drug Index {d_idx}: {count} violations")
