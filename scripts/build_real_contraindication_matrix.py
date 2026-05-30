"""
Build Real Drug-Lab Contraindication Prior Matrix based on clinical guidelines.
References: MedGCN, HSGNN, and standard clinical practice (e.g. Hyperkalemia -> avoid ACEi).
"""
import json
from pathlib import Path

# The 18 core labs in MIRROR (from lab_data_mimic3.pkl and dataset.py)
LAB_NAMES = [
    'Creatinine', 'BUN', 'ALT', 'AST', 'Bilirubin', 'Alk Phos', 'INR', 'PT', 'PTT', 
    'Sodium', 'Potassium', 'Magnesium', 'Calcium', 'Glucose', 'Albumin', 'Lactate', 
    'WBC', 'Hemoglobin'
]

# Drug Vocabulary (from cohort_mimic3.pkl)
# 46: C09A (ACE inhibitors)
# 47: C09C (ARBs)
# 39: C03D (Potassium-sparing diuretics)
# 90: M01A (NSAIDs)
# 20: A12B (Potassium supplements)
# 28: C01A (Cardiac glycosides / Digoxin)
# 14: A10B (Metformin)
# 100: N02B (Paracetamol / Acetaminophen)
# 23: B01A (Anticoagulants)

def main():
    matrix = {}

    # --- POTASSIUM (Index 10) ---
    # Hyperkalemia (High Potassium, Bin 3)
    # Avoid drugs that raise potassium further
    matrix["10_3"] = [46, 47, 39, 90, 20] 
    
    # Hypokalemia (Low Potassium, Bin 1)
    # Avoid Digoxin (increases toxicity risk)
    matrix["10_1"] = [28]

    # --- CREATININE (Index 0) ---
    # Renal Failure (High Creatinine, Bin 3)
    # Avoid nephrotoxic or renal-cleared drugs
    matrix["0_3"] = [90, 39, 14]

    # --- ALT / AST (Indices 2, 3) ---
    # Hepatic Failure (High ALT/AST, Bin 3)
    # Avoid hepatotoxic drugs
    matrix["2_3"] = [100]
    matrix["3_3"] = [100]

    # --- COAGULATION (INR/PT/PTT, Indices 6, 7, 8) ---
    # High Bleeding Risk (Bin 3)
    # Avoid Anticoagulants and NSAIDs
    matrix["6_3"] = [23, 90]
    matrix["7_3"] = [23, 90]
    matrix["8_3"] = [23, 90]

    out_dir = Path("src/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "contraindication_matrix.json"
    
    with open(out_path, "w") as f:
        json.dump(matrix, f, indent=2)
        
    print(f"Created REAL contraindication matrix at {out_path} with {len(matrix)} clinical rules.")
    for key, drugs in matrix.items():
        lab_idx = int(key.split('_')[0])
        bin_val = int(key.split('_')[1])
        state = "HIGH" if bin_val == 3 else "LOW"
        print(f"  - {LAB_NAMES[lab_idx]} {state}: {drugs}")

if __name__ == "__main__":
    main()
