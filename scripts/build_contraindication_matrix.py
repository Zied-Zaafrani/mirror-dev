"""
Build Drug-Lab Contraindication Prior Matrix (Medically Grounded).

This script generates a contraindication matrix mapping abnormal lab states
to specific ATC codes derived from clinical guidelines (e.g., Metformin vs Cr).
"""
import json
from pathlib import Path

# The 18 labs in MIRROR
LAB_NAMES = [
    'Anion Gap', 'Bicarbonate', 'Blood Urea Nitrogen', 'Calcium Total', 'Chloride',
    'Creatinine', 'Glucose', 'Magnesium', 'Phosphate', 'Potassium', 'Sodium',
    'Hematocrit', 'Hemoglobin', 'MCH', 'MCHC', 'MCV', 'Platelet Count', 'White Blood Cells'
]

# Mapping rules: { lab_idx: { bin_val: [atc_indices] } }
# BINS: 1=LOW, 3=HIGH
# ATC Mappings:
# 14: A10B (Metformin)
# 90: M01A (NSAIDs)
# 46: C09A (ACEi)
# 47: C09C (ARBs)
# 39: C03D (Spironolactone)
# 38: C03C (Furosemide)
# 23: B01A (Warfarin/Heparin/Aspirin)
# 67: H02A (Prednisone/Steroids)

CLINICAL_RULES = {
    5: { # Creatinine
        3: [14, 90] # HIGH -> Metformin, NSAIDs
    },
    9: { # Potassium
        3: [46, 47, 39], # HIGH -> ACEi, ARB, Spironolactone
        1: [38] # LOW -> Furosemide
    },
    16: { # Platelet Count
        1: [23] # LOW -> Anticoagulants/Antiplatelets
    },
    6: { # Glucose
        3: [67] # HIGH -> Steroids
    },
    10: { # Sodium
        1: [38, 65] # LOW -> Diuretics, Vasopressin analogs
    }
}

def main():
    matrix = {}
    
    for lab_idx, rules in CLINICAL_RULES.items():
        for bin_val, drug_indices in rules.items():
            key = f"{lab_idx}_{bin_val}"
            matrix[key] = drug_indices
            
    out_dir = Path("src/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "contraindication_matrix.json"
    
    with open(out_path, "w") as f:
        json.dump(matrix, f, indent=2)
        
    print(f"Created clinically-grounded contraindication matrix at {out_path} with {len(matrix)} rules.")

if __name__ == "__main__":
    main()
