"""
Build a static lab-drug prior matrix from cohort data.

This script generates a (num_labs, num_drugs) binary matrix encoding
clinically-established relationships between lab tests and drug classes.

IMPORTANT: The system uses dynamic `num_labs` (default 200 in production,
configurable via --num_labs). The lab set is determined by the lab PKL file,
NOT by the 18-lab `lab_ranges.py` (which is only for clinical binning).

The prior matrix is stored as `lab_drug_prior.npy` and can optionally be
loaded as a soft signal during training (though the primary drug-lab graph
edges are built dynamically by `build_drug_lab_edges()` in train.py).

Usage:
    python build_lab_drug_prior.py                          # Uses lab_vectors_200labs.pkl
    python build_lab_drug_prior.py --lab_pkl path/to/other.pkl
"""

import numpy as np
import pickle
from pathlib import Path

# ATC prefix -> clinical rationale for the drug-lab relationship.
# These map ATC drug class prefixes to the clinical lab domains they affect.
# When a lab test and a drug share a clinical domain, we set prior[lab, drug] = 1.
#
# The lab side is matched by NAME PATTERN (case-insensitive substring),
# not by hardcoded ITEMID, so it works with any lab set (18, 200, 446, etc.)
#
CLINICAL_RELATIONSHIPS = [
    # ── Kidney ──
    # Nephrotoxic drugs need renal monitoring
    {
        "lab_patterns": ["creatinine", "bun", "urea nitrogen"],
        "atc_prefixes": ["J01X", "J01G", "C09"],  # Vancomycin, Aminoglycosides, ACE inhibitors
        "rationale": "Nephrotoxic drugs require renal function monitoring",
    },
    # ── Liver ──
    # Hepatotoxic drugs need liver function monitoring
    {
        "lab_patterns": ["alt", "ast", "bilirubin", "alkaline phosphatase", "alk phos"],
        "atc_prefixes": ["C10A", "N02B"],  # Statins, Acetaminophen
        "rationale": "Hepatotoxic drugs require liver function monitoring",
    },
    # ── Coagulation ──
    {
        "lab_patterns": ["inr", "pt", "ptt", "prothrombin", "partial thromboplastin"],
        "atc_prefixes": ["B01A"],  # Anticoagulants
        "rationale": "Anticoagulants directly affect coagulation cascade",
    },
    # ── Electrolytes ──
    {
        "lab_patterns": ["sodium"],
        "atc_prefixes": ["C03", "H02A", "N03"],  # Diuretics, Mineralocorticoids, Antiepileptics
        "rationale": "Diuretics/carbamazepine cause hyponatremia",
    },
    {
        "lab_patterns": ["potassium"],
        "atc_prefixes": ["C09", "C03"],  # ACE inhibitors, Diuretics
        "rationale": "ACE inhibitors cause hyperkalemia, diuretics cause hypokalemia",
    },
    {
        "lab_patterns": ["magnesium"],
        "atc_prefixes": ["C01", "C03", "A02B"],  # Cardiac, Diuretics, PPIs
        "rationale": "PPIs and diuretics deplete magnesium; cardiac drugs need Mg monitoring",
    },
    {
        "lab_patterns": ["calcium"],
        "atc_prefixes": ["M05B", "A11C", "C01A"],  # Bisphosphonates, Vitamin D, Cardiac glycosides
        "rationale": "Calcium levels affect digoxin toxicity and bone metabolism drugs",
    },
    {
        "lab_patterns": ["phosph"],  # Phosphate, Phosphorus
        "atc_prefixes": ["M05B", "A11C", "C09"],  # Bisphosphonates, Vitamin D, ACE inhibitors
        "rationale": "Phosphate homeostasis linked to renal and bone drugs",
    },
    {
        "lab_patterns": ["chloride", "bicarbonate", "co2"],
        "atc_prefixes": ["C03"],  # Diuretics
        "rationale": "Diuretics affect acid-base and chloride balance",
    },
    # ── Metabolic ──
    {
        "lab_patterns": ["glucose"],
        "atc_prefixes": ["A10"],  # Antidiabetics
        "rationale": "Antidiabetics directly target glucose regulation",
    },
    {
        "lab_patterns": ["albumin"],
        "atc_prefixes": ["B05", "C03"],  # Blood substitutes/nutrition, Diuretics
        "rationale": "Albumin levels guide fluid management and drug dosing",
    },
    {
        "lab_patterns": ["lactate"],
        "atc_prefixes": ["C01C", "J01C", "J01D"],  # Vasopressors, Beta-lactams
        "rationale": "Lactate signals sepsis severity; guides vasopressor and antibiotic therapy",
    },
    {
        "lab_patterns": ["hba1c", "hemoglobin a1c", "glycated"],
        "atc_prefixes": ["A10"],  # Antidiabetics
        "rationale": "HbA1c is the gold-standard marker for glycemic control",
    },
    {
        "lab_patterns": ["uric acid", "urate"],
        "atc_prefixes": ["M04A", "C03"],  # Anti-gout, Diuretics
        "rationale": "Uric acid directly monitored for gout therapy; diuretics elevate levels",
    },
    {
        "lab_patterns": ["triglyceride", "cholesterol", "ldl", "hdl"],
        "atc_prefixes": ["C10A"],  # Statins / lipid-lowering
        "rationale": "Lipid panel is the primary efficacy marker for statins",
    },
    {
        "lab_patterns": ["tsh", "thyroid", "t3", "t4", "thyroxine"],
        "atc_prefixes": ["H03"],  # Thyroid drugs
        "rationale": "Thyroid function tests guide thyroid medication dosing",
    },
    # ── Blood / Hematology ──
    {
        "lab_patterns": ["wbc", "white blood cell", "leukocyte"],
        "atc_prefixes": ["L01", "L04"],  # Chemotherapy, Immunosuppressants
        "rationale": "Myelosuppressive drugs require WBC monitoring",
    },
    {
        "lab_patterns": ["hemoglobin", "hematocrit", "rbc", "red blood cell"],
        "atc_prefixes": ["B03"],  # Iron supplements / anti-anemics
        "rationale": "Anemia markers guide iron and EPO therapy",
    },
    {
        "lab_patterns": ["platelet"],
        "atc_prefixes": ["B01A", "L01"],  # Anticoagulants, Chemotherapy
        "rationale": "Thrombocytopenia risk from anticoagulants and chemotherapy",
    },
    {
        "lab_patterns": ["neutrophil"],
        "atc_prefixes": ["L01", "L04", "L03"],  # Chemo, Immunosuppressants, CSFs
        "rationale": "Neutropenia monitoring for myelosuppressive therapy",
    },
]


def match_lab_name(lab_name: str, patterns: list[str]) -> bool:
    """Check if a lab name matches any of the clinical patterns."""
    name_lower = lab_name.lower()
    return any(p.lower() in name_lower for p in patterns)


def build_prior(
    cohort_path: str = "data/processed/cohort_mimic3.pkl",
    lab_pkl_path: str | None = None,
    out_path: str = "data/processed/lab_drug_prior.npy",
):
    """Build the lab-drug prior matrix.
    
    If lab_pkl_path is provided, reads lab_itemids and lab_names from it
    to build a prior aligned with the actual model's lab dimensions.
    Otherwise, falls back to cohort-only mode with a warning.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load cohort for drug vocabulary
    with open(cohort_path, "rb") as f:
        cohort = pickle.load(f)
    
    drug_vocab = cohort["drug_vocab"]
    drug_voc = {idx: code for code, idx in drug_vocab.items()}
    num_drugs = cohort["num_drugs"]
    
    # Load lab PKL for lab names and ITEMIDs
    lab_names = []
    lab_itemids = []
    
    if lab_pkl_path is not None:
        lab_pkl = Path(lab_pkl_path)
    else:
        # Auto-discover: prefer 200labs, then any available
        processed = Path("data/processed")
        candidates = [
            processed / "lab_vectors_200labs.pkl",
            processed / "lab_vectors_150labs.pkl",
            processed / "lab_vectors_300labs.pkl",
        ]
        # Also glob for any lab PKL
        if processed.exists():
            candidates.extend(sorted(processed.glob("lab_vectors_*labs.pkl")))
        lab_pkl = None
        for c in candidates:
            if c.exists():
                lab_pkl = c
                break
    
    if lab_pkl is not None and lab_pkl.exists():
        with open(lab_pkl, "rb") as f:
            lab_data = pickle.load(f)
        lab_names = lab_data.get("lab_names", [])
        lab_itemids = lab_data.get("lab_itemids", [])
        num_labs = len(lab_names) if lab_names else len(lab_itemids)
        print(f"Loaded lab metadata from {lab_pkl}: {num_labs} labs")
    else:
        # Fallback: import from lab_ranges (18 labs)
        print("WARNING: No lab PKL found. Falling back to 18-lab lab_ranges.py.")
        print("  For production (200 labs), run with --lab_pkl data/processed/lab_vectors_200labs.pkl")
        try:
            from .lab_ranges import LAB_ITEMIDS, LAB_RANGES
        except ImportError:
            import sys
            sys.path.append(str(Path(__file__).resolve().parent))
            from lab_ranges import LAB_ITEMIDS, LAB_RANGES
        lab_itemids = LAB_ITEMIDS
        lab_names = [LAB_RANGES[iid][0] for iid in LAB_ITEMIDS]
        num_labs = len(LAB_ITEMIDS)
    
    if not lab_names:
        # Generate placeholder names from ITEMIDs
        lab_names = [f"Lab_{iid}" for iid in lab_itemids]
    
    print(f"Building prior matrix: ({num_labs} labs) x ({num_drugs} drugs)")
    prior = np.zeros((num_labs, num_drugs), dtype=np.float32)
    
    # Match each lab to clinical relationships by name pattern
    lab_matches = {}  # lab_idx -> list of matched ATC prefixes
    for lab_idx, name in enumerate(lab_names):
        matched_prefixes = []
        for rel in CLINICAL_RELATIONSHIPS:
            if match_lab_name(name, rel["lab_patterns"]):
                matched_prefixes.extend(rel["atc_prefixes"])
        if matched_prefixes:
            lab_matches[lab_idx] = list(set(matched_prefixes))
    
    # Fill the prior matrix
    for drug_idx, atc_code in drug_voc.items():
        for lab_idx, prefixes in lab_matches.items():
            if any(atc_code.startswith(p) for p in prefixes):
                prior[lab_idx, drug_idx] = 1.0
    
    # Diagnostic: per-lab coverage
    lab_coverage = prior.sum(axis=1)
    matched_count = (lab_coverage > 0).sum()
    print(f"\nLab-Drug Prior Coverage ({num_labs} labs):")
    print(f"  Labs with ≥1 drug link: {matched_count}/{num_labs} ({100*matched_count/max(num_labs,1):.1f}%)")
    print(f"  Total nonzero entries:  {int(prior.sum())}")
    print(f"\n  Top 20 labs by drug linkage:")
    sorted_labs = np.argsort(-lab_coverage)
    for rank, lab_idx in enumerate(sorted_labs[:20]):
        count = int(lab_coverage[lab_idx])
        name = lab_names[lab_idx] if lab_idx < len(lab_names) else f"Lab_{lab_idx}"
        print(f"    [{lab_idx:3d}] {name:30s}: {count:3d} drugs linked")
    
    unmatched = [lab_names[i] for i in range(num_labs) if lab_coverage[i] == 0]
    if unmatched:
        print(f"\n  Labs with NO drug links ({len(unmatched)}):")
        for name in unmatched[:10]:
            print(f"    - {name}")
        if len(unmatched) > 10:
            print(f"    ... and {len(unmatched) - 10} more")
    
    np.save(out_path, prior)
    print(f"\nSaved prior matrix to {out_path}: shape={prior.shape}, nonzero={int(prior.sum())}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build lab-drug prior matrix")
    parser.add_argument("--cohort", type=str, default="data/processed/cohort_mimic3.pkl")
    parser.add_argument("--lab_pkl", type=str, default=None,
                        help="Path to lab vectors PKL (e.g., data/processed/lab_vectors_200labs.pkl)")
    parser.add_argument("--output", type=str, default="data/processed/lab_drug_prior.npy")
    args = parser.parse_args()
    
    build_prior(
        cohort_path=args.cohort,
        lab_pkl_path=args.lab_pkl,
        out_path=args.output,
    )
