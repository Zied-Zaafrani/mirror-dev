"""
Quick comparison of MIMIC-III vs MIMIC-IV to surface structural differences
before committing to a full preprocessing run.

Checks:
  1. File existence and sizes for all expected tables
  2. Top-200 ATC-3 drug overlap between MIMIC-III and MIMIC-IV prescriptions
  3. Vocabulary size estimates (unique ICD codes, procedures)
  4. Note coverage (discharge summary count vs unique admissions)
  5. Lab coverage (top-200 labs ITEMID overlap)

Usage:
  python compare_mimic_datasets.py \\
    --mimic3_dir  DATASETS/mimic-iii-clinical-database-1.4 \\
    --mimic4_dir  DATASETS/mimic-iv-3.1 \\
    --notes4_dir  DATASETS/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note \\
    --external_dir data/external \\
    --sample_rows  200000
"""

import argparse
import ast
import pickle
from pathlib import Path

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_mb(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    mb = path.stat().st_size / 1_048_576
    return f"{mb:.0f} MB"


def _load_ndc2rxcui(path: Path) -> dict:
    content = path.read_text(encoding="utf-8", errors="replace").strip()
    if "u'" in content or 'u"' in content:
        if not content.startswith("{"):
            content = "{" + content + "}"
        try:
            raw = ast.literal_eval(content)
            return {k: str(v) for k, v in raw.items() if isinstance(k, str) and k != "idx"}
        except Exception:
            pass
    result = {}
    for line in content.splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            result[parts[0]] = parts[1]
    return result


def _get_atc3(external_dir: Path) -> dict[str, str]:
    """Build NDC -> ATC-3 mapping."""
    ndc2rxcui_path = external_dir / "ndc2RXCUI.txt"
    rxcui2atc4_path = external_dir / "RXCUI2atc4.csv"
    if not ndc2rxcui_path.exists() or not rxcui2atc4_path.exists():
        print("  WARNING: NDC mapping files not found in external_dir")
        return {}
    ndc2rxcui = _load_ndc2rxcui(ndc2rxcui_path)
    rxcui2atc4 = pd.read_csv(rxcui2atc4_path, dtype=str)
    cols = [c.upper().strip() for c in rxcui2atc4.columns]
    rxcui2atc4.columns = cols
    rxcui_col = next((c for c in cols if "RXCUI" in c), cols[-2])
    atc_col = next((c for c in cols if "ATC" in c), cols[-1])
    rxcui2atc4_dict = dict(zip(rxcui2atc4[rxcui_col].dropna(), rxcui2atc4[atc_col].dropna()))
    ndc2atc3 = {}
    for ndc, rxcui in ndc2rxcui.items():
        if rxcui in rxcui2atc4_dict:
            ndc2atc3[ndc] = rxcui2atc4_dict[rxcui][:4]
    return ndc2atc3


def _smiles_atc3_set(external_dir: Path) -> set[str]:
    """Return the set of ATC-3 codes that have SMILES (Carmen 131-drug list)."""
    smiles_path = external_dir / "idx2SMILES.pkl"
    if not smiles_path.exists():
        return set()
    with open(smiles_path, "rb") as f:
        idx2smiles = pickle.load(f)
    return set(idx2smiles.keys()) if isinstance(idx2smiles, dict) else set()


# ---------------------------------------------------------------------------
# Section 1: File check
# ---------------------------------------------------------------------------

def check_files(mimic3_dir: Path, mimic4_dir: Path, notes4_dir: Path):
    print("\n" + "="*60)
    print("SECTION 1: FILE EXISTENCE & SIZES")
    print("="*60)

    mimic3_files = {
        "ADMISSIONS":     "ADMISSIONS.csv.gz",
        "PRESCRIPTIONS":  "PRESCRIPTIONS.csv.gz",
        "DIAGNOSES_ICD":  "DIAGNOSES_ICD.csv.gz",
        "PROCEDURES_ICD": "PROCEDURES_ICD.csv.gz",
        "LABEVENTS":      "LABEVENTS.csv.gz",
        "NOTEEVENTS":     "NOTEEVENTS.csv.gz",
        "D_ICD_DIAGNOSES":"D_ICD_DIAGNOSES.csv.gz",
        "D_LABITEMS":     "D_LABITEMS.csv.gz",
    }
    mimic4_files = {
        "admissions":     "hosp/admissions.csv.gz",
        "prescriptions":  "hosp/prescriptions.csv.gz",
        "diagnoses_icd":  "hosp/diagnoses_icd.csv.gz",
        "procedures_icd": "hosp/procedures_icd.csv.gz",
        "labevents":      "hosp/labevents.csv.gz",
        "d_icd_diagnoses":"hosp/d_icd_diagnoses.csv.gz",
        "d_labitems":     "hosp/d_labitems.csv.gz",
        "notes_discharge":"discharge.csv.gz",  # notes4_dir
    }

    print(f"\n  {'Table':<20} {'MIMIC-III':>12} {'MIMIC-IV':>12}  Status")
    print(f"  {'-'*20} {'-'*12} {'-'*12}  {'-'*10}")
    for key in mimic3_files:
        m3p = mimic3_dir / mimic3_files[key]
        m4key = key.lower()
        if m4key == "noteevents":
            m4p = notes4_dir / "discharge.csv.gz"
        elif m4key in mimic4_files:
            fname = mimic4_files[m4key]
            m4p = (mimic4_dir / fname) if "/" in fname else (mimic4_dir / "hosp" / fname)
        else:
            m4p = Path("N/A")
        m3 = _file_mb(m3p)
        m4 = _file_mb(m4p)
        status = "OK" if "MB" in m3 and "MB" in m4 else ("MIMIC-III missing" if "MISSING" in m3 else "MIMIC-IV missing")
        print(f"  {key:<20} {m3:>12} {m4:>12}  {status}")


# ---------------------------------------------------------------------------
# Section 2: Drug overlap
# ---------------------------------------------------------------------------

def check_drug_overlap(
    mimic3_dir: Path, mimic4_dir: Path, external_dir: Path,
    sample_rows: int, smiles_atc3: set[str]
):
    print("\n" + "="*60)
    print("SECTION 2: TOP-200 DRUG OVERLAP (ATC-3)")
    print("="*60)

    ndc2atc3 = _get_atc3(external_dir)
    if not ndc2atc3:
        print("  SKIP: NDC->ATC3 mapping unavailable")
        return

    def _top_atc3_from_prescriptions(path: Path, ndc_col: str, top_k: int = 200) -> pd.Series:
        print(f"  Loading prescriptions from {path.name} (nrows={sample_rows:,}) ...")
        df = pd.read_csv(path, usecols=[ndc_col], compression="gzip",
                         nrows=sample_rows, dtype=str)
        df.columns = ["NDC"]
        df["NDC"] = df["NDC"].str.strip()
        df = df[~df["NDC"].isin(["0", "", "nan"])]
        df["ATC3"] = df["NDC"].map(ndc2atc3)
        df = df.dropna(subset=["ATC3"])
        counts = df["ATC3"].value_counts()
        print(f"    {len(df):,} mapped rows -> {len(counts):,} unique ATC-3 drugs")
        return counts.head(top_k)

    m3_top = _top_atc3_from_prescriptions(
        mimic3_dir / "PRESCRIPTIONS.csv.gz", "NDC", top_k=200
    )
    m4_top = _top_atc3_from_prescriptions(
        mimic4_dir / "hosp" / "prescriptions.csv.gz", "ndc", top_k=200
    )

    m3_set = set(m3_top.index)
    m4_set = set(m4_top.index)
    overlap = m3_set & m4_set
    only_m3 = m3_set - m4_set
    only_m4 = m4_set - m3_set

    print(f"\n  MIMIC-III top-200 drugs: {len(m3_set)}")
    print(f"  MIMIC-IV  top-200 drugs: {len(m4_set)}")
    print(f"  Overlap:    {len(overlap)} drugs ({100*len(overlap)/200:.1f}%)")
    print(f"  Only MIMIC-III: {len(only_m3)} drugs")
    print(f"  Only MIMIC-IV:  {len(only_m4)} drugs")

    # Check SMILES overlap
    if smiles_atc3:
        m3_smiles = smiles_atc3 & m3_set
        m4_smiles = smiles_atc3 & m4_set
        overlap_smiles = smiles_atc3 & overlap
        print(f"\n  Carmen SMILES list has {len(smiles_atc3)} drugs")
        print(f"    Covered by MIMIC-III top-200: {len(m3_smiles)} drugs")
        print(f"    Covered by MIMIC-IV  top-200: {len(m4_smiles)} drugs")
        print(f"    Covered by both:              {len(overlap_smiles)} drugs")
        if len(m4_smiles) == 131:
            print(f"  GOOD: All 131 Carmen drugs present in MIMIC-IV -> drug count will be 131")
        elif len(m4_smiles) >= 125:
            print(f"  OK: {len(m4_smiles)}/131 Carmen drugs in MIMIC-IV top-200 -> drug count ~{len(m4_smiles)}")
        else:
            print(f"  WARNING: Only {len(m4_smiles)}/131 Carmen drugs in MIMIC-IV top-200")
            missing = smiles_atc3 - m4_set
            print(f"    Missing from MIMIC-IV: {sorted(missing)[:10]}...")

    # Show top-10 in only MIMIC-IV (new drugs)
    if only_m4:
        print(f"\n  Top-10 drugs in MIMIC-IV but not in MIMIC-III top-200:")
        for atc3 in sorted(only_m4)[:10]:
            m4_rank = list(m4_top.index).index(atc3) + 1 if atc3 in m4_top.index else "?"
            print(f"    {atc3}  (MIMIC-IV rank #{m4_rank})")


# ---------------------------------------------------------------------------
# Section 3: ICD vocabulary size
# ---------------------------------------------------------------------------

def check_vocabulary_sizes(mimic3_dir: Path, mimic4_dir: Path, sample_rows: int):
    print("\n" + "="*60)
    print("SECTION 3: VOCABULARY SIZE ESTIMATES")
    print("="*60)

    # MIMIC-III diagnoses
    try:
        m3_diag = pd.read_csv(mimic3_dir / "DIAGNOSES_ICD.csv.gz",
                               usecols=["ICD9_CODE"], compression="gzip",
                               nrows=sample_rows, dtype=str)
        m3_diag_count = m3_diag["ICD9_CODE"].nunique()
        print(f"  MIMIC-III diagnoses unique ICD-9 codes (sample): {m3_diag_count:,}")
    except Exception as e:
        print(f"  MIMIC-III diagnoses: ERROR ({e})")

    # MIMIC-IV diagnoses (ICD-9 only)
    try:
        m4_diag = pd.read_csv(mimic4_dir / "hosp" / "diagnoses_icd.csv.gz",
                               usecols=["icd_code", "icd_version"], compression="gzip",
                               nrows=sample_rows, dtype={"icd_code": str, "icd_version": int})
        m4_diag_9  = m4_diag[m4_diag["icd_version"] == 9]["icd_code"].nunique()
        m4_diag_10 = m4_diag[m4_diag["icd_version"] == 10]["icd_code"].nunique()
        m4_hadm    = pd.read_csv(mimic4_dir / "hosp" / "diagnoses_icd.csv.gz",
                                  usecols=["hadm_id", "icd_version"], compression="gzip",
                                  nrows=sample_rows, dtype={"icd_version": int})
        n_hadm_9  = m4_hadm[m4_hadm["icd_version"] == 9]["hadm_id"].nunique()
        n_hadm_10 = m4_hadm[m4_hadm["icd_version"] == 10]["hadm_id"].nunique()
        print(f"  MIMIC-IV diagnoses (sample {sample_rows:,} rows):")
        print(f"    ICD-9  codes: {m4_diag_9:,} unique | admissions with ICD-9: {n_hadm_9:,}")
        print(f"    ICD-10 codes: {m4_diag_10:,} unique | admissions with ICD-10: {n_hadm_10:,}")
        if n_hadm_9 > 0 and n_hadm_10 > 0:
            print(f"    MIMIC-IV has MIXED ICD-9/10 -> --icd9_only gives ~{n_hadm_9:,}; full gives ~{n_hadm_9+n_hadm_10:,} admissions")
    except Exception as e:
        print(f"  MIMIC-IV diagnoses: ERROR ({e})")


# ---------------------------------------------------------------------------
# Section 4: Note coverage
# ---------------------------------------------------------------------------

def check_note_coverage(mimic3_dir: Path, notes4_dir: Path, sample_rows: int):
    print("\n" + "="*60)
    print("SECTION 4: NOTE COVERAGE")
    print("="*60)

    # MIMIC-III
    try:
        m3_notes = pd.read_csv(
            mimic3_dir / "NOTEEVENTS.csv.gz",
            usecols=["HADM_ID", "CATEGORY"],
            compression="gzip", nrows=sample_rows, dtype=str
        )
        disc = m3_notes[m3_notes["CATEGORY"] == "Discharge summary"]
        print(f"  MIMIC-III NOTEEVENTS sample ({sample_rows:,} rows):")
        print(f"    Total rows: {len(m3_notes):,}  | Discharge summaries: {len(disc):,}")
        print(f"    Unique HADMs with discharge notes: {disc['HADM_ID'].nunique():,}")
    except Exception as e:
        print(f"  MIMIC-III notes: ERROR ({e})")

    # MIMIC-IV
    notes4_path = notes4_dir / "discharge.csv.gz"
    if not notes4_path.exists():
        print(f"  MIMIC-IV notes: NOT FOUND at {notes4_path}")
    else:
        try:
            m4_notes = pd.read_csv(
                notes4_path,
                usecols=["hadm_id", "text"],
                compression="gzip", nrows=sample_rows, dtype=str
            )
            print(f"  MIMIC-IV discharge.csv.gz sample ({sample_rows:,} rows):")
            print(f"    Rows: {len(m4_notes):,} | Unique HADMs: {m4_notes['hadm_id'].nunique():,}")
            # Full count (fast: just count lines)
            import gzip
            with gzip.open(notes4_path, "rb") as f:
                total_lines = sum(1 for _ in f) - 1  # subtract header
            print(f"    Full file: ~{total_lines:,} rows")
        except Exception as e:
            print(f"  MIMIC-IV notes: ERROR ({e})")


# ---------------------------------------------------------------------------
# Section 5: Lab ITEMID overlap
# ---------------------------------------------------------------------------

def check_lab_overlap(mimic3_dir: Path, mimic4_dir: Path, sample_rows: int):
    print("\n" + "="*60)
    print("SECTION 5: LAB ITEMID OVERLAP (TOP-200 labs)")
    print("="*60)

    TARGET_IDS = [51221, 51301, 51265, 51222, 51249, 51279, 51250, 51248, 51277, 50971,
                  50983, 50902, 50882, 50868, 51006, 50912, 50931, 50960]
    print(f"  Checking presence of {len(TARGET_IDS)} champion ITEMID labs in each dataset ...")

    try:
        m3_labs = pd.read_csv(
            mimic3_dir / "LABEVENTS.csv.gz",
            usecols=["ITEMID", "HADM_ID"], compression="gzip",
            nrows=sample_rows, dtype={"ITEMID": int}
        )
        m3_present = set(m3_labs["ITEMID"].unique()) & set(TARGET_IDS)
        print(f"  MIMIC-III (sample): {len(m3_present)}/{len(TARGET_IDS)} champion ITEMIDs found")
    except Exception as e:
        print(f"  MIMIC-III labs: ERROR ({e})")

    try:
        m4_labs = pd.read_csv(
            mimic4_dir / "hosp" / "labevents.csv.gz",
            usecols=["itemid", "hadm_id"], compression="gzip",
            nrows=sample_rows, dtype={"itemid": int}
        )
        m4_present = set(m4_labs["itemid"].unique()) & set(TARGET_IDS)
        missing = set(TARGET_IDS) - m4_present
        print(f"  MIMIC-IV  (sample): {len(m4_present)}/{len(TARGET_IDS)} champion ITEMIDs found")
        if missing:
            print(f"    Missing in MIMIC-IV sample (may appear in full file): {sorted(missing)}")
        if len(m4_present) == len(TARGET_IDS):
            print(f"  GOOD: All champion lab ITEMIDs present in MIMIC-IV")
    except Exception as e:
        print(f"  MIMIC-IV labs: ERROR ({e})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare MIMIC-III vs MIMIC-IV datasets")
    parser.add_argument("--mimic3_dir",    type=str,
                        default=r"C:\Users\Zied Zaafrani\Desktop\Zied Project\master\MIRROR\datasets\mimic-iii-clinical-database-1.4")
    parser.add_argument("--mimic4_dir",    type=str,
                        default=r"C:\Users\Zied Zaafrani\Desktop\Zied Project\master\MIRROR\datasets\mimic-iv-3.1")
    parser.add_argument("--notes4_dir",    type=str,
                        default=r"C:\Users\Zied Zaafrani\Desktop\Zied Project\master\MIRROR\datasets\mimic-iv-note-deidentified-free-text-clinical-notes-2.2\note")
    parser.add_argument("--external_dir",  type=str,
                        default=r"C:\Users\Zied Zaafrani\Desktop\Zied Project\master\MIRROR\data\external")
    parser.add_argument("--sample_rows",   type=int, default=500_000,
                        help="Max rows to sample from large CSV files (default: 500K)")
    args = parser.parse_args()

    mimic3_dir   = Path(args.mimic3_dir)
    mimic4_dir   = Path(args.mimic4_dir)
    notes4_dir   = Path(args.notes4_dir)
    external_dir = Path(args.external_dir)

    print("\n" + "="*60)
    print("MIRROR: MIMIC-III vs MIMIC-IV Dataset Comparison")
    print("="*60)
    print(f"  MIMIC-III: {mimic3_dir}")
    print(f"  MIMIC-IV:  {mimic4_dir}")
    print(f"  Notes IV:  {notes4_dir}")
    print(f"  Sample:    {args.sample_rows:,} rows per file")

    smiles_atc3 = _smiles_atc3_set(external_dir)
    if smiles_atc3:
        print(f"  Carmen SMILES list: {len(smiles_atc3)} drugs loaded")

    check_files(mimic3_dir, mimic4_dir, notes4_dir)
    check_drug_overlap(mimic3_dir, mimic4_dir, external_dir, args.sample_rows, smiles_atc3)
    check_vocabulary_sizes(mimic3_dir, mimic4_dir, args.sample_rows)
    check_note_coverage(mimic3_dir, notes4_dir, args.sample_rows)
    check_lab_overlap(mimic3_dir, mimic4_dir, args.sample_rows)

    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
