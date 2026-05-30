"""
Preprocess MIMIC-IV v3.1 for drug recommendation following the Carmen/SafeDrug pipeline.

Differences from MIMIC-III:
  - Tables in hosp/ module with lowercase names
  - Mixed ICD-9 + ICD-10 codes (icd_version column)
  - Same NDC -> ATC mapping; our LLM embeddings handle mixed ICD naturally

Produces same output format as preprocess_mimic3.py for compatibility.

Usage:
  python preprocess_mimic4.py --mimic_dir DATASETS/mimic-iv-3.1 \
                              --external_dir data/external \
                              --output_dir data/processed
"""

import argparse
import ast
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Progress helper
# ---------------------------------------------------------------------------

def _progress(msg: str, current: int, total: int, t0: float | None = None):
    """Print an in-place progress line."""
    pct = current / total * 100 if total > 0 else 0
    bar_len = 30
    filled = int(bar_len * current / total) if total > 0 else 0
    bar = '=' * filled + '-' * (bar_len - filled)
    elapsed = f" ({time.time() - t0:.0f}s)" if t0 else ""
    print(f"\r  [{bar}] {pct:5.1f}% {msg}{elapsed}", end="", flush=True)
    if current >= total:
        print()  # newline when done


def _load_pickle_robust(path: Path) -> dict:
    """Load a pickle file with multiple fallback strategies."""
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception as e1:
            f.seek(0)
            try:
                return pickle.load(f, encoding="latin1")
            except Exception:
                pass
            f.seek(0)
            try:
                import dill
                return dill.load(f)
            except ImportError:
                print(f"  WARNING: {path.name} failed to load ({e1}). Install dill: pip install dill")
                return {}
            except Exception as e3:
                print(f"  WARNING: {path.name} failed all load strategies: {e1} / {e3}")
                return {}


# ---------------------------------------------------------------------------
# Step 1: Load MIMIC-IV tables
# ---------------------------------------------------------------------------

def load_prescriptions(mimic_dir: Path) -> pd.DataFrame:
    path = mimic_dir / "hosp" / "prescriptions.csv.gz"
    print(f"Loading {path} ...")
    df = pd.read_csv(path, usecols=["subject_id", "hadm_id", "ndc"],
                      compression="gzip", dtype={"ndc": str})
    df.columns = ["SUBJECT_ID", "HADM_ID", "NDC"]
    df = df.dropna(subset=["HADM_ID", "NDC"])
    df["HADM_ID"] = df["HADM_ID"].astype(int)
    df["NDC"] = df["NDC"].astype(str).str.strip()
    df = df[~df["NDC"].isin(["0", "", "nan"])]
    print(f"  {len(df):,} prescription rows, {df['HADM_ID'].nunique():,} admissions")
    return df


def load_diagnoses(mimic_dir: Path) -> pd.DataFrame:
    path = mimic_dir / "hosp" / "diagnoses_icd.csv.gz"
    print(f"Loading {path} ...")
    df = pd.read_csv(path, usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
                      compression="gzip", dtype={"icd_code": str, "icd_version": int})
    df.columns = ["SUBJECT_ID", "HADM_ID", "ICD_CODE", "ICD_VERSION"]
    df = df.dropna(subset=["HADM_ID", "ICD_CODE"])
    df["HADM_ID"] = df["HADM_ID"].astype(int)
    # Prefix with version for uniqueness: "9_4019" vs "10_I10"
    df["ICD_VERSIONED"] = df["ICD_VERSION"].astype(str) + "_" + df["ICD_CODE"]
    n9 = (df["ICD_VERSION"] == 9).sum()
    n10 = (df["ICD_VERSION"] == 10).sum()
    print(f"  {len(df):,} diagnosis rows ({n9:,} ICD-9, {n10:,} ICD-10), "
          f"{df['HADM_ID'].nunique():,} admissions")
    return df


def load_procedures(mimic_dir: Path) -> pd.DataFrame:
    path = mimic_dir / "hosp" / "procedures_icd.csv.gz"
    print(f"Loading {path} ...")
    df = pd.read_csv(path, usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
                      compression="gzip", dtype={"icd_code": str, "icd_version": int})
    df.columns = ["SUBJECT_ID", "HADM_ID", "ICD_CODE", "ICD_VERSION"]
    df = df.dropna(subset=["HADM_ID", "ICD_CODE"])
    df["HADM_ID"] = df["HADM_ID"].astype(int)
    df["ICD_VERSIONED"] = df["ICD_VERSION"].astype(str) + "_" + df["ICD_CODE"]
    print(f"  {len(df):,} procedure rows, {df['HADM_ID'].nunique():,} admissions")
    return df


# ---------------------------------------------------------------------------
# Step 2: NDC -> ATC-3 mapping (identical to MIMIC-III)
# ---------------------------------------------------------------------------

def _load_ndc2rxcui(path: Path) -> dict:
    """Parse ndc2RXCUI.txt — handles both tab-separated and Python 2 dict format."""
    content = path.read_text(encoding="utf-8", errors="replace").strip()
    if "u'" in content or 'u"' in content:
        if not content.startswith("{"):
            content = "{" + content + "}"
        try:
            raw = ast.literal_eval(content)
            return {k: str(v) for k, v in raw.items()
                    if isinstance(k, str) and k != "idx" and str(v)}
        except Exception:
            pass
    result = {}
    for line in content.splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            result[parts[0]] = parts[1]
    return result


def build_ndc_to_atc3(external_dir: Path, chain_only: bool = False) -> dict[str, str]:
    """Build NDC -> ATC-3 mapping via RXCUI intermediate + direct CSV NDC.

    chain_only=True: use only ndc2RXCUI.txt chain (2,929 mappings) — matches
    Carmen/ARMR/VITA/HI-DR behaviour and is required for the SOTA-comparable cohort.
    chain_only=False (default): also mines the NDC column in RXCUI2atc4.csv for
    an additional ~154K mappings (used for the 30K all-hospital cohort).
    """
    # --- Source 1: ndc2RXCUI.txt -> RXCUI -> ATC4 chain ---
    ndc2rxcui_path = external_dir / "ndc2RXCUI.txt"
    print(f"Loading NDC->RXCUI from {ndc2rxcui_path.name} ...")
    ndc2rxcui = _load_ndc2rxcui(ndc2rxcui_path)
    print(f"  {len(ndc2rxcui):,} NDC->RXCUI mappings")

    # RXCUI -> ATC-4 (detect columns by name)
    rxcui2atc4_path = external_dir / "RXCUI2atc4.csv"
    print(f"Loading RXCUI->ATC4 from {rxcui2atc4_path.name} ...")
    rxcui2atc = pd.read_csv(rxcui2atc4_path, dtype=str)
    cols = [c.upper().strip() for c in rxcui2atc.columns]
    rxcui2atc.columns = cols
    # Find the RXCUI and ATC columns by name
    rxcui_col = next((c for c in cols if 'RXCUI' in c), None)
    atc_col = next((c for c in cols if 'ATC' in c), None)
    if rxcui_col is None or atc_col is None:
        rxcui_col, atc_col = cols[-2], cols[-1]
    rxcui2atc4_dict = dict(zip(
        rxcui2atc[rxcui_col].dropna(),
        rxcui2atc[atc_col].dropna(),
    ))
    print(f"  {len(rxcui2atc4_dict):,} RXCUI->ATC4 mappings")

    # Chain: ndc2RXCUI -> RXCUI2atc4 -> ATC3
    ndc2atc3 = {}
    for ndc, rxcui in ndc2rxcui.items():
        if rxcui in rxcui2atc4_dict:
            atc4 = rxcui2atc4_dict[rxcui]
            ndc2atc3[ndc] = atc4[:4]
    print(f"  Chain (ndc->rxcui->atc3): {len(ndc2atc3):,} mappings")

    # --- Source 2: direct NDC column in RXCUI2atc4.csv ---
    # Skipped when chain_only=True (SOTA-comparable cohort, matches Carmen/ARMR/VITA/HI-DR)
    ndc_col = next((c for c in cols if 'NDC' in c), None)
    if not chain_only and ndc_col and atc_col:
        direct_count = 0
        for _, row in rxcui2atc.iterrows():
            ndc_raw = str(row[ndc_col]).replace('-', '').strip()
            if len(ndc_raw) < 9 or not ndc_raw.isdigit():
                continue
            ndc_norm = ndc_raw.zfill(11)
            if ndc_norm not in ndc2atc3:
                atc4 = str(row[atc_col]).strip()
                if len(atc4) >= 4:
                    ndc2atc3[ndc_norm] = atc4[:4]
                    direct_count += 1
        print(f"  Direct CSV NDC->ATC3: +{direct_count:,} new mappings")
    elif chain_only:
        print(f"  Source 2 (direct CSV) skipped — chain_only mode (SOTA-comparable)")

    print(f"  Total NDC->ATC3: {len(ndc2atc3):,} mappings")
    return ndc2atc3


def map_prescriptions_to_atc3(prescriptions: pd.DataFrame, ndc2atc3: dict) -> pd.DataFrame:
    prescriptions = prescriptions.copy()
    prescriptions["ATC3"] = prescriptions["NDC"].map(ndc2atc3)
    prescriptions = prescriptions.dropna(subset=["ATC3"])
    prescriptions = prescriptions.drop_duplicates(subset=["HADM_ID", "ATC3"])
    print(f"  After ATC3 mapping: {len(prescriptions):,} (admission, drug) pairs")
    return prescriptions


def filter_drugs_by_smiles(prescriptions: pd.DataFrame, external_dir: Path,
                           top_k: int = 300) -> tuple[pd.DataFrame, list[str]]:
    atc3_counts = prescriptions["ATC3"].value_counts()
    top_atc3 = set(atc3_counts.head(top_k).index)
    prescriptions = prescriptions[prescriptions["ATC3"].isin(top_atc3)]
    print(f"  After top-{top_k} ATC3 filter: {prescriptions['ATC3'].nunique()} unique drugs")

    smiles_path = external_dir / "idx2SMILES.pkl"
    if smiles_path.exists():
        idx2smiles = _load_pickle_robust(smiles_path)
        smiles_atc3 = set(idx2smiles.keys()) if isinstance(idx2smiles, dict) else set()
        if smiles_atc3:
            prescriptions = prescriptions[prescriptions["ATC3"].isin(smiles_atc3)]

    final_drugs = sorted(prescriptions["ATC3"].unique())
    print(f"  After SMILES filter: {len(final_drugs)} drugs (expected ~131)")
    return prescriptions, final_drugs


# ---------------------------------------------------------------------------
# Steps 4-5: Diagnoses and procedures (use ICD_VERSIONED for uniqueness)
# ---------------------------------------------------------------------------

def filter_top_diagnoses(diagnoses: pd.DataFrame, top_k: int = 2000) -> tuple[pd.DataFrame, list[str]]:
    code_counts = diagnoses["ICD_VERSIONED"].value_counts()
    top_codes = set(code_counts.head(top_k).index)
    diagnoses = diagnoses[diagnoses["ICD_VERSIONED"].isin(top_codes)]
    final_diag = sorted(diagnoses["ICD_VERSIONED"].unique())
    print(f"  Top-{top_k} diagnoses: {len(final_diag)} unique codes")
    return diagnoses, final_diag


def get_procedures(procedures: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    final_proc = sorted(procedures["ICD_VERSIONED"].unique())
    print(f"  Procedures (unfiltered): {len(final_proc)} unique codes")
    return procedures, final_proc


# ---------------------------------------------------------------------------
# Steps 6-7: Build patient records
# ---------------------------------------------------------------------------

def build_patient_records(prescriptions, diagnoses, procedures,
                          drug_vocab, diag_vocab, proc_vocab):
    med_hadms = set(prescriptions["HADM_ID"])
    diag_hadms = set(diagnoses["HADM_ID"])
    proc_hadms = set(procedures["HADM_ID"])
    common_hadms = med_hadms & diag_hadms & proc_hadms
    print(f"  Admissions with all 3 code types: {len(common_hadms):,}")

    med_per_hadm = prescriptions.groupby("HADM_ID")["ATC3"].apply(set).to_dict()
    diag_per_hadm = diagnoses.groupby("HADM_ID")["ICD_VERSIONED"].apply(set).to_dict()
    proc_per_hadm = procedures.groupby("HADM_ID")["ICD_VERSIONED"].apply(set).to_dict()

    hadm_to_subject = dict(zip(prescriptions["HADM_ID"], prescriptions["SUBJECT_ID"]))

    patient_hadms = defaultdict(list)
    for hadm_id in common_hadms:
        subj = hadm_to_subject.get(hadm_id)
        if subj is not None:
            patient_hadms[subj].append(hadm_id)

    for subj in patient_hadms:
        patient_hadms[subj].sort()

    multi_visit = {s: v for s, v in patient_hadms.items() if len(v) >= 2}
    print(f"  Patients with >=2 visits: {len(multi_visit):,}")

    records = []
    patient_visits = {}
    for subject_id in sorted(multi_visit.keys()):
        hadms = multi_visit[subject_id]
        patient_visits[subject_id] = hadms
        patient_record = []
        for hadm_id in hadms:
            diag_idx = sorted([diag_vocab[c] for c in diag_per_hadm.get(hadm_id, set())
                              if c in diag_vocab])
            proc_idx = sorted([proc_vocab[c] for c in proc_per_hadm.get(hadm_id, set())
                              if c in proc_vocab])
            med_idx = sorted([drug_vocab[c] for c in med_per_hadm.get(hadm_id, set())
                             if c in drug_vocab])
            if diag_idx and proc_idx and med_idx:
                patient_record.append([diag_idx, proc_idx, med_idx, hadm_id])
        if len(patient_record) >= 2:
            records.append(patient_record)

    total_visits = sum(len(p) for p in records)
    print(f"  Final: {len(records):,} patients, {total_visits:,} visits")
    return records, patient_visits


# ---------------------------------------------------------------------------
# DDI, co-occurrence, split (same logic as MIMIC-III)
# ---------------------------------------------------------------------------

def _load_cid_to_atc3(external_dir: Path) -> dict[str, set[str]]:
    """Parse drug-atc.csv (CID,ATC1[,ATC2,...]) -> {CID: {ATC3, ...}}."""
    atc_path = external_dir / "drug-atc.csv"
    cid2atc3: dict[str, set[str]] = {}
    if not atc_path.exists():
        print(f"  WARNING: {atc_path} not found.")
        return cid2atc3
    with open(atc_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            cid = parts[0].strip()
            for atc in parts[1:]:
                atc = atc.strip()
                if len(atc) >= 4:
                    cid2atc3.setdefault(cid, set()).add(atc[:4])
    print(f"  drug-atc.csv: {len(cid2atc3):,} CIDs with ATC3 codes")
    return cid2atc3


def build_ddi_matrix(external_dir, drug_vocab, num_drugs, topk_severe=40):
    """Build binary DDI adjacency from TWOSIDES via CID->ATC3."""
    ddi_path = external_dir / "drug-DDI.csv"
    if not ddi_path.exists():
        print(f"  WARNING: {ddi_path} not found. Returning empty DDI matrix.")
        return np.zeros((num_drugs, num_drugs), dtype=np.float32)

    cid2atc3 = _load_cid_to_atc3(external_dir)

    print(f"Loading DDI data from {ddi_path.name} ...")
    ddi_df = pd.read_csv(ddi_path, dtype=str)
    cols = ddi_df.columns.tolist()
    print(f"  {len(ddi_df):,} DDI rows")

    if len(cols) >= 3 and topk_severe > 0:
        se_col = cols[2]
        se_counts = ddi_df[se_col].value_counts()
        top_se = set(se_counts.head(topk_severe).index)
        before = len(ddi_df)
        ddi_df = ddi_df[ddi_df[se_col].isin(top_se)]
        print(f"  Top-{topk_severe} side effects: {len(ddi_df):,} rows (from {before:,})")

    ddi_pairs = ddi_df.iloc[:, :2].drop_duplicates()
    ddi_pairs.columns = ["CID1", "CID2"]

    cid_rows = [(c, a) for c, atcs in cid2atc3.items() for a in atcs]
    if not cid_rows:
        print("  WARNING: No CID->ATC3 mappings. DDI matrix will be empty.")
        return np.zeros((num_drugs, num_drugs), dtype=np.float32)
    cid_atc = pd.DataFrame(cid_rows, columns=["CID", "ATC3"])

    merged = ddi_pairs.merge(
        cid_atc.rename(columns={"CID": "CID1", "ATC3": "A1"}), on="CID1"
    ).merge(
        cid_atc.rename(columns={"CID": "CID2", "ATC3": "A2"}), on="CID2"
    )
    drug_set = set(drug_vocab.keys())
    merged = merged[(merged["A1"].isin(drug_set)) & (merged["A2"].isin(drug_set))
                    & (merged["A1"] != merged["A2"])]

    ddi_matrix = np.zeros((num_drugs, num_drugs), dtype=np.float32)
    unique_pairs = set()
    for a1, a2 in zip(merged["A1"], merged["A2"]):
        if a1 in drug_vocab and a2 in drug_vocab:
            i, j = drug_vocab[a1], drug_vocab[a2]
            ddi_matrix[i, j] = 1
            ddi_matrix[j, i] = 1
            unique_pairs.add((min(a1, a2), max(a1, a2)))

    print(f"  DDI pairs in vocabulary: {len(unique_pairs):,}")
    return ddi_matrix


def build_cooccurrence_matrix(train_records, num_drugs):
    cooccur = np.zeros((num_drugs, num_drugs), dtype=np.float32)
    if num_drugs == 0:
        print("  WARNING: 0 drugs, returning empty co-occurrence matrix.")
        return cooccur
    t0 = time.time()
    for pi, patient in enumerate(train_records):
        if pi % 500 == 0:
            _progress(f"{pi}/{len(train_records)} patients", pi, len(train_records), t0)
        for visit in patient:
            meds = visit[2]
            for i in range(len(meds)):
                for j in range(i + 1, len(meds)):
                    cooccur[meds[i], meds[j]] += 1
                    cooccur[meds[j], meds[i]] += 1
    _progress("done", len(train_records), len(train_records), t0)
    if cooccur.size > 0 and cooccur.max() > 0:
        cooccur = cooccur / cooccur.max()
    print(f"  Co-occurrence: {(cooccur > 0).sum() // 2:,} non-zero pairs")
    return cooccur


def split_by_patient(records, train_ratio=4/6, val_ratio=1/6, seed=42):
    rng = np.random.RandomState(seed)
    n = len(records)
    indices = rng.permutation(n)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = [records[i] for i in sorted(indices[:train_end])]
    val = [records[i] for i in sorted(indices[train_end:val_end])]
    test = [records[i] for i in sorted(indices[val_end:])]
    print(f"  Split: train={len(train)}, val={len(val)}, test={len(test)} patients")
    return train, val, test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess MIMIC-IV v3.1 (Carmen/SafeDrug pipeline)")
    parser.add_argument("--mimic_dir", type=str, required=True,
                        help="Path to mimic-iv-3.1/")
    parser.add_argument("--external_dir", type=str, required=True,
                        help="Path to external mapping files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for processed pickle files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--icd9_only", action="store_true", default=True,
                        help="Keep only ICD-9 visit codes (default: True). Produces ~30K patients "
                             "(admissions with any ICD-9 codes pass). "
                             "Use --no-icd9_only for full ICD-9+10 cohort.")
    parser.add_argument("--no-icd9_only", dest="icd9_only", action="store_false")
    parser.add_argument("--icd9_strict", action="store_true", default=False,
                        help="ICU-only cohort: filter to ICU stays (icustays.csv) + "
                             "chain-only NDC mapping. Reproduces Carmen/ARMR/VITA/HI-DR's "
                             "9,036-patient MIMIC-IV benchmark. Tag: mimic4_sota.")
    args = parser.parse_args()

    mimic_dir = Path(args.mimic_dir)
    external_dir = Path(args.external_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_t0 = time.time()

    # Step 1: Load
    print("\n=== Step 1: Load MIMIC-IV tables ===")
    t0 = time.time()
    prescriptions = load_prescriptions(mimic_dir)
    diagnoses = load_diagnoses(mimic_dir)
    procedures = load_procedures(mimic_dir)
    print(f"  Step 1 done ({time.time() - t0:.1f}s)")

    # ICU / ICD filtering
    chain_only_ndc = False
    if args.icd9_strict:
        # Carmen-compatible: filter ALL tables to ICU stays only, keep ICD-9+10 as-is,
        # use chain-only NDC mapping. Reproduces 9,036-patient SOTA benchmark.
        print("\n=== Filtering to ICU stays only (--icd9_strict / SOTA-comparable cohort) ===")
        icu_path = mimic_dir / "icu" / "icustays.csv.gz"
        if not icu_path.exists():
            print(f"  FATAL: {icu_path} not found. Cannot build SOTA-comparable cohort.")
            sys.exit(1)
        icustays = pd.read_csv(icu_path, usecols=["subject_id", "hadm_id"],
                               compression="gzip")
        icu_hadms = set(icustays["hadm_id"].dropna().astype(int))
        n_presc_before = prescriptions["HADM_ID"].nunique()
        n_diag_before  = diagnoses["HADM_ID"].nunique()
        prescriptions = prescriptions[prescriptions["HADM_ID"].isin(icu_hadms)].copy()
        diagnoses     = diagnoses[diagnoses["HADM_ID"].isin(icu_hadms)].copy()
        procedures    = procedures[procedures["HADM_ID"].isin(icu_hadms)].copy()
        print(f"  ICU hadm_ids loaded: {len(icu_hadms):,}")
        print(f"  Prescriptions admissions: {n_presc_before:,} -> {prescriptions['HADM_ID'].nunique():,}")
        print(f"  Diagnoses admissions:     {n_diag_before:,} -> {diagnoses['HADM_ID'].nunique():,}")
        print(f"  No ICD version filter applied (Carmen keeps ICD-9+10)")
        print(f"  Expected ~9,036 patients — Carmen/ARMR/VITA/HI-DR benchmark")
        chain_only_ndc = True   # Carmen drops NDC column, only uses RXCUI chain
    elif args.icd9_only:
        print("\n=== Filtering to ICD-9 only (--icd9_only, code-level) ===")
        n_before_diag = len(diagnoses)
        n_before_proc = len(procedures)
        diagnoses = diagnoses[diagnoses["ICD_VERSION"] == 9].copy()
        procedures = procedures[procedures["ICD_VERSION"] == 9].copy()
        print(f"  Diagnoses: {n_before_diag:,} -> {len(diagnoses):,} (dropped {n_before_diag - len(diagnoses):,} ICD-10 codes)")
        print(f"  Procedures: {n_before_proc:,} -> {len(procedures):,} (dropped {n_before_proc - len(procedures):,} ICD-10 codes)")
        print(f"  This gives ~30K patients (admissions with any ICD-9 codes pass)")

    # Step 2: NDC -> ATC-3
    print("\n=== Step 2: NDC -> ATC-3 mapping ===")
    t0 = time.time()
    ndc2atc3 = build_ndc_to_atc3(external_dir, chain_only=chain_only_ndc)
    prescriptions = map_prescriptions_to_atc3(prescriptions, ndc2atc3)
    print(f"  Step 2 done ({time.time() - t0:.1f}s)")

    # Step 3: Filter drugs
    print("\n=== Step 3: Filter drugs ===")
    t0 = time.time()
    prescriptions, final_drugs = filter_drugs_by_smiles(prescriptions, external_dir)
    if len(final_drugs) == 0:
        print("  FATAL: 0 drugs after filtering. Check NDC->RXCUI->ATC mapping chain.")
        sys.exit(1)
    # BUG-D: Validate drug count matches Carmen 131-drug MIMIC-III baseline.
    # MIMIC-IV prescribing patterns are similar (same hospital, BIDMC) so 131 is expected.
    # A significantly different count means the model's drug embedding matrix will be
    # a different size — not a crash (model auto-sizes), but worth flagging explicitly.
    if len(final_drugs) != 131:
        print(f"  WARNING: Drug count = {len(final_drugs)} (expected 131 for Carmen SMILES variant).")
        print(f"    This is OK — the model will auto-size to {len(final_drugs)} drugs.")
        print(f"    However, results will NOT be directly comparable to MIMIC-III baselines")
        print(f"    unless the drug set is the same. Proceeding.")
    else:
        print(f"  Drug count = 131 — matches MIMIC-III Carmen baseline. Good.")
    print(f"  Step 3 done ({time.time() - t0:.1f}s)")

    # Step 4-5: Diagnoses and procedures
    print("\n=== Step 4: Filter diagnoses ===")
    diagnoses, final_diag = filter_top_diagnoses(diagnoses)
    print("\n=== Step 5: Get procedures ===")
    procedures, final_proc = get_procedures(procedures)

    drug_vocab = {code: idx for idx, code in enumerate(final_drugs)}
    diag_vocab = {code: idx for idx, code in enumerate(final_diag)}
    proc_vocab = {code: idx for idx, code in enumerate(final_proc)}

    voc = {
        "diag_voc": {"idx2word": {v: k for k, v in diag_vocab.items()}, "word2idx": diag_vocab},
        "med_voc":  {"idx2word": {v: k for k, v in drug_vocab.items()}, "word2idx": drug_vocab},
        "pro_voc":  {"idx2word": {v: k for k, v in proc_vocab.items()}, "word2idx": proc_vocab},
    }

    # Step 6-7: Build records
    print("\n=== Step 6-7: Build patient records ===")
    t0 = time.time()
    records, patient_visits = build_patient_records(
        prescriptions, diagnoses, procedures,
        drug_vocab, diag_vocab, proc_vocab,
    )
    if len(records) == 0:
        print("  FATAL: 0 patients with >=2 visits. Check mapping chain above.")
        sys.exit(1)
    print(f"  Step 6-7 done ({time.time() - t0:.1f}s)")

    # Split
    print("\n=== Split (4:1:1 by patient) ===")
    train, val, test = split_by_patient(records, seed=args.seed)

    # Step 8: DDI
    print("\n=== Step 8: DDI matrix ===")
    t0 = time.time()
    num_drugs = len(final_drugs)
    ddi_matrix = build_ddi_matrix(external_dir, drug_vocab, num_drugs)
    print(f"  Step 8 done ({time.time() - t0:.1f}s)")

    # Step 9: Co-occurrence
    print("\n=== Step 9: Co-occurrence matrix ===")
    t0 = time.time()
    ehr_adj = build_cooccurrence_matrix(train, num_drugs)
    print(f"  Step 9 done ({time.time() - t0:.1f}s)")

    # Build hadm_id list with split labels
    all_hadm_ids = []
    all_splits = []
    for split_name, split_records in [("train", train), ("val", val), ("test", test)]:
        for patient in split_records:
            for visit in patient:
                all_hadm_ids.append(visit[3])
                all_splits.append(split_name)

    # Save
    print("\n=== Saving outputs ===")
    # Tag reflects filtering mode
    if args.icd9_strict:
        tag = "mimic4_sota"
        cohort_label = "ICU-only, chain-NDC (~9K, Carmen/ARMR/VITA/HI-DR benchmark)"
    elif args.icd9_only:
        tag = "mimic4"
        cohort_label = "ICD-9 code-level (~30K)"
    else:
        tag = "mimic4_full"
        cohort_label = "Full ICD-9+10"

    # MIRROR format: keep hadm_id in records [diag, proc, med, hadm_id]
    with open(output_dir / f"records_{tag}.pkl", "wb") as f:
        pickle.dump(records, f)
    with open(output_dir / f"voc_{tag}.pkl", "wb") as f:
        pickle.dump(voc, f)
    with open(output_dir / f"ddi_A_{tag}.pkl", "wb") as f:
        pickle.dump(ddi_matrix, f)
    with open(output_dir / f"ehr_adj_{tag}.pkl", "wb") as f:
        pickle.dump(ehr_adj, f)

    cohort = {
        "hadm_ids": np.array(all_hadm_ids),
        "split": all_splits,
        "patient_visits": patient_visits,
        "drug_vocab": drug_vocab,
        "diag_vocab": diag_vocab,
        "proc_vocab": proc_vocab,
        "num_drugs": num_drugs,
        "num_diag": len(final_diag),
        "num_proc": len(final_proc),
        "icd9_only": args.icd9_only,
        "icd9_strict": getattr(args, "icd9_strict", False),
    }
    with open(output_dir / f"cohort_{tag}.pkl", "wb") as f:
        pickle.dump(cohort, f)

    print(f"\nSaved to {output_dir}/ ({cohort_label}):")
    print(f"  records_{tag}.pkl  ({len(records):,} patients)")
    print(f"  voc_{tag}.pkl      (diag={len(final_diag)}, proc={len(final_proc)}, med={num_drugs})")
    print(f"  ddi_A_{tag}.pkl    ({num_drugs}x{num_drugs})")
    print(f"  ehr_adj_{tag}.pkl  ({num_drugs}x{num_drugs})")
    print(f"  cohort_{tag}.pkl   ({len(all_hadm_ids):,} admissions)")

    icd_label = "ICD-9 only" if args.icd9_only else "mixed ICD-9/10"
    visit_counts = [len(p) for p in records]
    med_counts = [len(v[2]) for p in records for v in p]
    print(f"\n=== Summary ({cohort_label}) ===")
    print(f"  Patients: {len(records):,}")
    print(f"  Visits: {sum(visit_counts):,}")
    print(f"  Avg visits/patient: {np.mean(visit_counts):.2f}")
    print(f"  Drugs: {num_drugs}")
    print(f"  Diagnoses: {len(final_diag)} ({icd_label})")
    print(f"  Procedures: {len(final_proc)}")
    print(f"  Avg meds/visit: {np.mean(med_counts):.1f}")
    print(f"  DDI pairs: {int(ddi_matrix.sum() / 2)}")
    print(f"\n  Total preprocessing time: {time.time() - pipeline_t0:.1f}s")


if __name__ == "__main__":
    main()
