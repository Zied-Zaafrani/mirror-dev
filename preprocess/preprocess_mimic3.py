"""
Preprocess MIMIC-III for drug recommendation following the Carmen/SafeDrug pipeline.

Produces the standard output consumed by all downstream baselines:
    - records_final.pkl  : patient visit sequences [diag_idx, proc_idx, med_idx, hadm_id]
  - voc_final.pkl      : vocabularies (code ↔ index mappings)
  - ddi_A_final.pkl    : DDI adjacency matrix (131 × 131)
  - ehr_adj_final.pkl  : drug co-occurrence matrix (from training set only)
    - cohort_mimic3.pkl  : cohort metadata (hadm_ids, split_indices, split_seed,
                                                 patient_visits, vocab maps, and code counts)
    - preprocess_manifest.json : reproducibility metadata for this preprocessing run

Requires external mapping files in data/external/:
  - ndc2RXCUI.txt          : NDC → RXCUI (from SafeDrug/Carmen repo)
  - RXCUI2atc4.csv         : RXCUI → ATC-4 (from SafeDrug/Carmen repo)
  - idx2SMILES.pkl         : ATC-3 → SMILES (Carmen variant → 131 drugs)
  - drug-DDI.csv           : TWOSIDES DDI data

Download these from: https://github.com/ycq091044/SafeDrug/tree/main/data

Usage:
  python preprocess_mimic3.py --mimic_dir DATASETS/mimic-iii-clinical-database-1.4 \
                              --external_dir data/external \
                              --output_dir data/processed
"""

import argparse
import ast
import json
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
# Step 1: Load MIMIC-III tables
# ---------------------------------------------------------------------------

def load_prescriptions(mimic_dir: Path) -> pd.DataFrame:
    path = mimic_dir / "PRESCRIPTIONS.csv.gz"
    print(f"Loading {path.name} ...")
    df = pd.read_csv(
        path,
        usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STARTDATE", "NDC"],
        compression="gzip",
        dtype={"NDC": str},
    )
    df = df.dropna(subset=["SUBJECT_ID", "HADM_ID"])
    df["SUBJECT_ID"] = df["SUBJECT_ID"].astype(int)
    df["HADM_ID"] = df["HADM_ID"].astype(int)
    df["ICUSTAY_ID"] = pd.to_numeric(df["ICUSTAY_ID"], errors="coerce")
    df["STARTDATE"] = pd.to_datetime(df["STARTDATE"], errors="coerce")
    df = df.sort_values(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STARTDATE"])

    df["NDC"] = df["NDC"].astype(str).str.strip()
    df = df[~df["NDC"].isin(["0"])].copy()
    df["NDC"] = df["NDC"].replace({"": np.nan, "nan": np.nan})
    # Match the original Carmen/SafeDrug behavior (forward-fill after sorting).
    df["NDC"] = df.groupby("SUBJECT_ID")["NDC"].ffill()
    df = df.dropna(subset=["NDC"])
    df = df.drop(columns=["ICUSTAY_ID", "STARTDATE"])
    df = df.drop_duplicates()
    print(f"  {len(df):,} prescription rows, {df['HADM_ID'].nunique():,} admissions")
    return df


def load_admissions(mimic_dir: Path) -> pd.DataFrame:
    path = mimic_dir / "ADMISSIONS.csv.gz"
    print(f"Loading {path.name} ...")
    df = pd.read_csv(
        path,
        usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME"],
        compression="gzip",
    )
    df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "ADMITTIME"])
    df["SUBJECT_ID"] = df["SUBJECT_ID"].astype(int)
    df["HADM_ID"] = df["HADM_ID"].astype(int)
    df["ADMITTIME"] = pd.to_datetime(df["ADMITTIME"], errors="coerce")
    df = df.dropna(subset=["ADMITTIME"]).drop_duplicates(subset=["SUBJECT_ID", "HADM_ID"])
    print(f"  {len(df):,} admission time rows")
    return df


def load_diagnoses(mimic_dir: Path) -> pd.DataFrame:
    path = mimic_dir / "DIAGNOSES_ICD.csv.gz"
    print(f"Loading {path.name} ...")
    df = pd.read_csv(path, usecols=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"], compression="gzip",
                      dtype={"ICD9_CODE": str})
    df = df.dropna(subset=["HADM_ID", "ICD9_CODE"])
    df["HADM_ID"] = df["HADM_ID"].astype(int)
    print(f"  {len(df):,} diagnosis rows, {df['HADM_ID'].nunique():,} admissions")
    return df


def load_procedures(mimic_dir: Path) -> pd.DataFrame:
    path = mimic_dir / "PROCEDURES_ICD.csv.gz"
    print(f"Loading {path.name} ...")
    df = pd.read_csv(path, usecols=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"], compression="gzip",
                      dtype={"ICD9_CODE": str})
    df = df.dropna(subset=["HADM_ID", "ICD9_CODE"])
    df["HADM_ID"] = df["HADM_ID"].astype(int)
    print(f"  {len(df):,} procedure rows, {df['HADM_ID'].nunique():,} admissions")
    return df


# ---------------------------------------------------------------------------
# Step 2: NDC → ATC-3 mapping
# ---------------------------------------------------------------------------

def _load_ndc2rxcui(path: Path) -> dict:
    """Parse ndc2RXCUI.txt — handles both tab-separated and Python 2 dict format."""
    content = path.read_text(encoding="utf-8", errors="replace").strip()
    # Python 2 dict literal format: {'ndc': u'rxcui', ...}
    if "u'" in content or 'u"' in content:
        if not content.startswith("{"):
            content = "{" + content + "}"
        try:
            raw = ast.literal_eval(content)
            return {k: str(v) for k, v in raw.items()
                    if isinstance(k, str) and k != "idx" and str(v)}
        except Exception:
            pass
    # Fall back to tab-separated
    result = {}
    for line in content.splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            result[parts[0]] = parts[1]
    return result


def build_ndc_to_atc3(external_dir: Path, mode: str = "canonical") -> dict[str, str]:
    """Build NDC → ATC-3 mapping.

    Modes:
      - canonical: Carmen/SafeDrug-style chain only (NDC→RXCUI→ATC), dedupe by RXCUI
      - mirror: canonical chain + direct CSV NDC fallback (legacy behavior)
    """
    # --- Source 1: ndc2RXCUI.txt → RXCUI → ATC4 chain ---
    ndc2rxcui_path = external_dir / "ndc2RXCUI.txt"
    print(f"Loading NDC→RXCUI from {ndc2rxcui_path.name} ...")
    ndc2rxcui = _load_ndc2rxcui(ndc2rxcui_path)
    print(f"  {len(ndc2rxcui):,} NDC→RXCUI mappings")

    # RXCUI → ATC-4 (detect columns by name)
    rxcui2atc4_path = external_dir / "RXCUI2atc4.csv"
    print(f"Loading RXCUI→ATC4 from {rxcui2atc4_path.name} ...")
    rxcui2atc = pd.read_csv(rxcui2atc4_path, dtype=str)
    cols = [c.upper().strip() for c in rxcui2atc.columns]
    rxcui2atc.columns = cols
    # Find the RXCUI and ATC columns by name
    rxcui_col = next((c for c in cols if 'RXCUI' in c), None)
    atc_col = next((c for c in cols if 'ATC' in c), None)
    if rxcui_col is None or atc_col is None:
        # Fallback: assume last two meaningful columns
        rxcui_col, atc_col = cols[-2], cols[-1]

    # Canonical behavior from Carmen: keep one ATC mapping per RXCUI.
    # NOTE: Do NOT sort before dedup. Carmen/SafeDrug/HiDR/VITA all use first-file-occurrence
    # (no sort), which preserves R02A mappings and yields 131 drugs. Sorting alphabetically
    # caused D06AX to beat R02AB, dropping R02A entirely and yielding only 130 drugs.
    if mode == "canonical":
        rxcui2atc = rxcui2atc.drop(columns=["YEAR", "MONTH", "NDC"], errors="ignore")
        rxcui2atc = rxcui2atc.drop_duplicates(subset=[rxcui_col], keep="first")

    rxcui2atc_pair = rxcui2atc[[rxcui_col, atc_col]].dropna()
    rxcui2atc4_dict = dict(zip(rxcui2atc_pair[rxcui_col], rxcui2atc_pair[atc_col]))
    print(f"  {len(rxcui2atc4_dict):,} RXCUI→ATC4 mappings")

    # Chain: ndc2RXCUI → RXCUI2atc4 → ATC3
    ndc2atc3 = {}
    for ndc, rxcui in ndc2rxcui.items():
        if rxcui in rxcui2atc4_dict:
            atc4 = rxcui2atc4_dict[rxcui]
            ndc2atc3[ndc] = atc4[:4]
    print(f"  Chain (ndc→rxcui→atc3): {len(ndc2atc3):,} mappings")

    if mode == "mirror":
        # Legacy extension: direct NDC column in RXCUI2atc4.csv.
        ndc_col = next((c for c in cols if 'NDC' in c), None)
        if ndc_col and atc_col:
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
            print(f"  Direct CSV NDC→ATC3: +{direct_count:,} new mappings")

    print(f"  Total NDC→ATC3: {len(ndc2atc3):,} mappings")
    return ndc2atc3


# ---------------------------------------------------------------------------
# Step 3: Map and filter drugs
# ---------------------------------------------------------------------------

def map_prescriptions_to_atc3(
    prescriptions: pd.DataFrame,
    ndc2atc3: dict[str, str],
) -> pd.DataFrame:
    """Map NDC codes to ATC-3 and deduplicate per admission."""
    prescriptions = prescriptions.copy()
    prescriptions["ATC3"] = prescriptions["NDC"].map(ndc2atc3)
    prescriptions = prescriptions.dropna(subset=["ATC3"])
    # Deduplicate: one ATC3 per admission
    prescriptions = prescriptions.drop_duplicates(subset=["HADM_ID", "ATC3"])
    print(f"  After ATC3 mapping: {len(prescriptions):,} (admission, drug) pairs")
    return prescriptions


def filter_subjects_by_min_visits(
    prescriptions: pd.DataFrame,
    min_visits: int = 2,
) -> pd.DataFrame:
    """Filter medication table to subjects with at least `min_visits` admissions.

    This mirrors Carmen's visit>=2 filtering before NDC→ATC conversion.
    """
    visit_counts = (
        prescriptions[["SUBJECT_ID", "HADM_ID"]]
        .drop_duplicates()
        .groupby("SUBJECT_ID")["HADM_ID"]
        .nunique()
    )
    keep_subjects = set(visit_counts[visit_counts >= min_visits].index)
    filtered = prescriptions[prescriptions["SUBJECT_ID"].isin(keep_subjects)].copy()
    print(
        f"  After >={min_visits} visit subject filter: "
        f"{len(keep_subjects):,} subjects, {filtered['HADM_ID'].nunique():,} admissions"
    )
    return filtered


def filter_drugs_by_smiles(
    prescriptions: pd.DataFrame,
    external_dir: Path,
    top_k: int = 300,
    order: str = "topk_then_smiles",
) -> tuple[pd.DataFrame, list[str]]:
    """Filter ATC3 drugs by frequency and structure availability.

    order:
      - smiles_then_topk (canonical Carmen sequence)
      - topk_then_smiles (legacy sequence)
    """
    if order not in {"smiles_then_topk", "topk_then_smiles"}:
        raise ValueError(f"Unknown drug filter order: {order}")

    smiles_atc3: set[str] = set()
    smiles_path = external_dir / "idx2SMILES.pkl"
    if smiles_path.exists():
        idx2smiles = _load_pickle_robust(smiles_path)
        smiles_atc3 = set(idx2smiles.keys()) if isinstance(idx2smiles, dict) else set()
    else:
        print(f"  WARNING: {smiles_path} not found. Using all top-{top_k} drugs.")

    if order == "smiles_then_topk":
        if smiles_atc3:
            prescriptions = prescriptions[prescriptions["ATC3"].isin(smiles_atc3)]
            print(f"  After SMILES filter: {prescriptions['ATC3'].nunique()} unique drugs")
        atc3_counts = prescriptions["ATC3"].value_counts()
        top_atc3 = set(atc3_counts.head(top_k).index)
        prescriptions = prescriptions[prescriptions["ATC3"].isin(top_atc3)]
        print(f"  After top-{top_k} ATC3 filter: {prescriptions['ATC3'].nunique()} unique drugs")
    else:
        atc3_counts = prescriptions["ATC3"].value_counts()
        top_atc3 = set(atc3_counts.head(top_k).index)
        prescriptions = prescriptions[prescriptions["ATC3"].isin(top_atc3)]
        print(f"  After top-{top_k} ATC3 filter: {prescriptions['ATC3'].nunique()} unique drugs")
        if smiles_atc3:
            prescriptions = prescriptions[prescriptions["ATC3"].isin(smiles_atc3)]
            print(f"  After SMILES filter: {prescriptions['ATC3'].nunique()} unique drugs")

    final_drugs = sorted(prescriptions["ATC3"].unique())
    print(f"  Final drug vocabulary: {len(final_drugs)} drugs (expected ~131 canonical)")
    return prescriptions, final_drugs


# ---------------------------------------------------------------------------
# Step 4-5: Filter diagnoses and procedures
# ---------------------------------------------------------------------------

def filter_top_diagnoses(diagnoses: pd.DataFrame, top_k: int = 2000) -> tuple[pd.DataFrame, list[str]]:
    """Keep top 2000 most frequent ICD-9 diagnosis codes."""
    code_counts = diagnoses["ICD9_CODE"].value_counts()
    top_codes = set(code_counts.head(top_k).index)
    diagnoses = diagnoses[diagnoses["ICD9_CODE"].isin(top_codes)]
    final_diag = sorted(diagnoses["ICD9_CODE"].unique())
    print(f"  Top-{top_k} diagnoses: {len(final_diag)} unique codes")
    return diagnoses, final_diag


def get_procedures(procedures: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Keep all procedure codes (consistent with Carmen pipeline)."""
    final_proc = sorted(procedures["ICD9_CODE"].unique())
    print(f"  Procedures (unfiltered): {len(final_proc)} unique codes")
    return procedures, final_proc


# ---------------------------------------------------------------------------
# Step 6-7: Patient filtering and inner join
# ---------------------------------------------------------------------------

def build_patient_records(
    prescriptions: pd.DataFrame,
    diagnoses: pd.DataFrame,
    procedures: pd.DataFrame,
    admissions: pd.DataFrame | None = None,
    min_visits: int = 1,
) -> tuple[list, dict, dict]:
    """Build per-patient visit sequences from the inner-joined cohort.

    Returns:
        records_raw: list of patients, each patient = list of visits,
                     each visit = [diag_codes, proc_codes, med_codes, hadm_id]
        patient_visits: {subject_id: [hadm_id_1, hadm_id_2, ...]}
        used_codes: dict with sets for diag/proc/med codes used in retained visits
    """
    med_key = prescriptions[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
    diag_key = diagnoses[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
    proc_key = procedures[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
    combined = med_key.merge(diag_key, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    combined = combined.merge(proc_key, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    print(f"  Admissions with all 3 code types: {len(combined):,}")

    prescriptions = prescriptions.merge(combined, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    diagnoses = diagnoses.merge(combined, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    procedures = procedures.merge(combined, on=["SUBJECT_ID", "HADM_ID"], how="inner")

    med_per_hadm = (
        prescriptions.groupby(["SUBJECT_ID", "HADM_ID"])["ATC3"]
        .apply(lambda x: sorted(set(x)))
        .to_dict()
    )
    diag_per_hadm = (
        diagnoses.groupby(["SUBJECT_ID", "HADM_ID"])["ICD9_CODE"]
        .apply(lambda x: sorted(set(x)))
        .to_dict()
    )
    proc_per_hadm = (
        procedures.groupby(["SUBJECT_ID", "HADM_ID"])["ICD9_CODE"]
        .apply(lambda x: sorted(set(x)))
        .to_dict()
    )

    if admissions is not None:
        adm = admissions[["SUBJECT_ID", "HADM_ID", "ADMITTIME"]].drop_duplicates(
            subset=["SUBJECT_ID", "HADM_ID"]
        )
        ordered = combined.merge(adm, on=["SUBJECT_ID", "HADM_ID"], how="left")
        ordered["ADMITTIME"] = pd.to_datetime(ordered["ADMITTIME"], errors="coerce")
        ordered = ordered.sort_values(["SUBJECT_ID", "ADMITTIME", "HADM_ID"])
    else:
        ordered = combined.sort_values(["SUBJECT_ID", "HADM_ID"])

    patient_hadms = defaultdict(list)
    for _, row in ordered.iterrows():
        patient_hadms[int(row["SUBJECT_ID"])].append(int(row["HADM_ID"]))

    records_raw = []
    patient_visits = {}
    used_diag = set()
    used_proc = set()
    used_med = set()

    for subject_id in sorted(patient_hadms.keys()):
        hadms = patient_hadms[subject_id]
        patient_visits[subject_id] = hadms
        patient_record = []
        for hadm_id in hadms:
            key = (subject_id, hadm_id)
            diag_codes = diag_per_hadm.get(key, [])
            proc_codes = proc_per_hadm.get(key, [])
            med_codes = med_per_hadm.get(key, [])
            if diag_codes and proc_codes and med_codes:
                patient_record.append([diag_codes, proc_codes, med_codes, hadm_id])
                used_diag.update(diag_codes)
                used_proc.update(proc_codes)
                used_med.update(med_codes)
        if len(patient_record) >= min_visits:
            records_raw.append(patient_record)

    total_visits = sum(len(p) for p in records_raw)
    print(f"  Final: {len(records_raw):,} patients, {total_visits:,} visits")
    used_codes = {
        "diag": used_diag,
        "proc": used_proc,
        "med": used_med,
    }
    return records_raw, patient_visits, used_codes


# ---------------------------------------------------------------------------
# Step 8: DDI matrix
# ---------------------------------------------------------------------------

def _load_cid_to_atc3(external_dir: Path) -> dict[str, set[str]]:
    """Parse drug-atc.csv (CID,ATC1[,ATC2,...]) → {CID: {ATC3, ...}}."""
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


def build_ddi_matrix(
    external_dir: Path,
    drug_vocab: dict[str, int],
    num_drugs: int,
    topk_severe: int = 40,
) -> np.ndarray:
    """Build binary DDI adjacency matrix.

    Primary: remap Carmen's pre-built ATC3-level DDI matrix (ddi_A_final_carmen.pkl)
    to our drug vocabulary.  This avoids the CID→ATC3 many-to-many join that
    inflates DDI density from ~5% to ~73%.

    Fallback: build from TWOSIDES drug-DDI.csv (only if Carmen's file is missing).
    """
    # ── Primary path: use Carmen's reference DDI matrix ──
    carmen_ddi_path = external_dir / "ddi_A_final_carmen.pkl"
    carmen_voc_path = external_dir / "voc_carmen_mimic3.pkl"
    if carmen_ddi_path.exists() and carmen_voc_path.exists():
        print("  Using Carmen reference DDI matrix (correct ATC3-level pairs)...")
        with open(carmen_ddi_path, "rb") as f:
            carmen_ddi = np.array(pickle.load(f))
        try:
            import dill
            with open(carmen_voc_path, "rb") as f:
                cvoc = dill.load(f)
        except ImportError:
            import sys, builtins
            sys.modules['__builtin__'] = builtins
            with open(carmen_voc_path, "rb") as f:
                cvoc = pickle.load(f)
        carmen_w2i = cvoc['med_voc'].word2idx
        # Remap: carmen_idx ↔ ATC3 ↔ our_idx
        ddi_matrix = np.zeros((num_drugs, num_drugs), dtype=np.float32)
        overlap = set(drug_vocab.keys()) & set(carmen_w2i.keys())
        for a1 in overlap:
            for a2 in overlap:
                if a1 == a2:
                    continue
                ci, cj = carmen_w2i[a1], carmen_w2i[a2]
                if carmen_ddi[ci, cj] > 0:
                    oi, oj = drug_vocab[a1], drug_vocab[a2]
                    ddi_matrix[oi, oj] = 1
                    ddi_matrix[oj, oi] = 1
        n_pairs = int(np.triu(ddi_matrix, k=1).sum())
        density = n_pairs / max(num_drugs * (num_drugs - 1) // 2, 1) * 100
        print(f"  DDI pairs: {n_pairs} ({density:.1f}% density)")
        return ddi_matrix

    # ── Fallback: build from TWOSIDES (CID-level) ──
    print("  WARNING: Carmen DDI files not found. Falling back to TWOSIDES build.")
    ddi_path = external_dir / "drug-DDI.csv"
    if not ddi_path.exists():
        print(f"  WARNING: {ddi_path} not found. Returning empty DDI matrix.")
        return np.zeros((num_drugs, num_drugs), dtype=np.float32)

    cid2atc3 = _load_cid_to_atc3(external_dir)

    print(f"Loading DDI data from {ddi_path.name} ...")
    ddi_df = pd.read_csv(ddi_path, dtype=str)
    cols = ddi_df.columns.tolist()
    print(f"  {len(ddi_df):,} DDI rows, columns: {cols}")

    if len(cols) >= 3 and topk_severe > 0:
        se_col = cols[2]
        se_counts = ddi_df[se_col].value_counts()
        top_se = set(se_counts.head(topk_severe).index)
        before = len(ddi_df)
        ddi_df = ddi_df[ddi_df[se_col].isin(top_se)]
        print(f"  Top-{topk_severe} side effects: {len(ddi_df):,} rows (from {before:,})")

    # Use UNIQUE CID pairs — one interaction per molecular pair
    ddi_pairs = ddi_df.iloc[:, :2].drop_duplicates()
    ddi_pairs.columns = ["CID1", "CID2"]
    print(f"  Unique CID pairs: {len(ddi_pairs):,}")

    cid_rows = [(c, a) for c, atcs in cid2atc3.items() for a in atcs]
    if not cid_rows:
        print("  WARNING: No CID→ATC3 mappings. DDI matrix will be empty.")
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

    n_pairs = len(unique_pairs)
    density = n_pairs / max(num_drugs * (num_drugs - 1) // 2, 1) * 100
    print(f"  DDI pairs in vocabulary: {n_pairs} ({density:.1f}% density)")
    if density > 20:
        print("  *** WARNING: DDI density > 20% — CID→ATC3 many-to-many join "
              "likely inflated pairs. Consider using Carmen reference DDI. ***")
    return ddi_matrix


# ---------------------------------------------------------------------------
# Step 9: Co-occurrence matrix and drug history
# ---------------------------------------------------------------------------

def build_cooccurrence_matrix(
    train_records: list,
    num_drugs: int,
) -> np.ndarray:
    """Build drug co-occurrence adjacency from TRAINING set only."""
    cooccur = np.zeros((num_drugs, num_drugs), dtype=np.float32)
    if num_drugs == 0:
        print("  WARNING: 0 drugs, returning empty co-occurrence matrix.")
        return cooccur
    t0 = time.time()
    for pi, patient in enumerate(train_records):
        if pi % 500 == 0:
            _progress(f"{pi}/{len(train_records)} patients", pi, len(train_records), t0)
        for visit in patient:
            meds = visit[2]  # medication indices
            for i in range(len(meds)):
                for j in range(i + 1, len(meds)):
                    cooccur[meds[i], meds[j]] += 1
                    cooccur[meds[j], meds[i]] += 1
    _progress("done", len(train_records), len(train_records), t0)
    # Normalize to [0, 1]
    if cooccur.size > 0 and cooccur.max() > 0:
        cooccur = cooccur / cooccur.max()
    print(f"  Co-occurrence: {(cooccur > 0).sum() // 2:,} non-zero pairs (from training set)")
    return cooccur


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_by_patient(
    records: list,
    train_ratio: float = 4 / 6,
    val_ratio: float = 1 / 6,
    seed: int = 42,
) -> tuple[list, list, list]:
    """Split patients into train/val/test (4:1:1 by patient)."""
    rng = np.random.RandomState(seed)
    n = len(records)
    indices = rng.permutation(n)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = sorted(indices[:train_end])
    val_idx = sorted(indices[train_end:val_end])
    test_idx = sorted(indices[val_end:])

    train = [records[i] for i in train_idx]
    val = [records[i] for i in val_idx]
    test = [records[i] for i in test_idx]

    print(f"  Split: train={len(train)}, val={len(val)}, test={len(test)} patients")
    return train, val, test


def split_indices_by_patient(
    n: int,
    train_ratio: float = 4 / 6,
    val_ratio: float = 1 / 6,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Return deterministic patient indices for train/val/test."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = sorted(indices[:train_end])
    val_idx = sorted(indices[train_end:val_end])
    test_idx = sorted(indices[val_end:])
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess MIMIC-III (Carmen/SafeDrug pipeline)")
    parser.add_argument("--mimic_dir", type=str, required=True,
                        help="Path to mimic-iii-clinical-database-1.4/")
    parser.add_argument("--external_dir", type=str, required=True,
                        help="Path to external mapping files (NDC→RXCUI, SMILES, DDI)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for processed pickle files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preprocess_mode",
        type=str,
        default="canonical",
        choices=["canonical", "mirror"],
        help="canonical: Carmen/SafeDrug-comparable filtering; mirror: legacy relaxed behavior",
    )
    parser.add_argument("--min_visits", type=int, default=2,
                        help="Minimum visits per patient for cohort inclusion")
    parser.add_argument("--top_k_atc3", type=int, default=300,
                        help="Top-K ATC3 codes for medication vocabulary")
    parser.add_argument("--top_k_diagnoses", type=int, default=2000,
                        help="Top-K diagnosis ICD codes")
    parser.add_argument(
        "--final_min_visits",
        type=int,
        default=1,
        help="Minimum visits per patient in final records (1 matches Carmen file semantics)",
    )
    args = parser.parse_args()

    mimic_dir = Path(args.mimic_dir)
    external_dir = Path(args.external_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_t0 = time.time()
    drug_filter_order = "smiles_then_topk" if args.preprocess_mode == "canonical" else "topk_then_smiles"
    print(
        f"\nPreprocess mode: {args.preprocess_mode} "
        f"(drug order: {drug_filter_order}, min_visits: {args.min_visits})"
    )

    # Step 1: Load tables
    print("\n=== Step 1: Load MIMIC-III tables ===")
    t0 = time.time()
    prescriptions = load_prescriptions(mimic_dir)
    diagnoses = load_diagnoses(mimic_dir)
    procedures = load_procedures(mimic_dir)
    admissions = load_admissions(mimic_dir)

    if args.preprocess_mode == "canonical":
        print("  Applying canonical pre-filter: subjects with >=2 visits before drug mapping")
        prescriptions = filter_subjects_by_min_visits(prescriptions, min_visits=args.min_visits)
    print(f"  Step 1 done ({time.time() - t0:.1f}s)")

    # Step 2: NDC → ATC-3
    print("\n=== Step 2: NDC → ATC-3 mapping ===")
    t0 = time.time()
    ndc2atc3 = build_ndc_to_atc3(external_dir, mode=args.preprocess_mode)
    prescriptions = map_prescriptions_to_atc3(prescriptions, ndc2atc3)
    print(f"  Step 2 done ({time.time() - t0:.1f}s)")

    # Step 3: Filter drugs (top 300 → SMILES filter → ~131)
    print("\n=== Step 3: Filter drugs ===")
    t0 = time.time()
    prescriptions, final_drugs = filter_drugs_by_smiles(
        prescriptions,
        external_dir,
        top_k=args.top_k_atc3,
        order=drug_filter_order,
    )
    if len(final_drugs) == 0:
        print("  FATAL: 0 drugs after filtering. Check NDC→RXCUI→ATC mapping chain.")
        sys.exit(1)
    print(f"  Step 3 done ({time.time() - t0:.1f}s)")

    # Step 4: Filter diagnoses (top 2000)
    print("\n=== Step 4: Filter diagnoses ===")
    diagnoses, final_diag = filter_top_diagnoses(diagnoses, top_k=args.top_k_diagnoses)

    # Step 5: Get procedures (unfiltered)
    print("\n=== Step 5: Get procedures ===")
    procedures, final_proc = get_procedures(procedures)

    # Step 6-7: Build records (inner join + chronological ordering)
    print("\n=== Step 6-7: Build patient records ===")
    t0 = time.time()
    records_raw, patient_visits, used_codes = build_patient_records(
        prescriptions,
        diagnoses,
        procedures,
        admissions=admissions,
        min_visits=args.final_min_visits,
    )
    if len(records_raw) == 0:
        print(f"  FATAL: 0 patients with >= {args.final_min_visits} visits. Check mapping chain above.")
        sys.exit(1)

    # Build vocabularies from joined cohort only (avoids dead codes)
    final_drugs = sorted(used_codes["med"])
    final_diag = sorted(used_codes["diag"])
    final_proc = sorted(used_codes["proc"])
    drug_vocab = {code: idx for idx, code in enumerate(final_drugs)}
    diag_vocab = {code: idx for idx, code in enumerate(final_diag)}
    proc_vocab = {code: idx for idx, code in enumerate(final_proc)}

    records = []
    for patient in records_raw:
        p = []
        for diag_codes, proc_codes, med_codes, hadm_id in patient:
            p.append([
                [diag_vocab[c] for c in diag_codes],
                [proc_vocab[c] for c in proc_codes],
                [drug_vocab[c] for c in med_codes],
                hadm_id,
            ])
        records.append(p)

    voc = {
        "diag_voc": {"idx2word": {v: k for k, v in diag_vocab.items()}, "word2idx": diag_vocab},
        "med_voc": {"idx2word": {v: k for k, v in drug_vocab.items()}, "word2idx": drug_vocab},
        "pro_voc": {"idx2word": {v: k for k, v in proc_vocab.items()}, "word2idx": proc_vocab},
    }

    print(f"  Step 6-7 done ({time.time() - t0:.1f}s)")

    # Split
    print("\n=== Split (4:1:1 by patient) ===")
    train_records, val_records, test_records = split_by_patient(records, seed=args.seed)
    train_idx, val_idx, test_idx = split_indices_by_patient(len(records), seed=args.seed)

    # Step 8: DDI matrix
    print("\n=== Step 8: DDI matrix ===")
    t0 = time.time()
    num_drugs = len(final_drugs)
    ddi_matrix = build_ddi_matrix(external_dir, drug_vocab, num_drugs)
    print(f"  Step 8 done ({time.time() - t0:.1f}s)")

    # Step 9: Co-occurrence (training set only)
    print("\n=== Step 9: Co-occurrence matrix ===")
    t0 = time.time()
    ehr_adj = build_cooccurrence_matrix(train_records, num_drugs)
    print(f"  Step 9 done ({time.time() - t0:.1f}s)")

    # Build ordered hadm_ids and split assignments for downstream use
    all_hadm_ids = []
    all_splits = []
    for split_name, split_records in [("train", train_records),
                                        ("val", val_records),
                                        ("test", test_records)]:
        for patient in split_records:
            for visit in patient:
                all_hadm_ids.append(visit[3])  # hadm_id
                all_splits.append(split_name)

    # Schema assertions (fail fast on corrupted artifacts)
    if any(len(v) != 4 for p in records for v in p):
        raise RuntimeError("Invalid records schema: each visit must have 4 fields [diag, proc, med, hadm_id].")
    if any(not isinstance(v[3], (int, np.integer)) for p in records for v in p):
        raise RuntimeError("Invalid hadm_id type in records: expected int.")
    if any((not v[0]) or (not v[1]) or (not v[2]) for p in records for v in p):
        raise RuntimeError("Invalid empty code list found in records after filtering.")

    # Save outputs
    print("\n=== Saving outputs ===")

    # MIRROR format: keep hadm_id in records [diag, proc, med, hadm_id]
    with open(output_dir / "records_final.pkl", "wb") as f:
        pickle.dump(records, f)
    with open(output_dir / "voc_final.pkl", "wb") as f:
        pickle.dump(voc, f)
    with open(output_dir / "ddi_A_final.pkl", "wb") as f:
        pickle.dump(ddi_matrix, f)
    with open(output_dir / "ehr_adj_final.pkl", "wb") as f:
        pickle.dump(ehr_adj, f)

    # MIRROR cohort metadata (for lab/note extraction)
    cohort = {
        "hadm_ids": np.array(all_hadm_ids),
        "split": all_splits,
        "split_seed": int(args.seed),
        "split_indices": {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        },
        "patient_visits": patient_visits,
        "drug_vocab": drug_vocab,
        "diag_vocab": diag_vocab,
        "proc_vocab": proc_vocab,
        "num_drugs": num_drugs,
        "num_diag": len(final_diag),
        "num_proc": len(final_proc),
    }
    with open(output_dir / "cohort_mimic3.pkl", "wb") as f:
        pickle.dump(cohort, f)

    # Save reproducibility manifest
    ddi_pairs = int(np.triu(ddi_matrix, k=1).sum())
    ddi_density = ddi_pairs / max(num_drugs * (num_drugs - 1) // 2, 1)
    ehr_pairs = int(np.triu((ehr_adj > 0).astype(np.int32), k=1).sum())
    manifest = {
        "preprocess_mode": args.preprocess_mode,
        "seed": int(args.seed),
        "min_visits": int(args.min_visits),
        "final_min_visits": int(args.final_min_visits),
        "top_k_atc3": int(args.top_k_atc3),
        "top_k_diagnoses": int(args.top_k_diagnoses),
        "patients": int(len(records)),
        "visits": int(sum(len(p) for p in records)),
        "avg_visits_per_patient": float(np.mean([len(p) for p in records])),
        "num_drugs": int(num_drugs),
        "num_diag": int(len(final_diag)),
        "num_proc": int(len(final_proc)),
        "split_patients": {
            "train": int(len(train_records)),
            "val": int(len(val_records)),
            "test": int(len(test_records)),
        },
        "split_seed": int(args.seed),
        "ddi_pairs": ddi_pairs,
        "ddi_density": float(ddi_density),
        "ehr_nonzero_pairs": ehr_pairs,
    }
    with open(output_dir / "preprocess_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved to {output_dir}/:")
    print(f"  records_final.pkl  ({len(records):,} patients)")
    print(f"  voc_final.pkl      (diag={len(final_diag)}, proc={len(final_proc)}, med={num_drugs})")
    print(f"  ddi_A_final.pkl    ({num_drugs}×{num_drugs})")
    print(f"  ehr_adj_final.pkl  ({num_drugs}×{num_drugs})")
    print(f"  cohort_mimic3.pkl  ({len(all_hadm_ids):,} admissions)")
    print(f"  preprocess_manifest.json")

    # Summary statistics
    visit_counts = [len(p) for p in records]
    med_counts = [len(v[2]) for p in records for v in p]
    print(f"\n=== Summary ===")
    print(f"  Patients: {len(records):,}")
    print(f"  Visits: {sum(visit_counts):,}")
    print(f"  Avg visits/patient: {np.mean(visit_counts):.2f}")
    print(f"  Drugs: {num_drugs}")
    print(f"  Diagnoses: {len(final_diag)}")
    print(f"  Procedures: {len(final_proc)}")
    print(f"  Avg meds/visit: {np.mean(med_counts):.1f}")
    print(f"  DDI pairs: {int(ddi_matrix.sum() / 2)}")
    print(f"\n  Total preprocessing time: {time.time() - pipeline_t0:.1f}s")


if __name__ == "__main__":
    main()
