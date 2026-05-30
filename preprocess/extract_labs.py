"""
Extract and preprocess lab values from MIMIC-III/IV for the MIRROR framework.

Default pipeline (36d per admission):
  - 18 curated lab tests → most recent value per admission
  - Two-tier outlier clipping (MIMIC-Extract ranges)
  - Z-score normalization (training set stats, computed after clipping)
  - Binary missingness flags

Ablation extension (72d per admission):
  - Adds per-test slope and variance across visit history (≥3 visits required)

Usage:
  python extract_labs.py --mimic_dir DATASETS/mimic-iii-clinical-database-1.4 \
                         --cohort_file data/processed/cohort_mimic3.pkl \
                         --output_dir data/processed \
                         --compute_trends
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from lab_ranges import (
    LAB_RANGES,
    ITEMID_TO_NAME,
    clip_lab_value,
)

TOP_LAB_SETS = {
    5: [51221, 51301, 51265, 51222, 51249],
    10: [51221, 51301, 51265, 51222, 51249, 51279, 51250, 51248, 51277, 50971],
    18: [51221, 51301, 51265, 51222, 51249, 51279, 51250, 51248, 51277, 50971, 50983, 50902, 50882, 50868, 51006, 50912, 50931, 50960],
    30: [51221, 51301, 51265, 51222, 51249, 51279, 51250, 51248, 51277, 50971, 50983, 50902, 50882, 50868, 51006, 50912, 50931, 50960, 50893, 50970, 51237, 51274, 51275, 51256, 51254, 51244, 51146, 51200, 50885, 50820],
    50: [51221, 51301, 51265, 51222, 51249, 51279, 51250, 51248, 51277, 50971, 50983, 50902, 50882, 50868, 51006, 50912, 50931, 50960, 50893, 50970, 51237, 51274, 51275, 51256, 51254, 51244, 51146, 51200, 50885, 50820, 51491, 50802, 50804, 50821, 50818, 51498, 50813, 50861, 50878, 50863, 50862, 50822, 50910, 50808, 50809, 51144, 50810, 50811, 50954, 50824]
}

def load_lab_names(mimic_dir: Path, mimic_version: int = 3) -> dict[int, str]:
    """Load ITEMID to Label mapping from D_LABITEMS."""
    if mimic_version == 3:
        path = mimic_dir / "D_LABITEMS.csv.gz"
        cols = ["ITEMID", "LABEL"]
    else:
        path = mimic_dir / "hosp" / "d_labitems.csv.gz"
        cols = ["itemid", "label"]
    
    if not path.exists():
        return {}
        
    print(f"Loading lab names from {path} ...")
    df = pd.read_csv(path, usecols=cols, compression="gzip")
    df.columns = df.columns.str.upper()
    return dict(zip(df["ITEMID"], df["LABEL"]))

def get_top_n_itemids(mimic_dir: Path, n: int, hadm_ids: set, mimic_version: int = 3) -> list[int]:
    """Find the top N most frequent lab ITEMIDs in the cohort (chunked read)."""
    if mimic_version == 3:
        path = mimic_dir / "LABEVENTS.csv.gz"
        cols = ["HADM_ID", "ITEMID"]
    else:
        path = mimic_dir / "hosp" / "labevents.csv.gz"
        cols = ["hadm_id", "itemid"]

    print(f"Finding top {n} labs from {path} (chunked) ...")
    # Count per (ITEMID, HADM_ID) pair — use a Counter to stay off large DataFrames.
    from collections import defaultdict
    itemid_hadm: dict = defaultdict(set)
    n_rows = 0
    for chunk in pd.read_csv(path, usecols=cols, compression="gzip", chunksize=100_000):
        chunk.columns = chunk.columns.str.upper()
        chunk = chunk[chunk["HADM_ID"].isin(hadm_ids)].dropna()
        for itemid, hadm_id in zip(chunk["ITEMID"], chunk["HADM_ID"]):
            itemid_hadm[int(itemid)].add(int(hadm_id))
        n_rows += len(chunk)
    counts = {iid: len(hadms) for iid, hadms in itemid_hadm.items()}
    top_itemids = sorted(counts, key=counts.get, reverse=True)[:n]
    top_freq = counts[top_itemids[0]] if top_itemids else 0
    print(f"  Identified {len(top_itemids)} labs. Top lab frequency: {top_freq}/{len(hadm_ids)}")
    return top_itemids



def load_labevents(mimic_dir: Path, lab_itemids: list, mimic_version: int = 3,
                   cohort_hadm_ids: set = None) -> pd.DataFrame:
    """Load LABEVENTS filtered to target ITEMIDs and cohort (chunked read)."""
    if mimic_version == 3:
        path = mimic_dir / "LABEVENTS.csv.gz"
        cols = ["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM"]
    else:
        path = mimic_dir / "hosp" / "labevents.csv.gz"
        cols = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"]

    itemid_set = set(lab_itemids)
    print(f"Loading LABEVENTS from {path} (chunked) ...")
    chunks = []
    n_rows = 0
    for chunk in pd.read_csv(path, usecols=cols, compression="gzip", chunksize=100_000):
        chunk.columns = chunk.columns.str.upper()
        # Filter to target items first (cheap int comparison), then cohort
        chunk = chunk[chunk["ITEMID"].isin(itemid_set)]
        if cohort_hadm_ids is not None:
            chunk = chunk[chunk["HADM_ID"].isin(cohort_hadm_ids)]
        chunk = chunk.dropna(subset=["HADM_ID", "VALUENUM"])
        if not chunk.empty:
            chunks.append(chunk)
        n_rows += 100_000
        if n_rows % 2_000_000 == 0:
            kept = sum(len(c) for c in chunks)
            print(f"  {n_rows:,} rows scanned, {kept:,} kept ...")

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM"]
    )
    df["HADM_ID"] = df["HADM_ID"].astype(int)
    df["CHARTTIME"] = pd.to_datetime(df["CHARTTIME"])

    print(f"  Loaded {len(df):,} lab measurements for {df['HADM_ID'].nunique():,} admissions")
    return df


def apply_outlier_clipping(df: pd.DataFrame, lab_itemids: list) -> pd.DataFrame:
    """Apply two-tier outlier clipping to lab values."""
    clipped = df.copy()
    removed_count = 0
    clipped_count = 0

    for itemid in lab_itemids:
        mask = clipped["ITEMID"] == itemid
        if not mask.any():
            continue
        original = clipped.loc[mask, "VALUENUM"].copy()

        if itemid in LAB_RANGES:
            _, outlier_lo, valid_lo, valid_hi, outlier_hi, _ = LAB_RANGES[itemid]
        else:
            # Fallback for labs without explicit ranges
            outlier_lo, valid_lo, valid_hi, outlier_hi = -1e9, -1e9, 1e9, 1e9

        # Tier 1: remove impossible values
        impossible = (original < outlier_lo) | (original > outlier_hi)
        removed_count += impossible.sum()
        clipped.loc[mask & impossible, "VALUENUM"] = np.nan

        # Tier 2: clip extreme-but-valid
        remaining = mask & ~impossible
        low_clip = remaining & (clipped["VALUENUM"] < valid_lo)
        high_clip = remaining & (clipped["VALUENUM"] > valid_hi)
        clipped_count += low_clip.sum() + high_clip.sum()
        clipped.loc[low_clip, "VALUENUM"] = valid_lo
        clipped.loc[high_clip, "VALUENUM"] = valid_hi

    clipped = clipped.dropna(subset=["VALUENUM"])
    print(f"  Outlier clipping: {removed_count:,} impossible values removed, "
          f"{clipped_count:,} extreme values clipped")
    return clipped


def extract_most_recent(df: pd.DataFrame) -> pd.DataFrame:
    """Per admission, keep only the most recent value for each test."""
    df = df.sort_values("CHARTTIME")
    most_recent = df.groupby(["HADM_ID", "ITEMID"]).last().reset_index()
    return most_recent[["HADM_ID", "ITEMID", "VALUENUM"]]


def pivot_to_matrix(df: pd.DataFrame, hadm_ids: np.ndarray, lab_itemids: list) -> tuple[np.ndarray, np.ndarray]:
    """Convert long-format lab values to (N, num_labs) value matrix + (N, num_labs) flag matrix.

    Returns:
        values: (N, num_labs) float array. Missing = 0.0 (will be overwritten after z-score).
        flags:  (N, num_labs) int array. 1 = missing, 0 = present.
    """
    hadm_to_idx = {h: i for i, h in enumerate(hadm_ids)}
    n = len(hadm_ids)
    itemid_to_col = {iid: j for j, iid in enumerate(lab_itemids)}
    num_labs = len(lab_itemids)

    values = np.full((n, num_labs), np.nan, dtype=np.float32)

    for _, row in df.iterrows():
        hadm_id = int(row["HADM_ID"])
        if hadm_id not in hadm_to_idx:
            continue
        i = hadm_to_idx[hadm_id]
        j = itemid_to_col[row["ITEMID"]]
        values[i, j] = row["VALUENUM"]

    flags = np.isnan(values).astype(np.float32)  # 1 = missing
    values = np.nan_to_num(values, nan=0.0)  # temporary; overwritten after z-score

    return values, flags


def compute_zscore_stats(values: np.ndarray, flags: np.ndarray, num_labs: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-test mean and std from non-missing values (training set only).

    Returns:
        means: (18,) array
        stds:  (18,) array (with floor of 1e-6 to avoid division by zero)
    """
    means = np.zeros(num_labs, dtype=np.float32)
    stds = np.ones(num_labs, dtype=np.float32)

    for j in range(num_labs):
        present = flags[:, j] == 0
        if present.sum() > 1:
            col = values[present, j]
            means[j] = col.mean()
            stds[j] = max(col.std(), 1e-6)

    return means, stds


def apply_zscore(values: np.ndarray, flags: np.ndarray,
                 means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Z-score normalize values; missing positions stay 0."""
    normalized = (values - means) / stds
    normalized[flags == 1] = 0.0  # missing → 0 (neutral after z-score)
    return normalized.astype(np.float32)


def build_lab_vectors(values_z: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """Concatenate z-scored values and missingness flags → 36d vectors."""
    return np.concatenate([values_z, flags], axis=1).astype(np.float32)  # (N, 36)


# ---------- Ablation: Trend features ----------

def compute_trend_features(
    lab_df: pd.DataFrame,
    cohort: dict,
    hadm_ids: np.ndarray,
    lab_itemids: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-test slope and variance across visit history.

    For each patient with ≥3 visits, compute linear regression slope
    and variance of each lab test across their visit sequence.

    Args:
        lab_df: Long-format lab values (HADM_ID, ITEMID, VALUENUM, CHARTTIME).
        cohort: Dict with patient visit sequences.
                Expected key: 'patient_visits' → {subject_id: [hadm_id_1, hadm_id_2, ...]}
        hadm_ids: Ordered array of all admission IDs.

    Returns:
        slopes:    (N, 18) array of per-test slopes
        variances: (N, 18) array of per-test variances
    """
    hadm_to_idx = {h: i for i, h in enumerate(hadm_ids)}
    n = len(hadm_ids)
    num_labs = len(lab_itemids)
    slopes = np.zeros((n, num_labs), dtype=np.float32)
    variances = np.zeros((n, num_labs), dtype=np.float32)

    # Build lookup: (hadm_id, itemid) → (charttime, value)
    lab_lookup: dict[tuple[int, int], tuple] = {}
    for _, row in lab_df.iterrows():
        key = (int(row["HADM_ID"]), int(row["ITEMID"]))
        lab_lookup[key] = (row["CHARTTIME"], row["VALUENUM"])

    itemid_to_col = {iid: j for j, iid in enumerate(lab_itemids)}

    patient_visits = cohort.get("patient_visits", {})
    for subject_id, visit_list in patient_visits.items():
        if len(visit_list) < 3:
            continue

        for j, itemid in enumerate(lab_itemids):
            times = []
            vals = []
            for hadm_id in visit_list:
                key = (hadm_id, itemid)
                if key in lab_lookup:
                    t, v = lab_lookup[key]
                    if isinstance(t, pd.Timestamp):
                        times.append(t.timestamp())
                    vals.append(v)

            if len(vals) >= 3:
                # Normalize times to days from first observation
                t_arr = np.array(times, dtype=np.float64)
                t_arr = (t_arr - t_arr[0]) / 86400.0  # seconds → days
                v_arr = np.array(vals, dtype=np.float64)

                slope_val = scipy_stats.linregress(t_arr, v_arr).slope
                var_val = np.var(v_arr)

                # Assign to the LAST visit in the sequence (prediction target)
                last_hadm = visit_list[-1]
                if last_hadm in hadm_to_idx:
                    idx = hadm_to_idx[last_hadm]
                    slopes[idx, j] = slope_val
                    variances[idx, j] = var_val

    return slopes, variances


def main():
    parser = argparse.ArgumentParser(description="Extract lab values from MIMIC")
    parser.add_argument("--mimic_dir", type=str, required=True,
                        help="Path to MIMIC-III or MIMIC-IV directory")
    parser.add_argument("--cohort_file", type=str, required=True,
                        help="Path to cohort pickle (from preprocess_mimic3.py)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for lab vectors")
    parser.add_argument("--mimic_version", type=int, default=3, choices=[3, 4],
                        help="MIMIC version (3 or 4)")
    parser.add_argument("--compute_trends", action="store_true",
                        help="Compute slope/variance trend features (72d ablation)")
    parser.add_argument("--suffix", type=str, default=None,
                        help="Override output suffix (default: _mimic{version})")
    parser.add_argument("--num_labs", type=int, default=18,
                        help="Number of labs to extract (default: 18). Set >50 for dynamic top-N.")
    args = parser.parse_args()

    mimic_dir = Path(args.mimic_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cohort
    print(f"Loading cohort from {args.cohort_file} ...")
    with open(args.cohort_file, "rb") as f:
        cohort = pickle.load(f)

    # Get ordered list of admission IDs and train/val/test split indices
    hadm_ids_list = cohort["hadm_ids"]
    hadm_ids_set = set(map(int, hadm_ids_list))
    hadm_ids = np.array(hadm_ids_list)
    train_mask = np.array(cohort["split"]) == "train"
    print(f"  Total admissions: {len(hadm_ids):,}, training: {train_mask.sum():,}")

    # Step 0: Get Lab Names
    all_lab_names = load_lab_names(mimic_dir, args.mimic_version)

    # Setup lab set
    num_labs = args.num_labs
    if num_labs in TOP_LAB_SETS:
        lab_itemids = TOP_LAB_SETS[num_labs]
    else:
        # Dynamic top-N
        lab_itemids = get_top_n_itemids(mimic_dir, num_labs, hadm_ids_set, args.mimic_version)
        num_labs = len(lab_itemids)
    
    # Step 1: Load and filter (pass cohort set for inline RAM-efficient filtering)
    lab_df = load_labevents(mimic_dir, lab_itemids, args.mimic_version,
                            cohort_hadm_ids=hadm_ids_set)

    # Step 2: Outlier clipping
    lab_df = apply_outlier_clipping(lab_df, lab_itemids)

    # Step 3: Most recent value per admission
    lab_recent = extract_most_recent(lab_df)
    print(f"  Most recent values: {len(lab_recent):,} (test, admission) pairs")

    # Step 4: Pivot to matrix
    values, flags = pivot_to_matrix(lab_recent, hadm_ids, lab_itemids)
    present_per_admission = (flags == 0).sum(axis=1)
    print(f"  Coverage: mean {present_per_admission.mean():.1f}/{num_labs} tests, "
          f"median {np.median(present_per_admission):.0f}/{num_labs}")

    # Step 5: Z-score from training set
    train_values = values[train_mask]
    train_flags = flags[train_mask]
    means, stds = compute_zscore_stats(train_values, train_flags, num_labs)
    values_z = apply_zscore(values, flags, means, stds)

    # Build 36d vectors
    lab_vectors_36d = build_lab_vectors(values_z, flags)
    has_lab = (flags == 0).any(axis=1).astype(np.float32)
    print(f"  Lab vectors (36d): shape {lab_vectors_36d.shape}")

    # Save
    lab_names = [all_lab_names.get(i, ITEMID_TO_NAME.get(i, f"Lab_{i}")) for i in lab_itemids]
    output = {
        "lab_vectors": lab_vectors_36d,  # (N, num_labs*2)
        "has_lab": has_lab,              # (N,) 1 if any target lab present in admission
        "hadm_ids": hadm_ids,
        "zscore_means": means,
        "zscore_stds": stds,
        "lab_itemids": lab_itemids,
        "lab_names": lab_names,
    }

    # Optional: compute trend features
    if args.compute_trends:
        print("Computing trend features (slope + variance) ...")
        slopes, variances = compute_trend_features(lab_df, cohort, hadm_ids, lab_itemids)

        # Z-score trends from training set
        train_slopes = slopes[train_mask]
        train_vars = variances[train_mask]
        slope_means = np.where(train_slopes.any(axis=0), train_slopes.mean(axis=0), 0)
        slope_stds = np.where(train_slopes.any(axis=0),
                              np.maximum(train_slopes.std(axis=0), 1e-6), 1.0)
        var_means = np.where(train_vars.any(axis=0), train_vars.mean(axis=0), 0)
        var_stds = np.where(train_vars.any(axis=0),
                            np.maximum(train_vars.std(axis=0), 1e-6), 1.0)

        slopes_z = (slopes - slope_means) / slope_stds
        variances_z = (variances - var_means) / var_stds

        # Where slopes/variances are 0 (not computed), keep as 0
        slopes_z[slopes == 0] = 0.0
        variances_z[variances == 0] = 0.0

        lab_vectors_72d = np.concatenate(
            [values_z, flags, slopes_z, variances_z], axis=1
        ).astype(np.float32)
        print(f"  Lab vectors (72d with trends): shape {lab_vectors_72d.shape}")

        output["lab_vectors_72d"] = lab_vectors_72d
        output["trend_zscore_stats"] = {
            "slope_means": slope_means, "slope_stds": slope_stds,
            "var_means": var_means, "var_stds": var_stds,
        }

    suffix = args.suffix or f"_{num_labs}labs"
    out_path = output_dir / f"lab_vectors{suffix}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(output, f)
    print(f"Saved to {out_path}")

    # Print summary
    print("\n=== Lab Extraction Summary ===")
    for j, iid in enumerate(lab_itemids):
        present = (flags[:, j] == 0).sum()
        pct = present / len(flags) * 100
        name = ITEMID_TO_NAME.get(iid, f"Lab_{iid}")
        print(f"  {name:20s}: {present:>6,}/{len(flags):,} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
