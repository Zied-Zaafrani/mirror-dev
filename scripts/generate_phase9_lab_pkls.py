"""
Phase 9 Local Lab PKL Generator
================================
Generates lab_vectors_150labs.pkl, 200labs, 250labs, 300labs, and {max}labs.pkl
locally using the cached LABEVENTS DataFrame (lab_cache.pkl) and the cohort.

Run from MIRROR root:
    python src/scripts/generate_phase9_lab_pkls.py

Output: data/processed/lab_vectors_{N}labs.pkl for each N in sweep.
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──
MIMIC_DIR     = Path("datasets/mimic-iii-clinical-database-1.4")
COHORT_PATH   = Path("data/processed/cohort_mimic3.pkl")
OUTPUT_DIR    = Path("data/processed")
CACHE_PATH    = MIMIC_DIR / "lab_cache.pkl"

# ── Add preprocess to path for clip_lab_value ──
sys.path.insert(0, str(Path("src/preprocess")))
try:
    from lab_ranges import LAB_RANGES, ITEMID_TO_NAME, clip_lab_value
    HAS_LAB_RANGES = True
    print("Loaded lab_ranges.py — outlier clipping enabled.")
except ImportError:
    HAS_LAB_RANGES = False
    print("WARNING: lab_ranges.py not found — skipping outlier clipping.")


def load_cohort():
    with open(COHORT_PATH, "rb") as f:
        cohort = pickle.load(f)
    hadm_ids = np.array(cohort["hadm_ids"])
    hadm_ids_set = set(map(int, hadm_ids))
    train_mask = np.array(cohort["split"]) == "train"
    return cohort, hadm_ids, hadm_ids_set, train_mask


def load_lab_cache(hadm_ids_set):
    print(f"\nLoading lab_cache.pkl ({CACHE_PATH.stat().st_size/1e6:.0f} MB)...")
    t0 = time.time()
    with open(CACHE_PATH, "rb") as f:
        df = pickle.load(f)
    print(f"  Loaded in {time.time()-t0:.1f}s — shape: {df.shape}")

    # Filter to cohort admissions with numeric values
    df = df.dropna(subset=["HADM_ID", "VALUENUM"])
    df["HADM_ID"] = df["HADM_ID"].astype(int)
    df = df[df["HADM_ID"].isin(hadm_ids_set)].copy()
    df["CHARTTIME"] = pd.to_datetime(df["CHARTTIME"])
    print(f"  After cohort filter: {len(df):,} rows, {df['HADM_ID'].nunique():,} admissions")
    return df


def get_top_n_itemids(df, n, hadm_ids_set):
    """Return top-N lab item IDs by number of distinct admissions."""
    counts = df.groupby("ITEMID")["HADM_ID"].nunique().sort_values(ascending=False)
    top = counts.head(n)
    print(f"\n  Top {n} labs — frequency range: "
          f"{top.iloc[-1]:,}–{top.iloc[0]:,} admissions "
          f"({100*top.iloc[-1]/len(hadm_ids_set):.1f}%–{100*top.iloc[0]/len(hadm_ids_set):.1f}%)")
    return top.index.tolist()


def get_max_itemids(df, hadm_ids_set, min_pct=1.0):
    """Return all lab item IDs present in >= min_pct% of cohort admissions."""
    n_adm = len(hadm_ids_set)
    counts = df.groupby("ITEMID")["HADM_ID"].nunique().sort_values(ascending=False)
    min_count = int(n_adm * min_pct / 100)
    top = counts[counts >= min_count]
    print(f"\n  Max labs (>={min_pct}% coverage, >={min_count} admissions): {len(top)}")
    return top.index.tolist()


def apply_clipping(df, lab_itemids):
    """Vectorized two-tier outlier clipping."""
    if not HAS_LAB_RANGES:
        return df
    
    t0 = time.time()
    # Build lookup arrays for efficiency
    # LAB_RANGES: {itemid: (_, outlier_lo, valid_lo, valid_hi, outlier_hi, _)}
    valid_range_map = {iid: (LAB_RANGES[iid][2], LAB_RANGES[iid][3]) for iid in lab_itemids if iid in LAB_RANGES}
    outlier_range_map = {iid: (LAB_RANGES[iid][1], LAB_RANGES[iid][4]) for iid in lab_itemids if iid in LAB_RANGES}

    # 1. Remove impossible outliers
    lo_impossible = df["ITEMID"].map(lambda x: outlier_range_map.get(x, (-1e9, 1e9))[0])
    hi_impossible = df["ITEMID"].map(lambda x: outlier_range_map.get(x, (-1e9, 1e9))[1])
    
    mask_valid = (df["VALUENUM"] >= lo_impossible) & (df["VALUENUM"] <= hi_impossible)
    clipped = df[mask_valid].copy()
    removed = len(df) - len(clipped)

    # 2. Clip extreme values to valid range
    lo_valid = clipped["ITEMID"].map(lambda x: valid_range_map.get(x, (-1e9, 1e9))[0])
    hi_valid = clipped["ITEMID"].map(lambda x: valid_range_map.get(x, (-1e9, 1e9))[1])
    
    # Efficient clipping
    original_vals = clipped["VALUENUM"].values
    clipped_vals = np.clip(original_vals, lo_valid.values, hi_valid.values)
    clipped_count = np.sum(original_vals != clipped_vals)
    clipped["VALUENUM"] = clipped_vals

    print(f"  Clipping done in {time.time()-t0:.1f}s: {removed:,} impossible removed, {clipped_count:,} extreme clipped")
    return clipped


def build_lab_vectors(most_recent_sub, hadm_ids, lab_itemids, train_mask):
    """Build (N, 2*L) lab vector matrix with z-score normalization."""
    n = len(hadm_ids)
    L = len(lab_itemids)
    hadm_to_idx = {int(h): i for i, h in enumerate(hadm_ids)}
    itemid_to_col = {iid: j for j, iid in enumerate(lab_itemids)}

    values = np.full((n, L), np.nan, dtype=np.float32)
    
    # Vectorized assignment
    h_idxs = most_recent_sub["HADM_ID"].map(hadm_to_idx).values
    iid_cols = most_recent_sub["ITEMID"].map(itemid_to_col).values
    vals = most_recent_sub["VALUENUM"].values
    
    # Only keep those that mapped successfully
    valid = (~np.isnan(h_idxs)) & (~np.isnan(iid_cols))
    h_idxs = h_idxs[valid].astype(int)
    iid_cols = iid_cols[valid].astype(int)
    vals = vals[valid]
    
    values[h_idxs, iid_cols] = vals

    flags = np.isnan(values).astype(np.float32)  # 1 = missing
    values = np.nan_to_num(values, nan=0.0)

    # Z-score from training set only
    means = np.zeros(L, dtype=np.float32)
    stds  = np.ones(L, dtype=np.float32)
    for j in range(L):
        present = (flags[train_mask, j] == 0)
        if present.sum() > 1:
            col = values[train_mask][present, j]
            means[j] = col.mean()
            stds[j]  = max(col.std(), 1e-6)

    values_z = ((values - means) / stds).astype(np.float32)
    values_z[flags == 1] = 0.0

    lab_vectors = np.concatenate([values_z, flags], axis=1)  # (N, 2*L)
    has_lab = (flags == 0).any(axis=1).astype(np.float32)

    present_per_adm = (flags == 0).sum(axis=1)
    print(f"  Coverage: mean {present_per_adm.mean():.1f}/{L}, "
          f"median {np.median(present_per_adm):.0f}/{L}, "
          f"has_lab={100*has_lab.mean():.1f}%")

    return lab_vectors, has_lab, means, stds, flags


def load_lab_names_from_dlabitems():
    path = MIMIC_DIR / "D_LABITEMS.csv.gz"
    if not path.exists():
        return {}
    df = pd.read_csv(path, usecols=["ITEMID", "LABEL"], compression="gzip")
    return dict(zip(df["ITEMID"], df["LABEL"]))


def generate_pkl(most_recent, hadm_ids, lab_itemids, train_mask,
                  lab_name_map, n_labs_label, output_dir):
    """Build and save one lab pkl file."""
    print(f"\n{'='*60}")
    print(f"Generating: lab_vectors_{n_labs_label}labs.pkl  ({len(lab_itemids)} labs)")
    print(f"{'='*60}")

    t0 = time.time()

    # Slice pre-computed most_recent
    most_recent_sub = most_recent[most_recent["ITEMID"].isin(set(lab_itemids))].copy()

    # Build vectors
    lab_vectors, has_lab, means, stds, flags = build_lab_vectors(
        most_recent_sub, hadm_ids, lab_itemids, train_mask
    )

    lab_names = [lab_name_map.get(iid, ITEMID_TO_NAME.get(iid, f"Lab_{iid}")
                                   if HAS_LAB_RANGES else f"Lab_{iid}")
                 for iid in lab_itemids]

    output = {
        "lab_vectors":  lab_vectors,
        "has_lab":      has_lab,
        "hadm_ids":     hadm_ids,
        "zscore_means": means,
        "zscore_stds":  stds,
        "lab_itemids":  lab_itemids,
        "lab_names":    lab_names,
    }

    out_path = output_dir / f"lab_vectors_{n_labs_label}labs.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(output, f)

    size_mb = out_path.stat().st_size / 1e6
    print(f"  Saved: {out_path} ({size_mb:.1f} MB) in {time.time()-t0:.1f}s")
    return out_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load cohort ──
    print("Loading cohort...")
    cohort, hadm_ids, hadm_ids_set, train_mask = load_cohort()
    N = len(hadm_ids)
    print(f"  {N:,} admissions, {train_mask.sum():,} training")

    # ── Load lab names ──
    lab_name_map = load_lab_names_from_dlabitems()
    print(f"  Loaded {len(lab_name_map)} lab name mappings from D_LABITEMS")

    # ── Load lab cache (only once) ──
    df = load_lab_cache(hadm_ids_set)

    # ── Pre-process Labs (Global Clipping & Most Recent) ──
    print("\nPre-processing all labs globally...")
    t0 = time.time()
    # Initial clipping for all found labs
    all_found_itemids = df["ITEMID"].unique().tolist()
    df_clipped = apply_clipping(df, all_found_itemids)
    
    # Pre-sort and group to get most recent values for every possible lab
    df_clipped = df_clipped.sort_values("CHARTTIME")
    most_recent_global = df_clipped.groupby(["HADM_ID", "ITEMID"])["VALUENUM"].last().reset_index()
    print(f"  Global most-recent done in {time.time()-t0:.1f}s — shape: {most_recent_global.shape}")

    # ── Count distinct labs for sweep planning ──
    counts = most_recent_global.groupby("ITEMID")["HADM_ID"].nunique().sort_values(ascending=False)
    print(f"\nDistinct lab types in cohort: {len(counts):,}")
    for pct in [50, 20, 10, 5, 1]:
        n = (counts >= N * pct / 100).sum()
        print(f"  >={pct}% coverage: {n} labs")

    max_lab_itemids = get_max_itemids(most_recent_global, hadm_ids_set, min_pct=0.0)
    MAX_LABS = len(max_lab_itemids)
    print(f"\n  => TRUE MAX LABS (absolute max): {MAX_LABS}")

    # ── Build sweep: full ablation suite ──
    # Standard top-N configs:
    #   5, 10, 20 — small-scale baselines
    #   50, 100, 150, 200, 250, 300, 350, 400 — main sweep
    #   MAX_LABS — true MIMIC maximum (no coverage filter)
    #
    # These counts were chosen to span the full range found in prior experiments.
    # The optimal lab count (200) was established empirically but all points are
    # needed for the ablation table in the thesis.
    STANDARD_COUNTS = [5, 10, 20, 50, 100, 150, 200, 250, 300, 350, 400]
    sweep = [n for n in STANDARD_COUNTS if n <= MAX_LABS]
    if MAX_LABS not in sweep:
        sweep.append(MAX_LABS)

    print(f"\nFull ablation sweep: {sweep}")

    # ── Generate each pkl ──
    generated = []
    for n_labs in sweep:
        out_path = OUTPUT_DIR / f"lab_vectors_{n_labs}labs.pkl"
        if out_path.exists():
            print(f"\n[SKIP] {out_path.name} already exists")
            generated.append(out_path)
            continue

        if n_labs == MAX_LABS:
            lab_itemids = max_lab_itemids
            label = f"{n_labs}_MAX"
        else:
            lab_itemids = get_top_n_itemids(most_recent_global, n_labs, hadm_ids_set)
            label = str(n_labs)

        out = generate_pkl(most_recent_global, hadm_ids, lab_itemids, train_mask,
                           lab_name_map, label, OUTPUT_DIR)
        generated.append(out)

    # ── Final summary ──
    print(f"\n\n{'='*60}")
    print(f"PHASE 9 PKL GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Generated {len(generated)} files:")
    for p in generated:
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
