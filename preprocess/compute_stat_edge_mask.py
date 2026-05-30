"""
Compute statistical edge mask for LabDrugCrossAttentionEncoder (Run 17B).

For each of the 18 lab tests, identify which drugs have a statistically
significant association with that lab value in the training set.

Method:
  For each (lab_i, drug_j) pair:
    - Compare mean z-score of lab_i for patients prescribed drug_j
      vs patients NOT prescribed drug_j
    - Use Welch's t-test (unequal variance) for significance
    - Keep edges with p < p_threshold (default 0.05, Bonferroni-corrected optional)

Additionally compute top-K edges per lab by absolute t-statistic for
a fixed-budget variant (useful for ablation).

Output:
  data/processed/stat_edge_mask.npy   — (18, num_drugs) bool array
  data/processed/stat_edge_tstat.npy  — (18, num_drugs) float32 |t| values
  data/processed/stat_edge_pval.npy   — (18, num_drugs) float32 p-values

Usage:
  python src/preprocess/compute_stat_edge_mask.py
  python src/preprocess/compute_stat_edge_mask.py --p_threshold 0.01 --top_k 20
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# ── Add project root to path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "preprocess"))

from lab_ranges import LAB_ITEMIDS, ITEMID_TO_NAME


NUM_LABS = 18
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_data():
    with open(DATA_DIR / "records_final.pkl", "rb") as f:
        records = pickle.load(f)
    with open(DATA_DIR / "cohort_mimic3.pkl", "rb") as f:
        cohort = pickle.load(f)
    with open(DATA_DIR / "lab_data_mimic3.pkl", "rb") as f:
        lab_data = pickle.load(f)
    return records, cohort, lab_data


def get_train_records(records, cohort):
    """Return training-split records using cohort split_indices."""
    split_idx = cohort.get("split_indices")
    if split_idx and all(k in split_idx for k in ("train", "val", "test")):
        train_idx = sorted(int(i) for i in split_idx["train"])
        print(f"  Using cohort split indices: {len(train_idx)} training patients")
    else:
        # Fallback: same permutation as train.py uses (seed 42)
        rng = np.random.RandomState(42)
        n = len(records)
        indices = rng.permutation(n)
        train_end = int(n * 4 / 6)
        train_idx = sorted(indices[:train_end])
        print(f"  Fallback to seed-42 split: {len(train_idx)} training patients")
    return [records[i] for i in train_idx]


def build_lab_drug_matrix(train_records, lab_data, num_drugs):
    """Build aligned (N, 18) lab z-score matrix and (N, num_drugs) drug label matrix.

    N = number of (patient, non-first-visit) training examples.
    Only includes examples where the target visit has lab data.

    Returns:
        X: (N, 18) float32 — lab z-scores for target visit
        Y: (N, num_drugs) float32 — drug labels for target visit
        present: (N, 18) bool — which labs are actually present (not missing)
    """
    hadm_to_lab = {}
    lab_vecs = lab_data["lab_vectors"]   # (M, 36)
    hadm_ids = lab_data["hadm_ids"]
    for i, hid in enumerate(hadm_ids):
        hadm_to_lab[int(hid)] = lab_vecs[i]

    X_list, Y_list, present_list = [], [], []
    skipped = 0

    for patient in train_records:
        for target_idx in range(1, len(patient)):
            target_visit = patient[target_idx]

            if len(target_visit) <= 3:
                skipped += 1
                continue
            hadm_id = int(target_visit[3])

            if hadm_id not in hadm_to_lab:
                skipped += 1
                continue

            lab_vec = hadm_to_lab[hadm_id]  # (36,)
            # Positions [0:18] = z-scores, [18:36] = missingness flags (1=missing, 0=present)
            z_scores = lab_vec[:18].astype(np.float32)
            missing_flags = lab_vec[18:36]  # 1.0 = missing

            # Skip if all labs missing
            if np.all(missing_flags > 0.5):
                skipped += 1
                continue

            # Drug labels
            drug_vec = np.zeros(num_drugs, dtype=np.float32)
            for m in target_visit[2]:
                if m < num_drugs:
                    drug_vec[m] = 1.0

            X_list.append(z_scores)
            Y_list.append(drug_vec)
            present_list.append(missing_flags < 0.5)  # True = present

    print(f"  Collected {len(X_list)} examples ({skipped} skipped: no hadm_id or all-missing labs)")
    X = np.stack(X_list)          # (N, 18)
    Y = np.stack(Y_list)          # (N, num_drugs)
    present = np.stack(present_list)  # (N, 18) bool
    return X, Y, present


def compute_ttest(X, Y, present):
    """Compute Welch's t-test between lab z-scores for prescribed vs not-prescribed.

    For each (lab_i, drug_j) pair:
      - pos_group: z-scores of lab_i for examples where drug_j=1 AND lab_i present
      - neg_group: z-scores of lab_i for examples where drug_j=0 AND lab_i present
      - t-stat, p-value from scipy.stats.ttest_ind (Welch's, unequal var)

    Returns:
        tstat: (18, num_drugs) float32 — absolute t-statistics
        pval:  (18, num_drugs) float32 — two-sided p-values
    """
    N, num_drugs = Y.shape
    num_labs = X.shape[1]

    tstat = np.zeros((num_labs, num_drugs), dtype=np.float32)
    pval = np.ones((num_labs, num_drugs), dtype=np.float32)

    for i in range(num_labs):
        lab_present_mask = present[:, i]  # (N,) bool — this lab is measured
        x_i = X[:, i]  # (N,) z-scores for lab i

        n_present = lab_present_mask.sum()
        if n_present < 20:
            print(f"    Lab {i} ({ITEMID_TO_NAME[LAB_ITEMIDS[i]]}): only {n_present} present — skipping")
            continue

        for j in range(num_drugs):
            pos_mask = lab_present_mask & (Y[:, j] > 0.5)
            neg_mask = lab_present_mask & (Y[:, j] < 0.5)

            pos_group = x_i[pos_mask]
            neg_group = x_i[neg_mask]

            if len(pos_group) < 5 or len(neg_group) < 5:
                continue

            t, p = stats.ttest_ind(pos_group, neg_group, equal_var=False)
            if np.isfinite(t) and np.isfinite(p):
                tstat[i, j] = abs(float(t))
                pval[i, j] = float(p)

        # Progress
        lab_name = ITEMID_TO_NAME[LAB_ITEMIDS[i]]
        n_sig = (pval[i] < 0.05).sum()
        print(f"    Lab {i:2d} ({lab_name:12s}): {n_present:5d} present, "
              f"{n_sig:3d} drugs p<0.05, max |t|={tstat[i].max():.2f}")

    return tstat, pval


def build_mask(pval, tstat, p_threshold=0.05, top_k=None, bonferroni=False):
    """Build edge mask from p-values and/or top-K.

    Args:
        p_threshold: significance level (applied after optional Bonferroni correction)
        top_k: if set, keep at least top_k edges per lab by |t| (regardless of p-value)
        bonferroni: if True, apply Bonferroni correction across 18 * num_drugs tests
    """
    num_labs, num_drugs = pval.shape
    n_tests = num_labs * num_drugs

    threshold = p_threshold
    if bonferroni:
        threshold = p_threshold / n_tests
        print(f"  Bonferroni correction: p_threshold {p_threshold} → {threshold:.2e}")

    mask = pval < threshold  # (18, num_drugs) bool

    if top_k is not None:
        # For each lab, ensure at least top_k edges even if not significant
        for i in range(num_labs):
            top_indices = np.argsort(tstat[i])[::-1][:top_k]
            mask[i, top_indices] = True
        print(f"  top_k={top_k}: ensures >={top_k} edges per lab")

    return mask


def main():
    parser = argparse.ArgumentParser(
        description="Compute statistical edge mask for lab-drug cross-attention"
    )
    parser.add_argument("--p_threshold", type=float, default=0.05,
                        help="p-value threshold for significance (default: 0.05)")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Keep at least top-K edges per lab by |t| (default: 20)")
    parser.add_argument("--bonferroni", action="store_true",
                        help="Apply Bonferroni multiple-test correction")
    parser.add_argument("--num_drugs", type=int, default=130,
                        help="Number of drugs in vocabulary (default: 130)")
    parser.add_argument("--output_dir", type=str, default=str(DATA_DIR),
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Statistical Edge Mask Computation ===")
    print(f"Settings: p_threshold={args.p_threshold}, top_k={args.top_k}, "
          f"bonferroni={args.bonferroni}, num_drugs={args.num_drugs}")
    print()

    print("[1] Loading data...")
    records, cohort, lab_data = load_data()
    print(f"  {len(records)} patients loaded")

    print("[2] Getting training split...")
    train_records = get_train_records(records, cohort)

    print("[3] Building lab-drug matrix from training examples...")
    X, Y, present = build_lab_drug_matrix(train_records, lab_data, args.num_drugs)
    print(f"  X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"  Drug prescription rate: {Y.mean(axis=0).mean():.4f} avg per drug")
    print(f"  Lab presence rate: {present.mean(axis=0).mean():.4f} avg per lab")

    print("\n[4] Computing Welch's t-tests (18 labs × 130 drugs)...")
    tstat, pval = compute_ttest(X, Y, present)

    print("\n[5] Building edge mask...")
    mask = build_mask(pval, tstat,
                      p_threshold=args.p_threshold,
                      top_k=args.top_k,
                      bonferroni=args.bonferroni)

    # Summary statistics
    edges_per_lab = mask.sum(axis=1)
    total_edges = mask.sum()
    density = total_edges / (NUM_LABS * args.num_drugs)

    print(f"\n=== Mask Summary ===")
    print(f"Total edges: {total_edges} / {NUM_LABS * args.num_drugs} "
          f"({density*100:.1f}% density)")
    print(f"Edges per lab:")
    for i in range(NUM_LABS):
        lab_name = ITEMID_TO_NAME[LAB_ITEMIDS[i]]
        top_drugs = np.where(mask[i])[0]
        print(f"  [{i:2d}] {lab_name:12s}: {edges_per_lab[i]:3d} edges | "
              f"max |t|={tstat[i].max():.2f} | "
              f"example drug indices: {top_drugs[:5].tolist()}")

    # Save
    mask_path = output_dir / "stat_edge_mask.npy"
    tstat_path = output_dir / "stat_edge_tstat.npy"
    pval_path = output_dir / "stat_edge_pval.npy"

    np.save(mask_path, mask.astype(np.bool_))
    np.save(tstat_path, tstat.astype(np.float32))
    np.save(pval_path, pval.astype(np.float32))

    print(f"\nSaved:")
    print(f"  {mask_path}  — shape {mask.shape}, dtype bool")
    print(f"  {tstat_path} — shape {tstat.shape}, dtype float32")
    print(f"  {pval_path}  — shape {pval.shape}, dtype float32")


if __name__ == "__main__":
    main()
