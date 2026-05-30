"""
center_note_embeddings.py — Run 15 preprocessing step.

Compute the global mean of ClinicalBERT note embeddings from the TRAINING SET ONLY
and save it alongside the original embeddings file.

Why: ClinicalBERT embeds all clinical notes on a high-dimensional cone.
  Mean cosine similarity between note embeddings = 0.9536.
  Global mean vector norm: 13.16 ≈ individual note norm: 13.18.
  This means ~99% of every note embedding is shared direction — no signal.
  Subtracting the global mean drops cosine similarity to near 0.

What this script does:
  1. Load note_embeddings_mimic3.pkl
  2. Load records_final.pkl + cohort_mimic3.pkl to get train split hadm_ids
  3. Compute mean over TRAINING set note embeddings only (no test leakage)
  4. Save note_global_mean.pkl (shape: 768,) to data/processed/

Usage:
    python src/scripts/center_note_embeddings.py
    python src/scripts/center_note_embeddings.py --data_dir data/processed --output_dir data/processed

The output file is then loaded in the Kaggle notebook and set on the model:
    note_global_mean = torch.tensor(np.load(...)).to(device)
    model.fusion.note_global_mean = note_global_mean
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_train_hadm_ids(records, cohort):
    """Replicate the 2/3 : 1/6 : 1/6 sequential split to get training hadm_ids."""
    # Records is list of patients, each patient is list of visits.
    # Visit format: (diags, procs, drugs, hadm_id) for 4-tuples.
    # Cohort may have split indices.

    # FIX-CENTER-SPLIT (Agent 6 finding): training uses cohort['split_indices']['train']
    # (10,002 hadm_ids), but the old fallback used a sequential first-2/3 split
    # (10,489 hadm_ids) — only ~70% overlap. ~30% of "train" patients in the saved
    # mean were actually val/test patients. Now prefer the real split indices.
    if isinstance(cohort, dict):
        if "split_indices" in cohort and isinstance(cohort["split_indices"], dict) and "train" in cohort["split_indices"]:
            train_idxs = list(cohort["split_indices"]["train"])
            print(f"  [FIX-CENTER-SPLIT] Using cohort['split_indices']['train'] ({len(train_idxs)} patients)")
        elif "train_indices" in cohort:
            train_idxs = cohort["train_indices"]
        else:
            n = len(records)
            train_end = int(n * 2 / 3)
            train_idxs = list(range(train_end))
            print(f"  [WARN] cohort lacks split_indices; falling back to sequential 2/3 split")
    elif isinstance(cohort, (list, tuple)):
        n = len(records)
        train_end = int(n * 2 / 3)
        train_idxs = list(range(train_end))
    else:
        n = len(records)
        train_end = int(n * 2 / 3)
        train_idxs = list(range(train_end))

    # Collect all hadm_ids from training patients
    train_hadm_ids = set()
    for i in train_idxs:
        patient = records[i]
        for visit in patient:
            if len(visit) > 3:
                train_hadm_ids.add(int(visit[3]))

    return train_hadm_ids


def main(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading note embeddings...")
    note_data = load_pkl(data_dir / "note_embeddings_mimic3.pkl")
    hadm_ids = note_data["hadm_ids"]
    embeddings = note_data["embeddings"]  # (N, 768) numpy array
    has_note = note_data["has_note"]

    print(f"  Total note embeddings: {len(hadm_ids)}")
    print(f"  Embedding shape: {np.array(embeddings).shape}")

    # Cosine similarity diagnostic (before centering)
    E = np.array(embeddings, dtype=np.float32)
    E_valid = E[np.array(has_note, dtype=bool)] if len(has_note) > 0 else E
    norms = np.linalg.norm(E_valid, axis=1, keepdims=True)
    E_normed = E_valid / (norms + 1e-8)
    # Sample 500 random pairs to estimate mean cosine similarity
    rng = np.random.default_rng(42)
    n_sample = min(500, len(E_valid))
    idxs = rng.choice(len(E_valid), size=n_sample, replace=False)
    E_sample = E_normed[idxs]
    cos_mat = E_sample @ E_sample.T
    upper_tri = cos_mat[np.triu_indices(n_sample, k=1)]
    print(f"  Mean cosine sim BEFORE centering: {upper_tri.mean():.4f}")

    # Load records + cohort for train split
    print("Loading records + cohort for train split...")
    try:
        records = load_pkl(data_dir / "records_final.pkl")
        cohort = load_pkl(data_dir / "cohort_mimic3.pkl")
        train_hadm_ids = get_train_hadm_ids(records, cohort)
        print(f"  Training hadm_ids: {len(train_hadm_ids)}")
    except FileNotFoundError as e:
        print(f"  WARNING: Could not load records/cohort ({e}). Using ALL embeddings for mean.")
        train_hadm_ids = None

    # Compute global mean from training set
    if train_hadm_ids is not None:
        train_mask = np.array([
            (has_note[i] and int(hadm_ids[i]) in train_hadm_ids)
            for i in range(len(hadm_ids))
        ], dtype=bool)
        train_embeds = E[train_mask]
        print(f"  Training notes used for mean: {train_mask.sum()}")
    else:
        # Fallback: use all embeddings with has_note=True
        valid_mask = np.array(has_note, dtype=bool)
        train_embeds = E[valid_mask]
        print(f"  All valid notes used for mean: {valid_mask.sum()}")

    global_mean = train_embeds.mean(axis=0)  # (768,)
    print(f"  Global mean norm: {np.linalg.norm(global_mean):.4f}")
    print(f"  Global mean shape: {global_mean.shape}")

    # Verify centering works
    E_centered = E_valid - global_mean[np.newaxis, :]
    norms_c = np.linalg.norm(E_centered, axis=1, keepdims=True)
    E_c_normed = E_centered / (norms_c + 1e-8)
    E_c_sample = E_c_normed[idxs]
    cos_mat_c = E_c_sample @ E_c_sample.T
    upper_tri_c = cos_mat_c[np.triu_indices(n_sample, k=1)]
    print(f"  Mean cosine sim AFTER centering: {upper_tri_c.mean():.4f}")

    # Save
    # BUG-B FIX: save with cohort-specific suffix so MIMIC-III and MIMIC-IV
    # note means are never confused. train.py prefers note_global_mean_{cohort_tag}.npy
    # and falls back to note_global_mean.npy for legacy MIMIC-III compatibility.
    cohort_tag = getattr(args, "cohort_tag", None) or "mimic3"
    output_path = output_dir / f"note_global_mean_{cohort_tag}.npy"
    np.save(output_path, global_mean)
    print(f"\nSaved: {output_path}")
    # Also save legacy name for MIMIC-III to keep backward compatibility
    if cohort_tag == "mimic3":
        legacy_path = output_dir / "note_global_mean.npy"
        np.save(legacy_path, global_mean)
        print(f"Saved legacy copy: {legacy_path}")
    print(f"  Shape: {global_mean.shape}, dtype: {global_mean.dtype}")
    print(f"\nTo use in notebook:")
    print(f"  note_global_mean = torch.from_numpy(np.load('{output_path}')).to(device)")
    print(f"  model.fusion.note_global_mean = note_global_mean")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--cohort_tag", type=str, default="mimic3",
                        choices=["mimic3", "mimic4", "mimic4_full"],
                        help="Cohort tag — determines output filename suffix")
    args = parser.parse_args()
    main(args)
