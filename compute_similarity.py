"""Build cross-patient retrieval index from Phase 1 embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np

from split_protocol import compute_split_indices


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _parse_optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s and s.lstrip("+-").isdigit():
            return int(s)
    return None


def _parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _is_under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _load_pickle_trusted(path: Path, trusted_roots: list[Path], allow_unsafe: bool):
    resolved = path.expanduser().resolve()
    trusted = any(_is_under_root(resolved, root) for root in trusted_roots)
    if not trusted and not allow_unsafe:
        roots_str = ", ".join(str(r.resolve()) for r in trusted_roots)
        raise RuntimeError(
            f"Refusing to load untrusted pickle path: {resolved}. "
            f"Trusted roots: {roots_str}. "
            "Set MIRROR_ALLOW_UNSAFE_DESERIALIZATION=1 or pass "
            "--allow_unsafe_deserialization only for trusted artifacts."
        )
    if not trusted:
        print(f"WARNING: loading untrusted pickle due to explicit override: {resolved}")

    with open(resolved, "rb") as f:
        return pickle.load(f)


def _resolve_manifest_path(
    embed_dir: Path,
    split_seed_used: object,
    explicit_manifest_path: str | None,
) -> Path | None:
    if explicit_manifest_path:
        path = Path(explicit_manifest_path)
        return path if path.exists() else None

    split_seed_num = _parse_optional_int(split_seed_used)
    if split_seed_num is not None:
        preferred = embed_dir / f"artifacts_manifest_seed{split_seed_num}.json"
        if preferred.exists():
            return preferred

    candidates = sorted(embed_dir.glob("artifacts_manifest_seed*.json"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def validate_manifest_hashes(
    manifest: dict,
    embed_dir: Path,
    strict: bool = True,
) -> dict[str, str]:
    files = manifest.get("files")
    if not isinstance(files, dict):
        if strict:
            raise ValueError("Manifest missing 'files' mapping.")
        return {}

    verified: dict[str, str] = {}
    for name, info in files.items():
        if not isinstance(name, str):
            continue
        expected = info.get("sha256") if isinstance(info, dict) else None
        if not expected:
            continue
        path = embed_dir / name
        if not path.exists():
            if strict:
                raise FileNotFoundError(f"Manifest expects missing artifact: {path}")
            continue
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"Manifest hash mismatch for {name}: expected {expected}, got {actual}"
            )
        verified[name] = actual
    return verified


def fit_zca_whitener(train_embeds: np.ndarray, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_embeds.ndim != 2:
        raise ValueError(f"Expected 2D train embeddings, got shape={train_embeds.shape}")
    mean = train_embeds.mean(axis=0, keepdims=True)
    centered = train_embeds - mean
    denom = max(centered.shape[0] - 1, 1)
    cov = (centered.T @ centered) / float(denom)
    u, s, _ = np.linalg.svd(cov, full_matrices=False)
    inv_sqrt = 1.0 / np.sqrt(s + float(eps))
    # (U * inv_sqrt) @ U^T is numerically stable and avoids explicit diag matrix.
    zca = (u * inv_sqrt[np.newaxis, :]) @ u.T
    return mean.astype(np.float32), zca.astype(np.float32), s.astype(np.float32)


def apply_whitener(embeds: np.ndarray, mean: np.ndarray, zca: np.ndarray) -> np.ndarray:
    if embeds.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape={embeds.shape}")
    return ((embeds - mean) @ zca.T).astype(np.float32)


def cosine_sim_batched(query: np.ndarray, keys: np.ndarray, batch_size: int = 512) -> np.ndarray:
    eps = 1e-8
    query_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + eps)
    keys_norm = keys / (np.linalg.norm(keys, axis=1, keepdims=True) + eps)

    n_q = query_norm.shape[0]
    sims = np.zeros((n_q, keys_norm.shape[0]), dtype=np.float32)
    for start in range(0, n_q, batch_size):
        end = min(start + batch_size, n_q)
        sims[start:end] = (query_norm[start:end] @ keys_norm.T).astype(np.float32)
    return sims


def jaccard_sim_batched(query_bool: np.ndarray, keys_bool: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Compute Jaccard similarity between boolean vectors: |A ∩ B| / |A ∪ B|."""
    n_q = query_bool.shape[0]
    n_k = keys_bool.shape[0]
    sims = np.zeros((n_q, n_k), dtype=np.float32)

    # Pre-compute sums (cardinality) for union calculation
    # J = Intersection / (SumA + SumB - Intersection)
    query_sums = query_bool.sum(axis=1).astype(np.float32)
    keys_sums = keys_bool.sum(axis=1).astype(np.float32)

    for start in range(0, n_q, batch_size):
        end = min(start + batch_size, n_q)
        # Intersection is the dot product of boolean vectors
        intersection = (query_bool[start:end].astype(np.float32) @ keys_bool.T.astype(np.float32))
        
        # Union = |A| + |B| - |A ∩ B|
        union = query_sums[start:end, np.newaxis] + keys_sums[np.newaxis, :] - intersection
        
        # Handle zero union (similarity = 0)
        valid = union > 0
        sims[start:end][valid] = intersection[valid] / union[valid]
        
    return sims


def build_examples(split_records: list, split_name: str) -> list[dict]:
    examples = []
    for pidx, patient in enumerate(split_records):
        for t in range(1, len(patient)):
            hadm_id = int(patient[t][3]) if len(patient[t]) > 3 else -1
            examples.append({
                "local_patient_idx": pidx,
                "target_visit_idx": t,
                "hadm_id": hadm_id,
            })
    print(f"  {split_name}: {len(examples)} examples from {len(split_records)} patients")
    return examples


def build_retrieval_for_split(
    query_embeds: np.ndarray,
    query_examples: list[dict],
    train_embeds: np.ndarray,
    train_labels: np.ndarray,
    train_patient_to_rows: dict[int, list[int]],
    top_k: int,
    exclude_mode: str,
    split_name: str,
    train_examples: list[dict] | None = None,  # F10: provenance of the retrieval pool
    # BUG-FIX (C3): Note similarity blending
    query_note_embeds: np.ndarray | None = None,
    train_note_embeds: np.ndarray | None = None,
    note_sim_weight: float = 0.0,
    # Phase 5.4: Lab Jaccard similarity blending
    query_lab_bools: np.ndarray | None = None,
    train_lab_bools: np.ndarray | None = None,
    lab_sim_weight: float = 0.0,
) -> dict[int, dict[str, np.ndarray]]:
    print(f"  {split_name}: computing {len(query_embeds)} x {len(train_embeds)} similarities...")
    sims = cosine_sim_batched(query_embeds, train_embeds)
    
    # EHR similarity weight
    # Phase 5.0 Hardening: Implement soft-normalization if weights > 1.0
    total_w = 1.0 + note_sim_weight + lab_sim_weight
    if (note_sim_weight + lab_sim_weight) > 1.0:
        norm_factor = 1.0 / (1.0 + note_sim_weight + lab_sim_weight)
        w_ehr = 1.0 * norm_factor
        w_note = note_sim_weight * norm_factor
        w_lab = lab_sim_weight * norm_factor
        print(f"    [Hardening] Weights sum to {note_sim_weight + lab_sim_weight + 1.0:.2f}. "
              f"Normalizing to EHR={w_ehr:.2f}, Note={w_note:.2f}, Lab={w_lab:.2f}")
    else:
        w_ehr = 1.0 - note_sim_weight - lab_sim_weight
        w_note = note_sim_weight
        w_lab = lab_sim_weight

    sims = w_ehr * sims

    # C3: blend note similarity when available
    if query_note_embeds is not None and train_note_embeds is not None and w_note > 0:
        sims_note = cosine_sim_batched(query_note_embeds, train_note_embeds)
        sims += w_note * sims_note
        print(f"    Blended note similarity (weight={w_note:.2f})")
        
    # Phase 5.4: blend lab similarity when available
    if query_lab_bools is not None and train_lab_bools is not None and w_lab > 0:
        sims_lab = jaccard_sim_batched(query_lab_bools, train_lab_bools)
        sims += w_lab * sims_lab
        print(f"    Blended lab similarity (weight={w_lab:.2f})")

    # F10 (Run 23): pre-compute per-row lookups so each entry can record which
    # training visits its neighbours came from. Enables HI-DR-style own-vs-cross
    # patient analysis downstream. Zero model-side cost; all audit.
    train_hadm_arr: np.ndarray | None = None
    train_pid_arr: np.ndarray | None = None
    if train_examples is not None:
        train_hadm_arr = np.asarray(
            [int(ex.get("hadm_id", -1)) for ex in train_examples], dtype=np.int64
        )
        train_pid_arr = np.asarray(
            [int(ex.get("local_patient_idx", -1)) for ex in train_examples], dtype=np.int64
        )

    result: dict[int, dict[str, np.ndarray]] = {}
    skipped = 0
    k_eff = min(top_k, train_embeds.shape[0])
    if k_eff <= 0:
        return result

    for row_i, ex in enumerate(query_examples):
        hadm_id = ex["hadm_id"]
        if hadm_id == -1:
            skipped += 1
            continue

        sim_row = sims[row_i].copy()
        if exclude_mode == "own_patient":
            own_rows = train_patient_to_rows.get(ex["local_patient_idx"], [])
            sim_row[own_rows] = -2.0
        elif exclude_mode == "self" and split_name == "train" and row_i < sim_row.shape[0]:
            sim_row[row_i] = -2.0

        top_indices = np.argsort(sim_row)[-k_eff:][::-1]
        entry: dict[str, np.ndarray] = {
            "similar_reprs": train_embeds[top_indices].copy(),
            "similar_multihots": train_labels[top_indices].copy(),
            "scores": sim_row[top_indices].copy(),
        }
        if train_hadm_arr is not None:
            entry["neighbor_hadm_ids"] = train_hadm_arr[top_indices].copy()
            entry["neighbor_patient_ids"] = train_pid_arr[top_indices].copy()
            entry["query_patient_id"] = np.int64(ex.get("local_patient_idx", -1))
        result[int(hadm_id)] = entry

    if skipped:
        print(f"    Skipped {skipped} examples with missing hadm_id")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cross-patient retrieval similarity index")
    parser.add_argument("--embeddings_dir", type=str, required=True,
                        help="Directory with patient_embeddings_{train,val,test}.pkl")
    parser.add_argument("--records", type=str, required=True, help="Path to records_*.pkl")
    parser.add_argument("--cohort", type=str, required=True, help="Path to cohort_*.pkl")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--output", type=str, default=None,
                        help="Output .pkl path (default: embeddings_dir/retrieval_index_top{k}.pkl)")
    parser.add_argument("--metadata_output", type=str, default=None,
                        help="Optional output path for provenance .json sidecar")
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional path to artifacts_manifest_seed*.json produced by extraction.",
    )
    parser.add_argument(
        "--enforce_manifest",
        action="store_true",
        help="Fail if manifest is missing or hashes do not match embedding artifacts.",
    )
    # BUG-FIX (C3): Note similarity blending for retrieval k-NN.
    # AMO: PSA (retrieval) is 1.7x more important than CAMO. EHR-only
    # similarity misses clinical narrative similarity — patients with
    # different ICD codes but similar notes may need similar drugs.
    parser.add_argument(
        "--note_embeddings",
        type=str,
        default=None,
        help="Path to note_embeddings_mimic3.pkl for note similarity blending.",
    )
    parser.add_argument(
        "--note_sim_weight",
        type=float,
        default=0.3,
        help="Weight for note similarity in blended retrieval (default: 0.3).",
    )
    parser.add_argument(
        "--lab_sim_weight",
        type=float,
        default=0.0,
        help="Weight for lab Jaccard similarity in blended retrieval (default: 0.0).",
    )
    parser.add_argument(
        "--lab_embeddings",
        type=str,
        default=None,
        help="Path to lab_vectors_200labs.pkl for accurate lab similarity blending.",
    )
    # F2 (Run 23): whitening is now DEFAULT ON. Run 22 shipped whitening code but
    # the flag defaulted False, so no run in the cap 5/10/15/20 sweep was actually
    # whitened — cosine_sim.std stayed near 0 and h4 never beat the drug-frequency
    # prior. --no_whiten stays available as a diagnostic / ablation control.
    whiten_group = parser.add_mutually_exclusive_group()
    whiten_group.add_argument(
        "--whiten",
        dest="whiten",
        action="store_true",
        default=True,
        help="Apply train-fit ZCA whitening (default: ON for Run 23).",
    )
    whiten_group.add_argument(
        "--no_whiten",
        dest="whiten",
        action="store_false",
        help="Disable ZCA whitening (diagnostic only; Run 22 behaviour).",
    )
    parser.add_argument(
        "--whitening_eps",
        type=float,
        default=1e-5,
        help="Stability epsilon for whitening (default: 1e-5).",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="hidr_vita",
        choices=["cohort", "permutation", "sequential", "hidr_vita"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--require_cohort_split",
        action="store_true",
        help="Fail fast if split_mode=cohort and split_indices are missing/invalid.",
    )
    parser.add_argument(
        "--train_exclude_mode",
        type=str,
        default="own_patient",
        choices=["none", "self", "own_patient"],
        help="Neighbor exclusion strategy for train split queries.",
    )
    parser.add_argument(
        "--allow_unsafe_deserialization",
        action="store_true",
        help="Allow loading pickle files outside trusted roots (dangerous; trusted artifacts only).",
    )
    parser.add_argument(
        "--trusted_root",
        action="append",
        default=None,
        help=(
            "Optional trusted root directory (repeatable). If omitted, defaults to the "
            "repository root containing this script."
        ),
    )
    args = parser.parse_args()

    embed_dir = Path(args.embeddings_dir)
    k = int(args.top_k)
    output_path = Path(args.output) if args.output else embed_dir / f"retrieval_index_top{k}.pkl"
    metadata_path = (
        Path(args.metadata_output)
        if args.metadata_output
        else output_path.with_suffix(output_path.suffix + ".meta.json")
    )

    print(f"\n{'=' * 60}")
    print(f"Phase 2: Cross-Patient Retrieval Index (top-{k})")
    print(f"{'=' * 60}\n")

    records_path = Path(args.records)
    cohort_path = Path(args.cohort)
    repo_root = Path(__file__).resolve().parents[1]
    if args.trusted_root:
        trusted_roots = [Path(p).expanduser().resolve() for p in args.trusted_root]
    else:
        trusted_roots = [repo_root]

    allow_unsafe_deserialization = bool(args.allow_unsafe_deserialization) or _parse_bool_env(
        "MIRROR_ALLOW_UNSAFE_DESERIALIZATION",
        default=False,
    )
    if not allow_unsafe_deserialization:
        roots_str = ", ".join(str(r) for r in trusted_roots)
        print(f"Trusted roots for pickle loading: {roots_str}")

    records = _load_pickle_trusted(records_path, trusted_roots, allow_unsafe_deserialization)
    cohort = _load_pickle_trusted(cohort_path, trusted_roots, allow_unsafe_deserialization)

    split = compute_split_indices(
        num_records=len(records),
        cohort=cohort,
        split_mode=args.split_mode,
        seed=args.seed,
        require_cohort_indices=args.require_cohort_split,
    )
    train_records = [records[i] for i in split.train_idx]
    val_records = [records[i] for i in split.val_idx]
    test_records = [records[i] for i in split.test_idx]
    print(
        f"Split ({split.split_source}, seed={split.split_seed_used}): "
        f"train={len(train_records)}, val={len(val_records)}, test={len(test_records)}"
    )

    print("\nBuilding example lists...")
    train_examples = build_examples(train_records, "train")
    val_examples = build_examples(val_records, "val")
    test_examples = build_examples(test_records, "test")

    train_patient_to_rows: dict[int, list[int]] = {}
    for row_i, ex in enumerate(train_examples):
        train_patient_to_rows.setdefault(ex["local_patient_idx"], []).append(row_i)

    print("\nLoading embeddings...")
    train_emb_path = embed_dir / "patient_embeddings_train.pkl"
    val_emb_path = embed_dir / "patient_embeddings_val.pkl"
    test_emb_path = embed_dir / "patient_embeddings_test.pkl"
    for p in (train_emb_path, val_emb_path, test_emb_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing embedding file: {p}")

    train_pkg = _load_pickle_trusted(train_emb_path, trusted_roots, allow_unsafe_deserialization)
    val_pkg = _load_pickle_trusted(val_emb_path, trusted_roots, allow_unsafe_deserialization)
    test_pkg = _load_pickle_trusted(test_emb_path, trusted_roots, allow_unsafe_deserialization)

    manifest_path = _resolve_manifest_path(embed_dir, split.split_seed_used, args.manifest)
    manifest_verified_hashes: dict[str, str] = {}
    if manifest_path is not None:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest_verified_hashes = validate_manifest_hashes(manifest, embed_dir, strict=True)
        print(f"Validated manifest hashes: {manifest_path}")
    elif args.enforce_manifest:
        raise FileNotFoundError(
            "Manifest enforcement requested, but no artifacts_manifest_seed*.json was found."
        )
    else:
        print("WARNING: No manifest found; continuing without hash verification.")

    train_embeds = np.asarray(train_pkg["embeddings"], dtype=np.float32)
    train_labels = np.asarray(train_pkg["labels"], dtype=np.float32)
    val_embeds = np.asarray(val_pkg["embeddings"], dtype=np.float32)
    test_embeds = np.asarray(test_pkg["embeddings"], dtype=np.float32)

    print(f"  train_embeds: {train_embeds.shape}")
    print(f"  val_embeds:   {val_embeds.shape}")
    print(f"  test_embeds:  {test_embeds.shape}")

    if len(train_embeds) != len(train_examples):
        raise ValueError(
            f"Train size mismatch: {len(train_embeds)} embeddings vs {len(train_examples)} examples"
        )
    if len(val_embeds) != len(val_examples):
        raise ValueError(
            f"Val size mismatch: {len(val_embeds)} embeddings vs {len(val_examples)} examples"
        )
    if len(test_embeds) != len(test_examples):
        raise ValueError(
            f"Test size mismatch: {len(test_embeds)} embeddings vs {len(test_examples)} examples"
        )

    # C2 fix: verify element-wise order match between the embeddings produced by
    # Phase 1 and the example list rebuilt here. Length equality alone can hide
    # a silent reorder that mispairs every retrieval. Phase 1 notebooks now save
    # `hadm_ids` alongside embeddings/labels.
    def _check_hadm_order(pkg_key: str, pkg: dict, examples: list[dict]) -> None:
        pkg_hadm = pkg.get("hadm_ids")
        if pkg_hadm is None:
            print(
                f"  WARNING: {pkg_key} embeddings have no 'hadm_ids' field. "
                f"Cannot verify ordering — rerun Phase 1 with the updated notebook."
            )
            return
        pkg_hadm_arr = np.asarray(pkg_hadm, dtype=np.int64)
        expected = np.asarray([int(ex["hadm_id"]) for ex in examples], dtype=np.int64)
        if pkg_hadm_arr.shape != expected.shape:
            raise ValueError(
                f"{pkg_key}: hadm_ids shape {pkg_hadm_arr.shape} != examples shape {expected.shape}"
            )
        if not np.array_equal(pkg_hadm_arr, expected):
            n_mismatch = int((pkg_hadm_arr != expected).sum())
            first_bad = int(np.argmax(pkg_hadm_arr != expected))
            raise ValueError(
                f"{pkg_key}: hadm_id order mismatch at {n_mismatch} positions. "
                f"First mismatch at index {first_bad}: "
                f"pkg={pkg_hadm_arr[first_bad]} vs expected={expected[first_bad]}. "
                f"Phase 1 embedding order does not match MIRRORDataset order — "
                f"retrievals would be mispaired. Rerun Phase 1."
            )

        # T0.2: optional patient_ids integrity check.
        pkg_patient_ids = pkg.get("patient_ids")
        if pkg_patient_ids is not None:
            pkg_pid_arr = np.asarray(pkg_patient_ids, dtype=np.int64)
            expected_pid = np.asarray(
                [int(ex["local_patient_idx"]) for ex in examples],
                dtype=np.int64,
            )
            if pkg_pid_arr.shape != expected_pid.shape:
                raise ValueError(
                    f"{pkg_key}: patient_ids shape {pkg_pid_arr.shape} != expected {expected_pid.shape}"
                )
            if not np.array_equal(pkg_pid_arr, expected_pid):
                n_mismatch = int((pkg_pid_arr != expected_pid).sum())
                first_bad = int(np.argmax(pkg_pid_arr != expected_pid))
                raise ValueError(
                    f"{pkg_key}: patient_ids order mismatch at {n_mismatch} positions. "
                    f"First mismatch at index {first_bad}: "
                    f"pkg={pkg_pid_arr[first_bad]} vs expected={expected_pid[first_bad]}."
                )

    _check_hadm_order("train", train_pkg, train_examples)
    _check_hadm_order("val", val_pkg, val_examples)
    _check_hadm_order("test", test_pkg, test_examples)

    whitening_info = {
        "enabled": bool(args.whiten),
        "eps": float(args.whitening_eps),
    }
    if args.whiten:
        train_mean, zca, eigvals = fit_zca_whitener(train_embeds, eps=float(args.whitening_eps))
        train_embeds = apply_whitener(train_embeds, train_mean, zca)
        val_embeds = apply_whitener(val_embeds, train_mean, zca)
        test_embeds = apply_whitener(test_embeds, train_mean, zca)
        whitening_info.update(
            {
                "train_mean_norm": float(np.linalg.norm(train_mean)),
                "eig_min": float(np.min(eigvals)),
                "eig_max": float(np.max(eigvals)),
            }
        )
        print(
            "Applied whitening: "
            f"eps={args.whitening_eps}, eig_min={whitening_info['eig_min']:.4e}, "
            f"eig_max={whitening_info['eig_max']:.4e}"
        )

    # BUG-FIX (C3): Load note embeddings for blended retrieval similarity.
    # When provided, sim_total = (1 - w) * sim_EHR + w * sim_note, where w = note_sim_weight.
    # This captures clinical narrative similarity that ICD-code-only EHR embeddings miss.
    note_train_embeds = note_val_embeds = note_test_embeds = None
    note_sim_weight = float(args.note_sim_weight)
    if args.note_embeddings:
        note_emb_path = Path(args.note_embeddings)
        if note_emb_path.exists():
            print(f"\nLoading note embeddings for blended retrieval (weight={note_sim_weight:.2f})...")
            note_pkg = _load_pickle_trusted(note_emb_path, trusted_roots, allow_unsafe_deserialization)

            # Build hadm_id → note_embedding lookup.
            # Format A: structured dict with 'embeddings' (N, 768) and 'hadm_ids' (N,)
            # Format B: flat dict {hadm_id: embedding}
            note_lookup: dict[int, np.ndarray] = {}
            if isinstance(note_pkg, dict) and "embeddings" in note_pkg and "hadm_ids" in note_pkg:
                # Format A (actual format of note_embeddings_mimic3.pkl)
                all_embeds = np.asarray(note_pkg["embeddings"], dtype=np.float32)
                all_hadm_ids = np.asarray(note_pkg["hadm_ids"], dtype=np.int64)
                for idx, hid in enumerate(all_hadm_ids):
                    note_lookup[int(hid)] = all_embeds[idx]
                print(f"  Loaded {len(note_lookup)} note embeddings (structured format)")
            elif isinstance(note_pkg, dict):
                # Format B (flat dict)
                for k, v in note_pkg.items():
                    if isinstance(k, (int, np.integer)):
                        note_lookup[int(k)] = np.asarray(v, dtype=np.float32)
                print(f"  Loaded {len(note_lookup)} note embeddings (flat dict format)")
            else:
                print(f"  WARNING: unrecognized note embeddings format ({type(note_pkg)}), skipping")

            if note_lookup:
                dim = 768
                def _align_note_embeds(examples: list[dict], lookup: dict) -> np.ndarray:
                    embeds = np.zeros((len(examples), dim), dtype=np.float32)
                    n_found = 0
                    for i, ex in enumerate(examples):
                        hadm_id = ex.get("hadm_id", -1)
                        if hadm_id in lookup:
                            embeds[i] = lookup[hadm_id][:dim]
                            n_found += 1
                    print(f"    Aligned {n_found}/{len(examples)} note embeddings")
                    return embeds
                note_train_embeds = _align_note_embeds(train_examples, note_lookup)
                note_val_embeds = _align_note_embeds(val_examples, note_lookup)
                note_test_embeds = _align_note_embeds(test_examples, note_lookup)
        else:
            print(f"  WARNING: note embeddings file not found: {note_emb_path}")


    # Phase 5.4: Extract lab features and compute abnormal bitmasks
    def _extract_abnormal_labs(records_list: list, examples: list[dict], lab_data_dict: dict | None = None, threshold: float = 2.0) -> np.ndarray:
        # Phase 5.0 Hardening: Use lab_data_dict (e.g. lab_vectors_200labs.pkl) if provided
        # otherwise fall back to scanning records_list.
        dim = 0
        lab_key = "lab_vectors"
        
        if lab_data_dict:
            # Detect key (lab_vectors or lab_vectors_72d)
            lab_key = "lab_vectors_72d" if "lab_vectors_72d" in lab_data_dict else "lab_vectors"
            if lab_key in lab_data_dict:
                dim = lab_data_dict[lab_key].shape[1]
                print(f"    [Hardening] Using lab data dict (dim={dim}, key={lab_key})")
        
        if dim == 0:
            # Fallback to record scan
            for patient in records_list:
                for t in range(1, len(patient)):
                    if len(patient[t]) > 4:
                        dim = len(patient[t][4])
                        break
                if dim > 0: break
        
        if dim == 0:
            print("    WARNING: No lab features found in records or lab_data_dict.")
            return np.zeros((len(examples), 1), dtype=bool)
            
        lab_bools = np.zeros((len(examples), dim), dtype=bool)
        n_found = 0
        for i, ex in enumerate(examples):
            hadm_id = ex.get("hadm_id", -1)
            # Try dict lookup first (Phase 5.0 Hardening)
            if lab_data_dict and hadm_id in lab_data_dict:
                lab_vals = np.asarray(lab_data_dict[hadm_id][lab_key], dtype=np.float32)
                lab_bools[i] = np.abs(lab_vals) > threshold
                n_found += 1
                continue
                
            # Fallback to record list
            p_idx = ex["local_patient_idx"]
            t = ex["target_visit_idx"]
            patient = records_list[p_idx]
            if len(patient[t]) > 4:
                # Abnormal = absolute z-score > threshold
                lab_vals = np.asarray(patient[t][4], dtype=np.float32)
                lab_bools[i] = np.abs(lab_vals) > threshold
                n_found += 1
        
        print(f"    Extracted lab bitmasks for {n_found}/{len(examples)} examples.")
        return lab_bools

    # Phase 5.4: Extract lab features and compute abnormal bitmasks
    lab_sim_weight = float(args.lab_sim_weight)
    lab_train_bools = lab_val_bools = lab_test_bools = None
    if lab_sim_weight > 0:
        print(f"\nExtracting abnormal lab bitmasks (threshold=2.0, weight={lab_sim_weight:.2f})...")
        
        lab_data_dict = None
        if args.lab_embeddings:
            lab_emb_path = Path(args.lab_embeddings)
            if lab_emb_path.exists():
                print(f"  Loading lab embeddings from {lab_emb_path} ...")
                lab_data_dict = _load_pickle_trusted(lab_emb_path, trusted_roots, allow_unsafe_deserialization)
            else:
                print(f"  WARNING: lab embeddings file not found: {lab_emb_path}")

        lab_train_bools = _extract_abnormal_labs(train_records, train_examples, lab_data_dict=lab_data_dict)
        lab_val_bools = _extract_abnormal_labs(val_records, val_examples, lab_data_dict=lab_data_dict)
        lab_test_bools = _extract_abnormal_labs(test_records, test_examples, lab_data_dict=lab_data_dict)
        
        if lab_train_bools.shape[1] > 1:
            print(f"  Train lab bitmasks: {lab_train_bools.shape}, % abnormal: {lab_train_bools.mean()*100:.1f}%")
        else:
            print(f"  WARNING: Lab bitmasks were zeroed out (dim=1). Check records/lab_data_dict.")

    print("\nBuilding retrieval index...")
    retrieval_index: dict[int, dict[str, np.ndarray]] = {}
    retrieval_index.update(
        build_retrieval_for_split(
            train_embeds,
            train_examples,
            train_embeds,
            train_labels,
            train_patient_to_rows,
            k,
            exclude_mode=args.train_exclude_mode,
            split_name="train",
            train_examples=train_examples,  # F10
            query_note_embeds=note_train_embeds,
            train_note_embeds=note_train_embeds,
            note_sim_weight=note_sim_weight,
            query_lab_bools=lab_train_bools,
            train_lab_bools=lab_train_bools,
            lab_sim_weight=lab_sim_weight,
        )
    )
    retrieval_index.update(
        build_retrieval_for_split(
            val_embeds,
            val_examples,
            train_embeds,
            train_labels,
            train_patient_to_rows,
            k,
            exclude_mode="none",
            split_name="val",
            train_examples=train_examples,  # F10
            query_note_embeds=note_val_embeds,
            train_note_embeds=note_train_embeds,
            note_sim_weight=note_sim_weight,
            query_lab_bools=lab_val_bools,
            train_lab_bools=lab_train_bools,
            lab_sim_weight=lab_sim_weight,
        )
    )
    retrieval_index.update(
        build_retrieval_for_split(
            test_embeds,
            test_examples,
            train_embeds,
            train_labels,
            train_patient_to_rows,
            k,
            exclude_mode="none",
            split_name="test",
            train_examples=train_examples,  # F10
            query_note_embeds=note_test_embeds,
            train_note_embeds=note_train_embeds,
            note_sim_weight=note_sim_weight,
            query_lab_bools=lab_test_bools,
            train_lab_bools=lab_train_bools,
            lab_sim_weight=lab_sim_weight,
        )
    )

    print(f"\nRetrieval index: {len(retrieval_index)} entries")
    if not retrieval_index:
        raise RuntimeError("Retrieval index is empty; check splits and embeddings.")

    sample_key = next(iter(retrieval_index))
    sample_val = retrieval_index[sample_key]
    print(f"  Sample hadm_id={sample_key}")
    print(f"    similar_reprs: {sample_val['similar_reprs'].shape}")
    print(f"    similar_multihots: {sample_val['similar_multihots'].shape}")

    all_scores = np.concatenate([v["scores"] for v in retrieval_index.values()])
    print(
        "\nScore distribution: "
        f"mean={all_scores.mean():.3f}, std={all_scores.std():.3f}, "
        f"min={all_scores.min():.3f}, max={all_scores.max():.3f}"
    )

    # Embed a lightweight provenance stamp inside the index itself so code that
    # loads the .pkl can verify the source without requiring the sidecar.
    retrieval_index["__meta__"] = {
        "schema_version": 2,  # F10: entries now include neighbor_hadm_ids / neighbor_patient_ids
        "top_k": k,
        "split_source": split.split_source,
        "split_seed_used": split.split_seed_used,
        "train_exclude_mode": args.train_exclude_mode,
        "whitening": whitening_info,
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "manifest_verified": bool(manifest_verified_hashes),
        "train_embeddings_sha256": file_sha256(train_emb_path),
        "num_entries": len({k for k in retrieval_index.keys() if isinstance(k, int)}),
    }

    with open(output_path, "wb") as f:
        pickle.dump(retrieval_index, f, protocol=4)
    size_mb = output_path.stat().st_size / 1e6
    print(f"\nSaved retrieval index: {output_path} ({size_mb:.1f} MB)")

    metadata = {
        "schema_version": 2,  # F10: neighbor_hadm_ids / neighbor_patient_ids in every entry
        "top_k": k,
        "split_mode_requested": args.split_mode,
        "split_source": split.split_source,
        "split_seed_used": split.split_seed_used,
        "train_exclude_mode": args.train_exclude_mode,
        "whitening": whitening_info,
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "manifest_verified_hashes": manifest_verified_hashes,
        "counts": {
            "records": len(records),
            "train_patients": len(train_records),
            "val_patients": len(val_records),
            "test_patients": len(test_records),
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "test_examples": len(test_examples),
            "retrieval_entries": len([k for k in retrieval_index if isinstance(k, int)]),
        },
        "embedding_shapes": {
            "train": list(train_embeds.shape),
            "val": list(val_embeds.shape),
            "test": list(test_embeds.shape),
            "labels_train": list(train_labels.shape),
        },
        "inputs": {
            "records_path": str(records_path),
            "cohort_path": str(cohort_path),
            "embeddings_dir": str(embed_dir),
            "train_embeddings_path": str(train_emb_path),
            "val_embeddings_path": str(val_emb_path),
            "test_embeddings_path": str(test_emb_path),
            "records_sha256": file_sha256(records_path),
            "cohort_sha256": file_sha256(cohort_path),
            "train_embeddings_sha256": file_sha256(train_emb_path),
            "val_embeddings_sha256": file_sha256(val_emb_path),
            "test_embeddings_sha256": file_sha256(test_emb_path),
        },
        "output": {
            "retrieval_index_path": str(output_path),
            "retrieval_index_sha256": file_sha256(output_path),
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved provenance metadata: {metadata_path}")


if __name__ == "__main__":
    main()
