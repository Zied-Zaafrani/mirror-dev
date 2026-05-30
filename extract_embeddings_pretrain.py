"""Phase 1.5: Extract patient embeddings from a trained PreTrain Phi checkpoint.

This script is aligned with the current Step 1 pipeline used by pretrain.py and
compute_similarity.py:
  - loads pretrain checkpoint + config
  - rebuilds PretrainMIRROR with current interfaces
  - reproduces split protocol deterministically
  - exports split-wise embedding packages with labels

Outputs:
  - patient_embeddings_train.pkl
  - patient_embeddings_val.pkl
  - patient_embeddings_test.pkl
Each file contains embeddings, labels, and ordering/provenance metadata used by
downstream retrieval integrity checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import MIRRORDataset, collate_fn
from model.pretrain_model import PretrainMIRROR
from split_protocol import compute_split_indices
from train import build_drug_graph, compute_directed_ehr_weights, resolve_drug_vocab

logger = logging.getLogger(__name__)


def _legacy_numpy_safe_globals() -> list[object]:
    """Allowlist legacy NumPy globals needed by older torch checkpoints.

    These are data-only reconstruction helpers/classes commonly present in
    tensor+ndarray checkpoint payloads exported before PyTorch 2.6 defaults.
    """
    safe_items: list[object] = [np.ndarray, np.dtype]

    np_core = getattr(np, "_core", None)
    if np_core is not None:
        multiarray = getattr(np_core, "multiarray", None)
        reconstruct = getattr(multiarray, "_reconstruct", None) if multiarray is not None else None
        if reconstruct is not None:
            safe_items.append(reconstruct)

    np_dtypes = getattr(np, "dtypes", None)
    if np_dtypes is not None:
        for cls_name in ("Float16DType", "Float32DType", "Float64DType", "Int32DType", "Int64DType"):
            cls = getattr(np_dtypes, cls_name, None)
            if cls is not None:
                safe_items.append(cls)

    return safe_items


def _safe_torch_load(path: Path, map_location: str | torch.device, allow_unsafe: bool):
    """Load torch artifacts with safe mode first, optional legacy fallback."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        try:
            with torch.serialization.safe_globals(_legacy_numpy_safe_globals()):
                return torch.load(path, map_location=map_location, weights_only=True)
        except Exception:
            pass
        if not allow_unsafe:
            raise RuntimeError(
                f"Refusing unsafe torch.load(weights_only=False) for {path}. "
                "Rerun with --allow_unsafe_torch_deserialize only for trusted artifacts."
            ) from exc
        logger.warning(
            "Falling back to unsafe torch.load(weights_only=False) for %s due to: %s",
            path,
            exc,
        )
        return torch.load(path, map_location=map_location, weights_only=False)


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


def _load_pickle(path: Path, trusted_roots: list[Path], allow_unsafe: bool):
    resolved = path.expanduser().resolve()
    trusted = any(_is_under_root(resolved, root) for root in trusted_roots)
    if not trusted and not allow_unsafe:
        roots_str = ", ".join(str(r.resolve()) for r in trusted_roots)
        raise RuntimeError(
            f"Refusing to load untrusted pickle path: {resolved}. "
            f"Trusted roots: {roots_str}. "
            "Set MIRROR_ALLOW_UNSAFE_DESERIALIZATION=1 or pass "
            "--allow_unsafe_torch_deserialize only for trusted artifacts."
        )
    if not trusted:
        logger.warning("Loading pickle from untrusted path due to explicit override: %s", resolved)

    with open(resolved, "rb") as f:
        return pickle.load(f)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resolve_file_tag(cohort_tag: str) -> str:
    mapping = {
        "mimic3": "final",
        "mimic4": "mimic4",
        "mimic4_full": "mimic4_full",
    }
    if cohort_tag not in mapping:
        raise ValueError(f"Unsupported cohort_tag '{cohort_tag}'. Expected one of: {sorted(mapping)}")
    return mapping[cohort_tag]


def _collect_hadm_ids_in_dataset_order(ds: MIRRORDataset) -> np.ndarray:
    hadm_ids = np.empty(len(ds), dtype=np.int64)
    for i, (pidx, t) in enumerate(ds.examples):
        visit = ds.records[pidx][t]
        hadm_ids[i] = int(visit[3]) if len(visit) > 3 else -1
    return hadm_ids


def _collect_patient_ids_in_dataset_order(ds: MIRRORDataset) -> np.ndarray:
    patient_ids = np.empty(len(ds), dtype=np.int64)
    for i, (pidx, _t) in enumerate(ds.examples):
        patient_ids[i] = int(pidx)
    return patient_ids


def _extract_embeddings(
    model: PretrainMIRROR,
    loader: DataLoader,
    device: torch.device,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    edge_weight: torch.Tensor,
    hidden_dim: int,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_embeds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            diag_seq = [t.to(device) for t in batch["diag_seq"]]
            proc_seq = [t.to(device) for t in batch["proc_seq"]]
            diag_mask = [t.to(device) for t in batch["diag_mask_seq"]]
            proc_mask = [t.to(device) for t in batch["proc_mask_seq"]]
            lengths = batch["lengths"].to(device)
            drug_history = batch["drug_history"].to(device)
            med_per_visit = batch["med_per_visit"].to(device) if "med_per_visit" in batch else None

            batch_size = lengths.size(0)
            note_embed = torch.zeros(batch_size, 768, device=device)
            has_note = torch.zeros(batch_size, dtype=torch.float32, device=device)
            lab_vector = torch.zeros(batch_size, 36, device=device)
            has_lab = torch.zeros(batch_size, dtype=torch.float32, device=device)

            model(
                diag_seq,
                proc_seq,
                diag_mask,
                proc_mask,
                lengths,
                note_embed,
                lab_vector,
                has_note,
                has_lab,
                drug_history,
                edge_index,
                edge_type,
                edge_weight=edge_weight,
                med_per_visit=med_per_visit,
            )

            all_embeds.append(model._aux_patient_repr.detach().cpu().numpy())
            all_labels.append(batch["target"].cpu().numpy())

    if not all_embeds:
        return np.zeros((0, hidden_dim), dtype=np.float32), np.zeros((0, model.num_drugs), dtype=np.float32)

    embeds = np.concatenate(all_embeds, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.float32)
    print(f"{split_name}: embeds={embeds.shape}, labels={labels.shape}")
    return embeds, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1.5: Extract patient embeddings")
    parser.add_argument("--model", type=str, required=True, help="Path to pretrain checkpoint (*.pt)")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir for patient_embeddings_*.pkl (default: processed_dir)")
    parser.add_argument("--split_mode", type=str, default=None,
                        choices=["cohort", "permutation", "sequential", "hidr_vita"],
                        help="Override split mode. Default uses checkpoint split_mode if available.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override split seed. Default uses checkpoint split seed/seed.")
    parser.add_argument("--require_cohort_split", action="store_true",
                        help="Fail fast if split_mode=cohort and split_indices are missing/invalid.")
    parser.add_argument("--cohort_tag", type=str, default=None,
                        choices=["mimic3", "mimic4", "mimic4_full"],
                        help="Override cohort tag if checkpoint does not contain it.")
    parser.add_argument("--allow_unsafe_deserialization", action="store_true",
                        help=(
                            "Allow unsafe fallback for BOTH torch checkpoint loading and pickle loading "
                            "outside trusted roots (dangerous; trusted artifacts only)."
                        ))
    parser.add_argument("--allow_unsafe_torch_deserialize", action="store_true",
                        help=(
                            "Legacy alias for --allow_unsafe_deserialization. "
                            "Prefer --allow_unsafe_deserialization."
                        ))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    checkpoint_path = Path(args.model)
    allow_unsafe_deserialization = (
        bool(args.allow_unsafe_deserialization)
        or bool(args.allow_unsafe_torch_deserialize)
        or _parse_bool_env("MIRROR_ALLOW_UNSAFE_DESERIALIZATION", default=False)
    )
    checkpoint = _safe_torch_load(checkpoint_path, map_location=device, allow_unsafe=allow_unsafe_deserialization)

    cfg = checkpoint.get("config")
    if not isinstance(cfg, dict):
        raise ValueError("Checkpoint is missing a valid 'config' dictionary.")

    processed_dir = Path(cfg["paths"]["processed_dir"])
    embeddings_dir = Path(cfg["paths"].get("embeddings_dir", "data/embeddings"))
    output_dir = Path(args.output_dir) if args.output_dir else processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    trusted_roots = [processed_dir.resolve()]

    cohort_tag = checkpoint.get("cohort_tag") or args.cohort_tag or "mimic3"
    file_tag = _resolve_file_tag(cohort_tag)

    records_path = processed_dir / f"records_{file_tag}.pkl"
    cohort_path = processed_dir / f"cohort_{cohort_tag}.pkl"
    ddi_path = processed_dir / f"ddi_A_{file_tag}.pkl"
    ehr_path = processed_dir / f"ehr_adj_{file_tag}.pkl"

    records = _load_pickle(records_path, trusted_roots, allow_unsafe_deserialization)
    cohort = _load_pickle(cohort_path, trusted_roots, allow_unsafe_deserialization)
    ddi_adj_np = _load_pickle(ddi_path, trusted_roots, allow_unsafe_deserialization)
    ehr_adj_np = _load_pickle(ehr_path, trusted_roots, allow_unsafe_deserialization)

    split_mode = args.split_mode or checkpoint.get("split_mode") or "hidr_vita"
    split_seed = args.seed
    if split_seed is None:
        ckpt_seed = checkpoint.get("split_seed_used")
        ckpt_seed_num = _parse_optional_int(ckpt_seed)
        if ckpt_seed_num is not None:
            split_seed = ckpt_seed_num
        else:
            split_seed = int(checkpoint.get("seed", 42))

    split = compute_split_indices(
        num_records=len(records),
        cohort=cohort,
        split_mode=split_mode,
        seed=int(split_seed),
        require_cohort_indices=args.require_cohort_split,
    )

    train_records = [records[i] for i in split.train_idx]
    val_records = [records[i] for i in split.val_idx]
    test_records = [records[i] for i in split.test_idx]

    print("\n=== Phase 1.5: Extract Embeddings ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Cohort tag: {cohort_tag} (file_tag={file_tag})")
    print(f"Split: {split.split_source} (seed={split.split_seed_used})")
    print(f"Patients: train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")

    embed_suffix = "" if file_tag == "final" else f"_{file_tag}"
    embed_file = embeddings_dir / f"code_embeddings{embed_suffix}.pt"
    embed_data = _safe_torch_load(embed_file, map_location="cpu", allow_unsafe=allow_unsafe_deserialization)

    if all(k in embed_data for k in ("diag_embeddings", "proc_embeddings", "drug_embeddings", "morgan_fingerprints")):
        diag_embeds = torch.as_tensor(embed_data["diag_embeddings"]) 
        proc_embeds = torch.as_tensor(embed_data["proc_embeddings"]) 
        drug_embeds = torch.as_tensor(embed_data["drug_embeddings"]) 
        morgan_fps = torch.as_tensor(embed_data["morgan_fingerprints"]) 
    elif all(k in embed_data for k in ("diag_embeddings_official", "proc_embeddings_official", "drug_embeddings_official", "morgan_fingerprints")):
        diag_embeds = torch.as_tensor(embed_data["diag_embeddings_official"]) 
        proc_embeds = torch.as_tensor(embed_data["proc_embeddings_official"]) 
        drug_embeds = torch.as_tensor(embed_data["drug_embeddings_official"]) 
        morgan_fps = torch.as_tensor(embed_data["morgan_fingerprints"]) 
    else:
        raise KeyError(f"Unsupported embedding keys in {embed_file}.")

    num_drugs = int(cohort["num_drugs"])
    model = PretrainMIRROR(
        diag_embeddings=diag_embeds,
        proc_embeddings=proc_embeds,
        drug_embeddings=drug_embeds,
        morgan_fingerprints=morgan_fps,
        ddi_adj=torch.tensor(ddi_adj_np, dtype=torch.float32),
        hidden_dim=cfg["model"]["hidden_dim"],
        embed_dim=cfg["model"]["embed_dim"],
        note_proj_dim=cfg["model"].get("note_proj_dim"),
        lab_proj_dim=cfg["model"].get("lab_proj_dim"),
        lab_input_dim=int(cfg["model"].get("lab_input_dim", 400 if "200" in str(processed_dir) else 36)),
        encoder_layers=cfg["model"].get("encoder_layers", 2),
        hgt_layers=cfg["model"]["hgt_layers"],
        hgt_heads=cfg["model"]["hgt_heads"],
        num_edge_types=cfg["model"]["num_edge_types"],
        dropout=cfg["model"]["dropout"],
        # F1 (Run 23): must match the pretrain-time flags so the loaded
        # checkpoint's fusion module is instantiated and its weights bind.
        # Run 22 forced False here which silently dropped the fusion layer
        # at extraction time → retrieval index was structural-only.
        use_notes=cfg["model"].get("use_notes", True),
        use_labs=cfg["model"].get("use_labs", True),
        use_copy=cfg["model"].get("copy_mechanism", True),
        finetune_embeddings=cfg["model"].get("finetune_embeddings", False),
        per_visit_copy=cfg["model"].get("per_visit_copy", True),
        max_visits=cfg["model"].get("max_visits", 30),
        use_projection_head=cfg["model"].get("use_projection_head", False),
        projection_dropout=cfg["model"].get("projection_dropout"),
    ).to(device)

    state_dict = checkpoint.get("model_state")
    if state_dict is None:
        raise ValueError("Checkpoint is missing 'model_state'.")
    model.load_state_dict(state_dict)

    ehr_directed = compute_directed_ehr_weights(train_records, num_drugs)
    drug_vocab = resolve_drug_vocab(cohort)
    if drug_vocab is None:
        print("WARNING: Could not resolve drug vocab schema; ATC edges may be skipped.")
    edge_index, edge_type, edge_weight = build_drug_graph(
        ddi_adj_np,
        ehr_adj_np,
        cooccur_threshold=0.05,
        drug_vocab=drug_vocab,
        add_self_loops=True,
        add_atc_edges=True,
        ehr_weights=ehr_directed,
    )
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
    edge_weight = edge_weight.to(device)

    train_ds = MIRRORDataset(train_records, cohort, None, None, num_drugs, "lab_vectors")
    val_ds = MIRRORDataset(val_records, cohort, None, None, num_drugs, "lab_vectors")
    test_ds = MIRRORDataset(test_records, cohort, None, None, num_drugs, "lab_vectors")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    hidden_dim = int(cfg["model"]["hidden_dim"])
    tr_emb, tr_lbl = _extract_embeddings(model, train_loader, device, edge_index, edge_type, edge_weight, hidden_dim, "train")
    va_emb, va_lbl = _extract_embeddings(model, val_loader, device, edge_index, edge_type, edge_weight, hidden_dim, "val")
    te_emb, te_lbl = _extract_embeddings(model, test_loader, device, edge_index, edge_type, edge_weight, hidden_dim, "test")

    tr_hadm = _collect_hadm_ids_in_dataset_order(train_ds)
    va_hadm = _collect_hadm_ids_in_dataset_order(val_ds)
    te_hadm = _collect_hadm_ids_in_dataset_order(test_ds)
    tr_pids = _collect_patient_ids_in_dataset_order(train_ds)
    va_pids = _collect_patient_ids_in_dataset_order(val_ds)
    te_pids = _collect_patient_ids_in_dataset_order(test_ds)

    assert len(tr_hadm) == len(tr_emb), "train hadm_ids length mismatch vs embeddings"
    assert len(va_hadm) == len(va_emb), "val hadm_ids length mismatch vs embeddings"
    assert len(te_hadm) == len(te_emb), "test hadm_ids length mismatch vs embeddings"

    outputs = {
        "train": (tr_emb, tr_lbl, tr_hadm, tr_pids),
        "val": (va_emb, va_lbl, va_hadm, va_pids),
        "test": (te_emb, te_lbl, te_hadm, te_pids),
    }
    selected_seed_num = _parse_optional_int(split.split_seed_used)
    selected_seed = selected_seed_num if selected_seed_num is not None else int(split_seed)
    pos_weight_cap = float(checkpoint.get("pos_weight_cap", cfg.get("training", {}).get("pos_weight_cap", 5.0)))
    use_cosine_lr = bool(checkpoint.get("use_cosine_lr", cfg.get("training", {}).get("use_cosine_lr", False)))
    extract_fused_repr = bool(getattr(model, "extract_fused_repr", False))

    for split_name, (embeds, labels, hadm_ids, patient_ids) in outputs.items():
        out_path = output_dir / f"patient_embeddings_{split_name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "embeddings": embeds,
                    "labels": labels,
                    "hadm_ids": hadm_ids,
                    "patient_ids": patient_ids,
                    "split": split_name,
                    "hidden_dim": int(embeds.shape[1]) if embeds.ndim == 2 else int(hidden_dim),
                    "extract_fused_repr": extract_fused_repr,
                    "selected_seed": selected_seed,
                    "pos_weight_cap": pos_weight_cap,
                    "use_cosine_lr": use_cosine_lr,
                },
                f,
                protocol=4,
            )
        print(f"Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    metadata = {
        "checkpoint": str(checkpoint_path),
        "cohort_tag": cohort_tag,
        "file_tag": file_tag,
        "split_mode_requested": split_mode,
        "split_source": split.split_source,
        "split_seed_used": split.split_seed_used,
        "counts": {
            "records": len(records),
            "train_patients": len(train_records),
            "val_patients": len(val_records),
            "test_patients": len(test_records),
            "train_examples": len(train_ds),
            "val_examples": len(val_ds),
            "test_examples": len(test_ds),
        },
        "embedding_shapes": {
            "train": list(tr_emb.shape),
            "val": list(va_emb.shape),
            "test": list(te_emb.shape),
            "labels_train": list(tr_lbl.shape),
            "hadm_ids_train": list(tr_hadm.shape),
            "patient_ids_train": list(tr_pids.shape),
        },
    }
    meta_path = output_dir / "patient_embeddings.meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata: {meta_path}")

    artifact_files = {
        "patient_embeddings_train.pkl": output_dir / "patient_embeddings_train.pkl",
        "patient_embeddings_val.pkl": output_dir / "patient_embeddings_val.pkl",
        "patient_embeddings_test.pkl": output_dir / "patient_embeddings_test.pkl",
    }
    manifest = {
        "schema_version": 1,
        "selected_seed": selected_seed,
        "pos_weight_cap": pos_weight_cap,
        "use_cosine_lr": use_cosine_lr,
        "split_mode": split_mode,
        "split_source": split.split_source,
        "split_seed_used": str(split.split_seed_used),
        "files": {},
    }
    for name, p in artifact_files.items():
        if p.exists():
            manifest["files"][name] = {
                "sha256": _sha256_file(p),
                "size_bytes": p.stat().st_size,
            }

    manifest_path = output_dir / f"artifacts_manifest_seed{selected_seed}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")

    repro_path = output_dir / f"repro_hashes_seed{selected_seed}.json"
    current_hashes = {
        name: info["sha256"]
        for name, info in manifest["files"].items()
        if "sha256" in info
    }
    if repro_path.exists():
        stored = json.loads(repro_path.read_text(encoding="utf-8"))
        mismatches = [
            (k, stored.get(k, "MISSING"), current_hashes.get(k, "MISSING"))
            for k in sorted(set(stored) | set(current_hashes))
            if stored.get(k) != current_hashes.get(k)
        ]
        if mismatches:
            details = "\n".join(f"  - {k}: old={old} new={new}" for k, old, new in mismatches)
            raise RuntimeError(
                "Reproducibility hash mismatch detected across repeated extraction runs:\n"
                + details
            )
        print(f"Repro hash check passed: {repro_path}")
    else:
        repro_path.write_text(json.dumps(current_hashes, indent=2), encoding="utf-8")
        print(f"Initialized repro hash baseline: {repro_path}")


if __name__ == "__main__":
    main()
