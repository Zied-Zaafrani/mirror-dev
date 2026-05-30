"""Phase 1 pretrain entrypoint for training Phi embeddings used by retrieval."""

import argparse
import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import MIRRORDataset, collate_fn
from model.predictor import MIRRORLoss
from model.pretrain_model import PretrainMIRROR
from split_protocol import compute_split_indices
from train import (
    build_drug_graph,
    compute_directed_ehr_weights,
    compute_pos_weight,
    evaluate_epoch,
    load_config,
    resolve_drug_vocab,
    train_epoch,
)

logger = logging.getLogger(__name__)


def _resolve_tag(mimic_version: int, mimic4_full: bool) -> tuple[str, str]:
    if mimic_version == 3:
        return "final", "mimic3"
    if mimic4_full:
        return "mimic4_full", "mimic4_full"
    return "mimic4", "mimic4"


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
            "--allow_unsafe_deserialization only for trusted artifacts."
        )
    if not trusted:
        logger.warning("Loading pickle from untrusted path due to explicit override: %s", resolved)

    with open(resolved, "rb") as f:
        return pickle.load(f)


def main() -> str:
    parser = argparse.ArgumentParser(description="Phase 1: PreTrain Phi for offline retrieval")
    parser.add_argument("--config", type=str, default="src/config_pretrain.yaml", help="Config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--mimic_version", type=int, default=3, choices=[3, 4])
    parser.add_argument("--mimic4_full", action="store_true", default=False)
    parser.add_argument(
        "--split_mode",
        type=str,
        default="hidr_vita",
        choices=["cohort", "permutation", "sequential", "hidr_vita"],
        help=(
            "Split protocol: hidr_vita (default), or cohort/permutation/sequential "
            "for strict HI-DR/VITA-style slicing"
        ),
    )
    parser.add_argument(
        "--require_cohort_split",
        action="store_true",
        help="Fail fast when split_mode=cohort but split_indices are missing/invalid.",
    )
    parser.add_argument(
        "--allow_unsafe_deserialization",
        action="store_true",
        help="Allow loading pickle files outside trusted roots (dangerous; trusted artifacts only).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, ablation="pretrain_only")
    processed_dir = Path(cfg["paths"]["processed_dir"])
    trusted_roots = [processed_dir.resolve()]
    allow_unsafe_deserialization = bool(args.allow_unsafe_deserialization) or _parse_bool_env(
        "MIRROR_ALLOW_UNSAFE_DESERIALIZATION",
        default=False,
    )
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = results_dir.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = requested_device

    file_tag, cohort_tag = _resolve_tag(args.mimic_version, args.mimic4_full)

    print(f"\n{'=' * 60}")
    print("Phase 1: PreTrain Phi for Offline Retrieval")
    print(f"{'=' * 60}")
    print(f"Config: {args.config}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Cohort tag: {cohort_tag}")
    if args.split_mode != "permutation":
        print(
            "NOTE: HI-DR paper comparability uses random split + multi-seed averaging. "
            "Use --split_mode permutation and run multiple seeds for paper-style numbers."
        )

    records_path = processed_dir / f"records_{file_tag}.pkl"
    cohort_path = processed_dir / f"cohort_{cohort_tag}.pkl"
    ddi_path = processed_dir / f"ddi_A_{file_tag}.pkl"
    ehr_path = processed_dir / f"ehr_adj_{file_tag}.pkl"

    diag_embed_path = processed_dir / f"diag_embeddings_{cohort_tag}.pkl"
    proc_embed_path = processed_dir / f"proc_embeddings_{cohort_tag}.pkl"
    drug_embed_path = processed_dir / f"drug_embeddings_{cohort_tag}.pkl"
    morgan_path = processed_dir / "morgan_fingerprints.pkl"

    print("\nLoading data artifacts...")
    records = _load_pickle(records_path, trusted_roots, allow_unsafe_deserialization)
    cohort = _load_pickle(cohort_path, trusted_roots, allow_unsafe_deserialization)
    ddi_adj_np = _load_pickle(ddi_path, trusted_roots, allow_unsafe_deserialization)
    ehr_adj_np = _load_pickle(ehr_path, trusted_roots, allow_unsafe_deserialization)
    diag_embeds = _load_pickle(diag_embed_path, trusted_roots, allow_unsafe_deserialization)
    proc_embeds = _load_pickle(proc_embed_path, trusted_roots, allow_unsafe_deserialization)
    drug_embeds = _load_pickle(drug_embed_path, trusted_roots, allow_unsafe_deserialization)
    morgan_fps = _load_pickle(morgan_path, trusted_roots, allow_unsafe_deserialization)

    num_drugs = int(cohort["num_drugs"])

    split = compute_split_indices(
        num_records=len(records),
        cohort=cohort,
        split_mode=args.split_mode,
        seed=args.seed,
        require_cohort_indices=args.require_cohort_split,
    )
    print(
        f"Split ({split.split_source}, seed={split.split_seed_used}): "
        f"train={len(split.train_idx)}, val={len(split.val_idx)}, test={len(split.test_idx)}"
    )

    train_records = [records[i] for i in split.train_idx]
    val_records = [records[i] for i in split.val_idx]
    if len(train_records) == 0 or len(val_records) == 0:
        raise ValueError("Pretrain split produced an empty train or val partition.")

    print("Building directed EHR graph weights...")
    ehr_directed = compute_directed_ehr_weights(train_records, num_drugs)
    drug_vocab = resolve_drug_vocab(cohort)
    if drug_vocab is None:
        print("  WARNING: Could not resolve drug vocab schema; ATC edges may be skipped.")
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

    print("Creating datasets...")
    train_ds = MIRRORDataset(train_records, cohort, None, None, num_drugs, "lab_vectors")
    val_ds = MIRRORDataset(val_records, cohort, None, None, num_drugs, "lab_vectors")
    print(f"  Train examples: {len(train_ds)} | Val examples: {len(val_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Pretrain dataset has zero examples after expansion.")

    batch_size = int(cfg["training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    print("\nBuilding pretrain model...")
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
        lab_input_dim=36,
        encoder_layers=cfg["model"].get("encoder_layers", 2),
        hgt_layers=cfg["model"]["hgt_layers"],
        hgt_heads=cfg["model"]["hgt_heads"],
        num_edge_types=cfg["model"]["num_edge_types"],
        dropout=cfg["model"]["dropout"],
        # F1 (Run 23): notes+labs must flow through pretrain so the extracted
        # fused repr (used as retrieval index) lives in the same multimodal
        # space as training-time queries. Run 22 hard-disabled these which
        # silently produced a structural-only index. Config can still override
        # for diagnostic structural-only ablations.
        use_notes=cfg["model"].get("use_notes", True),
        use_labs=cfg["model"].get("use_labs", True),
        use_copy=cfg["model"]["copy_mechanism"],
        finetune_embeddings=cfg["model"].get("finetune_embeddings", False),
        per_visit_copy=cfg["model"].get("per_visit_copy", True),
        max_visits=cfg["model"].get("max_visits", 30),
        use_projection_head=cfg["model"].get("use_projection_head", False),
        projection_dropout=cfg["model"].get("projection_dropout"),
    ).to(device)
    params = model.count_parameters()
    print(f"  Total parameters: {params['model_total']['total']:,}")
    print(f"  Trainable parameters: {params['model_total']['trainable']:,}")

    pos_weight_cap = float(cfg["training"].get("pos_weight_cap", 5.0))
    pos_weight = compute_pos_weight(train_records, num_drugs, max_cap=pos_weight_cap).to(device)
    loss_fn = MIRRORLoss(
        ddi_adj=torch.tensor(ddi_adj_np, dtype=torch.float32).to(device),
        bce_weight=cfg["training"]["bce_weight"],
        margin_weight=cfg["training"]["margin_weight"],
        label_smoothing=cfg["training"].get("label_smoothing", 0.0),
        pos_weight=pos_weight,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    use_cosine_lr = bool(cfg["training"].get("use_cosine_lr", False))
    if use_cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(cfg["training"]["epochs"]), 1),
            eta_min=float(cfg["training"].get("scheduler_min_lr", 1e-5)),
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(cfg["training"].get("scheduler_factor", 0.5)),
            patience=int(cfg["training"].get("scheduler_patience", 5)),
            min_lr=float(cfg["training"].get("scheduler_min_lr", 1e-5)),
        )

    best_val_jaccard = -1.0
    best_epoch = 0
    patience_counter = 0
    patience = int(cfg["training"]["patience"])
    min_epochs_before_estop = int(cfg["training"].get("min_epochs_before_estop", 0))
    epochs = int(cfg["training"]["epochs"])
    grad_clip = float(cfg["model"]["gradient_clip"])

    print(
        f"\nStarting pretrain ({epochs} epochs max) | "
        f"patience={patience}, min_epochs_before_estop={min_epochs_before_estop}...\n"
    )
    checkpoint_path = models_dir / f"pretrain_phi_seed{args.seed}_{cohort_tag}.pt"
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            edge_index,
            edge_type,
            grad_clip=grad_clip,
            edge_weight=edge_weight,
        )

        val_metrics = evaluate_epoch(
            model,
            val_loader,
            ddi_adj_np,
            device,
            edge_index,
            edge_type,
            threshold=float(cfg["training"].get("threshold", 0.5)),
            top_k=cfg["training"].get("top_k"),
            edge_weight=edge_weight,
        )
        val_jaccard = float(val_metrics["Jaccard"])
        if use_cosine_lr:
            scheduler.step()
        else:
            scheduler.step(val_jaccard)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d} | Train Loss: {train_loss['total']:.4f} "
            f"(BCE:{train_loss['bce']:.4f} Mrg:{train_loss['margin']:.4f}) | "
            f"Val Jaccard: {val_jaccard:.4f} | DDI: {val_metrics['DDI Rate']:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if val_jaccard > best_val_jaccard:
            best_val_jaccard = val_jaccard
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": cfg,
                    "seed": args.seed,
                    "val_jaccard": val_jaccard,
                    "cohort_tag": cohort_tag,
                    "split_mode": args.split_mode,
                    "split_source": split.split_source,
                    "split_seed_used": split.split_seed_used,
                    "pos_weight_cap": pos_weight_cap,
                    "use_cosine_lr": use_cosine_lr,
                    "records_path": str(records_path),
                },
                checkpoint_path,
            )
            print(f"  Saved checkpoint: {checkpoint_path}")
        else:
            if epoch >= min_epochs_before_estop:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
                    break

    if best_epoch == 0:
        raise RuntimeError("No checkpoint was saved during pretraining (best epoch never set).")

    print(f"\n{'=' * 60}")
    print("PreTrain complete")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Jaccard: {best_val_jaccard:.4f}")
    print(f"Model saved: {checkpoint_path}")
    print(f"{'=' * 60}\n")

    config_path = models_dir / f"pretrain_phi_seed{args.seed}_{cohort_tag}.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=str)

    return str(checkpoint_path)


if __name__ == "__main__":
    model_path = main()
    print(f"Ready for Phase 1.5 embedding extraction using {model_path}")
