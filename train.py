"""
MIRROR Training Script.

Handles:
  - Loading all preprocessed data (records, embeddings, notes, labs, DDI)
  - Building drug graph (DDI + co-occurrence + self-loops + ATC edges)
  - Model instantiation
  - Training loop with early stopping on val Jaccard
  - Multi-seed evaluation with mean ± std

Usage:
  python train.py --config src_final/config.yaml --seed 42 --device cpu
  python train.py --config src_final/config.yaml --seed 42 --device cuda
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

logger = logging.getLogger(__name__)

try:
    from .dataset import MIRRORDataset, collate_fn
    from .evaluate import evaluate_all, apply_threshold, evaluate_threshold_sweep
    from .model.model import MIRROR
    from .model.predictor import MIRRORLoss
    from .split_protocol import compute_split_indices
except ImportError:
    from dataset import MIRRORDataset, collate_fn
    from evaluate import evaluate_all, apply_threshold, evaluate_threshold_sweep
    from model.model import MIRROR
    from model.predictor import MIRRORLoss
    from split_protocol import compute_split_indices


def _empty_metrics() -> dict[str, float]:
    """Safe zero metrics when a split/loader has no samples."""
    return {
        "Jaccard": 0.0,
        "F1": 0.0,
        "PRAUC": 0.0,
        "DDI Rate": 0.0,
        "Avg Meds": 0.0,
        "Avg True Meds": 0.0,
        "Precision": 0.0,
        "Recall": 0.0,
    }


def load_config(config_path: str) -> dict:
    """Load config YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_drug_vocab(cohort: dict) -> dict | None:
    """Normalize cohort drug vocabulary into {idx2word, word2idx} schema.

    Supported input schemas:
      1) cohort['drug_vocab'] = {0: 'A02', 1: 'A03', ...}   — int keys, str values
      2) cohort['drug_vocab'] = {'A02': 0, 'A03': 1, ...}   — str keys, int values (inverted)
      3) cohort['med_voc'] = {'idx2word': {...}, 'word2idx': {...}}
      4) cohort['drug_vocab'] = {'idx2word': {...}, 'word2idx': {...}}
    """
    def _coerce_idx2word(raw: dict) -> dict[int, str]:
        idx2word: dict[int, str] = {}
        for k, v in raw.items():
            if not isinstance(v, str):
                continue
            try:
                idx = int(k)
            except (TypeError, ValueError):
                continue
            idx2word[idx] = v
        return idx2word

    for key in ("drug_vocab", "med_voc"):
        vocab = cohort.get(key)
        if not isinstance(vocab, dict):
            continue

        if isinstance(vocab.get("idx2word"), dict):
            idx2word = _coerce_idx2word(vocab["idx2word"])
            if not idx2word:
                continue
            word2idx = {str(word): int(idx) for idx, word in idx2word.items()}
            return {"idx2word": idx2word, "word2idx": word2idx}

        idx2word = _coerce_idx2word(vocab)
        if idx2word:
            word2idx = {str(word): int(idx) for idx, word in idx2word.items()}
            return {"idx2word": idx2word, "word2idx": word2idx}

        inverted = {v: k for k, v in vocab.items()
                    if isinstance(k, str) and isinstance(v, int)}
        if inverted:
            word2idx = {str(k): v for k, v in vocab.items()
                        if isinstance(k, str) and isinstance(v, int)}
            return {"idx2word": inverted, "word2idx": word2idx}

    return None


def compute_directed_ehr_weights(train_records: list, num_drugs: int) -> np.ndarray:
    """Compute directed conditional probability edge weights: w(i->j) = P(j|i).

    Returns (num_drugs, num_drugs) matrix where [i,j] = P(j|i).
    """
    co_count = np.zeros((num_drugs, num_drugs), dtype=np.float64)
    drug_count = np.zeros(num_drugs, dtype=np.float64)

    for patient in train_records:
        for visit in patient:
            meds = [m for m in visit[2] if m < num_drugs]
            for m in meds:
                drug_count[m] += 1
            for i in range(len(meds)):
                for j in range(len(meds)):
                    if i != j:
                        co_count[meds[i], meds[j]] += 1

    ehr_weights = np.zeros((num_drugs, num_drugs), dtype=np.float32)
    for i in range(num_drugs):
        if drug_count[i] > 0:
            ehr_weights[i] = co_count[i] / drug_count[i]

    asymmetric_pairs = np.sum(np.abs(ehr_weights - ehr_weights.T) > 1e-6) // 2
    nonzero_edges = np.count_nonzero(ehr_weights)
    print(f"  Directed EHR weights: {nonzero_edges} non-zero entries, "
          f"{asymmetric_pairs} asymmetric pairs")
    return ehr_weights


def build_drug_graph(
    ddi_adj: np.ndarray,
    ehr_adj: np.ndarray,
    ddi_threshold: float = 0.0,
    cooccur_threshold: float = 0.01,
    drug_vocab: dict | None = None,
    add_self_loops: bool = True,
    add_atc_edges: bool = True,
    ehr_weights: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build edge_index, edge_type, and edge_weight tensors for the drug graph.

    Edge type 0 = DDI, Edge type 1 = co-occurrence,
    Edge type 2 = self-loop, Edge type 3 = ATC-class (same therapeutic family).
    Co-occurrence edges get directed P(j|i) weights when ehr_weights is provided.
    """
    from collections import defaultdict

    edges_src: list[int] = []
    edges_tgt: list[int] = []
    edge_types: list[int] = []

    num_drugs = ddi_adj.shape[0]

    # DDI edges (type 0)
    for i in range(num_drugs):
        for j in range(i + 1, num_drugs):
            if ddi_adj[i, j] > ddi_threshold:
                edges_src.extend([i, j])
                edges_tgt.extend([j, i])
                edge_types.extend([0, 0])

    # Co-occurrence edges (type 1)
    for i in range(num_drugs):
        for j in range(i + 1, num_drugs):
            if ehr_adj[i, j] > cooccur_threshold:
                edges_src.extend([i, j])
                edges_tgt.extend([j, i])
                edge_types.extend([1, 1])

    # Self-loops (type 2)
    if add_self_loops:
        for i in range(num_drugs):
            edges_src.append(i)
            edges_tgt.append(i)
            edge_types.append(2)

    # ATC-class edges (type 3): drugs in the same ATC therapeutic group.
    # P15 FIX: ATC-3 singletons fall back to ATC-2 grouping.
    if add_atc_edges and drug_vocab is not None:
        idx2word: dict = {}
        if isinstance(drug_vocab, dict):
            nested = drug_vocab.get("idx2word")
            if isinstance(nested, dict):
                idx2word = nested
            else:
                idx2word = drug_vocab

        atc3_groups: dict = defaultdict(list)
        drug_names: dict = {}
        for idx in range(num_drugs):
            name = idx2word.get(idx, idx2word.get(str(idx), ""))
            drug_names[idx] = name if isinstance(name, str) else ""
            if isinstance(name, str) and len(name) >= 3:
                atc3_groups[name[:3]].append(idx)

        singleton_drugs: set = set()
        for group_drugs in atc3_groups.values():
            if len(group_drugs) == 1:
                singleton_drugs.add(group_drugs[0])

        atc2_groups: dict = defaultdict(list)
        for idx in singleton_drugs:
            name = drug_names.get(idx, "")
            if len(name) >= 2:
                atc2_groups[name[:2]].append(idx)

        atc_edge_count = 0
        covered_by_atc: set = set()

        for group_drugs in atc3_groups.values():
            if len(group_drugs) > 1:
                for i in group_drugs:
                    for j in group_drugs:
                        if i != j:
                            edges_src.append(i)
                            edges_tgt.append(j)
                            edge_types.append(3)
                            atc_edge_count += 1
                            covered_by_atc.add(i)

        atc2_fallback_count = 0
        for group_drugs in atc2_groups.values():
            if len(group_drugs) > 1:
                for i in group_drugs:
                    for j in group_drugs:
                        if i != j:
                            edges_src.append(i)
                            edges_tgt.append(j)
                            edge_types.append(3)
                            atc_edge_count += 1
                            atc2_fallback_count += 1
                            covered_by_atc.add(i)

        still_isolated = singleton_drugs - covered_by_atc
        print(f"  ATC coverage: {len(covered_by_atc)}/{num_drugs} drugs "
              f"({atc_edge_count} ATC-3 edges, {atc2_fallback_count} ATC-2 fallback, "
              f"{len(still_isolated)} true singletons)")

    if not edges_src:
        edges_src = list(range(num_drugs))
        edges_tgt = list(range(num_drugs))
        edge_types = [0] * num_drugs

    edge_index = torch.tensor([edges_src, edges_tgt], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)

    num_edges = len(edges_src)
    weights = np.ones(num_edges, dtype=np.float32)
    if ehr_weights is not None:
        for idx in range(num_edges):
            if edge_types[idx] == 1:
                src, tgt = edges_src[idx], edges_tgt[idx]
                weights[idx] = max(ehr_weights[src, tgt], 1e-8)
    edge_weight = torch.tensor(weights, dtype=torch.float32)

    counts = {t: sum(1 for et in edge_types if et == t) for t in sorted(set(edge_types))}
    labels = {0: "DDI", 1: "CoOccur", 2: "SelfLoop", 3: "ATC"}
    parts = [f"{labels.get(t, f'Type{t}')}={c}" for t, c in counts.items()]
    print(f"  Drug graph: {edge_index.size(1)} edges ({', '.join(parts)})")
    if ehr_weights is not None:
        cooccur_weights = weights[np.array(edge_types) == 1]
        if len(cooccur_weights) > 0:
            print(f"  Directed weights (co-occur): min={cooccur_weights.min():.4f}, "
                  f"max={cooccur_weights.max():.4f}, mean={cooccur_weights.mean():.4f}")
    return edge_index, edge_type, edge_weight


def compute_pos_weight(
    train_records: list,
    num_drugs: int,
    max_cap: float = 5.0,
    min_cap: float = 0.1,
) -> torch.Tensor:
    """Compute per-drug positive weight from training set (linear formula)."""
    pos_count = np.zeros(num_drugs, dtype=np.float32)
    total = 0
    for patient in train_records:
        for visit in patient:
            for m in visit[2]:
                if m < num_drugs:
                    pos_count[m] += 1
            total += 1

    neg_count = total - pos_count
    pos_weight = neg_count / np.maximum(pos_count, 1.0)
    pos_weight = np.clip(pos_weight, float(min_cap), float(max_cap))
    return torch.tensor(pos_weight, dtype=torch.float32)


def train_epoch(
    model,
    dataloader,
    loss_fn,
    optimizer,
    device,
    edge_index,
    edge_type,
    grad_clip: float,
    edge_weight=None,
):
    """Train for one epoch. Returns average loss dict."""
    model.train()
    total_losses: dict[str, float] = {}
    num_batches = 0

    for batch in dataloader:
        diag_seq = [t.to(device) for t in batch["diag_seq"]]
        proc_seq = [t.to(device) for t in batch["proc_seq"]]
        diag_mask = [t.to(device) for t in batch["diag_mask_seq"]]
        proc_mask = [t.to(device) for t in batch["proc_mask_seq"]]
        lengths = batch["lengths"].to(device)
        target = batch["target"].to(device)
        drug_history = batch["drug_history"].to(device)
        note_embed = batch["note_embed"].to(device)
        has_note = batch["has_note"].to(device)
        lab_vector = batch["lab_vector"].to(device)
        has_lab = batch["has_lab"].to(device)
        med_per_visit = batch["med_per_visit"].to(device) if "med_per_visit" in batch else None

        optimizer.zero_grad()

        logits, copy_gate = model(
            diag_seq, proc_seq, diag_mask, proc_mask, lengths,
            note_embed, lab_vector, has_note, has_lab,
            drug_history, edge_index, edge_type,
            edge_weight=edge_weight,
            med_per_visit=med_per_visit,
        )

        loss, loss_dict = loss_fn(logits, target)
        loss_dict["total"] = loss.item()

        loss.backward()

        has_bad_grad = False
        for name, p in model.named_parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                logger.warning(f"NaN/Inf gradient in {name}, skipping optimizer step")
                has_bad_grad = True
                break

        if has_bad_grad:
            optimizer.zero_grad()
        else:
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0) + v
        num_batches += 1

    if num_batches == 0:
        return {"bce": 0.0, "total": 0.0}

    epoch_results = {k: v / num_batches for k, v in total_losses.items()}
    if hasattr(model, "predictor") and hasattr(model.predictor, "get_head_weights"):
        epoch_results.update(model.predictor.get_head_weights())
    return epoch_results


@torch.no_grad()
def evaluate_epoch(
    model,
    dataloader,
    ddi_adj_np,
    device,
    edge_index,
    edge_type,
    top_k=None,
    threshold=0.5,
    return_raw=False,
    edge_weight=None,
):
    """Evaluate on a dataset split. Returns metrics dict."""
    model.eval()
    all_targets = []
    all_probs = []

    for batch in dataloader:
        diag_seq = [t.to(device) for t in batch["diag_seq"]]
        proc_seq = [t.to(device) for t in batch["proc_seq"]]
        diag_mask = [t.to(device) for t in batch["diag_mask_seq"]]
        proc_mask = [t.to(device) for t in batch["proc_mask_seq"]]
        lengths = batch["lengths"].to(device)
        drug_history = batch["drug_history"].to(device)
        note_embed = batch["note_embed"].to(device)
        has_note = batch["has_note"].to(device)
        lab_vector = batch["lab_vector"].to(device)
        has_lab = batch["has_lab"].to(device)
        med_per_visit = batch["med_per_visit"].to(device) if "med_per_visit" in batch else None

        logits, _ = model(
            diag_seq, proc_seq, diag_mask, proc_mask, lengths,
            note_embed, lab_vector, has_note, has_lab,
            drug_history, edge_index, edge_type,
            edge_weight=edge_weight,
            med_per_visit=med_per_visit,
        )

        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(batch["target"].numpy())

    if not all_probs or not all_targets:
        metrics = _empty_metrics()
        if return_raw:
            return metrics, np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
        return metrics

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = apply_threshold(all_probs, threshold=threshold, top_k=top_k)
    metrics = evaluate_all(all_targets, all_preds, all_probs, ddi_adj_np)

    if return_raw:
        return metrics, all_targets, all_probs
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train MIRROR")
    # Config & run identity
    parser.add_argument("--config", type=str, default="src_final/config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    # Path overrides
    parser.add_argument("--processed_dir", type=str, default=None)
    parser.add_argument("--embeddings_dir", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    # Cohort selection
    parser.add_argument("--mimic_version", type=int, default=3, choices=[3, 4])
    parser.add_argument("--mimic4_full", action="store_true", default=False,
                        help="Use full ICD-9+10 MIMIC-IV cohort")
    parser.add_argument("--mimic4_sota", action="store_true", default=False,
                        help="Strict admission-level ICD-9 MIMIC-IV cohort (~9K patients)")
    # Data file overrides
    parser.add_argument("--note_pkl", type=str, default=None,
                        help="Override note embedding pkl path.")
    parser.add_argument("--lab_pkl", type=str, default=None,
                        help="Override lab data pkl path.")
    # Champion sweep axes
    parser.add_argument("--ddi_alpha", type=float, default=0.0,
                        help="DDI loss weight (0 = monitoring-only; swept: 0.0, 0.2, 0.5).")
    # Training controls
    parser.add_argument("--visit_level_training", action="store_true",
                        help="Unroll sequences to visit-level prediction.")
    parser.add_argument("--num_labs", type=int, default=200,
                        help="Number of lab features (default 200).")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--ar_max_seq_len", type=int, default=None)
    # Architecture overrides (for ablation notebook)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--note_proj_dim", type=int, default=None)
    parser.add_argument("--lab_dim", type=int, default=None)
    parser.add_argument("--fusion_strategy", type=str, default=None,
                        help="Override fusion strategy (config default: film).")
    parser.add_argument("--aggregator_type", type=str, default=None,
                        help="Override aggregator type (config default: last).")
    parser.add_argument("--encoder_type", type=str, default=None,
                        help="Override temporal encoder type (config default: imdr_infused).")
    parser.add_argument("--predictor_type", type=str, default=None,
                        help="Override predictor type (config default: heidr).")
    parser.add_argument("--lab_encoder_type", type=str, default=None,
                        help="Override lab encoder type (config default: flat).")
    # Historical attention toggle (ablation)
    parser.add_argument("--no_historical_attention", action="store_true",
                        help="Disable within-patient historical visit attention.")
    # Loss weight overrides
    parser.add_argument("--bce_weight", type=float, default=None)
    parser.add_argument("--soft_jaccard_weight", type=float, default=None)
    parser.add_argument("--margin_weight", type=float, default=None)
    # Misc
    parser.add_argument("--finetune_embeddings", action="store_true")
    parser.add_argument("--target_ddi", type=float, default=None)
    parser.add_argument("--kp", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    processed_dir = Path(args.processed_dir or cfg["paths"]["processed_dir"])
    embeddings_dir = Path(args.embeddings_dir or cfg["paths"]["embeddings_dir"])
    results_dir = Path(args.results_dir or cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Seed: {seed}, Device: {device}")

    # ===== Cohort tag =====
    if args.mimic_version == 3:
        file_tag = "final"
        cohort_tag = "mimic3"
    elif args.mimic4_sota:
        file_tag = "mimic4_sota"
        cohort_tag = "mimic4_sota"
    elif args.mimic4_full:
        file_tag = "mimic4_full"
        cohort_tag = "mimic4_full"
    else:
        file_tag = "mimic4"
        cohort_tag = "mimic4"

    # ===== Load data =====
    print(f"\n=== Loading data (tag={file_tag}) ===")
    records_path = processed_dir / f"records_{file_tag}.pkl"
    cohort_path = processed_dir / f"cohort_{cohort_tag}.pkl"
    ddi_path = processed_dir / f"ddi_A_{file_tag}.pkl"
    ehr_path = processed_dir / f"ehr_adj_{file_tag}.pkl"

    with open(records_path, "rb") as f:
        records = pickle.load(f)
    with open(cohort_path, "rb") as f:
        cohort = pickle.load(f)
    with open(ddi_path, "rb") as f:
        ddi_adj_np = pickle.load(f)
    with open(ehr_path, "rb") as f:
        ehr_adj_np = pickle.load(f)

    num_drugs = cohort["num_drugs"]
    num_diag = cohort["num_diag"]
    num_proc = cohort["num_proc"]
    total_pairs = int(num_drugs * (num_drugs - 1) / 2)
    ddi_upper_pairs = int(np.count_nonzero(np.triu(ddi_adj_np, k=1) > 0))
    ddi_density = (ddi_upper_pairs / total_pairs) if total_pairs > 0 else 0.0
    print(f"  Records: {len(records)} patients, {num_drugs} drugs, "
          f"{num_diag} diag, {num_proc} proc")

    # Load embeddings (LLM-enriched path only)
    embed_suffix = "" if file_tag == "final" else f"_{file_tag}"
    embed_file = embeddings_dir / f"code_embeddings{embed_suffix}.pt"
    print(f"  Loading embeddings from {embed_file} ...")
    try:
        embed_data = torch.load(embed_file, map_location="cpu", weights_only=True)
    except Exception as exc:
        safe_items = [np.ndarray, np.dtype]
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
        with torch.serialization.safe_globals(safe_items):
            embed_data = torch.load(embed_file, map_location="cpu", weights_only=True)

    diag_embeds = torch.tensor(embed_data["diag_embeddings"])
    proc_embeds = torch.tensor(embed_data["proc_embeddings"])
    drug_embeds = torch.tensor(embed_data["drug_embeddings"])
    morgan_fps = torch.tensor(embed_data["morgan_fingerprints"])

    # Load notes
    note_file: Path | None = None
    note_data = None
    use_notes = cfg["preprocessing"].get("note_method", "none") != "none"
    if use_notes:
        note_file = Path(args.note_pkl) if args.note_pkl else processed_dir / f"note_embeddings_{cohort_tag}.pkl"
        if note_file.exists():
            print(f"  Loading note embeddings from {note_file} ...")
            with open(note_file, "rb") as f:
                note_data = pickle.load(f)
        else:
            print(f"  NOTE: {note_file} not found — running without notes")
            use_notes = False

    # Load labs
    lab_file: Path | None = None
    lab_data = None
    num_labs = args.num_labs  # default 200
    lab_dim = args.lab_dim or cfg["preprocessing"].get("lab_dim", 400)
    use_labs = lab_dim > 0
    lab_key = "lab_vectors"
    if use_labs:
        if args.lab_pkl:
            lab_file = Path(args.lab_pkl)
        else:
            lab_file = processed_dir / f"lab_vectors_{num_labs}labs.pkl"
        if lab_file.exists():
            print(f"  Loading lab data from {lab_file} ...")
            with open(lab_file, "rb") as f:
                lab_data = pickle.load(f)
            if lab_data and lab_key in lab_data:
                actual_dim = lab_data[lab_key].shape[1]
                if actual_dim != lab_dim:
                    print(f"  Auto-detecting lab_dim: {lab_dim} -> {actual_dim}")
                    lab_dim = actual_dim
        else:
            print(f"  NOTE: {lab_file} not found — running without labs")
            use_labs = False
            lab_dim = 400

    # ===== Split data =====
    print("\n=== Splitting data ===")
    split = compute_split_indices(
        num_records=len(records),
        cohort=cohort,
        split_mode="hidr_vita",
        seed=seed,
    )
    train_idx = split.train_idx
    val_idx = split.val_idx
    test_idx = split.test_idx
    split_source = split.split_source
    split_seed_used = split.split_seed_used
    print(f"  Split protocol: {split_source} (seed={split_seed_used})")

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    test_records = [records[i] for i in test_idx]
    if len(train_records) == 0 or len(val_records) == 0 or len(test_records) == 0:
        raise ValueError(
            f"Split produced empty partition(s): "
            f"train={len(train_records)}, val={len(val_records)}, test={len(test_records)}"
        )
    split_patient_counts = {
        "train": len(train_records),
        "val": len(val_records),
        "test": len(test_records),
    }
    print(f"  Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")

    batch_size = args.batch_size or cfg["training"]["batch_size"]
    train_ds = MIRRORDataset(train_records, cohort, note_data, lab_data, num_drugs, lab_key,
                             visit_level_scramble=args.visit_level_training,
                             num_labs=num_labs)
    val_ds = MIRRORDataset(val_records, cohort, note_data, lab_data, num_drugs, lab_key,
                           visit_level_scramble=args.visit_level_training,
                           num_labs=num_labs)
    test_ds = MIRRORDataset(test_records, cohort, note_data, lab_data, num_drugs, lab_key,
                            visit_level_scramble=args.visit_level_training,
                            num_labs=num_labs)
    split_example_counts = {
        "train": len(train_ds),
        "val": len(val_ds),
        "test": len(test_ds),
    }

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    # ===== Build drug graph =====
    print("\n=== Building drug graph ===")
    drug_vocab = resolve_drug_vocab(cohort)
    if drug_vocab is None:
        print("  WARNING: Could not resolve drug vocab schema; ATC edges may be skipped.")

    num_edge_types = cfg["model"].get("num_edge_types", 4)
    ehr_directed = compute_directed_ehr_weights(train_records, num_drugs)
    edge_index, edge_type, edge_weight = build_drug_graph(
        ddi_adj_np, ehr_adj_np,
        cooccur_threshold=0.05,
        drug_vocab=drug_vocab,
        add_self_loops=(num_edge_types >= 3),
        add_atc_edges=(num_edge_types >= 4),
        ehr_weights=ehr_directed,
    )

    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
    edge_weight = edge_weight.to(device)

    max_etype = int(edge_type.max().item()) if edge_type.numel() > 0 else 0
    if max_etype >= num_edge_types:
        raise ValueError(
            f"edge_type contains type {max_etype} but num_edge_types={num_edge_types}. "
            f"Increase num_edge_types in config to at least {max_etype + 1}."
        )

    # ===== Model =====
    print("\n=== Building model ===")
    use_copy = cfg["model"].get("copy_mechanism", True)
    fusion_strategy = args.fusion_strategy or cfg["model"].get("fusion_strategy", "film")
    use_hist_attn = cfg["model"].get("use_historical_attention", True)
    if args.no_historical_attention:
        use_hist_attn = False

    model = MIRROR(
        diag_embeddings=diag_embeds,
        proc_embeddings=proc_embeds,
        drug_embeddings=drug_embeds,
        morgan_fingerprints=morgan_fps,
        ddi_adj=torch.tensor(ddi_adj_np, dtype=torch.float32),
        ehr_adj=torch.tensor(ehr_adj_np, dtype=torch.float32),
        hidden_dim=args.hidden_dim or cfg["model"]["hidden_dim"],
        embed_dim=cfg["model"]["embed_dim"],
        note_proj_dim=args.note_proj_dim or cfg["model"].get("note_proj_dim"),
        lab_proj_dim=cfg["model"].get("lab_proj_dim"),
        lab_input_dim=lab_dim if use_labs else 400,
        encoder_layers=cfg["model"].get("encoder_layers", 2),
        hgt_layers=cfg["model"]["hgt_layers"],
        hgt_heads=cfg["model"]["hgt_heads"],
        num_edge_types=num_edge_types,
        dropout=args.dropout or cfg["model"]["dropout"],
        use_notes=use_notes,
        use_labs=use_labs,
        use_copy=use_copy,
        finetune_embeddings=args.finetune_embeddings or cfg["model"].get("finetune_embeddings", False),
        per_visit_copy=cfg["model"].get("per_visit_copy", True),
        max_visits=cfg["model"].get("max_visits", 30),
        fusion_strategy=fusion_strategy,
        use_historical_attention=use_hist_attn,
        att_tau=cfg["model"].get("att_tau", 20.0),
        gumbel_tau=cfg["model"].get("gumbel_tau", 0.6),
        lab_encoder_type=args.lab_encoder_type or cfg["model"].get("lab_encoder_type", "flat"),
        graph_encoder_type=cfg["model"].get("graph_encoder_type", "drug_gnn"),
        graph_layer_type=cfg["model"].get("graph_layer_type", "gcn"),
        encoder_type=args.encoder_type or cfg["model"].get("encoder_type", "imdr_infused"),
        predictor_type=args.predictor_type or cfg["model"].get("predictor_type", "heidr"),
        aggregator_type=args.aggregator_type or cfg["model"].get("aggregator_type", "last"),
        num_labs=num_labs,
    ).to(device)

    # Set note global mean for centering (fixes ClinicalBERT anisotropy)
    note_mean_path = processed_dir / f"note_global_mean_{cohort_tag}.npy"
    if not note_mean_path.exists():
        note_mean_path = processed_dir / "note_global_mean.npy"
    if note_mean_path.exists():
        note_global_mean = torch.from_numpy(np.load(note_mean_path)).to(device)
        if model.fusion is not None and hasattr(model.fusion, "note_global_mean"):
            model.fusion.note_global_mean = note_global_mean
            print(f"  Set fusion.note_global_mean (norm={note_global_mean.norm():.2f})")
        if hasattr(model.predictor, "note_global_mean"):
            model.predictor.note_global_mean = note_global_mean
            print(f"  Set predictor.note_global_mean (norm={note_global_mean.norm():.2f})")

    param_counts = model.count_parameters()
    for name, counts in param_counts.items():
        print(f"  {name}: {counts['trainable']:,} trainable / {counts['total']:,} total")

    # ===== Loss =====
    pos_weight_cap = float(cfg["training"].get("pos_weight_cap", 5.0))
    pos_weight = compute_pos_weight(train_records, num_drugs, max_cap=pos_weight_cap).to(device)

    bce_w = args.bce_weight if args.bce_weight is not None else float(cfg["training"]["bce_weight"])
    margin_w = args.margin_weight if args.margin_weight is not None else float(cfg["training"]["margin_weight"])
    jac_w = args.soft_jaccard_weight if args.soft_jaccard_weight is not None else float(cfg["training"].get("soft_jaccard_weight", 1.5))

    loss_fn = MIRRORLoss(
        ddi_adj=torch.tensor(ddi_adj_np, dtype=torch.float32).to(device),
        bce_weight=bce_w,
        margin_weight=margin_w,
        label_smoothing=float(cfg["training"].get("label_smoothing", 0.0)),
        pos_weight=pos_weight,
        ddi_weight=float(args.ddi_alpha),
        num_drugs=num_drugs,
        use_focal=bool(cfg["training"].get("use_focal", False)),
        focal_gamma_neg=float(cfg["training"].get("focal_gamma_neg", 2.0)),
        focal_gamma_pos=float(cfg["training"].get("focal_gamma_pos", 0.0)),
        soft_jaccard_weight=jac_w,
    )
    print(f"  [Loss] BCE={bce_w}  SoftJaccard={jac_w}  Margin={margin_w}  DDI={args.ddi_alpha}")
    if args.ddi_alpha > 0:
        print(f"  DDI loss ACTIVE in training objective (alpha={args.ddi_alpha})")
    else:
        print("  DDI monitoring-only (--ddi_alpha=0)")

    # ===== Optimizer & scheduler =====
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate or cfg["training"]["learning_rate"],
        weight_decay=args.weight_decay or cfg["training"]["weight_decay"],
    )

    epochs = args.epochs if args.epochs is not None else cfg["training"]["epochs"]
    lr_min = float(cfg["training"].get("lr_fallback", 1e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=8,
        min_lr=lr_min,
    )

    # ===== Training loop =====
    print("\n=== Training ===")
    best_val_jaccard = 0.0
    patience_counter = 0
    patience = args.patience or cfg["training"]["patience"]
    grad_clip = cfg["model"]["gradient_clip"]
    top_k = cfg["training"]["top_k"]
    threshold = cfg["training"]["threshold"]

    ablation_label = cfg.get("ablation", "full")
    run_name = f"seed{seed}_{ablation_label}_{cohort_tag}"
    best_model_path = results_dir / f"best_model_{run_name}.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer,
            device, edge_index, edge_type, grad_clip,
            edge_weight=edge_weight,
        )

        val_metrics = evaluate_epoch(
            model, val_loader, ddi_adj_np, device,
            edge_index, edge_type, top_k, threshold,
            edge_weight=edge_weight,
        )

        scheduler.step(val_metrics["Jaccard"])
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(
            f"  Epoch {epoch:3d}/{epochs} ({elapsed:.1f}s) | "
            f"Loss: {train_loss['total']:.4f} "
            f"(BCE:{train_loss.get('bce', train_loss.get('focal', 0.0)):.4f} "
            f"Jac:{train_loss.get('soft_jaccard', 0.0):.4f} "
            f"Mrg:{train_loss.get('soft_margin', 0.0):.4f} "
            f"DDI:{train_loss.get('ddi', 0.0):.3f}) | "
            f"LR: {current_lr:.2e}"
        )
        print(
            f"    Val -> Jac: {val_metrics['Jaccard']:.4f}  "
            f"F1: {val_metrics['F1']:.4f}  "
            f"PRAUC: {val_metrics['PRAUC']:.4f}  "
            f"DDI: {val_metrics['DDI Rate']:.4f}  "
            f"Prec: {val_metrics['Precision']:.3f}  "
            f"Rec: {val_metrics['Recall']:.3f}  "
            f"Meds: {val_metrics['Avg Meds']:.1f}/{val_metrics['Avg True Meds']:.1f}"
        )

        if val_metrics["Jaccard"] > best_val_jaccard:
            best_val_jaccard = val_metrics["Jaccard"]
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"    [*] New best val Jaccard: {best_val_jaccard:.4f} — saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    # ===== Test =====
    print("\n=== Testing ===")
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    test_metrics, test_targets, test_probs = evaluate_epoch(
        model, test_loader, ddi_adj_np, device, edge_index, edge_type,
        top_k, threshold, return_raw=True, edge_weight=edge_weight,
    )

    target_ddi = args.target_ddi or cfg["training"].get("target_ddi", 0.06)
    print(f"\n  Test Results (threshold={threshold}):")
    print(f"    Jaccard:    {test_metrics['Jaccard']:.4f}   <- primary metric")
    print(f"    F1:         {test_metrics['F1']:.4f}")
    print(f"    PRAUC:      {test_metrics['PRAUC']:.4f}")
    print(f"    DDI Rate:   {test_metrics['DDI Rate']:.4f}   (target <= {target_ddi:.2f})")
    print(f"    Precision:  {test_metrics['Precision']:.4f}")
    print(f"    Recall:     {test_metrics['Recall']:.4f}")
    print(f"    Avg Meds:   {test_metrics['Avg Meds']:.2f} predicted / "
          f"{test_metrics['Avg True Meds']:.2f} true")

    print(f"\n  Threshold sweep (Jaccard | Avg Meds | DDI Rate):")
    sweep = evaluate_threshold_sweep(test_targets, test_probs, ddi_adj_np)
    best_t = max(sweep, key=lambda t: sweep[t]["Jaccard"])
    avg_true = test_metrics["Avg True Meds"]
    cal_t = min(sweep, key=lambda t: abs(sweep[t]["Avg Meds"] - avg_true))
    for t, m in sweep.items():
        canonical = "  <- canonical" if t == 0.5 else ""
        best_marker = "  <- PEAK Jac" if t == best_t and t != 0.5 else ""
        cal_marker = "  <- COUNT-CAL" if t == cal_t and t != 0.5 else ""
        print(f"    t={t:.2f}: Jac={m['Jaccard']:.4f}  Meds={m['Avg Meds']:.1f}  "
              f"DDI={m['DDI Rate']:.4f}{canonical}{best_marker}{cal_marker}")
    print(f"  Count-calibrated threshold: t={cal_t:.2f}  "
          f"Jac={sweep[cal_t]['Jaccard']:.4f}  Meds={sweep[cal_t]['Avg Meds']:.1f}  "
          f"(true avg: {avg_true:.1f})")

    # ===== Save results =====
    loaded_artifacts = {
        "records": str(records_path),
        "cohort": str(cohort_path),
        "ddi": str(ddi_path),
        "ehr": str(ehr_path),
        "embeddings": str(embed_file),
    }
    if note_data is not None and note_file is not None:
        loaded_artifacts["notes"] = str(note_file)
    if lab_data is not None and lab_file is not None:
        loaded_artifacts["labs"] = str(lab_file)

    result = {
        "seed": seed,
        "ablation": ablation_label,
        "mimic_version": args.mimic_version,
        "mimic4_full": args.mimic4_full,
        "mimic4_sota": args.mimic4_sota,
        "cohort_tag": cohort_tag,
        "best_val_jaccard": best_val_jaccard,
        "test_metrics": test_metrics,
        "data_diagnostics": {
            "split_source": split_source,
            "split_seed_used": split_seed_used,
            "split_patient_counts": split_patient_counts,
            "split_example_counts": split_example_counts,
            "ddi_density": ddi_density,
            "ddi_upper_pairs": ddi_upper_pairs,
            "ddi_total_pairs": total_pairs,
            "loaded_artifacts": loaded_artifacts,
        },
        "param_counts": {k: v for k, v in param_counts.items()},
        "threshold_sweep": {
            str(t): {k: round(v, 6) for k, v in m.items()}
            for t, m in sweep.items()
        },
        "calibrated_threshold": cal_t,
        "calibrated_metrics": {k: round(v, 6) for k, v in sweep[cal_t].items()},
        "config": {
            "hidden_dim": args.hidden_dim or cfg["model"]["hidden_dim"],
            "use_notes": use_notes,
            "use_labs": use_labs,
            "use_copy": use_copy,
            "use_historical_attention": use_hist_attn,
            "fusion_strategy": fusion_strategy,
            "lab_dim": lab_dim,
            "ddi_alpha": args.ddi_alpha,
            "bce_weight": bce_w,
            "soft_jaccard_weight": jac_w,
            "margin_weight": margin_w,
        },
    }
    result_path = results_dir / f"result_{run_name}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()
