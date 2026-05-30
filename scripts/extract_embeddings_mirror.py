"""
Extract patient embeddings from a trained MIRROR checkpoint.

Unlike extract_embeddings.py (which only works with PretrainMIRROR checkpoints),
this script works with main MIRROR training checkpoints produced by train.py.

Captures model._aux_fused after each forward pass — the post-fusion patient
representation that blends EHR + notes + labs into a single hidden_dim vector.
This is the correct representation for cross-patient retrieval (same manifold as
the predictor query).

Output per split: patient_embeddings_{split}.pkl with keys:
  embeddings  — (N, hidden_dim) float32 numpy array
  labels      — (N, num_drugs) float32 multi-hot array
  hadm_ids    — (N,) int64 array
  split       — "train" / "val" / "test"
  hidden_dim  — int
  checkpoint  — str (source checkpoint path)

Usage:
  python src/scripts/extract_embeddings_mirror.py \
    --config src/config.yaml \
    --checkpoint data/embeddings/run41_best_checkpoint.pt \
    --output_dir data/embeddings \
    --encoder_type transformer \
    --device cuda

  # Kaggle: paths passed explicitly
  python src/scripts/extract_embeddings_mirror.py \
    --config src/config.yaml \
    --checkpoint /kaggle/input/run41/run41_best_checkpoint.pt \
    --processed_dir data/processed \
    --embeddings_dir data/embeddings \
    --output_dir data/embeddings \
    --encoder_type transformer \
    --device cuda
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

# Support both package-style and script-style imports
try:
    from ..dataset import MIRRORDataset, collate_fn
    from ..model.model import MIRROR
    from ..split_protocol import compute_split_indices
except ImportError:
    _src = Path(__file__).resolve().parent.parent
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    from dataset import MIRRORDataset, collate_fn
    from model.model import MIRROR
    from split_protocol import compute_split_indices


# ---------------------------------------------------------------------------
# Helpers mirrored from train.py
# ---------------------------------------------------------------------------

def _load_embeddings(embed_file: Path, allow_unsafe: bool = True):
    try:
        return torch.load(embed_file, map_location="cpu", weights_only=True)
    except Exception:
        if not allow_unsafe:
            raise
        return torch.load(embed_file, map_location="cpu", weights_only=False)


def _resolve_drug_vocab(cohort: dict):
    raw = cohort.get("drug_vocab") or cohort.get("med_voc", {})
    if isinstance(raw, dict) and "idx2word" in raw:
        return raw
    if isinstance(raw, dict):
        # Detect direction: if values are str → already idx2word
        sample = next(iter(raw.values()), None)
        if isinstance(sample, str):
            return {"idx2word": {int(k): v for k, v in raw.items()},
                    "word2idx": {v: int(k) for k, v in raw.items()}}
        else:
            return {"idx2word": {v: k for k, v in raw.items()},
                    "word2idx": {k: v for k, v in raw.items()}}
    return None


def _build_drug_graph(ddi_adj_np, ehr_adj_np, num_edge_types: int, drug_vocab,
                      train_records, num_drugs: int, device):
    # build_drug_graph and compute_directed_ehr_weights are defined in train.py
    try:
        from train import build_drug_graph, compute_directed_ehr_weights
    except ImportError:
        import importlib.util, os
        spec = importlib.util.spec_from_file_location(
            "train", os.path.join(os.path.dirname(__file__), "..", "train.py")
        )
        _train = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_train)
        build_drug_graph = _train.build_drug_graph
        compute_directed_ehr_weights = _train.compute_directed_ehr_weights

    ehr_directed = compute_directed_ehr_weights(train_records, num_drugs)
    edge_index, edge_type, edge_weight = build_drug_graph(
        ddi_adj_np, ehr_adj_np, cooccur_threshold=0.05,
        drug_vocab=drug_vocab,
        add_self_loops=(num_edge_types >= 3),
        add_atc_edges=(num_edge_types >= 4),
        ehr_weights=ehr_directed,
    )
    return edge_index.to(device), edge_type.to(device), edge_weight.to(device)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_split(model: nn.Module, loader, dataset, edge_index, edge_type,
                  device, split_name: str):
    """Run inference and collect _aux_fused, labels, and hadm_ids.

    hadm_ids come from dataset.examples (not the batch dict, which doesn't
    carry them). DataLoader with shuffle=False preserves example order.
    """
    model.eval()

    # Pre-collect hadm_ids in dataset order (shuffle=False guarantees this).
    # dataset.examples = [(patient_idx, target_visit_idx), ...]
    # target_visit = records[patient_idx][target_visit_idx]
    # hadm_id      = target_visit[3] if len(target_visit) > 3 else -1
    hadm_ids_ordered = []
    for pidx, tidx in dataset.examples:
        visit = dataset.records[pidx][tidx]   # pidx indexes into the split's records, not full records
        hadm_ids_ordered.append(int(visit[3]) if len(visit) > 3 else -1)

    all_fused  = []
    all_labels = []

    for batch in loader:
        diag_seq      = [t.to(device) for t in batch["diag_seq"]]
        proc_seq      = [t.to(device) for t in batch["proc_seq"]]
        diag_mask     = [t.to(device) for t in batch["diag_mask_seq"]]
        proc_mask     = [t.to(device) for t in batch["proc_mask_seq"]]
        lengths       = batch["lengths"].to(device)
        drug_hist     = batch["drug_history"].to(device)
        note_embed    = batch["note_embed"].to(device)
        has_note      = batch["has_note"].to(device)
        lab_vector    = batch["lab_vector"].to(device)
        has_lab       = batch["has_lab"].to(device)
        med_per_visit = batch["med_per_visit"].to(device) if "med_per_visit" in batch else None
        labels        = batch["target"]   # key is "target" in collate_fn, not "labels"

        _ = model(
            diag_seq, proc_seq, diag_mask, proc_mask, lengths,
            note_embed, lab_vector, has_note, has_lab,
            drug_hist, edge_index, edge_type,
            med_per_visit=med_per_visit,
        )

        all_fused.append(model._aux_fused.detach().cpu().float().numpy())
        all_labels.append(labels.cpu().float().numpy())

    embeddings = np.concatenate(all_fused,  axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    hadm_arr   = np.array(hadm_ids_ordered, dtype=np.int64)

    print(f"  [{split_name}] {len(embeddings)} samples, emb={embeddings.shape}, "
          f"hadm coverage={np.sum(hadm_arr >= 0)}/{len(hadm_arr)}")
    return embeddings, labels_arr, hadm_arr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from a MIRROR training checkpoint")
    parser.add_argument("--config",      required=True,  help="Path to config.yaml")
    parser.add_argument("--checkpoint",  required=True,  help="Path to best_model_*.pt from train.py")
    parser.add_argument("--output_dir",  required=True,  help="Where to save patient_embeddings_*.pkl")
    parser.add_argument("--processed_dir",  default=None)
    parser.add_argument("--embeddings_dir", default=None)
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--encoder_type", default=None,
                        help="Must match the encoder used during training (e.g. transformer)")
    parser.add_argument("--fusion_strategy", default=None,
                        help="Must match training (default: film)")
    parser.add_argument("--batch_size",  type=int, default=32)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    processed_dir  = Path(args.processed_dir  or cfg["paths"]["processed_dir"])
    embeddings_dir = Path(args.embeddings_dir or cfg["paths"]["embeddings_dir"])
    output_dir     = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Device: {device} | Seed: {args.seed}")
    print(f"Checkpoint: {args.checkpoint}")

    # --- Data loading ---
    print("\n=== Loading data ===")
    with open(processed_dir / "records_final.pkl",    "rb") as f: records   = pickle.load(f)
    with open(processed_dir / "cohort_mimic3.pkl",    "rb") as f: cohort    = pickle.load(f)
    with open(processed_dir / "ddi_A_final.pkl",      "rb") as f: ddi_adj   = pickle.load(f)
    with open(processed_dir / "ehr_adj_final.pkl",    "rb") as f: ehr_adj   = pickle.load(f)

    num_drugs = cohort["num_drugs"]
    num_diag  = cohort["num_diag"]
    num_proc  = cohort["num_proc"]
    print(f"  Records: {len(records)} patients, {num_drugs} drugs, "
          f"{num_diag} diag, {num_proc} proc")

    embed_data  = _load_embeddings(embeddings_dir / "code_embeddings.pt")
    diag_embeds = torch.tensor(embed_data["diag_embeddings"])
    proc_embeds = torch.tensor(embed_data["proc_embeddings"])
    drug_embeds = torch.tensor(embed_data["drug_embeddings"])
    morgan_fps  = torch.tensor(embed_data["morgan_fingerprints"])

    note_data = None
    use_notes = cfg["preprocessing"].get("note_method", "none") != "none"
    if use_notes:
        note_file = processed_dir / "note_embeddings_mimic3.pkl"
        if note_file.exists():
            with open(note_file, "rb") as f:
                note_data = pickle.load(f)
            print(f"  Note embeddings loaded: {note_data['embeddings'].shape}")
        else:
            print("  NOTE: note_embeddings_mimic3.pkl not found — notes disabled")
            use_notes = False

    lab_data = None
    lab_dim  = cfg["preprocessing"].get("lab_dim", 36)
    use_labs = lab_dim > 0
    if use_labs:
        # FIX-LAB-001: Use the per-N lab_vectors pkl, not the deleted lab_data_mimic3.pkl.
        _num_labs_embed = lab_dim // 2
        lab_file = processed_dir / f"lab_vectors_{_num_labs_embed}labs.pkl"
        if lab_file.exists():
            with open(lab_file, "rb") as f:
                lab_data = pickle.load(f)
            lab_key = "lab_vectors_72d" if lab_dim == 72 and "lab_vectors_72d" in lab_data else "lab_vectors"
            print(f"  Lab data loaded: {lab_file.name} (dim={lab_dim}, key={lab_key})")
        else:
            raise FileNotFoundError(
                f"\n[FATAL] Lab pkl not found: {lab_file}\n"
                f"  config.yaml lab_dim={lab_dim} → expects {_num_labs_embed} labs.\n"
                f"  Available: {list(processed_dir.glob('lab_vectors_*.pkl'))}"
            )

    # --- Split ---
    print("\n=== Splitting ===")
    # FIX-SPLIT-001: Use config split_mode, not hardcoded "hidr_vita".
    # Embedding extraction must use the SAME patient partition as training,
    # otherwise retrieval neighbours are drawn from a different split assignment.
    _embed_split_mode = cfg["training"].get("split_mode", "cohort")
    split = compute_split_indices(
        num_records=len(records), cohort=cohort,
        split_mode=_embed_split_mode, seed=args.seed,
        require_cohort_indices=False,
    )
    print(f"  [FIX-SPLIT-001] split_mode={_embed_split_mode!r} (from config, not hardcoded hidr_vita)")
    train_records = [records[i] for i in split.train_idx]
    val_records   = [records[i] for i in split.val_idx]
    test_records  = [records[i] for i in split.test_idx]
    print(f"  Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")

    # --- Drug graph ---
    print("\n=== Building drug graph ===")
    num_edge_types = cfg["model"].get("num_edge_types", 4)
    drug_vocab = _resolve_drug_vocab(cohort)
    edge_index, edge_type, edge_weight = _build_drug_graph(
        ddi_adj, ehr_adj, num_edge_types, drug_vocab,
        train_records, num_drugs, device,
    )

    # --- Datasets + loaders ---
    bs = args.batch_size
    lab_key_use = "lab_vectors_72d" if lab_dim == 72 and lab_data and "lab_vectors_72d" in lab_data else "lab_vectors"

    def _make_ds(recs):
        return MIRRORDataset(recs, cohort, note_data, lab_data, num_drugs, lab_key_use,
                             use_temporal_decay=False,
                             use_history_notes=False,
                             retrieval_data=None,
                             visit_level_scramble=True)  # all non-first visits → matches compute_similarity.py

    from torch.utils.data import DataLoader
    train_loader = DataLoader(_make_ds(train_records), batch_size=bs, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(_make_ds(val_records),   batch_size=bs, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(_make_ds(test_records),  batch_size=bs, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    # --- Build model (must match run41 config exactly) ---
    print("\n=== Building model ===")
    use_copy = cfg["model"].get("copy_mechanism", True)
    fusion   = args.fusion_strategy or cfg["model"].get("fusion_strategy", "film")
    enc_type = args.encoder_type    or cfg["model"].get("encoder_type", "gru")
    print(f"  encoder={enc_type}, fusion={fusion}, "
          f"notes={use_notes}, labs={use_labs}, copy={use_copy}")

    model = MIRROR(
        diag_embeddings=diag_embeds,
        proc_embeddings=proc_embeds,
        drug_embeddings=drug_embeds,
        morgan_fingerprints=morgan_fps,
        ddi_adj=torch.tensor(ddi_adj, dtype=torch.float32),
        hidden_dim=cfg["model"]["hidden_dim"],
        embed_dim=cfg["model"]["embed_dim"],
        note_proj_dim=cfg["model"]["note_proj_dim"],
        lab_proj_dim=cfg["model"]["lab_proj_dim"],
        lab_input_dim=lab_dim if use_labs else 36,
        encoder_layers=cfg["model"].get("encoder_layers", 2),
        hgt_layers=cfg["model"]["hgt_layers"],
        hgt_heads=cfg["model"]["hgt_heads"],
        num_edge_types=num_edge_types,
        dropout=cfg["model"]["dropout"],
        use_notes=use_notes,
        use_labs=use_labs,
        use_copy=use_copy,
        finetune_embeddings=cfg["model"].get("finetune_embeddings", False),
        per_visit_copy=cfg["model"].get("per_visit_copy", True),
        max_visits=cfg["model"].get("max_visits", 30),
        fusion_strategy=fusion,
        use_retrieval=False,
        retrieval_weight_init=1.0,
        use_camo=False,
        use_multi_view=False,
        use_drug_text=False,
        drug_text_embeddings=None,
        drug_text_weight_init=1.0,
        alignment_margin=0.2,
        use_hist_notes=False,
        use_historical_attention=cfg["model"].get("use_historical_attention", True),
        att_tau=cfg["model"].get("att_tau", 20.0),
        gumbel_tau=cfg["model"].get("gumbel_tau", 0.6),
        lab_encoder_type=cfg["model"].get("lab_encoder_type", "flat"),
        gnn_type=cfg["model"].get("gnn_type", "hgt"),
        use_tripartite=False,
        use_dcma=False,
        medgcn_layer_type=cfg["model"].get("medgcn_layer_type", "han"),
        use_lab_nodes=False,
        encoder_type=enc_type,
        predictor_type=cfg["model"].get("predictor_type", "dot_product"),
    ).to(device)

    # Load note_global_mean if present (same as train.py)
    # FIX-B18: must set BOTH fusion.note_global_mean AND predictor.note_global_mean.
    # Setting only fusion left the H2 scoring head computing `notes - zeros` (no centering),
    # operating on raw anisotropic ClinicalBERT embeddings (cos_sim ≈ 0.95) — every
    # extracted embedding had a corrupted H2 pathway.
    note_mean_path = processed_dir / "note_global_mean.npy"
    if note_mean_path.exists():
        note_global_mean = torch.from_numpy(np.load(note_mean_path)).float().to(device)
        if model.fusion is not None and hasattr(model.fusion, "note_global_mean"):
            model.fusion.note_global_mean = note_global_mean
        if hasattr(model, "predictor") and hasattr(model.predictor, "note_global_mean"):
            model.predictor.note_global_mean = note_global_mean

    # Load checkpoint
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"  Checkpoint loaded: {Path(args.checkpoint).name}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # --- Extract ---
    print("\n=== Extracting embeddings ===")
    checkpoint_name = str(args.checkpoint)

    splits = [
        ("train", train_loader, _make_ds(train_records)),
        ("val",   val_loader,   _make_ds(val_records)),
        ("test",  test_loader,  _make_ds(test_records)),
    ]
    for split_name, loader, ds in splits:
        embs, labs, hadms = extract_split(
            model, loader, ds, edge_index, edge_type, device, split_name,
        )
        out = {
            "embeddings": embs,
            "labels":     labs,
            "hadm_ids":   hadms.astype(np.int64),
            "split":      split_name,
            "hidden_dim": embs.shape[1],
            "checkpoint": checkpoint_name,
        }
        out_path = output_dir / f"patient_embeddings_{split_name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(out, f)
        print(f"  Saved {out_path.name}  ({embs.shape})")

    print("\nDone.")


if __name__ == "__main__":
    main()
