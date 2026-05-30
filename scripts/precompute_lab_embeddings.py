"""
MIRROR Lab Embedding Precomputer
=================================
Generates PubMedBERT embeddings for every MIRROR lab configuration.

NO DEFAULTS — you must specify --num_labs or --all.
If the required files are missing or the lab count is wrong, this script
CRASHES with a clear message rather than silently falling back to noise.

Output structure (one folder per lab configuration):
    processed/labs/top_N/
        lab_text_embeddings.pt          (N, 4, 768)  — 4 bin states × PubMedBERT 768d
        lab_description_embeddings.npy  (N, 768)     — one embed per lab name
        lab_names.json                               — ordered list of N lab names
        lab_itemids.json                             — ordered list of N MIMIC item IDs
        manifest.json                                — full metadata

For MAX configs (MIMIC-III / MIMIC-IV true maximum):
    processed/labs/max_N/
        (same files as above)

Bin semantics for lab_text_embeddings.pt:
    bin 0 = missing    → zero vector (no text, explicit absence)
    bin 1 = low        → "{name} value is low"
    bin 2 = normal     → "{name} value is normal"
    bin 3 = high       → "{name} value is high"

Description embedding (lab_description_embeddings.npy):
    Just the lab name itself: "{name}" encoded by PubMedBERT.
    This gives semantic diversity (Creatinine ≠ Potassium ≠ Glucose)
    without requiring clinical knowledge for all lab types.

Usage:
    python src/scripts/precompute_lab_embeddings.py --num_labs 200
    python src/scripts/precompute_lab_embeddings.py --num_labs 204 --is_max
    python src/scripts/precompute_lab_embeddings.py --all
    python src/scripts/precompute_lab_embeddings.py --list

MIMIC-IV note:
    If you are using MIMIC-IV data, run with the appropriate pkl path override:
        --pkl_dir /path/to/your/processed/  (defaults to processed/ relative to MIRROR_PROD root)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("precompute_lab_embeddings")

# ── Path resolution ────────────────────────────────────────────────────────────
# Script lives at src/src/scripts/, MIRROR_PROD root is 3 levels up
SCRIPT_DIR   = Path(__file__).resolve().parent
MIRROR_ROOT  = SCRIPT_DIR.parents[2]
PROCESSED_DIR = MIRROR_ROOT / "processed"
LABS_DIR      = PROCESSED_DIR / "labs"

logger.info("MIRROR_ROOT  = %s", MIRROR_ROOT)
logger.info("PROCESSED_DIR = %s", PROCESSED_DIR)
logger.info("LABS_DIR      = %s", LABS_DIR)

# ── Clinical threshold dictionary (the 18 well-known labs) ────────────────────
# These 18 labs have established clinical reference ranges. All other labs get
# the same low/normal/high text template — the difference is only in the
# description embedding (which embeds the lab name directly).
LAB_CLINICAL_THRESHOLDS = {
    "Creatinine", "BUN", "ALT", "AST", "Bilirubin", "Alk Phos",
    "INR", "PT", "PTT", "Sodium", "Potassium", "Magnesium",
    "Calcium", "Glucose", "Albumin", "Lactate", "WBC", "Hemoglobin",
}

# ── PKL discovery ─────────────────────────────────────────────────────────────

def discover_pkl_configs(pkl_dir: Path) -> dict:
    """
    Scan pkl_dir for all lab_vectors pkl files and return a map of:
        lab_count -> {pkl_path, folder_name, is_max}

    Naming conventions supported:
        lab_vectors_{N}labs.pkl       → top_N
        lab_vectors_{N}_MAXlabs.pkl   → max_N
    """
    configs = {}
    for p in sorted(pkl_dir.glob("lab_vectors_*.pkl")):
        name = p.stem  # e.g. "lab_vectors_200labs" or "lab_vectors_446_MAXlabs"
        # Strip prefix
        body = name.replace("lab_vectors_", "")
        is_max = "_MAX" in body.upper()

        # Extract numeric count
        num_str = body.replace("labs", "").replace("_MAX", "").replace("MAX", "")
        try:
            n = int(num_str)
        except ValueError:
            logger.warning("Skipping unrecognised pkl: %s", p.name)
            continue

        folder_name = f"max_{n}" if is_max else f"top_{n}"
        configs[n] = {
            "pkl_path":    p,
            "folder_name": folder_name,
            "is_max":      is_max,
            "num_labs":    n,
        }
        logger.debug("Discovered: %s → folder=%s  is_max=%s", p.name, folder_name, is_max)

    return configs


def load_pkl(pkl_path: Path) -> dict:
    """Load a lab_vectors pkl and validate its structure."""
    import pickle

    logger.info("Loading pkl: %s  (%.1f MB)", pkl_path.name, pkl_path.stat().st_size / 1e6)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    required_keys = {"lab_vectors", "hadm_ids", "lab_names", "lab_itemids",
                     "zscore_means", "zscore_stds"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"pkl {pkl_path.name} is missing keys: {missing}")

    lab_names = data["lab_names"]
    n_labs = len(lab_names)
    vec_dim = data["lab_vectors"].shape[1]
    expected_dim = n_labs * 2  # z-scores + flags

    logger.info("  lab_names: %d entries  (first 5: %s)", n_labs, lab_names[:5])
    logger.info("  lab_vectors: shape=%s  dtype=%s",
                data["lab_vectors"].shape, data["lab_vectors"].dtype)
    logger.info("  expected vector dim for %d labs: %d  actual: %d  → %s",
                n_labs, expected_dim, vec_dim,
                "✓ OK" if vec_dim == expected_dim else "✗ MISMATCH")

    if vec_dim != expected_dim:
        # Allow 4× for trend features (z + flag + slope + var)
        if vec_dim == n_labs * 4:
            logger.info("  Detected trend features (4×N dim) — this is OK.")
        else:
            raise ValueError(
                f"pkl {pkl_path.name}: vector dim {vec_dim} does not match "
                f"2×{n_labs}={expected_dim} or 4×{n_labs}={n_labs*4}"
            )

    # Log clinical overlap
    clinical_overlap = [nm for nm in lab_names if nm in LAB_CLINICAL_THRESHOLDS]
    logger.info("  Labs in clinical threshold dict: %d / %d  (%s)",
                len(clinical_overlap), n_labs, clinical_overlap or "none")

    non_clinical = [nm for nm in lab_names if nm not in LAB_CLINICAL_THRESHOLDS]
    logger.info("  Labs outside clinical threshold dict: %d  (will use same "
                "low/normal/high template)", len(non_clinical))

    return data


# ── PubMedBERT loading ─────────────────────────────────────────────────────────

def load_pubmedbert(device: str = "cpu"):
    """Load PubMedBERT tokenizer + model. Crashes with a clear message if unavailable."""
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        logger.critical(
            "transformers not installed. Cannot generate embeddings.\n"
            "Install it with:  pip install transformers\n"
            "Or run this script on Kaggle (has transformers pre-installed)."
        )
        sys.exit(1)

    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    logger.info("Loading PubMedBERT: %s", model_name)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    logger.info("  PubMedBERT loaded in %.1fs  (device=%s)", time.time() - t0, device)
    return tokenizer, model


def embed_text(texts: list[str], tokenizer, model, device: str,
               batch_size: int = 32) -> np.ndarray:
    """
    Embed a list of text strings using PubMedBERT pooler_output.

    Returns:
        np.ndarray of shape (len(texts), 768)
    """
    all_embeds = []
    n = len(texts)
    logger.info("  Embedding %d strings in batches of %d ...", n, batch_size)

    for start in range(0, n, batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # pooler_output is (batch, 768)
        embeds = outputs.pooler_output.cpu().float().numpy()
        all_embeds.append(embeds)

        if (start // batch_size) % 5 == 0 or start + batch_size >= n:
            logger.info("    Progress: %d / %d strings embedded", min(start + batch_size, n), n)

    result = np.concatenate(all_embeds, axis=0)
    logger.info("  Embedding complete: output shape=%s  dtype=%s", result.shape, result.dtype)
    return result


# ── Core generation ────────────────────────────────────────────────────────────

def generate_lab_embeddings(config: dict, tokenizer, model, device: str,
                             out_root: Path, force: bool = False) -> Path:
    """
    Generate all embedding files for one lab configuration and save them to
    out_root / config['folder_name'] /.

    Returns the output directory path.
    """
    n = config["num_labs"]
    folder_name = config["folder_name"]
    out_dir = out_root / folder_name
    manifest_path = out_dir / "manifest.json"

    logger.info("")
    logger.info("=" * 70)
    logger.info("CONFIG: %s  (%d labs)", folder_name, n)
    logger.info("=" * 70)

    # ── Skip if already done ──
    if manifest_path.exists() and not force:
        logger.info("  Already exists — skipping. Use --force to regenerate.")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("  Output directory: %s", out_dir)

    # ── Load pkl ──
    data = load_pkl(config["pkl_path"])
    lab_names  = data["lab_names"]
    lab_itemids = [int(x) for x in data["lab_itemids"]]
    assert len(lab_names) == n, (
        f"Expected {n} lab names from pkl but got {len(lab_names)}"
    )

    # ── Save lab_names.json and lab_itemids.json ──
    names_path = out_dir / "lab_names.json"
    ids_path   = out_dir / "lab_itemids.json"
    with open(names_path, "w") as f:
        json.dump(lab_names, f, indent=2)
    with open(ids_path, "w") as f:
        json.dump(lab_itemids, f, indent=2)
    logger.info("  Saved lab_names.json (%d names)", len(lab_names))
    logger.info("  Saved lab_itemids.json (%d IDs)", len(lab_itemids))

    # ─────────────────────────────────────────────────────────────────────────
    # 1) lab_description_embeddings.npy  (N, 768)
    #    Embed each lab name as a single sentence: "LABEL"
    #    Provides semantic distinctiveness between labs for per_lab_attn init.
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("── [1/2] Generating lab_description_embeddings.npy ──")
    desc_texts = []
    for i, name in enumerate(lab_names):
        text = name  # Just the lab name → PubMedBERT already understands medical vocabulary
        desc_texts.append(text)
        if i < 5 or i == n - 1:
            logger.info("    lab[%3d] = %r  →  embed text: %r", i, name, text)

    desc_embeds = embed_text(desc_texts, tokenizer, model, device)
    logger.info("  Shape check: expected (%d, 768)  got %s  → %s",
                n, desc_embeds.shape,
                "✓" if desc_embeds.shape == (n, 768) else "✗ WRONG SHAPE")
    assert desc_embeds.shape == (n, 768), (
        f"lab_description_embeddings shape {desc_embeds.shape} != ({n}, 768)"
    )

    desc_path = out_dir / "lab_description_embeddings.npy"
    np.save(str(desc_path), desc_embeds)
    logger.info("  Saved: %s  (%.2f MB)", desc_path.name, desc_path.stat().st_size / 1e6)

    # ─────────────────────────────────────────────────────────────────────────
    # 2) lab_text_embeddings.pt  (N, 4, 768)
    #    bin 0 = missing  → zero vector
    #    bin 1 = low      → "{name} value is low"
    #    bin 2 = normal   → "{name} value is normal"
    #    bin 3 = high     → "{name} value is high"
    #
    #    ALL labs (including non-clinical) get all 4 states.
    #    This is semantically valid: any quantitative lab can be low/normal/high.
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("── [2/2] Generating lab_text_embeddings.pt ──")

    # Build texts for bins 1, 2, 3 (bin 0 is always zero)
    texts_bin1 = []  # low
    texts_bin2 = []  # normal
    texts_bin3 = []  # high

    non_clinical_count = 0
    for i, name in enumerate(lab_names):
        if name in LAB_CLINICAL_THRESHOLDS:
            t1 = f"{name} is low"
            t2 = f"{name} is normal"
            t3 = f"{name} is high"
        else:
            # Non-clinical lab — same template, just using the lab name
            t1 = f"{name} value is low"
            t2 = f"{name} value is normal"
            t3 = f"{name} value is high"
            non_clinical_count += 1

        texts_bin1.append(t1)
        texts_bin2.append(t2)
        texts_bin3.append(t3)

        if i < 5 or i == n - 1:
            logger.info("    lab[%3d] %-35s  bin1=%r", i, repr(name), t1)

    logger.info("  Clinical labs (threshold-aware): %d", n - non_clinical_count)
    logger.info("  Non-clinical labs (generic template): %d", non_clinical_count)
    logger.info("")

    logger.info("  Embedding bin 1 (low) — %d strings ...", len(texts_bin1))
    emb_bin1 = embed_text(texts_bin1, tokenizer, model, device)

    logger.info("  Embedding bin 2 (normal) — %d strings ...", len(texts_bin2))
    emb_bin2 = embed_text(texts_bin2, tokenizer, model, device)

    logger.info("  Embedding bin 3 (high) — %d strings ...", len(texts_bin3))
    emb_bin3 = embed_text(texts_bin3, tokenizer, model, device)

    # Assemble (N, 4, 768):  bin0=zeros, bin1, bin2, bin3
    text_embeds = torch.zeros(n, 4, 768, dtype=torch.float32)
    text_embeds[:, 0, :] = 0.0                              # bin 0: missing → zero
    text_embeds[:, 1, :] = torch.from_numpy(emb_bin1)       # bin 1: low
    text_embeds[:, 2, :] = torch.from_numpy(emb_bin2)       # bin 2: normal
    text_embeds[:, 3, :] = torch.from_numpy(emb_bin3)       # bin 3: high

    logger.info("  Shape check: expected (%d, 4, 768)  got %s  → %s",
                n, tuple(text_embeds.shape),
                "✓" if text_embeds.shape == (n, 4, 768) else "✗ WRONG SHAPE")
    assert text_embeds.shape == (n, 4, 768), (
        f"lab_text_embeddings shape {tuple(text_embeds.shape)} != ({n}, 4, 768)"
    )

    # Sanity: bin 0 must be exactly zero
    assert text_embeds[:, 0, :].abs().sum() == 0, "bin 0 is not zero!"
    logger.info("  Sanity check: bin 0 is all-zero ✓")

    # Sanity: embeddings should be non-trivial
    bin1_norm = text_embeds[:, 1, :].norm(dim=-1).mean().item()
    logger.info("  Sanity check: mean L2 norm of bin-1 embeds = %.4f  (should be ~1.0)", bin1_norm)
    if bin1_norm < 0.01:
        raise RuntimeError(
            f"bin-1 embeddings have near-zero norm ({bin1_norm:.6f}) — "
            "PubMedBERT may have returned garbage. Aborting."
        )

    text_path = out_dir / "lab_text_embeddings.pt"
    torch.save(text_embeds, str(text_path))
    logger.info("  Saved: %s  (%.2f MB)", text_path.name, text_path.stat().st_size / 1e6)

    # ─────────────────────────────────────────────────────────────────────────
    # 3) manifest.json
    # ─────────────────────────────────────────────────────────────────────────
    manifest = {
        "num_labs":          n,
        "folder_name":       folder_name,
        "is_max_config":     config["is_max"],
        "pkl_source":        str(config["pkl_path"]),
        "lab_names":         lab_names,
        "lab_itemids":       lab_itemids,
        "clinical_labs":     [nm for nm in lab_names if nm in LAB_CLINICAL_THRESHOLDS],
        "non_clinical_labs": [nm for nm in lab_names if nm not in LAB_CLINICAL_THRESHOLDS],
        "files": {
            "lab_description_embeddings": str(desc_path),
            "lab_text_embeddings":        str(text_path),
            "lab_names":                  str(names_path),
            "lab_itemids":                str(ids_path),
        },
        "shapes": {
            "lab_description_embeddings": list(desc_embeds.shape),
            "lab_text_embeddings":        list(text_embeds.shape),
        },
        "model_name":   "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        "bin_semantics": {
            "0": "missing — zero vector",
            "1": "low — '{name} [value] is low'",
            "2": "normal — '{name} [value] is normal'",
            "3": "high — '{name} [value] is high'",
        },
        "generated_at":   time.strftime("%Y-%m-%d %H:%M:%S"),
        "mimic_source":   "MIMIC-III" if n <= 210 else "MIMIC-IV",
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("  Saved: manifest.json")

    logger.info("")
    logger.info("  ✓ Config %s COMPLETE", folder_name)
    logger.info("    lab_description_embeddings.npy : %s", desc_embeds.shape)
    logger.info("    lab_text_embeddings.pt         : %s", tuple(text_embeds.shape))
    logger.info("    lab_names.json                 : %d names", n)
    logger.info("    manifest.json                  : written")

    return out_dir


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--num_labs", type=int,
        help="Number of labs to process (must match an existing lab_vectors pkl).",
    )
    group.add_argument(
        "--all", action="store_true",
        help="Process ALL available lab configurations found in pkl_dir.",
    )
    group.add_argument(
        "--list", action="store_true",
        help="List all available lab configurations and exit.",
    )
    p.add_argument(
        "--is_max", action="store_true",
        help="When using --num_labs: flag this as a MAX config (folder = max_N, not top_N).",
    )
    p.add_argument(
        "--pkl_dir", type=Path, default=PROCESSED_DIR,
        help=f"Directory containing lab_vectors_*.pkl files. Default: {PROCESSED_DIR}",
    )
    p.add_argument(
        "--out_dir", type=Path, default=LABS_DIR,
        help=f"Root output directory for labs/ subfolders. Default: {LABS_DIR}",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Regenerate even if output already exists.",
    )
    p.add_argument(
        "--device", default="cpu",
        help="Torch device for PubMedBERT inference. Default: cpu. Use 'cuda' on GPU.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("")
    logger.info("=" * 70)
    logger.info("MIRROR Lab Embedding Precomputer")
    logger.info("=" * 70)
    logger.info("pkl_dir : %s", args.pkl_dir)
    logger.info("out_dir : %s", args.out_dir)
    logger.info("device  : %s", args.device)
    logger.info("force   : %s", args.force)
    logger.info("")

    # ── Discover available configs ──
    configs = discover_pkl_configs(args.pkl_dir)
    if not configs:
        logger.critical(
            "No lab_vectors_*.pkl files found in %s\n"
            "Expected files like: lab_vectors_200labs.pkl, lab_vectors_446_MAXlabs.pkl",
            args.pkl_dir,
        )
        sys.exit(1)

    logger.info("Discovered %d lab configs in %s:", len(configs), args.pkl_dir)
    for n in sorted(configs):
        c = configs[n]
        logger.info("  %-6d → %s  (pkl: %s)", n, c["folder_name"], c["pkl_path"].name)

    # ── --list mode ──
    if args.list:
        print("\nAvailable lab configurations:")
        for n in sorted(configs):
            c = configs[n]
            already = (args.out_dir / c["folder_name"] / "manifest.json").exists()
            status = "✓ already generated" if already else "  not yet generated"
            print(f"  --num_labs {n:4d}  {'--is_max' if c['is_max'] else '        '}  "
                  f"→ {c['folder_name']:12s}  {status}")
        sys.exit(0)

    # ── Resolve which configs to process ──
    if args.all:
        to_process = [configs[n] for n in sorted(configs)]
    else:
        # --num_labs N
        if args.num_labs not in configs:
            available = sorted(configs.keys())
            logger.critical(
                "--num_labs %d not found. Available: %s\n"
                "If you need to generate the pkl first, run:\n"
                "  python src/scripts/generate_phase9_lab_pkls.py",
                args.num_labs, available,
            )
            sys.exit(1)
        cfg = configs[args.num_labs].copy()
        if args.is_max:
            cfg["folder_name"] = f"max_{args.num_labs}"
            cfg["is_max"] = True
        to_process = [cfg]

    logger.info("")
    logger.info("Will process %d config(s):", len(to_process))
    for c in to_process:
        logger.info("  %s  (pkl: %s)", c["folder_name"], c["pkl_path"].name)

    # ── Load PubMedBERT once ──
    tokenizer, bert_model = load_pubmedbert(args.device)

    # ── Process each config ──
    t_total = time.time()
    results = []
    for i, cfg in enumerate(to_process):
        logger.info("")
        logger.info("Processing %d / %d ...", i + 1, len(to_process))
        t0 = time.time()
        try:
            out = generate_lab_embeddings(cfg, tokenizer, bert_model, args.device,
                                          args.out_dir, args.force)
            results.append(("OK", cfg["folder_name"], time.time() - t0, out))
        except Exception as e:
            logger.error("FAILED: %s — %s", cfg["folder_name"], e, exc_info=True)
            results.append(("FAIL", cfg["folder_name"], time.time() - t0, str(e)))

    # ── Final summary ──
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY  (total time: %.1fs)", time.time() - t_total)
    logger.info("=" * 70)
    ok  = [r for r in results if r[0] == "OK"]
    bad = [r for r in results if r[0] == "FAIL"]
    for status, name, elapsed, _ in results:
        logger.info("  %-5s  %-14s  %.1fs", status, name, elapsed)
    logger.info("")
    logger.info("  %d succeeded, %d failed", len(ok), len(bad))

    if bad:
        logger.critical("%d configs FAILED — see errors above.", len(bad))
        sys.exit(1)

    logger.info("")
    logger.info("All embedding files written to: %s", args.out_dir)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Upload processed/labs/ to your Kaggle dataset.")
    logger.info("  2. Run training with: python src/train.py --num_labs 200")
    logger.info("  3. The model will auto-load from processed/labs/top_200/")


if __name__ == "__main__":
    main()
