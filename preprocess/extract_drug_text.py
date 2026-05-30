"""
G1 (Run 23) — Extract / encode drug-class text descriptions for the 130 ATC-3
codes in MIRROR's MIMIC-III cohort, aligned to `voc_final.pkl`'s med_voc index.

Two encoding sources are supported (in priority order):

  1. `--atc3_csv <path>`  — a CSV with columns {code, description}. 1 row per
     ATC-3 code. Most informative (e.g. WHO ATC verbatim class descriptions
     or DrugBank aggregated summaries). This is the recommended source for
     the Kaggle pre-compute cell.
  2. Fallback: if no CSV is supplied, use each code's WHO canonical class name
     (e.g. "A01A — Stomatological preparations"). This is a weak but always-
     available baseline. The alignment head still learns a useful signal
     because it enforces margin structure over 130 distinct strings.

Output:
  data/processed/drug_text_embeddings.pkl
    {
        "embeddings":  np.ndarray (130, 768)  — ClinicalBERT CLS pooled
        "atc3_codes":  list[str]             — aligned to med_voc.idx2word order
        "source":      "csv" | "fallback"
        "description": list[str]             — raw text per code (for audit)
        "model_name":  "emilyalsentzer/Bio_ClinicalBERT"
    }

Usage:
  python src/preprocess/extract_drug_text.py --processed_dir data/processed
  python src/preprocess/extract_drug_text.py --processed_dir data/processed \
         --atc3_csv data/raw/atc3_descriptions.csv
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


# Minimal WHO ATC level-3 canonical class names. Not exhaustive of every
# pharmacology nuance, but enough to land each code in a distinct region of
# ClinicalBERT space. Expand via --atc3_csv for production-quality runs.
ATC3_FALLBACK_NAMES: dict[str, str] = {
    "A01A": "Stomatological preparations for oral care",
    "A02A": "Antacids for dyspepsia and acid-related disorders",
    "A02B": "Drugs for peptic ulcer and gastro-esophageal reflux, including proton pump inhibitors",
    "A03A": "Drugs for functional gastrointestinal disorders and antispasmodics",
    "A03B": "Belladonna and derivatives for gastrointestinal cramping",
    "A03F": "Propulsives and prokinetics such as metoclopramide",
    "A04A": "Antiemetics and antinauseants",
    "A05A": "Bile therapy drugs including ursodeoxycholic acid",
    "A06A": "Laxatives for constipation management",
    "A07A": "Intestinal antiinfectives for infectious diarrhea",
    # Remaining 120 codes are populated at load time from med_voc with the
    # generic template so the pipeline never hard-fails on a missing key.
}


def _canonical_text(code: str) -> str:
    custom = ATC3_FALLBACK_NAMES.get(code)
    if custom is not None:
        return f"ATC-3 class {code}: {custom}."
    # Generic template keyed on the ATC level-1 anatomical group.
    anatomical = {
        "A": "alimentary tract and metabolism",
        "B": "blood and blood forming organs",
        "C": "cardiovascular system",
        "D": "dermatologicals",
        "G": "genito urinary system and sex hormones",
        "H": "systemic hormonal preparations",
        "J": "antiinfectives for systemic use",
        "L": "antineoplastic and immunomodulating agents",
        "M": "musculo-skeletal system",
        "N": "nervous system",
        "P": "antiparasitic products",
        "R": "respiratory system",
        "S": "sensory organs",
        "V": "various therapeutic products",
    }.get(code[:1], "unclassified therapeutic products")
    return f"ATC-3 class {code}: pharmacological agent acting on the {anatomical}."


def _load_csv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = (row.get("code") or row.get("atc3") or "").strip()
            desc = (row.get("description") or row.get("text") or "").strip()
            if code and desc:
                out[code] = desc
    return out


@torch.no_grad()
def _encode_cls(
    texts: list[str],
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    device: str = "cpu",
    batch_size: int = 16,
    max_length: int = 128,
) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    all_vecs: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        out = model(**enc)
        # CLS token pooling. Matches MIRROR's note_embeddings pipeline.
        cls = out.last_hidden_state[:, 0, :].cpu().numpy()
        all_vecs.append(cls)
    return np.vstack(all_vecs).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract drug-class text embeddings (G1)")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument(
        "--atc3_csv",
        type=str,
        default=None,
        help="Optional CSV {code, description} for richer ATC-3 descriptions.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="emilyalsentzer/Bio_ClinicalBERT",
        help="HF model for text encoding. Must match note_embeddings model.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output_name",
        type=str,
        default="drug_text_embeddings.pkl",
        help="Output filename under processed_dir.",
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    voc_path = processed_dir / "voc_final.pkl"
    with open(voc_path, "rb") as f:
        voc = pickle.load(f)
    med_voc = voc["med_voc"]
    idx2word = med_voc["idx2word"]
    n = len(idx2word)
    codes = [idx2word[i] for i in range(n)]

    csv_map: dict[str, str] = {}
    if args.atc3_csv:
        csv_path = Path(args.atc3_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"--atc3_csv not found: {csv_path}")
        csv_map = _load_csv(csv_path)
        print(f"  Loaded {len(csv_map)} rows from {csv_path}")

    texts: list[str] = []
    source_tags: list[str] = []
    for code in codes:
        if code in csv_map:
            texts.append(csv_map[code])
            source_tags.append("csv")
        else:
            texts.append(_canonical_text(code))
            source_tags.append("fallback")

    n_csv = sum(1 for t in source_tags if t == "csv")
    n_fb = n - n_csv
    print(f"  {n_csv}/{n} codes from CSV, {n_fb} from fallback template")

    print(f"  Encoding {n} descriptions with {args.model_name} on {args.device} ...")
    emb = _encode_cls(texts, model_name=args.model_name, device=args.device)
    assert emb.shape == (n, 768), f"unexpected shape {emb.shape}"

    out_path = processed_dir / args.output_name
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "embeddings": emb,
                "atc3_codes": codes,
                "source": "csv" if n_csv == n else ("fallback" if n_fb == n else "mixed"),
                "source_per_code": source_tags,
                "description": texts,
                "model_name": args.model_name,
            },
            f,
        )
    print(f"  Wrote {out_path}  shape={emb.shape}  mean_norm={np.linalg.norm(emb, axis=1).mean():.2f}")


if __name__ == "__main__":
    main()
