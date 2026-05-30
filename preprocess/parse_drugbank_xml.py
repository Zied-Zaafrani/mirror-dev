"""
parse_drugbank_xml.py  —  Run 23 / G1 drug-text preparation

Streams DrugBank full-database XML (1.8 GB), aggregates drug descriptions
per ATC-3 code, then encodes each class with ClinicalBERT → (130, 768) matrix
saved to data/processed/drug_text_embeddings.pkl.

Usage (local, CPU):
  python src/preprocess/parse_drugbank_xml.py \
      --xml    "datasets/drugbank_all_full_database/full database.xml" \
      --processed_dir  data/processed

Output overwrites data/processed/drug_text_embeddings.pkl.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

NS = "http://www.drugbank.ca"


# ---------------------------------------------------------------------------
# 1. Stream-parse DrugBank XML → {atc3: [description strings]}
# ---------------------------------------------------------------------------

def _tag(local: str) -> str:
    return f"{{{NS}}}{local}"


def _text(elem, local: str) -> str:
    child = elem.find(_tag(local))
    return (child.text or "").strip() if child is not None else ""


def parse_drugbank(xml_path: Path) -> dict[str, list[str]]:
    """Return {atc3_code: [text_snippet, ...]} for all drugs in the XML."""
    atc3_texts: dict[str, list[str]] = {}
    n_drugs = 0

    context = ET.iterparse(str(xml_path), events=("end",))
    for event, elem in context:
        if elem.tag != _tag("drug"):
            continue
        # Skip <drug> elements that are nested (metabolite / salt entries)
        parent_type = elem.get("type", "")
        if parent_type not in ("small molecule", "biotech"):
            elem.clear()
            continue

        n_drugs += 1

        # --- collect ATC-3 codes for this drug ---
        atc3_codes: set[str] = set()
        for atc_code_elem in elem.findall(f".//{_tag('atc-code')}"):
            code = (atc_code_elem.get("code") or "").strip()
            if len(code) >= 4:
                atc3_codes.add(code[:4])  # first 4 chars = ATC-3

        if not atc3_codes:
            elem.clear()
            continue

        # --- build a descriptive snippet for this drug ---
        name        = _text(elem, "name")
        description = _text(elem, "description")
        indication  = _text(elem, "indication")
        pharmacodyn = _text(elem, "pharmacodynamics")
        mechanism   = _text(elem, "mechanism-of-action")

        # Prefer richer fields; fall back gracefully.
        parts: list[str] = []
        if name:
            parts.append(name)
        for field in [description, indication, pharmacodyn, mechanism]:
            if field:
                # Trim each field to ~150 chars to avoid one drug dominating the class
                parts.append(field[:300])
        if not parts:
            elem.clear()
            continue

        snippet = " ".join(parts)
        # Collapse whitespace
        snippet = re.sub(r"\s+", " ", snippet).strip()

        for code in atc3_codes:
            atc3_texts.setdefault(code, []).append(snippet)

        elem.clear()

    print(f"  Parsed {n_drugs} drugs, found {len(atc3_texts)} distinct ATC-3 codes")
    return atc3_texts


# ---------------------------------------------------------------------------
# 2. Build one description per ATC-3 code (aligned to voc_final.pkl order)
# ---------------------------------------------------------------------------

def build_class_descriptions(
    atc3_texts: dict[str, list[str]],
    codes: list[str],
) -> tuple[list[str], list[str]]:
    """
    For each code in `codes` (voc order), aggregate drug snippets into one
    class description. Returns (descriptions, source_per_code).
    """
    descriptions: list[str] = []
    sources: list[str] = []

    for code in codes:
        snippets = atc3_texts.get(code, [])
        if snippets:
            # Concatenate up to 5 drugs, total ≤ 512 chars for ClinicalBERT
            agg = " | ".join(snippets[:5])
            agg = agg[:512]
            descriptions.append(f"ATC-3 {code}: {agg}")
            sources.append("drugbank")
        else:
            # Graceful fallback for codes not in DrugBank
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
                "V": "various",
            }.get(code[:1], "unclassified")
            descriptions.append(
                f"ATC-3 class {code}: pharmacological agent acting on the {anatomical}."
            )
            sources.append("fallback")

    n_db  = sum(1 for s in sources if s == "drugbank")
    n_fb  = sum(1 for s in sources if s == "fallback")
    print(f"  {n_db}/{len(codes)} codes from DrugBank, {n_fb} from fallback template")
    return descriptions, sources


# ---------------------------------------------------------------------------
# 3. ClinicalBERT encoding  (same pipeline as note_embeddings)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_cls(
    texts: list[str],
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    device: str = "cpu",
    batch_size: int = 16,
    max_length: int = 128,
) -> np.ndarray:
    tok   = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    vecs: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        enc = tok(
            texts[i : i + batch_size],
            padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        cls = model(**enc).last_hidden_state[:, 0, :].cpu().numpy()
        vecs.append(cls)
    return np.vstack(vecs).astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml",           required=True,
                    help="Path to DrugBank full_database.xml")
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--model_name",    default="emilyalsentzer/Bio_ClinicalBERT")
    ap.add_argument("--device",        default="cpu")
    ap.add_argument("--output_name",   default="drug_text_embeddings.pkl")
    args = ap.parse_args()

    xml_path      = Path(args.xml)
    processed_dir = Path(args.processed_dir)

    # Load vocabulary to get the 130 ATC-3 codes in the correct order
    with open(processed_dir / "voc_final.pkl", "rb") as f:
        voc = pickle.load(f)
    med_voc  = voc["med_voc"]
    idx2word = med_voc["idx2word"]
    n        = len(idx2word)
    codes    = [idx2word[i] for i in range(n)]
    print(f"  Vocabulary: {n} ATC-3 codes")

    # Parse DrugBank XML
    print(f"  Streaming {xml_path} ...")
    atc3_texts = parse_drugbank(xml_path)

    # Build per-class descriptions aligned to voc order
    descriptions, sources = build_class_descriptions(atc3_texts, codes)

    # Encode with ClinicalBERT
    print(f"  Encoding {n} class descriptions on {args.device} ...")
    emb = encode_cls(descriptions, model_name=args.model_name, device=args.device)
    assert emb.shape == (n, 768), f"unexpected shape {emb.shape}"

    out_path = processed_dir / args.output_name
    with open(out_path, "wb") as f:
        pickle.dump({
            "embeddings":      emb,
            "atc3_codes":      codes,
            "source":          "drugbank" if all(s == "drugbank" for s in sources) else "mixed",
            "source_per_code": sources,
            "description":     descriptions,
            "model_name":      args.model_name,
        }, f)

    n_db = sum(1 for s in sources if s == "drugbank")
    print(f"  Wrote {out_path}")
    print(f"  shape={emb.shape}  drugbank={n_db}/{n}  mean_norm={np.linalg.norm(emb,axis=1).mean():.2f}")


if __name__ == "__main__":
    main()
