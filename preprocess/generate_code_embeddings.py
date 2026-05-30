"""
Generate code_embeddings.pt for any MIRROR cohort (MIMIC-III or MIMIC-IV).

This script handles the MIMIC-IV ICD prefix problem (Bug A):
  - MIMIC-III voc_final.pkl stores codes as raw ICD-9 strings: "4280", "V3000"
  - MIMIC-IV voc_mimic4.pkl stores codes as versioned strings:  "9_4280", "10_I50.0"
  - When looking up ICD descriptions, we strip the "9_" / "10_" prefix first
  - PubMedBERT then embeds the text description (not the raw code string)
    → semantically equivalent ICD-9/ICD-10 codes get similar embeddings

Output: code_embeddings_{cohort_tag}.pt  (or code_embeddings.pt for mimic3)
  {
    "diag_embeddings":          (num_diag, 768) float32
    "proc_embeddings":          (num_proc, 768) float32
    "drug_embeddings":          (num_drugs, 768) float32
    "morgan_fingerprints":      (num_drugs, 256) float32
    "diag_embeddings_official": same (alias expected by train.py --ablation official)
    "proc_embeddings_official": same
    "drug_embeddings_official": same
    "embed_dim": 768
    "morgan_bits": 256
    "pubmedbert_model": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    "cohort_tag": str
  }

Usage (MIMIC-III):
  python generate_code_embeddings.py \\
    --voc_file      data/processed/voc_final.pkl \\
    --mimic_dir     DATASETS/mimic-iii-clinical-database-1.4 \\
    --external_dir  data/external \\
    --output_dir    data/embeddings \\
    --cohort_tag    mimic3 \\
    --device        cpu

Usage (MIMIC-IV ICD-9 cohort):
  python generate_code_embeddings.py \\
    --voc_file      data/processed/voc_mimic4.pkl \\
    --mimic_dir     DATASETS/mimic-iv-3.1 \\
    --external_dir  data/external \\
    --output_dir    data/embeddings \\
    --cohort_tag    mimic4 \\
    --device        cpu

Usage (MIMIC-IV full cohort):
  python generate_code_embeddings.py \\
    --voc_file      data/processed/voc_mimic4_full.pkl \\
    --mimic_dir     DATASETS/mimic-iv-3.1 \\
    --external_dir  data/external \\
    --output_dir    data/embeddings \\
    --cohort_tag    mimic4_full \\
    --device        cpu
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# ICD prefix handling
# ---------------------------------------------------------------------------

def _strip_version_prefix(code: str) -> tuple[str, int]:
    """
    Strip the MIMIC-IV version prefix from a code string.

    "9_4280"   → ("4280",  9)   ICD-9
    "10_I50.0" → ("I50.0", 10)  ICD-10
    "4280"     → ("4280",  9)   MIMIC-III (no prefix → assume ICD-9)
    """
    if code.startswith("10_"):
        return code[3:], 10
    if code.startswith("9_"):
        return code[2:], 9
    # MIMIC-III or already stripped
    return code, 9


# ---------------------------------------------------------------------------
# ICD description lookup
# ---------------------------------------------------------------------------

def _load_icd_descriptions(mimic_dir: Path, mimic_version: int = 3) -> dict[str, str]:
    """
    Load ICD code → long_title mapping.
    Returns dict keyed by PLAIN code (no version prefix):
      {"4280": "Congestive heart failure, unspecified", "I50.0": "Congestive heart failure", ...}

    For MIMIC-III: single ICD-9 reference table
    For MIMIC-IV:  combined ICD-9 + ICD-10 reference table (hosp/)
    """
    desc: dict[str, str] = {}

    import pandas as pd

    if mimic_version == 3:
        for fname, code_col in [
            ("D_ICD_DIAGNOSES.csv.gz",  "ICD9_CODE"),
            ("D_ICD_PROCEDURES.csv.gz", "ICD9_CODE"),
        ]:
            fpath = mimic_dir / fname
            if fpath.exists():
                df = pd.read_csv(fpath, usecols=[code_col, "LONG_TITLE"],
                                 compression="gzip", dtype=str)
                df = df.dropna(subset=[code_col, "LONG_TITLE"])
                desc.update(dict(zip(df[code_col].str.strip(), df["LONG_TITLE"].str.strip())))
                print(f"  Loaded {len(df):,} ICD descriptions from {fname}")
        # Also try MIMIC-III-style short titles as fallback
        for fname, code_col in [
            ("D_ICD_DIAGNOSES.csv.gz",  "ICD9_CODE"),
            ("D_ICD_PROCEDURES.csv.gz", "ICD9_CODE"),
        ]:
            fpath = mimic_dir / fname
            if fpath.exists():
                df = pd.read_csv(fpath, usecols=[code_col, "SHORT_TITLE"],
                                 compression="gzip", dtype=str)
                df = df.dropna(subset=[code_col, "SHORT_TITLE"])
                for code, title in zip(df[code_col].str.strip(), df["SHORT_TITLE"].str.strip()):
                    if code not in desc:
                        desc[code] = title
    else:
        # MIMIC-IV: hosp/d_icd_diagnoses.csv.gz and hosp/d_icd_procedures.csv.gz
        for fname in ["d_icd_diagnoses.csv.gz", "d_icd_procedures.csv.gz"]:
            fpath = mimic_dir / "hosp" / fname
            if fpath.exists():
                df = pd.read_csv(fpath,
                                 usecols=["icd_code", "icd_version", "long_title"],
                                 compression="gzip", dtype=str)
                df = df.dropna(subset=["icd_code", "long_title"])
                df["icd_code"] = df["icd_code"].str.strip()
                df["long_title"] = df["long_title"].str.strip()
                desc.update(dict(zip(df["icd_code"], df["long_title"])))
                print(f"  Loaded {len(df):,} ICD descriptions from {fname}")

    print(f"  Total ICD descriptions: {len(desc):,}")
    return desc


def _get_description(code_raw: str, icd_descriptions: dict[str, str]) -> str:
    """
    Get a text description for an ICD code.
    Tries several normalisation variants to maximise lookup rate.
    Falls back to the raw code string if nothing found.
    """
    raw = code_raw.strip()

    # 1. Direct lookup
    if raw in icd_descriptions:
        return icd_descriptions[raw]

    # 2. Normalise: remove dots (ICD-10 codes sometimes stored with/without dots)
    nodot = raw.replace(".", "")
    if nodot in icd_descriptions:
        return icd_descriptions[nodot]

    # 3. Try zero-padded form (ICD-9 codes can be stored with/without leading zeros)
    padded = raw.zfill(5)
    if padded in icd_descriptions:
        return icd_descriptions[padded]

    # 4. Try without zero-padding
    unpadded = raw.lstrip("0") or "0"
    if unpadded in icd_descriptions:
        return icd_descriptions[unpadded]

    # 5. Fallback: use the raw code string — BERT will still produce a valid vector
    return f"ICD code {raw}"


# ---------------------------------------------------------------------------
# Drug (ATC-3) description lookup
# ---------------------------------------------------------------------------

def _load_atc3_descriptions(external_dir: Path) -> dict[str, str]:
    """
    Load ATC-3 → description mapping.
    Falls back to the code string if no dedicated description file is found.
    """
    desc: dict[str, str] = {}

    # Try WHO ATC description file (various naming conventions)
    for candidate in ["atc3_descriptions.csv", "atc_descriptions.csv", "drug-atc.csv"]:
        fpath = external_dir / candidate
        if fpath.exists():
            import pandas as pd
            try:
                df = pd.read_csv(fpath, dtype=str)
                df.columns = [c.strip().upper() for c in df.columns]
                code_col = next((c for c in df.columns if "ATC" in c), None)
                desc_col = next((c for c in df.columns if "NAME" in c or "TITLE" in c or "DESC" in c), None)
                if code_col and desc_col:
                    df = df.dropna(subset=[code_col, desc_col])
                    # Keep ATC-3 level (4 chars)
                    df = df[df[code_col].str.len() == 4]
                    desc.update(dict(zip(df[code_col].str.strip(), df[desc_col].str.strip())))
                    print(f"  Loaded {len(desc):,} ATC-3 descriptions from {candidate}")
                    break
            except Exception:
                pass

    if not desc:
        print("  No ATC description file found — using 'ATC code XXXX' fallback")

    return desc


# ---------------------------------------------------------------------------
# PubMedBERT embedding
# ---------------------------------------------------------------------------

def _embed_texts(
    texts: list[str],
    model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """
    Encode a list of text strings using PubMedBERT [CLS] pooling.
    Returns (N, 768) float32 numpy array.
    """
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        raise ImportError(
            "transformers not installed. Run: pip install transformers"
        )

    print(f"  Loading {model_name} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name).to(device).eval()

    all_embeds = []
    total = len(texts)
    print(f"  Encoding {total:,} texts (batch_size={batch_size}) ...")

    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch_texts = texts[start : start + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(device)
            out = bert_model(**enc)
            cls = out.last_hidden_state[:, 0, :].cpu().float().numpy()
            all_embeds.append(cls)
            pct = min(start + batch_size, total) / total * 100
            print(f"\r    {min(start + batch_size, total):>6,}/{total:,} ({pct:5.1f}%)", end="", flush=True)

    print()
    return np.concatenate(all_embeds, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Morgan fingerprints
# ---------------------------------------------------------------------------

def _compute_morgan_fingerprints(
    atc3_list: list[str],
    external_dir: Path,
    morgan_bits: int = 256,
    morgan_radius: int = 2,
) -> np.ndarray:
    """Compute Morgan fingerprints for each ATC-3 drug."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        raise ImportError("rdkit not installed. Run: pip install rdkit")

    idx2smiles_path = external_dir / "idx2SMILES.pkl"
    if not idx2smiles_path.exists():
        raise FileNotFoundError(f"idx2SMILES.pkl not found at {idx2smiles_path}")

    with open(idx2smiles_path, "rb") as f:
        idx2smiles = pickle.load(f)

    fps = np.zeros((len(atc3_list), morgan_bits), dtype=np.float32)
    missing = 0
    for i, atc3 in enumerate(atc3_list):
        smiles_list = idx2smiles.get(atc3)
        if smiles_list is None:
            missing += 1
            continue
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        # Use the first valid SMILES
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, morgan_radius, nBits=morgan_bits)
                fps[i] = np.array(fp, dtype=np.float32)
                break

    print(f"  Morgan fingerprints: {len(atc3_list) - missing}/{len(atc3_list)} with SMILES")
    return fps


# ---------------------------------------------------------------------------
# Vocabulary loading
# ---------------------------------------------------------------------------

def _load_voc(voc_file: Path) -> tuple[dict, dict, dict]:
    """Load vocabulary dict → returns (diag_voc, med_voc, pro_voc) as idx2word dicts."""
    with open(voc_file, "rb") as f:
        voc = pickle.load(f)

    def _to_idx2word(v) -> dict:
        if isinstance(v, dict):
            if "idx2word" in v:
                return v["idx2word"]
            # Possibly it's word2idx directly — invert it
            return {idx: word for word, idx in v.items()}
        # Assume Voc object with .idx2word attribute
        return getattr(v, "idx2word", {})

    diag_idx2word = _to_idx2word(voc.get("diag_voc", {}))
    med_idx2word  = _to_idx2word(voc.get("med_voc",  {}))
    pro_idx2word  = _to_idx2word(voc.get("pro_voc",  {}))

    return diag_idx2word, med_idx2word, pro_idx2word


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate code_embeddings.pt for MIRROR")
    parser.add_argument("--voc_file",     type=str, required=True,
                        help="Path to vocabulary pickle (voc_final.pkl / voc_mimic4.pkl)")
    parser.add_argument("--mimic_dir",    type=str, required=True,
                        help="Path to MIMIC-III root or MIMIC-IV root (contains hosp/ for IV)")
    parser.add_argument("--external_dir", type=str, required=True,
                        help="Path to external data directory (idx2SMILES.pkl, etc.)")
    parser.add_argument("--output_dir",   type=str, required=True,
                        help="Output directory for code_embeddings*.pt")
    parser.add_argument("--cohort_tag",   type=str, default="mimic3",
                        choices=["mimic3", "mimic4", "mimic4_full", "mimic4_sota"],
                        help="Cohort tag — determines output filename")
    parser.add_argument("--mimic_version", type=int, default=None, choices=[3, 4],
                        help="MIMIC version (auto-detected from cohort_tag if not set)")
    parser.add_argument("--device",       type=str, default="cpu",
                        help="Device for PubMedBERT encoding (cpu / cuda)")
    parser.add_argument("--batch_size",   type=int, default=64,
                        help="Encoding batch size")
    parser.add_argument("--morgan_bits",  type=int, default=256)
    parser.add_argument("--morgan_radius",type=int, default=2)
    parser.add_argument("--pubmedbert_model", type=str,
                        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    args = parser.parse_args()

    voc_file     = Path(args.voc_file)
    mimic_dir    = Path(args.mimic_dir)
    external_dir = Path(args.external_dir)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect MIMIC version
    mimic_version = args.mimic_version
    if mimic_version is None:
        mimic_version = 4 if args.cohort_tag.startswith("mimic4") else 3
    print(f"\n=== Generate Code Embeddings ===")
    print(f"  Cohort tag:    {args.cohort_tag}")
    print(f"  MIMIC version: {mimic_version}")
    print(f"  Voc file:      {voc_file.name}")
    print(f"  Device:        {args.device}")

    # Step 1: Load vocabularies
    print("\n--- Step 1: Load vocabularies ---")
    diag_idx2word, med_idx2word, pro_idx2word = _load_voc(voc_file)
    print(f"  Diag vocab: {len(diag_idx2word):,} codes")
    print(f"  Proc vocab: {len(pro_idx2word):,} codes")
    print(f"  Drug vocab: {len(med_idx2word):,} drugs")

    # Show sample codes to confirm prefix handling
    sample_diag = [diag_idx2word[i] for i in sorted(diag_idx2word)[:3]]
    print(f"  Sample diag codes (raw): {sample_diag}")
    sample_stripped = [_strip_version_prefix(c)[0] for c in sample_diag]
    print(f"  Sample diag codes (stripped): {sample_stripped}")

    # Step 2: Load ICD descriptions
    print("\n--- Step 2: Load ICD descriptions ---")
    icd_desc = _load_icd_descriptions(mimic_dir, mimic_version)

    # Step 3: Build text lists for diagnosis codes
    print("\n--- Step 3: Build diagnosis text list ---")
    diag_codes_raw = [diag_idx2word[i] for i in range(len(diag_idx2word))]
    diag_texts = []
    diag_miss = 0
    for code in diag_codes_raw:
        plain, version = _strip_version_prefix(code)
        desc = _get_description(plain, icd_desc)
        if desc == f"ICD code {plain}":
            diag_miss += 1
        diag_texts.append(desc)
    print(f"  Diagnosis texts: {len(diag_texts):,} total, {diag_miss:,} fallback (raw code string)")
    if diag_miss > 0:
        hit_rate = (len(diag_texts) - diag_miss) / len(diag_texts) * 100
        print(f"  Description hit rate: {hit_rate:.1f}%")

    # Step 4: Build text lists for procedure codes
    print("\n--- Step 4: Build procedure text list ---")
    proc_codes_raw = [pro_idx2word[i] for i in range(len(pro_idx2word))]
    proc_texts = []
    proc_miss = 0
    for code in proc_codes_raw:
        plain, _ = _strip_version_prefix(code)
        desc = _get_description(plain, icd_desc)
        if desc == f"ICD code {plain}":
            proc_miss += 1
        proc_texts.append(desc)
    print(f"  Procedure texts: {len(proc_texts):,} total, {proc_miss:,} fallback")

    # Step 5: Build drug text list (ATC-3 level descriptions)
    print("\n--- Step 5: Build drug text list ---")
    drug_codes = [med_idx2word[i] for i in range(len(med_idx2word))]
    atc_desc = _load_atc3_descriptions(external_dir)
    drug_texts = []
    for atc3 in drug_codes:
        if atc3 in atc_desc:
            drug_texts.append(atc_desc[atc3])
        else:
            drug_texts.append(f"ATC drug class {atc3}")

    # Step 6: PubMedBERT encoding
    print("\n--- Step 6: Encode with PubMedBERT ---")
    all_texts = diag_texts + proc_texts + drug_texts
    all_embeds = _embed_texts(all_texts, args.pubmedbert_model, args.device, args.batch_size)

    n_diag = len(diag_texts)
    n_proc = len(proc_texts)
    n_drug = len(drug_texts)

    diag_embeds = torch.from_numpy(all_embeds[:n_diag])
    proc_embeds = torch.from_numpy(all_embeds[n_diag : n_diag + n_proc])
    drug_embeds_bert = torch.from_numpy(all_embeds[n_diag + n_proc :])

    print(f"  diag_embeddings: {diag_embeds.shape}")
    print(f"  proc_embeddings: {proc_embeds.shape}")
    print(f"  drug_embeddings: {drug_embeds_bert.shape}")

    # Step 7: Morgan fingerprints
    print("\n--- Step 7: Morgan fingerprints ---")
    morgan_fps = torch.from_numpy(
        _compute_morgan_fingerprints(drug_codes, external_dir,
                                     args.morgan_bits, args.morgan_radius)
    )
    print(f"  morgan_fingerprints: {morgan_fps.shape}")

    # Step 8: Save
    print("\n--- Step 8: Save ---")
    suffix = "" if args.cohort_tag == "mimic3" else f"_{args.cohort_tag}"
    out_path = output_dir / f"code_embeddings{suffix}.pt"

    embed_data = {
        "diag_embeddings":          diag_embeds,
        "proc_embeddings":          proc_embeds,
        "drug_embeddings":          drug_embeds_bert,
        "morgan_fingerprints":      morgan_fps,
        # Aliases for --ablation official
        "diag_embeddings_official": diag_embeds,
        "proc_embeddings_official": proc_embeds,
        "drug_embeddings_official": drug_embeds_bert,
        "embed_dim":                768,
        "morgan_bits":              args.morgan_bits,
        "pubmedbert_model":         args.pubmedbert_model,
        "cohort_tag":               args.cohort_tag,
        "mimic_version":            mimic_version,
        "diag_miss_count":          diag_miss,
        "proc_miss_count":          proc_miss,
    }
    torch.save(embed_data, out_path)
    print(f"  Saved to {out_path}")
    print(f"\n  diag: {n_diag} codes ({diag_miss} fallback, "
          f"{100*(n_diag-diag_miss)/n_diag:.1f}% with real descriptions)")
    print(f"  proc: {n_proc} codes ({proc_miss} fallback, "
          f"{100*(n_proc-proc_miss)/n_proc:.1f}% with real descriptions)")
    print(f"  drug: {n_drug} drugs")
    print(f"\n  Upload this file to Kaggle alongside your processed data.")


if __name__ == "__main__":
    main()
