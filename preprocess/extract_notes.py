"""
Extract and preprocess discharge summaries from MIMIC-III/IV for the MIRROR framework.

Two-tier note encoding:
  Tier 1 (Default): Chunk + mean-pool ClinicalBERT
    - Split notes into 512-token chunks with 128-token overlap
    - Encode each chunk with Bio_ClinicalBERT → 768d
    - Mean-pool across chunks → single 768d embedding per admission

  Tier 2 (Ablation): MedGemma-4B full-context embeddings
    - See encode_notes_medgemma.py

This script handles:
  1. Loading and filtering discharge summaries
  2. Removing "Discharge Medications" section (data leakage prevention)
  3. Saving cleaned text per admission (for downstream encoding)
  4. Optionally running ClinicalBERT chunk+pool encoding

Usage:
  # Extract text only (fast, CPU):
  python extract_notes.py --mimic_dir DATASETS/mimic-iii-clinical-database-1.4 \
                          --cohort_file data/processed/cohort_mimic3.pkl \
                          --output_dir data/processed

  # Extract + encode with ClinicalBERT (requires GPU):
  python extract_notes.py --mimic_dir DATASETS/mimic-iii-clinical-database-1.4 \
                          --cohort_file data/processed/cohort_mimic3.pkl \
                          --output_dir data/processed \
                          --encode --device cuda
"""

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd


# --- Section removal patterns ---
# Match common discharge medication section headers in MIMIC notes.
# Each pattern: header followed by content until next section header or end-of-text.
# IMPORTANT: Must catch ALL variants to prevent target leakage.
_NEXT_SECTION = r"(?=\n\s*(?:[A-Z][A-Za-z][A-Za-z\-/ ]{2,}:|[A-Z][A-Z][A-Z\-/ ]{2,})|\Z)"
MEDICATION_SECTION_PATTERNS = [
    # "Discharge Medications:" / "Discharge Medication:"
    rf"(?is)discharge\s+medications?\s*:.*?{_NEXT_SECTION}",
    # "Medications on Discharge:" / "Medications at Discharge:"
    rf"(?is)medications?\s+(?:on|at)\s+discharge\s*:.*?{_NEXT_SECTION}",
    # "Discharge Meds:" (abbreviated)
    rf"(?is)discharge\s+meds?\s*:.*?{_NEXT_SECTION}",
    # "Patient's Medications at Discharge:" (possessive form)
    rf"(?is)patient'?s?\s+medications?\s+(?:on|at)\s+discharge\s*:.*?{_NEXT_SECTION}",
    # --- Admission/home medications (predictive confound, not discharge target but highly correlated) ---
    # Many patients continue the same drugs from admission to discharge. Leaving admission meds in the
    # note lets the model learn "admitted on X → discharged on X" without clinical reasoning.
    # These sections are removed to ensure the model uses physiological state, not prior drug lists.
    rf"(?is)medications?\s+on\s+admission\s*:.*?{_NEXT_SECTION}",
    rf"(?is)admission\s+medications?\s*:.*?{_NEXT_SECTION}",
    rf"(?is)home\s+medications?\s*:.*?{_NEXT_SECTION}",
    rf"(?is)outpatient\s+medications?\s*:.*?{_NEXT_SECTION}",
    rf"(?is)pre-?(?:admission|hospital)\s+medications?\s*:.*?{_NEXT_SECTION}",
    rf"(?is)current\s+medications?\s*:.*?{_NEXT_SECTION}",
    rf"(?is)medications?\s+prior\s+to\s+admission\s*:.*?{_NEXT_SECTION}",
]

# Post-clean leakage audit patterns (warning only).
LEAK_AUDIT_PATTERNS = [
    r"(?i)discharge\s+medications?\s*:",
    r"(?i)medications?\s+(?:on|at)\s+discharge\s*:",
    r"(?i)discharge\s+meds?\s*:",
    r"(?i)medications?\s+on\s+admission\s*:",
    r"(?i)admission\s+medications?\s*:",
    r"(?i)home\s+medications?\s*:",
]


def remove_discharge_medications(text: str) -> str:
    """Remove the Discharge Medications section from a discharge summary.

    This section lists the exact medications prescribed at discharge — including
    it would be data leakage since medications are our prediction target.
    """
    for pattern in MEDICATION_SECTION_PATTERNS:
        text = re.sub(pattern, "[MEDICATION_SECTION_REMOVED]", text, flags=re.DOTALL)
    return text


def load_discharge_notes(mimic_dir: Path, mimic_version: int = 3,
                         cohort_hadm_ids: set = None) -> pd.DataFrame:
    """Load discharge summaries from MIMIC-III or MIMIC-IV.

    cohort_hadm_ids: if provided, filter to only these admissions during chunked
    reading to avoid materialising the full file in RAM (~4-8 GB uncompressed).
    """
    if mimic_version == 3:
        path = mimic_dir / "NOTEEVENTS.csv.gz"
        print(f"Loading NOTEEVENTS from {path} (chunked to limit RAM) ...")
        # NOTEEVENTS contains all note types — load in chunks and filter inline
        # to avoid materializing 6+ GB for the full file.
        chunks = []
        for chunk in pd.read_csv(
            path,
            usecols=["SUBJECT_ID", "HADM_ID", "CATEGORY", "ISERROR", "CHARTDATE", "TEXT"],
            compression="gzip",
            chunksize=50_000,
        ):
            chunk.columns = chunk.columns.str.upper()
            chunk = chunk[chunk["CATEGORY"] == "Discharge summary"]
            chunk = chunk[chunk["ISERROR"] != 1]
            if cohort_hadm_ids is not None:
                chunk = chunk[chunk["HADM_ID"].isin(cohort_hadm_ids)]
            if not chunk.empty:
                chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
            columns=["SUBJECT_ID", "HADM_ID", "CATEGORY", "ISERROR", "CHARTDATE", "TEXT"]
        )
    else:
        # MIMIC-IV: check if mimic_dir already points to /note/ or hosp-level
        path = mimic_dir / "discharge.csv.gz"
        if not path.exists():
            path = mimic_dir / "note" / "discharge.csv.gz"
        if not path.exists():
            # MIMIC-IV notes might be in a separate directory
            alt = mimic_dir.parent / "mimic-iv-note-deidentified-free-text-clinical-notes-2.2" / "note" / "discharge.csv.gz"
            if alt.exists():
                path = alt
        print(f"Loading discharge notes from {path} (chunked to limit RAM) ...")
        # discharge.csv.gz unpacks to 4-8 GB. Track latest note per hadm_id in a
        # plain dict during chunking — never build the full DataFrame in memory.
        # Dict value: (charttime_str, text). Updated whenever a later note appears.
        latest: dict = {}  # hadm_id -> (charttime, text)
        n_rows = 0
        for chunk in pd.read_csv(
            path,
            usecols=["subject_id", "hadm_id", "charttime", "text"],
            compression="gzip",
            chunksize=10_000,
            low_memory=False,
        ):
            chunk.columns = chunk.columns.str.upper()
            chunk = chunk.dropna(subset=["HADM_ID", "TEXT"])
            chunk["HADM_ID"] = chunk["HADM_ID"].astype(int)
            if cohort_hadm_ids is not None:
                chunk = chunk[chunk["HADM_ID"].isin(cohort_hadm_ids)]
            for _, row in chunk.iterrows():
                hid = int(row["HADM_ID"])
                ct = row.get("CHARTTIME", "")
                txt = row["TEXT"]
                if hid not in latest or str(ct) > str(latest[hid][0]):
                    latest[hid] = (ct, txt)
            n_rows += len(chunk)
            if n_rows % 50_000 == 0:
                print(f"  {n_rows:,} cohort rows processed, {len(latest):,} unique admissions ...")
        # Build DataFrame from dict — one row per admission, no sort needed
        df = pd.DataFrame(
            [(hid, v[1]) for hid, v in latest.items()],
            columns=["HADM_ID", "TEXT"],
        ) if latest else pd.DataFrame(columns=["HADM_ID", "TEXT"])

    df = df.dropna(subset=["HADM_ID", "TEXT"])
    df["HADM_ID"] = df["HADM_ID"].astype(int)

    # MIMIC-IV path already deduplicates to latest note per admission in the dict.
    # MIMIC-III path may still have duplicates — deduplicate here if needed.
    if "CHARTDATE" in df.columns or "CHARTTIME" in df.columns:
        sort_col = "CHARTTIME" if "CHARTTIME" in df.columns else "CHARTDATE"
        df[sort_col] = pd.to_datetime(df[sort_col], errors="coerce")
        df = df.sort_values(sort_col).groupby("HADM_ID").last().reset_index()
    elif df["HADM_ID"].duplicated().any():
        df = df.groupby("HADM_ID").last().reset_index()

    print(f"  Loaded {len(df):,} discharge summaries")
    return df[["HADM_ID", "TEXT"]]


def clean_notes(df: pd.DataFrame) -> pd.DataFrame:
    """Clean discharge notes: remove medication section, normalize whitespace."""
    cleaned = df.copy()
    cleaned["TEXT"] = cleaned["TEXT"].apply(remove_discharge_medications)
    # Normalize excessive whitespace
    cleaned["TEXT"] = cleaned["TEXT"].str.replace(r"\n{3,}", "\n\n", regex=True)
    cleaned["TEXT"] = cleaned["TEXT"].str.strip()
    return cleaned


def audit_medication_leakage(df: pd.DataFrame) -> tuple[int, int]:
    """Return (#notes_with_residual_markers, total_notes)."""
    residual = 0
    for text in df["TEXT"].astype(str):
        if any(re.search(p, text) for p in LEAK_AUDIT_PATTERNS):
            residual += 1
    return residual, len(df)


def chunk_text(text: str, tokenizer, max_length: int = 512, overlap: int = 128) -> list[list[int]]:
    """Split text into overlapping token chunks.

    Returns list of token ID lists, each of length ≤ max_length.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    stride = max_length - overlap
    chunks = []
    for start in range(0, len(tokens), stride):
        chunk = tokens[start : start + max_length]
        if len(chunk) < 32:  # skip very short trailing chunks
            break
        chunks.append(chunk)
    if not chunks and tokens:
        chunks.append(tokens[:max_length])
    return chunks


def encode_with_clinicalbert(
    notes_df: pd.DataFrame,
    hadm_ids: np.ndarray,
    device: str = "cuda",
    batch_size: int = 16,
    max_length: int = 512,
    overlap: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode notes using chunk + mean-pool ClinicalBERT.

    Returns:
        embeddings: (N, 768) float array. All-zeros for missing notes.
        has_note:   (N,) binary array. 1 = has note, 0 = no note.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    print(f"Loading ClinicalBERT from {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    hadm_to_idx = {h: i for i, h in enumerate(hadm_ids)}
    note_lookup = dict(zip(notes_df["HADM_ID"], notes_df["TEXT"]))

    n = len(hadm_ids)
    embeddings = np.zeros((n, 768), dtype=np.float32)
    has_note = np.zeros(n, dtype=np.float32)

    processed = 0
    for hadm_id in hadm_ids:
        if hadm_id not in note_lookup:
            continue
        text = note_lookup[hadm_id]
        if not text or len(text.strip()) < 50:
            continue

        idx = hadm_to_idx[hadm_id]
        chunks = chunk_text(text, tokenizer, max_length, overlap)
        if not chunks:
            continue

        chunk_embeds = []
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            # Pad and create attention masks
            max_len = max(len(c) for c in batch_chunks)
            input_ids = []
            attention_masks = []
            for c in batch_chunks:
                padded = c + [tokenizer.pad_token_id] * (max_len - len(c))
                mask = [1] * len(c) + [0] * (max_len - len(c))
                input_ids.append(padded)
                attention_masks.append(mask)

            input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
            attention_mask_t = torch.tensor(attention_masks, dtype=torch.long, device=device)

            with torch.no_grad():
                output = model(input_ids=input_ids_t, attention_mask=attention_mask_t)
                # Mean pooling over non-pad tokens (better than CLS for patient-specific info).
                # CLS captures generic patterns; mean-pool preserves discriminative content.
                hidden = output.last_hidden_state  # (B, seq_len, 768)
                mask_expanded = attention_mask_t.unsqueeze(-1).float()  # (B, seq_len, 1)
                sum_hidden = (hidden * mask_expanded).sum(dim=1)  # (B, 768)
                count = mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
                mean_embeds = (sum_hidden / count).cpu().numpy()  # (B, 768)
                chunk_embeds.append(mean_embeds)

        # Attention-weighted chunk aggregation instead of naive mean-pool.
        # Long notes (4000+ tokens) get chunked into 10+ pieces; naive mean dilutes signal.
        # Use L2 norm of each chunk as a relevance proxy — chunks with more informative
        # content tend to have higher-norm embeddings after mean-pooling.
        all_chunks = np.concatenate(chunk_embeds, axis=0)  # (num_chunks, 768)
        if all_chunks.shape[0] == 1:
            embeddings[idx] = all_chunks[0]
        else:
            norms = np.linalg.norm(all_chunks, axis=1)  # (num_chunks,)
            weights = norms / (norms.sum() + 1e-8)  # normalized relevance weights
            embeddings[idx] = (weights[:, None] * all_chunks).sum(axis=0)
        has_note[idx] = 1.0

        processed += 1
        if processed % 1000 == 0:
            print(f"  Encoded {processed:,} notes ...")

    print(f"  Encoded {processed:,}/{n:,} admissions with notes")
    return embeddings, has_note


def main():
    parser = argparse.ArgumentParser(description="Extract discharge notes from MIMIC")
    parser.add_argument("--mimic_dir", type=str, required=True)
    parser.add_argument("--cohort_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mimic_version", type=int, default=3, choices=[3, 4])
    parser.add_argument("--suffix", type=str, default=None,
                        help="Override output suffix (default: _mimic{version})")
    parser.add_argument("--encode", action="store_true",
                        help="Run ClinicalBERT encoding (requires GPU)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    mimic_dir = Path(args.mimic_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cohort
    print(f"Loading cohort from {args.cohort_file} ...")
    with open(args.cohort_file, "rb") as f:
        cohort = pickle.load(f)
    hadm_ids = np.array(cohort["hadm_ids"])
    cohort_hadm_set = set(int(h) for h in hadm_ids)

    # Step 1: Load discharge summaries (filtered to cohort inline to save RAM)
    notes_df = load_discharge_notes(mimic_dir, args.mimic_version,
                                    cohort_hadm_ids=cohort_hadm_set)

    # Step 2: Clean (remove medication section)
    notes_df = clean_notes(notes_df)
    print(f"  After cleaning: {len(notes_df):,} notes")
    residual, total = audit_medication_leakage(notes_df)
    if total > 0:
        pct = residual / total * 100
        print(f"  Leakage audit (residual medication headers): {residual:,}/{total:,} ({pct:.2f}%)")
        if residual > 0:
            print("  WARNING: residual medication-section markers found after cleaning.")

    # Step 3: Save cleaned text
    suffix = args.suffix or f"_mimic{args.mimic_version}"
    text_path = output_dir / f"notes_text{suffix}.pkl"
    text_output = {
        "hadm_ids": hadm_ids,
        "notes": {
            int(row["HADM_ID"]): row["TEXT"]
            for _, row in notes_df.iterrows()
        },
    }
    with open(text_path, "wb") as f:
        pickle.dump(text_output, f)
    print(f"  Saved cleaned text to {text_path}")

    # Coverage stats
    cohort_hadms = set(hadm_ids)
    note_hadms = set(notes_df["HADM_ID"])
    coverage = len(cohort_hadms & note_hadms) / len(cohort_hadms) * 100
    print(f"  Note coverage: {len(cohort_hadms & note_hadms):,}/{len(cohort_hadms):,} ({coverage:.1f}%)")

    # Step 4: Optional ClinicalBERT encoding
    if args.encode:
        embeddings, has_note = encode_with_clinicalbert(
            notes_df, hadm_ids,
            device=args.device,
            batch_size=args.batch_size,
        )
        embed_path = output_dir / f"note_embeddings{suffix}.pkl"
        embed_output = {
            "embeddings": embeddings,  # (N, 768)
            "has_note": has_note,      # (N,)
            "hadm_ids": hadm_ids,
            "method": "clinicalbert_chunk_pool",
        }
        with open(embed_path, "wb") as f:
            pickle.dump(embed_output, f)
        print(f"  Saved ClinicalBERT embeddings to {embed_path}")

        # Stats
        mean_norm = np.linalg.norm(embeddings[has_note == 1], axis=1).mean()
        print(f"  Mean embedding L2 norm: {mean_norm:.2f}")


if __name__ == "__main__":
    main()
