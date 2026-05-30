"""
Phase 7 Preprocessing Script: Generate lab_text_embeddings.pt

This generates the (18, 4, 768) tensor needed by LabAsTextEncoder.
- Axis 0: 18 labs
- Axis 1: 4 bins (0=missing, 1=low, 2=normal, 3=high)
- Axis 2: 768-dim embedding from ClinicalBERT / existing lab_description_embeddings

Strategy:
  We already have `lab_description_embeddings.npy` of shape (18, 768) —
  these are embeddings of the lab name/description.

  For each bin we create a phrase like:
    bin 0 (missing): zero vector
    bin 1 (low):     embedding of "low {lab_name}"
    bin 2 (normal):  embedding of "normal {lab_name}"
    bin 3 (high):    embedding of "high {lab_name}"

  If ClinicalBERT is unavailable, we use the precomputed (18, 768) base
  embeddings and create the 4-bin tensor by:
    - bin 0: zeros
    - bin 1: base - 0.1  (shift in embedding space)
    - bin 2: base         (neutral)
    - bin 3: base + 0.1  (shift in embedding space)

  This is a deterministic, reproducible approximation.
  For production, pass --use_bert to run actual ClinicalBERT inference.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch

LAB_NAMES = [
    "Creatinine", "BUN", "ALT", "AST", "Bilirubin", "Alk Phos",
    "INR", "PT", "PTT", "Sodium", "Potassium", "Magnesium", "Calcium",
    "Glucose", "Albumin", "Lactate", "WBC", "Hemoglobin"
]

def build_from_existing(base_path: Path) -> torch.Tensor:
    """Build (18, 4, 768) from existing (18, 768) lab description embeddings."""
    base = np.load(base_path)  # (18, 768)
    assert base.shape == (18, 768), f"Expected (18,768), got {base.shape}"
    
    # L2-normalize the base embeddings first
    norms = np.linalg.norm(base, axis=1, keepdims=True).clip(min=1e-8)
    base_n = base / norms
    
    out = np.zeros((18, 4, 768), dtype=np.float32)
    # bin 0 = missing → all zeros
    out[:, 0, :] = 0.0
    # bin 1 = low → slight negative shift along a fixed direction
    out[:, 1, :] = base_n - 0.1 * np.roll(base_n, 1, axis=1)
    # bin 2 = normal → base embedding
    out[:, 2, :] = base_n
    # bin 3 = high → slight positive shift
    out[:, 3, :] = base_n + 0.1 * np.roll(base_n, 1, axis=1)
    
    print(f"  Built from existing embeddings: {base_path}")
    return torch.from_numpy(out)


def build_from_bert(device: str) -> torch.Tensor:
    """Build (18, 4, 768) using ClinicalBERT."""
    from transformers import AutoTokenizer, AutoModel
    
    MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
    print(f"  Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    
    BIN_LABELS = ["missing", "low", "normal", "high"]
    
    out = np.zeros((18, 4, 768), dtype=np.float32)
    
    with torch.no_grad():
        for i, lab in enumerate(LAB_NAMES):
            for j, bin_label in enumerate(BIN_LABELS):
                if j == 0:
                    out[i, j, :] = 0.0
                    continue
                text = f"{bin_label} {lab.lower()}"
                enc = tokenizer(text, return_tensors="pt").to(device)
                emb = model(**enc).last_hidden_state[:, 0, :]  # CLS
                out[i, j, :] = emb.cpu().float().numpy()
                print(f"    [{i:2d},{j}] '{text}' -> {emb.norm().item():.3f}")
    
    return torch.from_numpy(out)


def main():
    parser = argparse.ArgumentParser(description="Generate lab_text_embeddings.pt for Phase 7")
    parser.add_argument("--use_bert", action="store_true",
                        help="Use ClinicalBERT (requires transformers + GPU). Default: use existing npy.")
    parser.add_argument("--base_emb", default="data/processed/lab_description_embeddings.npy",
                        help="Path to existing (18, 768) lab description embeddings.")
    parser.add_argument("--out", default="data/processed/lab_text_embeddings.pt",
                        help="Output path for (18, 4, 768) tensor.")
    parser.add_argument("--device", default="cpu", help="Device for BERT (cpu or cuda).")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating lab_text_embeddings.pt ...")
    if args.use_bert:
        tensor = build_from_bert(args.device)
        method = "ClinicalBERT"
    else:
        base_path = Path(args.base_emb)
        if not base_path.exists():
            print(f"ERROR: {base_path} not found. Run with --use_bert or provide base embeddings.")
            sys.exit(1)
        tensor = build_from_existing(base_path)
        method = "existing_npy_approximation"

    torch.save(tensor, out_path)
    print(f"\nSaved: {out_path}")
    print(f"  Shape: {tuple(tensor.shape)}  Method: {method}")
    print(f"  Size: {out_path.stat().st_size / 1024:.1f} KB")
    print(f"\nBin layout: [0]=missing(zeros), [1]=low, [2]=normal, [3]=high")
    print(f"Labs: {LAB_NAMES}")


if __name__ == "__main__":
    main()
