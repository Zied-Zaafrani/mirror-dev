"""
Precompute ClinicalBERT embeddings for the 18 lab test clinical descriptions.

Each lab test gets a pharmacologically-focused description that explains:
  - What the test measures
  - What abnormal values indicate
  - Which drugs are affected (contraindicated, dose-adjusted, or indicated)

These embeddings initialize the lab node identity vectors in the LabDrugEncoder,
bridging the semantic gap between continuous z-scored lab values and drug representations
in PubMedBERT / pharmacological text space.

Output:
  data/processed/lab_description_embeddings.npy  — shape (18, 768)
  The 18 rows correspond exactly to LAB_ITEMIDS order in lab_ranges.py.

Usage:
  cd src/preprocess
  python encode_lab_descriptions.py --output_dir ../../data/processed
  # Or from project root:
  python src/preprocess/encode_lab_descriptions.py --output_dir data/processed
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# ── Add src/preprocess to path for lab_ranges import ─────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from lab_ranges import LAB_ITEMIDS, ITEMID_TO_NAME

# ── Clinical descriptions for the 18 lab tests ───────────────────────────────
# Written to maximize pharmacological content: each description focuses on
# what the test means for DRUG selection, not just what it measures.
# This ensures the ClinicalBERT embedding lands near relevant drug descriptions.

LAB_DESCRIPTIONS = {
    "Creatinine": (
        "Serum creatinine measures kidney filtration capacity and glomerular filtration rate. "
        "Elevated creatinine indicates acute kidney injury or chronic kidney disease requiring "
        "dose reduction or avoidance of renally-cleared medications including aminoglycosides, "
        "metformin, digoxin, and NSAIDs. Nephrotoxic drugs such as vancomycin are "
        "contraindicated in severe renal impairment."
    ),
    "BUN": (
        "Blood urea nitrogen reflects protein catabolism and renal nitrogen clearance. "
        "Elevated BUN with elevated creatinine confirms azotemia requiring avoidance of "
        "nephrotoxic agents and dose reduction of renally-excreted drugs. "
        "Disproportionately elevated BUN may indicate gastrointestinal bleeding or dehydration."
    ),
    "ALT": (
        "Alanine aminotransferase is a liver enzyme elevated in hepatocellular injury and "
        "impaired hepatic drug metabolism. High ALT requires dose reduction or avoidance of "
        "hepatotoxic drugs including acetaminophen, statins, isoniazid, valproate, and "
        "antifungals. Drugs with extensive first-pass hepatic metabolism require dose adjustment."
    ),
    "AST": (
        "Aspartate aminotransferase indicates hepatocellular or muscle damage. Combined "
        "elevation of AST and ALT indicates hepatitis requiring avoidance of statins, "
        "methotrexate, and other hepatotoxic medications. AST guides CYP450-metabolized "
        "drug dosing in liver impairment."
    ),
    "Bilirubin": (
        "Total bilirubin measures liver excretory function and bile flow. Elevated bilirubin "
        "indicates cholestasis or hepatocellular dysfunction requiring avoidance of drugs that "
        "compete for albumin binding including sulfonamides, and caution with protein-bound "
        "medications. Jaundice indicates severe hepatic impairment requiring significant drug "
        "dose adjustments."
    ),
    "Alk Phos": (
        "Alkaline phosphatase is elevated in cholestatic liver disease, biliary obstruction, "
        "and bone disorders. Elevated alkaline phosphatase with other liver tests indicates "
        "biliary disease requiring caution with cholestatic drugs including estrogens, "
        "erythromycin estolate, and chlorpromazine."
    ),
    "INR": (
        "International normalized ratio measures extrinsic coagulation pathway and warfarin "
        "anticoagulation. Elevated INR above therapeutic range indicates excessive "
        "anticoagulation or liver synthetic dysfunction requiring warfarin dose reduction, "
        "avoidance of additional anticoagulants, and caution with drugs that inhibit "
        "CYP2C9 warfarin metabolism."
    ),
    "PT": (
        "Prothrombin time measures coagulation factor synthesis reflecting liver function and "
        "vitamin K antagonist therapy. Prolonged PT indicates coagulation deficiency or warfarin "
        "effect requiring avoidance of additional anticoagulants, NSAIDs, and aspirin without "
        "coagulation correction."
    ),
    "PTT": (
        "Partial thromboplastin time measures intrinsic coagulation pathway and monitors "
        "heparin anticoagulation therapy. Prolonged PTT indicates therapeutic heparin effect or "
        "factor deficiency requiring heparin dose reduction, avoidance of thrombolytics, and "
        "caution with antiplatelet agents."
    ),
    "Sodium": (
        "Serum sodium reflects fluid balance and hypothalamic-renal osmoregulation. "
        "Hyponatremia requires avoidance of free water and caution with thiazide diuretics, "
        "SSRIs, and desmopressin which worsen hyponatremia. Hypernatremia requires hypotonic "
        "fluid replacement and avoidance of sodium-containing medications."
    ),
    "Potassium": (
        "Serum potassium is critical for cardiac membrane potential and rhythm. Hyperkalemia "
        "requires immediate avoidance of ACE inhibitors, angiotensin receptor blockers, "
        "potassium-sparing diuretics, and NSAIDs. Hypokalemia increases digoxin toxicity risk "
        "and requires potassium supplementation before initiating cardiac glycosides."
    ),
    "Magnesium": (
        "Serum magnesium regulates neuromuscular function and cardiac conduction. "
        "Hypomagnesemia increases digoxin toxicity risk and promotes arrhythmias requiring "
        "magnesium supplementation before cardiac drug therapy. Hypermagnesemia requires "
        "avoidance of magnesium-containing antacids, laxatives, and caution with "
        "neuromuscular blocking agents."
    ),
    "Calcium": (
        "Serum calcium regulates neuromuscular excitability and cardiac rhythm. Hypercalcemia "
        "increases digoxin sensitivity requiring digoxin dose reduction and avoidance of "
        "thiazide diuretics and calcium supplements. Hypocalcemia is caused by loop diuretics "
        "requiring calcium and vitamin D supplementation."
    ),
    "Glucose": (
        "Serum glucose monitors glycemic control and guides antidiabetic therapy selection. "
        "Hyperglycemia indicates diabetes or corticosteroid effect requiring insulin or oral "
        "antidiabetics including metformin, GLP-1 agonists, and SGLT2 inhibitors. "
        "Hypoglycemia requires reduction of insulin, sulfonylureas, and other hypoglycemic "
        "agents. Stress hyperglycemia guides insulin infusion protocols."
    ),
    "Albumin": (
        "Serum albumin reflects hepatic synthetic function and nutritional status. "
        "Hypoalbuminemia reduces protein binding of highly protein-bound drugs increasing "
        "free drug concentration including phenytoin, warfarin, valproate, and furosemide. "
        "Low albumin requires empiric dose reduction for drugs with narrow therapeutic windows "
        "and high protein binding."
    ),
    "Lactate": (
        "Serum lactate indicates tissue oxygen delivery and aerobic metabolism. Elevated "
        "lactate indicates lactic acidosis, sepsis, or cardiogenic shock. High lactate is an "
        "absolute contraindication to metformin use due to biguanide-associated lactic "
        "acidosis. Elevated lactate guides vasopressor selection, IV fluid resuscitation, and "
        "avoidance of drugs that impair mitochondrial function."
    ),
    "WBC": (
        "White blood cell count reflects immune status, infection, and bone marrow function. "
        "Leukocytosis indicates infection requiring targeted antibiotic selection. Leukopenia "
        "or neutropenia indicates bone marrow suppression requiring immediate avoidance of "
        "myelosuppressive drugs including cytotoxic chemotherapy, clozapine, carbamazepine, "
        "methimazole, and some antibiotics. Neutropenic fever mandates empiric "
        "broad-spectrum antibiotics."
    ),
    "Hemoglobin": (
        "Hemoglobin measures blood oxygen-carrying capacity and erythropoiesis. Anemia "
        "requires avoidance of further bone marrow suppression by myelotoxic drugs and drugs "
        "causing hemolysis in G6PD-deficient patients including dapsone, primaquine, and "
        "nitrofurantoin. Severe anemia guides erythropoietin administration, intravenous iron "
        "therapy, and blood transfusion decisions."
    ),
}


def mean_pool_tokens(last_hidden_state: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool over non-padding token positions (same as extract_notes.py)."""
    mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_hidden = (last_hidden_state * mask_expanded).sum(dim=1)
    count = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_hidden / count


def encode_descriptions(
    descriptions: list[str],
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    device: str = "cpu",
    batch_size: int = 8,
) -> np.ndarray:
    """Encode text descriptions with ClinicalBERT, mean-pool over tokens.

    Returns:
        embeddings: (N, 768) float32 array
    """
    print(f"Loading {model_name} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []
    for i in range(0, len(descriptions), batch_size):
        batch_texts = descriptions[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = mean_pool_tokens(
            outputs.last_hidden_state, inputs["attention_mask"]
        )
        all_embeddings.append(embeddings.cpu().float().numpy())
        print(f"  Encoded {min(i + batch_size, len(descriptions))}/{len(descriptions)}")

    return np.concatenate(all_embeddings, axis=0)


def cosine_sim_stats(embeddings: np.ndarray) -> tuple[float, float]:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-8)
    cos_sim = normed @ normed.T
    upper = cos_sim[np.triu_indices(len(embeddings), k=1)]
    return float(upper.mean()), float(upper.max())


def main():
    parser = argparse.ArgumentParser(
        description="Precompute ClinicalBERT embeddings for 18 lab test descriptions"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/processed",
        help="Output directory (default: data/processed)"
    )
    parser.add_argument(
        "--model_name", type=str,
        default="emilyalsentzer/Bio_ClinicalBERT",
        help="HuggingFace model name (default: Bio_ClinicalBERT)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device: 'cpu', 'cuda', or 'cuda:0' etc."
    )
    parser.add_argument(
        "--note_mean_path", type=str, default=None,
        help="Path to note_global_mean.npy. If provided, centers embeddings by "
             "the same mean used for note embeddings (recommended for consistency)."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect note_global_mean.npy if not specified
    note_mean_path = args.note_mean_path
    if note_mean_path is None:
        candidate = output_dir / "note_global_mean.npy"
        if candidate.exists():
            note_mean_path = str(candidate)
            print(f"Auto-detected note_global_mean.npy at {note_mean_path}")

    # Build ordered list of descriptions matching LAB_ITEMIDS order
    lab_names_ordered = [ITEMID_TO_NAME[iid] for iid in LAB_ITEMIDS]
    descriptions_ordered = []
    for name in lab_names_ordered:
        if name not in LAB_DESCRIPTIONS:
            raise KeyError(f"No description for lab '{name}'. Add it to LAB_DESCRIPTIONS.")
        descriptions_ordered.append(LAB_DESCRIPTIONS[name])

    print("=== Lab Description Encoding ===")
    print(f"Labs in order: {lab_names_ordered}")
    print()

    # Encode
    embeddings = encode_descriptions(
        descriptions_ordered,
        model_name=args.model_name,
        device=args.device,
    )

    print(f"\nEmbedding shape: {embeddings.shape}")  # (18, 768)
    print(f"Mean norm (raw): {np.linalg.norm(embeddings, axis=1).mean():.4f}")

    mean_sim, max_sim = cosine_sim_stats(embeddings)
    print(f"Pairwise cosine sim (raw) — mean: {mean_sim:.4f}, max: {max_sim:.4f}")
    print("(ClinicalBERT anisotropy: ~0.94-0.96 expected before centering)")

    # ── Center by own mean (same principle as drug embedding centering) ──────────
    # Drug embeddings are centered by mean(all 130 drug embeds) at model init.
    # Lab descriptions are centered by mean(these 18 vectors) — removes the shared
    # "biomedical text" component, leaving only the discriminative per-lab directions.
    #
    # Note: note_global_mean centers notes well (14,699 training samples → good estimate
    # of ClinicalBERT's discharge summary centroid). Lab descriptions are specialized
    # pharmacological text in a different region of ClinicalBERT space — self-centering
    # gives a better estimate of their shared component.
    lab_mean = embeddings.mean(axis=0, keepdims=True)
    embeddings_self_centered = embeddings - lab_mean
    mean_sim_self, max_sim_self = cosine_sim_stats(embeddings_self_centered)
    print(f"\nSelf-centering (subtract mean of 18 lab vectors):")
    print(f"  Pairwise cosine sim — mean: {mean_sim_self:.4f}, max: {max_sim_self:.4f}")
    print(f"  (Should be near 0.0 — same principle as drug embedding centering)")

    # Also report note_global_mean centering for comparison / ablation reference
    if note_mean_path is not None:
        note_mean = np.load(note_mean_path).astype(np.float32)
        embeddings_note_centered = embeddings - note_mean[np.newaxis, :]
        mean_sim_n, max_sim_n = cosine_sim_stats(embeddings_note_centered)
        print(f"\nNote-mean centering (note_global_mean from {note_mean_path}):")
        print(f"  Pairwise cosine sim — mean: {mean_sim_n:.4f}, max: {max_sim_n:.4f}")

    # Use self-centering as the saved output
    embeddings_to_save = embeddings_self_centered
    # Also save the lab_description_mean so the model can reconstruct raw embeddings
    # if needed, and for documentation
    lab_mean_path = output_dir / "lab_description_mean.npy"
    np.save(lab_mean_path, lab_mean.squeeze().astype(np.float32))
    print(f"\nSaved lab description mean to {lab_mean_path} (for reference)")
    centered = True

    # Save
    out_path = output_dir / "lab_description_embeddings.npy"
    np.save(out_path, embeddings_to_save.astype(np.float32))
    print(f"\nSaved {'centered' if centered else 'RAW'} embeddings to {out_path}")

    # Human-readable summary
    print(f"\n=== Per-Lab Embedding Summary ({'centered' if centered else 'raw'}) ===")
    for i, name in enumerate(lab_names_ordered):
        norm = np.linalg.norm(embeddings_to_save[i])
        print(f"  [{i:2d}] {name:12s}: norm={norm:.3f}")


if __name__ == "__main__":
    main()
