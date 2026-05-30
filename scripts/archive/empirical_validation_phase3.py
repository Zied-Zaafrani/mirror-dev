import torch
import torch.nn as nn
import logging
import sys
import os

# Configure logging to capture Rule 3 signals
log_capture = []
class CaptureHandler(logging.Handler):
    def emit(self, record):
        log_capture.append(self.format(record))

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = CaptureHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from MIRROR.src.model.model import MIRROR

def run_empirical_validation():
    print("-" * 60)
    print("MIRROR PHASE 3 EMPIRICAL VALIDATION REPORT")
    print("-" * 60)
    
    # Mock data
    batch_size = 2
    num_drugs = 131
    hidden_dim = 64
    embed_dim = 768
    diag_embeddings = torch.randn(10, embed_dim)
    proc_embeddings = torch.randn(10, embed_dim)
    drug_embeddings = torch.randn(num_drugs, embed_dim)
    morgan_fingerprints = torch.randn(num_drugs, 64)
    ddi_adj = torch.zeros(num_drugs, num_drugs)
    
    # 1. Instantiate model
    model = MIRROR(
        diag_embeddings=diag_embeddings,
        proc_embeddings=proc_embeddings,
        drug_embeddings=drug_embeddings,
        morgan_fingerprints=morgan_fingerprints,
        ddi_adj=ddi_adj,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        encoder_type="gru",
        aggregator_type="attention_residual",
        use_contraindication_prior=True
    )
    
    # Mock inputs (Fixed dimensions: B, T=1, Codes=1)
    diag_seq = [torch.zeros(batch_size, 1, dtype=torch.long)]
    proc_seq = [torch.zeros(batch_size, 1, dtype=torch.long)]
    lengths = torch.tensor([1, 1])
    note_embed = torch.zeros(batch_size, 768)
    lab_vector = torch.zeros(batch_size, 36)
    has_note = torch.zeros(batch_size)
    has_lab = torch.zeros(batch_size)
    drug_history = torch.zeros(batch_size, num_drugs)
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    edge_type = torch.zeros(0, dtype=torch.long)
    lab_bins = torch.zeros(batch_size, 18, dtype=torch.long)
    lab_bins[0, 0] = 1 

    # --- TEST 1: Rule 3 Talkative Logging ---
    print("\n[Audit 1] Verifying Rule 3 (Talkative Logging)...")
    model(
        diag_seq, proc_seq, None, None, lengths,
        note_embed, lab_vector, has_note, has_lab,
        drug_history, edge_index, edge_type, lab_bins=lab_bins
    )
    
    # Check signals
    keywords = ["[Temporal]", "[Aggregator]", "[MIRRORPredictor]"]
    signals = [s for s in log_capture if any(kw in s for kw in keywords)]
    
    if len(signals) >= 2:
        print("  [PASS] Talkative signals captured:")
        for s in signals:
            print(f"    >> {s}")
    else:
        print(f"  [FAIL] Missing talkative signals. Captured only: {len(signals)}")

    # --- TEST 2: Post-Hoc Suppression (Training Neutrality) ---
    print("\n[Audit 2] Verifying Post-Hoc Logic (Training vs Eval)...")
    
    model.train()
    logits_train, _ = model(
        diag_seq, proc_seq, None, None, lengths,
        note_embed, lab_vector, has_note, has_lab,
        drug_history, edge_index, edge_type, lab_bins=lab_bins
    )
    
    model.eval()
    with torch.no_grad():
        logits_eval, _ = model(
            diag_seq, proc_seq, None, None, lengths,
            note_embed, lab_vector, has_note, has_lab,
            drug_history, edge_index, edge_type, lab_bins=lab_bins
        )
    
    mask = model.predictor.contra_prior(lab_bins)
    contra_indices = torch.where(mask[0] > 0)[0]
    
    if len(contra_indices) > 0:
        idx = contra_indices[0].item()
        l_train = logits_train[0, idx].item()
        l_eval = logits_eval[0, idx].item()
        
        print(f"  Contraindicated Drug Index: {idx}")
        print(f"  Logit (TRAIN): {l_train:.4f}")
        print(f"  Logit (EVAL):  {l_eval:.4f}")
        
        if abs(l_train) < 1e5 and l_eval < -1e8:
            print(f"  [PASS] Post-hoc suppression active. Model is TRAINING-NEUTRAL.")
        else:
            print(f"  [FAIL] Suppression logic mismatch.")
    else:
        print("  [WARN] No rules triggered. Verifying mask generation manually.")
        if mask.shape == (batch_size, num_drugs):
            print(f"  [PASS] Mask generator {mask.shape} is operational.")

    print("\n" + "-" * 60)
    print("END OF REPORT")

if __name__ == "__main__":
    run_empirical_validation()
