import torch
import torch.nn as nn
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stdout
)

from MIRROR.src.model.model import MIRROR
from MIRROR.src.model.registry import TEMPORAL_ENCODERS, AGGREGATORS

def verify_logging():
    print("\n--- Phase 3.2: Rule 3 Logging Verification ---")
    
    # Mock data
    batch_size = 2
    num_drugs = 131
    hidden_dim = 64
    diag_embeddings = torch.randn(500, 768)
    proc_embeddings = torch.randn(500, 768)
    drug_embeddings = torch.randn(num_drugs, 768)
    morgan_fingerprints = torch.randn(num_drugs, 64)
    ddi_adj = torch.zeros(num_drugs, num_drugs)
    
    # 1. Test GRU + Attention
    print("\nTesting: GRU + attention_residual")
    model = MIRROR(
        diag_embeddings=diag_embeddings,
        proc_embeddings=proc_embeddings,
        drug_embeddings=drug_embeddings,
        morgan_fingerprints=morgan_fingerprints,
        ddi_adj=ddi_adj,
        hidden_dim=hidden_dim,
        encoder_type="gru",
        aggregator_type="attention_residual",
        use_contraindication_prior=True
    )
    
    # Mock forward inputs
    diag_seq = [torch.zeros(batch_size, dtype=torch.long)]
    proc_seq = [torch.zeros(batch_size, dtype=torch.long)]
    lengths = torch.tensor([1, 1])
    note_embed = torch.zeros(batch_size, 768)
    lab_vector = torch.zeros(batch_size, 36)
    has_note = torch.zeros(batch_size)
    has_lab = torch.zeros(batch_size)
    drug_history = torch.zeros(batch_size, num_drugs)
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    edge_type = torch.zeros(0, dtype=torch.long)
    lab_bins = torch.zeros(batch_size, 18, dtype=torch.long)
    
    # Run forward
    model(
        diag_seq, proc_seq, None, None, lengths,
        note_embed, lab_vector, has_note, has_lab,
        drug_history, edge_index, edge_type, lab_bins=lab_bins
    )
    
    # 2. Test Transformer + Last
    print("\nTesting: Transformer + last")
    model2 = MIRROR(
        diag_embeddings=diag_embeddings,
        proc_embeddings=proc_embeddings,
        drug_embeddings=drug_embeddings,
        morgan_fingerprints=morgan_fingerprints,
        ddi_adj=ddi_adj,
        hidden_dim=hidden_dim,
        encoder_type="transformer",
        aggregator_type="last",
        use_contraindication_prior=True
    )
    model2(
        diag_seq, proc_seq, None, None, lengths,
        note_embed, lab_vector, has_note, has_lab,
        drug_history, edge_index, edge_type, lab_bins=lab_bins
    )

    print("\n--- Phase 3.3: Contraindication Prior Suppression Check ---")
    model2.eval()
    with torch.no_grad():
        logits, _ = model2(
            diag_seq, proc_seq, None, None, lengths,
            note_embed, lab_vector, has_note, has_lab,
            drug_history, edge_index, edge_type, lab_bins=lab_bins
        )
    
    print(f"Eval logits (first drug): {logits[0, 0].item():.4f}")
    
    # Force a violation check
    if model2.predictor.contra_prior:
        mask = model2.predictor.contra_prior(lab_bins)
        print(f"Contra mask sum: {mask.sum().item()}")
    
    print("\nVerification Complete.")

if __name__ == "__main__":
    verify_logging()
