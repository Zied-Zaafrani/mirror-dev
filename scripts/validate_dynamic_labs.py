import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataset import MIRRORDataset, compute_lab_bins
from model.model import MIRROR
from model.registry import LAB_ENCODERS

def test_config(num_labs, trends=False):
    print(f"\n{'='*60}")
    print(f" TESTING: {num_labs} labs | trends={trends}")
    print(f"{'='*60}")
    
    # Calculate dimensions
    lab_dim = num_labs * 4 if trends else num_labs * 2
    n_labs = num_labs
    
    # 1. Mock Data
    hadm_ids = [100, 101, 102]
    lab_vecs = np.random.randn(3, lab_dim).astype(np.float32)
    # Set flags to 0 (present) for first half of batch
    lab_vecs[:2, n_labs : 2*n_labs] = 0.0
    # Set flags to 1 (missing) for last row
    lab_vecs[2, n_labs : 2*n_labs] = 1.0
    
    lab_data = {
        "hadm_ids": hadm_ids,
        "lab_vectors": lab_vecs,
        "lab_names": [f"Lab_{i}" for i in range(n_labs)],
        "zscore_means": np.zeros(n_labs),
        "zscore_stds": np.ones(n_labs)
    }
    
    # 2. Dataset Simulation
    # Format: List of Patients, where each Patient is a List of Visits
    # Visit: [diag_indices, proc_indices, med_indices, hadm_id]
    records = [[
        [[1, 2], [10, 11], [5, 6], 100],
        [[3, 4], [12, 13], [7, 8], 101],
        [[5, 6], [14, 15], [9, 10], 102], # target visit
    ]]
    cohort = {"num_drugs": 130, "num_diag": 100, "num_proc": 50}
    
    ds = MIRRORDataset(
        records=records,
        cohort=cohort,
        note_data=None,
        lab_data=lab_data,
        num_drugs=130,
        lab_key="lab_vectors",
        num_labs=num_labs
    )
    
    sample = ds[0]
    print(f"Dataset sample keys: {list(sample.keys())}")
    print(f"lab_vector shape: {sample['lab_vector'].shape} (Expected: {lab_dim})")
    print(f"lab_bins shape: {sample['lab_bins'].shape} (Expected: {n_labs})")
    print(f"lab_delta shape: {sample['lab_delta'].shape} (Expected: {n_labs})")
    
    assert sample['lab_vector'].shape[0] == lab_dim
    assert sample['lab_bins'].shape[0] == n_labs
    assert sample['lab_delta'].shape[0] == n_labs
    
    # 3. Model Simulation
    # Mock embeddings
    drug_embeds = torch.randn(130, 768)
    diag_embeds = torch.randn(100, 768)
    proc_embeds = torch.randn(50, 768)
    morgan_fps = torch.randn(130, 1024)
    ddi_adj = torch.zeros(130, 130)
    
    # Test different encoders
    for enc_type in ["flat", "per_lab_attn", "clinical_bin", "lab_as_text"]:
        print(f"\n  Checking encoder: {enc_type}...")
        model = MIRROR(
            diag_embeddings=diag_embeds,
            proc_embeddings=proc_embeds,
            drug_embeddings=drug_embeds,
            morgan_fingerprints=morgan_fps,
            ddi_adj=ddi_adj,
            lab_input_dim=lab_dim,
            lab_encoder_type=enc_type,
            num_labs=num_labs,
            use_labs=True,
            use_lab_impute_loss=True
        )
        
        # Fake batch (B=2)
        diag_seq = torch.randint(0, 100, (2, 2, 5)) # B, T, N
        proc_seq = torch.randint(0, 50, (2, 2, 5))
        lab_vector = torch.cat([torch.tensor(sample['lab_vector']).unsqueeze(0)] * 2, dim=0)
        lab_bins = torch.cat([torch.tensor(sample['lab_bins']).unsqueeze(0)] * 2, dim=0)
        lab_delta = torch.cat([torch.tensor(sample['lab_delta']).unsqueeze(0)] * 2, dim=0)
        has_lab = torch.tensor([1.0, 1.0]) # B=2
        
        # Forward with mixed lengths (T=1 and T=2) to test magnitude rebalancing (Phase 8.2)
        lengths = torch.tensor([1, 2])
        logits, _ = model(
            diag_seq=diag_seq,
            proc_seq=proc_seq,
            diag_mask_seq=torch.ones_like(diag_seq),
            proc_mask_seq=torch.ones_like(proc_seq),
            lengths=lengths,
            note_embed=torch.randn(2, 768),
            has_note=torch.tensor([1.0, 1.0]),
            drug_history=torch.zeros(2, 130),
            med_per_visit=torch.zeros(2, 2, 130), # B, T, num_drugs
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            edge_type=torch.tensor([0], dtype=torch.long),
            lab_vector=lab_vector,
            lab_bins=lab_bins,
            lab_delta=lab_delta,
            has_lab=has_lab
        )
        
        # Check patient_repr magnitude balance
        patient_repr = model.visit_encoder(
            diag_seq=[diag_seq[:, 0, :], diag_seq[:, 1, :]], # list of T tensors
            proc_seq=[proc_seq[:, 0, :], proc_seq[:, 1, :]],
            diag_mask_seq=[torch.ones_like(diag_seq[:, 0, :]), torch.ones_like(diag_seq[:, 1, :])],
            proc_mask_seq=[torch.ones_like(proc_seq[:, 0, :]), torch.ones_like(proc_seq[:, 1, :])],
            lengths=lengths
        )
        norms = torch.norm(patient_repr, dim=-1)
        ratio = norms.max() / (norms.min() + 1e-8)
        print(f"    Logits shape: {logits.shape} (Expected: (2, 130))")
        print(f"    Patient Repr Norms: {norms.tolist()} (Ratio: {ratio.item():.4f})")
        
        assert logits.shape == (2, 130)
        # Ratio should be reasonable (not 2.3x skew)
        assert ratio < 2.5, f"Magnitude imbalance detected: T=1/T=2 ratio {ratio.item():.4f} > 2.5"
        
        # Phase 2.2: Imputation head now lives in loss_fn, not model.
        # Verify the encoder exposes _lab_h and lab_h_dim (the ID Badge).
        lab_enc = model.predictor.lab_encoder
        assert hasattr(lab_enc, "lab_h_dim"), \
            f"Encoder '{enc_type}' missing lab_h_dim ID Badge!"
        assert hasattr(lab_enc, "_lab_h"), \
            f"Encoder '{enc_type}' did not set _lab_h after forward()!"
        if lab_enc._lab_h is not None:
            print(f"    _lab_h shape: {lab_enc._lab_h.shape} (lab_h_dim={lab_enc.lab_h_dim})")
        else:
            print(f"    _lab_h=None — encoder reported no lab hidden state.")

if __name__ == "__main__":
    # Test cases
    test_config(18, trends=False)
    test_config(18, trends=True)
    test_config(50, trends=False)
    test_config(100, trends=False)
    
    print("\n" + "#"*40)
    print(" ALL TESTS PASSED EMPIRICALLY! ")
    print("#"*40)
