import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

from model.model import MIRROR
from model.registry import GRAPH_ENCODERS, GRAPH_LAYERS, PREDICTORS

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def verify_modularity():
    print("=== Empirical Validation: MIRROR Modularity & Registry System ===\n")

    # 1. Verification of Component Registration
    print("[1] Verifying Registry Population...")
    print(f"    Graph Encoders: {list(GRAPH_ENCODERS._registry.keys())}")
    print(f"    Graph Layers:   {list(GRAPH_LAYERS._registry.keys())}")
    print(f"    Predictors:     {list(PREDICTORS._registry.keys())}")
    
    assert "none" in GRAPH_ENCODERS._registry, "NoGraphEncoder ('none') not registered!"
    assert "drug_gnn" in GRAPH_ENCODERS._registry, "DrugGNN not registered!"
    print("    [PASS] Registries are populated correctly.\n")

    # 2. Verification of "No-Graph" Ablation (The HI-DR -G baseline)
    print("[2] Verifying 'No-Graph' Ablation (none)...")
    
    # Mock data
    num_drugs = 20
    diag_embeds = torch.randn(50, 768)
    proc_embeds = torch.randn(30, 768)
    drug_embeds = torch.randn(num_drugs, 768)
    morgan_fps = torch.randn(num_drugs, 256)
    ddi_adj = torch.zeros(num_drugs, num_drugs)
    
    # Case A: Standard GNN
    print("\n--- CASE A: Standard GNN (DrugGNN + HGT) ---")
    model_gnn = MIRROR(
        diag_embeddings=diag_embeds,
        proc_embeddings=proc_embeds,
        drug_embeddings=drug_embeds,
        morgan_fingerprints=morgan_fps,
        ddi_adj=ddi_adj,
        graph_encoder_type="drug_gnn",
        graph_layer_type="hgt"
    )
    
    # Case B: No-Graph Ablation
    print("\n--- CASE B: No-Graph Ablation (none) ---")
    model_none = MIRROR(
        diag_embeddings=diag_embeds,
        proc_embeddings=proc_embeds,
        drug_embeddings=drug_embeds,
        morgan_fingerprints=morgan_fps,
        ddi_adj=ddi_adj,
        graph_encoder_type="none"
    )
    
    from model.graph_encoders.no_graph import NoGraphEncoder
    assert isinstance(model_none.drug_gnn, NoGraphEncoder), "Model failed to use NoGraphEncoder when requested!"
    print("    [PASS] Registry successfully dispatched NoGraphEncoder.\n")

    # 3. Verification of Learnable Projection Head in NoGraphEncoder
    print("[3] Verifying Learnable Projection Head in NoGraphEncoder...")
    # The NoGraphEncoder should have a learnable projection if hidden_dim != input_dim
    # In MIRROR, hidden_dim=256 (default), but NoGraphEncoder receives morgan (256) + centered (768) = 1024? 
    # Wait, let's check what features NoGraphEncoder gets.
    # It gets: torch.cat([morgan_fingerprints, drug_embeddings_centered], dim=1)
    
    print(f"    NoGraphEncoder proj: {model_none.drug_gnn.input_proj}")
    input_dim = 256 + 768 # morgan + centered
    assert model_none.drug_gnn.input_proj[0].in_features == input_dim, "Incorrect input dimension for NoGraphEncoder projection"
    print("    [PASS] Learnable projection head dimensions are correct.\n")

    # 4. Verification of Forward Pass without Graph
    print("[4] Verifying Forward Pass (No-Graph)...")
    bs = 4
    diag_seq = torch.randint(0, 50, (bs, 3, 5))
    proc_seq = torch.randint(0, 30, (bs, 3, 5))
    diag_mask = torch.ones(bs, 3, 5)
    proc_mask = torch.ones(bs, 3, 5)
    lengths = torch.tensor([3, 3, 3, 3])
    note_embed = torch.randn(bs, 768)
    lab_vector = torch.randn(bs, 400)
    has_note = torch.ones(bs)
    has_lab = torch.ones(bs)
    drug_history = torch.zeros(bs, 20)
    
    # Convert 3D tensors to lists of 2D tensors (timesteps)
    diag_seq_list = [diag_seq[:, t, :] for t in range(3)]
    proc_seq_list = [proc_seq[:, t, :] for t in range(3)]
    diag_mask_list = [diag_mask[:, t, :] for t in range(3)]
    proc_mask_list = [proc_mask[:, t, :] for t in range(3)]
    
    # We pass empty graph data for No-Graph
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_type = torch.zeros(0, dtype=torch.long)

    # Add med_per_visit mock data
    med_per_visit = torch.zeros(bs, 3, num_drugs)

    model_none.eval()
    with torch.no_grad():
        logits, _ = model_none(
            diag_seq_list, proc_seq_list, diag_mask_list, proc_mask_list, lengths,
            note_embed, lab_vector, has_note, has_lab,
            drug_history, edge_index, edge_type,
            med_per_visit=med_per_visit
        )
    
    assert logits.shape == (bs, num_drugs), f"Expected ({bs}, {num_drugs}), got {logits.shape}"
    print("    [PASS] Forward pass successful with 'none' encoder.\n")

    # 5. Verification of 200-Lab Default
    print("[5] Verifying 200-Lab Default...")
    assert model_gnn.num_labs == 200, f"Expected 200 labs, got {model_gnn.num_labs}"
    print("    [PASS] num_labs default is 200.\n")

    # 6. Verification of Trajectory Passing (traj_lstm)
    print("[6] Verifying Trajectory Passing (traj_lstm)...")
    model_traj = MIRROR(
        diag_embeddings=diag_embeds,
        proc_embeddings=proc_embeds,
        drug_embeddings=drug_embeds,
        morgan_fingerprints=morgan_fps,
        ddi_adj=ddi_adj,
        lab_encoder_type="traj_lstm",
        lab_input_dim=400,
        num_labs=200
    )
    
    lab_trajectory = torch.randn(bs, 5, 400) # (B, T, 200*2)
    lab_trajectory_len = torch.tensor([2, 3, 4, 5])
    
    model_traj.eval()
    with torch.no_grad():
        # Pass a 400-dim lab vector to match the new 200-lab standard
        logits, _ = model_traj(
            diag_seq_list, proc_seq_list, diag_mask_list, proc_mask_list, lengths,
            note_embed, torch.randn(bs, 400), has_note, has_lab,
            drug_history, edge_index, edge_type,
            lab_trajectory=lab_trajectory,
            lab_trajectory_len=lab_trajectory_len
        )
    print("    [PASS] Forward pass successful with traj_lstm and trajectory data.\n")

    print("\n=== ALL EMPIRICAL VALIDATION CHECKS PASSED ===")

if __name__ == "__main__":
    verify_modularity()
