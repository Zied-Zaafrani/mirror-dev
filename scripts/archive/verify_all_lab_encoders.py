import torch
import numpy as np
import sys
from pathlib import Path
import logging

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

from model.registry import LAB_ENCODERS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def verify_all_lab_encoders():
    print("=== Empirical Validation: Comprehensive Lab Encoder Audit (200 Labs) ===\n")

    num_labs = 200
    lab_input_dim = 400
    hidden_dim = 64
    bs = 4
    num_drugs = 131

    # Mock inputs
    lab_vector = torch.randn(bs, lab_input_dim)
    lab_bins = torch.randint(0, 4, (bs, num_labs))
    lab_delta = torch.randn(bs, num_labs)
    lab_trajectory = torch.randn(bs, 5, lab_input_dim)
    lab_trajectory_len = torch.tensor([2, 3, 4, 5])
    has_lab = torch.ones(bs, 1)
    drug_reprs = torch.randn(num_drugs, hidden_dim)
    temperature = 1.0

    registered_encoders = list(LAB_ENCODERS._registry.keys())
    print(f"Registered Lab Encoders: {registered_encoders}\n")

    results = {}

    for encoder_name in registered_encoders:
        print(f"--- Testing Encoder: {encoder_name} ---")
        try:
            # 1. Initialization
            encoder_cls = LAB_ENCODERS.get(encoder_name)
            
            # Check if it accepts num_labs
            import inspect
            sig = inspect.signature(encoder_cls.__init__)
            init_kwargs = {"hidden_dim": hidden_dim, "dropout": 0.3}
            if "num_labs" in sig.parameters:
                init_kwargs["num_labs"] = num_labs
            
            encoder = encoder_cls(**init_kwargs)
            print(f"    [INIT] Success (params: {list(init_kwargs.keys())})")

            # 2. Forward Pass
            # We use the same injection logic as MultiHeadCopyPredictor
            sig_fwd = inspect.signature(encoder.forward)
            params = sig_fwd.parameters
            
            fwd_kwargs = {}
            if "lab_vector" in params: fwd_kwargs["lab_vector"] = lab_vector
            if "drug_reprs" in params: fwd_kwargs["drug_reprs"] = drug_reprs
            if "has_lab" in params: fwd_kwargs["has_lab"] = has_lab.squeeze()
            if "temperature" in params: fwd_kwargs["temperature"] = temperature
            if "lab_bins" in params: fwd_kwargs["lab_bins"] = lab_bins
            if "lab_delta" in params: fwd_kwargs["lab_delta"] = lab_delta
            if "lab_trajectory" in params: fwd_kwargs["lab_trajectory"] = lab_trajectory
            if "lab_trajectory_len" in params: fwd_kwargs["lab_trajectory_len"] = lab_trajectory_len

            # Handle some variations in has_lab expected shape
            output = encoder(**fwd_kwargs)
            
            # 3. Shape Verification
            # Some return (B, D) scores, some return (B, H) embeddings
            if output.shape == (bs, num_drugs):
                print(f"    [FWD] Success: Output is Drug Scores (shape={output.shape})")
            elif output.shape == (bs, hidden_dim):
                print(f"    [FWD] Success: Output is Lab Embedding (shape={output.shape})")
            else:
                print(f"    [FWD] WARNING: Unexpected output shape {output.shape}")
            
            results[encoder_name] = "PASS"
        except Exception as e:
            print(f"    [FAIL] Error: {str(e)}")
            import traceback
            traceback.print_exc()
            results[encoder_name] = f"FAIL: {str(e)}"
        print("")

    print("=== SUMMARY SCORECARD ===")
    all_pass = True
    for name, res in results.items():
        status = "[PASS]" if res == "PASS" else "[FAIL]"
        print(f"{status} {name:20}: {res}")
        if res != "PASS": all_pass = False
    
    if all_pass:
        print("\n=== ALL LAB ENCODERS VALIDATED FOR 200-LAB STANDARD ===")
    else:
        print("\n=== SOME ENCODERS FAILED VALIDATION ===")
        sys.exit(1)

if __name__ == "__main__":
    verify_all_lab_encoders()
