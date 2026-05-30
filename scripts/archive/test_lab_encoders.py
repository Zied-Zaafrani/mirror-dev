import sys
sys.path.insert(0, 'src')
import torch
import model.lab_encoders
from model.registry import LAB_ENCODERS
from model.predictor import MultiHeadCopyPredictor

B, H, D = 4, 64, 131
print('Testing predictor dispatch for all encoders (use_copy=False)...')

for name in LAB_ENCODERS._registry:
    try:
        enc = LAB_ENCODERS.build(name, hidden_dim=H)
        pred = MultiHeadCopyPredictor(
            hidden_dim=H, num_drugs=D, note_input_dim=768,
            lab_input_dim=36, dropout=0.1, lab_encoder=enc,
            use_copy=False,
            per_visit_copy=False,
        )
        fused = torch.randn(B, H)
        drug_reprs = torch.randn(D, H)
        drug_history = torch.zeros(B, D)
        lab_vector = torch.randn(B, 36)
        lab_vector[..., 18:] = (torch.rand(B, 18) > 0.5).float()
        has_lab = torch.ones(B)
        lab_bins = torch.randint(0, 4, (B, 18))
        lab_delta = torch.randn(B, 18)
        lab_traj = torch.randn(B, 10, 36)
        lab_traj_len = torch.randint(1, 10, (B,))

        logits, cg = pred(fused, drug_reprs, drug_history,
                         lab_vector=lab_vector, has_lab=has_lab,
                         lab_bins=lab_bins, lab_delta=lab_delta,
                         lab_trajectory=lab_traj, lab_trajectory_len=lab_traj_len)
        ok = logits.shape == (B, D)
        status = 'OK' if ok else 'FAIL'
        print(f'  [{status}] {name}: logits={tuple(logits.shape)}')
    except Exception as e:
        print(f'  [FAIL] {name}: {e}')
