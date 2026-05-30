"""
Per-Lab Trajectory LSTM Encoder.

Extracts temporal patterns for each of the 18 labs independently across patient history
using a shared LSTM, processing `(B*18, T, 2)` features where 2 = `[zscore, is_present]`.
"""

import torch
import torch.nn as nn

from ..registry import LAB_ENCODERS
from .common import _split_lab_vec

import logging
logger = logging.getLogger(__name__)

@LAB_ENCODERS.register("traj_lstm")
class PerLabTrajectoryLSTM(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        num_labs: int = 200,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_labs = num_labs
        
        # Shared LSTM for all labs. Processes [zscore, is_present] -> hidden_dim
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        self.lab_token_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        # Property for MedGCN ablation
        self._lab_h = None
        self.lab_h_dim = hidden_dim

    def forward(
        self,
        lab_vector: torch.Tensor,    # (batch, 36)
        drug_reprs: torch.Tensor,    # (num_drugs, hidden_dim)
        has_lab: torch.Tensor,       # (batch,)
        temperature: "torch.Tensor | float" = 1.0,
        lab_trajectory: torch.Tensor | None = None,     # (batch, T_max, 36)
        lab_trajectory_len: torch.Tensor | None = None, # (batch,)
        **kwargs,
    ) -> torch.Tensor:
        # Talkative Logging
        if not hasattr(self, "_logged_flow"):
            traj_shape = lab_trajectory.shape if lab_trajectory is not None else "None"
            logger.info(f"  [PerLabTrajectoryLSTM] Active Flow:")
            logger.info(f"    - Lab Vector:  {lab_vector.shape}")
            logger.info(f"    - Trajectory:  {traj_shape}")
            logger.info(f"    - Num Labs:    {self.num_labs}")
            self._logged_flow = True
        B = lab_vector.size(0)
        
        if lab_trajectory is None or lab_trajectory_len is None:
            # Fallback if no trajectory provided: treat current lab as T=1
            lab_trajectory = lab_vector.unsqueeze(1)
            lab_trajectory_len = torch.ones(B, dtype=torch.long, device=lab_vector.device)
            
        T_max = lab_trajectory.size(1)
        
        # Split trajectory: (B, T, 36) -> (B, T, 18), (B, T, 18)
        # Note: _split_lab_vec expects 2D, so we reshape
        flat_traj = lab_trajectory.view(B * T_max, -1)
        flat_vals, flat_pres = _split_lab_vec(flat_traj, num_labs=self.num_labs)
        
        traj_vals = flat_vals.view(B, T_max, self.num_labs)
        traj_pres = flat_pres.view(B, T_max, self.num_labs)
        
        # Phase 7 Audit: Sheetrit 2023 Linear Interpolation / LOCF
        # Fill missing lab values with the last observation carried forward (LOCF)
        # to ensure the LSTM sees a continuous signal rather than abrupt drops to zero.
        with torch.no_grad():
            filled_vals = traj_vals.clone()
            for t in range(1, T_max):
                # If current is missing (pres=0), carry forward previous
                missing_mask = (traj_pres[:, t, :] < 0.5)
                filled_vals[:, t, :][missing_mask] = filled_vals[:, t-1, :][missing_mask]
        traj_vals = filled_vals

        # Stack features: (B, T, 18, 2)
        features = torch.stack([traj_vals, traj_pres], dim=-1)
        
        # Reshape for shared LSTM: (B*18, T, 2)
        features = features.transpose(1, 2).contiguous().view(B * self.num_labs, T_max, 2)
        
        # We need to gather the last valid timestep for each patient.
        # lab_trajectory_len is (B,)
        lens = lab_trajectory_len.repeat_interleave(self.num_labs) # (B*18,)
        # Handle edge case where lens=0 (no history at all)
        lens = torch.clamp(lens, min=1)
        
        # Pack sequence (optional but cleaner)
        # Here we just use the output and gather at length-1
        out, _ = self.lstm(features) # out is (B*18, T_max, H)
        
        # Gather last output
        idx = (lens - 1).view(-1, 1, 1).expand(-1, 1, self.hidden_dim)
        last_out = out.gather(1, idx).squeeze(1) # (B*18, H)
        
        # Reshape back to (B, 18, H)
        lab_tokens = last_out.view(B, self.num_labs, self.hidden_dim)
        
        # Project
        lab_tokens = self.lab_token_proj(lab_tokens)
        
        # Current visit presence (mask out labs entirely missing in current visit)
        _, curr_pres = _split_lab_vec(lab_vector, num_labs=self.num_labs)
        lab_tokens = lab_tokens * curr_pres.unsqueeze(-1)
        
        # Save _lab_h
        present_mask = curr_pres.unsqueeze(-1)
        self._lab_h = (lab_tokens * present_mask).sum(dim=1) / present_mask.sum(dim=1).clamp(min=1)

        if isinstance(temperature, torch.Tensor):
            temp = temperature.clamp(min=0.1)
        else:
            temp = max(temperature, 0.1)

        lab_drug_scores = (lab_tokens @ drug_reprs.T) / temp  # (B, 18, D)
        scores = lab_drug_scores.sum(dim=1)                   # (B, D)
        return scores * has_lab.unsqueeze(1)
