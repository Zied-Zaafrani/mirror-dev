import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from ..registry import SCORERS

logger = logging.getLogger(__name__)

@SCORERS.register("retrieval")
class RetrievalScorer(nn.Module):
    """
    Two-level health-status-aware drug scoring via cross-patient retrieval.
    Implements HI-DR (Hao 2023) Eq. 4-6.
    """
    is_pointer_generator = False

    def __init__(
        self,
        hidden_dim: int = 256,
        learnable_temperature: bool = False,
        fixed_att_tau: float = 10.0,
        **kwargs
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        fixed_tau = float(max(fixed_att_tau, 1e-3))
        init_log_temp = float(torch.log(torch.tensor(fixed_tau, dtype=torch.float32)).item())
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(init_log_temp))
        else:
            self.register_buffer("log_temperature", torch.tensor(init_log_temp))

    def forward(
        self,
        fused_patient: torch.Tensor,
        drug_reprs: torch.Tensor,
        gru_out: torch.Tensor | None = None,
        drug_history: torch.Tensor | None = None,
        similar_reprs: torch.Tensor | None = None,
        similar_multihots: torch.Tensor | None = None,
        **kwargs
    ) -> torch.Tensor:
        if not hasattr(self, "_logged_flow"):
            C = similar_reprs.size(1) if similar_reprs is not None else 0
            K = similar_reprs.size(2) if similar_reprs is not None else 0
            logger.info(
                f"[RetrievalScorer] Forward Flow | Channels={C} | Top-K={K} | "
                f"fused_patient={fused_patient.shape} | drug_reprs={drug_reprs.shape}"
            )
            self._logged_flow = True

        if similar_reprs is None or similar_multihots is None:
            B = fused_patient.size(0)
            D = drug_reprs.size(0)
            return torch.zeros(B, D, device=fused_patient.device)
            
        B, C, K, H = similar_reprs.shape
        
        # Temp for both health-status and channel attention
        temp = torch.exp(self.log_temperature.clamp(min=-1.0, max=5.3))
        
        # 1. Health-Status Scores (per channel)
        # (B, C, K, H) @ (B, H) -> (B, C, K)
        dot_l = torch.einsum("bckh,bh->bck", similar_reprs, fused_patient) / temp
        health_status_weights = F.softmax(dot_l, dim=-1)
        
        # 2. Channel Context & Candidate Drug Scores
        channel_contexts = torch.einsum("bck,bckh->bch", health_status_weights, similar_reprs)
        channel_drug_scores = torch.einsum("bck,bckd->bcd", health_status_weights, similar_multihots)
        
        # 3. Inter-channel Attention
        dot_c = torch.einsum("bch,bh->bc", channel_contexts, fused_patient) / temp
        channel_weights = F.softmax(dot_c, dim=-1)
        
        # 4. Final Blended Retrieval Score
        h4 = torch.einsum("bc,bcd->bd", channel_weights, channel_drug_scores)
        return h4
