import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from ..registry import LOSS_FUNCTIONS

logger = logging.getLogger(__name__)

@LOSS_FUNCTIONS.register("contrastive")
class MIRROR_InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss: patient_repr <-> drug embeddings.
    
    Bridges the patient-drug semantic gap by pulling patient
    representations toward their prescribed drugs in a shared space.
    """
    def __init__(self, num_negatives: int = 16, temperature: float = 0.07, **kwargs):
        super().__init__()
        self.num_negatives = num_negatives
        self.temperature = temperature

    def forward(
        self, 
        patient_repr: torch.Tensor, 
        drug_llm_embed: torch.Tensor, 
        drug_proj: nn.Module, 
        target: torch.Tensor, 
        **kwargs
    ) -> torch.Tensor:
        if not hasattr(self, "_logged_flow"):
            logger.info(f"  [InfoNCELoss] Active | num_neg={self.num_negatives} | temp={self.temperature}")
            self._logged_flow = True
            
        device = patient_repr.device
        # Project drug embeddings to hidden_dim and L2-normalize both
        drug_proj_embed = F.normalize(drug_proj(drug_llm_embed), dim=-1)  # (num_drugs, hidden)
        query = F.normalize(patient_repr, dim=-1)  # (B, hidden)

        # Compute all cosine similarities scaled by temperature
        all_sims = (query @ drug_proj_embed.T) / self.temperature  # (B, num_drugs)

        losses = []
        for i in range(patient_repr.size(0)):
            pos_mask = target[i].bool()
            neg_mask = ~pos_mask
            n_pos = pos_mask.sum().item()
            n_neg_total = neg_mask.sum().item()

            if n_pos == 0 or n_neg_total == 0:
                continue

            n_neg = min(self.num_negatives, n_neg_total)
            neg_idx = neg_mask.nonzero(as_tuple=True)[0]
            neg_sel = neg_idx[torch.randperm(n_neg_total, device=device)[:n_neg]]

            pos_sims = all_sims[i, pos_mask]          # (n_pos,)
            neg_sims = all_sims[i, neg_sel]            # (n_neg,)
            
            logits_nce = torch.cat(
                [pos_sims.unsqueeze(1), neg_sims.unsqueeze(0).expand(n_pos, -1)], dim=1
            )
            labels = torch.zeros(n_pos, dtype=torch.long, device=device)
            losses.append(F.cross_entropy(logits_nce, labels))

        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return torch.stack(losses).mean()

@LOSS_FUNCTIONS.register("margin_rank")
class MIRROR_MarginRankingLoss(nn.Module):
    """Margin ranking loss: widens gap between prescribed and non-prescribed drugs.
    
    Encourages predicted scores for true drugs to be higher than for non-true drugs.
    """
    def __init__(self, margin: float = 1.0, num_negatives: int = 8, **kwargs):
        super().__init__()
        self.margin = margin
        self.num_negatives = num_negatives

    def forward(self, logits: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        if not hasattr(self, "_logged_flow"):
            logger.info(f"  [MarginRankingLoss] Active | margin={self.margin} | num_neg={self.num_negatives}")
            self._logged_flow = True
            
        device = logits.device
        B = logits.size(0)

        losses = []
        for i in range(B):
            pos_idx = target[i].nonzero(as_tuple=True)[0]
            neg_idx = (target[i] == 0).nonzero(as_tuple=True)[0]

            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue

            n_neg = min(self.num_negatives, neg_idx.numel())
            neg_sel = neg_idx[torch.randperm(neg_idx.numel(), device=device)[:n_neg]]

            pos_scores = logits[i, pos_idx]   # (n_pos,)
            neg_scores = logits[i, neg_sel]   # (n_neg,)

            # Pairwise margin: (n_pos, n_neg)
            diff = self.margin - (pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0))
            losses.append(F.relu(diff).mean())

        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return torch.stack(losses).mean()

@LOSS_FUNCTIONS.register("alignment")
class MIRROR_CrossModalAlignmentLoss(nn.Module):
    """G1: cross-modal margin-ranking loss between patient text and drug text.
    
    Uses the predictor's note_proj so patient-text lives in the same projected
    space as drug-text representations.
    """
    def __init__(self, margin: float = 0.3, **kwargs):
        super().__init__()
        self.margin = margin

    def forward(
        self, 
        model: nn.Module, 
        note_embed: torch.Tensor, 
        target: torch.Tensor, 
        has_note: torch.Tensor | None = None, 
        **kwargs
    ) -> torch.Tensor:
        if not hasattr(self, "_logged_flow"):
            logger.info(f"  [CrossModalAlignmentLoss] Active | margin={self.margin}")
            self._logged_flow = True
            
        if (
            not getattr(model, "use_drug_text", False)
            or not hasattr(model, "drug_text_encoder")
            or model.drug_text_encoder is None
            or not hasattr(model, "_aux_drug_text_reprs")
        ):
            return torch.zeros((), device=note_embed.device)

        note_centered = note_embed - model.predictor.note_global_mean
        patient_text_repr = model.predictor.note_proj(note_centered)

        if has_note is not None:
            valid_idx = (has_note > 0).nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                return torch.zeros((), device=note_embed.device)
            patient_text_repr = patient_text_repr[valid_idx]
            target = target[valid_idx]

        from ..drug_text_encoder import cross_modal_alignment_loss
        return cross_modal_alignment_loss(
            patient_text_repr,
            model._aux_drug_text_reprs,
            target,
            margin=self.margin,
        )
