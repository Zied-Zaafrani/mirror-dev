import torch
import torch.nn as nn
import json
import logging

logger = logging.getLogger(__name__)

class ContraindicationPrior(nn.Module):
    """
    Clinical Contraindication Prior for MIRROR.
    
    Refactored for Phase 3.3 Hardening:
    - Acts as a multi-hot mask generator.
    - Designed for post-hoc suppression in the predictor head.
    - Training-neutral by default (Rule 1: Component Isolation).
    """
    def __init__(self, json_path: str, num_drugs: int):
        super().__init__()
        
        # Load rules from JSON
        # matrix keys: "lab_idx_binval" -> list of contraindicated drug indices
        try:
            with open(json_path, 'r') as f:
                rules = json.load(f)
        except Exception as e:
            logger.error(f"[ContraPrior] Failed to load rules from {json_path}: {e}")
            rules = {}
            
        # mask[i, b, d] = 1 if lab i at bin b contraindicates drug d
        self.num_rules = 18 # We only have clinical rules for the core 18 labs
        mask = torch.zeros(self.num_rules, 4, num_drugs)
        for key, drugs in rules.items():
            try:
                lab_idx, bin_val = map(int, key.split('_'))
                if lab_idx < self.num_rules:
                    for d in drugs:
                        if d < num_drugs:
                            mask[lab_idx, bin_val, d] = 1.0
            except ValueError:
                continue
                
        self.register_buffer('mask', mask)
        logger.info(f"[ContraPrior] Initialized with {len(rules)} clinical rules for {num_drugs} drugs.")
        
    def forward(self, lab_bins: torch.Tensor | None) -> torch.Tensor:
        """
        Computes the patient-specific contraindication mask.
        
        Args:
            lab_bins: (B, 18) indices [0-3] representing lab value bins.
        Returns:
            patient_mask: (B, num_drugs) binary mask (1.0 = contraindicated).
        """
        if lab_bins is None:
            # FIX-B15: caller is expected to pass batch-shaped tensor; if not,
            # return (1, num_drugs) which broadcasts safely in the subtraction.
            # Document the contract.
            return torch.zeros(1, self.mask.size(2), device=self.mask.device)

        B = lab_bins.size(0)
        num_drugs = self.mask.size(2)
        device = lab_bins.device
        
        # patient_mask[b, d] = 1 if any lab violation exists
        patient_mask = torch.zeros(B, num_drugs, device=device)
        
        # Vectorized gather: 
        # 1. Expand mask to (B, 18, 4, num_drugs)
        # 2. Use lab_bins to select bin index for each lab
        
        # We can iterate over the 18 labs (small constant) for clarity and safety
        for i in range(self.num_rules):
            bins = lab_bins[:, i] # (B,)
            # self.mask[i, bins] -> (B, num_drugs)
            patient_mask = torch.max(patient_mask, self.mask[i, bins])
            
        return patient_mask
