import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from ..registry import TEMPORAL_ENCODERS

logger = logging.getLogger(__name__)

class FastformerLayer(nn.Module):
    """Fastformer additive attention layer (O(N) complexity)."""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        
        self.q_attn = nn.Linear(hidden_dim, num_heads)
        self.k_attn = nn.Linear(hidden_dim, num_heads)
        
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, H)
            mask: (B, T) bool, True = ignore
        """
        B, T, H = x.shape
        
        # 1) Projections
        q = self.W_q(x)  # (B, T, H)
        k = self.W_k(x)  # (B, T, H)
        v = self.W_v(x)  # (B, T, H)
        
        # 2) Global Query (Q_g)
        alpha_q = self.q_attn(q)
        if mask is not None:
            alpha_q = alpha_q.masked_fill(mask.unsqueeze(-1), float("-inf"))
        alpha_q = F.softmax(alpha_q, dim=1)
        
        # Q_g: (B, heads, head_dim)
        q_reshaped = q.view(B, T, self.num_heads, self.head_dim)
        q_g = torch.einsum('bth,bthd->bhd', alpha_q, q_reshaped)
        q_g = q_g.view(B, 1, H) # (B, 1, H)
        
        # 3) Interaction between query and keys
        p = q_g * k
        
        # 4) Global Key (K_g)
        alpha_k = self.k_attn(p)
        if mask is not None:
            alpha_k = alpha_k.masked_fill(mask.unsqueeze(-1), float("-inf"))
        alpha_k = F.softmax(alpha_k, dim=1)
        
        # K_g: (B, heads, head_dim)
        p_reshaped = p.view(B, T, self.num_heads, self.head_dim)
        k_g = torch.einsum('bth,bthd->bhd', alpha_k, p_reshaped)
        k_g = k_g.view(B, 1, H) # (B, 1, H)
        
        # 5) Global Key and Values Interaction
        out = (k_g * v)
        out = self.proj(out)
        out = self.dropout(out)
        
        return x + out

@TEMPORAL_ENCODERS.register("fastformer")
class FastformerEncoder(nn.Module):
    """Fastformer encoder for efficient temporal processing."""
    def __init__(self, hidden_dim, num_layers=1, num_heads=4, dropout=0.3, use_cnn=False):
        super().__init__()
        self.num_layers = num_layers
        self.use_cnn = use_cnn
        if use_cnn:
            self.cnn = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            
        self.layers = nn.ModuleList([
            FastformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, lengths, **kwargs):
        """
        Args:
            x: (B, T, H)
            lengths: (B,)
        Returns:
            output: (B, T, H)
        """
        if not hasattr(self, "_logged_flow"):
            cnn_str = "+CNN" if self.use_cnn else ""
            logger.info(f"  [Temporal] Fastformer{cnn_str} Backbone Active | layers={self.num_layers}")
            self._logged_flow = True

        device = x.device
        if self.use_cnn:
            # (B, T, H) -> (B, H, T)
            x = x.transpose(1, 2)
            x = self.relu(self.cnn(x))
            x = x.transpose(1, 2)
            
        T = x.size(1)
        if lengths is not None:
            mask = torch.arange(T, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        else:
            mask = None
            
        for layer in self.layers:
            x = layer(x, mask)
            
        return x

@TEMPORAL_ENCODERS.register("cnn_fastformer")
class CNNFastformerEncoder(FastformerEncoder):
    """Fastformer encoder with 1D-CNN."""
    def __init__(self, hidden_dim, num_layers=1, num_heads=4, dropout=0.3):
        super().__init__(hidden_dim, num_layers, num_heads, dropout, use_cnn=True)
