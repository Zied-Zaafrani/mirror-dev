import torch
import torch.nn as nn
import math
import logging

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerTemporalEncoder(nn.Module):
    """Standard Transformer encoder for visit sequences."""
    def __init__(self, hidden_dim, num_layers=2, num_heads=4, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, lengths, **kwargs):
        """
        Args:
            x: (batch, T, hidden_dim)
            lengths: (batch,) actual visits
        Returns:
            output: (batch, T, hidden_dim)
        """
        if not hasattr(self, "_logged_flow"):
            logger.info(f"  [Temporal] Transformer Backbone Active | layers={self.num_layers} | heads={self.num_heads}")
            self._logged_flow = True

        T = x.size(1)
        device = x.device
        
        # 1) Transformer backbone handles temporal relationships via self-attention.
        x = self.pos_encoding(x)
        
        # 2) Causal mask (prevent future leakage)
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
        mask = mask == float("-inf")
        
        # 3) Padding mask (key_padding_mask: True = ignore)
        if lengths is not None:
            padding_mask = torch.arange(T, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        else:
            padding_mask = None
            
        # 4) Run transformer
        output = self.encoder(x, mask=mask, src_key_padding_mask=padding_mask)
        
        return output

class TransformerTemporalEncoder4L(TransformerTemporalEncoder):
    """4-layer, 8-head Transformer for Config 3 (deeper temporal encoder)."""
    def __init__(self, hidden_dim, num_layers=4, num_heads=8, dropout=0.3):
        super().__init__(hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout)
