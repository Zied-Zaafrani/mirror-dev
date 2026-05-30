import torch
import torch.nn as nn
import logging
from ..registry import TEMPORAL_ENCODERS

logger = logging.getLogger(__name__)

@TEMPORAL_ENCODERS.register("cnn_gru")
class CNNGRUEncoder(nn.Module):
    """CNN + GRU encoder (isolates CNN contribution)."""
    def __init__(self, hidden_dim, num_layers=1, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.cnn = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(
            input_size=hidden_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.0
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths, **kwargs):
        """
        Args:
            x: (B, T, H)
            lengths: (B,)
        Returns:
            output: (B, T, H)
        """
        if not hasattr(self, "_logged_flow"):
            logger.info(f"  [Temporal] CNN-GRU Backbone Active | layers={self.num_layers} | hidden={self.hidden_dim}")
            self._logged_flow = True

        T = x.size(1)
        # 1) CNN local context
        x = x.transpose(1, 2)
        x = self.relu(self.cnn(x))
        x = x.transpose(1, 2)
        
        # 2) GRU longitudinal
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            gru_out_packed, _ = self.gru(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                gru_out_packed, batch_first=True, total_length=T
            )
        else:
            output, _ = self.gru(x)
            
        return output
