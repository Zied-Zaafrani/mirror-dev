import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
import logging
from typing import Optional, Tuple
from ..registry import TEMPORAL_ENCODERS

logger = logging.getLogger(__name__)

# --- Minimal helpers to replace x_transformers/einops dependencies ---
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )
    def forward(self, x):
        return self.net(x)

def rearrange_b_n_hd(t, h):
    b, n, hd = t.shape
    d = hd // h
    return t.view(b, n, h, d).transpose(1, 2)

def rearrange_b_h_n_d(t):
    b, h, n, d = t.shape
    return t.transpose(1, 2).reshape(b, n, h * d)

def rotate_half(x):
    x = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(t, freqs):
    seq_len = t.shape[-2]
    freqs = freqs[-seq_len:, :]
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs

# --- SDRBT Recurrent Attention Block (BRT) Components ---

class RecurrentStateGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.main_proj = nn.Linear(dim, dim, bias = True)
        self.input_proj = nn.Linear(dim, dim, bias = True)
        self.forget_proj = nn.Linear(dim, dim, bias = True)
    
    def forward(self, x, state):
        z = torch.tanh(self.main_proj(x))
        i = torch.sigmoid(self.input_proj(x) - 1)
        f = torch.sigmoid(self.forget_proj(x) + 1)
        return torch.mul(state, f) + torch.mul(z, i)

class SDRBTAttention(nn.Module):
    def __init__(self, dim, *, dim_head = 64, heads = 8, causal = False, dropout = 0., null_kv = False):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        inner_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.null_kv = nn.Parameter(torch.randn(2, inner_dim)) if null_kv else None

    def forward(self, x, mask = None, context = None, pos_emb = None):
        b, device, h, scale = x.shape[0], x.device, self.heads, self.scale

        x = self.norm(x)
        kv_input = default(context, x)

        q = self.to_q(x)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q = rearrange_b_n_hd(q, h)
        k = rearrange_b_n_hd(k, h)
        v = rearrange_b_n_hd(v, h)

        q = q * scale

        if exists(pos_emb):
            q_pos_emb, k_pos_emb = pos_emb if isinstance(pos_emb, tuple) else (pos_emb, pos_emb)
            q = apply_rotary_pos_emb(q, q_pos_emb)
            k = apply_rotary_pos_emb(k, k_pos_emb)

        if exists(self.null_kv):
            nk, nv = self.null_kv.unbind(dim = 0)
            nk = nk.view(1, h, 1, -1).expand(b, h, 1, -1)
            nv = nv.view(1, h, 1, -1).expand(b, h, 1, -1)
            k = torch.cat((nk, k), dim = -2)
            v = torch.cat((nv, v), dim = -2)

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        mask_value = -torch.finfo(sim.dtype).max
        
        if exists(mask):
            i_len, j_len = sim.shape[-2], sim.shape[-1]
            if self.causal and i_len == j_len:
                tril_mask = torch.tril(torch.ones(j_len, j_len, device=device)).view(1, 1, j_len, j_len).bool()
                sim = sim.masked_fill(~tril_mask, mask_value)
            if exists(self.null_kv):
                mask = F.pad(mask, (1, 0), value=True)
            if mask.shape[-1] == j_len:
                mask_reshaped = mask.view(b, 1, 1, j_len)
                sim = sim.masked_fill(~mask_reshaped, mask_value)

        if self.causal and sim.shape[-2] == sim.shape[-1]:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones(i, j, device = device, dtype = torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange_b_h_n_d(out)
        return self.to_out(out), None

class SDRBTBlockRecurrentAttention(nn.Module):
    def __init__(self, dim: int, dim_state: int, dim_head: int = 64, state_len: int = 512, heads: int = 8, **kwargs):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dim = dim
        self.dim_state = dim_state
        self.heads = heads
        self.causal = True
        self.state_len = state_len
        rotary_emb_dim = max(dim_head // 2, 64)
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim)
        
        self.input_self_attn = SDRBTAttention(dim, heads = heads, causal = True)
        self.state_self_attn = SDRBTAttention(dim_state, heads = heads, causal = False)

        self.input_state_cross_attn = SDRBTAttention(dim, heads = heads, causal = False)
        self.state_input_cross_attn = SDRBTAttention(dim_state, heads = heads, causal = False)

        self.proj_gate = RecurrentStateGate(dim)
        self.ff_gate = RecurrentStateGate(dim)

        self.input_proj = nn.Linear(dim + dim_state, dim, bias = False)
        self.state_proj = nn.Linear(dim + dim_state, dim, bias = False)

        self.input_ff = FeedForward(dim)
        self.state_ff = FeedForward(dim_state)

    def forward(self, x, state: Optional[torch.Tensor] = None, mask = None, state_mask = None, rel_pos = None, rotary_pos_emb = None, prev_attn = None, mem = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, device = x.shape[0], x.shape[-2], x.device
        if not exists(state):
            state = torch.zeros((batch, self.state_len, self.dim_state)).to(x.device)

        # In SDRBT they passed lists, we just pass the length
        self_attn_pos_emb = self.rotary_pos_emb(seq_len, device=device)
        state_pos_emb = self.rotary_pos_emb(self.state_len, device=device)

        input_attn, _ = self.input_self_attn(x, mask = mask, pos_emb = self_attn_pos_emb)
        state_attn, _ = self.state_self_attn(state, mask = state_mask, pos_emb = state_pos_emb)

        input_as_q_cross_attn, _ = self.input_state_cross_attn(x, context = state, mask = mask)
        state_as_q_cross_attn, _ = self.state_input_cross_attn(state, context = x, mask = state_mask)

        projected_input = self.input_proj(torch.cat((input_as_q_cross_attn, input_attn), dim=2))
        projected_state = self.state_proj(torch.cat((state_as_q_cross_attn, state_attn), dim=2))

        input_residual = projected_input + x
        state_residual = self.proj_gate(projected_state, state)

        output = self.input_ff(input_residual) + input_residual
        next_state = self.ff_gate(self.state_ff(state_residual), state_residual)

        return output, next_state

# --- Wrapper for MIRROR ---

@TEMPORAL_ENCODERS.register("brt")
class BRTEncoder(nn.Module):
    """SDRBT Block Recurrent Transformer Encoder wrapper."""
    def __init__(self, hidden_dim, state_len=64, heads=4, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_len = state_len
        self.heads = heads
        self.brt = SDRBTBlockRecurrentAttention(
            dim=hidden_dim, 
            dim_state=hidden_dim, 
            dim_head=hidden_dim//heads,
            state_len=state_len,
            heads=heads
        )

    def forward(self, x, lengths, **kwargs):
        """
        Args:
            x: (B, T, H)
            lengths: (B,)
        Returns:
            output: (B, T, H)
        """
        if not hasattr(self, "_logged_flow"):
            logger.info(f"  [Temporal] SDRBT-BRT Backbone Active | state_len={self.state_len} | heads={self.heads}")
            self._logged_flow = True

        B, T, H = x.shape
        device = x.device
        
        if lengths is not None:
            mask = torch.arange(T, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        else:
            mask = None
            
        out, next_state = self.brt(x, state=None, mask=mask)
            
        return out
