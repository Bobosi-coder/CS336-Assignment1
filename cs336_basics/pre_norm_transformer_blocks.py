"""
实现 assignment 3.4 seciton 中的 RMSNorm, Swiglu, RotaryPositionalEmbedding,

"""

from xmlrpc.client import boolean

from numpy import full
from sympy import Line
import torch.nn as nn
import torch
from einops import rearrange, einsum
import einops
import math
from cs336_basics.basic_blocks import Linear

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.gain = nn.Parameter(torch.ones(d_model, device= device, dtype = dtype))
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(einops.reduce(x**2, "... d_model -> ... 1", "mean") + self.eps)

        result = x/rms * self.gain
        return result.to(in_dtype)
    
class Swiglu(nn.Module):
    def __init__(self, d_model: int, d_ff: int,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()

        self.d_model = d_model
        if d_ff:
            self.d_ff = d_ff
        else:
            d_ff = int(8/3 * d_model)
            self.d_ff = (d_ff + 63) // 64 * 64

        self.w_gate = Linear(self.d_model, self.d_ff)
        self.w_down = Linear(self.d_model, self.d_ff)
        self.w_up = Linear(self.d_ff, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated_value = self.w_gate(x)
        glu_value = gated_value * torch.sigmoid(gated_value) * self.w_down(x)
        return self.w_up(glu_value)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        
        seq_vector = torch.arange(max_seq_len, device = device)
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float().to(device) / d_k))

        freq_matrix = torch.outer(seq_vector, inv_freq)
        cos_matrix = torch.cos(freq_matrix)
        sin_matrix = torch.sin(freq_matrix)
        interleaved_cos_sin = rearrange([cos_matrix, sin_matrix], "pair ... d_half -> ... (d_half pair)")
        interleaved_neg_sin_cos = rearrange([-sin_matrix, cos_matrix], "pair ... d_half -> ... (d_half pair)")

        self.register_buffer("interleaved_cos_sin", interleaved_cos_sin, persistent=False)
        self.register_buffer("interleaved_neg_sin_cos", interleaved_neg_sin_cos, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_even = einops.repeat(x_even, "... d_half -> ... (d_half 2)")
        x_odd = einops.repeat(x_odd, "... d_half -> ... (d_half 2)")

        return x_even * self.interleaved_cos_sin[token_positions] + x_odd * self.interleaved_neg_sin_cos[token_positions] 
    

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_value, _ = torch.max(x, dim= dim, keepdim=True)
    x_scaled = x - max_value
    x_exp = torch.exp(x_scaled)
    x_exp_sum = torch.sum(x_exp, dim= dim, keepdim=True)
    return x_exp / x_exp_sum


def scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
    d_k = Q.shape[-1]

    self_attention_product = einsum(Q, K, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k")
    scaled_attention_product = self_attention_product/math.sqrt(d_k)

    if mask is not None:
        masked_scaled_attention_product = scaled_attention_product.masked_fill(~mask, float('-inf'))
    else: 
        masked_scaled_attention_product = scaled_attention_product

    attention_score = softmax(masked_scaled_attention_product, -1)
    return einsum(attention_score, V, "... seq_len_q seq_len, ... seq_len d_k -> ... seq_len_q d_k")

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_rope: bool = False,
                 max_seq_len: int = None, theta: float = None, 
                 device: torch.device | None = None):
        super().__init__()

        assert d_model % num_heads == 0

        self.use_rope = use_rope
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # in_feature_dim = int(self.d_model)
        # out_feature_dim = int(self.d_k * self.num_heads)

        self.Wq = Linear(self.d_model, self.d_k * self.num_heads)
        self.Wk = Linear(self.d_model, self.d_k * self.num_heads)
        self.Wv = Linear(self.d_model, self.d_k * self.num_heads)
        self.Wo = Linear(self.d_k * self.num_heads, self.d_model)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        mask_shape_ones = torch.ones(seq_len, seq_len, dtype= torch.bool)
        mask = torch.tril(mask_shape_ones, 0)

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        Q_multihead = rearrange(Q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", 
                              num_heads = self.num_heads)
        K_multihead = rearrange(K, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", 
                              num_heads = self.num_heads)
        V_multihead = rearrange(V, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", 
                              num_heads = self.num_heads)
        
        # 是否使用 rope
        if self.use_rope:
            d_k = Q_multihead.shape[-1]
            rope = RotaryPositionalEmbedding(self.theta, d_k= d_k, max_seq_len= self.max_seq_len,
                                             device = self.device)
            Q_multihead = rope(Q_multihead, token_positions)
            K_multihead = rope(K_multihead, token_positions)

        concated_attention_multihead = scaled_dot_product_attention(Q_multihead, K_multihead, V_multihead, mask)

        concated_attention = rearrange(concated_attention_multihead, 
                                       "... num_heads seq_len d_k -> ... seq_len (num_heads d_k)")
        result = self.Wo(concated_attention)
        return result
        
        
