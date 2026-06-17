import torch 
import torch.nn as nn
from einops import rearrange, einsum
import einops

from cs336_basics.basic_blocks import Embedding
from cs336_basics.pre_norm_transformer_blocks import *

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 use_rope: bool = False,
                 max_seq_len: int = None, theta: float = None, 
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()

        self.pre_attention_norm = RMSNorm(d_model=d_model, device= device, dtype=dtype)
        self.mha = MultiheadSelfAttention(d_model, num_heads, 
                                                use_rope= use_rope, max_seq_len= max_seq_len,
                                                theta= theta, device= device, dtype=dtype)
        self.pre_swiglu_norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.swiglu = Swiglu(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mha(self.pre_attention_norm(x))
        x = x + self.swiglu(self.pre_swiglu_norm(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 vocab_size: int, context_length: int, num_layers: int,
                 use_rope: bool = True,
                 theta: float = None, 
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.context_length = context_length
        self.use_rope = use_rope
        self.rope_theta = theta
        self.device = device

        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model,
                                   device= device, dtype = dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model = d_model,
                num_heads = num_heads,
                d_ff = d_ff,
                max_seq_len = context_length,
                theta = theta,
                use_rope = use_rope,
                device = device,
                dtype = dtype
            )
            for _ in range(num_layers)
        ])

        # 2. 按照 3.5 节规范：Pre-LN 架构要求在层堆叠之后、LM Head 之前，施加一个最终的 RMSNorm
        self.ln_f = RMSNorm(d_model=d_model, device=device, dtype=dtype)

        # 3. 语言模型输出头：将特征维度映射回词表大小
        self.lm_head = Linear(in_feature=d_model, out_feature=vocab_size, 
                              device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_features = self.embedding(x)
        for layer in self.layers:
            in_features = layer(in_features)
        
        in_features = self.ln_f(in_features)
        logits = self.lm_head(in_features)

        return logits
        
