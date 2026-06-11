"""
实现 assignment 3.3 seciton 中的 linear, embedding blocks
"""

from tkinter import NO

from numpy import full
from sympy import Line
import torch.nn as nn
import torch
from einops import rearrange, einsum
import einops
import math

class Linear(nn.Module):
    def __init__(self, in_feature: int, out_feature: int,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        std = math.sqrt(2/(in_feature + out_feature))

        self.weights = nn.Parameter(
                nn.init.trunc_normal_(
                    torch.empty(out_feature, in_feature,device=device, dtype=dtype),
            mean = 0, std = std, a = -3*std, b = 3 * std)
            )
    
    def forward(self, x : torch.tensor) -> torch.Tensor:
        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, 
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.embedding_matrix = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(num_embeddings, embedding_dim, device = device, dtype= dtype),
            mean = 0,
            std = 1, a = -3, b = 3)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids]        