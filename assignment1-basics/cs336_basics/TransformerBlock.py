import torch
import torch.nn as nn
from einops import rearrange, einsum
from jaxtyping import Float

from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.Linear import Linear
from cs336_basics.RoPE import RoPE
from cs336_basics.MultiHeadSelfAttention import MultiHeadSelfAttention
from cs336_basics.SwiGLU import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int = None,theta: float = None, max_seq_len: int = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.norm1 = RMSNorm(self.d_model)
        self.norm2 = RMSNorm(self.d_model)
        self.attn = MultiHeadSelfAttention(self.d_model, self.num_heads, self.max_seq_len, self.theta)
        self.ffn = SwiGLU(self.d_model, self.d_ff)

    def forward(self, x: Float[torch.Tensor, "batch_size seq_len d_model"], token_positions: Float[torch.Tensor, "batch_size seq_len"] = None) -> torch.Tensor:
        x_attn = self.attn(self.norm1(x), token_positions)
        x1 = x + x_attn

        x_ff = self.ffn(self.norm2(x1))
        x2 = x1 + x_ff

        return x2




