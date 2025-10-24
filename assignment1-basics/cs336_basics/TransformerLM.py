from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat, einsum
from jaxtyping import Float, Int

from cs336_basics.Embedding import Embedding
from cs336_basics.Linear import Linear
from cs336_basics.TransformerBlock import TransformerBlock
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.Softmax import Softmax

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int,
                 num_layers: int, num_heads: int, d_ff: int, rope_theta: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.embed = Embedding(self.vocab_size, self.d_model)
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.blocks.append(TransformerBlock(self.d_model, self.num_heads, self.d_ff, self.rope_theta, self.context_length))
        self.norm_final = RMSNorm(self.d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, tokens: Float[torch.Tensor, "batch_size seq_len"], token_positions: Optional[Int[torch.Tensor, "batch_size, seq_len"]] = None) -> torch.Tensor:
        x = self.embed(tokens)  # batch_size seq_len d_model
        if token_positions is None:
            batch_size = tokens.shape[0]
            seq_len = tokens.shape[1]
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        for block in self.blocks:
            x = block(x, token_positions)
        x = self.norm_final(x)
        x = self.lm_head(x) # batch_size seq_len vocab_size
        return x


