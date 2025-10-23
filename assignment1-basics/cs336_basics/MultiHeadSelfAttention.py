import torch
import torch.nn as nn
from einops import rearrange
from cs336_basics.ScaledDotProductAttention import ScaledDotProductAttention
from cs336_basics.RoPE import RoPE
from cs336_basics.Linear import Linear

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = None, theta: float = None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        self.d_k = self.d_v = self.d_model // self.num_heads
        self.q_proj = Linear(self.d_model, self.d_model)
        self.k_proj = Linear(self.d_model, self.d_model)
        self.v_proj = Linear(self.d_model, self.d_model)

        self.out_proj = Linear(self.d_model, self.d_model)

        self.rope = RoPE(theta, self.d_k, self.max_seq_len) if theta is not None and max_seq_len is not None else None

    def forward(self, x : torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        context_length = query.shape[1]
        queries_multi_head = rearrange(query, "b s (h d) -> b h s d", h=self.num_heads)
        keys_multi_head = rearrange(key, "b s (h d) -> b h s d", h=self.num_heads)
        values_multi_head = rearrange(value, "b s (h d) -> b h s d", h=self.num_heads)

        mask = torch.tril(torch.ones(context_length, context_length, device=x.device, dtype=torch.bool), diagonal=0)
        mask.unsqueeze_(0).unsqueeze_(0)
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(context_length, device=x.device).unsqueeze(0).expand(query.shape[0], -1)
                # token_positions = repeat(torch.arange(context_length, device=x.device), 's -> b s', b=x.shape[0])  # einops version
            q_rope = self.rope(queries_multi_head, token_positions)
            k_rope = self.rope(keys_multi_head, token_positions)
            out = ScaledDotProductAttention(q_rope, k_rope, values_multi_head, mask) # b h s d
        else:
            out = ScaledDotProductAttention(queries_multi_head, keys_multi_head, values_multi_head, mask)

        out = rearrange(out, "b h s d -> b s (h d)", h=self.num_heads, d=self.d_v)

        return self.out_proj(out)











