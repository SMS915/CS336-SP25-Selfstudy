import einx
import torch
import torch.nn as nn
from einops import rearrange


class RoPE(nn.Module):
    """
    RoPE (旋转位置编码) 模块。
    通过旋转输入向量的成对维度来注入位置信息，实现了用绝对位置编码相对位置注意力的能力。
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        """
        Args:
            d_k (int): 输入向量的维度 (必须是偶数)。
            max_seq_len (int): 模型支持的最大序列长度。
            theta (float): RoPE的基数，默认为10000.0。
        """
        super().__init__()
        self.theta = theta if theta else 10000.0
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        k_vec = torch.arange(0, self.d_k, 2) / self.d_k
        freq = 1.0 / self.theta ** k_vec

        rotation_matrix = torch.outer(torch.arange(0, max_seq_len), freq)
        cos_table = torch.cos(rotation_matrix)
        sin_table = torch.sin(rotation_matrix)
        self.register_buffer('cos_table', cos_table, persistent=False)
        self.register_buffer('sin_table', sin_table, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量。Shape: (batch_size, seq_len, d_k)
            token_positions (torch.Tensor): 每个token的绝对位置。Shape: (batch_size, seq_len)
        Returns:
            torch.Tensor: 应用RoPE后的张量。Shape: (batch_size, seq_len, d_k)
        """
        cos, sin = self.cos_table[token_positions], self.sin_table[token_positions]
        even_x, odd_x = rearrange(x, '... (d_half odd_even) -> odd_even ... d_half', odd_even=2)
        x1_rot = even_x * cos - odd_x * sin
        x2_rot = even_x * sin + odd_x * cos
        return einx.rearrange('... d_half, ... d_half -> ... (d_half (1 + 1))', x1_rot, x2_rot).contiguous()





