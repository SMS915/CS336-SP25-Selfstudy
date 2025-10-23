import torch
import torch.nn as nn
from cs336_basics.Linear import Linear


def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    """
    SwiGLU 前馈网络模块，遵循 LLaMA 等现代Transformer架构。
    公式: FFN(x) = W2( SiLU(x @ W1) ⊙ (x @ W3) )
    """
    def __init__(self, d_model: int, d_ff: int = None):
        """
        Args:
            d_model (int): 输入和输出维度。
            d_ff (int, optional): 隐藏层维度。若为None，则自动计算并对齐到64的倍数。
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = 64 *  ((round(self.d_model * 8/3) + 63) // 64) if d_ff is None else d_ff
        self.W1 = Linear(self.d_model, self.d_ff)
        self.W2 = Linear(self.d_ff, self.d_model)
        self.W3 = Linear(self.d_model, self.d_ff)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量。Shape: (..., d_model)
        Returns:
            torch.Tensor: 输出张量。Shape: (..., d_model)
        """
        return self.W2(SiLU(self.W1(x)) * self.W3(x))



