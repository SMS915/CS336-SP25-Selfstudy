import torch
import torch.nn as nn
from cs336_basics.Linear import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = 64 *  ((round(self.d_model * 8/3) + 63) // 64) if d_ff is None else d_ff
        self.W1 = Linear(self.d_model, self.d_ff)
        self.W2 = Linear(self.d_ff, self.d_model)
        self.W3 = Linear(self.d_model, self.d_ff)

    def SiLU(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(self.SiLU(self.W1(x)) * self.W3(x))



