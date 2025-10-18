import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.gamma = nn.Parameter(torch.ones(d_model).to(self.device), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_type = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(x.pow(2).mean(dim = -1, keepdim=True) + self.eps)
        x_norm = x / rms

        return (self.gamma * x_norm).to(input_type)