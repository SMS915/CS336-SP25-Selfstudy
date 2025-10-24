import math
import torch

def GradientClipping(parameter: torch.nn.parameter, max_norm: float, eps: float = 1e-6):
    total_norm = 0
    for p in parameter:
        if p.grad is not None:
            param_norm = p.grad.data.norm()
            total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5

    if total_norm > max_norm:
        scale_factor = max_norm / (total_norm + eps)
        for p in parameter:
            if p.grad is not None:
                p.grad.data.mul_(scale_factor)