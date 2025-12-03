from typing import Literal, Optional
import torch
from math import cos, pi
from jaxtyping import Float
import torch.nn.functional as F


def Softmax(x: torch.Tensor, i: int = 0) -> torch.Tensor:
    scaled_x = x - torch.max(x, dim = i, keepdim=True)[0]
    return torch.exp(scaled_x) / torch.sum(torch.exp(scaled_x), dim = i, keepdim=True)

# def cross_entropy_loss(inputs: Float[torch.Tensor, " batch_size vocab_size"], target: Float[torch.Tensor, " batch_size"]) -> torch.Tensor:
#     max_logit = torch.max(inputs, dim = 1, keepdim = True)[0]
#     log_sum_exp = torch.log(torch.sum(torch.exp(inputs - max_logit), dim = 1, keepdim = True))
#     correct_log = torch.gather(inputs, -1, target.unsqueeze(-1))
#     return (max_logit + log_sum_exp - correct_log).mean()

def cross_entropy_loss(inputs, target):
    return F.cross_entropy(inputs, target)

def get_lr_schedule(t: int, t_warm: int, t_cycle: int, lr_max: float, lr_min: float, choice: Literal['cosine', 'wsd'] = 'cosine', t_max: Optional[int] = None) -> Float:
    if choice == 'cosine':
        if t < t_warm:
            return t * lr_max / t_warm
        elif t > t_cycle:
            return lr_min
        else:
            return lr_min + (lr_max - lr_min) * (1 + cos((t - t_warm) * pi / (t_cycle - t_warm))) / 2
    elif choice == 'wsd':
        assert t_max is not None
        if t < t_warm:
            return t * lr_max / t_warm
        elif t < t_cycle:
            return lr_max
        elif t < t_max:
            return lr_max - (t - t_cycle) * (lr_max - lr_min) / (t_max - t_cycle)
        else:
            return lr_min 

def clip_gradient(parameter: torch.nn.parameter, max_norm: float, eps: float = 1e-6):
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

    return total_norm