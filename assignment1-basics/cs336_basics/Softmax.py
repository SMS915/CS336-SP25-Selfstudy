import torch

def Softmax(x: torch.Tensor, i: int = 0) -> torch.Tensor:
    sub_x = x - torch.max(x, dim = i, keepdim=True)[0]
    return torch.exp(sub_x) / torch.sum(torch.exp(sub_x), dim = i, keepdim=True)
