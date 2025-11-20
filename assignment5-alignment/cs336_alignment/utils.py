import torch
import torch.nn.functional as F
def pertoken_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim = -1)
    explog_prob = torch.exp(log_prob)
    multiplication = log_prob * explog_prob
    return -torch.sum(multiplication, dim=-1)
