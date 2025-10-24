import torch
import torch.nn as nn
from jaxtyping import Float
from cs336_basics.Softmax import Softmax


def cross_entropy_loss(inputs: Float[torch.Tensor, " batch_size vocab_size"], target: Float[torch.Tensor, " batch_size"]) -> torch.Tensor:
    max_logit = torch.max(inputs, dim = 1, keepdim = True)[0]
    log_sum_exp = torch.log(torch.sum(torch.exp(inputs - max_logit), dim = 1, keepdim = True))
    correct_log = torch.gather(inputs, -1, target.unsqueeze(-1))
    return (max_logit + log_sum_exp - correct_log).mean()




