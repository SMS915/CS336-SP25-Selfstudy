import torch
import torch.nn.functional as F
def pertoken_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim = -1) # shape: batch_size, seq_len, vocab_size
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return torch.nan_to_num(entropy, nan=0.0)# batch_size, seq_len
