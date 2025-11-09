__package__ = "cs336_basics"
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.checkpointing import *
from cs336_basics.BPE import BPETokenizer
from cs336_basics.utils import Softmax

def generate_text(model: TransformerLM, tokenizer: BPETokenizer, prompt: str, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> str:
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # Shape: (1, seq_len)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_tensor.size(1) > model.context_length:
                start_pos = input_tensor.size(1) - model.context_length
                idx_cond = input_tensor[:, -model.context_length:]
            else:
                start_pos = 0
                idx_cond = input_tensor

            token_positions = torch.arange(start_pos, input_tensor.size(1)).unsqueeze(0)
            logits = model(idx_cond, token_positions)  # Shape: (1, seq_len, vocab_size)
            next_token_logits = logits[:, -1, :]  # Shape: (vocab_size,)
            next_token_logits = next_token_logits / temperature
            if top_k is not None:
                top_k_values, _ = torch.topk(next_token_logits, top_k)
                kth_value = top_k_values[:, [-1]]
                next_token_logits[next_token_logits < kth_value] = -float('Inf')
            probabilities = Softmax(next_token_logits,-1)
            next_token_id = torch.multinomial(probabilities, num_samples=1).item()
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]], dtype=torch.long)], dim=1)

    generated_text = tokenizer.decode(input_tensor[0].tolist())
    return generated_text