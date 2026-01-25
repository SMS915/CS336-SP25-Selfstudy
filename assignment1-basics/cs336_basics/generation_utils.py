import torch
from cs336_basics.model import TransformerLM
from cs336_basics.checkpointing import *
from cs336_basics.BPE.bpe_fast import BPETokenizer
from cs336_basics.utils import Softmax

def generate_text(model: TransformerLM, tokenizer: BPETokenizer, prompt: str, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> str:
    """
    文本生成核心函数，给定提示词，循环进行自回归文本生成
    Args:
        model (TransformerLM): TransformerLM 实例
        tokenizer (BPETokenizer): BPE分词器实例
        prompt (str): 输入的提示词
        max_new_tokens (int): 最大文本生成长度
        temperature (float): 模型温度，间接控制模型输出logits的分布的尖锐性，以控制模型输出的确定性。
            调高可以使得模型输出更加多样，但可能会不够精确，连贯
            调低则可以使得模型输出确定化，保持连贯准确
        top_k (int): 控制模型从最可能的 k 个词中选取下一个词，k最大，生成文本越多样，否则则越确定性
    Returns:
        str:

    """
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # Shape: (1, seq_len)
    input_tensor = input_tensor.to(model.lm_head.weight.device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_tensor.size(1) > model.context_length:
                idx_cond = input_tensor[:, -model.context_length:]
            else:
                idx_cond = input_tensor
            current_len = idx_cond.size(1)
            token_positions = torch.arange(current_len, device=idx_cond.device).unsqueeze(0)
            logits = model(idx_cond, token_positions)  # Shape: (1, seq_len, vocab_size)
            next_token_logits = logits[:, -1, :]  # Shape: (vocab_size,)
            next_token_logits = next_token_logits / temperature
            if top_k is not None:
                top_k_values, _ = torch.topk(next_token_logits, top_k)
                kth_value = top_k_values[:, [-1]]
                next_token_logits[next_token_logits < kth_value] = -float('Inf')
            probabilities = Softmax(next_token_logits,-1)
            next_token_id = torch.multinomial(probabilities, num_samples=1).item()
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]], dtype=torch.long).to(model.lm_head.weight.device)], dim=1)

    generated_text = tokenizer.decode(input_tensor[0].tolist())
    return generated_text