import torch
import random
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Callable
from .utils import pertoken_entropy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerateDecoderOnlyOutput


def tokenize_prompt_and_output(prompt_strs: List[str], output_strs: List[str], tokenizer: PreTrainedTokenizerBase, max_length: int = 1024):
    batch_tokens = []
    batch_mask = []
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_c = prompt.strip()
        output_c = output.strip()

        clean_output = output
        if prompt_c.endswith('<think>') and output_c.startswith('<think>'):
            clean_output = output_c[7:].lstrip()

        prompt_ids = tokenizer.encode(prompt, add_special_tokens = False)
        output_ids = tokenizer.encode(clean_output, add_special_tokens = False)
        # eos_id = tokenizer.eos_token_id
        # assert isinstance(eos_id, int)
        # if tokenizer.eos_token_id is not None:
        #     output_ids.append(eos_id)
        # else:
        #     pass
        token_ids = prompt_ids + output_ids
        output_mask = [0] * len(prompt_ids) + [1] * len(output_ids)
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            output_mask = output_mask[:max_length]
            
        batch_tokens.append(torch.tensor(token_ids, dtype=torch.long))
        batch_mask.append(torch.tensor(output_mask, dtype=torch.long))

    max_len = max(len(ids) for ids in batch_tokens)
    padded_input_ids = []
    padded_masks = []
    padded_labels = []
    for input_ids, masks in zip(batch_tokens, batch_mask):
        pad_len = max_len - len(input_ids)
        padded_input = F.pad(input=input_ids, pad=(0, pad_len), value=tokenizer.pad_token_id) # 对最后一个dim,左填充0个，右填充pad_len个
        padded_mask = F.pad(input=masks, pad=(0, pad_len), value=0)

        padded_input_ids.append(padded_input)
        padded_masks.append(padded_mask)

    batch_input_ids = torch.stack(padded_input_ids)
    batch_masks = torch.stack(padded_masks)

    shifted_inputs = batch_input_ids[:, :-1]
    shifted_masks = batch_masks[:, 1:]
    labels = batch_input_ids[:, 1:]

    return {
        "input_ids": shifted_inputs,
        "labels": labels,
        "response_mask": shifted_masks
    }

def get_response_log_probs(model: torch.nn.Module, input_ids:torch.Tensor, labels: torch.Tensor, return_token_entropy: bool = False) -> Dict[str, torch.Tensor]:
    """
    Args:
        model: PreTrainedModel, HuggingFace model used for scoring (placed on the correct device
            and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor, shape (batch_size, sequence_length), concatenated prompt +
                response tokens as produced by your tokenization method.
        labels: torch.Tensor, shape (batch_size, sequence_length), labels as produced by your
                tokenization method.
        return_token_entropy: bool If True, also return per-token entropy by calling
                            compute_entropy.

    Returns:
        dict[str, torch.Tensor].
        "log_probs" shape (batch_size, sequence_length), conditional log-probabilities
        log pθ(xt | x<t).
        "token_entropy" optional, shape (batch_size, sequence_length), per-token entropy
        for each position (present only if return_token_entropy=True).
    """
    outputs = model.forward(input_ids)
    logits = outputs.logits # shape(batch_size, seq_len, vocab_size)

    all_log_probs = F.log_softmax(logits.to(torch.float32), dim=-1)
    labels_expanded = labels.unsqueeze(-1)
    selected_log_probs = torch.gather(all_log_probs, dim=-1, index=labels_expanded).squeeze(-1)
    
    result = {"log_probs": selected_log_probs}
    if return_token_entropy:
        token_entropy = pertoken_entropy(logits.to(torch.float32))
        result["token_entropy"] = token_entropy

    return result

def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, normalize_constant: float, dim: int | None = None) -> torch.Tensor:
    """
    Sum over a dimension and normalize by a constant, considering only those elements where mask
    == 1.
    Args:
        tensor: torch.Tensor The tensor to sum and normalize.
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the sum.
        normalize_constant: float the constant to divide by for normalization.
        dim: int | None the dimension to sum along before normalization. If None, sum over all
        dimensions.
    Returns:
        torch.Tensor the normalized sum, where masked elements (mask == 0) don’t contribute to
        the sum.
    """
    masked_tensor = tensor * mask
    if dim is None:
        summation = torch.sum(masked_tensor)
    else:
        summation = torch.sum(masked_tensor, dim=dim)

    return summation / normalize_constant

def sft_microbatch_train_step(policy_log_probs: torch.Tensor, response_mask: torch.Tensor, gradient_accumulation_steps: int, normalize_constant: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.
    Args:
        policy_log_probs: (batch_size, sequence_length), per-token log-probabilities from the
                          SFT policy being trained.
        response_mask: (batch_size, sequence_length), 1 for response tokens, 0 for
                       prompt/padding.   
        gradient_accumulation_steps: Number of microbatches per optimizer step.
        normalize_constant:The constant by which to divide the sum. It is fine to leave this as 1.0.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
    """
    batch_size = policy_log_probs.shape[0]
    pertoken_loss = -policy_log_probs
    masked_loss = response_mask * pertoken_loss
    sum_loss = masked_normalize(masked_loss, response_mask, normalize_constant)
    mean_loss = sum_loss / batch_size
    actual_loss = mean_loss / gradient_accumulation_steps
    actual_loss.backward()
    log = {
        "loss": mean_loss.detach()
    }

    return actual_loss, log

def log_generations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    ground_truths: List[str],
    reward_fn: Callable[[str, str], Dict[str, float]],
    num_examples_to_log: int = 4,
    max_new_tokens: int = 1024,  # 抽查时可以设短一点节省时间，或者保持 1024
) -> Dict[str, float]:
    """
    使用 PyTorch 原生 generate 进行抽样评估，计算奖励和熵。
    """
    # 1. 采样
    n = min(num_examples_to_log, len(prompts))
    indices = random.sample(range(len(prompts)), n)
    sampled_prompts = [prompts[i] for i in indices]
    sampled_truths = [ground_truths[i] for i in indices]

    # 2. 准备环境
    device = model.device
    was_training = model.training
    model.eval()  # 切换到评估模式

    total_rewards = []
    format_rewards = []
    answer_rewards = []
    lengths = []
    entropies = []

    print(f"\n[Log Generation] Sampling {n} examples...")

    # 3. 逐条生成 (Batch 生成实现稍繁琐，逐条对 log 来说更安全)
    for i in range(n):
        prompt = sampled_prompts[i]
        truth = sampled_truths[i]

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        with torch.no_grad():
            outputs = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,      # 采样模式
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,  # 关键：请求输出分数以计算熵
            )
        assert type(outputs) == GenerateDecoderOnlyOutput

        # 解析生成的文本 (去掉 Prompt 部分)
        generated_ids = outputs.sequences[0][input_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # --- 计算指标 ---
        
        # A. 长度
        lengths.append(len(generated_ids))

        # B. 熵 (Entropy)
        # outputs.scores 是一个 tuple，长度为 generated_len
        # 每个元素是 (batch_size=1, vocab_size) 的 logits
        if outputs.scores:
            # 堆叠为 (Seq, Vocab)
            stacked_logits = torch.stack(outputs.scores).squeeze(1)
            # 计算 Prob 和 LogProb
            probs = torch.nn.functional.softmax(stacked_logits, dim=-1)
            log_probs = torch.nn.functional.log_softmax(stacked_logits, dim=-1)
            # 熵公式: -sum(p * log_p)
            token_entropies = -(probs * log_probs).sum(dim=-1)
            # 取平均
            avg_entropy = token_entropies.mean().item()
            entropies.append(avg_entropy)
        else:
            entropies.append(0.0)

        # C. 奖励 (Reward)
        # 假设 reward_fn 接受 (output, truth)
        metrics = reward_fn(generated_text, truth)
        total_rewards.append(metrics.get("reward", 0.0))
        format_rewards.append(metrics.get("format_reward", 0.0))
        answer_rewards.append(metrics.get("answer_reward", 0.0))

        # D. 打印第一条作为直观展示
        if i == 0:
            print("-" * 40)
            print(f" Prompt: {prompt[:50]}...")
            print(f"Generated: {generated_text[:100]}... (Len: {len(generated_ids)})")
            print(f"Truth: {truth[:50]}...")
            print(f"Metrics: {metrics} | Entropy: {entropies[-1]:.2f}")
            print("-" * 40)

    # 4. 恢复训练模式
    if was_training:
        model.train()

    # 5. 汇总统计
    stats = {
        "eval/reward": np.mean(total_rewards),
        "eval/format_reward": np.mean(format_rewards),
        "eval/answer_reward": np.mean(answer_rewards),
        "eval/length": np.mean(lengths),
        "eval/entropy": np.mean(entropies) if entropies else 0.0,
    }
    
    return stats

