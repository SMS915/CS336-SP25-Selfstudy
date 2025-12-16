import torch
from typing import List
import torch.nn.functional as F
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def pertoken_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim = -1) # shape: batch_size, seq_len, vocab_size
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return torch.nan_to_num(entropy, nan=0.0)# batch_size, seq_len


def tokenize_prompt_and_output(prompt_strs: List[str], output_strs: List[str], tokenizer: PreTrainedTokenizerBase, max_length: int = 1024):
    """
    对提示词（Prompt）和输出（Output）进行分词、拼接、填充，并生成用于训练的掩码。

    该函数处理数据预处理的核心逻辑，包括：
    1. 清洗特定标签（如 <think>）。
    2. 将文本转换为 Token ID。
    3. 构建 Input IDs 和 Attention Mask。
    4. 构建 Response Mask（仅在计算 Loss 时考虑回复部分）。
    5. 执行 Padding（右填充）并进行移位（Shift）以适应因果语言模型训练。

    Args:
        prompt_strs (List[str]): 提示词字符串列表。
        output_strs (List[str]): 对应的回复/输出字符串列表。
        tokenizer (PreTrainedTokenizerBase): HuggingFace 分词器。
        max_length (int): 序列最大长度，超过此长度将被截断。

    Returns:
        Dict[str, torch.Tensor]: 包含模型输入和标签的字典：
            - "input_ids": (batch, seq_len) 模型输入 ID。
            - "attention_mask": (batch, seq_len) 注意力掩码。
            - "labels": (batch, seq_len) 训练标签（input_ids 向左移一位）。
            - "response_mask": (batch, seq_len) 回复掩码（仅 Output 部分为 1，用于 Loss 计算）。
    """
    batch_tokens = []
    batch_mask = []
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_c = prompt.strip()
        output_c = output.strip()
        # 处理 <think> 标签重复的问题
        clean_output = output
        if prompt_c.endswith('<think>') and output_c.startswith('<think>'):
            clean_output = output_c[7:].lstrip()

        # 分词，不自动添加特殊 token
        prompt_ids = tokenizer.encode(prompt, add_special_tokens = False)
        output_ids = tokenizer.encode(clean_output, add_special_tokens = False)

        token_ids = prompt_ids + output_ids
        # 构建掩码: Prompt 部分为 0，Output 部分为 1
        output_mask = [0] * len(prompt_ids) + [1] * len(output_ids)
        # 截断处理
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            output_mask = output_mask[:max_length]
            
        batch_tokens.append(torch.tensor(token_ids, dtype=torch.long))
        batch_mask.append(torch.tensor(output_mask, dtype=torch.long))
    # Padding (批处理填充)
    max_len = max(len(ids) for ids in batch_tokens)
    padded_input_ids = []
    padded_response_masks = []
    padded_attention_masks = []
    for input_ids, masks in zip(batch_tokens, batch_mask):
        pad_len = max_len - len(input_ids)

        # 执行右填充
        padded_input = F.pad(input=input_ids, pad=(0, pad_len), value=tokenizer.pad_token_id) # type: ignore # 对最后一个dim,左填充0个，右填充pad_len个
        pad_res_mask = F.pad(input=masks, pad=(0, pad_len), value=0)                          # mask 填充 0

        # 构建 Attention Mask: 有效内容为 1，Padding 部分为 0
        attn_mask = torch.cat([torch.ones(len(input_ids), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])

        padded_input_ids.append(padded_input)
        padded_response_masks.append(pad_res_mask)
        padded_attention_masks.append(attn_mask)

    # 堆叠成 Batch 张量
    batch_input_ids = torch.stack(padded_input_ids)
    batch_response_masks = torch.stack(padded_response_masks)
    batch_attention_masks = torch.stack(padded_attention_masks)

    # 移位处理
    # 输入: t_0, ..., t_{n - 1}
    shifted_inputs = batch_input_ids[:, :-1]
    shifted_attn_masks = batch_attention_masks[:, :-1]

    # 输出: t_1, ..., t_{n}
    labels = batch_input_ids[:, 1:]
    shifted_response_masks = batch_response_masks[:, 1:]

    return {
        "input_ids": shifted_inputs,
        "attention_mask": shifted_attn_masks,
        "labels": labels,
        "response_mask": shifted_response_masks
    }
