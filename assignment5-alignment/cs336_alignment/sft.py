import torch
import random
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Callable
from vllm import LLM, SamplingParams
from .utils import pertoken_entropy, optim_pertoken_entropy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerateDecoderOnlyOutput

def get_response_log_probs(model: PreTrainedModel, input_ids:torch.Tensor, attention_masks: torch.Tensor, labels: torch.Tensor, return_token_entropy: bool = False) -> Dict[str, torch.Tensor]:
    """
    计算给定输入和标签的对数概率（Log Probabilities）。

    Args:
        model (PreTrainedModel): 用于推理的 HuggingFace 模型。
        input_ids (torch.Tensor): 形状 (batch_size, seq_len)，模型输入。
        attention_masks (torch.Tensor): 形状 (batch_size, seq_len)，注意力掩码。
        labels (torch.Tensor): 形状 (batch_size, seq_len)，目标标签。
        return_token_entropy (bool): 是否计算并返回每个 token 的熵。

    Returns:
        Dict[str, torch.Tensor]:
            - "log_probs": (batch_size, seq_len) 目标 token 的条件对数概率 log p(y|x)。
            - "token_entropy": (batch_size, seq_len) 每个位置分布的熵（可选）。
    """
    # 前向传播
    outputs = model.forward(input_ids=input_ids, attention_mask=attention_masks)
    logits = outputs.logits # shape(batch_size, seq_len, vocab_size)

    logits_fp32 = logits.to(torch.float32)


    # 计算全词表的 Log Softmax
    all_log_probs = F.log_softmax(logits_fp32, dim=-1)
    labels_expanded = labels.unsqueeze(-1)

    # 在 vocab 维度上取 labels 指定索引的值
    selected_log_probs = torch.gather(all_log_probs, dim=-1, index=labels_expanded).squeeze(-1)
    if not return_token_entropy:
        result = {"log_probs": selected_log_probs}
    else :
        # 计算整个分布的熵，用于监控模型的不确定性
        # 启用的话会产生较大的显存开销，可能需要降低micro_batch_size / inference_batch_size
        token_entropy = optim_pertoken_entropy(logits_fp32)
        result = {"log_probs": selected_log_probs,
                  "token_entropy": token_entropy}

    return result

def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, normalize_constant: float, dim: int | None = None) -> torch.Tensor:
    """
    计算掩码加权和并进行归一化。

    仅统计 mask == 1 的位置。

    Args:
        tensor (torch.Tensor): 输入张量。
        mask (torch.Tensor): 掩码张量，形状同输入。
        normalize_constant (float): 归一化常数（分母）。
        dim (int | None): 求和维度。若为 None，则对所有维度求和。

    Returns:
        torch.Tensor: 归一化后的结果。
    """
    masked_tensor = tensor * mask
    if dim is None:
        summation = torch.sum(masked_tensor)
    else:
        summation = torch.sum(masked_tensor, dim=dim)

    return summation / normalize_constant

def compute_generation_entropy(scores: tuple | None) -> float:
    """
    从 generate 返回的 scores 中计算平均熵。
    Args:
        scores (tuple | None): generate 函数返回的 output_scores。
    Returns:
        float: 生成序列的平均熵值。
    """
    if not scores:
        return 0.0
    
    # 堆叠并转精度: (seq_len, batch_size, vocab_size) -> (seq_len, vocab_size)
    stacked_logits = torch.stack(scores).squeeze(1).to(torch.float32)
    
    # 计算概率分布
    log_probs = F.log_softmax(stacked_logits, dim=-1)
    probs = torch.exp(log_probs) 
    
    # 计算熵: -sum(p * log_p)
    token_entropies = -torch.sum(probs * log_probs, dim=-1)

    # 处理可能的 NaN (如 log(0))
    token_entropies = torch.nan_to_num(token_entropies, nan=0.0)
    
    return token_entropies.mean().item()

def sft_microbatch_train_step(policy_log_probs: torch.Tensor, response_mask: torch.Tensor, gradient_accumulation_steps: int, normalize_constant: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    执行 SFT (Supervised Fine-Tuning) 的微批次训练步骤。
    计算负对数似然损失（NLL Loss），并仅在 response_mask 指示的区域进行反向传播。

    Args:
        policy_log_probs (torch.Tensor): (batch, seq_len) 目标 token 的对数概率。
        response_mask (torch.Tensor): (batch, seq_len) 掩码，1 表示回复部分，0 表示提示/填充。
        gradient_accumulation_steps (int): 梯度累积步数，用于缩放 Loss。
        normalize_constant (int): 归一化常数（通常为 1.0 或 batch 大小等）。

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            - actual_loss: 用于反向传播的 Loss 张量。
            - log: 包含用于记录的 detach 后的 Loss 字典。
    """
    batch_size = policy_log_probs.shape[0]
    pertoken_loss = -policy_log_probs
    # masked_loss = response_mask * pertoken_loss

    # 计算有效 token 的总 Loss，并按常数归一化
    loss_sum  = masked_normalize(pertoken_loss, response_mask, normalize_constant=normalize_constant,dim=None)

    # 根据 batch_size 和 梯度累积步数 进一步平均
    actual_loss = loss_sum / gradient_accumulation_steps / batch_size
    actual_loss.backward()

    # 计算有效 token 数量用于显示平均 per-token loss
    valid_tokens_count = response_mask.sum().detach()
    if valid_tokens_count == 0:
        valid_tokens_count = 1

    loss_for_log = loss_sum.detach() / valid_tokens_count
    log = {
        "loss": loss_for_log
    }

    return actual_loss, log

def log_generations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    ground_truths: List[str],
    reward_fn: Callable[[str, str], Dict[str, float]],
    num_examples_to_log: int = 4,
    max_new_tokens: int = 1024,  
) -> Dict[str, float]:
    """
    使用模型生成回复并记录评估指标。

    抽样部分样本，执行推理，计算奖励（Reward）和熵（Entropy），并打印示例。

    Args:
        model (PreTrainedModel): 待评估模型。
        tokenizer (PreTrainedTokenizerBase): 分词器。
        prompts (List[str]): 提示词列表。
        ground_truths (List[str]): 标准答案列表。
        reward_fn (Callable): 奖励计算函数。
        num_examples_to_log (int): 抽样数量。
        max_new_tokens (int): 生成最大长度。

    Returns:
        Dict[str, float]: 包含平均奖励、长度、熵等统计指标的字典。
    """
    # try:
    #     ans_end_id = tokenizer.convert_tokens_to_ids("</answer>")
    #     print(f"Eval Tokenizer </answer> ID: {ans_end_id}")
    #     if ans_end_id == tokenizer.unk_token_id:
    #         print("FATAL: 评估脚本用的 Tokenizer 不认识 </answer>")
    # except:
    #     print("FATAL: Tokenizer 出错")

    # 随机抽样
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
    tag_count = 0
    run_away_count = 0
    print(f"\n[Log Generation] Sampling {n} examples...")

    # 3. 逐条生成与评估
    for i in range(n):
        prompt = sampled_prompts[i]
        truth = sampled_truths[i]

        # Tokenize 输入
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        with torch.no_grad():
            # 执行生成
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
                output_scores=True,  # 请求输出分数以计算熵
            )
        assert isinstance(outputs, GenerateDecoderOnlyOutput)

        # 解析生成的文本 (去掉 Input Prompt 部分)
        generated_ids = outputs.sequences[0][input_len:]
        generated_text = tokenizer.decode(generated_ids)
        has_tag = '</answer>' in generated_text
        
        if has_tag:
            tag_count += 1
            if len(generated_ids) >= int(max_new_tokens * 0.95):
                run_away_count += 1
        # 记录长度
        lengths.append(len(generated_ids))
        
        # 计算并记录熵
        entropy = compute_generation_entropy(outputs.scores)
        entropies.append(entropy)
        # 记录奖励
        # 预处理文本格式以适配奖励函数
        metrics = reward_fn(generated_text, truth)
        total_rewards.append(metrics.get("reward", 0.0))
        format_rewards.append(metrics.get("format_reward", 0.0))
        answer_rewards.append(metrics.get("answer_reward", 0.0))


        # print(f"Tail IDs: {outputs.sequences[0][-20:].tolist()}")
        # print(f"Tail Decode: {tokenizer.decode(outputs.sequences[0][-20:], skip_special_tokens=False)}")
        # ans_end_id = tokenizer.convert_tokens_to_ids("</answer>")
        # if ans_end_id in generated_ids:
        #     print(f"模型确实生成了 ID {ans_end_id}, 是解码或Reward函数的问题。")
        # else:
        #     print(f"模型根本没生成 ID {ans_end_id}, 是训练的问题。")


        print("-" * 40)
        print(f" Prompt: {prompt[:50]}...")
        print(f"Generated: {generated_text[-100:]}... (Len: {len(generated_ids)})")
        print(f"Truth: {truth[-100:]}...")
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
        # "eval/tag_rate": tag_count / n,
        # "eval/runaway_rate": run_away_count / n
    }
    
    return stats


def log_generations_vllm(
        llm: LLM,  # 传入训练脚本中的 vLLM 实例
        tokenizer,
        prompts: List[str],
        ground_truths: List[str],
        reward_fn: Callable[[str, str, bool], Dict[str, float]],
        num_examples_to_log: int = 4,
        max_new_tokens: int = 1024,
        verify: bool = False
) -> Dict[str, float]:
    """
    使用 vLLM 加速生成并记录评估指标。
    """
    # 1. 随机抽样
    n = min(num_examples_to_log, len(prompts))
    indices = random.sample(range(len(prompts)), n)
    sampled_prompts = [prompts[i] for i in indices]
    sampled_truths = [ground_truths[i] for i in indices]

    # 2. 配置 vLLM 采样参数
    # 开启 logprobs 以便计算熵
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        logprobs=5  # 每个 token 返回前 5 个候选的 logprob 用于估算熵
    )

    print(f"\n[Log Generation vLLM] Batch inferencing {n} examples...")

    # 3. 批量生成
    # 注意：在调用此函数前，外部应已执行 make_zero_copy_sync(policy, llm)
    outputs = llm.generate(sampled_prompts, sampling_params, use_tqdm=False)

    total_rewards = []
    format_rewards = []
    answer_rewards = []
    lengths = []
    entropies = []

    # 4. 后处理与评估
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        truth = sampled_truths[i]

        # 记录 Token 长度
        token_ids = output.outputs[0].token_ids
        lengths.append(len(token_ids))

        # 5. 计算熵 (vLLM 风格)
        # vLLM 的 logprobs 结构与 HF 不同，这里计算生成序列的平均 token 熵
        step_entropies = []
        if output.outputs[0].logprobs:
            for logprob_dict in output.outputs[0].logprobs:
                # logprob_dict 是 {token_id: LogprobObj}
                # 熵 H = -sum(p * log(p))
                probs = np.exp([lp.logprob for lp in logprob_dict.values()])
                # 归一化（因为只取了 top-k）
                probs = probs / np.sum(probs)
                ent = -np.sum(probs * np.log(probs + 1e-10))
                step_entropies.append(ent)

        avg_ent = np.mean(step_entropies) if step_entropies else 0.0
        entropies.append(avg_ent)

        # 6. 计算奖励
        metrics = reward_fn(generated_text, truth, verify)
        total_rewards.append(metrics.get("reward", 0.0))
        format_rewards.append(metrics.get("format_reward", 0.0))
        answer_rewards.append(metrics.get("answer_reward", 0.0))

        # 打印部分示例
        if i < 2:  # 仅打印前两个示例节省日志空间
            print("-" * 40)
            print(f" Prompt: {prompt[:50]}...")
            print(f"Generated: {generated_text[-150:]} (Len: {len(token_ids)})")
            print(f"Truth: {truth} | Reward: {metrics.get('reward'):.2f}")
            print("-" * 40)

    # 7. 汇总统计
    stats = {
        "eval/reward": np.mean(total_rewards),
        "eval/format_reward": np.mean(format_rewards),
        "eval/answer_reward": np.mean(answer_rewards),
        "eval/length": np.mean(lengths),
        "eval/entropy": np.mean(entropies) if entropies else 0.0,
    }

    return stats

