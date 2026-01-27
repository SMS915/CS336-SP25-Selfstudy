import torch
import numpy as np
import torch.nn.functional as F
from typing import Callable, List, Dict

def compute_reward_proxy(
    policy_log_probs: torch.Tensor, 
    ref_log_probs: torch.Tensor, 
    beta: float
) -> torch.Tensor:
    """
    计算奖励代理 î (reward proxy)。
    
    Args:
        policy_log_probs (torch.Tensor): 当前策略的序列对数概率 (sum over sequence length)。
        ref_log_probs (torch.Tensor): 参考策略的序列对数概率。
        beta (float): 超参数 β。

    Returns:
        torch.Tensor: 奖励代理 î 张量。
    """
    # log_probs 应该是每个样本序列的总和，形状为 (total_samples,)
    return beta * (policy_log_probs - ref_log_probs)

def compute_pacs_score(reward_proxy: torch.Tensor, group_size: int):
    # print(f"proxy size: {reward_proxy.shape}")
    assert reward_proxy.dim() == 1 and reward_proxy.shape[-1] % group_size == 0

    proxy_grouped = reward_proxy.view(-1, group_size)
    mean_tensor = proxy_grouped.mean(dim = -1, keepdim=True)

    score_grouped = (group_size / (group_size - 1.0)) * (proxy_grouped - mean_tensor)

    return score_grouped.flatten()

def get_raw_rewards(
    reward_fn: Callable, 
    rollout_responses: List[str], 
    repeated_ground_truths: List[str], 
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    从一组模型的响应中根据奖励函数得到每个响应的 reward。

    Args:
        reward_fn (Callable): 奖励函数，接受 (response, ground_truth) 并返回包含 'reward' 键的字典。
        rollout_responses (List[str]): 模型生成的回复列表，总长度应为 batch_size * group_size。
        repeated_ground_truths (List[str]): 对应的标准答案列表，长度与 rollout_responses 一致。

    Returns:
        tuple[torch.Tensor, Dict[str, float]]:
            - raw_tensor (torch.Tensor): 展平后的原始奖励张量，形状为 (total_samples,)。
            - meta_data (Dict[str, float]): 包含最大、最小和平均奖励的统计信息。
    """
    raw_rewards_list = []
    format_rewards_list = []
    for response, truth in zip(rollout_responses, repeated_ground_truths):
        reward_metric = reward_fn(response, truth, False)
        raw_rewards_list.append(reward_metric.get('reward', 0.0))
        format_rewards_list.append(reward_metric.get('format_reward', 0.0))
        
    # 将奖励转换为Tensor
    raw_tensor = torch.tensor(raw_rewards_list, dtype=torch.float32)
    mean_reward = raw_tensor.mean().item()
    format_rate = np.mean(format_rewards_list)
    meta_data = {'mean_reward': mean_reward,
                 "format_rate": format_rate}
    
    return raw_tensor, meta_data


def pacs_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        group_size: int,
        beta: float,
        raw_rewards: torch.Tensor,
    ):
    
    """
    执行 PACS 的单个微批次（Micro-batch）训练步骤。

    计算奖励代理和RLOO风格的类优势得分，并进行反向传播。

    Args:
        policy_log_probs (torch.Tensor): 模型生成的对数概率。
        old_log_probs (torch.Tensor | None): 参考模型的对数概率。
        response_mask (torch.Tensor): 掩码，用于指示哪些 token 是生成的回复（loss 仅在回复部分计算）。
        gradient_accumulation_steps (int): 梯度累积步数，用于归一化 loss。
        
        raw_rewards (torch.Tensor | None): 原始奖励。

    Returns:
        tuple:
            - actual_loss (torch.Tensor): 用于反向传播的最终标量损失。
            - approx_kl (torch.Tensor): 对KL散度的k2估计。
    """
    masked_policy_log_probs = policy_log_probs * response_mask
    masked_old_log_probs = old_log_probs * response_mask
    with torch.no_grad():
        log_ratio =  (policy_log_probs - old_log_probs.detach()).to(torch.float32)
        approx_kl = (log_ratio ** 2).mean() * 0.5
        del log_ratio

    reward_proxy = compute_reward_proxy(masked_policy_log_probs, masked_old_log_probs, beta).sum(dim=-1)

    score = compute_pacs_score(reward_proxy, group_size)

    positive_count = raw_rewards.sum().item()

    negative_count = raw_rewards.numel() - positive_count
    
    if positive_count:
        pos_weight = negative_count / positive_count
        pos_weight_tensor = torch.tensor([pos_weight], device=score.device, dtype=score.dtype)
        bce_with_logits_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()

    loss = bce_with_logits_loss(score, raw_rewards)

    actual_loss = loss / gradient_accumulation_steps

    actual_loss.backward()

    actual_kl = approx_kl / gradient_accumulation_steps

    return actual_loss, actual_kl
