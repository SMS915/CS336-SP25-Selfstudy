import torch
from .sft import get_response_log_probs
import torch.nn.functional as F
from typing import Callable, List, Dict, Literal


def compute_group_normalized_rewards(reward_fn: Callable, 
                                     rollout_responses: List[str], 
                                     repeated_ground_truths: List[str], 
                                     group_size: int, 
                                     advantage_eps: float,
                                     normalize_by_std: bool) -> tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    计算基于组归一化（Group Normalization）的奖励优势（Advantage）。

    该函数实现了 GRPO (Group Relative Policy Optimization) 的核心逻辑：
    对同一提示（Prompt）生成的多个回复（Group）进行打分，并计算每个回复相对于组内均值的优势。

    Args:
        reward_fn (Callable): 奖励函数，接受 (response, ground_truth) 并返回包含 'reward' 键的字典。
        rollout_responses (List[str]): 模型生成的回复列表，总长度应为 batch_size * group_size。
        repeated_ground_truths (List[str]): 对应的标准答案列表，长度与 rollout_responses 一致。
        group_size (int): 每个组包含的样本数量 (GRPO 中的 G)。
        advantage_eps (float): 防止除以零的小常数。
        normalize_by_std (bool): 是否使用标准差进行归一化。如果不使用，仅减去均值(Dr.GRPO)。

    Returns:
        tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
            - advantage (torch.Tensor): 展平后的优势张量，形状为 (total_samples,)。
            - raw_tensor (torch.Tensor): 展平后的原始奖励张量，形状为 (total_samples,)。
            - meta_data (Dict[str, float]): 包含最大、最小和平均奖励的统计信息。
    """

    raw_rewards_list = [
        reward_fn(response, truth).get('reward', 0.0)
        for response, truth in zip(rollout_responses, repeated_ground_truths)
    ]
    # 将奖励转换为Tensor, 并reshape到[batch_size, group_size]的形状
    raw_tensor = torch.tensor(raw_rewards_list, dtype=torch.float32).view(-1, group_size) # batch_size, group_size
    # 计算组内均值
    mean_tensor = raw_tensor.mean(dim=-1, keepdim=True) # (batch_size, 1)
    if not normalize_by_std:
        # 仅中心化，Dr.GRPO 做法
        advantage = raw_tensor - mean_tensor
    else:
        # 标准化，原始 GRPO 做法
        std_tensor = raw_tensor.std(dim = -1, keepdim = True)
        advantage = (raw_tensor - mean_tensor) / (std_tensor + advantage_eps)

    max_reward = raw_tensor.max().item()
    min_reward = raw_tensor.min().item()
    mean_reward = mean_tensor.mean().item()
    meta_data = {'mean_reward': mean_reward,
                 'max_reward': max_reward,
                 'min_reward': min_reward}
    
    return advantage.flatten(), raw_tensor.flatten(), meta_data

def compute_naive_policy_gradient_loss(raw_rewards_or_advantages: torch.Tensor,
                                       policy_log_probs: torch.Tensor,
                                       ) -> torch.Tensor:
    """
    计算朴素策略梯度损失（Policy Gradient Loss）。
    
    计算公式为: Loss = - Advantage * log(pi(action|state))

    Args:
        raw_rewards_or_advantages (torch.Tensor): 形状为 (batch_size, 1) 或可广播形状的标量。
            代表每个样本的奖励或优势。
        policy_log_probs (torch.Tensor): 形状为 (batch_size, sequence_length)。
            模型生成的每个 token 的对数概率。

    Returns:
        torch.Tensor: 形状为 (batch_size, sequence_length) 的 per-token 损失张量。
            注意：此处尚未进行 mask 处理或求和。
    """
    return -raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(advantages: torch.Tensor,
                           policy_log_probs: torch.Tensor,
                           old_log_probs: torch.Tensor,
                           cliprange: float) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    计算带有裁剪机制的 GRPO/PPO 损失。

    采用 PPO 风格的裁剪目标函数，防止策略更新步长过大。
    Loss = -min( ratio * A, clip(ratio, 1-eps, 1+eps) * A )

    Args:
        advantages (torch.Tensor): 优势函数张量。
        policy_log_probs (torch.Tensor): 当前策略模型的对数概率。
        old_log_probs (torch.Tensor): 参考策略（或旧策略）的对数概率，不参与梯度计算。
        cliprange (float): 裁剪范围 epsilon（例如 0.2）。

    Returns:
        tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            - loss (torch.Tensor): 计算后的 per-token 损失。
            - metadata (Dict): 包含裁剪比例、近似 KL 散度等监控指标。
    """
    # 计算重要性采样比率 (Importance Sampling Ratio): pi_new / pi_old
    ratio = torch.exp(policy_log_probs - old_log_probs.detach())
    # 未裁剪的目标部分
    part1 = ratio * advantages
    # 裁剪后的目标部分：将比率限制在 [1-clip, 1+clip] 范围内
    part2 = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) * advantages
    # 取两者中的较小值，确保更新保守
    clipped_object = torch.min(part1, part2)
     # 损失取负号，考虑到优化器是做最小化
    loss = -clipped_object

    with torch.no_grad():
        # 统计触发裁剪的 token 比例
        clipped_mask = (ratio > 1 + cliprange) | (ratio < 1 - cliprange)
        clipped_ratio = clipped_mask.float().mean()
        # 计算近似 KL 散度 http://joschu.net/blog/kl-approx.html
        log_ratio = policy_log_probs - old_log_probs
        approx_kl = (log_ratio ** 2).mean() * 0.5
    metadata = {
        "clip_mask": clipped_mask,
        "clip_ratio": clipped_ratio,
        "approx_kl": approx_kl,
    }
    return loss, metadata

def compute_policy_gradient_loss(policy_log_probs: torch.Tensor,
                                 loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
                                 raw_rewards: torch.Tensor | None = None,
                                 advantages: torch.Tensor | None = None,
                                 old_log_probs: torch.Tensor | None = None,
                                 cliprange: float | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    策略梯度损失计算的分发函数。

    根据 `loss_type` 选择具体的损失计算逻辑。

    Args:
        policy_log_probs (torch.Tensor): 当前策略的 log probs。
        loss_type (Literal): 损失类型：
            - 'no_baseline': 直接使用原始奖励 (Vanilla PG)。
            - 'reinforce_with_baseline': 使用优势函数 (REINFORCE)。
            - 'grpo_clip': 使用 GRPO/PPO 裁剪机制。
        raw_rewards (torch.Tensor | None): 原始奖励（用于 no_baseline）。
        advantages (torch.Tensor | None): 优势函数（用于 reinforce 或 grpo）。
        old_log_probs (torch.Tensor | None): 旧策略 log probs（用于 grpo_clip）。
        cliprange (float | None): 裁剪系数（用于 grpo_clip）。

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 损失张量和元数据字典。
    """
    metadata = {}
    if loss_type == 'no_baseline':
        # 必须提供原始奖励
        assert raw_rewards is not None
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
    elif loss_type == 'reinforce_with_baseline':
        # 必须提供优势函数
        assert advantages is not None
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
    else:
        # GRPO Clip 模式，需要优势、旧概率和裁剪范围
        assert advantages is not None and old_log_probs is not None and cliprange is not None
        loss, data = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
        for k, v in data.items():
            metadata[k] = v
        
    return loss, metadata

def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """
    计算掩码平均值（Masked Mean）。

    仅计算 mask 为 1（True）位置的元素平均值，忽略 padding 部分。

    Args:
        tensor (torch.Tensor): 输入张量。
        mask (torch.Tensor): 掩码张量，形状与 tensor 兼容。
        dim (int | None): 指定计算平均值的维度。如果为 None，则计算全局平均。

    Returns:
        torch.Tensor: 平均值张量。
    """
    masked_tensor = tensor * mask
    if dim is None:
        valid_count = mask.sum()
        valid_sum = masked_tensor.sum() # 注意平均的长度也要取mask后的有效长度
    else:
        valid_count = mask.sum(dim=dim)
        valid_sum = masked_tensor.sum(dim=dim)
    return valid_sum / valid_count

def grpo_microbatch_train_step(policy_log_probs: torch.Tensor,
                               response_mask: torch.Tensor,
                               gradient_accumulation_steps: int,
                               loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
                               raw_rewards: torch.Tensor | None = None,
                               advantages: torch.Tensor | None = None,
                               old_log_probs: torch.Tensor | None = None,
                               cliprange: float | None = None):
    
    """
    执行 GRPO 的单个微批次（Micro-batch）训练步骤。

    计算损失，处理掩码，进行反向传播。

    Args:
        policy_log_probs (torch.Tensor): 模型生成的对数概率。
        response_mask (torch.Tensor): 掩码，用于指示哪些 token 是生成的回复（loss 仅在回复部分计算）。
        gradient_accumulation_steps (int): 梯度累积步数，用于归一化 loss。
        loss_type (Literal): 损失函数类型。
        raw_rewards (torch.Tensor | None): 原始奖励。
        advantages (torch.Tensor | None): 优势函数。
        old_log_probs (torch.Tensor | None): 参考模型的对数概率。
        cliprange (float | None): 裁剪范围。

    Returns:
        tuple:
            - actual_loss (torch.Tensor): 用于反向传播的最终标量损失。
            - metadata (Dict): 包含损失数值和其他监控指标的字典。
    """

    batch_size = policy_log_probs.shape[0]
    # 计算每个 token 的策略梯度损失
    step_loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    # 应用掩码并计算每个样本的平均损失，只计算 response 部分的 loss
    perexample_loss = masked_mean(step_loss, response_mask, dim=-1)
    # 计算整个 batch 的平均损失
    mean_loss = perexample_loss.mean()
    # 根据梯度累积步数缩放损失
    actual_loss = mean_loss / gradient_accumulation_steps
    actual_loss.backward()
    # 记录用于显示的 detach 后的 loss
    metadata['loss'] = mean_loss.detach()
    
    return actual_loss, metadata

        
