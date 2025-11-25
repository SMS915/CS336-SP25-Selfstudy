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
    

    raw_rewards_list = [
        reward_fn(response, truth).get('reward', 0.0)
        for response, truth in zip(rollout_responses, repeated_ground_truths)
    ]

    raw_tensor = torch.tensor(raw_rewards_list, dtype=torch.float32).view(-1, group_size) # batch_size, group_size
    mean_tensor = raw_tensor.mean(dim=-1, keepdim=True) # (batch_size, 1)
    if not normalize_by_std:
        advantage = raw_tensor - mean_tensor
    else:
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
    Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either
    the raw reward or an already-normalized advantage.
    Args:
        raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1), scalar
        reward/advantage for each rollout response.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), logprobs for
        each token.
    Returns:
        torch.Tensor: Shape (batch_size, sequence_length), the per-token policy-gradient loss (to
        be aggregated across the batch and sequence dimensions in the training loop)
    """
    return -raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(advantages: torch.Tensor,
                           policy_log_probs: torch.Tensor,
                           old_log_probs: torch.Tensor,
                           cliprange: float) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    ratio = torch.exp(policy_log_probs - old_log_probs.detach())
    part1 = ratio * advantages
    part2 = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) * advantages
    clipped_object = torch.min(part1, part2)
    loss = -clipped_object
    with torch.no_grad():
        clipped_mask = (ratio > 1 + cliprange) | (ratio < 1 - cliprange)
        clipped_ratio = clipped_mask.float().mean()
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
    metadata = {}
    if loss_type == 'no_baseline':
        assert raw_rewards is not None
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
    elif loss_type == 'reinforce_with_baseline':
        assert advantages is not None
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
    else:
        assert advantages is not None and old_log_probs is not None and cliprange is not None
        loss, data = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
        for k, v in data.items():
            metadata[k] = v
        
    return loss, metadata

def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    masked_tensor = tensor * mask
    if dim is None:
        valid_count = mask.sum()
        valid_sum = masked_tensor.sum() # 注意平均的长度也要取mask后的有效长度
    else:
        valid_count = mask.sum(dim=dim)
        valid_sum = masked_tensor.sum(dim = dim)
    return valid_sum / valid_count

def grpo_microbatch_train_step(policy_log_probs: torch.Tensor,
                               response_mask: torch.Tensor,
                               gradient_accumulation_steps: int,
                               loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
                               raw_rewards: torch.Tensor | None = None,
                               advantages: torch.Tensor | None = None,
                               old_log_probs: torch.Tensor | None = None,
                               cliprange: float | None = None):
    batch_size = policy_log_probs.shape[0]
    step_loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    perexample_loss = masked_mean(step_loss, response_mask, dim=-1)
    mean_loss = perexample_loss.mean()
    actual_loss = mean_loss / gradient_accumulation_steps
    actual_loss.backward()
    metadata['loss'] = actual_loss.detach()
    
    return actual_loss, metadata

        
