from typing import Iterable, Callable, Any, Tuple
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float, betas: Tuple[float], eps: float, weight_decay: float) -> None:
        """
        初始化 AdamW 优化器。

        Args:
            params (iterable): 模型中需要优化的参数。
            lr (float): 学习率 α。
            beta1: 控制一阶矩估计的衰减率。
            beta2: 控制二阶矩估计的衰减率.
            eps (float): 为数值稳定性添加到分母的一个小常数 ε。
            weight_decay (float): 权重衰减系数 λ。
        """
        beta1, beta2 = betas[0], betas[1]
        if not lr >= 0.0:
            raise ValueError(f"无效学习率: {lr}")
        if not 0.0 <= beta1 <= 1.0:
            raise ValueError(f"无效的beta1值: {beta1}")
        if not 0.0 <= beta2 <= 1.0:
            raise ValueError(f"无效的beta2值: {beta2}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"无效的权重衰减值: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Callable[..., Any] = None) -> Any:
        loss = None if closure is None else closure()
        # 遍历参数组
        for group in self.param_groups:
            # 提取当前参数组的超参数
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            # 遍历参数组中的每个参数
            for p in group['params']:
                if p.grad is None:
                    continue

                # 状态初始化
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['v_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                m_avg, v_avg = state['m_avg'], state['v_avg']
                state['step'] += 1
                t = state['step']

                # Adam 更新逻辑
                grad = p.grad
                m_avg.mul_(beta1).add_(grad, alpha=1 - beta1)           # 更新一阶矩估计 m_t
                v_avg.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) # 更新二阶矩估计 v_t

                # 偏差修正
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                m_hat = m_avg / bias_correction1
                v_hat = v_avg / bias_correction2

                # 参数更新
                denom = v_hat.sqrt().add_(eps)
                p.data.addcdiv_(-lr, m_hat, denom)

                # 权重衰减解耦
                if weight_decay != 0.0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss



