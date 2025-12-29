import torch
from math import cos, pi
from jaxtyping import Float
from typing import Literal, Optional, Iterable


def Softmax(x: torch.Tensor, i: int = 0) -> torch.Tensor:
    """
    Softmax函数的手动实现，利用减去指定维度最大值的技巧保证数值稳定性。
    对输入张量对应维度上减去max(x)，由于softmax的平移不变性，变换前后相当于分子分母上同处以max(x), 结果相等
    但是避免了 exp(x) 在 x 为较大正数时可能导致的上溢问题，同时也减小了所有值都为较大负数时可能导致的下溢问题。
    由于softmax满足最大熵原理，没有引入任何无证据的先验信息，从而作出了最无偏的假设，其是最公平的概率分布选择。
    Args:
        x (torch.Tensor): 待归一化的输入张量。
        i (int): 指定的归一化维度，默认为0。

    Returns:
        torch.Tensor: 归一化后的张量
    """

    # x - max(x)
    scaled_x = x - torch.max(x, dim=i, keepdim=True)[0]
    return torch.exp(scaled_x) / torch.sum(torch.exp(scaled_x), dim=i, keepdim=True)

def cross_entropy_loss(inputs: Float[torch.Tensor, " batch_size seq_len vocab_size"], target: Float[torch.Tensor, " batch_size seq_len"]) -> torch.Tensor:
    """
    手动实现的交叉熵损失函数，计算目标分布与预测分布的交叉熵，采用了Log_Sum_Exp技巧，以提升数值稳定性
    标准的交叉熵损失计算涉及对 Softmax 的输出取对数，即 -log(softmax(z_i))，其中 z_i 是正确类别的logit, z_j是对所有类别的原始输出的logits
    Loss = -log(exp(z_i) / sum_j(exp(z_j)))
         = -(log(exp(z_i)) - log(sum_j(exp(z_j))))
         = log(sum_j(exp(z_j))) - z_i
    直接按该公式计算log项存在数值风险，z_j大时，可能出现数值溢出，z_j很小时，可能出现数值下溢，导致log(0) 后出现-inf
    为解决该问题，LogSumExp做法利用平移不变性，设 m = max(z_j)
    log(sum_j(exp(z_j)) = log(sum_j(exp(z_j + m - m)))
                        = log(sum_j(exp(z_j - m) * exp(m)))
                        = log(exp(m)) + log(sum_j(exp(z_j - m)))
                        = m + log(sum_j(exp(z_j - m)))

    所以           loss = m + log(sum_j(exp(z_j - m))) - z_i

    Args:
        inputs (torch.Tensor): 模型的预测原始输出logits, 形状为(batch_size, seq_len, vocab_size)
        target (torch.Tensor): 每个位置的真实标签索引，形状为(batch_size, seq_len)

    Returns:
        torch.Tensor: 一个标量张量，表示该批次所有位置损失的均值
    """

    # 第一项，m
    max_logit = torch.max(inputs, dim = -1, keepdim = True)[0]

    # 第二项，log(sum(exp(inputs - m)))
    log_sum_exp = torch.log(torch.sum(torch.exp(inputs - max_logit), dim=-1, keepdim = True))

    # 第三项，z_i, gather取到以target为下标的所有对应logit值
    correct_log = torch.gather(inputs, -1, target.unsqueeze(-1))

    # 对所有位置损失平均
    return (max_logit + log_sum_exp - correct_log).mean()


def get_wsd_schedule(t: int, t_warm: int, t_cycle: int, t_max: int, lr_max: float, lr_min: float) -> Float:
    """
    结合了预热策略和衰减策略的学习率调度器，分为预热 (Warmup) ，稳定 (Stable) 和 衰减 (Decay) 三个阶段。
    Warmup 阶段学习率线性增长，有助于在训练初期稳定模型参数，防止因大学习率导致梯度爆炸/训练发散。
    Stable 阶段学习率维持在高位，有助于模型充分探索优化空间，找到良好的收敛区间
    Decay 阶段学习率快速线性衰减，有助于模型精确收敛到良好的局部最优解
    Args:
        t (int): 当前训练步数
        t_warm (int): warm_up结束点，在0 - t_warm 范围内，学习率从 0 线性增长到 lr_max。
        t_cycle (int): 学习率高位阶段结束点，在 t_warm - t_cycle 内，学习率保持在 lr_max。
        t_max (int): 衰减阶段的结束节点，在 t_cycle - t_max 内，学习率快速线性衰减到 lr_min。
        lr_max (float): 整个训练周期内的最大学习率。
        lr_min (float): 训练末期的学习率，部分决定了衰减阶段的陡峭程度。

    Returns:
        float: 当前步数下的学习率
    """
    # Warmup 阶段: 学习率线性增长
    if t < t_warm:
        return t * lr_max / t_warm

    # Stable 阶段: 学习率维持高位
    elif t < t_cycle:
        return lr_max

    # Decay 阶段: 学习率快速衰减到lr_min
    elif t < t_max:
        return lr_max - (t - t_cycle) * (lr_max - lr_min) / (t_max - t_cycle)

    # 保底分支：确保在 t >= t_max 后，调度器仍返回一个确定的值，避免程序异常。
    else:
        return lr_min

def get_cosine_schedule(t: int, t_warm: int, t_cycle: int, lr_max: float, lr_min: float) -> Float:
    """
    结合了预热策略和退火策略的学习率调度器，分为预热 (Warmup)，退火 (Anneal) 和周期后维持 (Post-cycle) 三个阶段。
    Warmup 阶段学习率线性增长，有助于在训练初期稳定模型参数，防止因大学习率导致梯度爆炸/训练发散。
    Anneal 阶段学习率余弦衰减，有助于模型更稳定地收敛到良好的局部最优解。
    余弦衰减相比于线性衰减，其特点是在下降初期和末期速率较平缓，而在中期速率较快。
    这种平滑的衰减被认为有助于优化器更稳定地收敛到良好的局部最优解。
    Args:
        t (int): 当前训练步数
        t_warm (int): warm_up结束点，在0 - t_warm 范围内，学习率从 0 线性增长到 lr_max。
        t_cycle (int): 余弦周期结束点，在 t_warm - t_cycle 内，学习率从 lr_max 按余弦曲线的右半周期下降到 lr_min
        lr_max (float): 整个训练周期内的最大学习率
        lr_min (float): 余弦退火末期的恒定学习率，在 t > t_cycle 后学习率将保持在该水平

    Returns:
        float: 当前步数下的学习率
    """

    # Warmup 阶段: 学习率线性增长
    if t < t_warm:
        return t * lr_max / t_warm

    # 周期后维持阶段: 学习率保持在最低水平
    elif t > t_cycle:
        return lr_min

    # Anneal 阶段: 学习率按余弦曲线下降
    else:
        return lr_min + (lr_max - lr_min) * (1 + cos((t - t_warm) * pi / (t_cycle - t_warm))) / 2

def get_lr_schedule(t: int, t_warm: int, t_cycle: int, lr_max: float, lr_min: float, choice: Literal['cosine', 'wsd'] = 'cosine', t_max: Optional[int] = None) -> Float:
    """
    学习率调度策略分发函数，包含余弦退火和预热-稳定-衰减两个策略
    Args:
        t (int): 当前训练步数
        t_warm (int): warm_up结束点。
        t_cycle (int): 调度周期的关键时间点。
            - 在 'cosine' 策略下, 代表余弦周期的结束步数。
            - 在 'wsd' 策略下, 代表稳定周期的结束步数。
        t_max (Optional[int]): 训练总步数，仅在 wsd 策略下必须给出。
        lr_max (float): 整个训练周期内的最大学习率。
        lr_min (float): 训练末期的学习率，部分决定了衰减阶段的陡峭程度。
        choice (Literal['cosine', 'wsd']): 学习率调度选项。

    Returns:
        float: 当前步数下指定策略给出的学习率
    """
    assert t_warm <= t_cycle
    assert lr_min <= lr_max

    if choice == 'cosine':
        return get_cosine_schedule(t, t_warm, t_cycle, lr_max, lr_min)
    elif choice == 'wsd':
        assert t_max is not None, "wsd策略需指定最大步数"
        assert t_max >= t_cycle and t_max >= t_warm
        return get_wsd_schedule(t, t_warm, t_cycle, t_max, lr_max, lr_min)
    else:
        raise ValueError(f"未知的学习率调度策略 {choice}")

def clip_gradient(parameters: Iterable[torch.nn.parameter], max_norm: float, eps: float = 1e-6):
    """
    按 L2 范数对一组参数的梯度进行裁剪 (Gradient Clipping by Norm)。

    该方法是解决梯度爆炸问题的关键技术，常见于训练 RNN 或深度 Transformer 等模型。
    其核心思想是，如果所有参数梯度的总 L2 范数超过了一个预设的阈值 `max_norm`，
    则对所有梯度进行等比缩放，使得其总范数恰好等于 `max_norm`。

    这个过程的关键在于，它只改变了梯度向量的大小，而保持了其方向不变，
    从而在防止数值不稳定的同时，不会破坏优化的基本方向。

    计算公式:
    total_norm = sqrt(sum_i(||grad_i||_2^2))
    if total_norm > max_norm:
        scale_factor = max_norm / total_norm
        grad_i = grad_i * scale_factor

    Args:
        parameters (Iterable[torch.nn.Parameter]): 一个包含模型参数的可迭代对象，
            通常通过 `model.parameters()` 获取。
        max_norm (float): 梯度的最大允许 L2 范数。
        eps (float): 一个极小的浮点数，用于在除法中防止分母为0，提升数值稳定性。

    Returns:
        torch.Tensor: 一个标量张量，表示裁剪前的梯度总 L2 范数, 用于监控训练过程。
    """
    total_norm = 0

    # 累加所有可学习参数的梯度范数
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm()
            total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5


    if total_norm > max_norm:
        # 等比缩放的缩放比例
        scale_factor = max_norm / (total_norm + eps)
        for p in parameters:
            if p.grad is not None:
                # 缩放梯度
                p.grad.data.mul_(scale_factor)

    # 返回原始梯度范数用于监控
    return total_norm