# checkpointing.py

import os
import re
import typing
import warnings
import torch

from collections import defaultdict
from typing import Optional, List, Tuple

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    load_scaler: bool
) -> bool:
    """
    保存训练检查点（Model, Optimizer, Iteration），
    对于直接文件保存情况，采用原子写入策略（先写临时文件再重命名），防止因系统故障导致检查点损坏。
    预期磁盘存储占用(Per Parameter)：
    - FP32 训练: ~12 Bytes (4 Bytes 模型权重 + 8 Bytes 优化器状态)
    - BF16 训练: ~10 Bytes (2 Bytes 模型权重 + 8 Bytes 优化器状态)
     (注: 即使模型为 BF16，AdamW 状态通常仍需以 FP32 存储以保证精度)

    Args:
        model (torch.nn.Module):需要保存的模型实例，仅保存 state_dict 以实现解耦。
        optimizer (torch.optim.Optimizer): 需要保存的优化器实例。
            常规的 AdamW 含一阶(m)和二阶(v)动量，每个参数需 2 个 Float32，共 8 Bytes。
        iteration (int): 当前训练步数，用于恢复 LR Schedule 等状态
        out (str | os.PathLike | typing.BinaryIO | typing.IO[Bytes]): 输出路径或流对象。
            若传入二进制流，无法保证写入原子性。

    Returns:
        bool: 标记保存是否成功，二进制流默认返回成功
    """

    checkpoint_dict = {'model_state_dict': model.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(),
                       'iteration': iteration}

    if isinstance(out, (str, os.PathLike)):

        # 处理目标文件夹未创建的情况
        output_dir = os.path.dirname(out)
        os.makedirs(output_dir, exist_ok=True)

        # 采用原子化写入策略，以保证保存只有完全成功和失败两种状态，而不会出现损坏状态
        temp_path = str(out) + '.tmp'
        torch.save(checkpoint_dict, temp_path)
        os.rename(temp_path, out)

        return not os.path.exists(temp_path)

    else:
        # 写入二进制流，不保证结果的原子性
        torch.save(checkpoint_dict, out)
        return True

    
def save_amp_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    """
    保存混合精度(FP16)训练检查点（Model, Optimizer, Scaler, Iteration）

    相比标准检查点，额外保存了 GradScaler 状态，这对恢复 FP16 训练的数值稳定性至关重要。
    采用原子写入策略。

    预期磁盘存储占用(Per Parameter):
    FP16: ~ 10 Bytes (2 Bytes 模型权重 + 8 Bytes 优化器状态)

    Args:
        model (torch.nn.Module): 需要保存的模型实例，仅保存 state_dict 以解耦代码结构。
            在 FP16 混合精度下，模型权重通常存储为 Half (2 Bytes)。
        optimizer (torch.optim.Optimizer): 需要保存的优化器状态
        scaler (torch.cuda.amp.GradScaler): 梯度缩放器实例，保存了当前的缩放因子，增长/退避因子，计数器(均为单个数字) 等重要状态。
        iteration (int): 当前训练步数，用于恢复 LR Schedule 等状态
        out (str | os.PathLike | typing.BinaryIO | typing.IO[Bytes]): 输出路径或流对象。
            若传入二进制流，无法保证写入原子性。

    Returns:
        bool: 标记保存是否成功，二进制流默认返回成功

    """
    checkpoint_dict = {'model_state_dict': model.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(),
                       'scaler_state_dict': scaler.state_dict(),  # 多保存一个梯度缩放器状态
                       'iteration': iteration}

    if isinstance(out, (str, os.PathLike)):

        output_dir = os.path.dirname(out)
        os.makedirs(output_dir, exist_ok=True)

        # 同样原子化写入策略
        temp_path = str(out) + '.tmp'
        torch.save(checkpoint_dict, temp_path)
        os.rename(temp_path, out)

        return not os.path.exists(temp_path)
    else:
        # 写入二进制流，不保证结果的原子性
        torch.save(checkpoint_dict, out)
        return True


def load_checkpoint(
    path: str | os.PathLike,
    model: torch.nn.Module,
    model_compiled: Optional[bool] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None
) -> int:

    """
    从检查点加载模型、优化器和可选的梯度缩放器状态。

     主要功能包括：
     1. 通过 `map_location`支持设备无关加载
     2. 处理 `torch.compile()` 编译前后 `state_dict` 键名的兼容性问题。
     3. 对用户传入的 `model_compiled` 标志与模型的实际状态进行验证并发出警告。
     4. 支持选择性地加载优化器和梯度缩放器的状态。

    Args:
        path (str | os.PathLike): 检查点文件的路径。
        model (torch.nn.Module): 一个已初始化的模型实例。状态将被加载到此实例中。
        model_compiled (bool, optional):
            - None (默认): 自动检测模型是否被编译。
            - True: 强制认为模型期望带 '_orig_mod.' 前缀的权重。
            - False: 强制认为模型期望标准权重。
            用于在 DDP/FSDP 等复杂嵌套导致自动检测失效时，由用户接管控制权。
        optimizer (torch.optim.Optimizer):  可选。一个优化器实例，
            如果提供且检查点中存在，将加载其状态。
        scaler (torch.amp.GradScaler): 可选。一个梯度缩放器实例，
            如果提供且检查点中存在，将加载其状态。

    Returns:
        iteration (int): 继续训练的步数节点。
    """
    checkpoint = torch.load(path, map_location='cpu')

    ckpt_state_dict = checkpoint['model_state_dict']

    # 确认目标模型状态
    is_impl_compiled =  hasattr(model, "_orig_mod") or "OptimizedModule" in type(model).__name__

    # 用户强制指定模型的 compiled 状态
    if model_compiled is not None:
        expect_compiled = model_compiled
        # 如果用户强制指定与检测结果不一致，发出警告，但依然执行用户指令
        if expect_compiled != is_impl_compiled:
            warnings.warn(f"User provided `model_compiled={model_compiled}` but automatic detection found "
                          f"`is_model_compiled={is_impl_compiled}`. "
                          "Proceeding based on user input, but this may indicate an error.")
    else:
        expect_compiled = is_impl_compiled

    # 确认加载源状态
    is_ckpt_compiled = all(k.startswith('_orig_mod.') for k in ckpt_state_dict.keys() if 'weight' in k or 'bias' in k)

    sanitized_state_dict = {}
    prefix = '_orig_mod.'

    for k, v in ckpt_state_dict.items():

        # 权重是编译的，但模型(用户)断言不是编译的 -> 剥离前缀
        if is_ckpt_compiled and not expect_compiled:
            if k.startswith(prefix):
                sanitized_key = k[len(prefix):]
            else:
                sanitized_key = k

        # 权重是普通的，但是模型(用户)断言是编译的 -> 增加前缀
        elif not is_ckpt_compiled and expect_compiled:
            sanitized_key = prefix + k

        # 状态一致，直接传入
        else:
            sanitized_key = k

        sanitized_state_dict[sanitized_key] = v

    model.load_state_dict(sanitized_state_dict)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return checkpoint.get('iteration', 0)

def load_amp_checkpoint(
    path: str | os.PathLike,
    model: torch.nn.Module,
    optimizer: torch.optim.optimizer,
    scaler: torch.amp.GradScaler,
    model_compiled: bool
) -> int:
    """
    加载混合精度检查点，将模型 state_dict 和优化器状态加载回实例中，并健壮地处理compiled相关的逻辑
    Args:
        path (str | os.PathLike): 模型和优化器状态的保存地址
        model (torch.nn.Module): 一个已经初始化并移动到目标设备的模型实例
        optimizer (torch.optim.Optimizer): 一个已经初始化并移动到目标设备的优化器实例
        scaler:
        model_compiled (bool): 指示当前传入的model实例是否经过torch.compile()编译。
            函数将根据加载的权重的键名和目标模型的compiled情况，自动处理'_orig_mod'前缀，以处理相异compiled状态保存的检查点

    Returns:
        iteration (int): 继续训练的步数节点。
    """
    checkpoint = torch.load(path, map_location='cpu')
    
    state_dict = checkpoint['model_state_dict']
    
    # 前缀清洗
    is_model_compiled = hasattr(model, "_orig_mod") or "OptimizedModule" in type(model).__name__

    new_state_dict = {}
    
    for k, v in state_dict.items():
        if k.startswith("_orig_mod.") and not is_model_compiled:
            new_key = k[10:]

        elif not k.startswith("_orig_mod.") and is_model_compiled:
            new_key = "_orig_mod." + k

        else:
            new_key = k
            
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=True)
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
    return checkpoint['iteration']


CKPT_PATTERN = re.compile(
    r"ckpt_(?P<name>[\w-]+)_step_(?P<step>\d+)_loss_(?P<loss>\d+_\d+)\.pt"
)

def get_checkpoints_stats(ckpt_dir: str | os.PathLike, run_name: Optional[str] = None) -> List[Tuple[int, float, str]]:
    """
    遍历指定文件夹下所有文件，获取符合`ckpt_{name}_step_{step}_loss_{loss}.pt`格式的检查点。

    Args:
        ckpt_dir (str | os.PathLike): 检测模型的目标文件夹
        run_name (Optional[str]): 可选的运行名指定参数
            若指定了感兴趣的run_name，则只返回对应的 ckpt 的信息
            否则返回文件夹下所有检查点的信息。

    Returns:
        List[Tuple[int, float, str]]: 一个列表，其中每个元组包含了
            (步数, 损失值, 文件路径)。
    """
    checkpoint_stats: list[tuple[int, float, str]] = []

    if not os.path.exists(ckpt_dir):
        return checkpoint_stats


    for f in os.listdir(ckpt_dir):
        match = CKPT_PATTERN.match(f)

        if match:
            name = match.group('name')
            if run_name is None or run_name == name:
                step = int(match.group('step'))

                # 替换下划线为小数点，再转换为浮点数
                loss_str = match.group('loss').replace('_', '.', 1)
                loss = float(loss_str)

                full_path = os.path.join(ckpt_dir, f)
                checkpoint_stats.append((step, loss, full_path))

    return checkpoint_stats


def _get_extreme_checkpoint(
        ckpt_dir: str | os.PathLike,
        run_name: Optional[str],
        sort_key_index: int,
        reverse: bool
) -> Tuple[bool, str]:
    checkpoints = get_checkpoints_stats(ckpt_dir, run_name)
    if not checkpoints:
        return False, ""

    # 使用通用参数进行排序
    chosen_ckpt = sorted(checkpoints, key=lambda x: x[sort_key_index], reverse=reverse)[0]
    return True, chosen_ckpt[2]


def get_latest_checkpoint(ckpt_dir: str | os.PathLike, run_name: Optional[str] = None) -> Tuple[bool, str | os.PathLike]:
    """
    获取指定运行的最新检查点 (步数最大的) 并返回路径，用于恢复训练。

    Args:
        ckpt_dir (str | os.PathLike): 模型保存路径文件夹
        run_name (Optional[str]): 可选的运行名称参数，用于指定感兴趣的运行检查点

    Returns:
        bool: 指示变量，指示是否获取到一个合理路径
        str | os.PathLike: 获取到的路径名

    """
    return _get_extreme_checkpoint(ckpt_dir, run_name, sort_key_index=0, reverse=True)


def get_best_checkpoint(ckpt_dir: str | os.PathLike, run_name: Optional[str] = None) -> Tuple[bool, str | os.PathLike]:
    """
    获取指定运行的'最好'的检查点 (损失最小的)并返回路径，用于恢复训练。
    Args:
        ckpt_dir (str | os.PathLike): 模型保存路径文件夹
        run_name (Optional[str]): 可选的运行名称参数，用于指定感兴趣的运行检查点

    Returns:
        str | os.PathLike: 最佳检查点的路径
    """
    return _get_extreme_checkpoint(ckpt_dir, run_name, sort_key_index=1, reverse=False)
