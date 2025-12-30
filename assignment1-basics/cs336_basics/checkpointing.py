import os
import re
import typing
import torch

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
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
        None.
    """

    checkpoint_dict = {'model_state_dict': model.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(),
                       'iteration': iteration}

    if isinstance(out, (str, os.PathLike)):
        # 采用原子化写入策略，以保证保存只有完全成功和失败两种状态，而不会出现损坏状态
        temp_path = str(out) + '.tmp'
        torch.save(checkpoint_dict, temp_path)
        os.rename(temp_path, out)
    else:
        # 写入二进制流，不保证结果的原子性
        torch.save(checkpoint_dict, out)
    
def save_amp_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,scaler: torch.cuda.amp.GradScaler,
                    iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    """
    保存混合精度(FP16)训练检查点（Model, Optimizer, Scaler, Iteration）

    相比标准检查点，额外保存了 GradScaler 状态，这对恢复 FP16 训练的数值稳定性至关重要。
    采用原子写入策略。

    预期磁盘存储占用(Per Parameter):
    FP16: ~ 10 Bytes (2 Bytes 模型权重 + 8 Bytes 优化器状态)

    Args:
        model (torch.nn.Module): 需要保存的模型实例，仅保存 state_dict 以解耦代码结构。
            在 FP16 混合精度下，模型权重通常存储为 Half (2 Bytes)。

        optimizer (torch.optim.Optimizer): 需要保存的优化器状态, 大小一般约为 num_params * 8 Bytes (一阶动量和二阶动量，分别 4 Bytes)。
        scaler (torch.cuda.amp.GradScaler): 梯度缩放器实例，保存了当前的缩放因子，增长/退避因子，计数器(均为单个数字) 等重要状态。
        iteration (int): 当前训练步数，用于恢复 LR Schedule 等状态
        out (str | os.PathLike | typing.BinaryIO | typing.IO[Bytes]): 输出路径或流对象。
            若传入二进制流，无法保证写入原子性。

    Returns:
        None.

    """
    checkpoint_dict = {'model_state_dict': model.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(),
                       'scaler_state_dict': scaler.state_dict(),  # 多保存一个梯度缩放器状态
                       'iteration': iteration}

    if isinstance(out, (str, os.PathLike)):
        # 同样原子化写入策略
        temp_path = str(out) + '.tmp'
        torch.save(checkpoint_dict, temp_path)
        os.rename(temp_path, out)
    else:
        # 写入二进制流，不保证结果的原子性
        torch.save(checkpoint_dict, out)

def load_checkpoint(src: str | os.PathLike, model: torch.nn.Module, optimizer: torch.optim.Optimizer, compiled: bool = False):
    """
    加载常规检查点，将
    Args:
        src:
        model:
        optimizer:
        compiled:

    Returns:

    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration

# def load_amp_checkpoint(src: str | os.PathLike, model: torch.nn.Module, optimizer: torch.optim.Optimizer,scaler: torch.cuda.amp.GradScaler):
#     checkpoint = torch.load(src)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     scaler.load_state_dict(checkpoint['scaler_state_dict'])
#     iteration = checkpoint['iteration']
#     return iteration

def load_amp_checkpoint(path, model, optimizer=None, scaler=None):
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

CKPT_PATTERN = re.compile(r"ckpt_step_(\d+)_loss_([\d\.]+)\.pt")

def get_checkpoints(ckpt_dir):
    """
    扫描目录，返回所有符合命名规范的检查点文件及其解析出的信息。
    返回列表格式: [(step, loss, path), ...]
    """
    if not os.path.exists(ckpt_dir):
        return []

    checkpoints = []
    for f in os.listdir(ckpt_dir):
        match = CKPT_PATTERN.match(f)
        if match:
            step = int(match.group(1))
            loss = float(match.group(2))
            full_path = os.path.join(ckpt_dir, f)
            checkpoints.append((step, loss, full_path))
    return checkpoints

def get_latest_checkpoint(ckpt_dir):
    """找到'最新'的检查点 (步数最大的) -> 用于恢复训练"""
    checkpoints = get_checkpoints(ckpt_dir)
    if not checkpoints:
        return None
    # 按 step (元组的第0个元素) 降序排列，取第一个
    latest_ckpt = sorted(checkpoints, key=lambda x: x[0], reverse=True)[0]
    return latest_ckpt[2] # 返回 path

def get_best_checkpoint(ckpt_dir):
    """找到'最好'的检查点 (损失最小的) -> 用于推理/发布"""
    checkpoints = get_checkpoints(ckpt_dir)
    if not checkpoints:
        return None
    # 按 loss (元组的第1个元素) 升序排列，取第一个
    best_ckpt = sorted(checkpoints, key=lambda x: x[1])[0]
    return best_ckpt[2] # 返回 path
