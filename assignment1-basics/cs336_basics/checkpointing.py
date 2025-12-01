import os
import re
import typing
import torch
import torch.nn as nn

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration': iteration}, out)
    
def save_amp_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,scaler: torch.cuda.amp.GradScaler,
                    iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'iteration': iteration}, out)

def load_checkpoint(src: str | os.PathLike, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
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
