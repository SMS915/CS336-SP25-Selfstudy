import torch
import argparse
import yaml
import wandb
import os
import math
import numpy as np

from cs336_basics.utils import *
from cs336_basics.optimizer import AdamW
from cs336_basics.checkpointing import *
from cs336_basics.model import *
# from cs336_basics.data import DataLoader
from cs336_basics.fast_data import create_dataloader 

def main():
    parser = argparse.ArgumentParser(description="训练大语言模型的脚本")
    parser.add_argument('--config', type=str, required=True, help='YAML格式的配置文件路径')

    # 预留几个命令行参数
    parser.add_argument('--max_learning_rate', type=float, default=None, help='覆盖配置文件中的学习率')
    parser.add_argument('--batch_size', type=int, default=None, help='覆盖配置文件中的批次大小')
    parser.add_argument('--max_steps', type=int, default=None, help='覆盖配置文件中的最大训练步数')

    args = parser.parse_args()

    # 加载配置文件
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, value in vars(args).items():
        if key != 'config' and value is not None:
            print(f"从命令行覆盖配置: {key} = {value}")
            config[key] = value

    try:
        config['max_learning_rate'] = float(config['max_learning_rate'])
        config['min_learning_rate'] = float(config['min_learning_rate'])
        config['weight_decay'] = float(config['weight_decay'])
        config['beta1'] = float(config['beta1'])
        config['beta2'] = float(config['beta2'])
        config['eps'] = float(config['eps'])
        config['max_grad_norm'] = float(config['max_grad_norm'])
        
        config['batch_size'] = int(config['batch_size'])
        config['max_steps'] = int(config['max_steps'])
        config['eval_interval'] = int(config['eval_interval'])
        config['eval_steps'] = int(config['eval_steps'])
        config['checkpoint_interval'] = int(config['checkpoint_interval'])
        config['warmup_steps'] = int(config['warmup_steps'])
        config['cycle_steps'] = int(config['cycle_steps'])
    except (ValueError, TypeError, KeyError) as e:
        print(f"错误: 配置文件中的某个数值参数类型不正确或缺失: {e}")
        return

    # 设置设备
    device = torch.device(config.get('device', 'cpu'))
    torch.manual_seed(config.get('seed', 42))

    wandb.init(
        project=config.get('wandb_project', 'CS336-TransformerLM-Training'),
        config=config
    )

    # 初始化模型
    model = TransformerLM(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        num_layers=config['n_layers'],
        num_heads=config['n_heads'],
        d_ff = config['d_ff'],
        rope_theta=config['rope_theta'],
        post_norm=config.get('post_norm', False),
        no_norm = config.get('no_norm', False),
        gated_ffn=config.get('gated_ffn', True),
        activation=config.get('activation', 'silu'),
        tie_weights=config.get('Weight_Tying', False),
        num_kv_heads=config.get('n_kv_heads', None),
        pos_emb_type=config.get('pos_emb_type', 'rope'),
        layer_norm=config.get('layer_norm', False),
        bias=config.get('bias', False),
        gated_attn=config.get('gated_attn', False)
    ).to(device)
    model.count_params()
    # print(f"模型总参数量: {model.count_params()}")


    betas = (config.get('beta1', 0.9), config.get('beta2', 0.95))
    print(type(config['max_learning_rate']))
    
    optimizer = AdamW(
        model.parameters(),
        lr=config.get('max_learning_rate', 3e-4),
        weight_decay=config.get('weight_decay', 0.01),
        betas=betas,
        eps = config.get('eps', 1e-8)
    )
    ckpt_dir = config.get('checkpoint_path', 'checkpoints/')
    amp_enabled = (device.type == 'cuda')
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using AMP with dtype: {amp_dtype}")
    scaler = torch.amp.GradScaler(enabled=amp_enabled)
    if config.get('load_ckpt', False) == True:
        ckpt_list = os.listdir(ckpt_dir) if os.path.exists(ckpt_dir) else []
        if ckpt_list:
            ckpt_path = get_latest_checkpoint(ckpt_dir)
            checkpoint = torch.load(ckpt_path)
            if 'scaler_state_dict' in checkpoint:
                start_step = load_amp_checkpoint(ckpt_path, model, optimizer, scaler)
            else :
                start_step = load_checkpoint(ckpt_path, model, optimizer)
            print(f"从检查点 {ckpt_path} 恢复，起始迭代次数: {start_step}")
        else:
            start_step = 0
            print("未找到检查点，开始新的训练")
    else:
        start_step = 0

    print("尝试编译模型")
    model = torch.compile(model, mode='default')
    # start_step = 0

        # 初始化数据加载器
    # train_loader = DataLoader(
    #     dataset_path=config['train_data_path'],
    #     context_length=config['context_length'],
    #     batch_size=config['batch_size'],
    #     seed=config.get('seed', 42),
    #     device=device.type,
    #     token_dtype=np.dtype(config.get('token_dtype', 'uint16'))
    # )
    train_loader = create_dataloader(config, is_train=True)

    # val_loader = DataLoader(
    #     dataset_path=config['val_data_path'],
    #     context_length=config['context_length'],
    #     batch_size=config['batch_size'],
    #     seed=config.get('seed', 42),
    #     device=device.type,
    #     token_dtype=np.dtype(config.get('token_dtype', 'uint16'))
    # )
    val_loader = create_dataloader(config, is_train=False)


    # current_epoch = 0
    # train_iter = train_loader.__iter__()
    # for step in range(start_step, config['max_steps']):
    #     new_lr = get_lr_schedule(t=step, t_warm=config['warmup_steps'], t_cycle=config['cycle_steps'], lr_max=config['max_learning_rate'], lr_min=config['min_learning_rate'])
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = new_lr
    #     epoch_for_dataloader = step // config['steps_per_epoch']
    #     if epoch_for_dataloader != current_epoch:
    #         current_epoch = epoch_for_dataloader
    #         train_iter = train_loader.__iter__()
    #     with torch.amp.autocast(device_type='cuda', enabled=amp_enabled, dtype=amp_dtype):
    #         inputs, targets = next(train_iter)
    #         logits = model(inputs, token_positions = None)  # batch_size seq_len vocab_size
    #         loss = cross_entropy_loss(logits.view(-1, config['vocab_size']), targets.view(-1))
    #     # scaled_loss = loss * s
    #     optimizer.zero_grad()
    #     scaler.scale(loss).backward()  
    #     scaler.unscale_(optimizer)
    #     clip_gradient(model.parameters(), config.get('max_grad_norm', 1.0))
    #     # torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('max_grad_norm', 1.0))
    #     scaler.step(optimizer)
    #     scaler.update()

    #     wandb.log({'train/loss': loss.item(), 'learning_rate': optimizer.param_groups[0]['lr']}, step=step)
    #     if step % config.get('log_interval', 100) == 0:
    #         print(f"Step {step}: Train Loss = {loss.item():.4f}")

    #     if (step + 1) % config['eval_interval'] == 0:
    #         model.eval()
    #         val_loss = 0.0
    #         with torch.no_grad():
    #             val_iter = val_loader.__iter__()
    #             for _ in range(config['eval_steps']):
    #                 val_inputs, val_targets = next()
    #                 val_logits = model(val_inputs, token_positions = None)
    #                 val_loss += cross_entropy_loss(val_logits.view(-1, config['vocab_size']), val_targets.view(-1)).item()

    #             avg_val_loss = val_loss / config['eval_steps']
    #             perplexity = math.exp(avg_val_loss)
    #             wandb.log({'val/loss': avg_val_loss, 'val/perplexity': perplexity}, step=step)
    #         model.train()

    #         if (step + 1) % config['checkpoint_interval'] == 0:
    #             if not os.path.exists(ckpt_dir):
    #                 os.makedirs(ckpt_dir)
    #             ckpt_filename = f"ckpt_step_{step:07d}_loss_{val_loss:.4f}.pt"
    #             ckpt_path = os.path.join(ckpt_dir, ckpt_filename)
    #             if scaler is not None:
    #                 save_amp_checkpoint(model=model, optimizer=optimizer,scaler=scaler, iteration= step + 1, out=ckpt_path)
    #             else:
    #                 save_checkpoint(model=model, optimizer=optimizer, iteration= step + 1, out=ckpt_path)
    #             print(f"保存检查点: {ckpt_path}\n")
    grad_accum_steps = config.get('accumulate_size', 1)

    train_iter = iter(train_loader)

    for step in range(start_step, config['max_steps']):
        # 更新学习率
        new_lr = get_lr_schedule(
            t=step, 
            t_warm=config['warmup_steps'], 
            t_cycle=config['cycle_steps'], 
            lr_max=config['max_learning_rate'], 
            lr_min=config['min_learning_rate'],
            choice=config.get('lr_schedule', 'cosine'),
            t_max = config['max_steps']
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        optimizer.zero_grad()

        accum_loss = 0.0
        for micro_step in range(grad_accum_steps):
            try:
                inputs_cpu, targets_cpu = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs_cpu, targets_cpu = next(train_iter)

            # 异步移动数据
            inputs = inputs_cpu.to(device, dtype=torch.long, non_blocking=True)
            targets = targets_cpu.to(device, dtype=torch.long, non_blocking=True)

            # 前向传播
            with torch.amp.autocast(device_type='cuda', enabled=amp_enabled, dtype=amp_dtype):
                logits = model(inputs, token_positions=None)
                loss = cross_entropy_loss(logits.view(-1, config['vocab_size']), targets.view(-1))
                
                # Loss 归一化
                loss = loss / grad_accum_steps

            # 反向传播 (梯度会累加到 .grad 属性中)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 记录还原后的 Loss 用于日志
            accum_loss += loss.item() * grad_accum_steps 

        # 权重更新
        
        if scaler is not None:
            scaler.unscale_(optimizer)
            # 梯度裁剪
            curr_grad_norm = clip_gradient(model.parameters(), config.get('max_grad_norm', 1.0))
            scaler.step(optimizer)
            scaler.update()
        else:
            # 梯度裁剪
            curr_grad_norm =  clip_gradient(model.parameters(), config.get('max_grad_norm', 1.0))
            optimizer.step()

        # 计算平均 Loss
        avg_loss = accum_loss / grad_accum_steps

        # 日志记录
        wandb.log({'train/loss': avg_loss, 'learning_rate': new_lr, 'grad_norm': curr_grad_norm}, step=step)
        if step % config.get('log_interval', 100) == 0:
            print(f"Step {step}: Train Loss = {avg_loss:.4f}, grad_norm = {curr_grad_norm:.4f}")

        if (step + 1) % config['eval_interval'] == 0:
            model.eval()
            val_loss = 0.0
            
            # 每次评估时重置验证集迭代器
            val_iter = iter(val_loader)
            
            with torch.no_grad():
                for _ in range(config['eval_steps']):
                    try:
                        val_inputs_cpu, val_targets_cpu = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        val_inputs_cpu, val_targets_cpu = next(val_iter)
                    
                    # 对验证数据应用相同的传输优化
                    val_inputs = val_inputs_cpu.to(device, dtype=torch.long, non_blocking=True)
                    val_targets = val_targets_cpu.to(device, dtype=torch.long, non_blocking=True)
                    
                    with torch.amp.autocast(device_type='cuda', enabled=amp_enabled, dtype=amp_dtype):
                        val_logits = model(val_inputs, token_positions=None)
                        val_loss += cross_entropy_loss(val_logits.view(-1, config['vocab_size']), val_targets.view(-1)).item()

            avg_val_loss = val_loss / config['eval_steps']
            perplexity = math.exp(avg_val_loss)
            wandb.log({'val/loss': avg_val_loss, 'val/perplexity': perplexity}, step=step)
            model.train()
            if config.get('save_ckpt', False):
                if (step + 1) % config['checkpoint_interval'] == 0:
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
    
                    ckpt_filename = f"ckpt_step_{step+1:07d}_loss_{avg_val_loss:.4f}.pt"
                    ckpt_path = os.path.join(ckpt_dir, ckpt_filename)
                    
                    if scaler is not None:
                        save_amp_checkpoint(model=model, optimizer=optimizer, scaler=scaler, iteration=step + 1, out=ckpt_path)
                    else:
                        save_checkpoint(model=model, optimizer=optimizer, iteration=step + 1, out=ckpt_path)
                    print(f"保存检查点: {ckpt_path}\n")

    print("训练完成。")
if __name__ == "__main__":
    main()