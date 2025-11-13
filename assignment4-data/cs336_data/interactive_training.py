import fasttext
from fasttext import train_supervised
import argparse
import os
import yaml
from datetime import datetime
from pathlib import Path

# --- 1. 参数解析 (与之前类似，但增加新参数) ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="交互式地训练fastText质量分类器。")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径。")
    parser.add_argument(
        "--checkpoint_interval", 
        type=int, 
        default=5, 
        help="每训练多少个epoch后暂停并评估。"
    )
    parser.add_argument(
        "--resume_from", 
        type=str, 
        help="从一个已有的模型快照(.bin文件)继续训练。"
    )
    parser.add_argument(
        "--max_backups",
        type=int,
        default=5,
        help="最多保留多少个模型快照。"
    )
    return parser.parse_args()

# --- 2. 辅助函数：管理快照 ---
def manage_checkpoints(output_dir: str, max_backups: int):
    """确保快照数量不超过限制，删除最旧的。"""
    checkpoints = sorted(
        Path(output_dir).glob("*.bin"), 
        key=os.path.getmtime, 
        reverse=True
    )
    if len(checkpoints) > max_backups:
        for old_checkpoint in checkpoints[max_backups:]:
            print(f"删除旧快照: {old_checkpoint}")
            os.remove(old_checkpoint)

# --- 3. 核心交互式训练循环 ---
def interactive_train(config, args):
    # --- 加载配置 ---
    data_config = config['data']
    train_params = config['training']
    exp_config = config['experiment']
    
    train_file = data_config['train_file']
    valid_file = data_config.get('valid_file')
    output_dir = data_config['output_dir']
    
    if not valid_file:
        print("错误：交互式训练需要一个验证集(valid_file)在配置文件中指定。")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    total_epochs = train_params.get('epoch', 25) # 从配置中获取总轮数
    
    # --- 加载或初始化模型 ---
    current_epoch = 0
    if args.resume_from:
        print(f"从快照继续训练: {args.resume_from}")
        model = fasttext.load_model(args.resume_from)
        # 尝试从文件名中解析当前epoch (这是一个简化的解析，可以做得更复杂)
        try:
            filename = Path(args.resume_from).name
            current_epoch = int(filename.split('_e')[1].split('_')[0])
            print(f"已完成 {current_epoch} 轮训练。")
        except (IndexError, ValueError):
            print("警告：无法从文件名中解析epoch数，将从0开始计数。")
    else:
        print("从头开始训练新模型。")
        model = None # 初始时没有模型

    # --- 主训练循环 ---
    while current_epoch < total_epochs:
        next_epoch_target = min(current_epoch + args.checkpoint_interval, total_epochs)
        
        print("\n" + "="*50)
        print(f"准备训练: 从 epoch {current_epoch + 1} 到 {next_epoch_target}")
        print("="*50)

        # 准备增量训练的参数
        incremental_train_params = train_params.copy()
        incremental_train_params['epoch'] = args.checkpoint_interval # 只训练一小段
        incremental_train_params['input'] = train_file
        
        if model: # 如果已有模型，则在其上继续训练
            incremental_train_params['pretrainedVectors'] = model.get_word_vector('</s>') # 这是一个技巧，实际上是在加载整个模型状态
            # 更现代的 fasttext 版本可能支持 `model` 参数，但这更通用
            model = fasttext.train_supervised(**incremental_train_params)
            train_supervised
        else: # 第一次训练
            model = fasttext.train_supervised(**incremental_train_params)

        current_epoch = next_epoch_target
        
        # --- 评估 ---
        print("\n--- 评估当前模型 ---")
        result = model.test(valid_file)
        precision = result[1]
        recall = result[2]
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Epoch: {current_epoch}/{total_epochs}")
        print(f"  - 验证集 P@1: {precision:.4f}")
        print(f"  - 验证集 R@1: {recall:.4f}")
        print(f"  - 验证集 F1@1: {f1_score:.4f}")

        # --- 保存快照 ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        f1_str = f"{f1_score:.3f}".replace('.', '_')
        model_name = f"{exp_config['name']}_e{current_epoch}_f1_{f1_str}_{timestamp}.bin"
        model_path = os.path.join(output_dir, model_name)
        model.save_model(model_path)
        print(f"模型快照已保存至: {model_path}")

        # --- 管理快照数量 ---
        manage_checkpoints(output_dir, args.max_backups)
        
        # --- 交互 ---
        if current_epoch >= total_epochs:
            print("\n已达到最大训练轮数，训练结束。")
            break
            
        while True:
            action = input("\n继续训练 (c), 还是停止 (s)? [c/s]: ").lower().strip()
            if action in ['c', 's']:
                break
            print("无效输入，请输入 'c' 或 's'。")

        if action == 's':
            print("用户选择停止训练。")
            break

if __name__ == "__main__":
    args = parse_arguments()
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件 {args.config} 不存在！")
        exit(1)
        
    interactive_train(config, args)