import torch
import yaml
import argparse
import os

from cs336_basics.model import TransformerLM
from cs336_basics.BPE import BPETokenizer
from cs336_basics.checkpointing import get_latest_checkpoint, get_best_checkpoint
from cs336_basics.GenerateText import generate_text

def main():
    # --- 1. 设置命令行参数 ---
    parser = argparse.ArgumentParser(description="使用训练好的模型生成文本。")
    parser.add_argument('--config', type=str, required=True, help='用于训练模型的YAML配置文件路径。')
    parser.add_argument('--prompt', type=str, default="Once upon a time", help='初始文本提示。')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='要生成的新token的数量。')
    parser.add_argument('--temperature', type=float, default=0.8, help='采样温度。')
    parser.add_argument('--top_k', type=int, default=200, help='Top-K采样中的k值。')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='检查点目录，如果未提供，则使用config文件中的路径。')
    
    args = parser.parse_args()

    # --- 2. 加载配置 ---
    # 我们需要配置文件来获取模型的超参数 (d_model, n_layers, etc.)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get('device', 'cpu'))

    # --- 3. 准备分词器 (Tokenizer) ---
    print("正在加载分词器...")
    try:
        tokenizer = BPETokenizer.from_files(
            vocab_filepath='/root/autodl-tmp/CS336/CS336-SP25-Selfstudy/assignment1-basics/BPE_File/gpt2_vocab.json', # 假设config中有这些路径
            merges_filepath='/root/autodl-tmp/CS336/CS336-SP25-Selfstudy/assignment1-basics/BPE_File/gpt2_merges.txt',
            special_tokens=["<|endoftext|>"]
        )
        print("分词器加载成功。")
    except Exception as e:
        print(f"加载分词器失败: {e}")
        return

    # --- 4. 准备模型 (Model) ---
    print("正在构建模型架构...")
    model = TransformerLM(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        num_layers=config['n_layers'],
        num_heads=config['n_heads'],
        d_ff=config.get('d_ff'), # d_ff可能是None
        rope_theta=config['rope_theta']
    ).to(device)
    print("模型构建完成。")

    # --- 5. 加载最佳检查点 (Best Checkpoint) ---
    ckpt_dir = args.checkpoint_dir or config.get('checkpoint_path', './checkpoints')
    
    # 使用我们之前定义的 get_best_checkpoint 辅助函数
    best_ckpt_path = get_latest_checkpoint(ckpt_dir)
    
    if best_ckpt_path:
        print(f"正在从最佳检查点加载权重: {best_ckpt_path}")
        try:
            # 加载检查点
            checkpoint = torch.load(best_ckpt_path, map_location=device)
            # 将权重加载到模型中
            model.load_state_dict(checkpoint['model_state_dict'])
            print("权重加载成功。")
        except Exception as e:
            print(f"加载检查点失败: {e}")
            return
    else:
        print(f"警告: 在目录 '{ckpt_dir}' 中未找到任何有效的检查点。")
        print("将使用随机初始化的模型进行生成，结果可能无意义。")

    # --- 6. 执行生成 (Execution) ---
    print("\n--- 开始生成文本 ---")
    print(f"提示: '{args.prompt}'")
    
    # 将所有准备好的组件传入生成函数
    generated_output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )

    print("\n--- 生成结果 ---")
    print(generated_output)
    print("\n------------------")


if __name__ == '__main__':
    main()