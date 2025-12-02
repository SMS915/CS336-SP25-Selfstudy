# cs336_basics/train_bpe.py
import argparse
import os
import time
import datetime
from tqdm import tqdm
from cs336_basics.FastBPE import BPETrainer


def main():
    parser = argparse.ArgumentParser(description="Train a BPE Tokenizer")

    # 必需参数
    parser.add_argument("--input_path", type=str, required=True, help="Path to the training corpus (txt/bin)")
    parser.add_argument("--vocab_size", type=int, required=True, help="Target vocabulary size")
    parser.add_argument("--save_name", type=str, required=True, help="Prefix for the saved vocab/merges files")

    # 可选参数
    parser.add_argument("--special_tokens", nargs="+", default=["<|endoftext|>"], help="List of special tokens")

    args = parser.parse_args()

    print(f"--- BPE Training Configuration ---")
    print(f"Input: {args.input_path}")
    print(f"Vocab Size: {args.vocab_size}")
    print(f"Special Tokens: {args.special_tokens}")
    print(f"Save Prefix: {args.save_name}")
    print(f"--------------------------------")

    # 1. 实例化训练器
    trainer = BPETrainer()

    # 2. 开始训练 (你的代码中 train 方法返回了 vocab, merges)
    # 注意：确保你的 train 方法签名与这里匹配
    trainer.train(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens
    )

    # 3. 保存结果
    # 如果你已经在 train() 内部调用了 save()，这一步可以省略
    # 或者为了保险，在这里显式调用我们刚才写的 save()
    if hasattr(trainer, 'save'):
        trainer.save(args.save_name)
    else:
        print("Warning: Trainer has no 'save' method. Please implement it or rely on auto-save in train().")


if __name__ == "__main__":
    main()