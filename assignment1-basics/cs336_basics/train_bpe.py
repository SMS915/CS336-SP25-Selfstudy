# cs336_basics/train_bpe.py
import argparse
import os
import time
import datetime
from tqdm import tqdm
from cs336_basics.FastBPE import BPETokenizer


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

    # 实例化并训练训练器
    trainer = BPETokenizer.train(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens
    )

    # 获取vocab和merge，如果需要
    # vocab, merge = trainer._vocab, trainer._merges

    # 保存结果
    if hasattr(trainer, 'save'):
        trainer.save(args.save_name)
    else:
        print("警告: Trainer 没有实现 train 方法. 请先实现或者在调用train()时自动保存.")

if __name__ == "__main__":
    main()