# cs336_basics/train_bpe.py
import argparse
import os
import time
import datetime
from tqdm import tqdm
from typing import Literal
from cs336_basics.bpe_fast import BPETokenizer
from cs336_basics.bpe_naive import train_bpe_run


def main():
    parser = argparse.ArgumentParser(description="Train a BPE Tokenizer")

    # 必需参数
    parser.add_argument("--input_path", type=str, required=True, help="Path to the training corpus (txt/bin)")
    parser.add_argument("--vocab_size", type=int, required=True, help="Target vocabulary size")
    parser.add_argument("--save_name", type=str, required=True, help="Prefix for the saved vocab/merges files")
    # 可选参数
    parser.add_argument("--bpe_version", type=str, default="fast",
                        help="Select a version of bpe for training, using fast as default")
    parser.add_argument("--special_tokens", nargs="+", default=["<|endoftext|>"], help="List of special tokens")

    args = parser.parse_args()

    print(f"--- BPE Training Configuration ---")
    print(f"Input: {args.input_path}")
    print(f"Vocab Size: {args.vocab_size}")
    print(f"Special Tokens: {args.special_tokens}")
    print(f"BPE version: {args.bpe_version}")
    print(f"Save Prefix: {args.save_name}")
    print(f"--------------------------------")

    version: Literal["fast", "naive"] = args.bpe_version
    # 实例化并训练训练器
    if version == 'fast':
        trainer = BPETokenizer.train(
            input_path=args.input_path,
            vocab_size=args.vocab_size,
            special_tokens=args.special_tokens
        )
        vocab, merges = trainer._vocab, trainer._merges
        print("实际训练出的vocab size: ", len(vocab))
        print("实际训练出的merges size: ", len(merges))

        trainer.save(args.save_name)

    else:
        vocab, merges = train_bpe_run(input_path=args.input_path,
                                      vocab_size=args.vocab_size,
                                      special_tokens=args.special_tokens)

        print("实际训练出的vocab size: ", len(vocab))
        print("实际训练出的merges size: ", len(merges))


if __name__ == "__main__":
    main()