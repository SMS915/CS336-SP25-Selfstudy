import json
import os
import argparse
import numpy as np
from tqdm import tqdm

# 引入你的 Tokenizer
try:
    from cs336_basics.BPE.bpe_fast import BPETokenizer
except ImportError:
    raise ImportError("请确保 bpe_fast.py 存在且包含 BPETokenizer 类")


INPUT_FILE_PATH = "data/c4/c4_valid_full.jsonl"  # 支持 .jsonl 或解压后的文件
# 输出文件路径
OUTPUT_FILE_PATH = "c4_validation.bin"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=str, required=True, help="vocab.json路径")
    parser.add_argument("--merges", type=str, required=True, help="merges.txt路径")
    # 如果你想通过命令行覆盖上面的路径，可以使用这两个参数
    parser.add_argument("--input", type=str, default=INPUT_FILE_PATH, help="本地JSONL文件路径")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE_PATH, help="输出Bin文件路径")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误: 找不到输入文件: {args.input}")
        return

    eot_token = "<|endoftext|>"
    print(f"Loading tokenizer from {args.vocab}...")
    tokenizer = BPETokenizer.from_files(
        args.vocab, 
        args.merges, 
        special_tokens=[eot_token]
    )

    eot_encoded = tokenizer.encode(eot_token)
    assert len(eot_encoded) == 1, "Error: <|endoftext|> 被分词异常"
    eot_id = eot_encoded[0]
    print(f"EOT ID: {eot_id}")

    vocab_size = len(tokenizer.get_vocab())
    dtype = np.uint16 if vocab_size < 65535 else np.uint32
    print(f"Vocab size: {vocab_size}, dtype: {dtype}")

    buffer = []
    BUFFER_SIZE = 1_000_000 # 100万token写一次
    total_tokens = 0
    doc_count = 0

    print(f"Processing {args.input} -> {args.output} ...")
    
    # 自动处理 .gz 压缩文件或普通文本文件
    import gzip
    if args.input.endswith(".gz"):
        file_opener = gzip.open
        mode = 'rt' # 文本模式读取
    else:
        file_opener = open
        mode = 'r'

    with file_opener(args.input, mode, encoding='utf-8') as f_in, open(args.output, "wb") as f_out:
        pbar = tqdm(f_in, desc="Tokenizing", unit="doc")
        
        for line in pbar:
            if not line.strip(): continue
            
            try:
                data = json.loads(line)
                text = data.get('text', '')
                if not text: continue
                tokens = tokenizer.encode(text)
                tokens.append(eot_id)
                buffer.extend(tokens)
                doc_count += 1
                
                # 写入
                if len(buffer) >= BUFFER_SIZE:
                    arr = np.array(buffer, dtype=dtype)
                    f_out.write(arr.tobytes())
                    total_tokens += len(buffer)
                    buffer = []
                    
            except json.JSONDecodeError:
                print(f"跳过损坏的 JSON 行: {doc_count}")
                continue

        # 写入剩余
        if buffer:
            arr = np.array(buffer, dtype=dtype)
            f_out.write(arr.tobytes())
            total_tokens += len(buffer)

    print(f"\n完成！")
    print(f"文档总数: {doc_count}")
    print(f"Token总数: {total_tokens}")

if __name__ == "__main__":
    main()