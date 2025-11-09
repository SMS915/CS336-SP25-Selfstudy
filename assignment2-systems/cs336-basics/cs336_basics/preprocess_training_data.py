import os
import multiprocessing as mp
import numpy as np
from cs336_basics.BPE import BPETokenizer # 确保路径正确

# --- 配置 ---
TOKENIZER_DIR = "BPE_File"
DATA_DIR = "data"
# GPT-2的词汇表大小是50257，小于65536，所以可以用uint16
TOKEN_DTYPE = np.uint16
CHUNK_SIZE = 1000 * 1024 * 1024
NUM_WORKERS = mp.cpu_count()

# --- 1. 加载分词器 ---
print("正在加载分词器...")
tokenizer = BPETokenizer.from_files(
    vocab_filepath=os.path.join(TOKENIZER_DIR, "gpt2_vocab.json"),
    merges_filepath=os.path.join(TOKENIZER_DIR, "gpt2_merges.txt"),
    special_tokens=["<|endoftext|>"]
)
print(f"分词器加载成功，词汇表大小: {len(tokenizer._vocab)}")

# --- 2. 定义要处理的文件 ---
files_to_process = [
    "owt_train.txt",
    "owt_valid.txt",
    "TinyStoriesV2-GPT4-train.txt",
    "TinyStoriesV2-GPT4-valid.txt"
]

# --- 3. 循环处理 ---
for filename in files_to_process:
    input_path = os.path.join(DATA_DIR, filename)
    output_path = os.path.join(DATA_DIR, filename.replace('.txt', '.bin'))

    if not os.path.exists(input_path):
        print(f"文件未找到，跳过: {input_path}")
        continue

    total_tokens = 0
    print(f"正在流式处理文件: {input_path}...")
    # 以二进制追加模式 ('ab') 打开输出文件
    with open(output_path, 'wb') as f_out:
        # 以文本模式打开输入文件
        with open(input_path, 'r', encoding='utf-8') as f_in:
            while True:
                # a. 逐块读取
                text_chunk = f_in.read(CHUNK_SIZE)
                if not text_chunk:
                    # 文件读取完毕
                    break
                
                # b. 编码当前块
                token_ids = tokenizer.encode(text_chunk)
                
                # c. 转换为 Numpy 数组
                token_ids_arr = np.array(token_ids, dtype=TOKEN_DTYPE)
                
                # d. 将数组的二进制内容追加到输出文件
                f_out.write(token_ids_arr.tobytes())
                
                total_tokens += len(token_ids_arr)
                # 打印进度
                print(f"\r  - 已处理 token 数量: {total_tokens}", end="")
    
    print(f"\n处理完成！总共 {total_tokens} 个 token 已保存到: {output_path}")

print("\n所有文件预处理完成！")