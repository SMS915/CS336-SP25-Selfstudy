import os
import multiprocessing as mp
import numpy as np
import time
from bpe_fast import BPETokenizer, find_chunk_boundaries # 导入你的 FastBPE
from bpe_baseline import BPETokenizer as OriginalBPETokenizer

# --- 配置 ---
TOKENIZER_DIR = "BPE_File"
DATA_DIR = "data"
TOKEN_DTYPE = np.uint16
NUM_WORKERS = 14 # 留一个核给主进程和IO
CHUNK_MULTIPLIER = 4 # 任务块数量是核心数的倍数，让调度更灵活

def benchmark_worker(args):
    """
    Worker: 只负责 Encode 并返回 Token 数量，不返回巨大的数组以节省通信开销
    """
    filename, start, end, tokenizer = args
    
    try:
        with open(filename, 'rb') as f:
            f.seek(start)
            length = end - start
            if length <= 0:
                return 0
            bytes_data = f.read(length)
            
        # 解码
        text_chunk = bytes_data.decode('utf-8', errors='replace')
        
        # --- 核心差异点：调用原始 BPE 的 encode ---
        token_ids = tokenizer.encode(text_chunk)
        
        return len(token_ids)
        
    except Exception as e:
        print(f"\n[Error] {e}")
        return 0


def encode_worker(args):
    """
    Worker 进程执行的函数：读取文件片段 -> Encode -> 返回 Numpy 数组
    """
    filename, start, end, tokenizer, file_index = args
    
    try:
        with open(filename, 'rb') as f:
            f.seek(start)
            # 读取指定字节范围
            length = end - start
            if length <= 0:
                return np.array([], dtype=TOKEN_DTYPE)
            
            bytes_data = f.read(length)
            
        # 解码为字符串，忽略错误的 unicode 字符（防止切分点在多字节字符中间的极罕见情况）
        text_chunk = bytes_data.decode('utf-8', errors='replace')
        
        # 执行 BPE 编码
        token_ids = tokenizer.encode(text_chunk)
        
        return np.array(token_ids, dtype=TOKEN_DTYPE)
        
    except Exception as e:
        print(f"\n[Worker Error] 处理块 {start}-{end} 时出错: {e}")
        return np.array([], dtype=TOKEN_DTYPE)

# def main():
#     # --- 1. 加载分词器 ---
#     print(f"正在加载分词器 (PID: {os.getpid()})...")
#     # 注意：如果 merges.txt 有标题行，请加上 skip_first_row=True
#     tokenizer = BPETokenizer.from_files(
#         vocab_filepath=os.path.join(TOKENIZER_DIR, "gpt2_vocab.json"),
#         merges_filepath=os.path.join(TOKENIZER_DIR, "gpt2_merges.txt"),
#         special_tokens=["<|endoftext|>"]
#     )
#     print("分词器加载完成。")

#     # --- 2. 定义文件 ---
#     files_to_process = [
#         "owt_train.txt",
#         "owt_valid.txt",
#         "TinyStoriesV2-GPT4-train.txt",
#         "TinyStoriesV2-GPT4-valid.txt"
#     ]

#     # --- 3. 并行处理循环 ---
#     for filename in files_to_process:
#         input_path = os.path.join(DATA_DIR, filename)
#         output_path = os.path.join(DATA_DIR, filename.replace('.txt', '.bin'))

#         if not os.path.exists(input_path):
#             print(f"文件未找到，跳过: {input_path}")
#             continue

#         print(f"\n{'='*40}")
#         print(f"开始处理: {filename}")
#         t0 = time.time()

#         # A. 计算切分边界
#         # 我们希望把文件切分成很多小块，让多进程并行处理
#         # 使用 b'\n' 作为切分符，确保在行尾切断，解决单词被切断的问题
#         desired_chunks = NUM_WORKERS * CHUNK_MULTIPLIER
#         print(f"正在寻找文件切分边界 (目标: {desired_chunks} 块)...")
        
#         boundaries = find_chunk_boundaries(
#             input_path, 
#             desired_num_chunks=desired_chunks, 
#             split_special_token=b'\n' 
#         )
        
#         # 如果文件很小或者没有换行符，boundaries 可能很少，做一下安全检查
#         if len(boundaries) < 2:
#             boundaries = [0, os.path.getsize(input_path)]

#         print(f"文件被切分为 {len(boundaries)-1} 个任务块。")

#         # B. 准备任务参数
#         tasks = []
#         for i in range(len(boundaries) - 1):
#             start = boundaries[i]
#             end = boundaries[i+1]
#             # 将 tokenizer 传入 worker。
#             # 在 Linux 上利用 copy-on-write 很快，在 Win/Mac 上需要 pickle，BPETokenizer 支持 pickle。
#             tasks.append((input_path, start, end, tokenizer, i))

#         total_tokens = 0
        
#         # C. 启动多进程处理
#         # 使用 imap 保证结果有序返回 (order-preserving)
#         # 这样我们可以拿到第一个块的结果写入，再拿第二个... 保证二进制文件顺序正确
#         with mp.Pool(processes=NUM_WORKERS) as pool:
#             with open(output_path, 'wb') as f_out:
#                 # imap 返回的是一个迭代器
#                 for result_arr in pool.imap(encode_worker, tasks):
#                     if len(result_arr) > 0:
#                         f_out.write(result_arr.tobytes())
#                         total_tokens += len(result_arr)
                    
#                     # 简单的进度展示
#                     print(f"\r  - 已处理并写入 Token 数: {total_tokens:,}", end="", flush=True)

#         duration = time.time() - t0
#         print(f"\n完成！")
#         print(f"输出文件: {output_path}")
#         print(f"总耗时: {duration:.2f} 秒")
#         print(f"处理速度: {total_tokens / duration / 1000:.2f} k tokens/s")

def main():
    print(f"--- 原始 BPE 性能基准测试 (Processers: {NUM_WORKERS}) ---")

    # 1. 加载原始分词器
    print("正在加载原始 BPE 分词器...")
    t_load_start = time.time()
    tokenizer = OriginalBPETokenizer.from_files(
        vocab_filepath=os.path.join(TOKENIZER_DIR, "gpt2_vocab.json"),
        merges_filepath=os.path.join(TOKENIZER_DIR, "gpt2_merges.txt"),
        special_tokens=["<|endoftext|>"]
    )
    print(f"加载耗时: {time.time() - t_load_start:.2f}s")

    # 2. 指定只测试 TinyStories
    files_to_process = [
        "TinyStoriesV2-GPT4-train.txt",
        "TinyStoriesV2-GPT4-valid.txt"
    ]

    for filename in files_to_process:
        input_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(input_path):
            print(f"文件跳过 (不存在): {input_path}")
            continue

        print(f"\n开始测试文件: {filename}")
        file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        print(f"文件大小: {file_size_mb:.2f} MB")

        t0 = time.time()

        # A. 切分任务 (利用 FastBPE 的工具)
        desired_chunks = NUM_WORKERS * CHUNK_MULTIPLIER
        boundaries = find_chunk_boundaries(
            input_path, 
            desired_num_chunks=desired_chunks, 
            split_special_token=b'\n'
        )
        
        if len(boundaries) < 2:
            boundaries = [0, os.path.getsize(input_path)]

        tasks = []
        for i in range(len(boundaries) - 1):
            tasks.append((input_path, boundaries[i], boundaries[i+1], tokenizer))

        total_tokens = 0
        
        # B. 并行执行 (无写入 IO)
        with mp.Pool(processes=NUM_WORKERS) as pool:
            # 使用 imap_unordered 稍微快一点点，因为我们不在乎顺序，只在乎总数
            for count in pool.imap_unordered(benchmark_worker, tasks):
                total_tokens += count
                print(f"\r  - 已处理 Token: {total_tokens:,}", end="", flush=True)

        duration = time.time() - t0
        
        print(f"\n[完成] {filename}")
        print(f"  - 总耗时: {duration:.2f} 秒")
        print(f"  - 吞吐量: {total_tokens / duration / 1000:.2f} k tokens/s")



if __name__ == '__main__':
    mp.freeze_support()
    main()