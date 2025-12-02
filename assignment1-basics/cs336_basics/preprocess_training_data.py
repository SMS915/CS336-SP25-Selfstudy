import os
import multiprocessing as mp
import numpy as np
import time
import argparse
from typing import List, Tuple
from tqdm import tqdm  # 导入 tqdm

# 尝试导入 FastBPE，兼容 user 提供的路径或当前目录
try:
    from FastBPE import BPETokenizer, find_chunk_boundaries
except ImportError:
    try:
        from cs336_basics.FastBPE import BPETokenizer, find_chunk_boundaries
    except ImportError:
        print("错误: 无法导入 FastBPE。请确保 'FastBPE.py' (或 cs336_basics.FastBPE) 包含 BPETokenizer 和 find_chunk_boundaries。")
        exit(1)

# --- 全局配置 ---
TOKEN_DTYPE = np.uint16  # 对于 GPT-2 级别的词表 (50257)，uint16 (0-65535) 足够且节省空间
CHUNK_MULTIPLIER = 4     # 任务块数量是核心数的倍数，让调度更灵活

def encode_worker(args):
    """
    Worker 进程执行的函数：
    1. 读取指定字节范围的文件片段
    2. 解码为字符串
    3. 执行 BPE Encode
    4. 返回 Numpy 数组
    """
    filename, start, end, tokenizer, chunk_id = args
    
    try:
        with open(filename, 'rb') as f:
            f.seek(start)
            length = end - start
            if length <= 0:
                return np.array([], dtype=TOKEN_DTYPE)
            
            bytes_data = f.read(length)
            
        # 解码为字符串，使用 'replace' 忽略错误的 unicode 字符（防止切分点在多字节字符中间的极罕见情况）
        text_chunk = bytes_data.decode('utf-8', errors='replace')
        
        # 执行 FastBPE 编码
        token_ids = tokenizer.encode(text_chunk)
        
        return np.array(token_ids, dtype=TOKEN_DTYPE)
        
    except Exception as e:
        print(f"\n[Worker Error] 处理块 {chunk_id} ({start}-{end}) 时出错: {e}")
        return np.array([], dtype=TOKEN_DTYPE)

def process_file(input_path: str, tokenizer: BPETokenizer, num_workers: int):
    """
    处理单个文件的完整流程：切分 -> 并行编码 -> 写入 .bin
    """
    if not os.path.exists(input_path):
        print(f"文件未找到，跳过: {input_path}")
        return

    file_size = os.path.getsize(input_path)
    output_path = os.path.splitext(input_path)[0] + '.bin'
    
    print(f"\n{'='*40}")
    print(f"开始处理: {input_path}")
    print(f"文件大小: {file_size / (1024 * 1024):.2f} MB")
    print(f"输出路径: {output_path}")
    
    t0 = time.time()

    # 1. 计算切分边界
    # 我们希望把文件切分成很多小块，让多进程并行处理
    # 使用 b'\n' 作为切分符，确保在行尾切断，防止单词被切断
    desired_chunks = num_workers * CHUNK_MULTIPLIER
    print(f"正在计算文件切分边界 (目标: {desired_chunks} 块)...")
    
    boundaries = find_chunk_boundaries(
        input_path, 
        desired_num_chunks=desired_chunks, 
        split_special_token=b'\n' 
    )
    
    # 边界检查
    if not boundaries:
        boundaries = [0, file_size]
    if boundaries[-1] != file_size:
        boundaries.append(file_size)
        
    num_tasks = len(boundaries) - 1
    print(f"文件被切分为 {num_tasks} 个任务块。")

    # 2. 准备任务参数
    tasks = []
    for i in range(num_tasks):
        start = boundaries[i]
        end = boundaries[i+1]
        # 注意：tokenizer 对象会被序列化传给子进程
        tasks.append((input_path, start, end, tokenizer, i))

    total_tokens = 0
    
    # 3. 启动多进程处理
    # 使用 imap 保证结果有序返回 (order-preserving)，这对写入二进制文件至关重要
    print("启动多进程编码...")
    with mp.Pool(processes=num_workers) as pool:
        with open(output_path, 'wb') as f_out:
            # 使用 tqdm 包装循环
            # total=num_tasks: 进度条的总长度是任务块的数量
            # unit="chunk": 单位是块
            # desc: 进度条左侧的描述文字
            with tqdm(total=num_tasks, desc=f"Encoding {os.path.basename(input_path)}", unit="chunk") as pbar:
                # imap 返回的是一个迭代器，按任务提交顺序返回结果
                for result_arr in pool.imap(encode_worker, tasks):
                    if len(result_arr) > 0:
                        f_out.write(result_arr.tobytes())
                        total_tokens += len(result_arr)
                    
                    # 在进度条右侧实时更新已处理的 Token 总数
                    pbar.set_postfix(tokens=f"{total_tokens:,}")
                    pbar.update(1)

    duration = time.time() - t0
    speed = (file_size / (1024 * 1024)) / duration if duration > 0 else 0
    token_speed = total_tokens / duration / 1000 if duration > 0 else 0

    print(f"\n完成！")
    print(f"  - 总耗时: {duration:.2f} 秒")
    print(f"  - 处理速度: {speed:.2f} MB/s | {token_speed:.2f} k tokens/s")
    print(f"  - 生成文件: {output_path}")

def main():
    # 默认配置
    DATA_DIR = "data"
    DEFAULT_FILES = [
        # "owt_train.txt",
        # "owt_valid.txt",
        "TinyStoriesV2-GPT4-train.txt",
        # "TinyStoriesV2-GPT4-valid.txt"
    ]

    parser = argparse.ArgumentParser(description="使用 FastBPE 并行编码文本文件为二进制文件 (.bin)")
    
    parser.add_argument("--inputs", nargs='+', help="输入文本文件路径列表 (.txt)。如果不指定，默认使用代码中定义的 DATA_DIR 下的文件。")
    parser.add_argument("--model_dir", type=str, default="BPE_File", help="包含 vocab.json 和 merges.txt 的目录")
    parser.add_argument("--workers", type=int, default=15, help="并行工作的进程数") # 这里由于autodl的CPU核心数与系统信息不一致，需要手动指定自己被分配了几个CPU核心，否则进程太多反而会限制进程性能
    parser.add_argument("--vocab_file", type=str, default="gpt2_vocab.json", help="词表文件名")
    parser.add_argument("--merges_file", type=str, default="gpt2_merges.txt", help="Merges 文件名")
    
    args = parser.parse_args()

    vocab_path = os.path.join(args.model_dir, args.vocab_file)
    merges_path = os.path.join(args.model_dir, args.merges_file)

    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        print(f"错误: 找不到模型文件。\n请检查 {vocab_path} 和 {merges_path}")
        return

    # 1. 加载分词器
    print(f"正在加载分词器 (PID: {os.getpid()})...")
    t_load = time.time()
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=["<|endoftext|>"]
    )
    print(f"分词器加载完成，耗时 {time.time() - t_load:.2f} 秒。")

    # 2. 确定要处理的文件列表
    if args.inputs:
        # 如果命令行指定了文件，优先使用命令行参数
        files_to_process = args.inputs
    else:
        # 否则使用默认的 DATA_DIR 和文件列表
        print(f"未指定输入文件，使用默认目录 '{DATA_DIR}' 下的文件列表。")
        files_to_process = [os.path.join(DATA_DIR, f) for f in DEFAULT_FILES]

    # 3. 处理所有文件
    for input_file in files_to_process:
        process_file(input_file, tokenizer, args.workers)

if __name__ == '__main__':
    mp.freeze_support() # Windows 下必须
    main()