import os
import json
import multiprocessing
from functools import partial
from transformers.models.auto.tokenization_auto import AutoTokenizer
from tqdm import tqdm

# --- 配置 ---
INPUT_FILE = "data/MATH/sft.jsonl"
OUTPUT_FILE = "data/MATH/sft_v3.jsonl"
MODEL_PATH = "models/Qwen2.5-Math-1.5B" 
MAX_LEN = 2560
MIN_LEN = 300 
NUM_PROCESSES = 30

tokenizer = None

def init_worker(model_path):
    """子进程初始化函数：每个进程独立加载 Tokenizer"""
    global tokenizer
    # 禁用 Tokenizer 自身的并行，防止与 Multiprocessing 冲突
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def check_quality_worker(line):
    """
    工作函数：接收一行文本，解析并检查
    返回: (is_valid, reason, json_str_or_none)
    """
    global tokenizer
    try:
        data = json.loads(line)
    except Exception:
        return False, "json_error", None

    prompt = data.get('prompt', "")
    response = data.get('response', "")
    
    # 0. 基础判空
    if not prompt or not response:
        return False, "empty_data", None

    full_text = prompt + response
    
    # 1. 长度检查 (使用全局 tokenizer)
    try:
        tokens = tokenizer.encode(full_text)
        if len(tokens) > MAX_LEN: return False, "too_long", None
        if len(tokens) < MIN_LEN: return False, "too_short", None
    except Exception:
        return False, "tokenization_error", None
    
    # 2. 标签存在性 & 唯一性检查
    tags = ["<think>", "</think>", "<answer>", "</answer>"]
    for tag in tags:
        count = response.count(tag)
        if count == 0:
            return False, f"missing_{tag}", None
        if count > 1:
            return False, f"duplicate_{tag}", None # 拒绝标签重复
            
    # 3. 标签顺序检查
    t_start = response.find("<think>")
    t_end = response.find("</think>")
    a_start = response.find("<answer>")
    a_end = response.find("</answer>")
    
    # 顺序必须严格: <think> ... </think> ... <answer> ... </answer>
    if not (t_start < t_end < a_start < a_end):
        return False, "wrong_order", None
        
    # 4. Boxed 检查
    answer_content = response[a_start+8 : a_end]
    if "\\boxed" not in answer_content:
        return False, "no_boxed", None
    
    # 5. 空内容检查
    think_content = response[t_start+7 : t_end].strip()
    if not think_content:
        return False, "empty_think", None
        
    if not answer_content.strip():
        return False, "empty_answer", None

    # 检查通过，返回原始数据的 JSON 字符串（重新 dump 以保格式统一，或者直接用 line 也可以）
    return True, "pass", json.dumps(data, ensure_ascii=False)

def main():
    # 统计行数用于进度条 (如果文件极大，可以跳过这一步直接设 total=None)
    print("正在计算文件行数...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"开始并行处理数据 (进程数: {NUM_PROCESSES})...")
    
    valid_count = 0
    reject_stats = {}
    
    # 打开输入和输出文件
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        
        # 创建进程池
        # chunksize 稍微设大一点可以减少进程间通信开销
        with multiprocessing.Pool(processes=NUM_PROCESSES, initializer=init_worker, initargs=(MODEL_PATH,)) as pool:
            
            # 使用 imap 进行流式处理，保持顺序（如果顺序不重要可以用 imap_unordered 更快一点点）
            # chunksize=100 意味着一次给子进程发100行
            iterator = pool.imap(check_quality_worker, fin, chunksize=100)
            
            for is_valid, reason, result_str in tqdm(iterator, total=total_lines):
                if is_valid:
                    fout.write(result_str + "\n")
                    valid_count += 1
                else:
                    reject_stats[reason] = reject_stats.get(reason, 0) + 1

    # --- 输出统计报告 ---
    print("\n" + "=" * 40)
    print(f"处理完成！")
    print(f"原始数据总量: {total_lines}")
    print(f"保留数据总量: {valid_count}")
    print(f"保留率: {valid_count/total_lines:.2%}")
    print(f"输出文件: {OUTPUT_FILE}")
    print("-" * 40)
    print("拒绝原因分布:")
    # 按数量降序排列
    sorted_stats = sorted(reject_stats.items(), key=lambda x: x[1], reverse=True)
    for reason, count in sorted_stats:
        print(f"  {reason:<20}: {count}")
    print("=" * 40)

if __name__ == "__main__":
    # Windows下必须把代码放在 if __name__ == "__main__": 下
    # Linux下是个好习惯
    main()