import json
import argparse
import numpy as np
from transformers.models.auto.tokenization_auto import AutoTokenizer
from tqdm import tqdm
import os

def calculate_response_token_length(output_path, model_path):
    # 1. 加载 Tokenizer
    print(f"正在加载 Tokenizer: {model_path} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"加载 Tokenizer 失败: {e}")
        return

    if not os.path.exists(output_path):
        print(f"错误: 找不到结果文件 {output_path}")
        return

    # 2. 读取数据并计算长度
    print(f"正在读取并分析: {output_path} ...")
    
    token_lengths = []
    char_lengths = []
    empty_count = 0
    format_failed_count = []
    length_bound_count = 0
    
    with open(output_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"共发现 {len(lines)} 条记录")

    for line in tqdm(lines, desc="Tokenizing"):
        try:
            data = json.loads(line)
            # 提取生成的文本
            text = data.get("generated_text", "")
            metrics = data.get("metrics")
            
            if not text:
                empty_count += 1
                token_lengths.append(0)
                char_lengths.append(0)
                continue


            # 计算字符长度
            char_lengths.append(len(text))
            
            # 计算 Token 长度
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if metrics["format_reward"] == 0:
                format_failed_count.append(len(tokens))
                if len(tokens) >= 4050:
                    length_bound_count += 1
            token_lengths.append(len(tokens))
            
        except json.JSONDecodeError:
            print("跳过无效的 JSON 行")
            continue

    # 3. 统计分析
    if not token_lengths:
        print("未找到有效数据。")
        return

    token_lengths = np.array(token_lengths)
    char_lengths = np.array(char_lengths)

    print("\n" + "="*50)
    print(f"真实输出长度统计 (基于 {os.path.basename(output_path)})")
    print("="*50)
    print(f"样本总数: {len(token_lengths)}")
    print(f"空输出数: {empty_count}")
    print(f"被推理长度限制的failed数:   {length_bound_count}")

    print("-" * 50)
    print("【Token 维度】 (用于检查 max_tokens)")
    print(f"平均长度 (Mean):   {np.mean(token_lengths):.2f}")
    print(f"中位数 (Median):   {np.median(token_lengths):.2f}")
    print(f"最大长度 (Max):    {np.max(token_lengths)}")
    print(f"最小长度 (Min):    {np.min(token_lengths)}")
    print(f"格式错误平均长度 (Mean):   {np.mean(format_failed_count):.2f}")
    print(f"格式错误中位数 (Median):   {np.median(format_failed_count):.2f}")
    print(f"格式错误最大长度 (Max):    {np.max(format_failed_count)}")
    print(f"格式错误最小长度 (Min):    {np.min(format_failed_count)}")
    print("-" * 50)
    print("【字符维度】")
    print(f"平均字符数:        {np.mean(char_lengths):.2f}")
    print(f"字符/Token比率:    {np.mean(char_lengths)/np.mean(token_lengths):.2f} (约值)")
    print("="*50)
    
    # 4. 额外检查：是否有截断迹象？
    # 如果很多样本的长度恰好等于 max_tokens (例如 1024 或 2048)
    # 则说明可能发生了截断
    unique_lengths, counts = np.unique(token_lengths, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    print("\n出现频率最高的 5 个 Token 长度:")
    for i in range(min(5, len(unique_lengths))):
        idx = sorted_indices[i]
        print(f"长度: {unique_lengths[idx]} Tokens - 出现 {counts[idx]} 次")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True, help="你的评估结果jsonl文件路径")
    parser.add_argument("--model_path", type=str, default="checkpoints/sft_v2/epoch1", help="模型路径")
    # parser.add_argument("--model_path", type=str, default="models/Qwen2.5-Math-1.5B", help="模型路径")
    args = parser.parse_args()
    
    calculate_response_token_length(args.output_path, args.model_path)