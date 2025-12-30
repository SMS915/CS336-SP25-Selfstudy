import json
import argparse
import numpy as np
from transformers.models.auto.tokenization_auto import AutoTokenizer
from tqdm import tqdm
import os


def calculate_response_token_length(output_path, model_path, attempt_id_threshold=None):
    # 1. 加载 Tokenizer
    print(f"正在加载 Tokenizer: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"加载 Tokenizer 失败: {e}")
        return

    if not os.path.exists(output_path):
        print(f"错误: 找不到结果文件 {output_path}")
        return

    # 2. 读取数据并计算长度
    print(f"正在读取并分析: {output_path}")
    if attempt_id_threshold is not None:
        print(f"筛选条件: 仅统计 attempt_id <= {attempt_id_threshold} 的样本")

    token_lengths = []
    char_lengths = []
    format_failed_list = []
    answer_correct_list = []

    # 计数器
    total_lines = 0
    filtered_count = 0

    with open(output_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    print(f"文件共包含 {total_lines} 条记录")

    for line in tqdm(lines, desc="Processing & Tokenizing"):
        try:
            data = json.loads(line)

            # --- 新增优先筛选逻辑 ---
            current_attempt = data.get("attempt_id")
            # 如果设定了阈值且字段存在，执行筛选
            if attempt_id_threshold is not None and current_attempt is not None:
                if int(current_attempt) > attempt_id_threshold:
                    continue
            # ----------------------

            # 提取生成的文本
            text = data.get("generated_text") or data.get("response") or data.get("output") or ""
            metrics = data.get("metrics", {})

            # 计算字符长度
            char_lengths.append(len(text))

            # 计算 Token 长度
            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_count = len(tokens)

            # 统计逻辑
            format_reward = metrics.get("format_reward", 0)
            answer_reward = metrics.get("answer_reward", 0)

            if format_reward == 0:
                format_failed_list.append(token_count)
            elif answer_reward == 1.0:
                answer_correct_list.append(token_count)

            token_lengths.append(token_count)
            filtered_count += 1

        except json.JSONDecodeError:
            print("跳过无效的 JSON 行")
            continue

    # 3. 统计分析
    if not token_lengths:
        print("未找到匹配筛选条件的有效数据。")
        return

    token_lengths = np.array(token_lengths)
    char_lengths = np.array(char_lengths)

    print("\n" + "=" * 50)
    print(f"真实输出长度统计 (基于 {os.path.basename(output_path)})")
    if attempt_id_threshold is not None:
        print(f"筛选模式: Pass@{attempt_id_threshold} 子集分析")
    print("=" * 50)
    print(f"原始总行数: {total_lines}")
    print(f"筛选后样本数: {len(token_lengths)}")
    print(f"有效率 (筛选后/原始): {len(token_lengths) / total_lines:.2%}")
    print(f"正确率 (基于筛选后): {len(answer_correct_list) / len(token_lengths):.2%}")
    print(f"格式错误率: {len(format_failed_list) / len(token_lengths):.2%}")
    print("-" * 50)
    print("【Token 维度】")
    print(f"平均长度 (Mean):   {np.mean(token_lengths):.2f}")
    print(f"中位数 (Median):   {np.median(token_lengths):.2f}")
    print(f"最大长度 (Max):    {np.max(token_lengths)}")
    print(f"最小长度 (Min):    {np.min(token_lengths)}")
    print("\n")
    if answer_correct_list:
        print(f"答案正确平均长度:   {np.mean(answer_correct_list):.2f}")
    if format_failed_list:
        print(f"格式错误平均长度:   {np.mean(format_failed_list):.2f}")
    print("-" * 50)
    print("【字符维度】")
    print(f"平均字符数:        {np.mean(char_lengths):.2f}")
    print(f"字符/Token比率:    {np.mean(char_lengths) / np.mean(token_lengths):.2f}")
    print("=" * 50)

    unique_lengths, counts = np.unique(token_lengths, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    print("\n出现频率最高的 5 个 Token 长度:")
    for i in range(min(5, len(unique_lengths))):
        idx = sorted_indices[i]
        print(f"长度: {unique_lengths[idx]} Tokens - 出现 {counts[idx]} 次")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True, help="评估结果jsonl文件路径")
    parser.add_argument("--model_path", type=str, default="models/Qwen2.5-Math-1.5B", help="模型路径")

    # 新增参数：对应 pass_k 逻辑
    parser.add_argument("--attempt_id", "--pass_k", type=int, default=None,
                        help="仅提取 attempt_id 小于等于此值的样本进行分析 (Pass@K 筛选)")

    args = parser.parse_args()

    calculate_response_token_length(args.output_path, args.model_path, args.attempt_id)