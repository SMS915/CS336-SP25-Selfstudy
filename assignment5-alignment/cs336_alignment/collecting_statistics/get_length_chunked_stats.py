import json
import os
import argparse
import numpy as np
from transformers.models.auto.tokenization_auto import AutoTokenizer
from tqdm import tqdm


def get_length_limit(dataset_key):
    """根据任务关键词返回长度上限"""
    dk = dataset_key.lower()
    if "gsm8k" in dk:
        return 2048
    else:
        return 4096


def analyze_format_failures(file_path, tokenizer, pass_k, limit):
    """分析格式错误样本中的长度截断情况"""
    stats = {
        "pass_k_count": 0,
        "format_error_count": 0,
        "length_hit_count": 0,  # 触及长度限制的格式错误数
        "error_lengths": []
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)

                # 1. Pass@K 筛选
                current_attempt = data.get("attempt_id")
                if current_attempt is not None and int(current_attempt) > pass_k:
                    continue
                stats["pass_k_count"] += 1

                text = data.get("generated_text") or data.get("response") or ""
                metrics = data.get("metrics", {})

                # 2. 筛选格式错误 (format_reward != 1.0)
                # 注意：这里通常包括格式不完全正确或根本没写完的
                if metrics.get("format_reward", 0) < 1.0:
                    stats["format_error_count"] += 1

                    # 3. 计算 Token 长度并判断是否触及上限
                    token_count = len(tokenizer.encode(text, add_special_tokens=False))

                    # 考虑到 tokenizer 编码差异或 EOS 占位，通常达到 limit-1 或 limit 即可判定为截断
                    if token_count >= limit - 2:
                        stats["length_hit_count"] += 1
                        stats["error_lengths"].append(token_count)

    except Exception as e:
        print(f"处理文件 {file_path} 出错: {e}")
        return None

    return stats


def main():
    # 路径配置
    input_dir = "./results"
    output_log = "./results/format_failure_analysis.txt"

    # 复用你的配置列表
    configs = [
        # --- GSM8K (Limit: 2048) ---
        ("baseline", "gsm8k_pass_1", 1), ("sft", "gsm8k_pass_1", 1),
        ("grpo", "gsm8k_pass_1", 1), ("grpo_without_std_norm", "gsm8k_pass_1", 1),
        ("drgrpo", "gsm8k_pass_1", 1), ("drgrpo_best", "gsm8k_pass_1", 1),
        ("instruct", "gsm8k_pass_1", 1),

        # --- MATH-500 (Limit: 4096) ---
        ("baseline", "math500_pass_64", 64), ("sft", "math500_pass_64", 64),
        ("grpo", "math500_pass_64", 64), ("grpo_without_std_norm", "math500_pass_64", 64),
        ("drgrpo", "math500_pass_64", 64), ("drgrpo_best", "math500_pass_64", 64),
        ("instruct", "math500_pass_64", 64),

        # --- MathTest ---
        ("baseline", "MathTest_pass_8", 8), ("sft", "MathTest_pass_8", 8),
        ("grpo", "MathTest_pass_8", 8), ("grpo_without_std_norm", "MathTest_pass_8", 8),
        ("drgrpo", "MathTest_pass_8", 8),("drgrpo_best", "MathTest_pass_8", 8),
        ("instruct", "MathTest_pass_8", 8),

        # --- AMC ---
        ("baseline", "amc_pass_64", 64), ("sft", "amc_pass_64", 64),
        ("grpo", "amc_pass_64", 64), ("grpo_without_std_norm", "amc_pass_64", 64),
        ("drgrpo", "amc_pass_64", 64), ("drgrpo_best", "amc_pass_64", 64),
        ("instruct", "amc_pass_64", 64),

        # --- AIME 2024 ---
        ("baseline", "aime24_pass_64", 64), ("sft", "aime24_pass_64", 64),
        ("grpo", "aime24_pass_64", 64), ("grpo_without_std_norm", "aime24_pass_64", 64),
        ("drgrpo", "aime24_pass_64", 64), ("drgrpo_best", "aime24_pass_64", 64),
        ("instruct", "aime24_pass_64", 64),

        # --- AIME 2025 ---
        ("baseline", "aime25_pass_64", 64), ("sft", "aime25_pass_64", 64),
        ("grpo", "aime25_pass_64", 64), ("grpo_without_std_norm", "aime25_pass_64", 64),
        ("drgrpo", "aime25_pass_64", 64), ("drgrpo_best", "aime25_pass_64", 64),
        ("instruct", "aime25_pass_64", 64),
    ]

    tokenizer = AutoTokenizer.from_pretrained("models/Qwen2.5-Math-1.5B", trust_remote_code=True)

    with open(output_log, 'w', encoding='utf-8') as log_f:
        log_f.write("=" * 80 + "\n格式错误与长度上限 (Length Truncation) 分析报告\n" + "=" * 80 + "\n\n")

        for model_prefix, dataset_key, pass_k in configs:
            sub_dir = os.path.join(input_dir, model_prefix)
            if not os.path.exists(sub_dir): continue

            target_file = next((f for f in os.listdir(sub_dir) if dataset_key in f and f.endswith(".jsonl")), None)
            if not target_file: continue

            limit = get_length_limit(dataset_key)
            file_path = os.path.join(sub_dir, target_file)

            res = analyze_format_failures(file_path, tokenizer, pass_k, limit)

            if res and res['format_error_count'] > 0:
                hit_ratio = res['length_hit_count'] / res['format_error_count']
                report = (
                        f"【项目】: {model_prefix.upper()} | {dataset_key.upper()}\n"
                        f"  - 设定上限: {limit} tokens | Pass@{pass_k}\n"
                        f"  - 格式错误总数: {res['format_error_count']}\n"
                        f"  - 触及长度上限数: {res['length_hit_count']}\n"
                        f"  - 长度截断占比 (Hit/Format Error): {hit_ratio:.2%}\n"
                        + "-" * 40 + "\n"
                )
                log_f.write(report)
                print(f"分析完成: {model_prefix} - {dataset_key} (截断率: {hit_ratio:.2%})")
            elif res:
                print(f"分析完成: {model_prefix} - {dataset_key} (无格式错误)")

    print(f"\n报告已保存至: {output_log}")


if __name__ == "__main__":
    main()