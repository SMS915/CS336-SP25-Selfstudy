import json
import os
import argparse
import numpy as np
from transformers.models.auto.tokenization_auto import AutoTokenizer
from tqdm import tqdm


def analyze_file(file_path, tokenizer, attempt_id_threshold=None):
    """分析单个文件并返回统计数据字典"""
    token_lengths = []
    answer_correct_list = []
    format_failed_list = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)

                # 筛选 attempt_id
                current_attempt = data.get("attempt_id")
                if attempt_id_threshold is not None and current_attempt is not None:
                    if int(current_attempt) > attempt_id_threshold:
                        continue

                text = data.get("generated_text") or data.get("response") or ""
                metrics = data.get("metrics", {})

                # Tokenization
                tokens = tokenizer.encode(text, add_special_tokens=False)
                token_count = len(tokens)

                # 统计分类
                if metrics.get("format_reward", 0) == 0:
                    format_failed_list.append(token_count)
                elif metrics.get("answer_reward", 0) == 1.0:
                    answer_correct_list.append(token_count)

                token_lengths.append(token_count)
    except Exception as e:
        return {"error": str(e)}

    if not token_lengths:
        return None

    return {
        "count": len(token_lengths),
        "acc": len(answer_correct_list) / len(token_lengths),
        "format_err": len(format_failed_list) / len(token_lengths),
        "mean": np.mean(token_lengths),
        "median": np.median(token_lengths),
        "max": np.max(token_lengths),
        "correct_mean": np.mean(answer_correct_list) if answer_correct_list else 0,
        "failed_mean": np.mean(format_failed_list) if format_failed_list else 0
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./results", help="结果文件夹")
    parser.add_argument("--model_path", type=str, default="models/Qwen2.5-Math-1.5B")
    parser.add_argument("--output_file", type=str, default="./results/results_stats_summary.txt")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # 定义配置：(模型前缀, 数据集关键字, 筛选Pass@K)
    configs = [
        # --- GSM8K ---
        ("baseline", "gsm8k_pass_1", 1),
        ("sft", "gsm8k_pass_1", 1),
        ("grpo", "gsm8k_pass_1", 1),
        ("grpo_no_std_norm", "gsm8k_pass_1", 1),
        ("drgrpo", "gsm8k_pass_1", 1),
        ("drgrpo_best", "gsm8k_pass_1", 1),
        ("instruct", "gsm8k_pass_1", 1),

        # --- MATH-500 ---
        ("baseline", "math500_pass_64", 64),
        ("sft", "math500_pass_64", 64),
        ("grpo", "math500_pass_64", 64),
        ("grpo_no_std_norm", "math500_pass_64", 64),
        ("drgrpo", "math500_pass_64", 64),
        ("drgrpo_best", "math500_pass_64", 64),
        ("instruct", "math500_pass_64", 64),

        # --- MathTest ---
        ("baseline", "MathTest_pass_8", 8),
        ("sft", "MathTest_pass_8", 8),
        ("grpo", "MathTest_pass_8", 8),
        ("grpo_no_std_norm", "MathTest_pass_8", 8),
        ("drgrpo", "MathTest_pass_8", 8),
        ("drgrpo_best", "MathTest_pass_8", 8),
        ("instruct", "MathTest_pass_8", 8),


        # --- AMC (American Mathematics Competitions) ---
        ("baseline", "amc_pass_64", 64),
        ("sft", "amc_pass_64", 64),
        ("grpo", "amc_pass_64", 64),
        ("grpo_no_std_norm", "amc_pass_64", 64),
        ("drgrpo", "amc_pass_64", 64),
        ("drgrpo_best", "amc_pass_64", 64),
        ("instruct", "amc_pass_64", 64),

        # --- AIME 2024 ---
        ("baseline", "aime24_pass_64", 64),
        ("sft", "aime24_pass_64", 64),
        ("grpo", "aime24_pass_64", 64),
        ("grpo_no_std_norm", "aime24_pass_64", 64),
        ("drgrpo", "aime24_pass_64", 64),
        ("drgrpo_best", "aime24_pass_64", 64),
        ("instruct", "aime24_pass_64", 64),

        # --- AIME 2025 ---
        ("baseline", "aime25_pass_64", 64),
        ("sft", "aime25_pass_64", 64),
        ("grpo", "aime25_pass_64", 64),
        ("grpo_no_std_norm", "aime25_pass_64", 64),
        ("drgrpo", "aime25_pass_64", 64),
        ("drgrpo_best", "aime25_pass_64", 64),
        ("instruct", "aime25_pass_64", 64),
    ]

    all_files = os.listdir(args.input_dir)
    results_log = []

    print(f"开始批量处理 {args.input_dir} 中的文件...")

    with open(args.output_file, 'w', encoding='utf-8') as log_f:
        log_f.write("=" * 80 + "\n")
        log_f.write(f"模型微调阶段对比统计报告\n源目录: {args.input_dir}\n")
        log_f.write("=" * 80 + "\n\n")

        for model_prefix, dataset_key, pass_k in configs:
            # 1. 构建子文件夹路径 (例如: ./results/drgrpo)
            sub_dir = os.path.join(args.input_dir, model_prefix)

            if not os.path.exists(sub_dir):
                print(f"跳过: 文件夹不存在 {sub_dir}")
                continue

            # 2. 在子文件夹中检索文件
            target_file = None
            sub_all_files = os.listdir(sub_dir)
            for f in sub_all_files:
                # 匹配规则：文件名包含数据集关键字 且 以 .jsonl 结尾
                if dataset_key in f and f.endswith(".jsonl"):
                    target_file = f
                    break

            if not target_file:
                print(f"未找到匹配文件: {model_prefix} -> {dataset_key}")
                continue

            # 3. 拼接完整路径进行分析
            file_path = os.path.join(sub_dir, target_file)
            print(f"正在分析: {model_prefix}/{target_file} (Pass@{pass_k})")

            stats = analyze_file(file_path, tokenizer, pass_k)

            if stats:
                report = (
                        f"【项目】: {model_prefix.upper()} | {dataset_key.upper()} | Pass@{pass_k}\n"
                        f"  - 文件: {target_file}\n"
                        f"  - 样本数: {stats['count']}\n"
                        f"  - 正确率: {stats['acc']:.2%} | 格式错误率: {stats['format_err']:.2%}\n"
                        f"  - Token长度: 平均 {stats['mean']:.1f} | 中位数 {stats['median']:.1f} | 最大 {stats['max']}\n"
                        f"  - 成功样本平均Token: {stats['correct_mean']:.1f}\n"
                        f"  - 格式错误平均Token: {stats['failed_mean']:.1f}\n"
                        + "-" * 40 + "\n"
                )
                log_f.write(report)
                log_f.flush()

    print(f"分析完成！结果已写入 {args.output_file}")


if __name__ == "__main__":
    main()