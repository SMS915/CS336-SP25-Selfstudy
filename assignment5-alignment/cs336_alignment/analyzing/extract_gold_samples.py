import json
import os
import argparse
import numpy as np
from transformers.models.auto.tokenization_auto import AutoTokenizer
from tqdm import tqdm


# 复用你之前的重复率检测逻辑
def calculate_lcp_ratio(text, min_pattern_len=10):
    if not text or len(text) < min_pattern_len:
        return 0.0
    seen_patterns = set()
    repeated_chars = 0
    for i in range(len(text) - min_pattern_len):
        pattern = text[i: i + min_pattern_len]
        if pattern in seen_patterns:
            repeated_chars += 1
        else:
            seen_patterns.add(pattern)
    return repeated_chars / len(text)


def is_mode_collapse(text, ratio_threshold=0.2, length_threshold=3000):
    """判定是否为模式坍塌的伪正确样本"""
    ratio = calculate_lcp_ratio(text)
    return ratio > ratio_threshold and len(text) > length_threshold


def process_and_extract(file_path, tokenizer, pass_k, gold_dir, model_prefix, dataset_key):
    """分析并提取金块样本"""
    gold_samples = []
    stats = {
        "total": 0, "pass_k_count": 0, "correct_count": 0,
        "collapse_count": 0, "gold_count": 0, "lengths": []
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                stats["total"] += 1

                # 1. 筛选 Pass@K
                current_attempt = data.get("attempt_id")
                if current_attempt is not None and int(current_attempt) > pass_k:
                    continue
                stats["pass_k_count"] += 1

                text = data.get("generated_text") or data.get("response") or ""
                metrics = data.get("metrics", {})

                # 2. 筛选 正确答案且格式正确
                if metrics.get("answer_reward") == 1.0 and metrics.get("format_reward") == 1.0:
                    stats["correct_count"] += 1

                    # 3. 剔除模式坍塌 (Reward Hacking)
                    if is_mode_collapse(text):
                        stats["collapse_count"] += 1
                    else:
                        # 4. 记录金块样本
                        token_count = len(tokenizer.encode(text, add_special_tokens=False))
                        stats["lengths"].append(token_count)
                        gold_samples.append(data)
                        stats["gold_count"] += 1

        # 保存金块样本到独立文件
        if gold_samples:
            gold_filename = f"gold_{model_prefix}_{dataset_key}.jsonl"
            with open(os.path.join(gold_dir, gold_filename), 'w', encoding='utf-8') as gf:
                for s in gold_samples:
                    gf.write(json.dumps(s, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"处理出错: {e}")
        return None

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True, help="评估结果jsonl文件路径")
    parser.add_argument("--model_path", type=str, default="models/Qwen2.5-Math-1.5B", help="模型路径")
    # 对应 pass_k
    parser.add_argument("--attempt_id", "--pass_k", type=int, default=None,
                        help="仅提取 attempt_id 小于等于此值的样本进行分析 (Pass@K 筛选)")

    input_dir = "./results"
    gold_dir = "./results/gold_samples"
    os.makedirs(gold_dir, exist_ok=True)

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

    tokenizer = AutoTokenizer.from_pretrained("models/Qwen2.5-Math-1.5B", trust_remote_code=True)

    with open("./results/gold_extraction_summary.txt", 'w', encoding='utf-8') as log_f:
        log_f.write("=" * 80 + "\nRL 采样金块提取 (Gold Extraction) 报告\n" + "=" * 80 + "\n\n")

        for model_prefix, dataset_key, pass_k in configs:
            sub_dir = os.path.join(input_dir, model_prefix)
            if not os.path.exists(sub_dir): continue

            # 定位文件逻辑 (同你提供的代码)
            target_file = next((f for f in os.listdir(sub_dir) if dataset_key in f and f.endswith(".jsonl")), None)
            if not target_file: continue

            file_path = os.path.join(sub_dir, target_file)
            res = process_and_extract(file_path, tokenizer, pass_k, gold_dir, model_prefix, dataset_key)

            if res:
                clean_ratio = res['gold_count'] / res['correct_count'] if res['correct_count'] > 0 else 0
                report = (
                        f"【项目】: {model_prefix.upper()} | {dataset_key.upper()}\n"
                        f"  - 筛选范围: Pass@{pass_k} | 初始正确数: {res['correct_count']}\n"
                        f"  - 模式坍塌剔除: {res['collapse_count']} (清洗率: {clean_ratio:.2%})\n"
                        f"  - 最终金块数: {res['gold_count']}\n"
                        f"  - 金块平均长度: {np.mean(res['lengths']) if res['lengths'] else 0:.1f} tokens\n"
                        + "-" * 40 + "\n"
                )
                log_f.write(report)
                print(f"已完成: {model_prefix} - {dataset_key}")


if __name__ == "__main__":
    main()