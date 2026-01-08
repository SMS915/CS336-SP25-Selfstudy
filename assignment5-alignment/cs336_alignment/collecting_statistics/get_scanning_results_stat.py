import json
import os
import argparse
import numpy as np
from transformers.models.auto.tokenization_auto import AutoTokenizer
from tqdm import tqdm


def analyze_file(file_path, tokenizer):
    """分析单个文件并返回统计数据字典"""
    token_lengths = []
    answer_correct_list = []
    format_failed_list = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)

                # 筛选 attempt_id
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
    parser.add_argument("--input_dir", type=str, default="./results/scan_grpo_no_std_norm", help="结果文件夹")
    parser.add_argument("--model_path", type=str, default="models/Qwen2.5-Math-1.5B")
    parser.add_argument("--output_file", type=str, default="./results/grpo_no_std_norm_check_summary.txt")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # 定义配置：(模型前缀, 数据集关键字, 筛选Pass@K)
    configs = [
        ('aime2025', 50),
        ('aime2025', 100),
        ('aime2025', 150),
        ('aime2025', 200),
        ('aime2025', 250),
        ('aime2025', 300),
        ('aime2025', 350),
        ('aime2025', 400),
        ('aime2025', 450),
        ('aime2025', 500),
        ('aime2025', 550),

        ('math500', 50),
        ('math500', 100),
        ('math500', 150),
        ('math500', 200),
        ('math500', 250),
        ('math500', 300),
        ('math500', 350),
        ('math500', 400),
        ('math500', 450),
        ('math500', 500),
        ('math500', 550),
    ]

    all_files = os.listdir(args.input_dir)
    results_log = []

    print(f"开始批量处理 {args.input_dir} 中的文件...")

    with open(args.output_file, 'w', encoding='utf-8') as log_f:
        log_f.write("=" * 80 + "\n")
        log_f.write(f"模型微调阶段对比统计报告\n源目录: {args.input_dir}\n")
        log_f.write("=" * 80 + "\n\n")

        for dataset_key, ckpt_idx in configs:
            # 检索符合条件的文件
            target_file = None
            for f in all_files:
                if f.startswith(dataset_key) and '_' + str(ckpt_idx) in f and f.endswith(".jsonl"):
                    target_file = f
                    break

            if not target_file:
                continue

            file_path = os.path.join(args.input_dir, target_file)
            print(f"正在分析: {target_file}")

            stats = analyze_file(file_path, tokenizer)

            if stats:
                report = (
                        f"【项目】: drgrpo_curriculum step{ckpt_idx} | {dataset_key}\n"
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