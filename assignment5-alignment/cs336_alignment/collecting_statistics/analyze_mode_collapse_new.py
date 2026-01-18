import json
import os
import csv
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def calculate_lcp_ratio(text, min_pattern_len=10):
    """
    计算最大重复模式占比 (基于字符匹配)
    """
    if not text or len(text) < min_pattern_len:
        return 0.0

    seen_patterns = set()
    repeated_chars = 0

    # 统计字符级别的重复覆盖率
    for i in range(len(text) - min_pattern_len):
        pattern = text[i: i + min_pattern_len]
        if pattern in seen_patterns:
            repeated_chars += 1
        else:
            seen_patterns.add(pattern)

    return repeated_chars / len(text)


def is_format_error(data_row, text):
    """
    判定是否为格式错误
    """
    # 1. 优先检查 metrics
    if "metrics" in data_row and isinstance(data_row["metrics"], dict):
        fmt_reward = data_row["metrics"].get("format_reward")
        if fmt_reward is not None:
            return float(fmt_reward) == 0.0

    # 2. 回退机制
    if not text:
        return True
    if "<answer>" not in text or "</answer>" not in text:
        return True

    return False


def parse_filename(filename):
    """
    解析文件名: {model}_{dataset}_pass_{k}.jsonl
    """
    if not filename.endswith(".jsonl"):
        return None, None, None
    try:
        name_part = filename.replace(".jsonl", "")
        if "_pass_" in name_part:
            base_info, pass_k = name_part.split("_pass_")
            return base_info, pass_k
    except:
        pass
    return filename.replace(".jsonl", ""), "unknown"


def run_analysis(input_root, target_models, model_path, output_csv, collapse_threshold=2048):
    """
    核心分析函数
    :param target_models: 用户指定的子文件夹名称列表
    """
    # 1. 加载 Tokenizer
    print(f"正在加载 Tokenizer: {model_path} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"加载 Tokenizer 失败: {e}")
        return

    results_data = []

    # 2. 遍历指定的文件夹
    for model_name in target_models:
        current_dir = os.path.join(input_root, model_name)

        if not os.path.exists(current_dir):
            print(f"跳过: 目录不存在 -> {current_dir}")
            continue

        files = [f for f in os.listdir(current_dir) if f.endswith(".jsonl")]
        if not files:
            print(f"警告: 在 {current_dir} 中未找到 .jsonl 文件")
            continue

        for filename in files:
            file_path = os.path.join(current_dir, filename)
            base_info, pass_k = parse_filename(filename)
            dataset = base_info.replace(f"{model_name}_", "") if base_info else "unknown"

            print(f"分析中: [{model_name}] - {dataset} (Pass@{pass_k})")

            stats = {
                "total": 0,
                "format_error_total": 0,
                "collapse_total": 0,
                "fmt_err_collapse_count": 0,
                "fmt_err_collapse_token_sum": 0,
                "fmt_err_normal_count": 0,
                "fmt_err_normal_token_sum": 0
            }

            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in tqdm(lines, desc=f"  Processing {filename[:20]}...", leave=False):
                try:
                    data = json.loads(line)
                    text = data.get("generated_text") or data.get("response") or ""
                    if not text: continue

                    stats["total"] += 1

                    # --- Token 统计 ---
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    token_len = len(tokens)

                    # --- 坍塌判定 (基于 Token 长度阈值) ---
                    ratio = calculate_lcp_ratio(text)
                    is_collapsed = (ratio > 0.2 and token_len > collapse_threshold)

                    if is_collapsed:
                        stats["collapse_total"] += 1

                    # --- 格式错误判定 ---
                    is_fmt_err = is_format_error(data, text)

                    if is_fmt_err:
                        stats["format_error_total"] += 1
                        if is_collapsed:
                            stats["fmt_err_collapse_count"] += 1
                            stats["fmt_err_collapse_token_sum"] += token_len
                        else:
                            stats["fmt_err_normal_count"] += 1
                            stats["fmt_err_normal_token_sum"] += token_len

                except Exception:
                    continue

            # --- 计算指标 ---
            if stats["total"] == 0: continue

            collapse_share = (stats["fmt_err_collapse_count"] / stats["format_error_total"]) if stats[
                                                                                                    "format_error_total"] > 0 else 0
            avg_tok_collapse = (stats["fmt_err_collapse_token_sum"] / stats["fmt_err_collapse_count"]) if stats[
                                                                                                              "fmt_err_collapse_count"] > 0 else 0
            avg_tok_normal = (stats["fmt_err_normal_token_sum"] / stats["fmt_err_normal_count"]) if stats[
                                                                                                        "fmt_err_normal_count"] > 0 else 0
            token_gap = avg_tok_collapse - avg_tok_normal

            row = {
                "Model": model_name,
                "Dataset": dataset,
                "Pass_K": pass_k,
                "Total_Samples": stats["total"],
                "Total_Format_Errors": stats["format_error_total"],
                "Format_Error_Rate": f"{stats['format_error_total'] / stats['total']:.2%}",
                "Collapse_In_Errors_Count": stats["fmt_err_collapse_count"],
                "Collapse_In_Errors_Share": f"{collapse_share:.2%}",
                "Avg_Token_Collapse_Err": int(avg_tok_collapse),
                "Avg_Token_Normal_Err": int(avg_tok_normal),
                "Token_Gap": int(token_gap)
            }
            results_data.append(row)

    # 3. 写入 CSV
    keys = ["Model", "Dataset", "Pass_K", "Total_Samples", "Total_Format_Errors", "Format_Error_Rate",
            "Collapse_In_Errors_Count", "Collapse_In_Errors_Share", "Avg_Token_Collapse_Err",
            "Avg_Token_Normal_Err", "Token_Gap"]

    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results_data)

    print(f"\n✅ 分析完成！结果已保存至: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mode Collapse Analysis (Token-based & List-filtered)")
    parser.add_argument("--results_dir", type=str, default="./results", help="包含 jsonl 文件的结果根目录")
    parser.add_argument("--model_path", type=str, default="./models/Qwen2.5-Math-1.5B",
                        help="用于加载 Tokenizer 的路径")
    parser.add_argument("--output_csv", type=str, default="./results/collapse_token_stats.csv", help="输出 CSV 路径")
    parser.add_argument("--threshold", type=int, default=2048, help="判定坍塌的 Token 长度阈值")

    args = parser.parse_args()

    # --- 1. 在这里指定你要分析的子文件夹列表 ---
    TARGET_MODELS = [
        "baseline",
        "sft",
        "grpo",
        "grpo_without_std_norm",
        "drgrpo",
        "drgrpo_curriculum",
        "instruct"
    ]

    # --- 2. 运行分析 ---
    if os.path.exists(args.results_dir):
        run_analysis(
            input_root=args.results_dir,
            target_models=TARGET_MODELS,
            model_path=args.model_path,
            output_csv=args.output_csv,
            collapse_threshold=args.threshold
        )
    else:
        print(f"错误: 未找到目录 {args.results_dir}")