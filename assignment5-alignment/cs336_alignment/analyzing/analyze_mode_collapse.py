import json
import os
import csv
from tqdm import tqdm


def calculate_lcp_ratio(text, min_pattern_len=10):
    """
    计算最大重复模式占比 (Mode Collapse 核心指标)
    """
    if not text or len(text) < min_pattern_len:
        return 0.0

    seen_patterns = set()
    repeated_chars = 0

    # 步长为1，统计字符级别的重复覆盖率
    for i in range(len(text) - min_pattern_len):
        pattern = text[i: i + min_pattern_len]
        if pattern in seen_patterns:
            repeated_chars += 1
        else:
            seen_patterns.add(pattern)

    return repeated_chars / len(text)


def is_format_error(data_row):
    """
    判定是否为格式错误。
    优先使用 evaluate 脚本生成的 metrics 字段。
    """
    # 1. 优先检查 metrics 字段 (最准确)
    if "metrics" in data_row and isinstance(data_row["metrics"], dict):
        # 如果 format_reward 为 0.0，则为格式错误
        fmt_reward = data_row["metrics"].get("format_reward")
        if fmt_reward is not None:
            return float(fmt_reward) == 0.0

    # 2. 回退机制：如果没有 metrics 字段，使用文本匹配 heuristic
    # (防止部分原始日志没有 metrics)
    text = data_row.get("generated_text") or data_row.get("response") or ""
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
        # 根据你的命名习惯调整，这里尝试分割
        if "_pass_" in name_part:
            base_info, pass_k = name_part.split("_pass_")
            return base_info, pass_k
    except:
        pass
    return filename.replace(".jsonl", ""), "unknown"


def run_analysis(input_root, output_txt, output_csv):
    results_data = []

    with open(output_txt, 'w', encoding='utf-8') as log_f:
        log_f.write("=" * 100 + "\n")
        log_f.write(f"模型模式坍缩深度诊断 (基于 Format Reward)\n源目录: {input_root}\n")
        log_f.write("=" * 100 + "\n\n")

        subdirs = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]

        for model_name in subdirs:
            current_dir = os.path.join(input_root, model_name)
            files = [f for f in os.listdir(current_dir) if f.endswith(".jsonl")]

            for filename in files:
                file_path = os.path.join(current_dir, filename)
                base_info, pass_k = parse_filename(filename)
                dataset = base_info.replace(f"{model_name}_", "") if base_info else "unknown"

                print(f"正在分析 [{model_name}] - {dataset} (Pass@{pass_k})...")

                # 统计计数器
                stats = {
                    "total": 0,
                    "format_error_total": 0,
                    "collapse_total": 0,

                    # 细分：格式错误中的坍塌情况
                    "fmt_err_collapse_count": 0,
                    "fmt_err_collapse_len_sum": 0,

                    # 细分：格式错误中的非坍塌情况
                    "fmt_err_normal_count": 0,
                    "fmt_err_normal_len_sum": 0
                }

                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)

                            # 获取生成的文本内容用于计算长度和重复率
                            text = data.get("generated_text") or data.get("response") or ""
                            if not text: continue

                            stats["total"] += 1
                            text_len = len(text)

                            # --- 核心判定逻辑 ---

                            # 1. 判定模式坍塌 (Mode Collapse)
                            # 定义：长度超过3000且重复率超过20%
                            ratio = calculate_lcp_ratio(text)
                            is_collapsed = (ratio > 0.2 and text_len > 3000)

                            if is_collapsed:
                                stats["collapse_total"] += 1

                            # 2. 判定格式错误 (Format Error)
                            # 使用 metrics 中的 format_reward
                            is_fmt_err = is_format_error(data)

                            if is_fmt_err:
                                stats["format_error_total"] += 1

                                if is_collapsed:
                                    # Case A: 格式错了，且是因为死循环 (坍塌)
                                    stats["fmt_err_collapse_count"] += 1
                                    stats["fmt_err_collapse_len_sum"] += text_len
                                else:
                                    # Case B: 格式错了，但是正常的错误 (非坍塌)
                                    stats["fmt_err_normal_count"] += 1
                                    stats["fmt_err_normal_len_sum"] += text_len

                        except Exception as e:
                            # print(f"Error parsing line: {e}")
                            continue

                # --- 计算统计指标 ---

                # 1. 整体坍塌率
                collapse_ratio = stats["collapse_total"] / stats["total"] if stats["total"] > 0 else 0

                # 2. 格式错误中的坍塌占比 (Diagnosis Metric)
                # 解释：在所有格式错误的样本中，有多少是因为“疯了”(Mode Collapse)导致的？
                collapse_share_in_error = (stats["fmt_err_collapse_count"] / stats["format_error_total"]) if stats[
                                                                                                                 "format_error_total"] > 0 else 0

                # 3. 平均长度计算
                avg_len_collapse = (stats["fmt_err_collapse_len_sum"] / stats["fmt_err_collapse_count"]) if stats[
                                                                                                                "fmt_err_collapse_count"] > 0 else 0
                avg_len_normal = (stats["fmt_err_normal_len_sum"] / stats["fmt_err_normal_count"]) if stats[
                                                                                                          "fmt_err_normal_count"] > 0 else 0

                # 4. 长度剪刀差 (Length Gap)
                len_gap = avg_len_collapse - avg_len_normal

                # 记录数据
                row = {
                    "Model": model_name,
                    "Dataset": dataset,
                    "Pass_K": pass_k,
                    "Total_Samples": stats["total"],
                    "Total_Format_Errors": stats["format_error_total"],
                    "Format_Error_Rate": f"{stats['format_error_total'] / stats['total']:.2%}" if stats[
                                                                                                      'total'] > 0 else "0%",

                    "Collapse_In_Errors_Count": stats["fmt_err_collapse_count"],
                    "Collapse_In_Errors_Share": f"{collapse_share_in_error:.2%}",  # 核心诊断指标

                    "Avg_Len_Collapse_Err": int(avg_len_collapse),
                    "Avg_Len_Normal_Err": int(avg_len_normal),
                    "Length_Gap": int(len_gap)
                }
                results_data.append(row)

                # 写入日志
                report = (
                        f"【{model_name} | {dataset}】\n"
                        f"  - 样本总数: {stats['total']}\n"
                        f"  - 格式错误总数: {stats['format_error_total']} (占比 {stats['format_error_total'] / stats['total']:.1%})\n"
                        f"  - 错误归因: {collapse_share_in_error:.1%} 的格式错误源于模式坍塌\n"
                        f"  - 长度对比: 坍塌样本均长 {int(avg_len_collapse)} vs 普通错误均长 {int(avg_len_normal)}\n"
                        f"  >> 资源浪费剪刀差: +{int(len_gap)} Tokens\n"
                        + "-" * 50 + "\n"
                )
                log_f.write(report)
                log_f.flush()

    # 写入 CSV
    keys = [
        "Model", "Dataset", "Pass_K", "Total_Samples",
        "Total_Format_Errors", "Format_Error_Rate",
        "Collapse_In_Errors_Count", "Collapse_In_Errors_Share",
        "Avg_Len_Collapse_Err", "Avg_Len_Normal_Err", "Length_Gap"
    ]

    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results_data)

    print(f"\n分析完成！")
    print(f"统计 CSV 已保存至: {output_csv}")
    print(f"详细日志已保存至: {output_txt}")


if __name__ == "__main__":
    # 请确保此路径指向包含各模型子文件夹的 results 目录
    RESULTS_DIR = "./results"

    LOG_FILE = "./results/collapse_metrics_diagnosis.txt"
    CSV_FILE = "./results/collapse_metrics_stats.csv"

    if os.path.exists(RESULTS_DIR):
        run_analysis(RESULTS_DIR, LOG_FILE, CSV_FILE)
    else:
        print(f"错误: 未找到目录 {RESULTS_DIR}")