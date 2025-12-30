import json
import os
import csv
from tqdm import tqdm


def calculate_lcp_ratio(text, min_pattern_len=10):
    """
    优化后的重复性检测：使用集合记录模式，提升检测速度。
    """
    if not text or len(text) < min_pattern_len:
        return 0.0

    seen_patterns = set()
    repeated_chars = 0

    # 步长可适当调整以提升速度
    for i in range(len(text) - min_pattern_len):
        pattern = text[i: i + min_pattern_len]
        if pattern in seen_patterns:
            repeated_chars += 1
        else:
            seen_patterns.add(pattern)

    return repeated_chars / len(text)


def parse_filename(filename):
    """
    解析逻辑: {模型名}_{数据集}_pass_{k}.jsonl
    """
    if not filename.endswith(".jsonl"):
        return None, None, None

    name_part = filename.replace(".jsonl", "")
    # 寻找最后一个 _pass_ 标记来区分数据集和 K 值
    try:
        if "_pass_" in name_part:
            base_info, pass_k = name_part.split("_pass_")
            # 模型名通常在第一个下划线前，或者根据你的目录结构，直接从父目录获取模型名
            # 这里我们返回 base_info 供后续根据目录名二次修正
            return base_info, pass_k
    except:
        pass
    return None, None


def run_analysis(input_root, output_txt, output_csv):
    results_data = []

    # 准备写入 TXT 日志
    with open(output_txt, 'w', encoding='utf-8') as log_f:
        log_f.write("=" * 80 + "\n")
        log_f.write(f"模型生成模式坍缩 (Mode Collapse) 分析报告\n源目录: {input_root}\n")
        log_f.write("=" * 80 + "\n\n")

        # 遍历子文件夹 (baseline, drgrpo, sft 等)
        subdirs = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]

        for model_name in subdirs:
            current_dir = os.path.join(input_root, model_name)
            files = [f for f in os.listdir(current_dir) if f.endswith(".jsonl")]

            for filename in files:
                file_path = os.path.join(current_dir, filename)
                base_info, pass_k = parse_filename(filename)

                # 提取数据集名称 (去掉前缀中的模型名)
                dataset = base_info.replace(f"{model_name}_", "") if base_info else "unknown"

                print(f"正在分析 [{model_name}] 的 {dataset} (Pass@{pass_k})...")

                collapse_count = 0
                total_samples = 0

                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            text = data.get("generated_text") or data.get("response") or ""
                            if not text: continue

                            total_samples += 1
                            ratio = calculate_lcp_ratio(text)

                            # 判定标准：重复率 > 0.2 且 长度 > 3000
                            if ratio > 0.2 and len(text) > 3000:
                                collapse_count += 1
                        except:
                            continue

                collapse_ratio = collapse_count / total_samples if total_samples > 0 else 0

                # 记录到列表用于 CSV
                results_data.append({
                    "Model": model_name,
                    "Dataset": dataset,
                    "Pass_K": pass_k,
                    "Total": total_samples,
                    "Collapse_Count": collapse_count,
                    "Collapse_Ratio": f"{collapse_ratio:.2%}"
                })

                # 写入 TXT 日志 (参考你提供的格式)
                report = (
                        f"【项目】: {model_name.upper()} | {dataset.upper()} | Pass@{pass_k}\n"
                        f"  - 对应文件: {filename}\n"
                        f"  - 样本总数: {total_samples}\n"
                        f"  - 模式坍缩数: {collapse_count}\n"
                        f"  - 坍缩占比: {collapse_ratio:.2%}\n"
                        + "-" * 40 + "\n"
                )
                log_f.write(report)
                log_f.flush()

    # 写入 CSV
    keys = ["Model", "Dataset", "Pass_K", "Total", "Collapse_Count", "Collapse_Ratio"]
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results_data)

    print(f"\n分析完成！")
    print(f"日志已保存至: {output_txt}")
    print(f"表格已保存至: {output_csv}")


if __name__ == "__main__":
    # 配置路径
    RESULTS_DIR = "./results"
    LOG_FILE = "./results/collapse_summary.txt"
    CSV_FILE = "./results/collapse_analysis.csv"

    if os.path.exists(RESULTS_DIR):
        run_analysis(RESULTS_DIR, LOG_FILE, CSV_FILE)
    else:
        print("未找到 results 文件夹")