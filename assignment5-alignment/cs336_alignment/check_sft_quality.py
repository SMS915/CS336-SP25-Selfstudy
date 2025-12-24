import json
import argparse
import sys
import numpy as np
from tqdm import tqdm
from typing import List, Dict

# 尝试导入 reward_fn，确保路径正确
try:
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
except ImportError:
    print("错误: 无法导入 r1_zero_reward_fn。请确保 cs336_alignment 文件夹在当前路径下或 PYTHONPATH 中。")
    sys.exit(1)

def load_data(file_path: str) -> List[Dict]:
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"警告: 第 {i+1} 行 JSON 解析失败，已跳过。")
    return examples

def evaluate_sft_dataset(file_path: str, output_path: str = None, max_samples: int = None):
    """
    评估 SFT 数据集的质量。
    逻辑完全复刻 evaluate_vllm，只是把 'generated_text' 换成了数据集里的 'response'。
    """
    print(f"正在加载数据: {file_path}")
    examples = load_data(file_path)
    
    if max_samples:
        examples = examples[:max_samples]
        print(f"仅评估前 {max_samples} 条样本")

    results = []
    correct_count = 0      # reward == 1.0 (格式对 + 答案对)
    ans_error_count = 0    # format_reward == 1.0 (格式对 + 答案错)
    format_error_count = 0 # format_reward == 0.0 (格式错)
    
    lengths = []
    failed_cases = [] # 记录反例用于分析

    print(f"开始评估 {len(examples)} 条数据...")
    
    for i, example in enumerate(tqdm(examples, desc="Evaluating")):
        # 1. 获取 SFT 数据中的回复
        # 兼容常见字段名: response, output, completion
        response = example.get('response') or example.get('output') or example.get('completion')
        
        if not response:
            print(f"警告: 第 {i} 条数据缺少 response 字段")
            continue

        # 2. 获取标准答案 (Ground Truth)
        # SFT 数据集通常包含 solution。如果没有，我们无法判断答案对错，只能判断格式。
        truth = example.get('solution') or example.get('answer')
        
        # 如果数据集中没有 solution，我们尝试从 response 中提取 boxed 内容作为临时 truth
        # 这样至少能跑通流程，虽然判断正确率意义不大，但能判断 "自洽性"
        if not truth:
            # 这是一个 fallback，如果你的数据只有 prompt/response
            truth = "UNKNOWN_TRUTH" 

        # =======================================================
        # 核心逻辑：完全复刻 run_evaluate 中的处理方式
        # =======================================================
        
        # 1. 长度统计
        lengths.append(len(response))
        
        # 2. 文本预处理 (补丁)
        # 这一步至关重要，因为 reward_fn 对空格敏感
        text_for_evaluate = response.replace("</think><answer>", "</think> <answer>")
        
        # 3. 调用 Reward Function
        metrics = r1_zero_reward_fn(text_for_evaluate, truth)
        
        # 4. 统计指标 (复刻 evaluate_vllm 的判断逻辑)
        if metrics.get("reward", 0.0) == 1.0:
            correct_count += 1
        elif metrics.get("format_reward", 0.0) == 1.0:
            ans_error_count += 1
            # 记录答案错误的例子（可能是 SFT 数据本身算错了，或者是 solution 字段有问题）
            if len(failed_cases) < 5:
                failed_cases.append({
                    "type": "Wrong Answer",
                    "response": response[-200:], 
                    "truth": truth
                })
        else:
            format_error_count += 1
            # 记录格式错误的例子
            if len(failed_cases) < 5:
                failed_cases.append({
                    "type": "Format Error",
                    "response": response[-200:], # 只看末尾
                    "truth": truth
                })

        # 保存详细结果
        results.append({
            "original_data": example,
            "metrics": metrics
        })

    # =======================================================
    # 输出统计报告
    # =======================================================
    total = len(results)
    if total == 0: return

    accuracy = correct_count / total
    format_rate = (correct_count + ans_error_count) / total
    avg_len = np.mean(lengths)

    print("\n" + "="*40)
    print("SFT 数据集质量评估报告")
    print("="*40)
    print(f"文件路径: {file_path}")
    print(f"样本总数: {total}")
    print("-" * 40)
    print(f"完全正确 (数据自洽): {accuracy:.2%}  <-- 重点关注")
    print(f"格式正确，答案不匹配: {ans_error_count / total:.2%}")
    print(f"格式错误 (Format Error): {format_error_count / total:.2%}")
    print("-" * 40)
    print(f"格式遵从率 (Format Rate): {format_rate:.2%}")
    print(f"平均字符长度: {avg_len:.2f}")
    print("="*40)

    if failed_cases:
        print("\n[问题样本示例 (Top 5)]")
        for case in failed_cases:
            print(f"类型: {case['type']}")
            print(f"Truth: {case['truth']}")
            print(f"Resp (尾部): {case['response']}")
            print("-" * 20)

    # 如果指定了输出路径，保存带 metrics 的新 jsonl
    if output_path:
        print(f"正在保存带评分的详细数据至: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for res in results:
                # 将 metrics 合并进原始数据
                data = res['original_data']
                data['eval_metrics'] = res['metrics']
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SFT dataset quality using reward_fn.")
    parser.add_argument("file_path", type=str, help="Path to SFT jsonl file.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save evaluated jsonl.")
    parser.add_argument("--max", type=int, default=None, help="Max samples to check.")
    
    args = parser.parse_args()
    
    evaluate_sft_dataset(args.file_path, args.output_path, args.max)