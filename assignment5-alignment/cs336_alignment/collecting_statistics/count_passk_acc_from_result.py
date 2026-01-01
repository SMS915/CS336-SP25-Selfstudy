import json
import argparse
import os
from collections import Counter

def analyze_pass_k_results(file_path):
    """
    自动识别最大 K 值并统计 Pass@k，考虑空缺尝试的情况
    """
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 {file_path}")
        return

    total_problems = 0
    # 记录每个 attempt_id 对应的首次解决数
    solve_at_attempt_counts = Counter()
    
    # 用来确定该文件实际跑到了 Pass@几
    max_k_detected = 0

    print(f"正在分析文件: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError: continue
            
            total_problems += 1
            metrics = data.get("metrics", {})
            # 在早停逻辑中，未解出的题 attempt_id 通常等于设定的最大 K
            # 已解出的题则记录解出的那一刻的 attempt_id
            attempt_id = data.get("attempt_id", 1)
            
            # 自动探测文件的最大 K
            if attempt_id > max_k_detected:
                max_k_detected = attempt_id
            
            # 只有当 reward 为 1.0 时才算解决
            reward = metrics.get("reward", metrics.get("answer_reward", 0.0))
            if reward == 1.0:
                solve_at_attempt_counts[attempt_id] += 1

    if total_problems == 0:
        print("文件中没有有效数据。")
        return

    print("\n" + "="*60)
    print(f"项目统计报告 | 总题目数: {total_problems} | 探测到最大 K: {max_k_detected}")
    print("="*60)
    print(f"{'指标':<12} | {'新增解决':<10} | {'累计解决':<10} | {'准确率 (Pass@k)':<15}")
    print("-" * 60)

    cumulative_solved = 0
    results_for_plot = [] # 方便你后续复制到绘图脚本中

    # 从 1 循环到探测出的最大 K，确保中间空缺的 k 也能被统计
    for k in range(1, max_k_detected + 1):
        newly_solved = solve_at_attempt_counts.get(k, 0)
        cumulative_solved += newly_solved
        pass_k_acc = cumulative_solved / total_problems
        results_for_plot.append(round(pass_k_acc, 4))
        
        # 只打印关键节点或每隔几步打印，如果 K 很大（如 64）建议对齐打印
        print(f"Pass@{k:<8} | {newly_solved:<10} | {cumulative_solved:<10} | {pass_k_acc:.2%}")

    formatted_items = []
    for i, x in enumerate(results_for_plot):
        val = f"{x:.3f}"
        # 如果是最后一个元素，不加逗号
        if i == len(results_for_plot) - 1:
            formatted_items.append(val)
        # 如果是每行的第 8 个元素（索引 7, 15, 23...），加逗号并换行
        elif i % 8 == 7:
            formatted_items.append(val + ",\n ")
        # 其他情况，加逗号和空格
        else:
            formatted_items.append(val + ", ")

    # 使用空字符串 join，因为逗号已经根据逻辑放好了
    formatted_str_tuple = "".join(formatted_items)

    print("-" * 60)
    print("\n[绘图专用数据格式]:")
    print(f"'{os.path.basename(file_path)}':\n ({formatted_str_tuple})")
    print("="*60 + "\n")
    # for i,acc in enumerate(passk_acc_list):
    #     print(f'{acc:.3f}', end=', ') if i != len(passk_acc_list) - 1 else print(f'{acc:.3f}',  end=']\n')

def main():
    parser = argparse.ArgumentParser(description="统计 Pass@k 结果文件")
    parser.add_argument("input", type=str, help="jsonl 结果文件路径")
    args = parser.parse_args()

    analyze_pass_k_results(args.input)

if __name__ == "__main__":
    main()