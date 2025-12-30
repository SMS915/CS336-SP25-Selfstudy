import matplotlib.pyplot as plt
import numpy as np

def plot_mask_strategy_comparison():
    # 数据填充自项目统计
    labels = ['Mask <think> (v4.1)', 'No-Mask <think> (v4.2)']
    accuracy = [42.30, 44.10]
    format_error = [46.16, 42.16]
    avg_len_success = [1187.4, 1082.4]
    avg_len_error = [3825.1, 3567.3]

    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 图1：准确率与格式错误率对比
    rects1 = ax1.bar(x - width/2, accuracy, width, label='Accuracy (Pass@1)', color='#4c72b0')
    rects2 = ax1.bar(x + width/2, format_error, width, label='Format Error Rate', color='#c44e52')
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Accuracy vs Format Error by Mask Strategy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # 图2：成功与失败样本的平均长度对比
    rects3 = ax2.bar(x - width/2, avg_len_success, width, label='Success Sample Length', color='#55a868')
    rects4 = ax2.bar(x + width/2, avg_len_error, width, label='Error Sample Length', color='#dd8452')
    ax2.set_ylabel('Avg Tokens')
    ax2.set_title('Avg Generation Length: Success vs Failure')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.axhline(y=4096, color='r', linestyle='--', label='Max Window (4k)')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('mask_strategy_analysis.png', dpi=300)
    print("已生成消融实验对比图：mask_strategy_analysis.png")

if __name__ == "__main__":
    plot_mask_strategy_comparison()