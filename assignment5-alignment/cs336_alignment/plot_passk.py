import matplotlib.pyplot as plt

def plot_pass_k_curve(acc_list, model_name="Qwen-2.5-Math-1.5B"):
    """
    Args:
        acc_list: 每一轮的累计准确率数组 [0.60, 0.65, 0.68, ...]
    """
    k_values = list(range(1, len(acc_list) + 1))
    
    # 转换为百分比
    percentages = [acc * 100 for acc in acc_list]

    plt.figure(figsize=(10, 6))
    
    # 绘制折线图
    plt.plot(k_values, percentages, marker='o', linestyle='-', color='#2c7bb6', 
             linewidth=2, markersize=8, label='Cumulative Pass@k')

    # 在每个点上标注具体数值
    for i, val in enumerate(percentages):
        plt.text(k_values[i], val + 0.5, f'{val:.1f}%', ha='center', fontsize=10)

    # 图表装饰
    plt.title(f'Pass@k Evaluation Trend - {model_name}', fontsize=14)
    plt.xlabel('Number of Attempts (k)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(k_values)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 保存图片
    plt.savefig('pass_k_trend.png', dpi=300)
    print("图表已保存为 pass_k_trend.png")
    plt.show()

if __name__ == "__main__":
    # 这里替换成你跑出来的真实数据
    # 假设 Pass@1 是 60%，Pass@8 达到了 78%
    my_data = [0.602, 0.654, 0.691, 0.720, 0.743, 0.761, 0.775, 0.784] 
    
    plot_pass_k_curve(my_data)