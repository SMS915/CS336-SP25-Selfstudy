import matplotlib.pyplot as plt


def plot_pass_k_curve(results_list, show_k_list=[1, 2, 4, 8], model_name="Qwen-2.5-Math-1.5B"):
    """
    Args:
        results_list: 包含多个结果字典的列表
        show_k_list: 列表，指定要展示哪些 k 值，例如 [1, 2, 4, 8]
        model_name: 用于标题的模型名称
    """
    plt.figure(figsize=(10, 6))
    plot_x = [0] + show_k_list
    # 定义颜色
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    # 确保 show_k_list 是排序的
    show_k_list = sorted(show_k_list)
    x_labels = ['0'] + [str(k) for k in show_k_list]
    x_indices = range(len(x_labels))  # 用索引作为实际绘图的 X 坐标

    for idx, result in enumerate(results_list):
        name = list(result.keys())[0]
        full_data = list(result.values())[0]

        # 根据 show_k_list 提取数据 (k是从1开始的，索引从0开始)
        display_data = [0.0]
        for k in show_k_list:
            if k <= len(full_data):
                display_data.append(full_data[k - 1] * 100)
            else:
                print(f"Warning: k={k} 超出了 {name} 的数据范围")

        # 绘制曲线
        color = colors[idx % len(colors)]
        plt.plot(x_indices, display_data, marker='v', linestyle='-',
                 linewidth=2, markersize=8, label=name, color=color)
        # 标注数值
        for i, val in enumerate(display_data[1:], start=1):
            if val is not None:
                plt.text(i, val + 2.5, f'{val:.1f}%', color=color,
                         fontsize=9, ha='center', fontweight='bold')

    # 图表装饰
    plt.title('Pass@k Performance Comparison(MATH-500)', fontsize=16)
    plt.xlabel('Number of Attempts (k)', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.tick_params(labelsize=14)

    # 设置横坐标刻度只显示列表中要求的数值
    plt.xticks(x_indices, x_labels)

    plt.ylim(0, 105)
    plt.xlim(0, len(x_indices) - 0.5)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize=11)

    plt.tight_layout()

    # 保存图片
    save_name = 'Math-500 Pass@k对比图.png'
    plt.savefig(save_name, dpi=300)
    print(f"图表已保存为 {save_name}")
    plt.show()

if __name__ == "__main__":
    baseline_full_pass8_result = {'baseline': (0.123, 0.2156, 0.2942, 0.3478, 0.3916, 0.4282, 0.4542, 0.4820)}
    sft_full_pass8_result      = {'sft':      (0.4452, 0.5872, 0.6542, 0.6952, 0.7224, 0.7426, 0.758, 0.7702)}
    drgrpo_full_pass8_result   = {'drgrpo'  : (0.619, 0.7314, 0.7802, 0.8034, 0.8214, 0.8332, 0.8432, 0.8514)}

    baseline_500_pass64_result = {'baseline': (0.144, 0.262, 0.320, 0.414, 0.482, 0.524, 0.578, 0.614,
                                               0.642, 0.664, 0.700, 0.710, 0.714, 0.734, 0.750, 0.758,
                                               0.774, 0.784, 0.790, 0.800, 0.806, 0.814, 0.818, 0.822,
                                               0.826, 0.828, 0.832, 0.840, 0.846, 0.854, 0.858, 0.862,
                                               0.864, 0.868, 0.870, 0.874, 0.878, 0.878, 0.882, 0.886,
                                               0.890, 0.892, 0.896, 0.896, 0.896, 0.898, 0.902, 0.902,
                                               0.902, 0.904, 0.906, 0.906, 0.906, 0.906, 0.908, 0.908,
                                               0.910, 0.910, 0.910, 0.912, 0.912, 0.914, 0.914, 0.916)}

    sft_500_pass64_result      = {'sft':      (0.668, 0.816, 0.874, 0.896, 0.914, 0.926, 0.936, 0.942,
                                               0.944, 0.950, 0.952, 0.952, 0.954, 0.956, 0.956, 0.958,
                                               0.960, 0.962, 0.962, 0.964, 0.968, 0.970, 0.970, 0.970,
                                               0.970, 0.970, 0.972, 0.972, 0.974, 0.974, 0.976, 0.976,
                                               0.976, 0.976, 0.976, 0.976, 0.976, 0.978, 0.978, 0.978,
                                               0.978, 0.980, 0.982, 0.982, 0.982, 0.982, 0.982, 0.982,
                                               0.982, 0.982, 0.982, 0.982, 0.982, 0.982, 0.982, 0.982,
                                               0.982, 0.982, 0.982, 0.982, 0.982, 0.982, 0.982, 0.982)}

    drgrpo_500_pass64_result   = {'drgrpo':   (0.834, 0.916, 0.952, 0.956, 0.962, 0.966, 0.966, 0.968,
                                               0.968, 0.972, 0.974, 0.974, 0.974, 0.974, 0.974, 0.974,
                                               0.974, 0.976, 0.976, 0.976, 0.976, 0.976, 0.976, 0.976,
                                               0.976, 0.976, 0.976, 0.976, 0.976, 0.978, 0.978, 0.980,
                                               0.980, 0.980, 0.980, 0.980, 0.980, 0.980, 0.980, 0.980,
                                               0.980, 0.980, 0.980, 0.982, 0.982, 0.982, 0.982, 0.984,
                                               0.984, 0.984, 0.984, 0.986, 0.986, 0.988, 0.988, 0.988,
                                               0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 0.988)}
    # results = [baseline_full_pass8_result, sft_full_pass8_result, drgrpo_full_pass8_result]
    results = [baseline_500_pass64_result, sft_500_pass64_result, drgrpo_500_pass64_result]
    target_k = [1, 2, 4, 8, 16, 32, 64]
    plot_pass_k_curve(results, show_k_list=target_k)