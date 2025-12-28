import os

import matplotlib.pyplot as plt
import numpy as np


# def plot_pass_k_curve(results_list, task_name: str, show_k_list=[1, 2, 4, 8]):
#     """
#     Args:
#         results_list: 包含多个结果字典的列表
#         task_name:   任务名
#         show_k_list: 列表，指定要展示哪些 k 值，例如 [1, 2, 4, 8]
#         model_name: 用于标题的模型名称
#     """
#     plt.figure(figsize=(10, 6))
#     plot_x = [0] + show_k_list
#
#     # 顺序：1.灰色/深红(Base) 2.蓝色(SFT) 3.绿色(RL) 4.紫色(Official)
#     # colors = ['#EF4444', '#3B82F6', '#10B981', '#8B5CF6', '#F59E0B']
#
#     # 推荐1: 柔和 Pastel 风（类似你图片中的风格，适合学术/技术报告）
#     # colors = ['#fd7f6f', '#7eb0d5', '#b2e061', '#bd7ebe']
#
#     # 推荐2: Matplotlib/Seaborn 默认风格（AI论文最常见，高对比）
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
#
#     # 推荐3: IBM Carbon 风格（企业报告常用，颜色盲友好）
#     # colors = ['#6929c4', '#1192e8', '#005d5d', '#9f1853']
#
#     # 推荐4: 更现代的深色友好方案（深浅主题都好看）
#     # colors = ['#8fb1e6', '#ffb482', '#a0d8a0', '#e69fc4']
#
#     # 确保 show_k_list 是排序的
#     show_k_list = sorted(show_k_list)
#     x_labels = ['0'] + [str(k) for k in show_k_list]
#     length = len(x_labels)
#     x_indices = range(length)  # 用索引作为实际绘图的 X 坐标
#     occupied_y = {i: [] for i in range(length)}
#     best_performance = 0.0
#     min_distance = 3.5
#
#     for idx, result in enumerate(results_list):
#         name = list(result.keys())[0]
#         full_data = list(result.values())[0]
#         best_performance = max(best_performance, np.max(full_data))
#
#         # 根据 show_k_list 提取数据 (k是从1开始的，索引从0开始)
#         display_data = [0.0]
#         for k in show_k_list:
#             if k <= len(full_data):
#                 display_data.append(full_data[k - 1] * 100)
#             else:
#                 print(f"Warning: k={k} 超出了 {name} 的数据范围")
#
#         # 绘制曲线
#         color = colors[idx % len(colors)]
#         plt.plot(x_indices, display_data,
#                  marker='o' if idx == 3 else 'v',
#                  linestyle='-',
#                  linewidth=2,
#                  markersize=8,
#                  label=name,
#                  color=color)
#         # 标注数值
#         for i, val in enumerate(display_data[1:], start=1):
#             current_y = val
#             for existing_y in occupied_y[i]:
#                 if abs(current_y - existing_y) < min_distance:
#                     # 如果重叠，尝试向上弹开，如果已到顶部则向下
#                     if current_y < 95:
#                         current_y += min_distance
#                     else:
#                         current_y -= min_distance
#             occupied_y[i].append(current_y)
#             if val is not None:
#                 plt.text(i, current_y, f'{val:.1f}%', color=color,
#                          fontsize=9, ha='center', fontweight='bold')
#
#     # 图表装饰
#     plt.title(f'Pass@k Performance Comparison({task_name})', fontsize=16)
#     plt.xlabel('Number of Attempts (k)', fontsize=14)
#     plt.ylabel('Accuracy (%)', fontsize=14)
#     plt.tick_params(labelsize=14)
#
#     # 设置横坐标刻度只显示列表中要求的数值
#     plt.xticks(x_indices, x_labels)
#
#     plt.ylim(0, 105)
#     plt.xlim(0, len(x_indices) - 0.5)
#     plt.grid(True, linestyle='--', alpha=0.5)
#     if best_performance > 0.5:
#         plt.legend(loc='lower right',fontsize=11)
#     else:
#         plt.legend(loc='upper right', fontsize=11)
#
#     plt.tight_layout()
#
#     # 保存图片
#     save_name = f'{task_name} Pass@{show_k_list[-1]}对比图.png'
#     save_path = os.path.join(os.getcwd(), 'asset/', save_name)
#     plt.savefig(save_path, dpi=300)
#     print(f"图表已保存为 {save_name}")
#     # plt.show()

def plot_pass_k_curve(results_list, task_name: str, show_k_list=[1, 2, 4, 8]):
    """
    绘制 Pass@k 性能对比图，优化了重叠标签避让、线型区分和视觉层次。
    """
    plt.figure(figsize=(11, 6.5), dpi=300)

    # 颜色保留你之前的推荐风格
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # 确保 show_k_list 是排序的
    show_k_list = sorted(show_k_list)
    x_labels = ['0'] + [str(k) for k in show_k_list]
    length = len(x_labels)
    x_indices = range(length)

    # 标签避让系统
    occupied_y = {i: [] for i in range(length)}
    best_performance = 0.0
    min_distance = 3.8  # 稍微增大间距以适应加粗字体

    for idx, result in enumerate(results_list):
        name = list(result.keys())[0]
        full_data = list(result.values())[0]

        if len(full_data) == 0: continue  # 跳过空数据
        best_performance = max(best_performance, np.max(full_data))

        display_data = [0.0]
        for k in show_k_list:
            if k <= len(full_data):
                display_data.append(full_data[k - 1] * 100)
            else:
                display_data.append(None)

        color = colors[idx % len(colors)]

        # 使用虚线和特殊标记官方 instruct 模型
        is_instruct = 'instruct' in name.lower()
        line_style = '--' if is_instruct else '-'
        marker_style = 'D' if is_instruct else ('o' if 'no_std_norm_grpo' in name.lower() else 'v')
        z_order = 5 if 'no_std_norm_grpo' in name.lower() or is_instruct else 3
        alpha = 1.0 if 'no_std_norm_grpo' in name.lower() or is_instruct else 0.7

        plt.plot(x_indices, display_data,
                 marker=marker_style,
                 linestyle=line_style,
                 linewidth=2.5 if not is_instruct else 2.0,
                 markersize=7,
                 label=name,
                 color=color,
                 alpha=alpha,
                 zorder=z_order)

        # 标签避让与边界保护
        for i, val in enumerate(display_data[1:], start=1):
            if val is None: continue

            current_y = val
            # 简单的碰撞检测迭代
            attempts = 0
            while attempts < 10:  # 最多尝试10次位移
                collision = False
                for existing_y in occupied_y[i]:
                    if abs(current_y - existing_y) < min_distance:
                        # 碰撞了：如果接近天花板则向下弹，否则向上弹
                        direction = -1 if current_y > 96 else 1
                        current_y += direction * 2.0
                        collision = True
                        break
                if not collision: break
                attempts += 1

            # 边界保护：确保不超出绘图区
            current_y = max(2, min(103, current_y))
            occupied_y[i].append(current_y)

            plt.text(i, current_y, f'{val:.1f}%', color=color,
                     fontsize=9, ha='center', va='bottom', fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.5))


    plt.title(f'Pass@k Performance Evolution: {task_name}', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Number of Attempts (k)', fontsize=13)
    plt.ylabel('Accuracy (%)', fontsize=13)

    plt.xticks(x_indices, x_labels, fontsize=12)
    plt.yticks(range(0, 110, 10), fontsize=12)

    plt.ylim(0, 108)
    plt.xlim(0, len(x_indices) - 0.5)

    # 使用柔和的网格线
    plt.grid(True, linestyle=':', alpha=0.6, color='#94A3B8')

    # 去除顶部和右侧边框
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 图例位置优化
    loc = 'lower right' if best_performance > 0.5 else 'upper left'
    plt.legend(loc=loc, fontsize=11, frameon=True, shadow=False, borderpad=1)

    plt.tight_layout()

    # 路径处理
    save_dir = os.path.join(os.getcwd(), 'asset/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_name = f'{task_name} Pass@{show_k_list[-1]}对比图.png'
    # save_name = f'{task_name} Pass@{show_k_list[-1]}对比图-无标注.png'
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=300)
    print(f"图表已成功保存至: {save_path}")

if __name__ == "__main__":
    baseline_full_pass8_result = {'baseline': (0.123, 0.216, 0.294, 0.348, 0.392, 0.428, 0.454, 0.482)}
    sft_full_pass8_result      = {'sft'     : (0.445, 0.587, 0.654, 0.695, 0.722, 0.743, 0.758, 0.770)}
    no_std_norm_grpo_full_pass8_result   = {'no_std_norm_grpo'  : (0.619, 0.731, 0.780, 0.803, 0.821, 0.833, 0.843, 0.851)}
    grpo_full_pass8_result = {'grpo': (0.598, 0.721, 0.769, 0.795, 0.814, 0.825, 0.838, 0.847)}

    baseline_500_pass64_result = {'baseline': (0.144, 0.262, 0.320, 0.414, 0.482, 0.524, 0.578, 0.614,
                                               0.642, 0.664, 0.700, 0.710, 0.714, 0.734, 0.750, 0.758,
                                               0.774, 0.784, 0.790, 0.800, 0.806, 0.814, 0.818, 0.822,
                                               0.826, 0.828, 0.832, 0.840, 0.846, 0.854, 0.858, 0.862,
                                               0.864, 0.868, 0.870, 0.874, 0.878, 0.878, 0.882, 0.886,
                                               0.890, 0.892, 0.896, 0.896, 0.896, 0.898, 0.902, 0.902,
                                               0.902, 0.904, 0.906, 0.906, 0.906, 0.906, 0.908, 0.908,
                                               0.910, 0.910, 0.910, 0.912, 0.912, 0.914, 0.914, 0.916)}

    sft_500_pass64_result      = {'sft'     : (0.668, 0.816, 0.874, 0.896, 0.914, 0.926, 0.936, 0.942,
                                               0.944, 0.950, 0.952, 0.952, 0.954, 0.956, 0.956, 0.958,
                                               0.960, 0.962, 0.962, 0.964, 0.968, 0.970, 0.970, 0.970,
                                               0.970, 0.970, 0.972, 0.972, 0.974, 0.974, 0.976, 0.976,
                                               0.976, 0.976, 0.976, 0.976, 0.976, 0.978, 0.978, 0.978,
                                               0.978, 0.980, 0.982, 0.982, 0.982, 0.982, 0.982, 0.982,
                                               0.982, 0.982, 0.982, 0.982, 0.982, 0.982, 0.982, 0.982,
                                               0.982, 0.982, 0.982, 0.982, 0.982, 0.982, 0.982, 0.982)}

    no_std_norm_grpo_500_pass64_result   = {'no_std_norm_grpo'  :
                                              (0.834, 0.916, 0.952, 0.956, 0.962, 0.966, 0.966, 0.968,
                                               0.968, 0.972, 0.974, 0.974, 0.974, 0.974, 0.974, 0.974,
                                               0.974, 0.976, 0.976, 0.976, 0.976, 0.976, 0.976, 0.976,
                                               0.976, 0.976, 0.976, 0.976, 0.976, 0.978, 0.978, 0.980,
                                               0.980, 0.980, 0.980, 0.980, 0.980, 0.980, 0.980, 0.980,
                                               0.980, 0.980, 0.980, 0.982, 0.982, 0.982, 0.982, 0.984,
                                               0.984, 0.984, 0.984, 0.986, 0.986, 0.988, 0.988, 0.988,
                                               0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 0.988)}

    baseline_aime24_pass64_result = {'baseline':(0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                                                 0.000, 0.000, 0.033, 0.033, 0.033, 0.033, 0.033, 0.067,
                                                 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067,
                                                 0.067, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100,
                                                 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100,
                                                 0.133, 0.133, 0.133, 0.133, 0.133, 0.133, 0.133, 0.133,
                                                 0.133, 0.133, 0.133, 0.133, 0.133, 0.133, 0.133, 0.133,
                                                 0.133, 0.133, 0.133, 0.133, 0.133, 0.133, 0.133, 0.133)}


    sft_aime24_pass64_result     = {'sft'     : (0.000, 0.000, 0.033, 0.033, 0.033, 0.100, 0.100, 0.100,
                                                 0.100, 0.133, 0.133, 0.133, 0.133, 0.133, 0.167, 0.167,
                                                 0.167, 0.167, 0.167, 0.167, 0.167, 0.167, 0.167, 0.167,
                                                 0.167, 0.167, 0.167, 0.167, 0.167, 0.167, 0.167, 0.167,
                                                 0.167, 0.167, 0.167, 0.167, 0.167, 0.167, 0.167, 0.167,
                                                 0.167, 0.167, 0.167, 0.167, 0.200, 0.233, 0.233, 0.233,
                                                 0.233, 0.233, 0.233, 0.233, 0.233, 0.233, 0.233, 0.233,
                                                 0.233, 0.233, 0.267, 0.267, 0.267, 0.267, 0.267, 0.267)}


    no_std_norm_grpo_aime24_pass64_result  = {'no_std_norm_grpo':
                                                (0.033, 0.033, 0.067, 0.133, 0.133, 0.167, 0.167, 0.200,
                                                 0.200, 0.200, 0.233, 0.233, 0.267, 0.300, 0.333, 0.367,
                                                 0.367, 0.367, 0.367, 0.367, 0.367, 0.367, 0.367, 0.367,
                                                 0.367, 0.367, 0.367, 0.367, 0.400, 0.400, 0.400, 0.400,
                                                 0.400, 0.400, 0.400, 0.400, 0.400, 0.400, 0.400, 0.400,
                                                 0.400, 0.400, 0.400, 0.400, 0.400, 0.400, 0.400, 0.400,
                                                 0.400, 0.400, 0.400, 0.400, 0.400, 0.400, 0.400, 0.400,
                                                 0.400, 0.400, 0.400, 0.400, 0.400, 0.400, 0.400, 0.400)}


    baseline_aime25_pass64_result= {'baseline': (0.033, 0.033, 0.033, 0.033, 0.033, 0.033, 0.033, 0.033,
                                                 0.033, 0.033, 0.033, 0.033, 0.033, 0.033, 0.033, 0.033,
                                                 0.033, 0.033, 0.033, 0.033, 0.033, 0.033, 0.033, 0.033,
                                                 0.033, 0.033, 0.033, 0.033, 0.033, 0.033, 0.033, 0.033,
                                                 0.033, 0.033, 0.033, 0.033, 0.033, 0.033, 0.033, 0.033,
                                                 0.033, 0.033, 0.033, 0.033, 0.033, 0.033, 0.033, 0.033,
                                                 0.033, 0.033, 0.033, 0.033, 0.033, 0.033, 0.033, 0.033,
                                                 0.033, 0.033, 0.033, 0.033, 0.033, 0.033, 0.033, 0.033)}


    sft_aime25_pass64_result     = {'sft'     : (0.000, 0.033, 0.067, 0.067, 0.100, 0.100, 0.100, 0.100,
                                                 0.100, 0.133, 0.133, 0.133, 0.167, 0.167, 0.167, 0.167,
                                                 0.167, 0.167, 0.167, 0.167, 0.200, 0.200, 0.200, 0.200,
                                                 0.200, 0.200, 0.200, 0.200, 0.200, 0.200, 0.200, 0.200,
                                                 0.200, 0.200, 0.200, 0.200, 0.200, 0.200, 0.200, 0.200,
                                                 0.200, 0.200, 0.233, 0.233, 0.233, 0.233, 0.233, 0.233,
                                                 0.233, 0.233, 0.233, 0.233, 0.233, 0.233, 0.233, 0.233,
                                                 0.233, 0.233, 0.233, 0.233, 0.233, 0.233, 0.233, 0.233)}

    no_std_norm_grpo_aime25_pass64_result = {'no_std_norm_grpo'   :
                                                (0.033, 0.100, 0.100, 0.100, 0.167, 0.167, 0.200, 0.200,
                                                 0.200, 0.200, 0.233, 0.233, 0.233, 0.233, 0.233, 0.233,
                                                 0.233, 0.233, 0.233, 0.233, 0.233, 0.233, 0.233, 0.233,
                                                 0.267, 0.267, 0.267, 0.267, 0.267, 0.267, 0.267, 0.267,
                                                 0.267, 0.267, 0.267, 0.267, 0.267, 0.267, 0.267, 0.267,
                                                 0.267, 0.267, 0.267, 0.267, 0.267, 0.267, 0.267, 0.267,
                                                 0.267, 0.267, 0.267, 0.267, 0.267, 0.267, 0.267, 0.267,
                                                 0.267, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300)}

    baseline_AMC12_pass64_result = {'baseline': (0.024, 0.084, 0.108, 0.108, 0.120, 0.133, 0.145, 0.193,
                                                 0.205, 0.205, 0.205, 0.253, 0.253, 0.301, 0.301, 0.313,
                                                 0.325, 0.337, 0.349, 0.349, 0.349, 0.349, 0.361, 0.373,
                                                 0.373, 0.398, 0.398, 0.398, 0.398, 0.398, 0.422, 0.434,
                                                 0.434, 0.434, 0.434, 0.434, 0.434, 0.458, 0.482, 0.482,
                                                 0.494, 0.494, 0.494, 0.506, 0.506, 0.518, 0.518, 0.518,
                                                 0.518, 0.518, 0.518, 0.518, 0.530, 0.542, 0.542, 0.542,
                                                 0.542, 0.542, 0.554, 0.554, 0.554, 0.566, 0.566, 0.566)}

    sft_AMC12_pass64_result      = {'sft'     : (0.217, 0.253, 0.325, 0.349, 0.361, 0.386, 0.434, 0.482,
                                                 0.482, 0.506, 0.506, 0.530, 0.530, 0.530, 0.530, 0.542,
                                                 0.578, 0.590, 0.590, 0.602, 0.602, 0.602, 0.602, 0.602,
                                                 0.602, 0.602, 0.614, 0.614, 0.627, 0.651, 0.651, 0.651,
                                                 0.651, 0.651, 0.651, 0.651, 0.663, 0.663, 0.663, 0.663,
                                                 0.663, 0.663, 0.663, 0.663, 0.687, 0.699, 0.699, 0.699,
                                                 0.699, 0.699, 0.699, 0.699, 0.699, 0.711, 0.711, 0.723,
                                                 0.723, 0.735, 0.735, 0.735, 0.735, 0.735, 0.735, 0.735)}

    no_std_norm_grpo_AMC12_pass64_result   = {'no_std_norm_grpo'  :
                                                (0.325, 0.398, 0.470, 0.530, 0.554, 0.566, 0.566, 0.590,
                                                 0.602, 0.614, 0.627, 0.627, 0.651, 0.663, 0.675, 0.675,
                                                 0.675, 0.675, 0.687, 0.699, 0.699, 0.699, 0.699, 0.699,
                                                 0.699, 0.723, 0.723, 0.723, 0.723, 0.723, 0.723, 0.723,
                                                 0.723, 0.723, 0.723, 0.723, 0.723, 0.723, 0.735, 0.735,
                                                 0.735, 0.735, 0.735, 0.735, 0.735, 0.735, 0.735, 0.735,
                                                 0.735, 0.747, 0.747, 0.747, 0.747, 0.747, 0.747, 0.759,
                                                 0.759, 0.759, 0.759, 0.759, 0.759, 0.759, 0.759, 0.759)}
    instruct_math_pass8_result = {'instruct':   (0.749, 0.809, 0.833, 0.849, 0.861, 0.872, 0.877, 0.882)}

    instruct_500_pass64_result = {'instruct':   (0.920, 0.954, 0.968, 0.972, 0.972, 0.976, 0.976, 0.978,
                                                 0.980, 0.982, 0.982, 0.982, 0.982, 0.982, 0.982, 0.982,
                                                 0.982, 0.982, 0.982, 0.982, 0.982, 0.984, 0.984, 0.984,
                                                 0.984, 0.984, 0.984, 0.984, 0.984, 0.984, 0.984, 0.984,
                                                 0.984, 0.984, 0.986, 0.986, 0.988, 0.990, 0.990, 0.990,
                                                 0.990, 0.990, 0.990, 0.990, 0.990, 0.990, 0.990, 0.990,
                                                 0.990, 0.990, 0.990, 0.990, 0.990, 0.990, 0.990, 0.990,
                                                 0.990, 0.990, 0.990, 0.990, 0.990, 0.990, 0.990, 0.990)}

    instruct_AMC_12_pass64_result = {'instruct': (0.446, 0.542, 0.639, 0.663, 0.675, 0.699, 0.735, 0.735,
                                                  0.747, 0.747, 0.747, 0.747, 0.759, 0.771, 0.771, 0.771,
                                                  0.771, 0.783, 0.807, 0.807, 0.819, 0.819, 0.819, 0.819,
                                                  0.819, 0.819, 0.819, 0.819, 0.819, 0.819, 0.831, 0.831,
                                                  0.831, 0.831, 0.843, 0.843, 0.843, 0.843, 0.843, 0.843,
                                                  0.843, 0.843, 0.843, 0.843, 0.843, 0.843, 0.843, 0.843,
                                                  0.843, 0.843, 0.843, 0.843, 0.843, 0.843, 0.843, 0.843,
                                                  0.843, 0.843, 0.843, 0.843, 0.843, 0.843, 0.843, 0.843)}

    instruct_aime24_pass64_result = {'instruct': (0.067, 0.100, 0.100, 0.133, 0.167, 0.200, 0.233, 0.233,
                                                  0.233, 0.267, 0.267, 0.300, 0.300, 0.300, 0.333, 0.333,
                                                  0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333,
                                                  0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.367, 0.367,
                                                  0.367, 0.367, 0.367, 0.367, 0.367, 0.367, 0.367, 0.400,
                                                  0.400, 0.400, 0.400, 0.400, 0.400, 0.400, 0.400, 0.400,
                                                  0.400, 0.433, 0.433, 0.433, 0.433, 0.433, 0.433, 0.433,
                                                  0.433, 0.433, 0.433, 0.433, 0.433, 0.467, 0.467, 0.467)}

    instruct_aime25_pass64_result = {'instruct': (0.067, 0.067, 0.100, 0.100, 0.100, 0.133, 0.133, 0.167,
                                                  0.200, 0.200, 0.233, 0.233, 0.233, 0.233, 0.267, 0.267,
                                                  0.267, 0.300, 0.333, 0.333, 0.333, 0.333, 0.333, 0.367,
                                                  0.367, 0.400, 0.400, 0.400, 0.400, 0.400, 0.400, 0.400,
                                                  0.400, 0.400, 0.400, 0.400, 0.400, 0.400, 0.400, 0.400,
                                                  0.400, 0.400, 0.400, 0.400, 0.400, 0.400, 0.400, 0.400,
                                                  0.400, 0.400, 0.400, 0.400, 0.400, 0.400, 0.400, 0.400,
                                                  0.400, 0.400, 0.400, 0.400, 0.400, 0.400, 0.400, 0.400)}

    grpo_500_pass64_result        = {'grpo':     (0.808, 0.920, 0.942, 0.952, 0.958, 0.966, 0.972, 0.972,
                                                  0.980, 0.980, 0.982, 0.984, 0.984, 0.984, 0.986, 0.988,
                                                  0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 0.990,
                                                  0.990, 0.990, 0.990, 0.990, 0.992, 0.992, 0.992, 0.992,
                                                  0.992, 0.992, 0.992, 0.992, 0.992, 0.992, 0.992, 0.994,
                                                  0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994,
                                                  0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994,
                                                  0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994)}

    grpo_AMC12_pass64_result      = {'grpo':      (0.361, 0.446, 0.482, 0.530, 0.554, 0.590, 0.590, 0.602,
                                                   0.614, 0.639, 0.651, 0.651, 0.651, 0.651, 0.651, 0.651,
                                                   0.651, 0.651, 0.663, 0.663, 0.675, 0.687, 0.687, 0.687,
                                                   0.687, 0.699, 0.699, 0.711, 0.723, 0.735, 0.735, 0.735,
                                                   0.735, 0.735, 0.735, 0.735, 0.735, 0.735, 0.735, 0.735,
                                                   0.735, 0.735, 0.735, 0.735, 0.735, 0.735, 0.747, 0.759,
                                                   0.759, 0.759, 0.759, 0.759, 0.759, 0.759, 0.759, 0.759,
                                                   0.759, 0.759, 0.759, 0.759, 0.759, 0.759, 0.759, 0.759)}

    grpo_aime24_pass64_result     = {'grpo':      (0.133, 0.167, 0.167, 0.167, 0.167, 0.167, 0.200, 0.200,
                                                   0.200, 0.200, 0.200, 0.200, 0.200, 0.200, 0.200, 0.200,
                                                   0.200, 0.200, 0.267, 0.267, 0.267, 0.267, 0.267, 0.267,
                                                   0.267, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300,
                                                   0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300,
                                                   0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300,
                                                   0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300,
                                                   0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300)}

    grpo_aime25_pass64_result     = {'grpo':       (0.033, 0.067, 0.067, 0.067, 0.100, 0.167, 0.167, 0.167,
                                                    0.167, 0.167, 0.167, 0.167, 0.167, 0.233, 0.233, 0.233,
                                                    0.233, 0.233, 0.233, 0.267, 0.267, 0.267, 0.267, 0.267,
                                                    0.267, 0.267, 0.267, 0.267, 0.267, 0.267, 0.267, 0.267,
                                                    0.267, 0.267, 0.267, 0.267, 0.267, 0.267, 0.267, 0.267,
                                                    0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300,
                                                    0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300,
                                                    0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300)}



    math_full_pass8_results  = [baseline_full_pass8_result, sft_full_pass8_result, no_std_norm_grpo_full_pass8_result, grpo_full_pass8_result, instruct_math_pass8_result]
    math500_pass64_results   = [baseline_500_pass64_result, sft_500_pass64_result, no_std_norm_grpo_500_pass64_result, grpo_500_pass64_result, instruct_500_pass64_result]
    amc12_pass64_results     = [baseline_AMC12_pass64_result, sft_AMC12_pass64_result, no_std_norm_grpo_AMC12_pass64_result, grpo_AMC12_pass64_result, instruct_AMC_12_pass64_result]
    aime_2025_pass64_results = [baseline_aime25_pass64_result, sft_aime25_pass64_result, no_std_norm_grpo_aime25_pass64_result, grpo_aime25_pass64_result, instruct_aime25_pass64_result]
    aime_2024_pass64_results = [baseline_aime24_pass64_result, sft_aime24_pass64_result, no_std_norm_grpo_aime24_pass64_result, grpo_aime24_pass64_result, instruct_aime24_pass64_result]




    pass64_target_k = [1, 2, 4, 8, 16, 32, 64]
    pass16_target_k = pass64_target_k[:-2]
    pass8_target_k = pass64_target_k[:-3]

    # MATH Pass@8
    plot_pass_k_curve(math_full_pass8_results, 'MATH-test', pass8_target_k)

    # MATH-500 Pass@64
    plot_pass_k_curve(math500_pass64_results,'MATH-500', pass64_target_k)

    # AMC12 Pass@64
    plot_pass_k_curve(amc12_pass64_results, 'AMC12', pass64_target_k)

    # AIME-2024 Pass@64
    plot_pass_k_curve(aime_2024_pass64_results, 'AIME-2024', pass64_target_k)

    # AIME-2025 Pass@64
    plot_pass_k_curve(aime_2025_pass64_results, 'AIME-2025', pass64_target_k)


