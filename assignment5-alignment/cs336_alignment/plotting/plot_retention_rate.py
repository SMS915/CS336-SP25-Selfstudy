import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.lines import Line2D
from adjustText import adjust_text
from matplotlib.ticker import ScalarFormatter
import matplotlib.patheffects as path_effects

# --- 1. 基础配置与同色系定义 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
main_blue = '#4A90E2'
dark_navy = '#1A3350'
light_bg_blue = '#F0F5FF'

# --- 2. 数据处理 ---
base_date = datetime(2024, 2, 1)
aime24_exam_date = datetime(2024, 2, 10)
aime24_days = (aime24_exam_date - base_date).days + 1

models_data = {
    # 早期基准与窗口期填补
    'DS-Math-7B-RL': {'params': 7, 'date_str': '24.02.05', 'greedy_r': 0.0, 'upper_r': 0.0},
    'Claude 3 Opus': {'params': 175, 'date_str': '24.03.04', 'greedy_r': 29.7, 'upper_r': 38.2},
    'GPT-4o-2024-05': {'params': 128, 'date_str': '24.05.13', 'greedy_r': 42.9, 'upper_r': 55.4},
    'Numina-72B-CoT': {'params': 72, 'date_str': '24.07.15', 'greedy_r': 0.0, 'upper_r': 98.6},
    'Llama-3.1-70B-I': {'params': 70, 'date_str': '24.07.23', 'greedy_r': 42.9, 'upper_r': 51.7},

    # 2024.09 推理革命期
    'OpenAI o1-mini': {'params': 32, 'date_str': '24.09.12', 'greedy_r': 77.4, 'upper_r': 71.5},
    'Qwen2.5-7B-I': {'params': 7, 'date_str': '24.09.19', 'greedy_r': 60.4, 'upper_r': 95.8},
    'Qwen2.5-32B-I': {'params': 32, 'date_str': '24.09.19', 'greedy_r': 180.2, 'upper_r': 104.1},
    'Qwen2.5-72B-I': {'params': 72, 'date_str': '24.09.19', 'greedy_r': 150.4, 'upper_r': 98.5},
    'Qwen2.5-Math-7B-I': {'params': 7, 'date_str': '24.09.19', 'greedy_r': 180.2, 'upper_r': 176.9},
    'Qwen2.5-Math-72B-I': {'params': 72, 'date_str': '24.09.19', 'greedy_r': 66.5, 'upper_r': 85.5},

    # 2024 年末与新架构期
    'Mistral-Large-2411': {'params': 123, 'date_str': '24.11.18', 'greedy_r': 100.0, 'upper_r': 127.9},
    'Skywork-o1': {'params': 72, 'date_str': '24.11.20', 'greedy_r': 119.8, 'upper_r': 141.2},
    'QWQ-32B-Preview': {'params': 32, 'date_str': '24.11.28', 'greedy_r': 60.1, 'upper_r': 81.4},
    'Llama-3.3-70B-I': {'params': 70, 'date_str': '24.12.06', 'greedy_r': 30.2, 'upper_r': 36.7},

    # 2025 年初最前沿
    'InternLM-3-8B-I': {'params': 8, 'date_str': '25.01.15', 'greedy_r': 119.8, 'upper_r': 147.8},
    'DS Distill 1.5B': {'params': 1.5, 'date_str': '25.01.22', 'greedy_r': 150.0, 'upper_r': 79.5},
    'DS Distill 32B': {'params': 32, 'date_str': '25.01.22', 'greedy_r': 75.1, 'upper_r': 84.1},
    'Qwen2.5-Max': {'params': 110, 'date_str': '25.01.28', 'greedy_r': 59.9, 'upper_r': 89.9},
}

print('[', end='')
for i, model_name in enumerate(models_data.keys()):
    if i != len(models_data) - 1:
        print(f"\'{model_name}\', ", end="")
    else:
        print(f"\'{model_name}\'", end="")
print(']')


print('[', end='')
for model_name, model_dict in models_data.items():
    if model_name != 'Qwen2.5-Max':
        print(f"{model_dict['greedy_r']}, ", end="")
    else:
        print(f"{model_dict['greedy_r']}", end="")
print(']')


print('[', end='')
for model_name, model_dict in models_data.items():
    if model_name != 'Qwen2.5-Max':
        print(f"{model_dict['upper_r']}, ", end="")
    else:
        print(f"{model_dict['upper_r']}", end="")
print(']')


def apply_y_repulsion(text_objects, threshold_y=8, iterations=10, force=0.5):
    """
    对标签施加垂直方向的斥力
    threshold_y: 触发斥力的最小纵向距离（单位通常是坐标系单位）
    iterations: 迭代次数，越多越分散
    force: 每次移动的力度
    """
    for _ in range(iterations):
        # 按 y 坐标排序，确保处理顺序
        text_objects.sort(key=lambda t: t.get_position()[1])

        for i in range(len(text_objects)):
            for j in range(i + 1, len(text_objects)):
                t1 = text_objects[i]
                t2 = text_objects[j]
                x1, y1 = t1.get_position()
                x2, y2 = t2.get_position()

                # 因为 X 轴是对数坐标，判断横向接近需转换到 Log 空间
                # 如果横向距离（视觉上）非常接近
                if abs(np.log10(x1) - np.log10(x2)) < 0.05:
                    dist_y = y2 - y1  # t2 在 t1 上方
                    if abs(dist_y) < threshold_y:
                        # 计算需要移动的距离
                        push = (threshold_y - abs(dist_y)) * force
                        t1.set_position((x1, y1 - push))
                        t2.set_position((x2, y2 + push))

for name, data in models_data.items():
    p = data['date_str'].split('.')
    data['days'] = (datetime(2000 + int(p[0]), int(p[1]), int(p[2])) - base_date).days + 1

min_s, max_s = 250, 1600

def get_s(p):
    # 计算逻辑保持不变，但限制结果最小值为 min_s (250)
    val = min_s + (max_s - min_s) * np.log(p / 7) / np.log(110 / 7)
    return max(80, val) # 即使是 1.5B 模型，也给它一个 80 的可见大小

# --- 3. 绘图主体 ---
fig, ax = plt.subplots(figsize=(22, 13))
ax.set_facecolor('#FCFCFC')

texts = []
for name, data in models_data.items():
    d, g, u = data['days'], data['greedy_r'], data['upper_r']
    s = get_s(data['params'])

    if g != u:
        safe_s = max(0, s)
        shrink_val = np.sqrt(safe_s) / 2

        ann = ax.annotate('', xy=(d, u), xytext=(d, g),
                          arrowprops=dict(arrowstyle='->,head_width=0.3,head_length=0.5',
                                          color=dark_navy, linewidth=2.5, alpha=0.8,
                                          shrinkA=shrink_val, shrinkB=shrink_val), zorder=8)
        # 关键美学优化：为箭头添加白色描边，以在复杂背景中脱颖而出
        ann.arrow_patch.set_path_effects([
            path_effects.Stroke(linewidth=4, foreground='white', alpha=0.7),
            path_effects.Normal()
        ])

    ax.scatter(d, g, s=s, color=main_blue, alpha=0.4, edgecolors=dark_navy, linewidth=0.8, zorder=10)

    disp = f"{name}\n({data['date_str']})"
    if name == 'DS-Math-7B-RL':
        ax.annotate(disp, xy=(d, g), xytext=(0, 15 + np.sqrt(s) / 2),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=12, fontweight='bold', color=dark_navy, zorder=15)
    else:
        t = ax.text(d, g, disp, fontsize=12, ha='center', va='center', fontweight='bold', color='#444')
        texts.append(t)

cluster_names = [
    'OpenAI o1-mini', 'DS Distill 32B', 'Qwen2.5-Math-72B-I',
    'QWQ-32B-Preview', 'Qwen2.5-Max', 'Llama-3.1-70B-I', 'Numina-72B-CoT'
]

cluster_points = []
for name in cluster_names:
    if name in models_data:
        d = models_data[name]['days']
        g = models_data[name]['greedy_r']
        # X轴是对数坐标，计算重心时必须取 log 才能保证视觉中心正确
        cluster_points.append([np.log10(d), g, name])

# --- 针对右侧五个点集群的重心放射算法 ---

# 1. 精确定义这五个点的名称
target_cluster = [
    'OpenAI o1-mini',
    'Qwen2.5-Math-72B-I',
    'QWQ-32B-Preview',
    'DS Distill 32B',
    'Qwen2.5-Max'
]

# 2. 提取这五个点的坐标并转入 Log-Linear 空间
cluster_coords = []
for name in target_cluster:
    d = models_data[name]['days']
    g = models_data[name]['greedy_r']
    cluster_coords.append([np.log10(d), g, name])

cluster_coords = np.array(cluster_coords, dtype=object)

# 3. 计算这五个点的重心 (Centroid)
centroid_log_x = np.mean(cluster_coords[:, 0].astype(float))
centroid_y = np.mean(cluster_coords[:, 1].astype(float))

# 4. 遍历 texts 对象，仅对这五个目标进行大幅度推移
repel_dist = 5 # 推离距离系数，数值越大标签离原点越远
x_stretch = 0.05  # X轴在Log空间的额外拉伸比例（防止标签在水平方向太挤）

for t in texts:
    full_text = t.get_text()
    model_name = full_text.split('\n')[0]

    if model_name in target_cluster:
        # 获取原始点坐标
        orig_days = models_data[model_name]['days']
        orig_greedy = models_data[model_name]['greedy_r']

        # 计算从重心指向点的向量 (在 Log-X 空间)
        vec_x = (np.log10(orig_days) - centroid_log_x) * 100  # 放大系数便于计算
        vec_y = orig_greedy - centroid_y

        # 向量归一化
        norm = np.sqrt(vec_x ** 2 + vec_y ** 2)
        if norm == 0: continue

        dir_x = vec_x / norm
        dir_y = vec_y / norm

        # 计算新坐标：在原有位置基础上，沿重心放射方向推离
        new_log_x = np.log10(orig_days) + dir_x * x_stretch
        new_y = orig_greedy + dir_y * repel_dist
        curr_x, curr_y = 10 ** new_log_x, new_y
        # 应用新坐标（X轴转回线性）
        if 'QWQ' in full_text:
            t.set_position((curr_x * 0.85, curr_y - 8))
        else:
            t.set_position((curr_x, curr_y))


        # 自动调整对齐方式，增强可读性
        t.set_ha('left' if dir_x > 0 else 'right')
        t.set_va('bottom' if dir_y > 0 else 'top')

        # 为这几个重点模型加个稍微深一点的颜色区分
        t.set_color('#222')

adjust_text(texts, only_move={'points': 'y', 'text': 'y'}, expand_points=(1.5, 1.8),force_text=0.1,
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.4))

# --- 4. 坐标轴与标签美化 ---
ax.set_xscale('log')
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_xticks([1, 10, 50, 100, 200, 400])
ax.set_xlim(1, 600)
ax.set_ylim(-35, 230)
ax.grid(True, which="both", ls=":", alpha=0.4, color='#AAA', zorder=0)

# **【信息补充】添加X轴和Y轴标题**
ax.set_xlabel('自 2024年2月1日 后的天数 (对数坐标)', fontsize=14, labelpad=15)
ax.set_ylabel('推理留存率 (%)', fontsize=14, labelpad=15)

# **【信息补充】添加辅助线标签**
ax.axvline(x=aime24_days, color='#800080', ls='--', lw=1.5, alpha=0.3)
ax.axhline(y=100, color='#D0021B', ls='--', lw=1, alpha=0.4)

ax.text(aime24_days, ax.get_ylim()[1], ' AIME 24 考试日\n (数据污染分界线)',
        color='purple', fontsize=12, ha='left', va='top', alpha=0.7)
ax.text(ax.get_xlim()[1], 100, ' 100% 理想稳定线 ',
        color='#D0021B', fontsize=12, ha='right', va='center',
        bbox=dict(facecolor='#FCFCFC', edgecolor='none', pad=0))


# --- 5. 视觉1:1校准图例 ---
def s_to_ms(s): return np.sqrt(s)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='7B 参数量',
           markerfacecolor=main_blue, alpha=0.4, markeredgecolor=dark_navy, markersize=s_to_ms(min_s)),
    Line2D([0], [0], marker='o', color='w', label='~70B 参数量',
           markerfacecolor=main_blue, alpha=0.4, markeredgecolor=dark_navy, markersize=s_to_ms(get_s(70))),
    Line2D([0], [0], marker='o', color='w', label='110B 参数量',
           markerfacecolor=main_blue, alpha=0.4, markeredgecolor=dark_navy, markersize=s_to_ms(max_s)),
    # **【信息补充】优化图例说明文字**
    Line2D([0], [0], color=dark_navy, lw=2, marker='^', markersize=6, label='演化路径 (起点: Greedy, 终点: Upper)')
]

leg = ax.legend(handles=legend_elements, loc='upper left', title='图例 (视觉1:1校准)',
                fontsize=10, title_fontsize=12, frameon=True, shadow=False,
                labelspacing=3.8, handletextpad=2.2, borderpad=1.5)
leg.get_frame().set_facecolor(light_bg_blue)
leg.get_frame().set_linewidth(0.5)

# **【信息补充】优化主标题**
ax.set_title('模型推理稳定性随时间演化与采样深度的诊断 (点大小代表参数量)',
             fontsize=18, pad=25, fontweight='bold', color=dark_navy)

plt.tight_layout()
plt.show()