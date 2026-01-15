import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 定义题目领域映射 (基于 AIME 2024 & 2025 典型题型)
# 1: 代数 (Algebra), 2: 数论 (Number Theory), 3: 几何 (Geometry), 4: 组合 (Combinatorics)
domain_map = {
    # AIME 2024
    2401: 1, 2407: 1, 2408: 1, 2413: 1, 2425: 1,  # 代数
    2419: 3, 2426: 3,                             # 几何
    2409: 4, 2410: 4, 2423: 4, 2424: 4, 2427: 4,  # 组合
    # AIME 2025
    2501: 1, 2503: 1, 2504: 1, 2508: 1,           # 代数
    2505: 2, 2506: 2, 2510: 2,                    # 数论
    2502: 3, 2516: 3, 2520: 3,                    # 几何
    2509: 4, 2517: 4                              # 组合
}

# 2. 原始数据定义
aime24_data = {'Baseline': [8, 9, 13, 19], 'SFT': [1, 7, 8, 10, 13, 23, 25, 27], 'DRGRPO': [1, 7, 8, 9, 10, 13, 19, 23, 24, 25, 26, 27]}
aime25_data = {'Baseline': [1], 'SFT': [1, 3, 4, 6, 9, 16, 17], 'DRGRPO': [1, 2, 3, 4, 5, 6, 8, 9, 16, 17, 20]}

def create_categorized_matrix(data, year_prefix, num_problems=30):
    """将数据转换为分类矩阵 (0:未解出, 1:代数, 2:数论, 3:几何, 4:组合)"""
    matrix = np.zeros((3, num_problems))
    models = ['Baseline', 'SFT', 'DRGRPO']
    for i, model in enumerate(models):
        for prob in data[model]:
            key = year_prefix * 100 + prob
            # 根据领域地图填色，若未定义则默认为代数(1)
            matrix[i, prob - 1] = domain_map.get(key, 1)
    return matrix

# 3. 生成矩阵
matrix_24 = create_categorized_matrix(aime24_data, 24)
matrix_25 = create_categorized_matrix(aime25_data, 25)

# 4. 颜色配置
# 0: 灰(未解出), 1: 森林绿(代数), 2: 深绿(数论), 3: 深橄榄绿(几何), 4: 灰绿(组合)
colors = ["#f2f2f2", "#228B22", "#006400", "#556B2F", "#8FBC8F"]
cmap = plt.matplotlib.colors.ListedColormap(colors)

# 5. 绘图
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
models = ['Baseline', 'SFT', 'DRGRPO']
titles = ['AIME 2024: Problem-level Distribution (Potential Data Contamination)',
          'AIME 2025: Problem-level Distribution (OOD Generalization & Logical Scaling)']

for ax, matrix, title in zip(axes, [matrix_24, matrix_25], titles):
    sns.heatmap(matrix, cmap=cmap, cbar=False, linewidths=2, linecolor='white',
                xticklabels=range(1, 31), yticklabels=models, ax=ax, vmin=0, vmax=4)
    ax.set_title(title, fontsize=18, fontweight='bold', pad=25)
    ax.set_ylabel('Model Stage', fontsize=14, fontweight='bold')
    plt.setp(ax.get_yticklabels(), rotation=0)

axes[1].set_xlabel('Problem Number (Ordered by Historical Difficulty $\\rightarrow$)', fontsize=14, fontweight='bold', labelpad=15)

# 6. 添加 Legend
labels = ['Incorrect/Unsolved', 'Algebra (代数)', 'Number Theory (数论)', 'Geometry (几何)', 'Combinatorics (组合)']
patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(colors))]
fig.legend(handles=patches, loc='lower center', ncol=5, fontsize=14, frameon=True, bbox_to_anchor=(0.5, -0.05))

plt.subplots_adjust(hspace=0.6, bottom=0.1)
plt.savefig('asset/aime_domain_distribution.png', dpi=300, bbox_inches='tight')
print("多领域分类分布图已生成：aime_domain_distribution.png")