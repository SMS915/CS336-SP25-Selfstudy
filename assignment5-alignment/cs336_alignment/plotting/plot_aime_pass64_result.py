import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 原始数据定义 (基于实验观测)
# AIME 2024 数据：展现出明显的“跳跃式”得分特征 (数据污染指控证据)
aime24_data = {
    'Baseline': [8, 9, 13, 19],
    'SFT': [1, 7, 8, 10, 13, 23, 25, 27],
    'DRGRPO': [1, 7, 8, 9, 10, 13, 19, 23, 24, 25, 26, 27]
}

# AIME 2025 数据：展现出稳健的“能力前沿”单调推进 (泛化性证明)
aime25_data = {
    'Baseline': [1],
    'SFT': [1, 3, 4, 6, 9, 16, 17],
    'DRGRPO': [1, 2, 3, 4, 5, 6, 8, 9, 16, 17, 20]
}


def create_binary_matrix(data, num_problems=30):
    """将正确题号列表转换为 3x30 的二值矩阵"""
    matrix = np.zeros((3, num_problems))
    models = ['Baseline', 'SFT', 'DRGRPO']
    for i, model in enumerate(models):
        for prob in data[model]:
            if 1 <= prob <= num_problems:
                matrix[i, prob - 1] = 1
    return matrix


# 2. 生成矩阵
matrix_24 = create_binary_matrix(aime24_data)
matrix_25 = create_binary_matrix(aime25_data)

# 3. 绘图配置
# 设置画布大小 (宽16, 高9)，确保上下间距充足
fig, axes = plt.subplots(2, 1, figsize=(16, 9))
models = ['Baseline', 'SFT', 'DRGRPO']

# 颜色配置：浅灰 (#F2F2F2) 表示未解出，深森林绿 (#145A32) 表示正确
# 深绿色能显著强化“记忆孤岛”的视觉特征
cmap = sns.color_palette(["#f2f2f2", "#145a32"])

# 4. 循环渲染两个年份的分布图
titles = [
    'AIME 2024: Problem-level Distribution (Potential Data Contamination)',
    'AIME 2025: Problem-level Distribution (OOD Generalization & Logical Scaling)'
]

for ax, matrix, title in zip(axes, [matrix_24, matrix_25], titles):
    # linewidths=2 为每个格子增加白色边框，增加“打孔卡”质感
    sns.heatmap(matrix, cmap=cmap, cbar=False, linewidths=2, linecolor='white',
                xticklabels=range(1, 31), yticklabels=models, ax=ax)

    # 样式细节调整
    ax.set_title(title, fontsize=18, fontweight='bold', pad=25)
    ax.set_ylabel('Model Stage', fontsize=13, fontweight='bold')
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=12)
    plt.setp(ax.get_xticklabels(), fontsize=10)

# 设置横轴标签
axes[1].set_xlabel('Problem Number (Ordered by Historical Difficulty $\\rightarrow$)',
                   fontsize=14, fontweight='bold', labelpad=15)

# 5. 调整布局：显式拉开子图之间的垂直距离
plt.subplots_adjust(hspace=0.6)

# 6. 保存高质量图片
plt.savefig('aime_distribution_final_report.png', dpi=300, bbox_inches='tight')
print("科研级分布图已生成：aime_distribution_final_report.png")