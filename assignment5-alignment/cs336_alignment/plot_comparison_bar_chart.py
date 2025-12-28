import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. 环境准备 ---
# 创建 assets 文件夹
output_dir = "assets"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 设置全局样式 (类似 DeepSeek/OpenAI 技术报告风格)
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

# 定义配色方案
color_baseline = '#E5E7EB'  # 浅灰 (Neutral)
color_sft = '#3B82F6'  # 蓝色 (Structure)
color_drgrpo = '#10B981'  # 绿色 (Optimization)

# --- 2. 数据准备 ---
datasets = ['GSM8K', 'MATH-500', 'AMC', 'AIME 24', 'AIME 25']
# 正确率数据
acc_baseline = [20.85, 91.60, 56.63, 13.33, 3.33]
acc_sft = [45.49, 98.20, 73.49, 26.67, 23.33]
acc_drgrpo = [79.61, 98.80, 75.90, 40.00, 30.00]
# Token 长度数据
len_baseline = [154, 198, 420, 776, 658]
len_sft = [889, 1102, 1899, 2619, 2931]
len_drgrpo = [599, 817, 1531, 1581, 2297]


# --- 3. 绘制图 1: Accuracy Comparison (簇状条形图) ---
def plot_accuracy():
    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)

    rects1 = ax.bar(x - width, acc_baseline, width, label='Baseline', color=color_baseline, edgecolor='white')
    rects2 = ax.bar(x, acc_sft, width, label='SFT (v6)', color=color_sft, edgecolor='white')
    rects3 = ax.bar(x + width, acc_drgrpo, width, label='Dr.GRPO (Final)', color=color_drgrpo, edgecolor='white')

    ax.set_ylabel('Pass@k Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Evolution', fontsize=15, pad=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.legend(frameon=False, loc='upper left', fontsize=10)
    ax.grid(axis='y')
    ax.set_ylim(0, 115)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=8,
                        fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
    print(f"✔️ 已保存: {output_dir}/accuracy_comparison.png")


# --- 4. 绘制图 2: Token Efficiency (折线面积图) ---
def plot_efficiency():
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)

    # 绘制趋势线
    ax.plot(datasets, len_baseline, marker='o', label='Baseline', color='#9CA3AF', linestyle='--', linewidth=2,
            alpha=0.6)
    ax.plot(datasets, len_sft, marker='s', label='SFT (v6)', color=color_sft, linewidth=3)
    ax.plot(datasets, len_drgrpo, marker='D', label='Dr.GRPO (Final)', color=color_drgrpo, linewidth=3)

    # 填充 SFT 到 RL 的提升区域 (强调逻辑压缩)
    ax.fill_between(datasets, len_drgrpo, len_sft, color=color_drgrpo, alpha=0.1,
                    label='Efficiency Gain (Logical Pruning)')

    ax.set_title('Inference Token Consumption Trend', fontsize=15, fontweight='bold', pad=20)
    ax.set_ylabel('Average Token Length', fontsize=12, fontweight='bold')
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, axis='y')

    # 在 AIME 24 处添加标注，强调“拒绝死循环”
    ax.annotate('Dramatic efficiency gain\non complex reasoning',
                xy=('AIME 24', 1650), xytext=('MATH-500', 2500),
                arrowprops=dict(arrowstyle='->', color='black', lw=1),
                fontsize=10, fontstyle='italic',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#10B981", alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'token_efficiency.png'))
    print(f"✔️ 已保存: {output_dir}/token_efficiency.png")


if __name__ == "__main__":
    plot_accuracy()
    plot_efficiency()
    print("\n🚀 所有资产已生成，请刷新 assets 文件夹查看。")