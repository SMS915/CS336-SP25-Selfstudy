import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def main():
    # 1. 模拟数据加载 (请确保你的文件名正确)
    df = pd.read_csv('results/collapse_analysis.csv')
    if df['Collapse_Ratio'].dtype == 'object':
        df['Collapse_Ratio'] = df['Collapse_Ratio'].str.rstrip('%').astype('float')
    # 2. 设置绘图风格
    sns.set_theme(style="whitegrid", palette="muted")

    # 3. 按任务难度排序 (从易到难)
    dataset_order = ['gsm8k', 'math500', 'MathTest', 'amc', 'aime24', 'aime25']

    # 3. 绘制柱状图
    # 使用更高级的颜色映射，区分 SFT 与 RL
    palette = {
        # 官方/基准系列
        "baseline": "#4F4F4F",  # 深灰 (Base)
        "instruct": "#607D8B",  # 蓝灰 (官方 Instruct，保持独立感)

        # 用户训练系列
        "sft": "#D32F2F",  # 红色 (用户 SFT 基础)

        "grpo": "#81D4FA",   # 浅蓝(标准 RL)
        "grpo_no_std_norm": "#7db8ef",  # 深蓝
        "drgrpo": "#82a1e2" ,
        "drgrpo_best": "#F57C00" # 橙色 (你的最终优化版本，最显眼)
    }

    hue_order = ['baseline', 'sft', 'grpo', 'grpo_no_std_norm', 'drgrpo', 'drgrpo_best', 'instruct']

    plt.figure(figsize=(12, 16))  # 增加高度
    ax = sns.barplot(
        data=df,
        y='Dataset',  # 任务名作为 Y 轴
        x='Collapse_Ratio',  # 比例作为 X 轴
        hue='Model',
        order=dataset_order,
        hue_order=hue_order,
        palette=palette,
        orient='h'  # 强制水平方向
    )

    # 标注数字移动到柱子右侧
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=5, fontsize=14)

    plt.xlabel('Mode Collapse Ratio (%)', fontsize=16)
    plt.ylabel('Datasets', fontsize=16)
    plt.legend(title='Model Stage', loc='upper right', frameon=True)

    plt.tick_params(labelsize=14)

    plt.tight_layout()
    plt.savefig('asset/collapse_analysis.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    main()