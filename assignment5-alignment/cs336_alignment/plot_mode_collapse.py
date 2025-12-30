import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# data = [
#     ['baseline', 'aime24', 20.00], ['baseline', 'aime25', 13.33], ['baseline', 'amc', 8.43],
#     ['baseline', 'gsm8k', 0.38], ['baseline', 'math500', 0.40], ['baseline', 'MathTest', 3.98],
#     ['drgrpo', 'aime24', 83.33], ['drgrpo', 'aime25', 83.33], ['drgrpo', 'amc', 60.24],
#     ['drgrpo', 'gsm8k', 9.55], ['drgrpo', 'math500', 13.20], ['drgrpo', 'MathTest', 32.88],
#     ['grpo', 'aime24', 73.33], ['grpo', 'aime25', 83.33], ['grpo', 'amc', 60.24],
#     ['grpo', 'gsm8k', 8.19], ['grpo', 'math500', 10.40], ['grpo', 'MathTest', 28.92],
#     ['grpo_no_std_norm', 'aime24', 83.33], ['grpo_no_std_norm', 'aime25', 83.33],
#     ['grpo_no_std_norm', 'amc', 55.42], ['grpo_no_std_norm', 'gsm8k', 12.81],
#     ['grpo_no_std_norm', 'math500', 13.20], ['grpo_no_std_norm', 'MathTest', 30.76],
#     ['instruct', 'amc', 14.46], ['instruct', 'gsm8k', 0.53], ['instruct', 'math500', 0.60],
#     ['instruct', 'MathTest', 2.66],
#     ['sft', 'aime24', 93.33], ['sft', 'aime25', 90.00], ['sft', 'amc', 72.29],
#     ['sft', 'gsm8k', 42.84], ['sft', 'math500', 40.40], ['sft', 'MathTest', 59.36]
# ]
# df = pd.DataFrame(data, columns=['Model', 'Dataset', 'Collapse_Ratio'])

# 1. 模拟数据加载 (请确保你的文件名正确)
df = pd.read_csv('results/collapse_analysis.csv')
# 2. 设置绘图风格
sns.set_theme(style="whitegrid", palette="muted")
plt.figure(figsize=(14, 7))

# 3. 按任务难度排序 (从易到难)
dataset_order = ['gsm8k', 'math500', 'MathTest', 'amc', 'aime24', 'aime25']

# 4. 绘制分组柱状图
ax = sns.barplot(
    data=df,
    x='Dataset',
    y='Collapse_Ratio',
    hue='Model',
    order=dataset_order,
    palette="viridis"
)

# 5. 图表美化
plt.title('Comparison of Mode Collapse Ratios Across Models and Datasets', fontsize=16, fontweight='bold')
plt.ylabel('Collapse Ratio (%)', fontsize=12)
plt.xlabel('Evaluation Datasets (Ordered by Difficulty)', fontsize=12)
plt.ylim(0, 100)
plt.legend(title='Model Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# 在柱状图上方标注数值 (可选，适合展示具体数据)
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f'{p.get_height():.1f}%',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    fontsize=8, color='black', rotation=90, xytext=(0, 15),
                    textcoords='offset points')

plt.tight_layout()
plt.savefig('collapse_ratio_comparison.png', dpi=300)
plt.show()