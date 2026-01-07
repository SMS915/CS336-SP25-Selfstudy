import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据准备
models = [
    'DS-Math-7B-RL', 'Claude 3 Opus', 'GPT-4o-2024-05', 'Numina-72B-CoT', 'Llama-3.1-70B-I',
    'OpenAI o1-mini', 'Qwen2.5-7B-I', 'Qwen2.5-32B-I', 'Qwen2.5-72B-I', 'Qwen2.5-Math-7B-I',
    'Qwen2.5-Math-72B-I', 'Mistral-Large-2411', 'Skywork-o1', 'QWQ-32B-Preview', 'Llama-3.3-70B-I',
    'InternLM-3-8B-I', 'DS Distill 1.5B', 'DS Distill 32B', 'Qwen2.5-Max'
]
pass1_acc = [2.2, 21.1, 26.7, 2.2, 15.6, 60.3, 11.1, 11.1, 13.3, 11.1, 20.0, 13.3, 11.1, 44.4, 22.2, 11.1, 17.8, 62.2, 22.2]
passk_acc = [16.3, 34.1, 43.3, 21.3, 41.2, 86.7, 26.3, 32.0, 33.7, 20.8, 35.2, 15.4, 22.1, 74.3, 37.1, 20.5, 68.7, 86.3, 44.4]
greedy_r = [0.0, 29.7, 42.9, 0.0, 42.9, 77.4, 60.4, 180.2, 150.4, 180.2, 66.5, 100.0, 119.8, 60.1, 30.2, 119.8, 150.0, 75.1, 59.9]
upper_r = [0.0, 38.2, 55.4, 98.6, 51.7, 71.5, 95.8, 104.1, 98.5, 176.9, 85.5, 127.9, 141.2, 81.4, 36.7, 147.8, 79.5, 84.1, 89.9]

df = pd.DataFrame({
    'Model': models,
    'Pass1_Acc': pass1_acc,
    'Passk_Acc': passk_acc,
    'Rg': greedy_r,
    'Ru': upper_r
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# 子图1: Rg vs Pass@1 (直觉稳定性)
for i, row in df.iterrows():
    color = 'tab:red' if any(x in row['Model'] for x in ['Qwen', 'QWQ', 'Distill']) else 'tab:blue'
    ax1.scatter(row['Pass1_Acc'], row['Rg'], c=color, s=120, edgecolors='black', alpha=0.7)
    ax1.text(row['Pass1_Acc']+0.5, row['Rg']+2, row['Model'], fontsize=9)

ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Greedy Accuracy (Pass@1) %', fontsize=12)
ax1.set_ylabel('Greedy Retention (Rg) %', fontsize=12)
ax1.set_title('图 A：直觉可靠性分布 (Rg vs Accuracy)', fontsize=14)

# 子图2: Ru vs Pass@k (潜力稳定性)
for i, row in df.iterrows():
    color = 'tab:red' if any(x in row['Model'] for x in ['Qwen', 'QWQ', 'Distill']) else 'tab:blue'
    ax2.scatter(row['Passk_Acc'], row['Ru'], c=color, s=120, edgecolors='black', alpha=0.7)
    ax2.text(row['Passk_Acc']+0.5, row['Ru']+2, row['Model'], fontsize=9)

ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Pass@k Accuracy %', fontsize=12)
ax2.set_ylabel('Upper Retention (Ru) %', fontsize=12)
ax2.set_title('图 B：上限探索稳定性 (Ru vs Accuracy)', fontsize=14)

plt.suptitle('推理能力的一体两面：直觉与潜力的泛化偏差对比分析', fontsize=18, y=1.02)
plt.tight_layout()
plt.savefig('retention_dual_comparison.png')
print("Plot saved as retention_dual_comparison.png")