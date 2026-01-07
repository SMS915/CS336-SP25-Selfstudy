import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 1. 数据准备 ---
models_data = {
    'DS-Math-7B-RL': 7, 'Claude 3 Opus': 175, 'GPT-4o-2024-05': 128,
    'Numina-72B-CoT': 72, 'Llama-3.1-70B-I': 70, 'OpenAI o1-mini': 32,
    'Qwen2.5-7B-I': 7, 'Qwen2.5-32B-I': 32, 'Qwen2.5-72B-I': 72,
    'Qwen2.5-Math-7B-I': 7, 'Qwen2.5-Math-72B-I': 72, 'Mistral-Large-2411': 123,
    'Skywork-o1': 72, 'QWQ-32B-Preview': 32, 'Llama-3.3-70B-I': 70,
    'InternLM-3-8B-I': 8, 'DS Distill 1.5B': 1.5, 'DS Distill 32B': 32, 'Qwen2.5-Max': 110
}

pass1_acc = [2.2, 21.1, 26.7, 2.2, 15.6, 60.3, 11.1, 11.1, 13.3, 11.1, 20.0, 13.3, 11.1, 44.4, 22.2, 11.1, 17.8, 62.2, 22.2]
greedy_r = [0.0, 29.7, 42.9, 0.0, 42.9, 77.4, 60.4, 180.2, 150.4, 180.2, 66.5, 100.0, 119.8, 60.1, 30.2, 119.8, 150.0, 75.1, 59.9]

df = pd.DataFrame({
    'Params': list(models_data.values()),
    'Acc': pass1_acc,
    'Rg': greedy_r
})

# --- 2. 计算演化矩阵 ---
thresholds = [7, 8, 14, 32, 70, 72, 110, 125]
evo_data = []

for t in thresholds:
    low_group = df[df['Params'] <= t]
    high_group = df[df['Params'] > t]
    c_low = low_group['Acc'].corr(low_group['Rg']) if len(low_group) > 2 else np.nan
    c_high = high_group['Acc'].corr(high_group['Rg']) if len(high_group) > 2 else np.nan
    evo_data.append({'Threshold': f"{t}B", '<= threshold': c_low, '> threshold': c_high})

evo_df = pd.DataFrame(evo_data).set_index('Threshold').T

# --- 3. 绘图设置 ---
# 定义字号常量
UPPER_ANNOT_SIZE = 22  # 上方热力图内部数字大小
TICK_LABEL_SIZE = 16   # 坐标轴刻度标签大小

fig = plt.figure(figsize=(18, 12))
# 通过 hspace 和 wspace 增加子图间距
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# Subplot 1: 7B Boundary
ax1 = fig.add_subplot(gs[0, 0])
t7_low = df[df['Params'] <= 7]
sns.heatmap(t7_low[['Acc', 'Rg']].corr(), annot=True, cmap='RdBu_r', center=0, ax=ax1, cbar=False,
            annot_kws={"size": UPPER_ANNOT_SIZE})
ax1.set_title("Evolution Step 1: 7B Boundary", fontsize=14)
ax1.tick_params(labelsize=TICK_LABEL_SIZE)

# Subplot 2: 32B Boundary
ax2 = fig.add_subplot(gs[0, 1])
t32_high = df[df['Params'] > 32]
sns.heatmap(t32_high[['Acc', 'Rg']].corr(), annot=True, cmap='RdBu_r', center=0, ax=ax2, cbar=False,
            annot_kws={"size": UPPER_ANNOT_SIZE})
ax2.set_title("Evolution Step 2: 32B Boundary", fontsize=14)
ax2.tick_params(labelsize=TICK_LABEL_SIZE)

# Subplot 3: 72B Boundary
ax3 = fig.add_subplot(gs[0, 2])
t72_high = df[df['Params'] > 72]
sns.heatmap(t72_high[['Acc', 'Rg']].corr(), annot=True, cmap='RdBu_r', center=0, ax=ax3, cbar=False,
            annot_kws={"size": UPPER_ANNOT_SIZE})
ax3.set_title("Evolution Step 3: 72B Boundary", fontsize=14)
ax3.tick_params(labelsize=TICK_LABEL_SIZE)

# Subplot 4: Continuous Evolution Heatmap (底部长图)
ax_evo = fig.add_subplot(gs[1, :])
sns.heatmap(evo_df, annot=True, cmap='RdBu_r', center=0, fmt=".2f", linewidths=1, ax=ax_evo,
            annot_kws={"size": 14, "weight": "bold"}) # 底部数字保持原样
ax_evo.set_title("Continuous Evolution Heatmap", fontsize=18, pad=20)
ax_evo.tick_params(labelsize=TICK_LABEL_SIZE)

plt.tight_layout(pad=3.0) # 进一步优化边缘间距
plt.savefig('asset/retention_vs_acc_dynamic_dashboard.png', dpi=300)
plt.show()