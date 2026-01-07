import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 0. 环境设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 数据提取 (基于表5: Pass@16 潜力评估) ---
# 包含：模型名, 参数规模(B), 25 Upper (Acc), 留存率 (Ru)
raw_data = [
    ('Qwen2.5-Math-7B-I', 7, 36.8, 176.9),
    ('Qwen2.5-Math-72B-I', 72, 30.1, 85.5),
    ('Qwen2.5-32B-Instruct', 32, 33.3, 104.1),
    ('Qwen2.5-72B-Instruct', 72, 33.2, 98.5),
    ('Qwen2.5-7B-Instruct', 7, 25.2, 95.8),
    ('Qwen2.5-Max', 300, 39.9, 89.9), # 假设 Max 为 300B 规模
    ('OpenAI o1-mini', 32, 62.0, 71.5),
    ('QwQ-32B-Preview', 32, 60.5, 81.4),
    ('DS Distill Qwen-1.5B', 1.5, 54.6, 79.5),
    ('DS Distill Qwen-32B', 32, 72.6, 84.1),
    ('Mistral-Large-2411', 123, 19.7, 127.9),
    ('InternLM-3-8B-I', 8, 30.3, 147.8),
    ('Skywork-o1-8B', 8, 31.2, 141.2),
    ('Llama-3.1-70B-I', 70, 21.3, 51.7),
    ('Llama-3.3-70B-I', 70, 13.6, 36.7),
    ('DS-Math-7B-RL', 7, 0.0, 0.0), # 表格显示 AIME25 表现极其异常
    ('Numina-72B-CoT', 72, 21.0, 98.6),
    ('Claude 3 Opus', 172, )
]

df = pd.DataFrame(raw_data, columns=['Model', 'Params', 'Acc', 'Ru'])

# --- 2. 计算演化矩阵 ---
thresholds = [1.5, 7, 8, 32, 70, 72, 123]
evo_data = []
for t in thresholds:
    low = df[df['Params'] <= t]
    high = df[df['Params'] > t]
    c_low = low['Acc'].corr(low['Ru']) if len(low) > 2 else np.nan
    c_high = high['Acc'].corr(high['Ru']) if len(high) > 2 else np.nan
    evo_data.append({'阈值': f"{t}B", '规模 ≤ 阈值': c_low, '规模 > 阈值': c_high})

evo_df = pd.DataFrame(evo_data).set_index('阈值').T

# --- 3. 绘图 (沿用大字号与中文标题) ---
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

UPPER_ANNOT_SIZE = 22
TICK_LABEL_SIZE = 16

# 子图 1: 7B 边界 (小模型)
ax1 = fig.add_subplot(gs[0, 0])
corr1 = df[df['Params'] <= 7][['Acc', 'Ru']].corr()
corr1.index = corr1.columns = ['AIME25', '留存率']
sns.heatmap(corr1, annot=True, cmap='RdBu_r', center=0, ax=ax1, cbar=False, annot_kws={"size": UPPER_ANNOT_SIZE})
ax1.set_title("阶段 1：7B 边界\n(推理潜力一致性)", fontsize=16, fontweight='bold')
ax1.tick_params(labelsize=TICK_LABEL_SIZE)

# 子图 2: 32B 边界 (o1/推理模型集中区)
ax2 = fig.add_subplot(gs[0, 1])
corr2 = df[df['Params'] > 32][['Acc', 'Ru']].corr()
corr2.index = corr2.columns = ['AIME25', '留存率']
sns.heatmap(corr2, annot=True, cmap='RdBu_r', center=0, ax=ax2, cbar=False, annot_kws={"size": UPPER_ANNOT_SIZE})
ax2.set_title("阶段 2：32B 边界\n(高准确率下的逻辑解耦)", fontsize=16, fontweight='bold')
ax2.tick_params(labelsize=TICK_LABEL_SIZE)

# 子图 3: 72B 边界 (大参数逻辑陷阱)
ax3 = fig.add_subplot(gs[0, 2])
corr3 = df[df['Params'] > 72][['Acc', 'Ru']].corr()
corr3.index = corr3.columns = ['AIME25', '留存率']
sns.heatmap(corr3, annot=True, cmap='RdBu_r', center=0, ax=ax3, cbar=False, annot_kws={"size": UPPER_ANNOT_SIZE})
ax3.set_title("阶段 3：72B 边界\n(规模悖论：高参低效)", fontsize=16, fontweight='bold')
ax3.tick_params(labelsize=TICK_LABEL_SIZE)

# 底部全景图
ax_evo = fig.add_subplot(gs[1, :])
sns.heatmap(evo_df, annot=True, cmap='RdBu_r', center=0, fmt=".2f", linewidths=1, ax=ax_evo,
            annot_kws={"size": 16, "weight": "bold"})
ax_evo.set_title("Pass@16 逻辑演化全景：AIME25 潜力与留存率的相关性偏移", fontsize=20, pad=25, fontweight='bold')
ax_evo.set_xlabel("参数规模阈值 (十亿 / B)", fontsize=16)
ax_evo.set_ylabel("分析分组", fontsize=16)
ax_evo.tick_params(labelsize=TICK_LABEL_SIZE)

plt.tight_layout(pad=3.0)
plt.savefig('asset/aime25_pass16_evolution_dashboard.png', dpi=300)
plt.show()