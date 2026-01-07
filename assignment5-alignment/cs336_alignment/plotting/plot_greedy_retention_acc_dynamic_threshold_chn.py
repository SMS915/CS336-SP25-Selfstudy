import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 0. 环境设置：确保中文显示 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文黑体
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

# --- 1. 数据准备 (保持不变) ---
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

df = pd.DataFrame({'Params': list(models_data.values()), 'Acc': pass1_acc, 'Rg': greedy_r})

# --- 2. 计算演化矩阵 (重命名索引以匹配中文) ---
thresholds = [7, 8, 14, 32, 70, 72, 110, 125]
evo_data = []
for t in thresholds:
    low = df[df['Params'] <= t]
    high = df[df['Params'] > t]
    c_low = low['Acc'].corr(low['Rg']) if len(low) > 2 else np.nan
    c_high = high['Acc'].corr(high['Rg']) if len(high) > 2 else np.nan
    evo_data.append({'阈值': f"{t}B", '模型规模 ≤ 阈值': c_low, '模型规模 > 阈值': c_high})

evo_df = pd.DataFrame(evo_data).set_index('阈值').T

# --- 3. 绘图 ---
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

UPPER_ANNOT_SIZE = 22  # 上方热力图内数字
TICK_LABEL_SIZE = 16   # 坐标轴刻度

# Subplot 1: 7B 边界
ax1 = fig.add_subplot(gs[0, 0])
t7_low = df[df['Params'] <= 7][['Acc', 'Rg']].corr()
t7_low.index = t7_low.columns = ['准确率', '留存率'] # 重命名坐标轴
sns.heatmap(t7_low, annot=True, cmap='RdBu_r', center=0, ax=ax1, cbar=False, annot_kws={"size": UPPER_ANNOT_SIZE})
ax1.set_title("阶段 1：7B 边界\n(小模型确定性)", fontsize=16, fontweight='bold')
ax1.tick_params(labelsize=TICK_LABEL_SIZE)

# Subplot 2: 32B 边界
ax2 = fig.add_subplot(gs[0, 1])
t32_high = df[df['Params'] > 32][['Acc', 'Rg']].corr()
t32_high.index = t32_high.columns = ['准确率', '留存率']
sns.heatmap(t32_high, annot=True, cmap='RdBu_r', center=0, ax=ax2, cbar=False, annot_kws={"size": UPPER_ANNOT_SIZE})
ax2.set_title("阶段 2：32B 边界\n(逻辑解耦阶段)", fontsize=16, fontweight='bold')
ax2.tick_params(labelsize=TICK_LABEL_SIZE)

# Subplot 3: 72B 边界
ax3 = fig.add_subplot(gs[0, 2])
t72_high = df[df['Params'] > 72][['Acc', 'Rg']].corr()
t72_high.index = t72_high.columns = ['准确率', '留存率']
sns.heatmap(t72_high, annot=True, cmap='RdBu_r', center=0, ax=ax3, cbar=False, annot_kws={"size": UPPER_ANNOT_SIZE})
ax3.set_title("阶段 3：72B 边界\n(涌现悖论 / 逻辑逆转)", fontsize=16, fontweight='bold')
ax3.tick_params(labelsize=TICK_LABEL_SIZE)

# Subplot 4: 演化全景图
ax_evo = fig.add_subplot(gs[1, :])
sns.heatmap(evo_df, annot=True, cmap='RdBu_r', center=0, fmt=".2f", linewidths=1, ax=ax_evo,
            annot_kws={"size": 16, "weight": "bold"})
ax_evo.set_title("全景图：准确率与留存率相关性随参数规模的动态偏移", fontsize=20, pad=25, fontweight='bold')
ax_evo.set_xlabel("参数规模阈值 (十亿 / B)", fontsize=16)
ax_evo.set_ylabel("分析分组", fontsize=16)
ax_evo.tick_params(labelsize=TICK_LABEL_SIZE)

# 保存
if not os.path.exists('asset'): os.makedirs('asset')
plt.tight_layout(pad=3.0)
plt.savefig('asset/logic_evolution_chinese_dashboard.png', dpi=300)
plt.show()