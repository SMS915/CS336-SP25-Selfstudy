import matplotlib.pyplot as plt
import matplotlib
import os

# ==========================================
# 1. 样式与清晰度全局配置 (严格保留验证后的细节)
# ==========================================
font_list = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
preferred_font = "SimHei"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei"],
    "axes.unicode_minus": False,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 200
})

# ==========================================
# 2. 数据准备 (Length 数据)
# ==========================================
datasets = ['GSM8K(Pass@1)', 'MATH500(Pass@64)', 'MATHTEST(Pass@64)', 'AMC(Pass@64)', 'AIME24(Pass@64)', 'AIME25(Pass@64)']
models = ['GRPO', 'w/o_std_norm', 'DRGRPO']

# 成功样本平均长度 (Succ Avg Token)
succ_len = {
    'GRPO': [521.7, 904.4, 870.4, 1205.6, 1400.2, 1396.3],
    'w/o_std_norm': [551.6, 948.0, 896.1, 1217.6, 1279.8, 1248.2],
    'DRGRPO': [605.5, 946.7, 922.3, 1201.0, 1446.8, 1315.8]
}

# 失败样本平均长度 (Fail Avg Token)
fail_len = {
    'GRPO': [1976.1, 3584.2, 3438.6, 3383.6, 3464.9, 3680.9],
    'w/o_std_norm': [2035.0, 3549.4, 2868.3, 3097.2, 2832.5, 3473.2],
    'DRGRPO': [2036.5, 3364.6, 3517.6, 3767.3, 3407.9, 3270.7]
}

# ==========================================
# 3. 绘图逻辑 (严格保留验证后的布局细节)
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), sharex=True)

styles = {
    'GRPO': {'color': '#1f77b4', 'marker': 'o'},
    'w/o_std_norm': {'color': '#ff7f0e', 'marker': 's'},
    'DRGRPO': {'color': '#2ca02c', 'marker': '^'}
}

# 绘图
for model in models:
    # 上图：成功长度
    ax1.plot(datasets, succ_len[model], label=model, linewidth=2.2, markersize=7, **styles[model])
    # 下图：失败长度 (保持虚线风格以区分指标)
    ax2.plot(datasets, fail_len[model], label=model, linewidth=2.2, markersize=7, linestyle='--', **styles[model])

# 上图装饰
ax1.set_title('GRPO 变体成功样本平均长度 (Succ Avg Token)', pad=15, fontweight='bold')
ax1.set_ylabel('Tokens', labelpad=8)
ax1.grid(True, alpha=0.2, linestyle='--')
ax1.legend(loc='upper left', frameon=True, framealpha=0.9)

# 下图装饰
ax2.set_title('GRPO 变体失败样本平均长度 (Fail Avg Token)', pad=15, fontweight='bold')
ax2.set_ylabel('Tokens', labelpad=8)
ax2.set_xlabel('数据集 (按难度排序)', labelpad=10, fontsize=13)
ax2.grid(True, alpha=0.2, linestyle='--')
ax2.legend(loc='upper left', frameon=True, framealpha=0.9)

# ==========================================
# 4. 精细布局调整 (严格保留验证后的比例参数)
# ==========================================
plt.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.16, hspace=0.4)

# ==========================================
# 5. 路径处理与保存
# ==========================================
save_path = 'asset/GRPOlike_Comparison/ResponseLengthComparison.png'

# 保存不使用 bbox_inches='tight'，确保完全符合手调的 adjust 比例
plt.savefig(save_path, dpi=300)
print(f"图表已按验证布局保存至: {save_path}")

plt.show()