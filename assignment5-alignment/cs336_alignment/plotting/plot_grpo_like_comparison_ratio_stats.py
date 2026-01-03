import matplotlib.pyplot as plt
import matplotlib
import os

# ==========================================
# 1. 样式与清晰度全局配置
# ==========================================
font_list = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
preferred_font = "SimHei"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [preferred_font],
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 200
})

# ==========================================
# 2. 数据准备
# ==========================================
datasets = ['GSM8K(Pass@1)', 'MATH500(Pass@64)', 'MATHTEST(Pass@64)', 'AMC(Pass@64)', 'AIME24(Pass@64)', 'AIME25(Pass@64)']
models = ['GRPO', 'w/o_std_norm', 'DRGRPO']

acc_data = {
    'GRPO': [80.36, 92.60, 84.68, 75.90, 30.00, 30.00],
    'w/o_std_norm': [80.59, 93.60, 85.14, 75.90, 40.00, 30.00],
    'DRGRPO': [79.15, 92.00, 84.36, 74.70, 40.00, 36.67]
}

err_data = {
    'GRPO': [7.28, 4.00, 6.84, 14.46, 36.67, 36.67],
    'w/o_std_norm': [8.72, 3.40, 6.98, 14.46, 13.33, 46.67],
    'DRGRPO': [8.49, 4.60, 7.14, 13.25, 30.00, 33.33]
}

# ==========================================
# 3. 绘图逻辑
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), sharex=True)

styles = {
    'GRPO': {'color': '#1f77b4', 'marker': 'o'},
    'w/o_std_norm': {'color': '#ff7f0e', 'marker': 's'},
    'DRGRPO': {'color': '#2ca02c', 'marker': '^'}
}

# 绘图
for model in models:
    ax1.plot(datasets, acc_data[model], label=model, linewidth=2.2, markersize=7, **styles[model])
    ax2.plot(datasets, err_data[model], label=model, linewidth=2.2, markersize=7, linestyle='--', **styles[model])

# 上图装饰
ax1.set_title('GRPO 变体正确率对比 (Accuracy %)', pad=15, fontweight='bold')
ax1.set_ylabel('百分比 (%)', labelpad=8)
ax1.grid(True, alpha=0.2, linestyle='--')
ax1.legend(loc='lower left', frameon=True, framealpha=0.9) # 移到左下防止遮挡下降曲线

# 下图装饰
ax2.set_title('GRPO 变体格式错误率对比 (Format Error %)', pad=15, fontweight='bold')
ax2.set_ylabel('百分比 (%)', labelpad=8)
ax2.set_xlabel('数据集 (按难度排序)', labelpad=15, fontsize=13) # 增加字号和间距
ax2.grid(True, alpha=0.2, linestyle='--')
ax2.legend(loc='upper left', frameon=True, framealpha=0.9)

# ==========================================
# 4. 精细布局调整 (替代 tight_layout)
# ==========================================
# left: 增加左边距，防止Y轴标题出界
# bottom: 增加底边距，给横轴标题留位置
# hspace: 增加子图间的垂直间距，解决“归属感”问题
plt.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.16, hspace=0.4)

# ==========================================
# 5. 路径处理与保存
# ==========================================
save_path = 'asset/GRPOlike_Comparison/RatioComparison.png'

plt.savefig(save_path, dpi=300)
print(f"图表已优化并保存至: {save_path}")

plt.show()