import matplotlib.pyplot as plt
import pandas as pd
import re
import io

# ==========================================
# 1. 原始数据 (直接嵌入)
# ==========================================
raw_data = """
【项目】: drgrpo_curriculum step50 | aime2025
  - 正确率: 6.67% | 格式错误率: 86.67%
  - Token长度: 平均 3313.2 | 中位数 3821.0 | 最大 3948
  - 成功样本平均Token: 1461.5
  - 格式错误平均Token: 3589.1
----------------------------------------
【项目】: drgrpo_curriculum step100 | aime2025
  - 正确率: 20.00% | 格式错误率: 60.00%
  - Token长度: 平均 2811.9 | 中位数 3216.0 | 最大 3948
  - 成功样本平均Token: 1519.5
  - 格式错误平均Token: 3659.4
----------------------------------------
【项目】: drgrpo_curriculum step150 | aime2025
  - 正确率: 16.67% | 格式错误率: 50.00%
  - Token长度: 平均 2491.6 | 中位数 2637.0 | 最大 3931
  - 成功样本平均Token: 1263.2
  - 格式错误平均Token: 3478.2
----------------------------------------
【项目】: drgrpo_curriculum step200 | aime2025
  - 正确率: 16.67% | 格式错误率: 46.67%
  - Token长度: 平均 2351.6 | 中位数 1806.5 | 最大 3923
  - 成功样本平均Token: 1301.0
  - 格式错误平均Token: 3332.0
----------------------------------------
【项目】: drgrpo_curriculum step250 | aime2025
  - 正确率: 16.67% | 格式错误率: 60.00%
  - Token长度: 平均 2559.5 | 中位数 2136.5 | 最大 3948
  - 成功样本平均Token: 1393.6
  - 格式错误平均Token: 3280.8
----------------------------------------
【项目】: drgrpo_curriculum step300 | aime2025
  - 正确率: 20.00% | 格式错误率: 53.33%
  - Token长度: 平均 2497.2 | 中位数 2622.0 | 最大 3948
  - 成功样本平均Token: 1337.3
  - 格式错误平均Token: 3421.1
----------------------------------------
【项目】: drgrpo_curriculum step350 | aime2025
  - 正确率: 20.00% | 格式错误率: 46.67%
  - Token长度: 平均 2397.3 | 中位数 1701.5 | 最大 3931
  - 成功样本平均Token: 1364.5
  - 格式错误平均Token: 3600.3
----------------------------------------
【项目】: drgrpo_curriculum step400 | aime2025
  - 正确率: 13.33% | 格式错误率: 46.67%
  - Token长度: 平均 2341.5 | 中位数 2141.5 | 最大 3908
  - 成功样本平均Token: 1223.2
  - 格式错误平均Token: 3339.2
----------------------------------------
【项目】: drgrpo_curriculum step450 | aime2025
  - 正确率: 10.00% | 格式错误率: 43.33%
  - Token长度: 平均 2211.2 | 中位数 1654.0 | 最大 3935
  - 成功样本平均Token: 1072.0
  - 格式错误平均Token: 3375.8
----------------------------------------
【项目】: drgrpo_curriculum step500 | aime2025
  - 正确率: 16.67% | 格式错误率: 40.00%
  - Token长度: 平均 2183.7 | 中位数 1517.5 | 最大 3937
  - 成功样本平均Token: 1028.2
  - 格式错误平均Token: 3555.8
----------------------------------------
【项目】: drgrpo_curriculum step550 | aime2025
  - 正确率: 16.67% | 格式错误率: 56.67%
  - Token长度: 平均 2506.8 | 中位数 2511.5 | 最大 3923
  - 成功样本平均Token: 1332.6
  - 格式错误平均Token: 3325.1
----------------------------------------
【项目】: drgrpo_curriculum step50 | math500
  - 正确率: 35.80% | 格式错误率: 57.20%
  - Token长度: 平均 2650.4 | 中位数 3824.5 | 最大 3994
  - 成功样本平均Token: 966.8
  - 格式错误平均Token: 3880.2
----------------------------------------
【项目】: drgrpo_curriculum step100 | math500
  - 正确率: 50.20% | 格式错误率: 40.00%
  - Token长度: 平均 2081.2 | 中位数 1182.0 | 最大 3992
  - 成功样本平均Token: 882.8
  - 格式错误平均Token: 3828.6
----------------------------------------
【项目】: drgrpo_curriculum step150 | math500
  - 正确率: 54.20% | 格式错误率: 34.20%
  - Token长度: 平均 1908.9 | 中位数 1094.0 | 最大 3991
  - 成功样本平均Token: 882.5
  - 格式错误平均Token: 3805.3
----------------------------------------
【项目】: drgrpo_curriculum step200 | math500
  - 正确率: 55.60% | 格式错误率: 32.80%
  - Token长度: 平均 1820.9 | 中位数 1006.0 | 最大 3990
  - 成功样本平均Token: 840.0
  - 格式错误平均Token: 3755.4
----------------------------------------
【项目】: drgrpo_curriculum step250 | math500
  - 正确率: 55.00% | 格式错误率: 31.40%
  - Token长度: 平均 1799.7 | 中位数 981.0 | 最大 3988
  - 成功样本平均Token: 853.0
  - 格式错误平均Token: 3764.3
----------------------------------------
【项目】: drgrpo_curriculum step300 | math500
  - 正确率: 58.80% | 格式错误率: 27.80%
  - Token长度: 平均 1699.0 | 中位数 1006.0 | 最大 3990
  - 成功样本平均Token: 857.3
  - 格式错误平均Token: 3768.2
----------------------------------------
【项目】: drgrpo_curriculum step350 | math500
  - 正确率: 57.60% | 格式错误率: 28.40%
  - Token长度: 平均 1723.5 | 中位数 1001.5 | 最大 3988
  - 成功样本平均Token: 842.4
  - 格式错误平均Token: 3759.2
----------------------------------------
【项目】: drgrpo_curriculum step400 | math500
  - 正确率: 54.80% | 格式错误率: 29.20%
  - Token长度: 平均 1720.3 | 中位数 1006.5 | 最大 3988
  - 成功样本平均Token: 859.7
  - 格式错误平均Token: 3713.8
----------------------------------------
【项目】: drgrpo_curriculum step450 | math500
  - 正确率: 56.00% | 格式错误率: 26.60%
  - Token长度: 平均 1669.0 | 中位数 957.5 | 最大 3988
  - 成功样本平均Token: 852.4
  - 格式错误平均Token: 3763.0
----------------------------------------
【项目】: drgrpo_curriculum step500 | math500
  - 正确率: 58.60% | 格式错误率: 26.60%
  - Token长度: 平均 1672.1 | 中位数 990.5 | 最大 3991
  - 成功样本平均Token: 868.9
  - 格式错误平均Token: 3752.5
----------------------------------------
【项目】: drgrpo_curriculum step550 | math500
  - 正确率: 57.20% | 格式错误率: 28.20%
  - Token长度: 平均 1720.0 | 中位数 956.0 | 最大 3991
  - 成功样本平均Token: 830.9
  - 格式错误平均Token: 3828.8
"""


# ==========================================
# 2. 数据解析
# ==========================================
def parse_raw_data(text):
    data = []
    blocks = text.split('----------------------------------------')
    for block in blocks:
        if '【项目】' not in block: continue
        try:
            step = int(re.search(r'step(\d+)', block).group(1))
            # 兼容 aime2025 和 math500
            dataset_match = re.search(r'\| (aime\d+|math500)', block)
            if not dataset_match: continue
            dataset = dataset_match.group(1)

            acc = float(re.search(r'正确率: ([\d.]+)%', block).group(1))
            err = float(re.search(r'格式错误率: ([\d.]+)%', block).group(1))
            avg_l = float(re.search(r'Token长度: 平均 ([\d.]+)', block).group(1))
            succ_l = float(re.search(r'成功样本平均Token: ([\d.]+)', block).group(1))
            fail_l = float(re.search(r'格式错误平均Token: ([\d.]+)', block).group(1))
            if dataset == 'aime2025':
                experient_name = 'aime2025 Pass@8'
            else:
                experient_name = 'math500 Pass@1'
            data.append({
                'step': step, 'dataset': experient_name, 'acc': acc, 'err': err,
                'avg_len': avg_l, 'succ_len': succ_l, 'fail_len': fail_l
            })
        except Exception as e:
            continue
    return pd.DataFrame(data).sort_values('step')


df = parse_raw_data(raw_data)

# ==========================================
# 3. 绘图风格设置 (Metric-Based)
# ==========================================
# 定义不同指标的配色和样式
METRIC_STYLES = {
    'acc': {'color': '#2ca02c', 'marker': 'o', 'label': 'Accuracy (%)', 'title': 'Accuracy Trend'},
    'err': {'color': '#d62728', 'marker': 'x', 'label': 'Format Error (%)', 'title': 'Format Instability'},
    'avg_len': {'color': '#1f77b4', 'marker': 's', 'label': 'Total Avg Length', 'title': 'Global Token Usage'},
    'succ_len': {'color': '#9467bd', 'marker': '^', 'label': 'Success Length', 'title': 'Reasoning Efficiency'},
    'fail_len': {'color': '#ff7f0e', 'marker': 'v', 'label': 'Error Length', 'title': 'Reasoning Efficiency'}
}


def create_training_dashboard(df, dataset_name):
    """
    生成 2x2 的单一模型训练动态仪表盘
    """
    subset = df[df['dataset'] == dataset_name]
    if subset.empty:
        print(f"No data for {dataset_name}")
        return

    # 创建 2x2 画布
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # ---------------------------
    # Plot 1: Accuracy (Top Left)
    # ---------------------------
    ax = axes[0]
    style = METRIC_STYLES['acc']
    ax.plot(subset['step'], subset['acc'], color=style['color'], marker=style['marker'],
            linewidth=2.5, markersize=8, label=style['label'])

    # 标注最高点
    max_acc = subset['acc'].max()
    max_step = subset.loc[subset['acc'].idxmax(), 'step']
    ax.annotate(f'Best: {max_acc}% (Step {max_step})',
                xy=(max_step, max_acc), xytext=(max_step, max_acc - (max_acc * 0.2)),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=12, fontweight='bold', ha='center')

    ax.set_title(style['title'], fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right')

    # ---------------------------
    # Plot 2: Error Rate (Top Right)
    # ---------------------------
    ax = axes[1]
    style = METRIC_STYLES['err']
    ax.plot(subset['step'], subset['err'], color=style['color'], marker=style['marker'],
            linewidth=2.5, markersize=8, label=style['label'])
    ax.set_title(style['title'], fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # ---------------------------
    # Plot 3: Total Avg Length (Bottom Left)
    # ---------------------------
    ax = axes[2]
    style = METRIC_STYLES['avg_len']
    ax.plot(subset['step'], subset['avg_len'], color=style['color'], marker=style['marker'],
            linewidth=2.5, markersize=8, label=style['label'])
    ax.set_title(style['title'], fontsize=14, fontweight='bold')
    ax.set_ylabel('Token Count', fontsize=12)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # ---------------------------
    # Plot 4: Success vs Fail Length (Bottom Right)
    # ---------------------------
    ax = axes[3]
    s_style = METRIC_STYLES['succ_len']
    f_style = METRIC_STYLES['fail_len']

    ax.plot(subset['step'], subset['succ_len'], color=s_style['color'], marker=s_style['marker'],
            linewidth=2, label=s_style['label'])
    ax.plot(subset['step'], subset['fail_len'], color=f_style['color'], marker=f_style['marker'],
            linestyle=':', linewidth=2, label=f_style['label'])

    ax.set_title('Efficiency Analysis (Succ vs Err)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Token Count', fontsize=12)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # 全局标题和布局
    plt.suptitle(f'Dr.GRPO Training Dynamics: {dataset_name.upper()}',
                 fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存
    filename = f'drgrpo_dashboard_{dataset_name}.png'
    plt.savefig(filename, dpi=150)
    print(f"Generated dashboard: {filename}")
    plt.close()


# ==========================================
# 4. 执行生成
# ==========================================
if not df.empty:
    for ds in df['dataset'].unique():
        create_training_dashboard(df, ds)
else:
    print("No data parsed.")