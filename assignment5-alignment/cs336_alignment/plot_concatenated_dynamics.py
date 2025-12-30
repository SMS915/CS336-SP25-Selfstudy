import matplotlib.pyplot as plt
import pandas as pd

# 设定统一配色和风格 (Color Blind Friendly / Professional)
METHOD_STYLES = {
    'drgrpo': {'color': '#d62728', 'marker': 'o', 'label': 'Dr.GRPO'},
    'grpo': {'color': '#2ca02c', 'marker': 's', 'label': 'GRPO (Standard)'},
    'grpo_no_std_norm': {'color': '#1f77b4', 'marker': '^', 'label': 'GRPO (No Std Norm)'}
}

METRIC_MAP = {
    'accuracy': {'title': 'Accuracy (%)', 'ylabel': 'Acc %'},
    'error_rate': {'title': 'Format Error Rate (%)', 'ylabel': 'Err %'},
    'avg_len': {'title': 'Total Avg Token Length', 'ylabel': 'Tokens'},
    'succ_avg_len': {'title': 'Success Sample Avg Length', 'ylabel': 'Tokens'},
    'err_avg_len': {'title': 'Error Sample Avg Length', 'ylabel': 'Tokens'}
}


def create_readme_composite(df, dataset_name,
                            metrics_to_plot=['accuracy', 'error_rate', 'succ_avg_len', 'err_avg_len']):
    """
    df: 包含所有数据的 DataFrame
    dataset_name: 'aime2024' 或 'math500'
    metrics_to_plot: 想要展示的 4 个指标列表
    """
    subset = df[df['dataset'] == dataset_name]
    methods = sorted(subset['method'].unique())

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()
    handles, labels = [], []

    for i, metric_key in enumerate(metrics_to_plot):
        ax = axes[i]
        meta = METRIC_MAP[metric_key]

        for m in methods:
            m_data = subset[subset['method'] == m]
            style = METHOD_STYLES.get(m, {})
            line, = ax.plot(m_data['step'], m_data[metric_key],
                            color=style['color'], marker=style['marker'],
                            linewidth=2.5, markersize=8, label=style['label'])

            # 只在第一张图中收集图例句柄，确保全局唯一
            if i == 0:
                handles.append(line)
                labels.append(style['label'])

        ax.set_title(meta['title'], fontsize=14, fontweight='bold', pad=10)
        ax.set_ylabel(meta['ylabel'], fontsize=12)
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)

    # 底部统一放置唯一图例
    fig.legend(handles, labels, loc='lower center', ncol=3,
               fontsize=13, bbox_to_anchor=(0.5, -0.02), frameon=True, shadow=True)

    plt.suptitle(f'RL Training Dynamics Summary: {dataset_name.upper()}',
                 fontsize=22, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(f'readme_summary_{dataset_name}.png', bbox_inches='tight', dpi=200)
    plt.close()

# 调用示例
# create_readme_composite(df, 'aime2024')
# create_readme_composite(df, 'math500')