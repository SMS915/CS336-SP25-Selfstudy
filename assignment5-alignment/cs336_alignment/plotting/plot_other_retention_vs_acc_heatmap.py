import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 0. 环境设置：确保中文显示 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 数据整合 ---
params_data = {
    'DeepSeek-Math-7b-RL': 7, 'NuminaMath-72B-CoT': 72, 'Llama-3.1-8B-Instruct': 8,
    'Gemma-2-27B-it': 27, 'InternLM-3-8B-Instruct': 8, 'Qwen2.5-32B-Instruct': 32,
    'Qwen2.5-7B-Instruct': 7, 'Qwen2.5-Math-7B-Instruct': 7, 'Skywork-o1': 8,
    'Mistral-Large-Instruct-2411': 123, 'Qwen2.5-72B-Instruct': 72, 'Gemini-1.5-Pro-Latest': 175,
    'Llama-3.1-70B-Instruct': 70, 'Qwen2.5-Math-72B-Instruct': 72, 'Llama-3.3-70B-Instruct': 70,
    'Qwen2.5-Max': 300,  # Assumption
    'DeepSeek Distill Qwen-1.5B': 1.5, 'QwQ-32B-Preview': 32, 'DeepSeek Distill LLama-8B': 8,
    'DeepSeek Distill Qwen-7B': 7, 'OpenAI o1-mini': 100,  # Assumption
    'DeepSeek Distill Qwen-14B': 14, 'DeepSeek Distill LLama-70B': 70, 'DeepSeek Distill Qwen-32B': 32,
    'Yi-1.5-34B-Chat': 34
    }

# 数据源 2: AIME24 准确率
aime24_acc = {
    'Yi-1.5-34B-Chat': {'p1': 2.2, 'p16': 5.9}, 'Llama-3.1-8B-Instruct': {'p1': 4.4, 'p16': 5.9},
    'Gemma-2-27B-it': {'p1': 6.7, 'p16': 8.3}, 'InternLM-3-8B-Instruct': {'p1': 11.1, 'p16': 13.3},
    'Qwen2.5-32B-Instruct': {'p1': 11.1, 'p16': 14.9}, 'Mistral-Large-Instruct-2411': {'p1': 13.3, 'p16': 11.1},
    'Qwen2.5-7B-Instruct': {'p1': 11.1, 'p16': 11.2}, 'Llama-3.1-70B-Instruct': {'p1': 15.6, 'p16': 23.5},
    'Gemini-1.5-Pro-Latest': {'p1': 13.3, 'p16': 26.5}, 'Qwen2.5-72B-Instruct': {'p1': 13.3, 'p16': 16.3},
    'Llama-3.3-70B-Instruct': {'p1': 22.2, 'p16': 28.7}, 'Qwen2.5-Max': {'p1': 22.2, 'p16': 25.2},
    'DeepSeek-Math-7b-RL': {'p1': 2.2, 'p16': 4.4}, 'NuminaMath-72B-CoT': {'p1': 2.2, 'p16': 4.8},
    'Qwen2.5-Math-7B-Instruct': {'p1': 11.1, 'p16': 8.5}, 'Qwen2.5-Math-72B-Instruct': {'p1': 20.0, 'p16': 24.8},
    'Skywork-o1': {'p1': 11.1, 'p16': 13.6}, 'DeepSeek Distill Qwen-1.5B': {'p1': 17.8, 'p16': 41.6},
    'QwQ-32B-Preview': {'p1': 44.4, 'p16': 59.3}, 'DeepSeek Distill LLama-8B': {'p1': 44.4, 'p16': 72.6},
    'DeepSeek Distill Qwen-7B': {'p1': 44.4, 'p16': 73.9}, 'OpenAI o1-mini': {'p1': 60.3, 'p16': 80.0},
    'DeepSeek Distill Qwen-14B': {'p1': 62.2, 'p16': 79.3}, 'DeepSeek Distill LLama-70B': {'p1': 62.2, 'p16': 77.0},
    'DeepSeek Distill Qwen-32B': {'p1': 62.2, 'p16': 79.7}
}

# 数据源 3: AIME25 准确率
aime25_acc = {
    'Llama-3.1-8B-Instruct': {'p1': 0.0, 'p16': 8.9}, 'Gemma-2-27B-it': {'p1': 0.0, 'p16': 9.5},
    'InternLM-3-8B-Instruct': {'p1': 13.3, 'p16': 30.3}, 'Qwen2.5-32B-Instruct': {'p1': 20.0, 'p16': 33.3},
    'Qwen2.5-7B-Instruct': {'p1': 6.7, 'p16': 25.2}, 'Llama-3.1-70B-Instruct': {'p1': 6.7, 'p16': 21.3},
    'Gemini-1.5-Pro-Latest': {'p1': 20.0, 'p16': 40.9}, 'Qwen2.5-72B-Instruct': {'p1': 20.0, 'p16': 33.2},
    'Llama-3.3-70B-Instruct': {'p1': 6.7, 'p16': 13.6}, 'Qwen2.5-Max': {'p1': 13.3, 'p16': 39.9},
    'Mistral-Large-Instruct-2411': {'p1': 13.3, 'p16': 19.7}, 'DeepSeek-Math-7b-RL': {'p1': 0.0, 'p16': 0.0},
    'NuminaMath-72B-CoT': {'p1': 0.0, 'p16': 21.0}, 'Qwen2.5-Math-7B-Instruct': {'p1': 20.0, 'p16': 36.8},
    'Qwen2.5-Math-72B-Instruct': {'p1': 13.3, 'p16': 30.1}, 'Skywork-o1': {'p1': 13.3, 'p16': 31.2},
    'DeepSeek Distill Qwen-1.5B': {'p1': 26.7, 'p16': 54.6}, 'QwQ-32B-Preview': {'p1': 26.7, 'p16': 60.5},
    'DeepSeek Distill LLama-8B': {'p1': 40.0, 'p16': 62.2}, 'DeepSeek Distill Qwen-7B': {'p1': 46.7, 'p16': 62.1},
    'OpenAI o1-mini': {'p1': 46.7, 'p16': 62.0}, 'DeepSeek Distill Qwen-14B': {'p1': 46.7, 'p16': 67.7},
    'DeepSeek Distill LLama-70B': {'p1': 46.7, 'p16': 75.1}, 'DeepSeek Distill Qwen-32B': {'p1': 46.7, 'p16': 72.6},
    'Yi-1.5-34B-Chat': {'p1':0.0, 'p16':14.8}
}


# --- 2. 创建主 DataFrame 并计算所有衍生指标 ---
data = []
for model, params in params_data.items():
    if model in aime24_acc and model in aime25_acc:
        data.append({
            'Model': model, 'Params': params,
            'AIME24_P1': aime24_acc[model]['p1'], 'AIME25_P1': aime25_acc[model]['p1'],
            'AIME24_P16': aime24_acc[model]['p16'], 'AIME25_P16': aime25_acc[model]['p16']
        })
df = pd.DataFrame(data)

# 基础留存率计算
df['Rg'] = df.apply(lambda r: r['AIME25_P1'] / r['AIME24_P1'] if r['AIME24_P1'] > 0 else 0, axis=1)
df['Ru'] = df.apply(lambda r: r['AIME25_P16'] / r['AIME24_P16'] if r['AIME24_P16'] > 0 else 0, axis=1)

# 新增计算：|Ru - Rg|, |100 - Ru|, |100 - Rg| (结果放大100倍以对应百分比含义)
df['Delta_R'] = (df['Ru'] - df['Rg']).abs() * 100
df['Loss_Ru'] = (1 - df['Ru']).abs() * 100
df['Loss_Rg'] = (1 - df['Rg']).abs() * 100

# --- 3. 通用绘图函数 ---
def generate_evolution_plot_v2(dataframe, acc_col, metric_col, title_prefix, metric_label, filename):
    df_plot = dataframe[['Params', acc_col, metric_col]].copy()
    df_plot.rename(columns={acc_col: 'AIME25 准确率', metric_col: metric_label}, inplace=True)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)

    UPPER_ANNOT_SIZE = 22
    TICK_LABEL_SIZE = 16

    # 阶段性分析 (子图 1-3)
    ranges = [("≤ 7B", df_plot['Params'] <= 7), ("> 32B", df_plot['Params'] > 32), ("> 72B", df_plot['Params'] > 72)]
    for i, (name, mask) in enumerate(ranges):
        ax = fig.add_subplot(gs[0, i])
        corr = df_plot[mask][['AIME25 准确率', metric_label]].corr()
        sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, ax=ax, cbar=False, annot_kws={"size": UPPER_ANNOT_SIZE}, fmt=".2f")
        ax.set_title(f"阶段 {i+1}：{name}", fontsize=16, fontweight='bold')
        ax.tick_params(labelsize=TICK_LABEL_SIZE)

    # 底部全景演变图
    ax_evo = fig.add_subplot(gs[1, :])
    thresholds = [7, 8, 14, 27, 32, 70, 72, 100, 123, 175]
    evo_data = []
    for t in thresholds:
        low, high = df_plot[df_plot['Params'] <= t], df_plot[df_plot['Params'] > t]
        c_low = low.corr().iloc[0, 1] if len(low) > 2 else np.nan
        c_high = high.corr().iloc[0, 1] if len(high) > 2 else np.nan
        evo_data.append({'阈值': f"{t}B", '规模 ≤ 阈值': c_low, '规模 > 阈值': c_high})

    evo_df = pd.DataFrame(evo_data).set_index('阈值').T
    sns.heatmap(evo_df, annot=True, cmap='RdBu_r', center=0, fmt=".2f", linewidths=1, ax=ax_evo, annot_kws={"size": 16})
    fig.suptitle(f"{title_prefix}：AIME25 准确率与指标的相关性演变", fontsize=24, fontweight='bold')
    ax_evo.set_xlabel("参数规模阈值 (B)", fontsize=16)
    ax_evo.set_ylabel("分析分组", fontsize=16)
    ax_evo.tick_params(labelsize=TICK_LABEL_SIZE)
    # 保存路径：asset/heatmap
    save_dir = 'asset/heatmap'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.savefig(f'{save_dir}/{filename}.png', dpi=300)
    plt.close()

# --- 4. 执行生成任务 ---
# 生成 |Ru - Rg| 演变图
generate_evolution_plot_v2(df, 'AIME25_P16', 'Delta_R', '差距 |Ru - Rg|', '|Ru - Rg|', 'delta_r_evolution')
# 生成 |100 - Ru| 演变图
generate_evolution_plot_v2(df, 'AIME25_P16', 'Loss_Ru', '损失 |100 - Ru|', '|100 - Ru|', 'loss_ru_evolution')
# 生成 |100 - Rg| 演变图
generate_evolution_plot_v2(df, 'AIME25_P1', 'Loss_Rg', '损失 |100 - Rg|', '|100 - Rg|', 'loss_rg_evolution')