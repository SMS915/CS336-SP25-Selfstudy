import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 数据准备 ---
models = [
    'DS-Math-7B-RL', 'Claude 3 Opus', 'GPT-4o-2024-05', 'Numina-72B-CoT', 'Llama-3.1-70B-I',
    'OpenAI o1-mini', 'Qwen2.5-7B-I', 'Qwen2.5-32B-I', 'Qwen2.5-72B-I', 'Qwen2.5-Math-7B-I',
    'Qwen2.5-Math-72B-I', 'Mistral-Large-2411', 'Skywork-o1', 'QWQ-32B-Preview', 'Llama-3.3-70B-I',
    'InternLM-3-8B-I', 'DS Distill 1.5B', 'DS Distill 32B', 'Qwen2.5-Max'
]

pass1_acc = [2.2, 21.1, 26.7, 2.2, 15.6, 60.3, 11.1, 11.1, 13.3, 11.1, 20.0, 13.3, 11.1, 44.4, 22.2, 11.1, 17.8, 62.2, 22.2]
passk_acc = [16.3, 34.1, 43.3, 21.3, 41.2, 86.7, 26.3, 32.0, 33.7, 20.8, 35.2, 15.4, 22.1, 74.3, 37.1, 20.5, 68.7, 86.3, 44.4]
greedy_r_list = [0.0, 29.7, 42.9, 0.0, 42.9, 77.4, 60.4, 180.2, 150.4, 180.2, 66.5, 100.0, 119.8, 60.1, 30.2, 119.8, 150.0, 75.1, 59.9]
upper_r_list = [0.0, 38.2, 55.4, 98.6, 51.7, 71.5, 95.8, 104.1, 98.5, 176.9, 85.5, 127.9, 141.2, 81.4, 36.7, 147.8, 79.5, 84.1, 89.9]


df_all = pd.DataFrame({
    'Model': models,
    'Pass@1_准确率': pass1_acc,
    'Pass@k_准确率': passk_acc,
    'Greedy留存率_Rg': greedy_r_list,
    'Upper留存率_Ru': upper_r_list
})

qwen_mask = df_all['Model'].str.contains('Qwen|QWQ|Distill', case=False)
corrs = [df_all.drop(columns='Model').corr(),
         df_all[qwen_mask].drop(columns='Model').corr(),
         df_all[~qwen_mask].drop(columns='Model').corr()]

titles = ['全样本相关性对比', '仅 Qwen 系列相关性', '排除 Qwen 系列相关性']

# --- 2. 绘图 ---
fig = plt.figure(figsize=(22, 8)) # 稍微收窄画布宽度
gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.04])

axes = [plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[2])]
cbar_ax = plt.subplot(gs[3])

heatmap_kwargs = {
    'annot': True,
    'cmap': 'RdBu_r',
    'center': 0,
    'fmt': ".2f",
    'linewidths': 0.8,
    'annot_kws': {"size": 11},
    'vmin': -1, 'vmax': 1
}

for i in range(3):
    sns.heatmap(corrs[i], ax=axes[i],
                cbar=(i == 2), cbar_ax=(cbar_ax if i == 2 else None),
                **heatmap_kwargs)
    axes[i].set_title(titles[i], fontsize=16, pad=20)
    plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')
    plt.setp(axes[i].get_yticklabels(), rotation=0)

plt.suptitle('模型推理能力与留存率的相关性对比分析', fontsize=22, y=0.98)

# 调整间距：wspace 压缩至 0.35
plt.subplots_adjust(top=0.82, bottom=0.25, wspace=0.37, right=0.94, left=0.06)

plt.savefig('asset/compressed_acc_vs_R_heatmaps.png', bbox_inches='tight')
plt.show()