import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    # "axes.spines.top": False,
    # "axes.spines.right": False,
    # "grid.alpha": 0.3,
    # "grid.linestyle": "--"
})

plt.rcParams["font.sans-serif"] = ["SimHei"]

# 数据来源：表4 模型分任务推理行为统计
stages = ['BASELINE', 'SFT', 'GRPO变体平均', 'INSTRUCT']
datasets = ['GSM8K', 'MATH500', 'MATHTEST', 'AMC', 'AIME24', 'AIME25']

# 格式错误样本平均长度 (Fail Avg Token) 数据
# 注意：表中INSTRUCT在MATH500, AMC, AIME25显示为0.0或N/A，此处用0表示
data_fail = {
    'GSM8K': [249.8, 2048.0, 2018.0, 2047.9],
    'MATH500': [459.4, 3747.5, 3490.2, 0.0],
    'MATHTEST': [596.9, 2976.8, 3275.6, 3526.4],
    'AMC': [776.2, 3511.5, 3406.0, 0.0],
    'AIME24': [1225.3, 3429.8, 3338.2, 2905.0],
    'AIME25': [820.9, 3682.7, 3480.6, 0.0]
}

plt.figure(figsize=(12, 7))

markers = ['o', 's', '^', 'D', 'v', '>']


# 商务风
# 依次为：深蓝、橙色、绿色、红色、紫色、棕色 (经典的 Tab10 配色)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# 学术风
# 对应：天蓝、朱红、蓝绿、赤红、红紫、黄色
# colors = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#F0E442']

# 深邃风
# 对应：深蓝、土橘、草绿、深红、暗紫、褐色
# colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']

for i, dataset in enumerate(datasets):
    plt.plot(stages, data_fail[dataset], marker=markers[i], label=dataset, color=colors[i], linewidth=2, linestyle='--')

plt.title('格式错误样本平均长度演化 (Fail Avg Token)', fontsize=16)
plt.xlabel('模型演化阶段', fontsize=14)
plt.ylabel('平均 Token 长度', fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.grid(True, linestyle='--')
plt.legend(fontsize=12)
# 设置Y轴下限为0，以便清楚显示INSTRUCT的0值
plt.ylim(bottom=0)
plt.savefig('asset/LengthEvolution/format_wrong_length_evolution.png')
plt.show()