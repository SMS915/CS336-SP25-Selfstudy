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

# 成功样本平均长度 (Succ Avg Token) 数据
data_success = {
    'GSM8K': [122.8, 801.6, 559.4, 291.8],
    'MATH500': [179.4, 1198.2, 933.1, 532.9],
    'MATHTEST': [162.5, 1136.7, 896.2, 515.6],
    'AMC': [215.1, 1371.1, 1208.1, 757.4],
    'AIME24': [420.2, 1423.9, 1373.4, 959.6],
    'AIME25': [640.0, 1409.4, 1319.8, 896.6]
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
    plt.plot(stages, data_success[dataset], marker=markers[i], label=dataset, color=colors[i], linewidth=2)

plt.title('成功样本平均长度演化 (Succ Avg Token)', fontsize=16)
plt.xlabel('模型演化阶段', fontsize=14)
plt.ylabel('平均 Token 长度', fontsize=14)
plt.tick_params(axis='both', labelsize=14)
plt.grid(True, linestyle='--')
plt.legend(fontsize=12)
plt.savefig('asset/LengthEvolution/correct_length_evolution.png')
plt.show()