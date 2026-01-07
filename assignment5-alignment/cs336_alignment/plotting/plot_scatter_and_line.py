import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data
models = [
    'DS-Math-7B-RL', 'Claude 3 Opus', 'GPT-4o-2024-05', 'Numina-72B-CoT', 'Llama-3.1-70B-I',
    'OpenAI o1-mini', 'Qwen2.5-7B-I', 'Qwen2.5-32B-I', 'Qwen2.5-72B-I', 'Qwen2.5-Math-7B-I',
    'Qwen2.5-Math-72B-I', 'Mistral-Large-2411', 'Skywork-o1', 'QWQ-32B-Preview', 'Llama-3.3-70B-I',
    'InternLM-3-8B-I', 'DS Distill 1.5B', 'DS Distill 32B', 'Qwen2.5-Max'
]
pass1_acc = [2.2, 21.1, 26.7, 2.2, 15.6, 60.3, 11.1, 11.1, 13.3, 11.1, 20.0, 13.3, 11.1, 44.4, 22.2, 11.1, 17.8, 62.2, 22.2]
greedy_r = [0.0, 29.7, 42.9, 0.0, 42.9, 77.4, 60.4, 180.2, 150.4, 180.2, 66.5, 100.0, 119.8, 60.1, 30.2, 119.8, 150.0, 75.1, 59.9]

# Plot
plt.figure(figsize=(10, 8))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Scatter
for i, model in enumerate(models):
    color = 'tab:red' if any(x in model for x in ['Qwen', 'QWQ', 'Distill']) else 'tab:blue'
    plt.scatter(pass1_acc[i], greedy_r[i], c=color, s=100, edgecolors='black', alpha=0.7)
    plt.text(pass1_acc[i]+0.8, greedy_r[i], model, fontsize=9)

# The "1/3 Slope" Line: y = 1/3 * x + 200/3
x_vals = np.array([0, 100])
y_vals = (1/3) * x_vals + (200/3)
plt.plot(x_vals, y_vals, color='green', linestyle='--', linewidth=2, label='泛化理想演化线 (斜率 ~ 1/3)')

# Aesthetics
plt.axhline(y=100, color='gray', linestyle=':', alpha=0.5, label='100% 理想稳定线')
plt.xlabel('Greedy Accuracy (Pass@1) %', fontsize=12)
plt.ylabel('Greedy Retention Rate (Rg) %', fontsize=12)
plt.title('推理能力的“一体两面”：Rg 与 Accuracy 的分布及理想斜率线', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-5, 75)
plt.ylim(-10, 200)

plt.savefig('retention_with_ideal_slope.png')
print("Plot saved as retention_with_ideal_slope.png")