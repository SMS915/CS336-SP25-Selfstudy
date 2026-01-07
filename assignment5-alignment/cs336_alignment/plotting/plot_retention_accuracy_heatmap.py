import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

# --- 0. 环境设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, title: str, filename: str):
    """
    通用热力图绘制函数
    """
    heatmap_args = {
        'annot': True,
        'cmap': 'RdBu_r',
        'center': 0,
        'fmt': ".2f",
        'linewidths': 0.8,
        'annot_kws': {"size": 14},
        'vmin': -1, 'vmax': 1
    }

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, **heatmap_args)

    plt.title(title, fontsize=18, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()

    # 确保保存目录存在
    save_path = os.path.join('asset', filename)
    if not os.path.exists('asset'):
        os.makedirs('asset')

    plt.savefig(save_path, dpi=300)
    print(f"图像已保存至: {save_path}")
    plt.show()


def get_data():
    """
    初始化数据并返回 DataFrame
    """
    models = [
        'DS-Math-7B-RL', 'Claude 3 Opus', 'GPT-4o-2024-05', 'Numina-72B-CoT', 'Llama-3.1-70B-I',
        'OpenAI o1-mini', 'Qwen2.5-7B-I', 'Qwen2.5-32B-I', 'Qwen2.5-72B-I', 'Qwen2.5-Math-7B-I',
        'Qwen2.5-Math-72B-I', 'Mistral-Large-2411', 'Skywork-o1', 'QWQ-32B-Preview', 'Llama-3.3-70B-I',
        'InternLM-3-8B-I', 'DS Distill 1.5B', 'DS Distill 32B', 'Qwen2.5-Max'
    ]
    pass1_acc = [2.2, 21.1, 26.7, 2.2, 15.6, 60.3, 11.1, 11.1, 13.3, 11.1, 20.0, 13.3, 11.1, 44.4, 22.2, 11.1, 17.8,
                 62.2, 22.2]
    passk_acc = [16.3, 34.1, 43.3, 21.3, 41.2, 86.7, 26.3, 32.0, 33.7, 20.8, 35.2, 15.4, 22.1, 74.3, 37.1, 20.5, 68.7,
                 86.3, 44.4]
    greedy_r_list = [0.0, 29.7, 42.9, 0.0, 42.9, 77.4, 60.4, 180.2, 150.4, 180.2, 66.5, 100.0, 119.8, 60.1, 30.2, 119.8,
                     150.0, 75.1, 59.9]
    upper_r_list = [0.0, 38.2, 55.4, 98.6, 51.7, 71.5, 95.8, 104.1, 98.5, 176.9, 85.5, 127.9, 141.2, 81.4, 36.7, 147.8,
                    79.5, 84.1, 89.9]

    df = pd.DataFrame({
        'Model': models,
        'Pass@1_准确率': pass1_acc,
        'Pass@k_准确率': passk_acc,
        'Greedy留存率_Rg': greedy_r_list,
        'Upper留存率_Ru': upper_r_list
    })
    return df


def main():
    # 配置命令行参数
    parser = argparse.ArgumentParser(description='绘制模型相关性热力图')
    parser.add_argument('--type', type=str, choices=['all', 'qwen', 'non_qwen', 'run_all'],
                        default='all', help='选择要分析的样本类型 (default: all)')

    args = parser.parse_args()
    df_all = get_data()

    # 定义 Qwen 系列的特征词
    qwen_keywords = 'Qwen|QWQ|Distill'
    qwen_mask = df_all['Model'].str.contains(qwen_keywords, case=False)

    # 处理逻辑
    tasks = []
    if args.type == 'all':
        tasks.append((df_all, '全样本相关性对比', 'full_sample_correlation_heatmap.png'))
    elif args.type == 'qwen':
        tasks.append((df_all[qwen_mask], 'Qwen系列相关性对比', 'qwen_sample_correlation_heatmap.png'))
    elif args.type == 'non_qwen':
        tasks.append((df_all[~qwen_mask], '非Qwen系列相关性对比', 'non_qwen_sample_correlation_heatmap.png'))
    elif args.type == 'run_all':
        tasks = [
            (df_all, '全样本相关性对比', 'full_sample_correlation_heatmap.png'),
            (df_all[qwen_mask], 'Qwen系列相关性对比', 'qwen_sample_correlation_heatmap.png'),
            (df_all[~qwen_mask], '非Qwen系列相关性对比', 'non_qwen_sample_correlation_heatmap.png')
        ]

    # 执行绘图
    for data, title, filename in tasks:
        # 去除非数值列计算相关性
        corr = data.drop(columns='Model').corr()
        plot_correlation_heatmap(corr, title, filename)


if __name__ == '__main__':
    main()