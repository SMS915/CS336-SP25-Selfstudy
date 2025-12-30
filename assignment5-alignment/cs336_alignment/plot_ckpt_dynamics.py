import matplotlib.pyplot as plt
import pandas as pd
import re


# 1. 数据解析函数
def parse_log(filename, method_name):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    data = []
    blocks = content.split('----------------------------------------')
    for block in blocks:
        if '【项目】' not in block: continue
        try:
            step = int(re.search(r'step(\d+)', block).group(1))
            dataset = re.search(r'\| (aime2024|math500)', block).group(1)
            acc = float(re.search(r'正确率: ([\d.]+)%', block).group(1))
            err = float(re.search(r'格式错误率: ([\d.]+)%', block).group(1))
            avg_l = float(re.search(r'Token长度: 平均 ([\d.]+)', block).group(1))
            succ_l = float(re.search(r'成功样本平均Token: ([\d.]+)', block).group(1))
            fail_l = float(re.search(r'格式错误平均Token: ([\d.]+)', block).group(1))
            data.append({'method': method_name, 'step': step, 'dataset': dataset,
                         'acc': acc, 'err': err, 'avg_len': avg_l,
                         'succ_len': succ_l, 'fail_len': fail_l})
        except:
            continue
    return data


# 合并数据
all_data = parse_log('drgrpo_check_summary.txt', 'drgrpo') + \
           parse_log('grpo_no_std_norm_check_summary.txt', 'grpo_no_std_norm') + \
           parse_log('grpo_check_summary.txt', 'grpo')
df = pd.DataFrame(all_data).sort_values('step')


# 2. 绘图函数
def plot_metrics(dataset):
    sub = df[df['dataset'] == dataset]
    methods = sub['method'].unique()

    metrics = [('acc', 'Accuracy (%)'), ('err', 'Format Error Rate (%)'),
               ('avg_len', 'Avg Length'), ('succ_len', 'Success Avg Length'),
               ('fail_len', 'Format Error Avg Length')]

    # 分别画五张图
    for col, title in metrics:
        plt.figure(figsize=(8, 5))
        for m in methods:
            m_sub = sub[sub['method'] == m]
            plt.plot(m_sub['step'], m_sub[col], marker='o', label=m)
        plt.title(f'{title} vs Steps ({dataset})')
        plt.xlabel('Steps');
        plt.ylabel(title);
        plt.legend();
        plt.grid(True)
        plt.savefig(f'{col}_{dataset}.png')
        plt.close()

    # 概率合并图 (Acc & Err)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    for m in methods:
        m_sub = sub[sub['method'] == m]
        ax1.plot(m_sub['step'], m_sub['acc'], marker='o', label=f'{m} Acc')
        ax2.plot(m_sub['step'], m_sub['err'], marker='x', linestyle='--', label=f'{m} Err')
    ax1.set_ylabel('Accuracy (%)', color='blue');
    ax2.set_ylabel('Error Rate (%)', color='red')
    plt.title(f'Acc & Error Rate Comparison ({dataset})')
    ax1.legend(loc='upper left');
    ax2.legend(loc='upper right');
    plt.grid(True)
    plt.savefig(f'combined_prob_{dataset}.png');
    plt.close()

    # 三个长度合并图
    plt.figure(figsize=(10, 6))
    colors = {'drgrpo': 'r', 'grpo': 'g', 'grpo_no_std_norm': 'b'}
    for m in methods:
        m_sub = sub[sub['method'] == m]
        c = colors[m]
        plt.plot(m_sub['step'], m_sub['avg_len'], color=c, marker='o', label=f'{m} Total')
        plt.plot(m_sub['step'], m_sub['succ_len'], color=c, linestyle='--', marker='s', label=f'{m} Succ')
        plt.plot(m_sub['step'], m_sub['fail_len'], color=c, linestyle=':', marker='x', label=f'{m} Fail')
    plt.title(f'Length Evolution ({dataset})');
    plt.legend(bbox_to_anchor=(1.05, 1));
    plt.tight_layout()
    plt.grid(True);
    plt.savefig(f'combined_len_{dataset}.png');
    plt.close()


# 执行绘图
for ds in ['aime2024', 'math500']:
    plot_metrics(ds)