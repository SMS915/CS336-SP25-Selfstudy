import pandas as pd

# 提取表 4 中的原始数据 (GRPO, GRPO_NO_STD_NORM, DRGRPO)
data = [
    # 格式: [任务, 模型, 样本数N, 正确率, 格式错误率, 平均长度, 成功样本长度, 格式错误长度]
    ["GSM8K", "GRPO", 1319, 80.36, 7.28, 643.1, 521.7, 1976.1],
    ["GSM8K", "GRPO_NO_STD_NORM", 1319, 80.59, 8.72, 691.4, 551.6, 2035.0],
    ["GSM8K", "DRGRPO", 1319, 79.15, 8.49, 742.7, 605.5, 2036.5],

    ["MATH500", "GRPO", 500, 92.60, 4.00, 1012.3, 904.4, 3584.2],
    ["MATH500", "GRPO_NO_STD_NORM", 500, 93.60, 3.40, 1040.6, 948.0, 3549.4],
    ["MATH500", "DRGRPO", 500, 92.00, 4.60, 1062.4, 946.7, 3364.6],

    ["MATHTEST", "GRPO", 5000, 84.68, 6.84, 1074.5, 870.4, 3438.6],
    ["MATHTEST", "GRPO_NO_STD_NORM", 5000, 85.14, 6.98, 1058.2, 896.1, 2868.3],
    ["MATHTEST", "DRGRPO", 5000, 84.36, 7.14, 1136.5, 922.3, 3517.6],

    ["AMC", "GRPO", 83, 75.90, 14.46, 1530.6, 1205.6, 3383.6],
    ["AMC", "GRPO_NO_STD_NORM", 83, 75.90, 14.46, 1530.7, 1217.6, 3097.2],
    ["AMC", "DRGRPO", 83, 74.70, 13.25, 1576.0, 1201.0, 3767.3],

    ["AIME24", "GRPO", 30, 30.00, 36.67, 2036.9, 1400.2, 3464.9],
    ["AIME24", "GRPO_NO_STD_NORM", 30, 40.00, 13.33, 1580.6, 1279.8, 2832.5],
    ["AIME24", "DRGRPO", 30, 40.00, 30.00, 1974.5, 1446.8, 3407.9],

    ["AIME25", "GRPO", 30, 30.00, 36.67, 2208.6, 1396.3, 3680.9],
    ["AIME25", "GRPO_NO_STD_NORM", 30, 30.00, 46.67, 2297.3, 1248.2, 3473.2],
    ["AIME25", "DRGRPO", 30, 36.67, 33.33, 1941.6, 1315.8, 3270.7],
]

columns = ["Dataset", "Model", "N", "Accuracy", "FormatErr", "AvgLen", "SuccAvgLen", "FailAvgLen"]
df = pd.DataFrame(data, columns=columns)


def calculate_averages(group):
    # 1. 计算简单平均的指标
    avg_acc = group["Accuracy"].mean()
    avg_fer = group["FormatErr"].mean()
    avg_len = group["AvgLen"].mean()

    # 2. 计算加权平均：成功样本长度
    # 成功样本数 = 总数 * 正确率 / 100
    succ_counts = group["N"] * group["Accuracy"] / 100
    weighted_succ_len = (group["SuccAvgLen"] * succ_counts).sum() / succ_counts.sum()

    # 3. 计算加权平均：格式错误长度
    # 格式错误样本数 = 总数 * 格式错误率 / 100
    fail_counts = group["N"] * group["FormatErr"] / 100
    weighted_fail_len = (group["FailAvgLen"] * fail_counts).sum() / fail_counts.sum()

    return pd.Series({
        "平均正确率 (%)": round(avg_acc, 2),
        "平均格式错误率 (%)": round(avg_fer, 2),
        "平均生成长度 (Tokens)": round(avg_len, 2),
        "加权成功样本长度": round(weighted_succ_len, 2),
        "加权格式错误长度": round(weighted_fail_len, 2)
    })


# 按测试集分组并应用计算
results = df.groupby("Dataset").apply(calculate_averages)

# 打印结果
print("三个 RL 变体在各测试中的平均指标汇总：")
print(results)