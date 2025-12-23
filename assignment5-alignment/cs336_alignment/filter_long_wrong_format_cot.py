import json
from transformers import AutoTokenizer
from typing import Tuple
from tqdm import tqdm

# 配置
INPUT_FILE = "data/MATH/sft.jsonl"  # 你的 v2 数据源
OUTPUT_FILE = "data/MATH/sft_v3_high_quality.jsonl"
MODEL_PATH = "/models/Qwen2.5-Math-1.5B"  # 用于计算 token 长度
MAX_LEN = 2048
MIN_LEN = 300  # 避免过短的劣质 CoT

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


def check_quality(data) -> Tuple[bool, str]:
    prompt = data['prompt']
    response = data['response']
    full_text = prompt + response

    # 1. 长度检查 (Token 级)
    tokens = tokenizer.encode(full_text)
    if len(tokens) > MAX_LEN or len(tokens) < MIN_LEN:
        return False, "length_mismatch"

    # 2. 标签存在性检查
    tags = ["<think>", "</think>", "<answer>", "</answer>"]
    for tag in tags:
        count = response.count(tag)
        if count == 0:
            return False, f"missing_{tag}"
        if count > 1:
            return False, f"duplicate_{tag}"  # 拒绝标签重复的样本

    # 3. 标签顺序检查
    # find() 返回索引，必须保证索引递增
    t_start = response.find("<think>")
    t_end = response.find("</think>")
    a_start = response.find("<answer>")
    a_end = response.find("</answer>")

    if not (t_start < t_end < a_start < a_end):
        return False, "wrong_order"

    # 4. Boxed 检查
    # 提取 answer 标签内的内容
    answer_content = response[a_start + 8: a_end]
    if "\\boxed" not in answer_content:
        return False, "no_boxed"

    return True, "pass"


valid_count = 0
total_count = 0
reject_stats = {}

with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
        open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
    for line in tqdm(fin):
        total_count += 1
        data = json.loads(line)

        is_valid, reason = check_quality(data)

        if is_valid:
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            valid_count += 1
        else:
            reject_stats[reason] = reject_stats.get(reason, 0) + 1

print(f"原始数据: {total_count}")
print(f"保留数据: {valid_count} (保留率: {valid_count / total_count:.2%})")
print("拒绝原因统计:", reject_stats)