import json
import re
from tqdm import tqdm

# --- 配置 ---
INPUT_FILE = "data/CoT/sft_v4.jsonl"  # 基于你之前合并了solution的版本
OUTPUT_FILE = "data/CoT/sft_v5_platinum.jsonl"  # 白金版数据
BAD_FILE = "data/CoT/sft_v5_rejected.jsonl"  # 被剔除的数据（用于分析）

# 阈值设置
MAX_ANSWER_LEN = 100  # <answer> 标签内的内容不应超过这个长度
FORBIDDEN_IN_THINK = [r"\\boxed\{"]  # <think> 里不该出现的东西


def clean_and_filter(data):
    response = data.get('response', '')

    # 1. 提取标签内容
    # 假设已经是标准格式：<think>...</think><answer>...</answer>
    try:
        think_start = response.find("<think>") + 7
        think_end = response.find("</think>")
        answer_start = response.find("<answer>") + 8
        answer_end = response.find("</answer>")

        if think_start == 6 or think_end == -1 or answer_start == 7 or answer_end == -1:
            return False, "broken_tags", response

        think_content = response[think_start:think_end]
        answer_content = response[answer_start:answer_end]
    except Exception:
        return False, "parse_error", response

    # 2. 检查 Thinking 纯净度 (No Early Boxed)
    # R1 有时候会在思考过程中写 \boxed，这其实是可以接受的（作为中间步骤），
    # 但如果你追求极致的“思考-结论”分离，可以过滤掉。
    # 稍微放宽一点：如果 think 的最后 10% 内容里出现了 boxed，说明它在思考里就把题结了。
    if "\\boxed" in think_content:
        # 策略：如果 think 里有 boxed，我们认为这可能导致模型偷懒。
        # 但为了不误杀太多（因为 math 经常用 boxed 标记中间结果），
        # 我们可以检查是否 think_content 的末尾就是 boxed。
        # 这里采取严厉策略：直接拒绝，或者统计一下。
        return False, "boxed_in_think", response

    # 3. 检查 Answer 纯净度 (Conciseness)
    # 策略：Answer 必须短，且不能包含换行符（换行意味着废话多）
    if len(answer_content) > MAX_ANSWER_LEN:
        # 尝试修复：只提取 boxed 部分
        boxed_match = re.search(r"(\\boxed\{.*?\})", answer_content)
        if boxed_match:
            # 修复：重写 answer 部分，只保留 boxed
            clean_answer = boxed_match.group(1)
            # 重组 response
            new_response = f"<think>{think_content}</think>\n<answer>{clean_answer}</answer>"
            data['response'] = new_response
            return True, "fixed_verbose_answer", new_response
        else:
            return False, "verbose_answer_no_boxed", response

    if "\n" in answer_content.strip():
        # 尝试修复：去除换行
        clean_answer = answer_content.replace("\n", " ").strip()
        new_response = f"<think>{think_content}</think>\n<answer>{clean_answer}</answer>"
        data['response'] = new_response
        return True, "fixed_newline_answer", new_response

    return True, "clean", response


# --- 执行 ---
stats = {}
valid_data = []

print("开始构建白金版数据集...")
with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
        open(BAD_FILE, 'w', encoding='utf-8') as fbad:
    for line in tqdm(fin):
        data = json.loads(line)
        is_ok, reason, new_response = clean_and_filter(data)

        stats[reason] = stats.get(reason, 0) + 1

        if is_ok:
            valid_data.append(data)
        else:
            fbad.write(json.dumps(data, ensure_ascii=False) + "\n")

# 写入通过的数据
with open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
    for data in valid_data:
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")

print("\n" + "=" * 40)
print(f"原始数据: {sum(stats.values())}")
print(f"白金数据: {len(valid_data)}")
print(f"保留率: {len(valid_data) / sum(stats.values()):.2%}")
print("-" * 40)
print("统计分布:")
for k, v in sorted(stats.items(), key=lambda x: x[1], reverse=True):
    print(f"  {k}: {v}")