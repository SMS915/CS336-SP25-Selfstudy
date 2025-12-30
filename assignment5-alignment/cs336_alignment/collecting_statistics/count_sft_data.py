import json
import numpy as np
from transformers.models.auto.tokenization_auto import AutoTokenizer
from tqdm import tqdm

# 配置
TRAIN_DATA_PATH = "data/MATH/sft_filtered_4096.jsonl"  # 你的训练数据路径
MODEL_PATH = "models/Qwen2.5-Math-1.5B"
PROMPT_TEMPLATE_PATH = "cs336_alignment/prompts/r1_zero.prompt"

def check_data_length():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    with open(PROMPT_TEMPLATE_PATH, 'r') as f:
        template = f.read()

    lengths = []
    
    print("正在统计训练数据长度...")
    with open(TRAIN_DATA_PATH, 'r') as f:
        # 如果是 Bespoke-Stratos 原始格式可能略有不同，这里假设是你转换好的 {prompt, response} 或 {problem, solution}
        # 这里按你代码里的 SFTDataset 格式假设：
        for line in tqdm(f):
            item = json.loads(line)
            # 兼容不同的字段名
            prompt_content = item.get("prompt") or item.get("problem")
            response_content = item.get("response") or item.get("solution")
            
            # 拼接完整 Prompt
            if "{question}" in template:
                full_prompt = template.replace("{question}", prompt_content)
            else:
                full_prompt = prompt_content
                
            full_text = full_prompt + response_content
            
            # 计算 Token 数
            ids = tokenizer.encode(full_text, add_special_tokens=True)
            lengths.append(len(ids))

    lengths = np.array(lengths)
    p95 = np.percentile(lengths, 95)
    p99 = np.percentile(lengths, 99)
    p100 = np.max(lengths)

    print("\n" + "="*40)
    print("训练数据 Token 长度统计")
    print("="*40)
    print(f"平均长度: {np.mean(lengths):.2f}")
    print(f"95% 分位数: {p95:.2f}")
    print(f"99% 分位数: {p99:.2f}")
    print(f"最大长度: {p100}")
    print(f"当前设置 (4096) 覆盖率: {np.sum(lengths <= 4096) / len(lengths) * 100:.2f}%")
    print("="*40)

    if p99 > 4096:
        print("⚠️ 警告：超过 1% 的数据会被截断，建议增加 max_length！")
    else:
        print("✅ 4096 是安全的，无需增加。")

if __name__ == "__main__":
    check_data_length()