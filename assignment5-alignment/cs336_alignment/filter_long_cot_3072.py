import json
import argparse
import os
import numpy as np
from transformers.models.auto.tokenization_auto import AutoTokenizer
from tqdm import tqdm

def preprocess_dataset(
    input_path, 
    output_path, 
    model_path, 
    prompt_template_path, 
    max_length=4096,
    target_samples=None  # 如果限制最终数量
):
    print(f"Loading Tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print(f"Loading Prompt Template from {prompt_template_path}...")
    with open(prompt_template_path, 'r') as f:
        template = f.read()

    print(f"Reading data from {input_path}...")
    raw_data = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line))
    
    print(f"原始数据共 {len(raw_data)} 条。开始筛选 (Max Length <= {max_length})...")
    
    kept_data = []
    lengths = []
    dropped_count = 0
    
    # 进度条
    for item in tqdm(raw_data):
        # 兼容不同的字段名
        prompt = item.get("prompt") or item.get("problem")
        response = item.get("response") or item.get("solution")
        
        if not prompt or not response:
            continue

        # 1. 拼接完整文本 (模拟训练时的输入)
        # 注意：这里要和训练时的 collate_fn 逻辑一致
        if "{question}" in template:
            formatted_prompt = template.replace("{question}", prompt)
        else:
            formatted_prompt = prompt
            
        full_text = formatted_prompt + response
        
        # 2. 计算 Token 长度
        # add_special_tokens=True 会加上 BOS，更接近真实长度
        ids = tokenizer.encode(full_text, add_special_tokens=True)
        length = len(ids)
        
        # 3. 筛选逻辑
        # 留一点余量 (比如 50 tokens) 给 EOS 或其他可能的 padding
        if length <= (max_length - 50):
            lengths.append(length)
            kept_data.append(item)
        else:
            dropped_count += 1
            
        # 如果设置了目标数量，且已经凑够了，可以提前退出
        # if target_samples and len(kept_data) >= target_samples:
        #     break

    # 如果有目标数量限制，进行截取（通常为了做实验）
    if target_samples and len(kept_data) > target_samples:
        print(f"筛选出 {len(kept_data)} 条，按要求截取前 {target_samples} 条。")
        kept_data = kept_data[:target_samples]
        lengths = lengths[:target_samples]

    # 保存文件
    print(f"\n正在保存至 {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in kept_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 打印统计
    lengths = np.array(lengths)
    print("\n" + "="*40)
    print("预处理统计报告")
    print("="*40)
    print(f"原始总数:   {len(raw_data)}")
    print(f"保留总数:   {len(kept_data)} (保留率: {len(kept_data)/len(raw_data):.2%})")
    print(f"丢弃总数:   {dropped_count}")
    print("-" * 40)
    if len(lengths) > 0:
        print(f"保留数据平均长度: {np.mean(lengths):.2f}")
        print(f"保留数据最大长度: {np.max(lengths)}")
        print(f"保留数据最小长度: {np.min(lengths)}")
    print("="*40)
    print(f"✅ 文件已准备好：{output_path}")

if __name__ == "__main__":
    # 在这里配置你的路径
    INPUT_FILE = "data/MATH/sft.jsonl"       # 建议换成全量 17k 文件路径
    OUTPUT_FILE = "data/MATH/sft_filtered.jsonl"
    MODEL_DIR = "models/Qwen2.5-Math-1.5B"
    TEMPLATE_FILE = "cs336_alignment/prompts/r1_zero.prompt"
    MAX_LEN = 3072
    
    preprocess_dataset(INPUT_FILE, OUTPUT_FILE, MODEL_DIR, TEMPLATE_FILE, MAX_LEN)