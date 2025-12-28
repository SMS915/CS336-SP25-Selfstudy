import json
import os
from tqdm import tqdm

# --- 配置路径 ---
# 你的本地源文件路径
INPUT_FILE = "data/gsm8k/train.jsonl"
# 输出给评测脚本用的文件路径
OUTPUT_FILE = "data/gsm8k/train_clean.jsonl"

def process_gsm8k():
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到文件 {INPUT_FILE}")
        return

    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print(f"正在处理: {INPUT_FILE} ...")
    
    valid_count = 0
    total_count = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        
        # 读取所有行以显示进度条
        lines = fin.readlines()
        total_count = len(lines)
        
        for line in tqdm(lines, desc="Converting"):
            if not line.strip(): continue
            
            try:
                data = json.loads(line)
                
                # 1. 提取原始字段
                # GSM8K 原始字段通常是 "question" 和 "answer"
                question = data.get('question')
                raw_answer = data.get('answer')
                
                if not question or not raw_answer:
                    continue

                # 2. 核心逻辑：提取 #### 后的最终答案
                if "####" in raw_answer:
                    final_answer = raw_answer.split("####")[-1].strip()
                    # 去掉数字中的逗号 (例如 1,200 -> 1200)，便于后续数值比对
                    final_answer = final_answer.replace(",", "")
                else:
                    # 极少数情况如果没有分隔符，暂时保留原样或跳过
                    print(f"⚠️ Warning: No '####' found in answer: {raw_answer[:50]}...")
                    final_answer = raw_answer

                # 3. 构造新对象 (适配你的 run_evaluate.py)
                new_entry = {
                    "problem": question,          # 评测脚本用这个填 Prompt
                    "solution": raw_answer,       # 保留完整 CoT 以备查验
                    "answer": final_answer        # 评测脚本优先用这个做 Ground Truth
                }
                
                fout.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
                valid_count += 1
                
            except json.JSONDecodeError:
                print("Skipping invalid JSON line")
                continue

    print("-" * 40)
    print(f"✅ 转换完成！")
    print(f"输入: {total_count} 条")
    print(f"输出: {valid_count} 条")
    print(f"结果已保存至: {OUTPUT_FILE}")
    
    # 预览第一条
    if valid_count > 0:
        print("\n[数据预览]")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            first = json.loads(f.readline())
            print(f"Problem: {first['problem'][:50]}...")
            print(f"Clean Answer: {first['answer']}")

if __name__ == "__main__":
    process_gsm8k()