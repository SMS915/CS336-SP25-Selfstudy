from datasets import load_dataset
import json
import os

# 这是目前最标准的 AIME 2024 数据集源
DATASET_NAMES = ["HuggingFaceH4/aime_2024","HuggingFaceH4/aime_2025"]
OUTPUT_FILENAMES = ["data/AIME/aime2024_test.jsonl", "data/AIME/aime2025_test.jsonl"]

def prepare_aime(dataset_name: str, output_filename: str):
    print(f"正在下载 {dataset_name} ...")
    try:
        # 数据集只有 train split 30 道题
        ds = load_dataset(output_filename, split="train")
    except Exception as e:
        print(f"下载失败: {e}")
        return

    os.makedirs("data/AIME", exist_ok=True)
    
    print(f"正在转换 {len(ds)} 条数据...")
    with open(output_filename, 'w', encoding='utf-8') as f:
        for item in ds:
            entry = {
                "problem": item['problem'],
                "answer": item['answer'], 
                "solution": item['solution'],
                "id": item.get('id', '')
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"转换完成: {output_filename}")

if __name__ == "__main__":
    for dateset_name, output_filename in zip(DATASET_NAMES, OUTPUT_FILENAMES):
        prepare_aime(dateset_name, output_filename)