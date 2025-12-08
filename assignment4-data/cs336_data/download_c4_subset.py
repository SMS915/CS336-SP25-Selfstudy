import json
from datasets import load_dataset
from tqdm import tqdm

output_file = "data/c4_valid_subset.jsonl"
num_samples = 2000  # 取 2000 条足以做初步 PPL 验证

# 加载官方 validation split
ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)

print(f"正在抽取前 {num_samples} 条数据保存到 {output_file} ...")

with open(output_file, 'w', encoding='utf-8') as f:
    for i, example in tqdm(enumerate(ds), total=num_samples):
        if i >= num_samples:
            break
        # 写入 jsonl
        json.dump(example, f, ensure_ascii=False)
        f.write('\n')

print("完成！现在你有一个固定的、轻量级的 C4 验证集了。")