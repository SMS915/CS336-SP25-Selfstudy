from datasets import load_dataset
import json
import os

# 这是目前最标准的 AIME 2024 数据集源
DATASET_NAME = "HuggingFaceH4/aime_2024"
OUTPUT_FILE = "data/MATH/aime2024_test.jsonl"


def prepare_aime_2024():
    print(f"🚀 正在下载 {DATASET_NAME} ...")
    try:
        # 这个数据集通常只有 train split，其实就是那 30 道题
        ds = load_dataset(DATASET_NAME, split="train")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return

    os.makedirs("data/MATH", exist_ok=True)

    print(f"正在转换 {len(ds)} 条数据...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in ds:
            # 字段映射：
            # problem -> problem
            # answer -> solution (H4的库里 answer 通常是标准答案 0-999)

            entry = {
                "problem": item['problem'],
                # 注意：H4 数据集里 'answer' 是短答案，'solution' 是解析
                # 我们的评估脚本兼容两者，但为了保险，把 answer 字段填好
                "answer": item['answer'],
                "solution": item['answer'],  # 既然是整数，solution 放答案也没事
                "id": item.get('id', '')
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ 转换完成: {OUTPUT_FILE}")
    print("请使用以下命令评估:")
    print(
        f"python run_evaluate.py --example_path {OUTPUT_FILE} --output_path results/aime2024_pass64.jsonl --pass_k 64 ...")


if __name__ == "__main__":
    prepare_aime_2024()