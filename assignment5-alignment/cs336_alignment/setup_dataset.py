import os
import json
from datasets import load_dataset
from huggingface_hub import login

# 如果你需要设置镜像
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def prepare_math_dataset():
    dataset_name = "xDAN2099/lighteval-MATH"
    print(f"正在下载 {dataset_name} ...")
    
    try:
        # 加载数据集
        dataset = load_dataset(dataset_name)
    except Exception as e:
        print(f"❌ 下载出错: {e}")
        return

    # 定义输出目录
    output_dir = "data/MATH"
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. 处理验证集 (Test Split -> validation.jsonl) ---
    # 作业 Baseline 需要用这 5000 条数据
    print("正在处理 Validation set (原 test split)...")
    val_data = dataset["test"]
    val_path = os.path.join(output_dir, "validation.jsonl")
    
    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            entry = {
                "problem": item["problem"],
                "solution": item["solution"],
                "level": item["level"],
                "type": item["type"]
            }
            f.write(json.dumps(entry) + "\n")
    print(f"✅ 已保存: {val_path} (共 {len(val_data)} 条)")

    # --- 2. 处理训练集 (Train Split -> train.jsonl) ---
    # 后续步骤会用到
    print("正在处理 Train set...")
    train_data = dataset["train"]
    train_path = os.path.join(output_dir, "train.jsonl")
    
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            entry = {
                "problem": item["problem"],
                "solution": item["solution"],
                "level": item["level"],
                "type": item["type"]
            }
            f.write(json.dumps(entry) + "\n")
    print(f"✅ 已保存: {train_path} (共 {len(train_data)} 条)")

if __name__ == "__main__":
    prepare_math_dataset()