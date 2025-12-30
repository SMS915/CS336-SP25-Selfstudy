import os
import json
from datasets import load_dataset
from huggingface_hub import login

# 设置镜像
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def prepare_math_full_dataset():
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

    # 处理验证集 (Test Split -> validation.jsonl)
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

    #处理训练集 (Train Split -> train.jsonl)
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

def prepare_math_500_dataset(output_dir: str = "data/MATH", filename: str = "math500_test.jsonl"):
    """
    下载 HuggingFaceH4/math-500 数据集并保存为 JSONL 格式到本地。
    这个数据集包含了清洗过的 'answer' 字段，非常适合直接评测。
    """
    # 确保目录存在
    if not os.path.exists(output_dir):
        print(f"目录不存在，正在创建: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    print("正在从 HuggingFace 下载 HuggingFaceH4/math-500 ...")
    
    try:
        dataset = load_dataset("HuggingFaceH4/math-500", split="test")
    except Exception as e:
        print(f"❌ 下载出错: {e}")
        print("请检查网络连接，或确保已登录 huggingface-cli login")
        return

    output_path = os.path.join(output_dir, filename)
    print(f"正在处理并写入: {output_path}")

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset:
            # 该数据集包含: problem, solution, answer, subject, level
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1

    print("-" * 40)
    print(f"✅ 成功完成！")
    print(f"文件位置: {output_path}")
    print(f"样本数量: {count}")
    
    if count > 0:
        print("\n[数据示例]")
        first_item = dataset[0]
        print(f"Keys: {list(first_item.keys())}")
        print(f"Problem: {first_item['problem'][:50]}...")
        print(f"Answer (Clean): {first_item['answer']}")
        print(f"Solution (Full): {first_item['solution'][:50]}...")

if __name__ == "__main__":
    # prepare_math_full_dataset()
    prepare_math_500_dataset()