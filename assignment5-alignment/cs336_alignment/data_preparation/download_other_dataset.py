import json
import os
from datasets import load_dataset

# 配置
DATA_DIR = "data/OTHER_BENCHMARKS"
os.makedirs(DATA_DIR, exist_ok=True)

TASKS = {
    # "GSM8K": ("gsm8k", "main", "test"),
    "AMC12": ("AI-MO/aimo-validation-amc", None, "train"),
    "OmniMATH": ("KbsdJames/Omni-MATH", None, "test")
}


def prepare_benchmarks():
    for name, (path, config, split) in TASKS.items():
        print(f"⬇️  正在下载处理 {name} ...")
        try:
            ds = load_dataset(path, config, split=split)
            output_file = os.path.join(DATA_DIR, f"{name}.jsonl")

            with open(output_file, 'w', encoding='utf-8') as f:
                for item in ds:
                    entry = {}

                    # 1. GSM8K 格式适配
                    if name == "GSM8K":
                        entry["problem"] = item["question"]
                        # GSM8K solution 包含推导，answer 需要提取 #### 后的内容
                        entry["solution"] = item["answer"]
                        entry["answer"] = item["answer"].split("####")[-1].strip()

                    # 2. AMC / OmniMATH (AI-MO 格式)
                    else:
                        entry["problem"] = item.get("problem") or item.get("question")
                        entry["solution"] = item.get("solution") or item.get("answer")
                        # 尝试提取 answer 字段，如果没有则假设 solution 是答案
                        entry["answer"] = item.get("answer") or item.get("final_answer")

                    if entry["problem"]:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            print(f"✅ {name} 就绪: {output_file} ({len(ds)} 条)")

        except Exception as e:
            print(f"❌ {name} 处理失败: {e}")


if __name__ == "__main__":
    prepare_benchmarks()