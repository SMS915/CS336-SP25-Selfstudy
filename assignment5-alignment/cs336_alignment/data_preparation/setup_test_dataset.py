import json
import os
from cs336_alignment.hf_mirror import configure_hf_mirror

# --- 全局配置 ---
# 所有数据集存放的总目录
BASE_DATA_DIR = "data"

configure_hf_mirror()

from datasets import load_dataset
from tqdm import tqdm


def prepare_aime_datasets():
    """
    准备 AIME 数据集 (2024, 2025)。
    """
    print("\n" + "="*20 + " 准备 AIME 数据集 " + "="*20)
    output_dir = os.path.join(BASE_DATA_DIR, "AIME")
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = {
        "aime2024_test": "HuggingFaceH4/aime_2024",
        "aime2025_test": "yentinglin/aime_2025"
    }

    for filename, dataset_name in tasks.items():
        print(f"正在处理: {dataset_name} ...")
        output_path = os.path.join(output_dir, f"{filename}.jsonl")
        try:
            ds = load_dataset(dataset_name, split="train")
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in ds:
                    entry = {
                        "problem": item.get('problem'),
                        "answer": item.get('answer'), 
                        "solution": item.get('solution'),
                        "id": item.get('id', '')
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"✅ {filename} 已保存至 {output_path} ({len(ds)} 条)")
        except Exception as e:
            print(f"❌ 下载或处理 {dataset_name} 失败: {e}")

def prepare_math_datasets():
    """
    准备 MATH 数据集：
    1. math-500: 一个高质量的评测集。
    2. full_split: 完整的训练集和测试集。
    """
    print("\n" + "="*20 + " 准备 MATH 数据集 " + "="*20)
    output_dir = os.path.join(BASE_DATA_DIR, "MATH")
    os.makedirs(output_dir, exist_ok=True)

    # 1. 准备 math-500 (首选评测集)
    print("正在处理: HuggingFaceH4/math-500 ...")
    output_path_500 = os.path.join(output_dir, "math500-test.jsonl")
    try:
        ds_500 = load_dataset("HuggingFaceH4/math-500", split="test")
        with open(output_path_500, 'w', encoding='utf-8') as f:
            for item in ds_500:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"✅ math-500 已保存至 {output_path_500} ({len(ds_500)} 条)")
    except Exception as e:
        print(f"❌ 下载或处理 math-500 失败: {e}")

    # 2. 准备完整的 MATH 训练集和测试集
    print("\n正在处理: xDAN2099/lighteval-MATH (full train/test splits) ...")
    splits_to_process = {"test": "test_split.jsonl", "train": "train_split.jsonl"}
    for split_name, filename in splits_to_process.items():
        output_path = os.path.join(output_dir, filename)
        print(f"--> 正在处理 {split_name} split...")
        try:
            ds = load_dataset("xDAN2099/lighteval-MATH", split=split_name)
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in tqdm(ds, desc=f"Converting {split_name}"):
                    entry = {
                        "problem": item.get("problem"),
                        "solution": item.get("solution"),
                        "level": item.get("level"),
                        "type": item.get("type")
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"✅ {split_name} split 已保存至 {output_path} ({len(ds)} 条)")
        except Exception as e:
            print(f"❌ 下载或处理 lighteval-MATH 的 {split_name} split 失败: {e}")

def prepare_gsm8k_dataset():
    """
    准备 GSM8K 训练集和测试集，并对答案进行清洗。
    """
    print("\n" + "="*20 + " 准备 GSM8K 数据集 " + "="*20)
    name, path, config = "GSM8K", "gsm8k", "main"
    output_dir = os.path.join(BASE_DATA_DIR, path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义需要下载和处理的 split 列表
    splits_to_process = ["train", "test"]

    for split in splits_to_process:
        output_file = os.path.join(output_dir, f"{split}_clean.jsonl")
        print(f"正在处理: {name} ({path}) - {split} split ...")
        
        try:
            # 下载对应的 split
            ds = load_dataset(path, config, split=split)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in tqdm(ds, desc=f"Converting {name} ({split})"):
                    solution = item.get("answer")
                    entry = { "problem": item.get("question"), "solution": solution }
                    
                    # 提取并清洗最终答案
                    if solution and "####" in solution:
                        final_answer = solution.split("####")[-1].strip().replace(",", "")
                        entry["answer"] = final_answer
                    else:
                        entry["answer"] = str(solution).strip().replace(",", "") # 后备处理
                        
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"✅ {name} {split} 已保存至 {output_file} ({len(ds)} 条)")
            
        except Exception as e:
            print(f"❌ 处理 {name} {split} 失败: {e}")

def prepare_amc_dataset():
    """
    准备 AMC (AI-MO format) 验证集。
    """
    print("\n" + "="*20 + " 准备 AMC 数据集 " + "="*20)
    name, path, config, split = "AMC", "AI-MO/aimo-validation-amc", None, "train"
    output_dir = os.path.join(BASE_DATA_DIR, name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{name.upper()}12.jsonl")
    
    print(f"正在处理: {name} ({path}) ...")
    try:
        ds = load_dataset(path, config, split=split)
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tqdm(ds, desc=f"Converting {name}"):
                entry = {
                    "problem": item.get("problem") or item.get("question"),
                    "solution": item.get("solution") or item.get("answer"),
                    "answer": item.get("answer") or item.get("final_answer")
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"✅ {name} 已保存至 {output_file} ({len(ds)} 条)")
    except Exception as e:
        print(f"❌ 处理 {name} 失败: {e}")

def prepare_omnimath_dataset():
    """
    准备 OmniMATH 测试集。
    """
    print("\n" + "="*20 + " 准备 OmniMATH 数据集 " + "="*20)
    name, path, config, split = "OmniMATH", "KbsdJames/Omni-MATH", None, "test"
    output_dir = os.path.join(BASE_DATA_DIR, name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{name}.jsonl")
    
    print(f"正在处理: {name} ({path}) ...")
    try:
        ds = load_dataset(path, config, split=split)
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tqdm(ds, desc=f"Converting {name}"):
                entry = {
                    "problem": item.get("problem") or item.get("question"),
                    "solution": item.get("solution") or item.get("answer"),
                    "answer": item.get("answer") or item.get("final_answer")
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"✅ {name} 已保存至 {output_file} ({len(ds)} 条)")
    except Exception as e:
        print(f"❌ 处理 {name} 失败: {e}")

if __name__ == "__main__":
    print("--- 开始准备所有评测数据集 ---")
    
    # 依次调用各个数据集的独立准备函数
    prepare_aime_datasets()
    prepare_math_datasets()
    prepare_gsm8k_dataset()
    prepare_amc_dataset()
    prepare_omnimath_dataset()
    
    print("\n--- 所有数据集准备完成！---")
    print(f"所有数据均已保存在 '{BASE_DATA_DIR}' 文件夹下，并按数据集分类。")
