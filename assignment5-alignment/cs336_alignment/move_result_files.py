import os
import shutil
import argparse

def organize_files(input_dir, configs):
    # 1. 确保基础逻辑：非匹配的 jsonl 文件夹
    unnormalized_dir = os.path.join(input_dir, "unnormalized")
    
    # 获取目录下所有文件
    all_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    
    count_moved = 0
    count_unnormalized = 0

    for filename in all_files:
        # 仅处理 .jsonl 文件，其他类型原地不动
        if not filename.endswith(".jsonl"):
            continue

        target_folder = None
        
        # 2. 按照提供的逻辑进行匹配
        for model_prefix, dataset_key, _ in configs:
            if filename.startswith(model_prefix) and dataset_key in filename:
                target_folder = model_prefix
                break
        
        # 3. 确定最终目的地
        if target_folder:
            dest_path = os.path.join(input_dir, target_folder)
            count_moved += 1
        else:
            dest_path = unnormalized_dir
            count_unnormalized += 1

        # 4. 执行移动操作
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
            
        src_file = os.path.join(input_dir, filename)
        dst_file = os.path.join(dest_path, filename)
        
        print(f"正在移动: {filename} -> {dest_path}")
        shutil.move(src_file, dst_file)

    print("\n" + "="*30)
    print(f"整理完成！")
    print(f"匹配并移动: {count_moved} 个文件")
    print(f"未匹配移至 unnormalized: {count_unnormalized} 个文件")
    print("="*30)

if __name__ == "__main__":
    # 这里保留你原始代码中的配置定义
    configs = [
        # --- GSM8K ---
        ("baseline", "gsm8k_pass1", 1),
        ("sft", "gsm8k_pass1", 1),
        ("grpo", "gsm8k_pass1", 1),
        ("grpo_no_std_norm", "gsm8k_pass1", 1),
        ("drgrpo", "gsm8k_pass1", 1),
        ("instruct", "gsm8k_pass1", 1),

        # --- MATH-500 ---
        ("baseline", "math500_pass_64", 64),
        ("sft", "math500_pass_64", 64),
        ("grpo", "math500_pass_64", 64),
        ("grpo_no_std_norm", "math500_pass_64", 64),
        ("drgrpo", "math500_pass_64", 64),
        ("instruct", "math500_pass_64", 64),

        # --- MathTest ---
        ("baseline", "MathTest_pass_8", 8),
        ("sft", "MathTest_pass_8", 8),
        ("grpo", "MathTest_pass_8", 8),
        ("grpo_no_std_norm", "MathTest_pass_8", 8),
        ("drgrpo", "MathTest_pass_8", 8),
        ("instruct", "MathTest_pass_8", 64),

        # --- AMC ---
        ("baseline", "amc_pass_64", 64),
        ("sft", "amc_pass_64", 64),
        ("grpo", "amc_pass_64", 64),
        ("grpo_no_std_norm", "amc_pass_64", 64),
        ("drgrpo", "amc_pass_64", 64),
        ("instruct", "amc_pass_64", 64),

        # --- AIME 2024 ---
        ("baseline", "aime24_pass_64", 64),
        ("sft", "aime24_pass_64", 64),
        ("grpo", "aime24_pass_64", 64),
        ("grpo_no_std_norm", "aime24_pass_64", 64),
        ("drgrpo", "aime24_pass_64", 64),
        ("instruct", "aime24_pass_64", 64),

        # --- AIME 2025 ---
        ("baseline", "aime25_pass_64", 64),
        ("sft", "aime25_pass_64", 64),
        ("grpo", "aime25_pass_64", 64),
        ("grpo_no_std_norm", "aime25_pass_64", 64),
        ("drgrpo", "aime25_pass_64", 64),
        ("instruct", "aime25_pass_64", 64),
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./results", help="需要整理的文件夹路径")
    args = parser.parse_args()

    if os.path.exists(args.input_dir):
        organize_files(args.input_dir, configs)
    else:
        print(f"错误: 找不到目录 {args.input_dir}")