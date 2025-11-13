import random
import argparse
import os

def parse_arguments():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="将一个 fastText 格式的数据文件分割成训练集和验证集。"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="需要被分割的原始输入文件路径。"
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.9,
        help="训练集所占的比例 (例如, 0.9 表示 90% 的数据用于训练)。"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="用于保证分割结果可复现的随机种子。"
    )
    return parser.parse_args()

def split_file(input_path: str, train_ratio: float, seed: int):
    """
    读取、打乱并分割文件。
    """
    print(f"--- 开始分割文件: {input_path} ---")
    print(f"训练集比例: {train_ratio:.0%}")
    print(f"随机种子: {seed}")

    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误: 输入文件 '{input_path}' 不存在！")
        return

    # 1. 读取所有行到内存
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"成功读取 {len(lines)} 行数据。")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    # 2. 设置随机种子并打乱
    random.seed(seed)
    random.shuffle(lines)
    print("数据已随机打乱。")

    # 3. 计算分割点
    split_index = int(len(lines) * train_ratio)

    # 4. 分割数据
    train_lines = lines[:split_index]
    valid_lines = lines[split_index:]

    print(f"分割结果: {len(train_lines)} 条训练样本, {len(valid_lines)} 条验证样本。")

    # 5. 定义输出文件名
    base_name, ext = os.path.splitext(input_path)
    train_output_path = f"{base_name}.train_split"
    valid_output_path = f"{base_name}.valid_split"

    # 6. 写入文件
    try:
        with open(train_output_path, 'w', encoding='utf-8') as f:
            f.writelines(train_lines)
        print(f"训练集已保存至: {train_output_path}")

        with open(valid_output_path, 'w', encoding='utf-8') as f:
            f.writelines(valid_lines)
        print(f"验证集已保存至: {valid_output_path}")
    except Exception as e:
        print(f"写入文件时发生错误: {e}")

    print("--- 分割完成！ ---")

if __name__ == "__main__":
    args = parse_arguments()
    split_file(args.input_file, args.split_ratio, args.seed)