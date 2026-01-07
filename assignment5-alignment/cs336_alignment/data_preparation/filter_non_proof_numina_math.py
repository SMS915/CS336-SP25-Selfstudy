import json
import os
from tqdm import tqdm  # 用于显示进度条，如果没有安装建议 pip install tqdm


def filter_jsonl(input_path, output_path):
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误: 找不到文件 {input_path}")
        return

    total_count = 0
    kept_count = 0

    # 获取文件总行数用于进度条（可选步骤，如果文件极大可以跳过这步以节省时间）
    print("正在计算文件行数...")
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    print(f"开始筛选，总行数: {total_lines}")

    with open(input_path, 'r', encoding='utf-8') as f_in, \
            open(output_path, 'w', encoding='utf-8') as f_out:

        # 使用 tqdm 显示进度
        for line in tqdm(f_in, total=total_lines, desc="Processing"):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                total_count += 1

                # 核心筛选逻辑：获取 answer 字段，判断是否不等于 "proof"
                # 使用 .get() 防止某行数据缺失 answer 字段报错
                if data.get("answer") != "proof":
                    # 将保留的数据写入新文件，ensure_ascii=False 保证中文正常显示
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    kept_count += 1

            except json.JSONDecodeError:
                print(f"警告: 跳过无法解析的行: {line[:50]}...")
                continue

    print("-" * 30)
    print(f"筛选完成！")
    print(f"原始数据条数: {total_count}")
    print(f"保留数据条数: {kept_count}")
    print(f"剔除 'proof' 条数: {total_count - kept_count}")
    print(f"输出文件路径: {output_path}")


if __name__ == "__main__":
    # 定义文件路径
    input_file = "data/NuminaMath-1.5/raw/numina_raw.jsonl"
    output_file = "data/NuminaMath-1.5/numina_filtered.jsonl"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    filter_jsonl(input_file, output_file)