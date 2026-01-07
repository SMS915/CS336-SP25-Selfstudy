import json
import os


def clean_problem_prefixes(input_file, output_file):
    count_processed = 0
    count_modified = 0

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"正在读取: {input_file} ...")

    try:
        with open(input_file, 'r', encoding='utf-8') as fin, \
                open(output_file, 'w', encoding='utf-8') as fout:

            for line in fin:
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                    count_processed += 1

                    original_problem = item.get('problem', '')

                    # 只有当 problem 是字符串时才处理
                    if isinstance(original_problem, str):
                        # split('. ', 1) 表示只在第一个 ". " 处切分
                        parts = original_problem.split('. ', 1)

                        # 如果切分出了超过1个部分，说明找到了分隔符
                        if len(parts) > 1:
                            # parts[0] 是 ". " 之前的内容（也就是要删掉的）
                            # parts[1] 是 ". " 之后的内容（也就是要保留的）
                            item['problem'] = parts[1]
                            count_modified += 1

                    # 写入文件
                    fout.write(json.dumps(item, ensure_ascii=False) + '\n')

                except json.JSONDecodeError:
                    print(f"警告: JSON 格式错误，已跳过某行。")
                    continue

    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_file}")
        return

    print("-" * 30)
    print(f"清洗完成！")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"总处理行数: {count_processed}")
    print(f"修改(清洗)行数: {count_modified}")


if __name__ == "__main__":
    # 输入文件是上一步生成的纯数字版文件
    input_path = 'data/NuminaMath-1.5/numina_numbers_only.jsonl'
    # 输出文件是清洗后的最终版
    output_path = 'data/NuminaMath-1.5/numina_cleaned.jsonl'

    clean_problem_prefixes(input_path, output_path)