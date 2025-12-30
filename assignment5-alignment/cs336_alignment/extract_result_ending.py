import json
import argparse
import os
from tqdm import tqdm


def extract_tail(input_file, output_file, key_name="generated_text", length=30, pass_k=None):
    print(f"正在读取: {input_file}")
    print(f"提取字段: {key_name} (最后 {length} 个字符)")
    if pass_k:
        print(f"筛选条件: attempt_id <= {pass_k} (若无此字段则默认保留)")
    else:
        print("筛选条件: 提取所有行")

    count = 0
    extracted_count = 0

    try:
        with open(input_file, 'r', encoding='utf-8') as fin, \
                open(output_file, 'w', encoding='utf-8') as fout:

            lines = fin.readlines()
            for i, line in enumerate(tqdm(lines, desc="Processing")):
                line = line.strip()
                if not line: continue

                try:
                    data = json.loads(line)
                    # 获取 attempt_id (如果没有，默认为 1，确保兼容旧数据)
                    current_attempt = data.get("attempt_id")

                    # 如果指定了 pass_k，且当前 attempt_id > pass_k，则跳过
                    if pass_k is not None and current_attempt is not None:
                        if current_attempt > pass_k:
                            continue

                    # 兼容常见字段名
                    content = data.get(key_name) or data.get("generatedtext") or data.get("response") or data.get(
                        "output")

                    if content:
                        # 提取最后 N 个字符
                        tag_idx = content.rfind('</think>')
                        if tag_idx != -1:
                            tail = content[tag_idx:]
                        else:
                            tail = content[-length:]

                        # 将换行符转义，保证 txt 一行对应一条数据
                        tail_escaped = tail.replace('\n', '\\n').replace('\r', '\\r')

                        # 在前面加上 attempt_id 方便查看
                        prefix = f"[id:{current_attempt}] " if current_attempt is not None else ""

                        fout.write(f"{prefix}{tail_escaped}\n")
                        extracted_count += 1
                    else:
                        fout.write(f"<EMPTY_OR_MISSING_FIELD_LINE_{i + 1}>\n")

                    count += 1

                except json.JSONDecodeError:
                    print(f"Line {i + 1}: JSON 解析错误")
                    continue

        print("-" * 40)
        print(f"完成！结果已保存至: {output_file}")
        print(f"原始行数: {len(lines)}")
        print(f"提取行数: {extracted_count}")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取 JSONL 中生成文本的末尾字符，支持按 attempt_id 筛选")

    parser.add_argument("input_file", type=str, help="输入的 jsonl 文件路径")
    parser.add_argument("--output", type=str, default="tails_output.txt", help="输出的 txt 文件路径")
    parser.add_argument("--key", type=str, default="generated_text", help="要提取的 JSON 字段名")
    parser.add_argument("--len", type=int, default=40, help="提取末尾字符的长度")

    # 新增参数
    parser.add_argument("--pass_k", type=int, default=None, help="仅提取 attempt_id <= k 的样本。不填则提取所有。")

    args = parser.parse_args()

    extract_tail(args.input_file, args.output, args.key, args.len, args.pass_k)