import json
import re
import os
from tqdm import tqdm  # 如果没安装 tqdm，可以注释掉这行以及下面的 tqdm 包装


def filter_pure_numbers(input_file, output_file):
    # 正则表达式定义：
    # ^-?       : 可选的负号开头
    # \d+       : 至少一个数字
    # (\.\d+)?  : 可选的小数部分 (.后跟数字)
    # $         : 字符串结束
    # 这样可以匹配 "123", "-5", "12.34", "0.5" 等，但不匹配 "1/2", "1,000", "5 apples"
    number_pattern = re.compile(r'^-?\d+(\.\d+)?$')

    count_total = 0
    count_kept = 0

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"正在处理: {input_file} ...")

    try:
        with open(input_file, 'r', encoding='utf-8') as fin, \
                open(output_file, 'w', encoding='utf-8') as fout:

            # 如果安装了 tqdm，这行会显示进度条；如果报错，请去掉 tqdm() 包裹
            # for line in tqdm(fin):
            for line in fin:
                count_total += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)

                    # 获取 answer 字段并转为字符串去空格
                    answer = str(item.get('answer', '')).strip()

                    # 使用正则判断是否为纯数字
                    if number_pattern.match(answer):
                        # 写入原样数据
                        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                        count_kept += 1

                except json.JSONDecodeError:
                    print(f"警告: 第 {count_total} 行 JSON 格式错误，已跳过。")
                    continue

    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        return

    print("-" * 30)
    print(f"处理完成！")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"总行数: {count_total}")
    print(f"筛选出纯数字行数: {count_kept}")


if __name__ == "__main__":
    # 配置路径
    input_path = 'data/NuminaMath-1.5/numina_filtered.jsonl'
    output_path = 'data/NuminaMath-1.5/numina_numbers_only.jsonl'

    filter_pure_numbers(input_path, output_path)