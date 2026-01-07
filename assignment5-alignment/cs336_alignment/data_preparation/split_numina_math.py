import json
import os


def split_by_difficulty(input_file, output_dir):
    # 1. 定义难度映射规则
    difficulty_map = {
        # Level 1: 基础 (Elementary)
        'orca_math': 1,
        'synthetic_math': 1,

        # Level 2: 进阶 (Intermediate) - 包含默认未知来源
        'cn_k12': 2,
        'metamath': 2,

        # Level 3: 挑战 (Advanced)
        'amc_aime': 3,
        'aops_forum': 3,

        # Level 4: 专家 (Expert)
        'olympiads': 4,
        'olympiads_ref': 4,
        'cn_contest': 4,
        'inequalities': 4,
        'number_theory': 4
    }

    # 2. 准备输出文件路径
    os.makedirs(output_dir, exist_ok=True)
    file_handles = {}
    filenames = {
        1: 'numina_level_1_elementary.jsonl',
        2: 'numina_level_2_intermediate.jsonl',
        3: 'numina_level_3_advanced.jsonl',
        4: 'numina_level_4_expert.jsonl'
    }

    # 统计计数器
    stats = {1: 0, 2: 0, 3: 0, 4: 0}

    print(f"正在读取文件: {input_file} ...")
    print(f"正在拆分到目录: {output_dir}")

    try:
        # 打开所有输出文件的句柄
        for level, fname in filenames.items():
            path = os.path.join(output_dir, fname)
            file_handles[level] = open(path, 'w', encoding='utf-8')

        with open(input_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)

                    # 获取来源
                    source = item.get('source', '').strip()

                    # 判定难度，默认为 2
                    level = difficulty_map.get(source, 2)

                    # 可选：顺便把 difficulty 字段写进去，方便后续查看
                    item['difficulty'] = level

                    # 写入对应的文件句柄
                    file_handles[level].write(json.dumps(item, ensure_ascii=False) + '\n')
                    stats[level] += 1

                except json.JSONDecodeError:
                    continue

    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_file}")

    finally:
        # 关闭所有输出文件
        for f in file_handles.values():
            f.close()

    # 3. 输出统计结果
    print("-" * 30)
    print("拆分完成！统计如下：")
    total = sum(stats.values())
    for level in range(1, 5):
        count = stats[level]
        percent = (count / total * 100) if total > 0 else 0
        print(f"Level {level} ({filenames[level]}): {count} 条 ({percent:.2f}%)")


if __name__ == "__main__":
    # 输入文件：之前清洗过前缀且只保留纯数字答案的文件
    input_path = 'data/NuminaMath-1.5/numina_cleaned.jsonl'

    # 输出目录：拆分后的文件存放位置
    output_dir = 'data/NuminaMath-1.5/split_by_difficulty'

    split_by_difficulty(input_path, output_dir)