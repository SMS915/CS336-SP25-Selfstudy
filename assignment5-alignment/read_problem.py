import json

# 请将 'your_file.jsonl' 替换为你的实际 JSONL 文件路径
file_paths = ['data/AIME/aime2024_test.jsonl', 'data/AIME/aime2025_test.jsonl']
for path in file_paths:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            print(path)
            for i, line in enumerate(f, start=1):
                if line.strip():  # 跳过空行
                    data = json.loads(line.strip())
                    problem = data.get('problem', '')  # 获取 problem 字段，如果不存在则为空
                    print(f"{i}. {problem}")
            print("=" * 20)
    except FileNotFoundError:
        print("错误：文件未找到，请检查文件路径是否正确。")
    except json.JSONDecodeError as e:
        print(f"错误：JSON 解析失败（第 {i} 行）：{e}")
    except Exception as e:
        print(f"发生未知错误：{e}")