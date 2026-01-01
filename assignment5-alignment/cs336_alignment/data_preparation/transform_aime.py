import pandas as pd
import os


INPUT_FILE = "data/AIME/AIME-2025.parquet"
OUTPUT_FILE = "data/AIME/aime2025_test.jsonl"


def convert_parquet():
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到文件 {INPUT_FILE}")
        print("请修改脚本中的 INPUT_FILE 变量为你实际下载的文件路径。")
        return

    print(f"正在读取 Parquet 文件: {INPUT_FILE} ...")

    df = pd.read_parquet(INPUT_FILE)

    print(f"数据加载成功，共 {len(df)} 条。")
    print(f"包含字段: {df.columns.tolist()}")

    print(f"正在转换并保存至: {OUTPUT_FILE} ...")
    df.to_json(OUTPUT_FILE, orient='records', lines=True, force_ascii=False)

    print("转换完成！")

    print("\n[数据预览 - 第一条]")
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        print(f.readline())


if __name__ == "__main__":
    convert_parquet()