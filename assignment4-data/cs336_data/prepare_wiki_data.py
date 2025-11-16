import gzip
import random
from pathlib import Path
random.seed(42)  # 设置随机种子以确保可重复性

INPUT_URL_GZ_FILE = Path("data/wiki/enwiki-20240420-extracted_urls.txt.gz")

NUM_SAMPLES = 15000

def sample_urls_from_gz(input_path: Path, k: int) -> None:
    """
    使用水塘抽样从.gz文件中随机抽取k行URL。

    Args:
        input_path (Path): 输入的.gz文件路径。
        output_path (Path): 输出的抽样URL文件路径。
        k (int): 要抽取的样本数量。
    """
    print(f"开始从 {input_path} 中抽样 {k} 条URL...")
    OUTPUT_SAMPLED_URLS_FILE = Path(f"data/wiki/subsampled_positive_{k}_urls.txt")
    
    # 确保输出目录存在
    OUTPUT_SAMPLED_URLS_FILE.parent.mkdir(parents=True, exist_ok=True)

    reservoir = []  # 水塘
    lines_seen = 0

    # 使用gzip.open直接读取压缩文件
    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        for line in f:
            lines_seen += 1
            
            # 水塘抽样
            if len(reservoir) < k:
                reservoir.append(line.strip())
            else:
                j = random.randint(0, lines_seen - 1)
                if j < k:
                    reservoir[j] = line.strip()

            # 打印进度
            if lines_seen % 1_000_000 == 0:
                print(f"  ...已处理 {lines_seen // 1_000_000}M 行...")

    if not reservoir:
        print("错误：没有从输入文件中读取到任何URL！")
        return

    # 将抽样结果写入输出文件
    print(f"抽样完成！共处理 {lines_seen} 行。正在将 {len(reservoir)} 条URL写入 {OUTPUT_SAMPLED_URLS_FILE}...")
    with open(OUTPUT_SAMPLED_URLS_FILE, 'w', encoding='utf-8') as f:
        for url in reservoir:
            f.write(url + '\n')
            
    print("完成！")


if __name__ == "__main__":
    if not INPUT_URL_GZ_FILE.exists():
        print(f"错误：输入文件不存在于 {INPUT_URL_GZ_FILE}")
        print("请先运行下载脚本，或检查路径是否正确。")
    else:
        sample_urls_from_gz(INPUT_URL_GZ_FILE, NUM_SAMPLES)