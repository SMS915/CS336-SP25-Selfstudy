import gzip
import random
import argparse
import os
from pathlib import Path


def sample_and_generate_script(
        manifest_path: str,
        num_samples: int,
        output_dir: str,
        output_script_path: str,
        skip: int = 0,
        seed: int = 42,
        download_warc: bool = False,
):
    """
    生成一个包含 URL 列表的 .txt 文件，并创建一个轻量级的下载脚本。
    """
    base_url = "https://data.commoncrawl.org"

    # 构造 txt 文件路径：将脚本名的 .sh 换成 .txt
    url_list_path = output_script_path.replace(".sh", ".txt")

    print(f"从 '{manifest_path}' 中读取所有路径")
    try:
        with gzip.open(manifest_path, 'rt', encoding='utf-8') as f:
            all_paths = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"错误: 清单文件 '{manifest_path}' 未找到！")
        return

    if num_samples > len(all_paths):
        num_samples = len(all_paths)

    random.seed(seed)
    random.shuffle(all_paths)

    start_index = skip
    end_index = skip + num_samples

    if start_index >= len(all_paths):
        print(f"警告: 跳过数量({skip})过大。")
        return

    sampled_wet_paths = all_paths[start_index:end_index]

    # --- 1. 生成纯文本 URL 列表 (.txt) ---
    print(f"--- 正在生成 URL 列表文件: {url_list_path} ---")
    with open(url_list_path, 'w', encoding='utf-8') as txt_file:
        for wet_path in sampled_wet_paths:
            txt_file.write(f"{base_url}/{wet_path}\n")
            if download_warc:
                warc_path_intermediate = wet_path.replace('/wet/', '/warc/')
                if warc_path_intermediate.endswith('.warc.wet.gz'):
                    warc_path = warc_path_intermediate.removesuffix('.warc.wet.gz') + '.warc.gz'
                    txt_file.write(f"{base_url}/{warc_path}\n")

    # --- 2. 生成极简下载脚本 (.sh) ---
    print(f"--- 正在生成轻量级下载脚本: {output_script_path} ---")
    with open(output_script_path, 'w', encoding='utf-8') as script_file:
        script_file.write("#!/bin/bash\n")
        script_file.write(f"# 自动生成的轻量级下载脚本 (仅调用文本清单)\n")
        script_file.write(f"mkdir -p {output_dir}\n\n")
        # 使用 wget 的 -i 参数直接读取文件中的 URL
        # -c 并行断点续传，-P 指定目录，-i 指定输入列表文件
        script_file.write(f"wget -c -P \"{output_dir}\" -i \"{url_list_path}\"\n")

    os.chmod(output_script_path, 0o755)
    print(f"\n成功！脚本现在仅包含命令，链接已存入 {url_list_path}。")


if __name__ == "__main__":
    # ... (参数解析部分保持不变) ...
    parser = argparse.ArgumentParser(description="从本地CC清单文件中抽样并生成wget下载脚本。")
    parser.add_argument("manifest_file", type=str, help="本地的.paths.gz清单文件路径。")
    parser.add_argument("-n", "--num-samples", type=int, default=20, help="要抽样的样本数量。")
    parser.add_argument("--skip", type=int, default=0, help="从随机排序的列表开头跳过指定数量的样本。")
    parser.add_argument("--output-script", type=str, default="scripts/download_wet_file.sh",
                        help="输出的bash脚本文件名。")
    parser.add_argument("--output-dir", type=str, default="data/crawls/wet", help="下载文件的目标目录。")
    parser.add_argument("--download_warc", action="store_true", help="同时下载对应的WARC文件。")

    args = parser.parse_args()
    sample_and_generate_script(
        manifest_path=args.manifest_file,
        num_samples=args.num_samples,
        output_script_path=args.output_script,
        output_dir=args.output_dir,
        skip=args.skip,
        download_warc=args.download_warc
    )