# sample_cc_paths.py
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
    从本地的.paths.gz清单文件中进行确定性随机抽样，并生成一个可执行的wget下载脚本。
    """
    base_url = "https://data.commoncrawl.org"

    print(f"从 '{manifest_path}' 中读取所有路径")
    try:
        with gzip.open(manifest_path, 'rt', encoding='utf-8') as f:
            all_paths = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"错误: 清单文件 '{manifest_path}' 未找到！")
        return
    
    if num_samples > len(all_paths):
        print(f"警告: 请求的样本数 ({num_samples}) 大于文件中的总路径数 ({len(all_paths)})。将使用所有路径。")
        num_samples = len(all_paths)

    # 确定性随机排序
    print(f"--- 使用固定种子({seed})对所有 {len(all_paths)} 条路径进行确定性随机排序 ---")
    random.seed(seed)
    random.shuffle(all_paths) # 对整个列表进行一次性随机打乱

    # --- 核心抽样逻辑 ---
    print(f"--- 随机抽取 {num_samples} 条路径 ---")

    start_index = skip
    end_index = skip + num_samples

    if start_index >= len(all_paths):
        print(f"警告: 跳过数量({skip})已超过或等于总路径数({len(all_paths)})。没有可供抽样的路径。")
        return
        
    sampled_wet_paths = all_paths[start_index:end_index]
    print(f"--- 从排序后的列表中，跳过前 {skip} 条，选取了 {len(sampled_wet_paths)} 条路径 ---")

    with open(output_script_path, 'w', encoding='utf-8') as script_file:
        script_file.write("#!/bin/bash\n")
        script_file.write("# 自动生成的Common Crawl样本下载脚本\n")
        script_file.write(f"# 来源清单: {manifest_path}\n")
        script_file.write(f"# 抽样范围: 从第 {start_index + 1} 到 {end_index} 条\n\n")
        # 确保目标目录存在
        script_file.write(f"mkdir -p {output_dir}\n\n")
        for wet_path in sampled_wet_paths:
            full_wet_url = f"{base_url}/{wet_path}"
            # 写入 wget 命令
            script_file.write(f"# 下载样本: {Path(wet_path).name}\n")
            script_file.write(f"wget -c -P \"{output_dir}\" \"{full_wet_url}\"\n")

            if download_warc:
                # 若选择下载WARC文件，则需要从从WET路径推断出WARC路径
                warc_path_intermediate = wet_path.replace('/wet/', '/warc/')
                if warc_path_intermediate.endswith('.warc.wet.gz'):
                    warc_path = warc_path_intermediate.removesuffix('.warc.wet.gz') + '.warc.gz'
                else:
                    print(f"无法从 '{wet_path}' 推断出正确的WARC路径。")
                    continue
                full_warc_url = f"{base_url}/{warc_path}"
                script_file.write(f"wget -c -P \"{output_dir}\" \"{full_warc_url}\"\n")

            script_file.write("\n")

    os.chmod(output_script_path, 0o755)
    print("\n--- 下载脚本生成完毕！ ---")
    print(f"脚本已保存至: {output_script_path}")
    print("请运行以下命令来启动后台下载:")
    print(f"  chmod +x {output_script_path}")
    print(f"  nohup ./{output_script_path} > cc_download.log 2>&1 &")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从本地CC清单文件中抽样并生成wget下载脚本。")
    parser.add_argument("manifest_file", type=str, help="本地的.paths.gz清单文件路径。")
    parser.add_argument("-n", "--num-samples", type=int, default=20, help="要抽样的样本数量。")
    parser.add_argument("--skip", type=int, default=0, help="从随机排序的列表开头跳过指定数量的样本。(检查data/crawls/wet 中的文件数量以确定)")
    parser.add_argument("--output-script", type=str, default="scripts/download_wet_file.sh", help="输出的bash脚本文件名。")
    parser.add_argument("--output-dir", type=str, default="data/crawls/wet", help="下载文件的目标目录。")

    parser.add_argument(
        "--download_warc", 
        action="store_true", # 如果出现此标志，则值为True，否则为False
        help="同时下载与WET文件对应的WARC文件。"
    )
    args = parser.parse_args()
    sample_and_generate_script(
        manifest_path = args.manifest_file, 
        num_samples = args.num_samples, 
        output_script_path = args.output_script,
        output_dir = args.output_dir,
        skip = args.skip,
        download_warc = args.download_warc
    )