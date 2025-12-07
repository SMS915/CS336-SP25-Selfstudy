# prepare_wiki_data.py (最终升级版)
import gzip
import random
import argparse
import os
from pathlib import Path

def prepare_wiki_download_script(
    url_manifest_path: str,
    num_samples: int,
    output_txt_path: str,
    output_script_path: str,
    warc_prefix: str,
    seed: int = 42
):
    """
    从维基百科URL清单中抽样，生成URL列表文件和对应的wget下载脚本。
    """
    print(f"--- 步骤1: 从 '{url_manifest_path}' 中抽样 {num_samples} 条URL ---")
    
    # 确保输出目录存在
    Path(output_txt_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        with gzip.open(url_manifest_path, 'rt', encoding='utf-8') as f:
            all_urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"错误: URL清单文件 '{url_manifest_path}' 未找到！")
        return

    # 使用确定性随机排序 + 截取
    print(f"使用固定种子({seed})对所有 {len(all_urls)} 条URL进行确定性随机排序...")
    random.seed(seed)
    random.shuffle(all_urls)
    
    sampled_urls = all_urls[:num_samples]
    
    # --- 写入抽样后的URL列表文件 ---
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.writelines(url + '\n' for url in sampled_urls)
    print(f"抽样完成！已将 {len(sampled_urls)} 条URL写入 '{output_txt_path}'。")

    # --- 步骤2: 生成 wget 下载脚本 ---
    print(f"\n--- 步骤2: 生成 wget 下载脚本到 '{output_script_path}' ---")
    with open(output_script_path, 'w', encoding='utf-8') as f:
        f.write("#!/bin/bash\n")
        f.write("# 自动生成的维基百科引用页面下载脚本\n\n")
        f.write("# 日志将输出到 wget_wiki.log 文件中\n")
        f.write("wget \\\n")
        f.write("    --timeout=15 \\\n")
        f.write("    --tries=2 \\\n")
        f.write("    --max-redirect=5 \\\n")
        f.write("    --quota=30G \\\n")
        f.write(r"    --reject-regex '\.(pdf|zip|gz|rar|exe|iso|mp3|mp4|avi|mov)$' \\")
        f.write("\n")
        f.write(f"    -i \"{output_txt_path}\" \\\n")
        f.write(f"    --warc-file=\"{warc_prefix}\" \\\n")
        f.write("    -O /dev/null\n")

    os.chmod(output_script_path, 0o755)
    
    print("\n--- 下载脚本生成完毕！ ---")
    print("请运行以下命令来启动后台下载:")
    print(f"  nohup ./{output_script_path} > wget_wiki.log 2>&1 &")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从维基百科URL清单中抽样并生成wget下载脚本。")
    parser.add_argument(
        "--manifest-file",
        type=str,
        default="data/cc_path/enwiki-20240420-extracted_urls.txt.gz",
        help="输入的URL清单.gz文件路径。"
    )
    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=15000,
        help="要抽样的URL数量。"
    )
    parser.add_argument(
        "--output-txt",
        type=str,
        default="data/wiki/sampled_wiki_urls.txt",
        help="输出的抽样URL列表文件名。"
    )
    parser.add_argument(
        "--output-script",
        type=str,
        default="download_wiki_pages.sh",
        help="输出的bash脚本文件名。"
    )
    parser.add_argument(
        "--warc-prefix",
        type=str,
        default="data/wiki/wiki_cited_pages",
        help="wget命令中--warc-file参数的前缀。"
    )
    args = parser.parse_args()
    
    prepare_wiki_download_script(
        args.manifest_file,
        args.num_samples,
        args.output_txt,
        args.output_script,
        args.warc_prefix
    )