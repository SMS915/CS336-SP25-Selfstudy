# 建议将此文件保存为 cs336_data/utils.py

import os
import requests
from tqdm import tqdm
from pathlib import Path  # 导入 pathlib

# --- 动态计算项目路径 ---
# 获取当前脚本文件 (utils.py) 所在的目录
SCRIPT_DIR = Path(__file__).parent.resolve()

# 从脚本目录往上一级，就得到了项目根目录
PROJECT_ROOT = SCRIPT_DIR.parent

# 构造分类器和数据爬取文件的目标目录
CLASSIFIERS_DIR = PROJECT_ROOT / "data" / "classifiers"
CRAWLS_DIR = PROJECT_ROOT / "data" / "crawls"
WIKI_DIR = PROJECT_ROOT / "data" / "wiki"


# --- 定义模型URL ---
LANGUAGE_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
NSFW_MODEL_URL = "https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_nsfw_final.bin"
HATESPEECH_MODEL_URL = "https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_hatespeech_final.bin"
WIKI_URL_LINK = "https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment4/enwiki-20240420-extracted_urls.txt.gz"

# 将所有需要下载的模型统一管理
MODELS_TO_DOWNLOAD = {
    "语言检测模型": (LANGUAGE_MODEL_URL, CLASSIFIERS_DIR),
    "NSFW 分类模型": (NSFW_MODEL_URL, CLASSIFIERS_DIR),
    "有害言论分类模型": (HATESPEECH_MODEL_URL, CLASSIFIERS_DIR),
    "维基百科 URL 列表": (WIKI_URL_LINK, WIKI_DIR)
}

# --- 更新下载函数 (几乎不变，只修改了默认路径) ---
def download_file(url: str, desc: str, dest_dir: Path = CLASSIFIERS_DIR) -> Path | None:
    """
    从给定的 URL 下载文件，并显示进度条。
    如果文件已存在，则跳过下载。
    """
    # 确保目标目录存在
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    file_name = os.path.basename(url)
    file_path = dest_dir / file_name

    if file_path.exists():
        print(f"文件已存在于: {file_path}")
        return file_path

    print(f"文件不存在。开始从 {url} 下载 [{desc}]...")
    
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=desc) as progress_bar:
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        progress_bar.update(len(chunk))
                        f.write(chunk)
            
            if total_size != 0 and progress_bar.n != total_size:
                print(f"错误：下载的文件 '{file_name}' 大小与预期不符。")
                os.remove(file_path)
                return None
    except requests.exceptions.RequestException as e:
        print(f"下载 '{file_name}' 失败: {e}")
        if file_path.exists():
            os.remove(file_path)
        return None

    print(f"文件成功下载并保存至: {file_path}")
    return file_path

if __name__ == "__main__":
    print(f"项目根目录被识别为: {PROJECT_ROOT}")
    print(f"分类器将下载至: {CLASSIFIERS_DIR}")
    print("-" * 20)

    for desc, file in MODELS_TO_DOWNLOAD.items():
        download_file(file[0], desc, dest_dir=file[1])
        print("-" * 10)
    
    print("所有模型检查/下载完毕。")