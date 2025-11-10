import os
import requests
from tqdm import tqdm

LANGUAGE_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
NSFW_MODEL_URL = "https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_nsfw_final.bin"
HATESPEECH_MODEL_URL = "https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_hatespeech_final.bin"


url_dict = {"语言检测模型": LANGUAGE_MODEL_URL,
            "NSFW 分类模型": NSFW_MODEL_URL,
            "有害言论分类模型": HATESPEECH_MODEL_URL}


def download_file(url: str, desc: str, dest_dir: str = "data/classifiers/") -> str:
    """
    从给定的 URL 下载文件，并显示进度条。
    如果文件已存在，则跳过下载。

    Args:
        url (str): 文件的下载链接。
        dest_dir (str): 文件下载后存放的目标目录。
        desc (str): tqdm 进度条的描述文字，默认为文件名。

    Returns:
        str: 下载完成的文件的完整路径。如果下载失败则返回空字符串。
    """
    # 确保目标目录存在
    os.makedirs(dest_dir, exist_ok=True)
    
    # 获取文件名并构建完整的本地路径
    file_name = os.path.basename(url)
    file_path = os.path.join(dest_dir, file_name)

    # 检查文件是否已经存在
    if os.path.exists(file_path):
        print(f"文件已存在于: {file_path}")
        return file_path

    # 如果文件不存在，则开始下载
    if desc is None:
        desc = file_name
    print(f"文件不存在。开始从 {url} 下载...")
    
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
                os.remove(file_path) # 删除不完整的文件
                return ""
    
    except requests.exceptions.RequestException as e:
        print(f"下载 '{file_name}' 失败: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return ""

    print(f"文件成功下载并保存至: {file_path}")
    return file_path

if __name__ == "__main__":
    for desc, url in url_dict.items():
        download_file(url, desc)