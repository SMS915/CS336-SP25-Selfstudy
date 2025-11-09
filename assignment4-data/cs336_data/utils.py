import os
import requests
from tqdm import tqdm

def download_fasttext_model(dest_dir: str = "classifiers") -> str:
    """
    下载fastText语言识别模型 lid.176.bin。

    Args:
        dest_dir (str): 模型下载后存放的目标目录。

    Returns:
        str: 下载完成的模型文件的完整路径。
    """
    # 模型的直接下载链接
    MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    
    # 确保目标目录存在
    os.makedirs(dest_dir, exist_ok=True)
    
    # 获取文件名并构建完整的本地路径
    file_name = os.path.basename(MODEL_URL)
    model_path = os.path.join(dest_dir, file_name)

    # 1. 检查文件是否已经存在
    if os.path.exists(model_path):
        print(f"模型文件已存在于: {model_path}")
        return model_path

    # 2. 如果文件不存在，则开始下载
    print(f"模型文件不存在。开始从 {MODEL_URL} 下载...")
    try:
        # 使用stream=True来处理大文件，避免一次性加载到内存
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()  # 如果请求失败 (如404)，会抛出异常

            # 获取文件总大小，用于tqdm进度条
            total_size = int(r.headers.get('content-length', 0))

            # 设置tqdm进度条
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=file_name)

            # 以二进制写模式打开文件，开始写入
            with open(model_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    # chunk_size可以根据网络情况调整
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            
            progress_bar.close()

            if total_size != 0 and progress_bar.n != total_size:
                print("错误：下载的文件大小与预期不符。")
                # 这里可以添加删除不完整文件的逻辑
                # os.remove(model_path)
                return ""
    
    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")
        # 如果下载失败，确保不留下损坏的文件
        if os.path.exists(model_path):
            os.remove(model_path)
        return ""


    print(f"模型成功下载并保存至: {model_path}")
    return model_path

# 你可以这样直接运行这个文件来测试下载功能
if __name__ == "__main__":
    download_fasttext_model()