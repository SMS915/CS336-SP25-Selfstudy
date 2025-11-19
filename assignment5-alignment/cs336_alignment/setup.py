import os
from huggingface_hub import snapshot_download
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

model_path = snapshot_download(
    repo_id="Qwen/Qwen2.5-Math-1.5B",
    local_dir="models/Qwen2.5-Math-1.5B",
    local_dir_use_symlinks=False  # 确保下载的是实体文件
)

print(f"模型已下载至: {model_path}")