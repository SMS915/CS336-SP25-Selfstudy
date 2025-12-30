import os
from huggingface_hub import snapshot_download

# 设置环境变量，切换到国内镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

snapshot_download(
    repo_id="Qwen/Qwen2.5-Math-1.5B-Instruct",
    local_dir="models/Qwen2.5-Math-1.5B-Instruct"
)