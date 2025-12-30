import os
from datasets import load_dataset
from huggingface_hub import login, snapshot_download

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

target_dir = "data/Bespoke-Stratos-17k"
os.makedirs(target_dir, exist_ok=True)
# 下载数据集到指定文件夹
snapshot_download(
   repo_id="bespokelabs/Bespoke-Stratos-17k", # 数据集名称
   repo_type="dataset", # 类型为数据集
   local_dir=target_dir, # 指定保存路径
   local_dir_use_symlinks=False, # 禁用符号链接
   resume_download=True # 支持断点续传
)

# "bespokelabs/Bespoke-Stratos-17k"