from cs336_alignment.hf_mirror import configure_hf_mirror, get_hf_snapshot_download_kwargs

configure_hf_mirror()

from huggingface_hub import snapshot_download

model_path = snapshot_download(
    repo_id="Qwen/Qwen3-1.7B",
    local_dir="models/Qwen3-1.7B",
    **get_hf_snapshot_download_kwargs(),
)

print(f"模型已下载至: {model_path}")
