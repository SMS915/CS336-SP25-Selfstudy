from datasets import load_dataset
import os

def download_dataset(repo_id="AI-MO/NuminaMath-CoT", save_path="./data/NuminaMath-CoT/raw"):
    print(f"开始从 {repo_id} 下载数据集...")
    # 859k条数据，约 1.23GB
    dataset = load_dataset(repo_id, split="train")

    # 导出为本地格式，方便后续脚本直接读取
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dataset.to_json(os.path.join(save_path, "numina_cot_raw.jsonl"), force_ascii=False)
    print(f"下载完成！原始数据已保存至: {save_path}/numina_cot_raw.jsonl")


if __name__ == "__main__":
    download_dataset()