import json
import os
from datasets import load_dataset
from tqdm import tqdm
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
output_file = "data/c4_valid_full.jsonl"
num_samples = 5000  # 取 5000 条做初步 PPL 验证
os.makedirs(os.path.dirname(output_file), exist_ok=True)

print("正在从 hf-mirror.com 流式加载 C4 验证集...")

try:
    # 加载官方 validation split (此时会自动走镜像下载元数据)
    # streaming=True 确保不下载整个数据集，只流式读取
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)

    print(f"开始下载并保存到 {output_file} ...")

    # 这里的 total 是硬编码的参考值，用于显示进度条
    total_examples = 364608

    with open(output_file, 'w', encoding='utf-8') as f:
        # [修改点] 去掉了 .take()，直接遍历整个 dataset
        for i, example in tqdm(enumerate(ds), total=total_examples, unit="doc"):
            json.dump(example, f, ensure_ascii=False)
            f.write('\n')

    print(f"✅ 完成！完整验证集已保存至: {output_file}")
    print(f"总计文档数: {i+1}")

    print(f"完成！验证集已保存至: {output_file}")

except Exception as e:
    print(f"下载出错: {e}")
    print("提示: 请确保你已经安装了 datasets 库 (pip install datasets)")
    print("如果是连接超时，请检查服务器是否能访问 hf-mirror.com")