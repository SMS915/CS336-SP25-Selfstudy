import random
import os
from tqdm import tqdm
from typing import Optional, List

from cs336_data.build_classifier_dataset import judge_high_quality, extract_text
from fastwarc.warc import ArchiveIterator, WarcRecordType

random.seed(42)

# 从已有的训练文件中加载特定标签的样本
def load_samples_from_file(file_path: str, target_label: str) -> List[str]:
    """
    从一个fastText格式的文件中，只加载指定标签的样本行。

    Args:
        file_path (str): 原始训练文件的路径。
        target_label (str): 想要加载的标签 (例如, '__label__high_quality')。

    Returns:
        List[str]: 包含所有匹配样本的列表，每行是一个完整的fastText格式字符串。
    """
    print(f"从 {file_path} 加载标签为 '{target_label}' 的样本...")
    
    samples = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 检查行是否以目标标签开头
                if line.strip().startswith(target_label):
                    samples.append(line.strip())
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到！")
        return []
        
    print(f"成功加载 {len(samples)} 条样本。")
    return samples


def build_filtered_negative_data(warc_path: str, max_samples: int) -> List[str]:
    """
    从Common Crawl中筛选出通过了高质量检查的页面，作为负样本。
    """
    print(f"开始从 {warc_path} 筛选高质量的【负样本】...")
    
    negative_texts = []
    
    with open(warc_path, 'rb') as warc_file:
        iterator = ArchiveIterator(warc_file)
        progress = tqdm(iterator, desc="筛选高质量负样本", unit=" rec")
        
        for record in progress:
            if record.record_type != WarcRecordType.response or record.http_content_type != 'text/html':
                continue

            html_bytes = record.reader.read()
            text = extract_text(html_bytes)
            
            if text and not text.isspace():
                if judge_high_quality(text):
                    clean_text = text.replace('\n', ' ').replace('\r', ' ').strip()
                    formatted_text = f'__label__low_quality {clean_text}'
                    negative_texts.append(formatted_text)
                    progress.set_postfix(found=len(negative_texts), refresh=True)
            
            if len(negative_texts) >= max_samples:
                break
                
    print(f"从 {warc_path} 中提取到 {len(negative_texts)} 条高质量负样本。")
    return negative_texts


def remix_classifier_dataset(
    original_dataset_path: str,
    output_dir: str,
    output_base_name: str,
    output_suffix: str
):
    positive_samples = load_samples_from_file(original_dataset_path, '__label__high_quality')
    
    num_positive = len(positive_samples)
    if num_positive == 0:
        print("错误：未能从原始数据集中加载任何正样本。")
        return

    cc_warc_path = 'data/crawls/CC-MAIN-20250417135010-20250417165010-00065.warc.gz'
    negative_samples = build_filtered_negative_data(cc_warc_path, max_samples=num_positive)

    print("\n合并新的正负样本...")
    all_samples = positive_samples + negative_samples
    random.shuffle(all_samples)
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_count = len(all_samples)
    output_file_path = os.path.join(output_dir, f"{output_base_name}_{total_count}.{output_suffix}")
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.writelines(line + '\n' for line in all_samples)
    
    print(f"\n“重混”数据集已保存到 {output_file_path}，总样本数: {total_count}")
    print(f"  - 正样本 (复用): {len(positive_samples)}")
    print(f"  - 负样本 (新生成): {len(negative_samples)}")

if __name__ == "__main__":
    remix_classifier_dataset(
        original_dataset_path='data/classifiers_dataset/quality_classifier_large_2114.original',
        output_dir='data/classifiers_dataset',
        output_base_name='quality_classifier_remixed',
        output_suffix='original'
    )