import random
import os
from tqdm import tqdm
from typing import Optional, Dict, Any
from fastwarc.warc import ArchiveIterator, WarcRecordType
from cs336_data.extraction import extract_text
from cs336_data.content_filter import identify_language, classify_nsfw, classify_toxic_speech
from cs336_data.quality_filter import gopher_quality_filter

random.seed(42)

def judge_high_quality(text: str, lang_threshold: float = 0.9, nsfw_threshold: float = 0.02, toxic_threshold: float = 0.02) -> bool:
    """
    Judge if the given text is of high quality based on classification functions.

    Args:
        text (str): The text content to evaluate.
    Returns:
        bool: True if the text is high quality, False otherwise.
    """
    
    pass_gopher = gopher_quality_filter(text)
    if not pass_gopher:
        return False
    
    lang_code, lang_score = identify_language(text)
    if lang_code != 'en' or lang_score <= lang_threshold:
        return False
    
    nsfw_label, nsfw_score = classify_nsfw(text)
    transformed_nsfw_score = nsfw_score if nsfw_label == 'nsfw' else (1 - nsfw_score)
    if transformed_nsfw_score > nsfw_threshold:
        return False
    
    toxic_label, toxic_score = classify_toxic_speech(text)
    transformed_toxic_score = toxic_score if toxic_label == 'toxic' else (1 - toxic_score)
    if transformed_toxic_score > toxic_threshold:
        return False

    return True



def build_positive_data(warc_file_path: str, quality_label: str, fasttext_format_str: str, max_sample: Optional[int] = None) -> tuple[list[str], int]:
    """
    Build a dataset from a WARC file by extracting text content.

    Args:
        warc_file_path (str): The path to the WARC file.
    Returns:
        tuple: A tuple containing a list of extracted text contents and the count of valid records.
    """
    texts = []
    valid_count = 0
    
    print(f'正在从{warc_file_path}构建{quality_label}数据集...')
    
    with open(warc_file_path, 'rb') as warc_file:
        iterator = ArchiveIterator(warc_file)
        # 将tqdm包裹在迭代器上，让它自动处理输入进度
        progress_bar = tqdm(iterator, desc=f'正在扫描 {os.path.basename(warc_file_path)}', unit=' records')
        
        # 【修正】只有一个循环！
        for record in progress_bar:
            if record.record_type != WarcRecordType.response or record.http_content_type != 'text/html':
                continue

            html_bytes = record.reader.read()
            if not html_bytes:
                continue
                
            text = extract_text(html_bytes)
            
            if text and not text.isspace():
                # 【性能优化】在这里进行判断，而不是在函数内部重复计算
                if judge_high_quality(text):
                    clean_text = text.replace('\n', ' ').replace('\r', ' ').strip()
                    formatted_text = f'{fasttext_format_str} {clean_text}'
                    texts.append(formatted_text)
                    valid_count += 1
            
            # 使用postfix实时更新找到了多少个样本
            progress_bar.set_postfix(found=valid_count, refresh=True)
            
            if max_sample is not None and valid_count >= max_sample:
                break
                
    print(f"从 {warc_file_path} 中提取到 {valid_count} 条初筛后的{quality_label} 样本。")
    return texts, valid_count


def build_negative_data(warc_file_path: str, quality_label: str, fasttext_format_str: str, max_sample: int) -> tuple[list[str], int]:
    """
    从WARC文件中构建【真正随机】的负样本。
    不对内容进行高质量筛选，只做最基本的有效性检查（非空）。
    """
    negative_texts = []
    
    with open(warc_file_path, 'rb') as warc_file:
        iterator = ArchiveIterator(warc_file)
        # 不知道总共有多少有效记录，所以只显示迭代进度
        progress_bar = tqdm(iterator, desc=f'正在扫描 {warc_file_path} 以获取随机负样本', unit=' records')
        
        # 使用水塘抽样来保证公平性
        reservoir = []
        items_seen = 0

        for record in progress_bar:
            if record.record_type != WarcRecordType.response or record.http_content_type != 'text/html':
                continue

            html_bytes = record.reader.read()
            text = extract_text(html_bytes)

            # 只进行最基本的检查
            if text and not text.isspace():
                items_seen += 1
                # 水塘抽样逻辑
                if len(reservoir) < max_sample:
                    reservoir.append(text)
                else:
                    j = random.randint(0, items_seen - 1)
                    if j < max_sample:
                        reservoir[j] = text

    for txt in reservoir:
        clean_text = txt.replace('\n', ' ').replace('\r', ' ').strip()
        negative_texts.append(f"{fasttext_format_str} {clean_text}")

    negative_count = len(negative_texts)
        
    print(f"从 {warc_file_path} 中随机抽样出 {negative_count} 条{quality_label}。")
    return negative_texts, negative_count

def build_classifier_dataset(
    output_dir: str,
    output_base_name: str,
    output_suffix: str,
    max_sample: Optional[int] = None
):
    os.makedirs(output_dir, exist_ok=True)

    wiki_warc_path = 'data/wiki/subsampled_positive_15000_pages.warc.gz'
    positive_samples, positive_count =  build_positive_data(wiki_warc_path, \
                          '高质量样本', '__label__high_quality', max_sample)
    print(f"找到的正样本数量: {positive_count}")


    common_crawl_warc_path = 'data/crawls/CC-MAIN-20250417135010-20250417165010-00065.warc.gz'
    negative_samples, negative_count = build_negative_data(common_crawl_warc_path,\
                    '低质量样本', '__label__low_quality', max_sample = positive_count)
    print(f"找到的负样本数量: {negative_count}")

    all_samples = positive_samples + negative_samples
    random.shuffle(all_samples)
    total_count = len(all_samples)
    output_file_path = os.path.join(output_dir, f"{output_base_name}_{total_count}.{output_suffix}")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(sample + '\n')
    
    print(f"已将数据集保存到 {output_file_path}，总样本数: {total_count}")

if __name__ == "__main__":
    build_classifier_dataset(output_dir = 'data/classifiers_dataset/',output_base_name = 'quality_classifier_large', output_suffix = 'train')
