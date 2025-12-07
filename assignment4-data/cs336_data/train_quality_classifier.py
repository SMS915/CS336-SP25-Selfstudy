import random
import os
import argparse
from tqdm import tqdm
from typing import List, Literal, Optional, Tuple
from fastwarc.warc import ArchiveIterator, WarcRecordType
from .extraction import extract_text
from .filter import judge_high_quality

random.seed(42)


def build_filtered_data(warc_file_path: str, quality_label: str, source_label: str, fasttext_format_str: str, max_sample: Optional[int] = None) -> tuple[list[str], int]:
    """
    从WARC文件中筛选并通过所有高质量检查的文本内容，构建一个数据集。
    """
    texts = []
    
    # 检查文件是否存在
    if not os.path.exists(warc_file_path):
        raise FileNotFoundError(f"未找到文件: {warc_file_path}")

    with open(warc_file_path, 'rb') as warc_file:
        iterator = ArchiveIterator(warc_file)
        progress_bar = tqdm(iterator, desc=f'正在扫描 {os.path.basename(warc_file_path)}', unit=' records')
 
        for record in progress_bar:
            if record.record_type != WarcRecordType.response or record.http_content_type != 'text/html':
                continue

            html_bytes = record.reader.read()
            if not html_bytes:
                continue
                
            text = extract_text(html_bytes)
            
            if text and not text.isspace():
                if judge_high_quality(text):
                    clean_text = text.replace('\n', ' ').replace('\r', ' ').strip()
                    formatted_text = f'{fasttext_format_str} {clean_text}'
                    texts.append(formatted_text)
                    progress_bar.set_postfix(found=len(texts), refresh=True)
            
            if max_sample is not None and len(texts) >= max_sample:
                break
        
        positive_count = len(texts)
                
    print(f"从 {warc_file_path} 中提取到 {positive_count} 条{quality_label}{source_label} 样本。")
    return texts, positive_count


def build_unfiltered_data(warc_file_path: str, quality_label: str, source_label: str, fasttext_format_str: str, max_sample: int) -> tuple[list[str], int]:
    """
    从WARC文件中进行水塘抽样，构建一个无筛选的数据集。
    """
    negative_texts = []
    
    if not os.path.exists(warc_file_path):
        raise FileNotFoundError(f"未找到文件: {warc_file_path}")
        
    with open(warc_file_path, 'rb') as warc_file:
        iterator = ArchiveIterator(warc_file)
        progress_bar = tqdm(iterator, desc=f'正在扫描 {os.path.basename(warc_file_path)} 以获取随机负样本', unit=' records')

        reservoir = []
        items_seen = 0

        for record in progress_bar:
            if record.record_type != WarcRecordType.response or record.http_content_type != 'text/html':
                continue

            html_bytes = record.reader.read()
            text = extract_text(html_bytes)

            if text and not text.isspace():
                items_seen += 1
                # 水塘抽样
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
        
    print(f"从 {warc_file_path} 中随机抽样出 {negative_count} 条{quality_label}{source_label} 样本。")
    return negative_texts, negative_count


def build_filtered_wiki_data(wiki_file_path: str,  max_sample: Optional[int]) -> Tuple[List[str],int]:
    """"
    从维基百科(Wiki)的WARC文件中构建高质量正样本。
    """
    print(f"开始从 {wiki_file_path} 筛选高质量wiki样本...")
    texts, count = build_filtered_data(wiki_file_path, quality_label='高质量',source_label='wiki',fasttext_format_str='__label__wiki', max_sample = max_sample)
    return texts, count


def build_filtered_cc_data(cc_file_path: str, max_sample: Optional[int]) -> Tuple[List[str], int]:
    """
    用于从通用爬取(Common Crawl)的WARC文件中构建高质量负样本。
    """
    print(f"开始从 {cc_file_path} 筛选高质量cc样本...")
    texts, count = build_filtered_data(cc_file_path, quality_label='高质量',source_label='cc',fasttext_format_str='__label__cc', max_sample = max_sample)
    return texts, count


def build_unfiltered_cc_data(cc_file_path: str, max_sample: int) -> Tuple[List[str], int]:
    """
    用于从通用爬取(Common Crawl)的WARC文件中构建随机负样本。
    """
    assert max_sample > 0
    print(f"开始从 {cc_file_path} 筛选随机cc样本...")
    texts, count = build_unfiltered_data(cc_file_path, quality_label='低质量', source_label='cc', fasttext_format_str='__label__cc', max_sample = max_sample)
    return texts, count


def build_classifier_dataset(
    wiki_warc_path: str,  # 新增参数：不再硬编码
    common_crawl_warc_path: str, # 新增参数：不再硬编码
    output_dir: str,
    output_base_name: str,
    output_suffix: str,
    train_ratio: float = 0.9,
    negative_sample_strategy: Literal['filtered', 'unfiltered'] = 'filtered',
    max_sample: Optional[int] = None,
    split: bool = False,
    train_suffix: str = 'train',
    test_suffix: str = 'test'
):
    """
    构建并保存一个用于训练文本质量分类器的标签化数据集。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 步骤1：获取正样本 (Wiki)
    positive_samples, positive_count = build_filtered_wiki_data(wiki_warc_path, max_sample=max_sample)
    print(f"找到的wiki样本数量: {positive_count}")

    if positive_count == 0:
        print("警告: 未找到正样本，程序停止。")
        return

    # 步骤2：获取负样本 (CC)
    if negative_sample_strategy == 'filtered':
        nagative_samples, negative_count = build_filtered_cc_data(common_crawl_warc_path, positive_count)
    elif negative_sample_strategy == 'unfiltered':
        nagative_samples, negative_count = build_unfiltered_cc_data(common_crawl_warc_path, positive_count)
    else:
        raise ValueError(f'未知的负样本采样策略: {negative_sample_strategy}')
    
    print(f"找到的cc样本数量: {negative_count}")

    # 步骤3：合并与打乱
    all_samples = positive_samples + nagative_samples
    random.shuffle(all_samples)
    total_count = len(all_samples)
    
    # 构建输出文件名
    output_prefix = os.path.join(output_dir, f"{output_base_name}_{total_count}samples")

    # 步骤4：保存文件
    if split:
        split_index = int(len(all_samples) * train_ratio)
        print(f'选取{split_index}条训练样本, {total_count - split_index}条测试样本')
        
        # 注意：这里保留原始逻辑，文件名会自动加上 .train 或 .test
        train_file_path = f"{output_prefix}.{train_suffix}" 
        test_file_path = f"{output_prefix}.{test_suffix}"
        
        with open(train_file_path, 'w', encoding='utf-8') as train_f:
            for line in all_samples[:split_index]:
                train_f.write(line + '\n')
        
        with open(test_file_path, 'w', encoding='utf-8') as test_f:
            for line in all_samples[split_index:]:
                test_f.write(line + '\n')
    else:
        output_file_path = f"{output_prefix}.{output_suffix}"
        with open(output_file_path, 'w', encoding='utf-8') as output_f:
            for sample in all_samples:
                output_f.write(sample + '\n')
    
    print(f"已将数据集保存到 {output_dir}，总样本数: {total_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="构建用于文本质量分类的FastText格式数据集 (Wiki vs Common Crawl)"
    )

    # 输入文件路径参数
    parser.add_argument(
        '--wiki_path', 
        type=str, 
        required=True,
        help="维基百科WARC文件的路径 (正样本源)"
    )
    parser.add_argument(
        '--cc_path', 
        type=str, 
        required=True,
        help="Common Crawl WARC文件的路径 (负样本源)"
    )

    # 输出参数
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='data/dataset', 
        help="输出目录路径 (默认: data/dataset)"
    )
    parser.add_argument(
        '--output_base', 
        type=str, 
        default='quality_classifier', 
        help="输出文件名的前缀 (默认: quality_classifier)"
    )
    parser.add_argument(
        '--output_suffix', 
        type=str, 
        default='txt', 
        help="不分割时的文件后缀 (默认: txt)"
    )

    # 采样与策略参数
    parser.add_argument(
        '--strategy', 
        type=str, 
        choices=['filtered', 'unfiltered'], 
        default='filtered',
        help="负样本采样策略: 'filtered' (同样经过高质量筛选) 或 'unfiltered' (随机抽样) (默认: filtered)"
    )
    parser.add_argument(
        '--max_sample', 
        type=int, 
        default=None, 
        help="最大正样本数量。如果未指定，将使用所有可用样本。"
    )

    # 数据集分割参数
    parser.add_argument(
        '--split', 
        action='store_true', 
        help="是否将数据集分割为训练集和测试集"
    )
    parser.add_argument(
        '--train_ratio', 
        type=float, 
        default=0.9, 
        help="训练集比例 (0.0 - 1.0) (默认: 0.9)"
    )
    parser.add_argument(
        '--train_suffix', 
        type=str, 
        default='train', 
        help="训练集文件后缀 (默认: train)"
    )
    parser.add_argument(
        '--test_suffix', 
        type=str, 
        default='test', 
        help="测试集文件后缀 (默认: test)"
    )

    args = parser.parse_args()

    # 调用主函数
    build_classifier_dataset(
        wiki_warc_path=args.wiki_path,
        common_crawl_warc_path=args.cc_path,
        output_dir=args.output_dir,
        output_base_name=args.output_base,
        output_suffix=args.output_suffix,
        train_ratio=args.train_ratio,
        negative_sample_strategy=args.strategy,
        max_sample=args.max_sample,
        split=args.split,
        train_suffix=args.train_suffix,
        test_suffix=args.test_suffix
    )