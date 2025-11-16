import random
import os
from tqdm import tqdm
from typing import List, Literal, Optional, Tuple
from fastwarc.warc import ArchiveIterator, WarcRecordType
from .extraction import extract_text
from .filter import judge_high_quality

random.seed(42)


def build_filtered_data(warc_file_path: str, quality_label: str, source_label: str, fasttext_format_str: str, max_sample: Optional[int] = None) -> tuple[list[str], int]:
    """
    从WARC文件中筛选并通过所有高质量检查的文本内容，构建一个数据集。

    遍历指定的WARC文件，对每个页面执行评定函数中定义的所有过滤规则（语言、Gopher、
    NSFW、Toxic）。只有通过所有检查的页面才会被保留。

    Args:
        warc_file_path (str): 输入的WARC文件路径。
        quality_label (str): 在进度条中显示的标签。
        fasttext_format_str (str): 用于格式化输出的fastText标签字符串 (例如, '__label__wiki')。
        max_sample (Optional[int]): 要提取的最大样本数量。一旦达到此数量，将提前停止扫描。

    Returns:
        tuple[list[str], int]: 一个元组，包含：
                               - 格式化后的高质量文本行列表。
                               - 实际找到的高质量样本数量。
    """
    texts = []
    
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


def build_unfiltered_data(warc_file_path: str, quality_label: str, source_label: str, fasttext_format_str: str, max_sample: int) -> tuple[list[str], int]
    """
    从WARC文件中进行水塘抽样，构建一个无筛选的数据集。

    使用水塘抽样算法来确保在只遍历一次文件的情况下，对所有有效页面（非空）进行公平的随机抽样。
    此函数不会进行任何内容质量过滤。

    Args:
        warc_file_path (str): 输入的WARC文件路径。
        quality_label (str): 在进度条中显示的标签。
        fasttext_format_str (str): 用于格式化输出的fastText标签字符串 (例如, '__label__cc')。
        max_sample (int): 要抽样的确切样本数量。

    Returns:
        tuple[list[str], int]: 一个元组，包含：
                               - 格式化后的随机文本行列表。
                               - 实际抽样到的样本数量。
    """
    negative_texts = []
    
    with open(warc_file_path, 'rb') as warc_file:
        iterator = ArchiveIterator(warc_file)
        progress_bar = tqdm(iterator, desc=f'正在扫描 {warc_file_path} 以获取随机负样本', unit=' records')

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
    用于从通用爬取(Common Crawl)的WARC文件中构建高质量负样本,通过调用带过滤函数，找到同样通过了严格质量检查的普通网页。
    用于分类器学习 wiki引用页面与常规CC页面 在基础文本质量外深层特征分辨能力的训练策略。
    """
    print(f"开始从 {cc_file_path} 筛选高质量cc样本...")
    texts, count = build_filtered_data(cc_file_path, quality_label='高质量',source_label='cc',fasttext_format_str='__label__cc', max_sample = max_sample)
    return texts, count


def build_unfiltered_cc_data(cc_file_path: str, max_sample: int) -> Tuple[List[str], int]:
    """
    用于从通用爬取(Common Crawl)的WARC文件中构建随机负样本,通过调用不加过滤机制的函数来获取完全随机的、未经任何质量筛选的网页。
    用于分类器学习干净文本与“混沌”文本分辨能力的训练策略。
    """
    assert max_sample > 0
    print(f"开始从 {cc_file_path} 筛选高质量cc样本...")
    texts, count = build_unfiltered_data(cc_file_path, quality_label='低质量', source_label='cc', fasttext_format_str='__label__cc', max_sample = max_sample)
    return texts, count


def build_classifier_dataset(
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

    此函数执行一个端到端的数据集生成流程。它首先从维基百科源文件中提取
    经过高质量标准筛选的正样本。随后，根据指定的负样本采样策略，
    从Common Crawl源文件中生成与正样本数量相匹配的负样本。

    最终，函数将合并后的正负样本随机打乱，并根据 'split' 参数决定是保存为
    单个文件，还是分割为训练集和测试集两个文件。

    Args:
        output_dir (str): 数据集文件的输出目录路径。
        output_base_name (str): 输出文件的主名称，不包含样本数量和后缀。
        output_suffix (str): 当不进行分割时，使用的文件后缀。
        train_ratio (float, optional): 若分割，训练集在分割后所占的比例。
                                       默认为 0.9。
        negative_sample_strategy (Literal['filtered', 'unfiltered'], optional): 
            指定负样本的生成策略。默认为 'filtered'。
            - 'filtered': 负样本同样经过与正样本相同的高质量标准筛选。
            - 'unfiltered': 负样本通过对原始数据进行随机抽样生成，不应用内容质量过滤。
        max_sample (Optional[int], optional): 要提取的正样本的最大数量。如果为 `None`，
                                              则处理所有源文件以提取所有符合条件的
                                              正样本。负样本的数量将与最终提取到的
                                              正样本数量相匹配。默认为 `None`。
        split (bool, optional): 是否将最终的数据集分割为训练集和测试集。
                                如果为 `True`，将生成两个文件。默认为 `False`。
        train_suffix (str, optional): 当 `split=True` 时，训练集的文件后缀。默认为 'train'。
        test_suffix (str, optional): 当 `split=True` 时，测试集的文件后缀。默认为 'test'。

    Returns:
        None: 此函数没有返回值，其结果是直接写入到文件系统的一个或多个文件。

    Raises:
        ValueError: 当提供了未知的 `negative_sample_strategy` 时抛出。
    """
    os.makedirs(output_dir, exist_ok=True)

    wiki_warc_path = 'data/wiki/subsampled_positive_15000_pages.warc.gz'
    positive_samples, positive_count =  build_filtered_wiki_data(wiki_warc_path, max_sample=max_sample)
    print(f"找到的wiki样本数量: {positive_count}")


    common_crawl_warc_path = 'data/crawls/CC-MAIN-20250417135010-20250417165010-00065.warc.gz'

    if negative_sample_strategy == 'filtered':
        nagative_samples, negative_count = build_filtered_cc_data(common_crawl_warc_path, positive_count)
    elif negative_sample_strategy == 'unfiltered':
        nagative_samples, negative_count = build_unfiltered_cc_data(common_crawl_warc_path, positive_count)
    else:
        raise('未知的负样本采样策略')
    
    print(f"找到的cc样本数量: {negative_count}")

    all_samples = positive_samples + nagative_samples
    random.shuffle(all_samples)
    total_count = len(all_samples)
    output_prefix = os.path.join(output_dir, f"{output_base_name}_{total_count}samples")

    if split:
        split_index = int(len(all_samples) * train_ratio)
        print(f'选取{split_index}条训练样本, {total_count - split_index}条测试样本')
        train_file_path = os.path.join(output_prefix, f'.{train_suffix}')
        test_file_path = os.path.join(output_prefix, f'.{test_suffix}')
        with open(train_file_path, 'w', encoding='utf-8') as train_f, open(test_file_path, 'w', encoding='utf-8') as test_f:
            for line in range(total_count):
                if line <= split_index:
                    train_f.write(all_samples[line] + '\n')
                else:
                    test_f.write(all_samples[line] + '\n')
    else:
        output_file_path = os.path.join(output_prefix, f'.{output_suffix}')
        with open(output_file_path, 'w', encoding='utf-8') as output_f:
            for sample in all_samples:
                output_f.write(sample + '\n')
    
    print(f"已将数据集保存到 {output_dir}，总样本数: {total_count}")


