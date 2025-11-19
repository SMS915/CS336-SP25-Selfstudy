import csv
from typing import Optional
from tqdm import tqdm
from extraction import extract_text
from .filter import identify_language, classify_nsfw, classify_toxic_speech, gopher_quality_filter
from fastwarc.warc import ArchiveIterator, WarcRecordType

def process_warc_to_csv(warc_path: str, csv_base_name: str, source_type: str,sample_size: Optional[int] = None, gopher_prefilter: bool = False) -> None:
    """
    处理WARC文件，对每个文档应用全套分类器与过滤器，并将详细结果保存至CSV文件。

    主要用于数据探索与调试阶段，为确定最佳过滤阈值提供量化依据。

    具体来说，遍历指定的WARC文件，对每个有效文档执行所有已实现的质量评估函数
    （语言识别、NSFW/Toxic分类、Gopher规则等），并捕获每一个评估的原始输出
    （包括标签和置信度分数），连同原始文本结构化地写入一个CSV行中。

    生成CSV文件旨在被加载到数据分析环境中，以便
    可视化不同数据源（例如'wiki' vs 'cc'）下，各项质量分数的数据分布。
    手动检查处于决策边界（高分/低分）的样本，以验证分类器的性能。
    通过敏感性分析，系统性地评估不同阈值组合对数据保留率的影响。
    
    Args:
        warc_path (str): 需要处理的输入WARC文件的路径。
        csv_base_name (str): 用于构建输出CSV文件名的基础名称。最终文件名将包含
                             采样大小和预过滤策略等信息。
        source_type (str): 为此数据源指定的标签（例如, 'wiki', 'common_crawl'）。
                           该标签将被写入CSV的每一行，对于后续在分析中区分和
                           对比不同来源的数据至关重要。
        sample_size (Optional[int]): 指定要处理并写入CSV的最大样本数量。用于在大型
                                     WARC文件上进行快速的初步分析。如果为 `None`，
                                     则会处理整个文件。默认为 `None`。
        gopher_prefilter (bool): 一个控制实验行为的开关。如果为 `True`，则只有通过了
                                 计算成本较低的Gopher过滤器后，文档才会被送入计算
                                 成本较高的fastText分类器，从而便于评估
                                 该预过滤策略对性能和最终样本分布的影响。
                                 默认为 `False`。

    Returns:
        None: 此函数没有返回值。

    Side Effects:
        在文件系统中创建一个CSV文件。文件的具体路径和名称由输入参数动态构建。
    """
    sample_count_str = f"{sample_size}" if sample_size is not None else "all"
    filter_flag = 'filtered' if gopher_prefilter else 'unfiltered'
    output_csv_path = f"{csv_base_name}_{filter_flag}_samples_{sample_count_str}.csv"
        
    # 定义CSV的表头
    fieldnames = [
        'source_type', 'url', 'lang_code', 'lang_score', 'nsfw_label',
        'nsfw_score', 'toxic_label', 'toxic_score', 'gopher_pass', 'text'
    ]
    sample_count = 0
    # 打开WARC文件和CSV文件
    with open(warc_path, 'rb') as warc_file, \
         open(output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        # 遍历WARC文件中的每条记录
        for record in tqdm(ArchiveIterator(warc_file), desc=f"Processing to {output_csv_path}"):
            if record.record_type != WarcRecordType.response or record.http_content_type != 'text/html':
                continue

            html_bytes = record.reader.read()
            text = extract_text(html_bytes)
            if not text or text.isspace():
                continue
            gopher_passed = gopher_quality_filter(text)
            if gopher_prefilter and not gopher_passed:
                continue
            lang_code, lang_score = identify_language(text)
            nsfw_label, nsfw_score = classify_nsfw(text)
            toxic_label, toxic_score = classify_toxic_speech(text)

            # 将所有结果组织成一个字典
            row_data = {
                'source_type': source_type,
                'url': record.headers.get('WARC-Target-URI', ''),
                'lang_code': lang_code,
                'lang_score': lang_score,
                'nsfw_label': nsfw_label,
                'nsfw_score': nsfw_score,
                'toxic_label': toxic_label,
                'toxic_score': toxic_score,
                'gopher_pass': gopher_passed,
                'text': text,
            }
            writer.writerow(row_data)
            sample_count += 1

            if sample_size and sample_count >= sample_size:
                break           

    print(f"处理完成！结果已保存到 {output_csv_path}")

if __name__ == "__main__":
    # 处理WIKI样本
    process_warc_to_csv(
        warc_path='data/wiki/subsampled_positive_500_urls.warc.gz',
        csv_base_name='data/sample_csv/wiki_sample',
        source_type='wiki',
        gopher_prefilter=True,
        sample_size=500,
    )

    process_warc_to_csv(
        warc_path='data/wiki/subsampled_positive_500_urls.warc.gz',
        csv_base_name='data/sample_csv/wiki_sample',
        source_type='wiki',
        gopher_prefilter=False,
        sample_size=500,
    )
    
    # 处理 Common Crawl样本
    process_warc_to_csv(
        warc_path='data/crawls/CC-MAIN-20250417135010-20250417165010-00065.warc.gz',
        csv_base_name='data/sample_csv/cc_sample',
         source_type='common_crawl',
        gopher_prefilter=True,
        sample_size=1000,
    )

    process_warc_to_csv(
        warc_path='data/crawls/CC-MAIN-20250417135010-20250417165010-00065.warc.gz',
        csv_base_name='data/sample_csv/cc_sample',
        source_type='common_crawl',
        gopher_prefilter=False,
        sample_size=1000,
    )