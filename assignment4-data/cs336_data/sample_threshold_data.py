import csv
from typing import Optional
from tqdm import tqdm
from cs336_data.extraction import extract_text
from .filter import identify_language, classify_nsfw, classify_toxic_speech, gopher_quality_filter
from fastwarc.warc import ArchiveIterator, WarcRecordType

def process_warc_to_csv(warc_path: str, csv_base_name: str, source_type: str,sample_size: Optional[int] = None, gopher_prefilter: bool = False) -> None:
    """
    Processes a WARC file, applies all filters, and saves the results to a CSV file.

    Args:
        warc_path (str): Path to the input WARC file.
        output_csv_path (str): Path to the output CSV file.
        source_type (str): The label for this data source, e.g., 'positive' or 'negative'.
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

            # 5. 将所有结果组织成一个字典
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