import random
from fastwarc.warc import ArchiveIterator, WarcRecordType
from cs336_data.extraction import extract_text
from cs336_data.content_filter import identify_language

def analyze_languages_in_warc(warc_file_path: str, sample_size: int = 10):
    """
    Analyze the languages of text extracted from a WARC file.

    Args:
        warc_file_path (str): The path to the WARC file.
        sample_size (int): The number of records to sample for language identification.
    Returns:
        list of tuples: A list containing tuples of (language_code, score) for each sampled record.
    """
    results = []

    with open(warc_file_path, 'rb') as warc_file:
        for record in ArchiveIterator(warc_file):
            if record.record_type == WarcRecordType.response and record.http_content_type == 'text/html':
                html_bytes = record.reader.read()
                if not html_bytes:
                    continue
                text = extract_text(html_bytes)
                if text and not text.isspace():
                    lang_code, score = identify_language(text)
                    results.append({"url": record.headers.get('WARC-Target-URI', ''),
                                    "prefix": text[:200],
                                    "language_code": lang_code,
                                    "score": score})
            if len(results) > 10 * sample_size:
                break

    samples = random.sample(results, sample_size)

    print(f"--- 从 {warc_file_path} 中随机抽取 {len(samples)} 个样本进行分析 ---")
    for i, sample in enumerate(samples):
        print(f"\n--- 样本 {i+1} ---")
        print(f"URL: {sample['url']}")
        print(f"模型预测: {sample['language_code']} (置信度: {sample['score']:.4f})")
        print(f"文本片段: {sample['prefix']}")

if __name__ == "__main__":
    # 1. 指定模型路径和WARC文件路径
    WARC_FILE_PATH = 'data/crawls/CC-MAIN-20250417135010-20250417165010-00065.warc.gz' 
    analyze_languages_in_warc(WARC_FILE_PATH, 20)

        