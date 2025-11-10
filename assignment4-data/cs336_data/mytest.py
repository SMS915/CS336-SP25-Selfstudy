import random
from tqdm import tqdm
from fastwarc.warc import ArchiveIterator, WarcRecordType
from cs336_data.extraction import extract_text
from cs336_data.content_filter import identify_language, classify_nsfw
from cs336_data.quality_filter import gopher_quality_filter

def analyze_languages_in_warc(warc_file_path: str, sample_size: int = 20, sample_factor: float = 5) -> None:
    """
    Analyze the languages of text extracted from a WARC file.

    Args:
        warc_file_path (str): The path to the WARC file.
        sample_size (int): The number of records to sample for language identification.
    Returns:
        list of tuples: A list containing tuples of (language_code, score) for each sampled record.
    """
    chosen = []
    total_size = round(sample_factor * sample_size)
    with open(warc_file_path, 'rb') as warc_file:
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc='遍历文件') as progress_bar:
            for record in ArchiveIterator(warc_file):
                if record.record_type == WarcRecordType.response and record.http_content_type == 'text/html':
                    html_bytes = record.reader.read()
                    if not html_bytes:
                        continue
                    text = extract_text(html_bytes)
                    if text and not text.isspace():
                        chosen.append(text)
                    progress_bar.update(1)
                if len(chosen) > total_size:
                    break

    samples = random.sample(chosen[10:], sample_size)
    samples = chosen[:10] + samples
    results = []
    for text in samples:
        lang_code, score = identify_language(text)
        nsfw, nsfw_score = classify_nsfw(text)
        results.append({"url": record.headers.get('WARC-Target-URI', ''),
                        "prefix": text[:200],
                        "language_code": lang_code,
                        "score_lang": score,
                        'nsfw_label': nsfw,
                        'score_nsfw': nsfw_score})

    print(f"--- 从 {warc_file_path} 中随机抽取 {len(results)} 个样本进行分析 ---")
    for i, result in enumerate(results):
        print(f"\n--- 样本 {i+1} ---")
        print(f"URL: {result['url']}")
        print(f"语言识别模型预测: {result['language_code']} (分数: {result['score_lang']:.4f})")
        print(f"NSFW 分类模型预测: {result['nsfw_label']} (分数: {result['score_nsfw']:.4f})")
        print(f"文本片段: {result['prefix']}")


def test_geropher_quality_filter(warc_file_path: str, sample_size: int = 20) -> None:
    chosen = []
    with open(warc_file_path, 'rb') as warc_file:
        with tqdm(total=sample_size, unit='iB', unit_scale=True, desc='遍历文件') as progress_bar:
            for record in ArchiveIterator(warc_file):
                if record.record_type == WarcRecordType.response and record.http_content_type == 'text/html':
                    html_bytes = record.reader.read()
                    if not html_bytes:
                        continue
                    text = extract_text(html_bytes)
                    if text and not text.isspace():
                        if gopher_quality_filter(text):
                            chosen.append(text)
                            progress_bar.update(1)
                if len(chosen) > sample_size:
                    break

    print(f"--- 从 {warc_file_path} 中检测 {len(chosen)} 个样本进行分析 ---")
    for i, result in enumerate(chosen):
        print(f"\n--- 样本 {i+1} ---")
        print(f"文本: {result[:1000]}")


if __name__ == "__main__":
    # 1. 指定模型路径和WARC文件路径
    WARC_FILE_PATH = 'data/crawls/CC-MAIN-20250417135010-20250417165010-00065.warc.gz' 
    # analyze_languages_in_warc(WARC_FILE_PATH, 20)
    test_geropher_quality_filter(WARC_FILE_PATH, 20)

        