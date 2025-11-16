from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract import html2text
from resiliparse.parse import encoding

def extract_text(byte_string: bytes) -> str:
    """
    接受一个包含html内容的字节流，输出提取后的文本
    Args:
        byte_string (bytes): 一个包含html内容的字节流.
    Returns:
        str: 从html中提取出的文本.
    """
    if not byte_string:
        return ""

    try:
        html_string = byte_string.decode('utf-8')
    except UnicodeDecodeError:
        byte_encoding = encoding.detect_encoding(byte_string)
        html_string = byte_string.decode(byte_encoding, errors='ignore')

    extracted_text = html2text.extract_plain_text(html_string)
    return extracted_text