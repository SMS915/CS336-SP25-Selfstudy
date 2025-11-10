from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract import html2text
from resiliparse.parse import encoding

def extract_text(byte_string: bytes) -> str:
    """
    A function that takes a byte string containing HTML and returns a string containing the extracted text.
    Args:
        byte_string (bytes): A byte string containing HTML content.
    Returns:
        str: A string containing the extracted text.
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