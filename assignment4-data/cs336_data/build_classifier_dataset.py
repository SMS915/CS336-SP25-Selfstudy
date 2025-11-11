import random
from tqdm import tqdm
from typing import Optional, Dict, Any
from fastwarc.warc import ArchiveIterator, WarcRecordType
from cs336_data.extraction import extract_text
from cs336_data.content_filter import identify_language, classify_nsfw, classify_toxic_speech
from cs336_data.quality_filter import gopher_quality_filter

def judge_high_quality(text: str, lang_threshold: float = 0.9, nsfw_threshold: float = 0.02, toxic_threshold: float = 0.02) -> bool:
    """
    Judge if the given text is of high quality based on classification functions.

    Args:
        text (str): The text content to evaluate.
    Returns:
        bool: True if the text is high quality, False otherwise.
    """
    pass_gopher = gopher_quality_filter(text) #让我自己写
    lang_code, lang_score = identify_language(text)
    nsfw_label, nsfw_score = classify_nsfw(text)
    toxic_label, toxic_score = classify_toxic_speech(text)

    nsfw_score = nsfw_score if nsfw_label == 'nsfw' else (1 - nsfw_score)
    toxic_score = toxic_score if toxic_label == 'toxic' else (1 - toxic_score)
    is_high_quality = (pass_gopher and
                       lang_code == 'en' and
                       lang_score >= lang_threshold and
                       nsfw_score <= nsfw_threshold and
                       toxic_score <= toxic_threshold)
    return is_high_quality



def build_positive_dataset(warc_file_path: str, label: str, max_sample: Optional[int] = None) -> tuple[list[str], int]:
    """
    Build a positive dataset from a WARC file by extracting text content.

    Args:
        warc_file_path (str): The path to the WARC file.
    Returns:
        tuple: A tuple containing a list of extracted text contents and the count of valid records.
    """
    
    

