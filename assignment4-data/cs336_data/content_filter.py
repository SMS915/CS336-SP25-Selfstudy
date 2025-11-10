import fasttext
import regex
from fasttext import FastText as FTModel

def identify_language(text: str, model_path: str = 'data/classifiers/lid.176.bin'):
    """
    Identify the language of the given text using a FastText language identification model.

    Args:
        text (str): The input text whose language needs to be identified.
        model_path (str): The path to the FastText language identification model file.

    Returns:
        str: The identified language code.
    """
    text = text.replace('\n', ' ').strip()
    model= FTModel.load_model(model_path)
    predictions = model.predict(text)
    language_code = predictions[0][0].replace('__label__', '')
    score = predictions[1][0].item()
    return language_code, score

def mask_email(text: str, replace_str: str = '|||EMAIL_ADDRESS|||') -> tuple[str, int]:
    pattern = regex.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    masked_text, num_subs = pattern.subn(replace_str, text)
    return masked_text, num_subs

def mask_phone_number(text: str, replace_str: str = '|||PHONE_NUMBER|||') -> tuple[str, int]:
    pattern = regex.compile(r'(?<!\d)(?:\+?1[-\s\.]*)?(?:\(\d{3}\)|\d{3})[-\s\.]*\d{3}[-\s\.]*\d{4}\b')
    masked_text, num_subs = pattern.subn(replace_str, text)
    return masked_text, num_subs

def mask_ip_address(text:str, replace_str: str = '|||IP_ADDRESS|||') -> tuple[str, int]:
    pattern = regex.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    masked_text, num_subs = pattern.subn(replace_str, text)
    return masked_text, num_subs


    

