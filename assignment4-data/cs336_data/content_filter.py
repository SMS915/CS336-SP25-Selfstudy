import fasttext
import regex
from fasttext import FastText as FTModel


def classify_text(text: str, type: str) -> tuple[str, float]:
    """
    Classify the given text using a FastText model.

    Args:
        text (str): The input text to classify.
        model_path (str): The path to the FastText classification model file.

    Returns:
        tuple: A tuple containing the predicted label and the associated confidence score.
    """
    path_dict = {
        'language': 'data/classifiers/lid.176.bin',
        'nsfw': 'data/classifiers/jigsaw_fasttext_bigrams_nsfw_final.bin',
        'toxic_speech': 'data/classifiers/jigsaw_fasttext_bigrams_hatespeech_final.bin'}
    
    model_path = path_dict.get(type)

    if model_path is None:
        raise ValueError(f"Unsupported classification type: {type}")
    text = text.replace('\n', ' ').strip()
    model = FTModel.load_model(model_path)
    predictions = model.predict(text)
    label = predictions[0][0].replace('__label__', '')
    score = predictions[1][0].item()
    return label, score

def identify_language(text: str) -> tuple[str, float]:
    return classify_text(text, 'language')

def classify_nsfw(text: str) -> tuple[str, float]:
    return classify_text(text, 'nsfw')

def classify_toxic_speech(text: str) -> tuple[str, float]:
    return classify_text(text, 'toxic_speech')

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


    

