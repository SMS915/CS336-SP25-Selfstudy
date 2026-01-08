import regex
import nltk
from fasttext import FastText as FTModel
from typing import Dict
LANG_MODEL_PATH = 'data/classifiers/lid.176.bin'
NSFW_MODEL_PATH = 'data/classifiers/jigsaw_fasttext_bigrams_nsfw_final.bin'
TOXIC_MODEL_PATH = 'data/classifiers/jigsaw_fasttext_bigrams_hatespeech_final.bin'

_MODELS = {
    'lang': None,
    'nsfw': None,
    'toxic': None
}

# 硬编码路径
MODEL_PATHS = {
    'lang': 'data/classifiers/lid.176.bin',
    'nsfw': 'data/classifiers/jigsaw_fasttext_bigrams_nsfw_final.bin',
    'toxic': 'data/classifiers/jigsaw_fasttext_bigrams_hatespeech_final.bin'
}

PHONE_PATTERN = regex.compile(r'(?<!\d)(?:\+?1[-\s\.]*)?(?:\(\d{3}\)|\d{3})[-\s\.]*\d{3}[-\s\.]*\d{4}\b')
EMAIL_PATTERN = regex.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
IP_PATTERN = regex.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')


def _get_model(model_type: str) -> FTModel._FastText:
    """
    单例模式惰性加载模型。
    确保模型只在第一次被需要时加载，且在内存中只有一份。
    """
    if _MODELS[model_type] is None:
        path = MODEL_PATHS.get(model_type)
        if not path:
            raise ValueError(f"Unknown model type: {model_type}")
        try:
            _MODELS[model_type] = FTModel.load_model(path)  # type: ignore
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load {model_type} model from {path}: {e}")
            exit(1)
    return _MODELS[model_type]  # type: ignore

def classify_text(model: FTModel._FastText, text: str) -> tuple[str, float]:
    """
    基于fasttext预加载模型对文本进行分类，返回分数最高的标签与对应分数

    Args:
        text (str): 待分类的输入文本
        model_path (str): Fasttext分类模型路径.

    Returns:
        tuple: 一个元组，包括模型对文本分数最高的标签与相应分数.
    """
    predictions = model.predict(text)
    label = predictions[0][0].replace('__label__', '')  # type: ignore
    score = predictions[1][0].item()
    return label, score


def identify_language(text: str) -> tuple[str, float]:
    # return classify_text(text, 'language')
    text = text.replace('\n', ' ').strip()
    model = _get_model('lang')  # 惰性获取
    return classify_text(model, text)


def classify_nsfw(text: str) -> tuple[str, float]:
    # return classify_text(text, 'nsfw')
    text = text.replace('\n', ' ').strip()
    model = _get_model('nsfw')
    return classify_text(model, text)


def classify_toxic_speech(text: str) -> tuple[str, float]:
    # return classify_text(text, 'toxic_speech')
    text = text.replace('\n', ' ').strip()
    model = _get_model('toxic')
    return classify_text(model, text)


def gopher_quality_filter(text: str) -> bool:
    """
    使用 Gopher 模型对文本进行质量过滤,简单实现,基于四个启发式规则:
    词数统计(Word Count): 字符串的词数应该在50-100000之间。
    平均词长(Mean word length): 词的平均长度应在3到10个字符之间。
    省略号行数(Ellipsis Lines): 文本中省略号结尾的行数不应超过总行数的30%。
    包含字母的单词比例(Alphabetic Words): 包含字母的单词应占总单词数的至少80%。

    Args:
        text (str): 要评估的输入文本。
    Returns:
        bool: 如果文本通过质量过滤则返回 True，否则返回 False。
    """
    # nltk.download('punkt', quiet=True)
    # nltk.download('punkt_tab', quiet=True)

    # words = nltk.word_tokenize(text) # word_tokenize开销偏大，换成split更快
    words = text.split()
    word_count = len(words)

    # 文本的词数应该在50-100000之间。
    if word_count < 50 or word_count > 100000:
        return False

    total_length = 0
    alpha_count = 0
    for word in words:
        total_length += len(word)
        if any(c.isalpha() for c in word):
            alpha_count += 1

    mean_word_length = total_length / word_count
    # 词的平均长度应在3到10个字符之间。
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    lines = text.splitlines()
    ellipsis_lines_count = sum(1 for line in lines if line.strip().endswith('...'))
    # 文本中省略号结尾的行数不应超过总行数的30%。
    if (ellipsis_lines_count / len(lines)) > 0.3:
        return False

    # 包含字母的单词应占总单词数的至少80%。
    if alpha_count / word_count < 0.8:
        return False

    return True


def judge_high_quality(text: str, lang_threshold: float = 0.9, nsfw_threshold: float = 0.02,
                       toxic_threshold: float = 0.02, build_classifier_dataset:bool = False) -> bool:
    """
    结合gopher模型, 语言, nsfw, toxic评分信息, 对文本质量进行判断。

    Args:
        text (str): 用于评估的文本内容.
    Returns:
        bool: 返回布尔值，通过检测代表文本质量有基本保证.
    """

    if build_classifier_dataset:
        if len(text.split()) < 250: 
            return False

    text_lower = text.lower()
    if "performance & security by cloudflare" in text_lower or \
       "attention required!" in text_lower or \
       "enable cookies" in text_lower or "403 error" in text_lower:
        return False # 这是一个已知的坏样本，立即拒绝

    pass_gopher = gopher_quality_filter(text)
    if not pass_gopher:
        return False

    lang_code, lang_score = identify_language(text)
    if lang_code != 'en' or lang_score <= lang_threshold:
        return False

    nsfw_label, nsfw_score = classify_nsfw(text)
    transformed_nsfw_score = nsfw_score if nsfw_label == 'nsfw' else (1 - nsfw_score)
    if transformed_nsfw_score > nsfw_threshold:
        return False

    toxic_label, toxic_score = classify_toxic_speech(text)
    transformed_toxic_score = toxic_score if toxic_label == 'toxic' else (1 - toxic_score)
    if transformed_toxic_score > toxic_threshold:
        return False

    return True


def contains_alphabetic(word: str) -> bool:
    return any(char.isalpha() for char in word)


def mask_email(text: str, replace_str: str = '|||EMAIL_ADDRESS|||') -> tuple[str, int]:
    # [a-zA-Z0-9._%+-]+   - 匹配用户名部分：一个或多个字母、数字或特殊符号(._%+-)。
    # @                  - 匹配'@'符号。
    # [a-zA-Z0-9.-]+     - 匹配域名部分：一个或多个字母、数字、点或连字符。
    # \.                 - 匹配域名和顶级域名之间的点。
    # [a-zA-Z]{2,}       - 匹配顶级域名：至少两个字母。
    pattern = regex.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    masked_text, num_subs = pattern.subn(replace_str, text)
    return masked_text, num_subs


def mask_phone_number(text: str, replace_str: str = '|||PHONE_NUMBER|||') -> tuple[str, int]:
    # (?<!\d)                        边界检查:负向后顾，要求匹配的号码前没有数字，避免从长数字中间匹配。
    # (?: \+?1 [-\s\.]* )?           逻辑段1 (可选): 匹配美国国家代码 (+1 或 1) 和其后的分隔符。
    # (?: \(\d{3}\) | \d{3} )        逻辑段2: 匹配区号, 格式为 "(123)" 或 "123"。
    # [-\s\.]*                       逻辑段3: 匹配区号和号码主体之间的分隔符。
    # \d{3}                          逻辑段4: 匹配号码主体的前三位数字。
    # [-\s\.]*                       逻辑段5: 匹配号码主体中间的分隔符。
    # \d{4}                          逻辑段6: 匹配号码主体的后四位数字。
    # \b                             边界检查: 确保号码在此处结束，是一个完整的数字块。
    pattern = regex.compile(r'(?<!\d)(?:\+?1[-\s\.]*)?(?:\(\d{3}\)|\d{3})[-\s\.]*\d{3}[-\s\.]*\d{4}\b')
    masked_text, num_subs = pattern.subn(replace_str, text)
    return masked_text, num_subs


def mask_ip_address(text: str, replace_str: str = '|||IP_ADDRESS|||') -> tuple[str, int]:
    # \b               边界检查
    # (?:\d{1,3}\.){3} 匹配 ip地址前面的三个 数字. 模式
    # \d{1,3}          匹配最后的数字
    # \b               边界检查,确保结尾是一个完整的数字块
    pattern = regex.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    masked_text, num_subs = pattern.subn(replace_str, text)
    return masked_text, num_subs

def mask_all_pii(text: str) -> tuple[str, Dict[str, int]]:
    phone_masked_text, num_phones = PHONE_PATTERN.subn('|||PHONE_NUMBER|||', text)
    email_masked_text, num_emails = EMAIL_PATTERN.subn('|||EMAIL_ADDRESS|||', text)
    ip_masked_text, num_ips = IP_PATTERN.subn('|||IP_ADDRESS|||', text)
    return phone_masked_text, {'phone_count': num_phones, 'email_count': num_emails, 'ip_count': num_ips}
