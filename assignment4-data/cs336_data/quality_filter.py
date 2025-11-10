import nltk

def gopher_quality_filter(text: str) -> bool:
    """
    使用 Gopher 模型对文本进行质量过滤。
    返回 True 表示文本通过质量过滤，False 表示未通过。
    在该简短实现中包括四个规则:
    Word Count次数统计: 字符串的词数应该在50-100000之间。
    Mean word length平均词长: 词的平均长度应在3到10个字符之间。
    Ellipsis Lines省略号行数: 文本中省略号结尾的行数不应超过总行数的30%。
    Alphabetic Words包含字母的单词比例: 包含字母的单词应占总单词数的至少80%。
    
    Args:
        text (str): 要评估的输入文本。
    Returns:
        bool: 如果文本通过质量过滤则返回 True，否则返回 False。
    """
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    words = nltk.word_tokenize(text)
    word_count = len(words)
    if word_count < 50 or word_count > 100000:
        return False
    
    mean_word_length = sum(len(word) for word in words) / word_count
    if mean_word_length < 3 or mean_word_length > 10:
        return False
    
    lines = text.splitlines()
    ellipsis_lines_count = sum(1 for line in lines if line.strip().endswith('...'))
    if (ellipsis_lines_count / len(lines)) > 0.3:
        return False
    
    alphabetic_word_count = sum(1 for word in words if contains_alphabetic(word))
    if (alphabetic_word_count / word_count) < 0.8:
        return False
    
    return True

def contains_alphabetic(word: str) -> bool:
    return any(char.isalpha() for char in word)