import os
from collections import defaultdict
from typing import List, Dict, Tuple, Set
import regex
import pickle


def train_bpe_run(input_path : str,
                  vocab_size : int,
                  special_tokens : list[str]
                  ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    给定一个包含训练语料的地址，训练字节对编码器并返回词汇表与融合情况
    given a path to corpus, train a bpe and return its vocabulary and merges

    Args:
        input_path (str | os.PathLike): Path to a text file with BPE tokenizer training data.
        vocab_size (int): A positive integer,defines maximum total number of items in the tokenizer's vocabulary, including special tokens.
        special_tokens (list[str]): List of special tokens used in the tokenizer,
        which will never be merged with other tokens or be split into multiple tokens.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
            merges (list[tuple[bytes, bytes]]): A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>),
                                                representing that <token1> was merged with <token2>. Merges are ordered by order of creation.
    """

    if vocab_size <= 0 or not isinstance(vocab_size, int):
        raise ValueError('vocab_size must be a positive integer')

    vocab : dict[int, bytes] = {i : bytes([i]) for i in range(256)} #初始化256个基础byte
    next_token_id : int = 256

    # 统计每个word出现的频率，每个word以bytes元组形式表示，类型为Dict[Tuple[bytes], int]
    word_freq_table = defaultdict(int)
    # 用于查询一个符号是否存在于vocab中
    existing_token_values : Set[bytes] = set(vocab.values())

    # 添加特殊字符
    for special_str in special_tokens:
        if len(vocab) >= vocab_size: break
        st_bytes = special_str.encode("utf-8")
        if st_bytes not in existing_token_values:
            vocab[next_token_id] = st_bytes
            existing_token_values.add(st_bytes)
            next_token_id += 1

    # 读取语料库
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        text = ""

    # pre-tokenization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}+|\s+(?!\S)|\s+"""
    chunks = regex.split('|'.join(map(regex.escape,special_tokens)),text)  # 按照special_tokens 进行一次分割
    for chunk in chunks:
        for word in regex.findall(PAT, chunk):
            word_bytes = word.encode("utf-8")
            byte_lists = [bytes([b]) for b in word_bytes] #编码每个单词，并转换为bytes
            word_freq_table[tuple(byte_lists)] += 1

    merges : list[tuple[bytes, bytes]] = []
    pair_counts = defaultdict(int)

    for word in word_freq_table.keys():
        for i in range(len(word) - 1):
            pair_counts[(word[i], word[i + 1])] += word_freq_table[word] #直接累加word频率，无需遍历全文

    # 训练bpe
    while len(vocab) < vocab_size:
        if not pair_counts:
            break
        max_count = max(pair_counts.values())
        candidates = [k for k, v in pair_counts.items() if v == max_count]
        greatest_pair = max(candidates)

        merges.append(greatest_pair)

        new_token_bytes = greatest_pair[0] + greatest_pair[1]
        vocab[next_token_id] = new_token_bytes
        next_token_id += 1

        affected_words = [] # 在该次合并后，需要进行修改的tokens

        for word, freq in word_freq_table.items():
            for i in range(len(word) - 1):
                if (word[i], word[i + 1]) == greatest_pair:
                    affected_words.append((word, freq))
                    break  # 找到一个就够了，不需要继续检查

        for word, freq in affected_words:
            i = 0
            new_word_list = [] # 修改pair_count的同时构建新词，替换到频率字典中
            while i < len(word) - 1:
                if (word[i], word[i + 1]) == greatest_pair:
                    new_word_list.append(new_token_bytes)
                    if i > 0:
                        prev_token = word[i - 1]
                        pair_counts[prev_token, word[i]] -= freq
                        pair_counts[prev_token, new_token_bytes] += freq

                    if i < len(word) - 2:
                        next_token = word[i + 2]
                        pair_counts[word[i + 1], next_token] -= freq
                        pair_counts[new_token_bytes, next_token] += freq

                    i += 2
                else:
                    new_word_list.append(word[i])
                    i += 1
            if i == len(word) - 1:
                new_word_list.append(word[-1])

            new_word = tuple(new_word_list)
            del word_freq_table[word]
            word_freq_table[new_word] += freq

        del pair_counts[greatest_pair]

    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open("merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    return vocab, merges

if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe_run("../data/owt_train.txt", 20000, special_tokens = special_tokens)

    print(vocab)
    print(merges)


