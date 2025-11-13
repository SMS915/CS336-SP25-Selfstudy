import os
import nltk
import random
import mmh3
import numpy as np
from nltk import ngrams
from tqdm import tqdm
from hashlib import md5,sha256
from functools import partial
from typing import Optional, List, Dict, Tuple, Set, Callable
from itertools import combinations
from collections import defaultdict
random_seed = 42
random.seed(random_seed)


def exact_line_deduplication(input_files: List[os.PathLike], output_directory: os.PathLike) -> None:
    """
    Perform exact line deduplication on the input text files and save the deduplicated
    versions to the output directory.

    Args:
        input_files (List[os.PathLike]): List of input text file paths.
        output_directory (os.PathLike): Directory to save deduplicated files.
    """
    line_counts = {}
    os.makedirs(output_directory, exist_ok=True)
    for input_file in tqdm(input_files, desc="正在运行精确行去重"):
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line_hash = md5(line.strip().encode('utf-8')).hexdigest()
                line_counts[line_hash] = line_counts.get(line_hash, 0) + 1

    for input_file in input_files:
        output_file = os.path.join(output_directory, os.path.basename(input_file))
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout:
            for line in fin:
                line_hash = md5(line.strip().encode('utf-8')).hexdigest()
                if line_counts[line_hash] == 1:
                    fout.write(line)


def convert_text_into_n_gram(n: int, text: str) -> set[tuple[str, ...]]:
    """
    将输入文本转换为词级别的n-gram集合。
    对文本进行清洗并分割，提取n-grams，并存储在集合中

    Args:
        n (int): n-gram中单词的数量(n > 0) 
        text (str): 共处理的原始文本
    Returns:
        set[tuple[str, ...]]: 一个包含所有唯一n-gram的集合。
                             每个n-gram本身是一个长度为n的字符串元组。
                              如果分词后的单词数少于n，则返回一个空集合。
    """
    clean_words = text.replace('\n', ' ').replace('\r', ' ').lower().split()
    if len(clean_words) < n:
        return set()
    
    n_gram_set = set(ngrams(clean_words, n))

    return n_gram_set


def compute_minhash_signature(n_grams: Set[Tuple[str,...]], hash_functions: List[Callable[[str], int]]) -> np.ndarray:
    """
    计算给定n-gram集合的MinHash签名。

    Args:
        n_grams (Set[Tuple[str,...]]): 输入的n-gram集合。
        hash_functions (List[Callable[[str], int]]): 用于计算MinHash签名的哈希函数列表。
    Returns:
        np.ndarray: MinHash签名数组，其中每个元素对应一个哈希函数的最小哈希值。
    """
    max_hash_val = 2**32 - 1
    signature = np.full(len(hash_functions), max_hash_val, dtype=np.uint32) # num_hashes长度的数组，初始值为最大哈希值

    for n_gram in n_grams:
        n_gram_str = ' '.join(n_gram)
        all_hashes = np.array([hash_func(n_gram_str) for hash_func in hash_functions], dtype=np.uint32) # 计算该n-gram在所有哈希函数下的哈希值，长度同样为num_hashes
        signature = np.minimum(signature, all_hashes) # 向量化更新最小值

    return signature


def generate_hash_functions(num_hashes: int, max_hash: int = 2**32 - 1) -> List[Callable[[str], int]]:
    """
    基于mmh3生成一组高性能的哈希函数，用于MinHash签名计算。

    Args:
        num_hashes (int): 要生成的哈希函数数量。
        max_hash (int): 哈希值的最大范围，默认为2^32 - 1。 
    Returns:
        List[Callable[[str], int]]: 生成的哈希函数列表。
    """
    hash_functions = []
    for _ in range(num_hashes):
        seed = random.randint(0, max_hash)
        hash_func = partial(mmh3.hash, seed=seed, signed=False)
        hash_functions.append(hash_func)
    return hash_functions

def generate_signature_for_texts(input_files: List[os.PathLike],
                                 n: int,
                                 num_hashed:int,
                                 ) -> Tuple[Dict[int, np.ndarray], Dict[int, os.PathLike]]:
    """
    为输入文本文件生成MinHash签名。
    Args:
        input_files (List[os.PathLike]): 输入文本文件路径列表。
        n (int): n-gram中单词的数量(n > 0)。
        num_hashed (int): 用于MinHash签名计算的哈希函数数量。
    Returns:
        Dict[str, np.ndarray]: 包含每个文件MinHash签名的字典，键为文件路径，值为对应的MinHash签名数组。
    """
    hash_functions = generate_hash_functions(num_hashed)
    id_signatures_dict = {}
    path_ids_dict = {}
    for i, input_file in tqdm(enumerate(input_files), desc="正在为文本生成MinHash签名"):
        doc_id = i
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
            n_gram_set = convert_text_into_n_gram(n, text)
            signature = compute_minhash_signature(n_gram_set, hash_functions)
            id_signatures_dict[doc_id] = signature
            path_ids_dict[doc_id] = input_file

    return id_signatures_dict, path_ids_dict


def get_lsh_candidate_pairs(num_band: int, num_hashes: int, id_signatures_dict: dict[int, np.ndarray]) -> set[tuple[int, ...]]:
    assert num_hashes % num_band == 0
    rows = num_hashes // num_band

    buckets = defaultdict(list)
    candidate_pairs = set()
    for id, signature in id_signatures_dict.items():
        for b in range(num_band):
            band = tuple(signature[b * rows: (b + 1) * rows])
            band_hash = hash(band)
            band_key = (b, band_hash)

            buckets[band_key].append(id)
    
    for collision_doc_list in buckets.values():
        collision_pairs = combinations(collision_doc_list, 2)
        for pair in collision_pairs:
            candidate_pairs.add(pair)

    return candidate_pairs





# if __name__ == '__main__':
#     test_n_gram_set = convert_text_into_n_gram(3, 'test converting \nthis string into n-gram test converting this')
#     for n_gram in test_n_gram_set:
#         print(type(n_gram))
#         print(n_gram[i] for i in range(len(n_gram)))

