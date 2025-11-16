import os
import nltk
import random
import mmh3
import unicodedata
import numpy as np
import regex
from nltk import ngrams
from tqdm import tqdm
from hashlib import md5
from .UF import UnionFind
from functools import partial
from typing import Optional, List, Dict, Tuple, Set, List, Callable, Hashable, cast
from itertools import combinations
from collections import defaultdict
random_seed = 42
random.seed(random_seed)

def normalize_text_for_duplication(text: str) -> str:
    """
    对文本进行全面的标准化，为去重做准备。
    遵循作业要求：小写、去标点、标准化空格、去重音、NFD范式。
    """
    text = unicodedata.normalize('NFD', text.lower()) # 应用NFD Unicode标准化，并将文本转为小写

    text = ''.join(c for c in text if not unicodedata.combining(c)) # 移除重音符号 (组合标记)

    text = regex.sub(r'[^\w\s]', ' ', text) # 匹配并移除任何不是字母、数字、下划线或空白字符的字符，即标点符号

    text = regex.sub(r'\s+', ' ', text).strip() # 标准化空白字符
    
    return text

def exact_line_deduplication(input_files: List[os.PathLike], output_directory: os.PathLike) -> None:
    """
    对输入的文本文件进行精确行去重并写入到指定的输出文件夹中。

    Args:
        input_files (List[os.PathLike]): List of input text file paths.
        output_directory (os.PathLike): Directory to save deduplicated files.
    """
    line_counts = {}
    os.makedirs(output_directory, exist_ok=True)
    for input_file in input_files:
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
    clean_words = normalize_text_for_duplication(text).split()
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
                                 num_hashes:int,
                                 ) -> Tuple[Dict[int, np.ndarray], Dict[int, os.PathLike], Dict[int, tuple[tuple[str, ...]]], List[int]]:
    """
    为输入文本文件生成MinHash签名。
    Args:
        input_files (List[os.PathLike]): 输入文本文件路径列表。
        n (int): n-gram中单词的数量(n > 0)。
        num_hashes (int): 用于MinHash签名计算的哈希函数数量。
    Returns:
        Dict[str, np.ndarray]: 包含每个文件MinHash签名的字典，键为文件路径，值为对应的MinHash签名数组。
    """
    hash_functions = generate_hash_functions(num_hashes)
    all_doc_ids = []
    id_signatures_map = {}
    id_paths_map = {}
    id_n_gram_map = {}
    for i, input_file in tqdm(enumerate(input_files), desc="正在为文本生成MinHash签名"):
        doc_id = i
        all_doc_ids.append(doc_id)
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
            n_gram_set = convert_text_into_n_gram(n, text)
            signature = compute_minhash_signature(n_gram_set, hash_functions)
            id_signatures_map[doc_id] = signature
            id_paths_map[doc_id] = input_file
            id_n_gram_map[doc_id] = tuple(n_gram_set)

    return id_signatures_map, id_paths_map, id_n_gram_map, all_doc_ids

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

def find_true_duplicate_pair(candidate_pairs: set[tuple[int,...]], id_n_gram_map: Dict[int, tuple[tuple[str,...]]], jaccard_threshold: float = 0.8) -> list[tuple[int, int]]:
    true_duplicate_pairs = []
    for pair in candidate_pairs:
        doc_id1, doc_id2 = pair
        ngrams_1_set, ngrams_2_set = set(id_n_gram_map[doc_id1]), set(id_n_gram_map[doc_id2])

        intersection_set = ngrams_1_set & ngrams_2_set
        union_set = ngrams_1_set | ngrams_2_set
        jacccard_similiarity = len(intersection_set) / len(union_set)
        if(jacccard_similiarity >= jaccard_threshold):
            true_duplicate_pairs.append((doc_id1, doc_id2))

    return true_duplicate_pairs

def get_similar_cluster(all_doc_ids: List[Hashable], true_duplicate_pairs: List[Tuple[int, int]]) -> List[Set[int]]:
    uf = UnionFind(all_doc_ids)
    for (doc_id_1, doc_id_2) in true_duplicate_pairs:
        uf.union(doc_id_1, doc_id_2)
    
    all_clusters = uf.get_clusters()
    duplicate_clusters = [cluster for cluster in all_clusters if len(cluster) > 1]

    return cast(List[Set[int]], duplicate_clusters)

def remove_duplicate_ids(all_doc_ids: List[int], duplicate_clusters: List[Set[int]]) -> Set[int]:
    all_doc_id_set = set(all_doc_ids)
    remove_id_set = set()
    for cluster in duplicate_clusters:
        doc_list = sorted(list(cluster))
        remove_id_set.update(doc_list[1:])

    keep_id_set = all_doc_id_set - remove_id_set
    return keep_id_set

def minhash_deduplication(input_files: list[os.PathLike], num_hashes: int, num_bands: int, n: int, output_dir: os.PathLike, jaccard_threshold: float = 0.8):
    id_signatures_map, id_paths_map, id_n_gram_map, all_doc_list = generate_signature_for_texts(input_files, n, num_hashes)
    all_doc_hashable_list = cast(List[Hashable], all_doc_list)
    candidate_pairs = get_lsh_candidate_pairs(num_hashes=num_hashes, num_band=num_bands, id_signatures_dict=id_signatures_map)
    duplicate_pairs = find_true_duplicate_pair(candidate_pairs, id_n_gram_map, jaccard_threshold)
    doc_id_to_keep = remove_duplicate_ids(all_doc_list, get_similar_cluster(all_doc_hashable_list, duplicate_pairs))

    for doc_id in doc_id_to_keep:
        input_file = id_paths_map[doc_id]
        output_base_name = os.path.basename(input_file)
        output_file = os.path.join(output_dir, output_base_name)
        with open(input_file, 'r', encoding='utf-8') as in_file, open(output_file, 'w', encoding='utf-8') as out_file:
            content = in_file.read()
            out_file.write(content)


