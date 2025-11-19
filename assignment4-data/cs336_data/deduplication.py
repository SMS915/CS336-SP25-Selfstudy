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
from typing import List, Dict, Tuple, Set, List, Callable, Hashable, cast
from itertools import combinations
from collections import defaultdict
random_seed = 42
random.seed(random_seed)

def normalize_text_for_duplication(text: str) -> str:
    """
    对文本进行全面的标准化，为去重做准备。
    遵循作业要求：小写、去标点、标准化空格、去重音、NFD范式。

    使用NFD是为了将Unicode字符分解为基础字符和组合标记，从而使得有多种表示的字符可以被一致处理
    Args:
        text (str): 输入的带标准化的文本
    Returns:
        str: 标准化后的文本
    """
    text = unicodedata.normalize('NFD', text.lower()) # 应用NFD Unicode标准化，并将文本转为小写

    text = ''.join(c for c in text if not unicodedata.combining(c)) # 移除重音符号 (组合标记)

    text = regex.sub(r'[^\w\s]', ' ', text) # 匹配并移除任何不是字母、数字、下划线或空白字符的字符，即标点符号

    text = regex.sub(r'\s+', ' ', text).strip() # 标准化空白字符
    
    return text

def exact_line_deduplication(input_files: List[os.PathLike], output_directory: os.PathLike) -> None:
    """
    对输入的文本文件进行精确行去重并写入到指定的输出文件夹中。
    采用两阶段行为
    第一阶段，遍历所有输入文件，计算每个处理后的行在整个文件集合中出现的总频率
    第二阶段，再次遍历所有输入文件，只保留那些在第一阶段统计为频率为1的行，并将结果写入到输出目录对应的文件
    目的是过滤那些语料库中可能出现的结构化，样板化文字，比如 导航栏，页脚等

    Args:
        input_files (List[os.PathLike]): 一个包含所有输入文档路径的列表
        output_directory (os.PathLike):  用于存放输出文件的文件夹路径
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
    为了提高性能，采用循环交换的优化策略。它只遍历一次n-gram集合，
    对于每个n-gram，一次性计算出其在所有哈希函数下的哈希值，然后通过
    NumPy的向量化操作来高效更新整个签名向量
    避免了num_hashes * len(n_grams)次嵌套循环和重复的字符串操作。
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
    生成并采用不同的随机种子，构成一组基于mmh3的高性能哈希函数，用于MinHash签名计算。

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
        Tuple[...]: 一个元组，包含四个核心的数据结构：
        - id_signatures_map: 将文档ID映射到其MinHash签名向量(np.ndarray)。
        - id_paths_map: 一个反向查找表，将文档ID映射回其原始文件路径。
        - id_n_gram_map: 将文档ID映射到其原始的n-gram集合（以元组形式存储，
                         以便在字典中作为值）。这是后续进行精确Jaccard验证所必需的。
        - all_doc_ids: 一个包含所有已处理文档ID的列表。
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
            id_n_gram_map[doc_id] = tuple(n_gram_set) # 转换为tuple以便存储

    return id_signatures_map, id_paths_map, id_n_gram_map, all_doc_ids

def get_lsh_candidate_pairs(num_band: int, num_hashes: int, id_signatures_dict: dict[int, np.ndarray]) -> set[tuple[int, ...]]:
    """
    对MinHash签名执行局部敏感哈希（LSH），以高效地找出可能重复的候选对。

    LSH的核心思想是设计一种特殊的哈希方案，使得原本相似的项有很高的概率
    被哈希到同一个“桶”中。本函数采用“分段哈希”（Bands and Rows）策略：
    1. 将每个MinHash签名向量切分成 `num_band` 个“段”（band）。
    2. 对每个段独立地计算哈希值。
    3. 如果两个文档在【至少一个】段上哈希值相同（即发生了“碰撞”），
    它们就被认为是一对候选对。

    这个过程将一个O(N^2)的全局比较问题，显著地降维为一个只需要对少数
    桶内元素进行比较的问题。
    当分段数越多，段数b越大，一个band内就越容易匹配，碰撞概率更高，能找到更多的候选对，但是后续精确需要判断更多的候选对，开销更大
    当分段数越少，宽度r越大，成为候选对的门槛就越高，碰撞概率更低，但找到的候选对较少，从而降低后续精确判断开销，但是有漏网之鱼风险

    -----------------------------------------------------
    num_band (b) 和 rows (r) 的选择决定了LSH算法的敏感性，这是一个
    精确率(Precision)与召回率(Recall)之间的权衡。

    设两个文档的真实Jaccard相似度为 s。
    - 一对文档在一个band（包含r行）中完全匹配的概率为: p_band = s^r
    - 一对文档在所有b个band中都不匹配的概率为: (1 - s^r)^b
    - 因此，一对文档在至少一个band中匹配（即成为候选对）的概率为:
      P(candidate) = 1 - (1 - s^r)^b

    这个概率函数近似于一个S形曲线，其拐点的位置和陡峭程度由b和r控制。
    1.  b 增大, r 减小 (例如 b=50, r=2): "广撒网"策略
        - p_band (s^r) 相对较高，P(candidate) 整体偏高。
        - 效果: 提高召回率（不易错过真正的重复项），但降低精确率
          （会引入大量仅中低度相似的候选对），导致后续精确验证阶段的
          计算开销增大。

    2.  b 减小, r 增大 (例如 b=5, r=20): "精准打击"策略
        - p_band (s^r) 极低，P(candidate) 整体偏低，但对高相似度的s增长更敏感。
        - 效果: 提高精确率（候选对大概率是真的重复项），但降低召回率
          （可能错过部分相似度在阈值边缘的重复项），从而减少后续验证
          阶段的计算开销。

    通常选择b和r，使得S形曲线的拐点大致位于感兴趣的Jaccard相似度阈值附近。
    在克制候选对的数量同时，从而直接决定了精确Jaccard验证这个最昂贵阶段的计算开销。
    """
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
    """
    去重管道的精确验证阶段，对LSH生成的候选对进行精确的Jaccard相似度验证。

    接收由LSH算法筛选出的、可能重复的文档对（候选对），并对每一对计算其真实的、基于n-gram集合的
    Jaccard相似度。只有当相似度达到或超过指定的阈值时，该对才被确认为“真实重复对”。

    这个步骤是计算密集型的，因为它涉及到大集合的交集和并集运算，但其处理
    的数据量远小于全局的两两比较。

    Args:
        candidate_pairs (set[tuple[int,...]]): LSH生成的候选文档ID对的集合。
        id_n_gram_map (Dict[int, tuple[tuple[str,...]]]): 将文档ID映射到其原始n-gram
                                                          集合（以元组形式存储）的字典。
        jaccard_threshold (float, optional): 判断两个文档是否为重复的Jaccard相似度
                                             阈值。默认为 0.8。

    Returns:
        list[tuple[int, int]]: 一个列表，包含所有通过了Jaccard相似度验证的
                               “真实重复对”。
    """
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
    """
    去重管道聚类阶段，根据真实重复对的列表，使用并查集算法将文档聚类成簇。

    此函数是去重管道的“聚类”阶段。它接收一个代表图中“边”的重复对列表，
    并高效地找出图中所有的连通分量，每一个连通分量即为一个重复簇。

    例如，如果输入是 `[(1, 2), (2, 3), (4, 5)]`，此函数将识别出
    `{1, 2, 3}` 和 `{4, 5}` 这两个独立的重复簇。

    Args:
        all_doc_ids (List[Hashable]): 包含所有文档ID的列表，用于初始化并查集。
        true_duplicate_pairs (List[Tuple[int, int]]): 经过Jaccard验证的真实重复对列表。

    Returns:
        List[Set[int]]: 一个列表，其中每个元素是一个代表重复簇的集合（Set），
                        该集合包含了簇内所有文档的ID。只返回大小大于1的簇。
    """
    uf = UnionFind(all_doc_ids)
    for (doc_id_1, doc_id_2) in true_duplicate_pairs:
        uf.union(doc_id_1, doc_id_2)
    
    all_clusters = uf.get_clusters()
    duplicate_clusters = [cluster for cluster in all_clusters if len(cluster) > 1]

    return cast(List[Set[int]], duplicate_clusters)

def remove_duplicate_ids(all_doc_ids: List[int], duplicate_clusters: List[Set[int]]) -> Set[int]:
    """
    根据聚类结果，确定需要保留的唯一文档ID集合。

    此函数是去重管道的“筛选”阶段。它遍历每个重复簇，并为每个簇选择
    一个“幸存者”文档予以保留，而将簇中的所有其他文档标记为待删除。

    幸存者的选择策略是确定性的：在每个簇中，对所有文档ID进行排序，
    并始终保留ID最小（或最大，取决于排序方式）的那个。这确保了去重
    过程的可复现性。

    Args:
        all_doc_ids (List[int]): 包含所有原始文档ID的列表。
        duplicate_clusters (List[Set[int]]): `get_similar_cluster`函数生成的
                                              重复簇列表。

    Returns:
        Set[int]: 一个包含所有应被【保留】的文档ID的集合。
    """
    all_doc_id_set = set(all_doc_ids)
    remove_id_set = set()
    for cluster in duplicate_clusters:
        doc_list = sorted(list(cluster))
        remove_id_set.update(doc_list[1:])

    keep_id_set = all_doc_id_set - remove_id_set
    return keep_id_set

def minhash_deduplication(input_files: list[os.PathLike], num_hashes: int, num_bands: int, n: int, output_dir: os.PathLike, jaccard_threshold: float = 0.8):
    """
    对一组输入的文本文件执行端到端的近似去重流程，并输出唯一的文件。

    本函数是整个模糊去重管道的总控制器。它整合了MinHash和LSH（局部敏感哈希）
    算法，旨在高效地识别并移除内容上高度相似（而非完全相同）的文档。

    该流程遵循以下核心步骤：
    1.  【签名生成】(Signaturing): 遍历所有输入文件，将每个文件转换为n-gram集合，
        并为其计算一个紧凑的数字“指纹”——MinHash签名。同时，建立必要的ID映射
        以便后续处理。
        
    2.  【候选发现】(Bucketing): 使用LSH算法，将具有相似MinHash签名的文档ID高效地
        分入同一个“桶”中。只有在同一个桶中相遇的文档，才会被认为是可能重复的
        “候选对”。这一步是避免全局O(N^2)比较的关键。

    3.  【精确验证】(Verification): 对所有候选对，通过计算它们原始n-gram集合的
        真实Jaccard相似度，来进行精确的最终确认。
        
    4.  【聚类】(Clustering): 根据通过验证的“真实重复对”，使用并查集（Union-Find）
        数据结构，将所有具有传递性重复关系的文档聚合为“重复簇”。

    5.  【筛选与输出】(Filtering & Output): 遍历所有识别出的重复簇，为每个簇
        确定一个唯一的“幸存者”文档并予以保留。最后，将所有被确定为需要保留的
        原始文件内容，复制到指定的输出目录中。

    Args:
        input_files (list[os.PathLike]): 需要进行去重处理的输入文本文件的路径列表。
        num_hashes (int): 用于生成MinHash签名的哈希函数数量。
        num_bands (int): LSH阶段中，将MinHash签名分割成的“段”（band）的数量。
        n (int): 用于将文本分割为n-gram时，n的值。
        output_dir (os.PathLike): 用于存放去重后唯一文件的输出目录的路径。
        jaccard_threshold (float, optional): 判断两个文档是否为真实重复的Jaccard
                                             相似度阈值。默认为 0.8。

    Returns:
        None: 此函数没有返回值，其结果是直接在输出目录中创建去重后的文件。
    """
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


