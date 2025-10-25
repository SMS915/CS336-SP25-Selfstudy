import json
import pickle
from collections import defaultdict

import regex
from array import array
from typing import Tuple, List, Dict, Iterable, Iterator, Set

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPETokenizer:
    @staticmethod
    def _bytes_to_unicode():
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
            range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(2 ** 8):
            if b not in bs:
                bs.append(b)
                cs.append(2 ** 8 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    _BYTE_TO_UNICODE_MAP = _bytes_to_unicode()
    _UNICODE_TO_BYTES_MAP = {v: k for k, v in _BYTE_TO_UNICODE_MAP.items()}

    def __init__(self, vocab, merges, special_tokens = None):
        self._vocab : Dict[int, bytes] = vocab
        self._merges : List[tuple[bytes, bytes]] = merges
        self._byte_special_tokens: List[bytes] = []
        self._tokens_to_id = {token : id for id, token in vocab.items()}
        self._next_token_id = max(vocab.keys()) + 1 if vocab else 0

        if special_tokens:
            for st_str in special_tokens:
                byte_str = st_str.encode('utf-8')
                self._byte_special_tokens.append(byte_str)
                if byte_str not in self._tokens_to_id:
                    self._vocab[self._next_token_id] = byte_str
                    self._tokens_to_id[byte_str] = self._next_token_id
                    self._next_token_id += 1

        # 为了实现高效的merge规则应用，创建字典以快速查找
        self._int_merges_priority_map : Dict[Tuple[int, int], int] = {
            (self._tokens_to_id[p1], self._tokens_to_id[p2]) : i for i, (p1, p2) in enumerate(self._merges)
        }

        self._merges_to_new_id_map : Dict[Tuple[int, int], int] = {
            (self._tokens_to_id[p1], self._tokens_to_id[p2]) : self._tokens_to_id[p1 + p2] for p1, p2 in self._merges
        }

        #首先按照special_tokens进行初次切分
        if self._byte_special_tokens:
            sorted_bytes = sorted([s.decode('utf-8') for s in self._byte_special_tokens], key=len, reverse=True)
            self._first_chunking_pattern = regex.compile('(' + '|'.join(regex.escape(s) for s in sorted_bytes) + ')')
        else:
            self._first_chunking_pattern = None


    def pretokenize(self, text: str) -> List[str]:
        words = regex.findall(PAT, text)
        return words


    def encode_segment(self, text_bytes : bytes) -> List[int]:
        if not text_bytes:
            return []
        try:
            token_ids = array('H', [self._tokens_to_id[bytes([b])] for b in text_bytes])
        except Exception as e:
            raise ValueError(f"byte {e.args[0]} not in initial vocabulary")

        while True:
            if len(token_ids) < 2:
                break
            best_rank = float('inf')
            best_pos = -1

            for i in range(len(token_ids) - 1):
                pair = (token_ids[i], token_ids[i + 1])
                cur_rank = self._int_merges_priority_map.get(pair)

                if cur_rank is not None and cur_rank < best_rank:
                    best_rank = cur_rank
                    best_pos = i

            if best_pos == -1:
                break

            best_pair_to_merge = (token_ids[best_pos], token_ids[best_pos + 1])
            new_id = self._merges_to_new_id_map.get(best_pair_to_merge)

            token_ids[best_pos:best_pos+2] = array('H', [new_id])

        return list(token_ids)


    def encode(self, text : str) -> List[int]:
        encoded_ids : List[int] = []
        if self._first_chunking_pattern:
            segments = self._first_chunking_pattern.split(text)
        else:
            segments = [text]

        for i, segment in enumerate(segments):
            if not segment:
                continue
            segment_bytes = segment.encode('utf-8')
            if segment_bytes in self._byte_special_tokens:
                encoded_ids.append(self._tokens_to_id[segment_bytes])
            else:
                pretokenized_chunk = self.pretokenize(segment)
                for text in pretokenized_chunk:
                    text_bytes = text.encode('utf-8')
                    encoded_list = self.encode_segment(text_bytes)
                    encoded_ids.extend(encoded_list)

        return encoded_ids


    def encode_iterable(self, iterable : Iterable[str]) -> Iterator[int]:
        for line in iterable:
            yield from self.encode(line)


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # 1. 加载 vocab.json
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)  # 这是 str -> int

        # 将 str token 转换回 bytes
        bytes_to_id = {}
        for token_str, token_id in vocab_json.items():
            token_bytes = bytearray([cls._UNICODE_TO_BYTES_MAP[c] for c in token_str])
            bytes_to_id[bytes(token_bytes)] = token_id

        # 翻转得到 vocab: int -> bytes
        vocab = {v: k for k, v in bytes_to_id.items()}

        # 2. 加载 merges.txt
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()  # 跳过第一行

        merges = []
        for line in lines:
            p1_str, p2_str = line.strip().split()
            p1 = bytes(bytearray([cls._UNICODE_TO_BYTES_MAP[c] for c in p1_str]))
            p2 = bytes(bytearray([cls._UNICODE_TO_BYTES_MAP[c] for c in p2_str]))
            merges.append((p1, p2))

        return cls(vocab, merges, special_tokens)

    def decode(self, token_ids : List[int]):
        decoded_bytes = []
        for token_id in token_ids:
            token_bytes = self._vocab[token_id]
            if token_bytes is None:
                decoded_bytes.append(b"")
            else:
                decoded_bytes.append(token_bytes)
        return b''.join(decoded_bytes).decode('utf-8', errors = 'replace')


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



