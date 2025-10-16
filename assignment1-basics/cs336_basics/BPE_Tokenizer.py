import os
from collections import defaultdict
from typing import Tuple, Set, List, Dict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens = None):
        self._vocab : Dict[int, bytes] = vocab
        self._merges : List[tuple[bytes, bytes]] = merges
        self._byte_special_tokens: List[bytes] = []
        self._tokens_to_id = {token: id for id, token in vocab.items()}
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
        self._merges_priority_map : Dict[Tuple[bytes, bytes], int] = {
            merge_pair : i for i, merge_pair in enumerate(self._merges)
        }



