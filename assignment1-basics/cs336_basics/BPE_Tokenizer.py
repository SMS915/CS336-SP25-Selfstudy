import json
import os
from collections import defaultdict
from array import array
from typing import Tuple, Set, List, Dict, Iterable, Iterator

import regex
from sympy.simplify.hyperexpand import try_lerchphi

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPETokenizer:
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
                pretokenized_text = self.pretokenize(segment)
                for text in pretokenized_text:
                    text_bytes = text.encode('utf-8')
                    encoded_list = self.encode_segment(text_bytes)
                    encoded_ids.extend(encoded_list)

        return encoded_ids


    def encode_iterable(self, iterable : Iterable[str]) -> Iterator[int]:
        for line in iterable:
            yield from self.encode(line)

    @classmethod
    def from_files(self, cls, vocab_filepath : str | os.PathLike, merges_filepath : str | os.PathLike, special_tokens = None):
        vocab: Dict[int, bytes] = {}
        with open(vocab_filepath, encoding="utf-8") as f:
            vocab_json = json.load(f)
            for tokens_str, token_id in vocab_json.items():
                str_bytes = b''
                for char_repr in tokens_str:
                    if char_repr in cls._UNICODE_TO_BYTES_MAP:
                        str_bytes += cls._UNICODE_TO_BYTES_MAP[char_repr]
                vocab[token_id] = str_bytes

        merges: List[Tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    tokens = line.split(' ')
                    if len(tokens) != 2:
                        raise ValueError(f"Invalid line {line}")

                    token1_str, token2_str = tokens[0], tokens[1]

                    actual_byte1 = b''
                    for char_repr in token1_str:
                        if char_repr in cls._UNICODE_TO_BYTES_MAP:
                            actual_byte1 = cls._UNICODE_TO_BYTES_MAP[char_repr]

                    actual_byte2 = b''
                    for char_repr in token2_str:
                        if char_repr in cls._UNICODE_TO_BYTES_MAP:
                            actual_byte2 = cls._UNICODE_TO_BYTES_MAP[char_repr]

                    merges.append((actual_byte1, actual_byte2))
                except ValueError as e:
                    print(f"Invalid merges line {line} - {e}")
                    continue

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



