import heapq
import json
import regex
import os
import multiprocessing
from typing import Iterable, List, Dict, Iterator, Set, Tuple, BinaryIO
from collections import defaultdict, Counter

BPAT = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
Compiled_Byte_PAT = regex.compile(BPAT)
Compiled_Str_PAT = regex.compile(PAT)

def find_chunk_boundaries(
    filename: str | os.PathLike,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> List[int]:
    """计算文件切分的字节偏移量，确保不切断指定的特殊token"""
    with open(filename, "rb") as file:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size == 0:
            return [0]

        chunk_size = file_size // desired_num_chunks
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size
        
        mini_chunk_size = 8192

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)
            while True:
                mini_chunk = file.read(mini_chunk_size)
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break
                
                # 寻找分割点 (例如 <|endoftext|>)
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    # 边界定在特殊token之后，确保完整包含
                    chunk_boundaries[bi] = initial_position + found_at + len(split_special_token)
                    break
                initial_position += mini_chunk_size
                
    return sorted(set(chunk_boundaries))

def _process_chunk_worker(args) -> Dict[bytes, int]:
    """
    Worker 进程执行的函数：
    1. 读取指定范围的文件
    2. 移除特殊 Token
    3. 运行正则分词
    4. 返回局部词频统计
    """
    filename, start, end, special_tokens_bytes = args
    # local_freqs = defaultdict(int)
    local_freqs = Counter()
    
    # 预编译特殊字符移除正则 (bytes模式)
    if special_tokens_bytes:
        special_pattern = b"|".join(regex.escape(s) for s in special_tokens_bytes)
        compiled_special = regex.compile(special_pattern)
    else:
        compiled_special = None

    try:
        with open(filename, 'rb') as f:
            f.seek(start)
            # 只读取分配给当前进程的块
            chunk_data = f.read(end - start)
            
            # 按照特殊 Token 切分，防止合并跨越特殊 Token
            if compiled_special:
                # regex.split 返回一个列表，其中包含非特殊字符的片段
                segments = compiled_special.split(chunk_data)
            else:
                segments = [chunk_data]
            
            # 对每个片段进行 BPE 预分词正则匹配
            for seg_bytes in segments:
                if not seg_bytes: continue
                
                try:
                    seg_str = seg_bytes.decode('utf-8', errors='replace')
                except UnicodeDecodeError:
                    continue
                tokens_str = Compiled_Str_PAT.findall(seg_str)
                # for token in tokens_str:
                #     token_bytes = token.encode('utf-8')
                #     local_freqs[token_bytes] += 1
                tokens_bytes = [t.encode('utf-8') for t in tokens_str]
                local_freqs.update(tokens_bytes)
                    
    except Exception as e:
        print(f"处理文件{start}-{end}部分时遇到问题: {e}")
        
    return local_freqs


class Word:
    """编码单词在BPE训练中的状态。
    
    Attribute:
        id(int): 单词在词表中的索引
    包括组成的token id,该单词的id和频率"""
    def __init__(self, token_ids: List[int], frequency: int, index: int):
        self.id = index
        self.freq = frequency
        self.tokens = token_ids

    def get_pairs(self) -> set[tuple[int, int]]:
        """获取当前单词中所有相邻的token对"""
        return set(zip(self.tokens[:-1], self.tokens[1:]))
    
    def merge(self, target_pair: tuple[int, int], new_token_id: int) -> tuple[Dict, Dict, Set]:
        """
        在单词序列中执行合并操作，并计算产生的Pair变化

        执行单次线性扫描，将序列中的所有对应pair替换成new_token_id
        同时计算合并导致的断裂的对和生成的对，存入removed 和 add

        Args:
            pair (tuple[int, int]): 需要被合并的token_id对
            new_token_id: 合并后新的token_id

        Returns:
            tuple[Dict, Dict, Set[Tuple[int, int]]]
            包含三个元素的元组
            1. removed (Dict[Tuple, int])
               记录减少的Pair与减少的数量
            2. added (Dict[Tuple, int])
               记录新增的Pair和增加的数量
            3. new_pairs (Set[Tuple[int, int]])
               单词更新后的所有pair集合，用于更新倒排索引
        """
        if len(self.tokens) < 2:
            return {}, {}, set(zip(self.tokens[:-1], self.tokens[1:]))

        # 存储频率变化量: key=pair, value=变化次数delta
        freq_deltas = defaultdict(int)
        
        generated_tokens = []
        i = 0
        n = len(self.tokens)
        target_left, target_right = target_pair
        # last_orig_suffix: 上一个处理完的片段在"原序列"中的最后一个 Token。
        # 用于与"当前片段"的原序列头部进行连接检查。
        last_orig_suffix = None

        while i < n:
            # 检查当前位置是否匹配目标 Pair
            # 只有不越界且左右都匹配时，才判定为合并
            is_merge = (i < n - 1 and self.tokens[i] == target_left and self.tokens[i+1] == target_right)
            
            if is_merge:
                # case1: 合并
                # [target_left, target_right] -> 生成 Token: [new_token_id]
                token_to_append = new_token_id

                # 记录该片段在原序列中的"头"和"尾"，用于后续的边界一致性检查
                curr_orig_prefix = target_left
                curr_orig_suffix = target_right
                
                # Pair 本身被合并掉了，频率 -1
                freq_deltas[target_pair] -= 1
                
                step = 2 # 消耗两个旧token
            else:
                # case2: 不合并
                # [..., t, ...] -> [..., t, ...]
                current_token = self.tokens[i]

                token_to_append = current_token
                curr_orig_prefix = current_token
                curr_orig_suffix = current_token
                step = 1 # 消耗一个旧token

            # 邻居边界检查
            # 处理当前片段与上一个片段连接处的频率更新
            # 若generated_tokens非空，则有左邻居片段，需要检查连接
            if generated_tokens:
                # 取出左侧刚刚生成的 Token
                last_gen_suffix = generated_tokens[-1]

                # 对比 "新序列的连接" 与 "原序列的连接" 是否一致
                # 新连接: (last_gen_suffix, token_to_append)
                # 旧连接: (last_orig_suffix, curr_orig_prefix)
                # 如果不一致，说明因为合并发生了连接变化，需要记录
                if (last_gen_suffix != last_orig_suffix) or (token_to_append != curr_orig_prefix):
                    freq_deltas[(last_gen_suffix, token_to_append)] += 1
                    freq_deltas[(last_orig_suffix, curr_orig_prefix)] -= 1

            # 更新状态
            generated_tokens.append(token_to_append)

            # 当前片段处理完毕，它的"原尾部"变成下一轮循环的"上一个原尾部"
            last_orig_suffix = curr_orig_suffix
            i += step

        # 更新单词内部的token序列表示
        self.tokens = generated_tokens

        removed_counts = {}
        added_counts = {}

        current_pairs_set = set(zip(self.tokens[:-1], self.tokens[1:]))
        for p, delta in freq_deltas.items():
            if delta < 0:
                removed_counts[p] = -delta
            elif delta > 0:
                added_counts[p] = delta
        
        return removed_counts, added_counts, current_pairs_set


def pretokenizer(text: bytes):
    for s in Compiled_Byte_PAT.finditer(text):
        yield s.group(0)

def single_process_pretokenize_iter(path: os.PathLike, special_tokens: List[str], split_token: bytes = b'<|endoftext|>'):
    file = open(path, 'rb')
    chunk = file.read()
    token_to_remove = b"|".join(regex.escape(s.encode()) for s in special_tokens)
    chunks = regex.split(token_to_remove, chunk)
    for chunk in chunks:
        for m in pretokenizer(chunk):
            yield m
    file.close()

def get_pretoken_freq(pretokenize_iter: Iterator[bytes]) -> Dict[tuple[bytes], int]:
    """获取预词元频率"""
    freq = {}
    for p in pretokenize_iter:
        t = tuple(bytes([b]) for b in p)
        freq[p] = freq.get(t, 0) + 1
    return freq

class BPETrainer:
    def __init__(self):
        self._words: List[Word] = []
        self._pair_count: Dict[Tuple[int, int], int] = defaultdict(int) # pair计数
        self._pair_index: Dict[Tuple[int, int], Set[int]] = defaultdict(set) # 倒排索引，记录pair和所有包含其的Word id集合
        self._freq_buckets: Dict[int, set[Tuple[int, int]]] = defaultdict(set) # 频率桶 key: 频率 values: pairs
        self._max_freq = 0
        self._vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self._merges: List[Tuple[bytes, bytes]] = []
    
    def _update_pair_freq(self, pair: Tuple[int, int], delta: int):
        """更新频率桶频率"""
        old_freq = self._pair_count.get(pair, 0)
        new_freq = old_freq + delta

        if old_freq > 0:
            self._freq_buckets[old_freq].discard(pair)
            # 处理空桶
            if not self._freq_buckets[old_freq]:
                del self._freq_buckets[old_freq]

        if new_freq > 0:
            self._pair_count[pair] = new_freq
            self._freq_buckets[new_freq].add(pair)
            # 更新最大频率
            if new_freq > self._max_freq:
                self._max_freq = new_freq
        else:
            if pair in self._pair_count:
                del self._pair_count[pair]
                

    def _get_max_freq_pair(self):
        while self._max_freq > 0:
            if self._max_freq in self._freq_buckets.keys():
                best_pair = max(self._freq_buckets[self._max_freq], key = lambda p: (self._vocab[p[0]], self._vocab[p[1]]))
                break
            self._max_freq -= 1

        return best_pair
    
    def index_word(self, word: Word):
        word_pairs_count = Counter(zip(word.tokens[:-1], word.tokens[1:]))
        for pair, count in word_pairs_count.items():
            self._pair_index[pair].add(word.id)
            self._update_pair_freq(pair, word.freq * count)


    def train(self, input_path: str | os.PathLike , vocab_size: int, special_tokens: List[str], split_token: bytes = b"<|endoftext|>"):
        print("开始训练BPE")
        special_tokens_bytes = [s.encode('utf-8') for s in special_tokens]
        num_processes = max(1, os.cpu_count() - 1) # type: ignore
        specitial_token_list = [split_token]
        
        print("寻找分块边界")
        boundaries = find_chunk_boundaries(input_path, num_processes, split_token)
        tasks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            tasks.append((input_path, start, end, specitial_token_list))

        global_word_freqs = Counter()

        print("开始并行预分词")
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(_process_chunk_worker, tasks, chunksize=1)

        for local_freq in results:
            global_word_freqs.update(local_freq)

        print(f"共有{len(global_word_freqs)}个不同的单词")

        for word_bytes, freq in global_word_freqs.items():
            token_ids = list(word_bytes)
            word = Word(token_ids, freq, len(self._words))
            self._words.append(word)
            self.index_word(word)

        print("开始合并循环")
        next_token_id = 256
        for special_token_byte in special_tokens_bytes:
            if len(self._vocab) > vocab_size:
                break
            self._vocab[next_token_id] = special_token_byte
            next_token_id += 1

        while len(self._vocab) < vocab_size:
            best_pair = self._get_max_freq_pair()
            self._merges.append((self._vocab[best_pair[0]], self._vocab[best_pair[1]]))
            new_token_bytes = self._vocab[best_pair[0]] + self._vocab[best_pair[1]]
            self._vocab[next_token_id] = new_token_bytes

            affected_word_ids = list(self._pair_index[best_pair])

            # 清除该pair的索引
            del self._pair_index[best_pair]

            for word_id in affected_word_ids:
                word = self._words[word_id]
                removed, added, new_pairs = word.merge(best_pair, next_token_id)

                for p, count in removed.items():
                    decrease = -count
                    self._update_pair_freq(p, decrease * word.freq)
                    if p not in new_pairs:
                        if p in self._pair_index:
                            self._pair_index[p].discard(word_id)
                            if not self._pair_index[p]:
                                del self._pair_index[p]

                for p, count in added.items():
                    increase = count
                    self._update_pair_freq(p, increase * word.freq)
                    self._pair_index[p].add(word_id)

            next_token_id += 1 
        
        return self._vocab, self._merges
    
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

    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens = None):
        self._vocab = vocab
        self._merges = merges
        self.tokens_to_id = {v: k for k, v in self._vocab.items()}
        self.rank_map: Dict[Tuple[int, int], int] = {} # merge的顺序
        self.merge_to_vocab = {}
        self.cache: Dict[bytes, List[int]] = {} # 单词bytes到最优token组成的map
        self.cache_limit = 1_000_000
        self.pat = Compiled_Str_PAT
        for rank, (p1, p2) in enumerate(self._merges):
            id1 = self.tokens_to_id[p1]
            id2 = self.tokens_to_id[p2]
            merged = p1 + p2
            pair = (id1, id2)
            if merged in self.tokens_to_id:
                self.rank_map[pair] = rank
                self.merge_to_vocab[pair] = self.tokens_to_id[merged]

        self._b_special_tokens = []
        if special_tokens is not None:
            next_token_id = max(self._vocab.keys()) + 1 if self._vocab else 0
            for t in special_tokens:
                b = t.encode('utf-8')
                self._b_special_tokens.append(b)
                if b not in self.tokens_to_id:
                    self._vocab[next_token_id] = b
                    self.tokens_to_id[b] = next_token_id
                    next_token_id += 1

        self._special_tokens_set = set(special_tokens) if special_tokens else set()
        self._special_pattern = None
        if self._special_tokens_set:
            sorted_special = sorted(self._special_tokens_set, key=len, reverse=True)
            escaped_specials = [regex.escape(s) for s in sorted_special]
            self._special_pattern = regex.compile('(' + '|'.join(escaped_specials) + ')')

        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens = None, skip_first_row = False):
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)

        bytes_to_id = {}
        for token_str, token_id in vocab_json.items():
            token_bytes = bytearray([cls._UNICODE_TO_BYTES_MAP[c] for c in token_str])
            bytes_to_id[bytes(token_bytes)] = token_id

        vocab = {v: k for k, v in bytes_to_id.items()}

        with open(merges_filepath, 'r', encoding='utf-8') as f:
            if skip_first_row:
                f.readline()
            lines = f.readlines()
            
        merges = []
        for line in lines:
            p1_str, p2_str = line.strip().split()
            p1 = bytes(bytearray([cls._UNICODE_TO_BYTES_MAP[c] for c in p1_str]))
            p2 = bytes(bytearray([cls._UNICODE_TO_BYTES_MAP[c] for c in p2_str]))
            merges.append((p1, p2))

        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> List[int]:
        encoded_ids = []
        if not self._special_pattern:
            segments = [text]
        else:
            segments = self._special_pattern.split(text)

        for segment in segments:
            if not segment:
                continue
                
            if segment in self._special_tokens_set:
                special_id = self.tokens_to_id.get(segment.encode('utf-8'))
                encoded_ids.append(special_id)
            else:
                encoded_ids.extend(self.encode_segment(segment))

        return encoded_ids
    
    def encode_segment(self, text: str) -> List[int]:
        # b_text = text.encode('utf-8')
        # text = "".join([self.byte_encoder[b] for b in b_text])
        ids = []
        for word_match in self.pat.finditer(text):
            word = word_match.group(0)
            word_bytes = word.encode('utf-8')

            if word_bytes in self.cache:
                ids.extend(self.cache[word_bytes])
            else:
                word_ids = self.encode_word(word_bytes)
                if len(self.cache) > self.cache_limit:
                    self.cache.clear()
                self.cache[word_bytes] = word_ids
                ids.extend(word_ids)

        return ids
    
    def encode_word(self, word_bytes: bytes):
        """
        使用最小堆 + 双向链表优化的BPE 编码算法。
        """
        raw_ids = [self.tokens_to_id[bytes([b])] for b in word_bytes]

        n = len(raw_ids)

        if n < 2:
            return raw_ids
        
        ids = list(raw_ids)
        # i 位置的后继和前驱索引,模拟链表，初始化
        next_pos = list(range(1, n + 1))
        prev_pos = list(range(-1, n - 1))

        pq = []
        for i in range(n - 1):
            pair = (ids[i], ids[i + 1])
            rank = self.rank_map.get(pair)
            if rank is not None:
                heapq.heappush(pq, (rank, i))

        while pq:
            rank, left_idx = heapq.heappop(pq) # 取出rank最小的左节点

            # lazy check
            # 越界，无法合并
            if ids[left_idx] == -1:
                continue

            right_idx = next_pos[left_idx]
            if right_idx >= n:
                continue
            if ids[right_idx] == -1:
                continue

            current_pair = (ids[left_idx], ids[right_idx])
            # 过期pair，直接移除
            if self.rank_map.get(current_pair) != rank:
                continue

            new_id = self.merge_to_vocab[current_pair]
            ids[left_idx] = new_id
            ids[right_idx] = -1 # 标记无效

            next_neighbor = next_pos[right_idx]
            next_pos[left_idx] = next_neighbor
            if next_neighbor < n:
                prev_pos[next_neighbor] = left_idx

            # 局部检查新token是否可构成新的pair
            prev_neighbor = prev_pos[left_idx]
            if prev_neighbor != -1:
                new_pair_left = (ids[prev_neighbor], new_id)
                left_pair_rank = self.rank_map.get(new_pair_left)
                if left_pair_rank is not None:
                    heapq.heappush(pq, (left_pair_rank, prev_neighbor))

            if next_neighbor < n:
                new_pair_right = (new_id, ids[next_neighbor])
                right_pair_rank = self.rank_map.get(new_pair_right)
                if right_pair_rank is not None:
                    heapq.heappush(pq, (right_pair_rank, left_idx))

        results = []
        curr_idx = 0
        while curr_idx < n:
            if ids[curr_idx] != -1:
                results.append(ids[curr_idx])
            curr_idx = next_pos[curr_idx]

        return results
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            yield from self.encode(line)

    def decode(self, token_ids: List[int]) -> str:
        decoded_bytes = []
        for token_id in token_ids:
            token_bytes = self._vocab[token_id]
            if token_bytes == None:
                decoded_bytes.append(b"")
            else:
                decoded_bytes.append(token_bytes)

        return b"".join(decoded_bytes).decode(encoding='utf-8', errors='replace')

        



                            




            
                


            

            









