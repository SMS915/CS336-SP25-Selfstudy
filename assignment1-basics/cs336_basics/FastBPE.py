import heapq
import json
import regex
import os
import multiprocessing
from typing import Iterable, List, Dict, Iterator, Set, Tuple, BinaryIO
from collections import defaultdict, Counter
from tqdm import tqdm  # 引入 tqdm

# --- 正则与全局常量 ---
BPAT = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
Compiled_Byte_PAT = regex.compile(BPAT)
Compiled_Str_PAT = regex.compile(PAT)

# --- 辅助函数 (保持不变) ---

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
                
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at + len(split_special_token)
                    break
                initial_position += mini_chunk_size
                
    return sorted(set(chunk_boundaries))

def _process_chunk_worker(args) -> Dict[bytes, int]:
    """Worker 进程执行的函数"""
    filename, start, end, special_tokens_bytes = args
    local_freqs = Counter()
    
    if special_tokens_bytes:
        special_pattern = b"|".join(regex.escape(s) for s in special_tokens_bytes)
        compiled_special = regex.compile(special_pattern)
    else:
        compiled_special = None

    try:
        with open(filename, 'rb') as f:
            f.seek(start)
            chunk_data = f.read(end - start)
            
            if compiled_special:
                segments = compiled_special.split(chunk_data)
            else:
                segments = [chunk_data]
            
            for seg_bytes in segments:
                if not seg_bytes: continue
                try:
                    seg_str = seg_bytes.decode('utf-8', errors='replace')
                except UnicodeDecodeError:
                    continue
                tokens_str = Compiled_Str_PAT.findall(seg_str)
                tokens_bytes = [t.encode('utf-8') for t in tokens_str]
                local_freqs.update(tokens_bytes)
                    
    except Exception as e:
        print(f"处理文件{start}-{end}部分时遇到问题: {e}")
        
    return local_freqs


class Word:
    """编码单词在BPE训练中的状态。"""
    def __init__(self, token_ids: List[int], frequency: int, index: int):
        self.id = index
        self.freq = frequency
        self.tokens = token_ids

    def get_pairs(self) -> set[tuple[int, int]]:
        return set(zip(self.tokens[:-1], self.tokens[1:]))
    
    def merge(self, target_pair: tuple[int, int], new_token_id: int) -> tuple[Dict, Dict, Set]:
        if len(self.tokens) < 2:
            return {}, {}, set(zip(self.tokens[:-1], self.tokens[1:]))

        freq_deltas = defaultdict(int)
        generated_tokens = []
        i = 0
        n = len(self.tokens)
        target_left, target_right = target_pair
        last_orig_suffix = None

        while i < n:
            is_merge = (i < n - 1 and self.tokens[i] == target_left and self.tokens[i+1] == target_right)
            
            if is_merge:
                token_to_append = new_token_id
                curr_orig_prefix = target_left
                curr_orig_suffix = target_right
                freq_deltas[target_pair] -= 1
                step = 2
            else:
                current_token = self.tokens[i]
                token_to_append = current_token
                curr_orig_prefix = current_token
                curr_orig_suffix = current_token
                step = 1

            if generated_tokens:
                last_gen_suffix = generated_tokens[-1]
                if (last_gen_suffix != last_orig_suffix) or (token_to_append != curr_orig_prefix):
                    freq_deltas[(last_gen_suffix, token_to_append)] += 1
                    freq_deltas[(last_orig_suffix, curr_orig_prefix)] -= 1

            generated_tokens.append(token_to_append)
            last_orig_suffix = curr_orig_suffix
            i += step

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


# --- BPETokenizer (集成了 Training 逻辑) ---

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
        self.rank_map: Dict[Tuple[int, int], int] = {} 
        self.merge_to_vocab = {}
        self.cache: Dict[bytes, List[int]] = {}
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
            # 确保special tokens被添加到词表中（如果尚未存在）
            # 注意：这里的逻辑假设输入vocab已经包含了训练好的tokens
            # 如果是Train方法生成的实例，vocab通常已经包含了special tokens
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

    # --- Training Interface ---

    @classmethod
    def train(cls, input_path: str | os.PathLike, vocab_size: int, special_tokens: List[str], split_token: bytes = b"<|endoftext|>"):
        """
        训练 BPE 模型并返回一个 BPETokenizer 实例。
        集成 tqdm 显示训练进度。
        """
        
        # 内部类：用于封装训练过程中的复杂状态（倒排索引、频率桶等）
        # 这样不会污染 BPETokenizer 实例的命名空间
        class _TrainerBackend:
            def __init__(self):
                self._words: List[Word] = []
                self._pair_count: Dict[Tuple[int, int], int] = defaultdict(int)
                self._pair_index: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
                self._freq_buckets: Dict[int, set[Tuple[int, int]]] = defaultdict(set)
                self._max_freq = 0
                self._vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
                self._merges: List[Tuple[bytes, bytes]] = []

            def _update_pair_freq(self, pair: Tuple[int, int], delta: int):
                old_freq = self._pair_count.get(pair, 0)
                new_freq = old_freq + delta

                if old_freq > 0:
                    self._freq_buckets[old_freq].discard(pair)
                    if not self._freq_buckets[old_freq]:
                        del self._freq_buckets[old_freq]

                if new_freq > 0:
                    self._pair_count[pair] = new_freq
                    self._freq_buckets[new_freq].add(pair)
                    if new_freq > self._max_freq:
                        self._max_freq = new_freq
                else:
                    if pair in self._pair_count:
                        del self._pair_count[pair]

            def _get_max_freq_pair(self):
                while self._max_freq > 0:
                    if self._max_freq in self._freq_buckets.keys():
                        # Tie-breaking: 频率相同时，选字典序最小的
                        best_pair = max(self._freq_buckets[self._max_freq], 
                                      key=lambda p: (self._vocab[p[0]], self._vocab[p[1]]))
                        return best_pair
                    self._max_freq -= 1
                return None

            def index_word(self, word: Word):
                word_pairs_count = Counter(zip(word.tokens[:-1], word.tokens[1:]))
                for pair, count in word_pairs_count.items():
                    self._pair_index[pair].add(word.id)
                    self._update_pair_freq(pair, word.freq * count)

            def run(self, input_path, vocab_size, special_tokens, split_token):
                print(f"开始训练 BPE (目标词表大小: {vocab_size})")
                special_tokens_bytes = [s.encode('utf-8') for s in special_tokens]
                num_processes = 14
                
                # 1. 计算文件边界
                print("Step 1/3: 寻找分块边界...")
                boundaries = find_chunk_boundaries(input_path, num_processes, split_token)
                tasks = []
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    tasks.append((input_path, start, end, [split_token])) # 这里的split_token用作切分

                # 2. 并行预分词统计
                print("Step 2/3: 并行预分词统计...")
                global_word_freqs = Counter()
                # 使用 tqdm 显示预分词进度 (虽然 chunksize=1 可能跳得快，但有反馈总是好的)
                with multiprocessing.Pool(processes=num_processes) as pool:
                    results = list(tqdm(pool.imap(_process_chunk_worker, tasks, chunksize=1), 
                                      total=len(tasks), desc="Processing chunks"))
                
                for local_freq in results:
                    global_word_freqs.update(local_freq)

                print(f"预分词完成，共有 {len(global_word_freqs)} 个不同的单词。")

                # 初始化 Words
                for word_bytes, freq in tqdm(global_word_freqs.items(), desc="Indexing words"):
                    token_ids = list(word_bytes)
                    word = Word(token_ids, freq, len(self._words))
                    self._words.append(word)
                    self.index_word(word)

                # 3. 合并循环
                print("Step 3/3: 执行 BPE 合并...")
                
                # 初始化进度条
                pbar = tqdm(total=vocab_size, initial=256, desc="Building Vocab")
                
                next_token_id = 256
                
                # 添加特殊 tokens 到词表（如果不参与合并，只是占位）
                for special_token_byte in special_tokens_bytes:
                    if len(self._vocab) >= vocab_size:
                        break
                    # 如果特殊 token 已经在基础字节中，这步可能会覆盖或跳过，通常特殊 token 是额外定义的
                    # 简单起见，这里直接分配 ID
                    self._vocab[next_token_id] = special_token_byte
                    next_token_id += 1
                    pbar.update(1)

                while len(self._vocab) < vocab_size:
                    best_pair = self._get_max_freq_pair()
                    if best_pair is None:
                        print("没有更多可合并的 Pair，提前结束。")
                        break

                    # 记录 Merge
                    self._merges.append((self._vocab[best_pair[0]], self._vocab[best_pair[1]]))
                    new_token_bytes = self._vocab[best_pair[0]] + self._vocab[best_pair[1]]
                    self._vocab[next_token_id] = new_token_bytes

                    affected_word_ids = list(self._pair_index[best_pair])
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
                    pbar.update(1)
                
                pbar.close()
                return self._vocab, self._merges

        # --- End of Internal Class ---

        # 实例化后端并运行
        trainer = _TrainerBackend()
        vocab, merges = trainer.run(input_path, vocab_size, special_tokens, split_token)
        
        # 返回新的 Tokenizer 实例
        return cls(vocab, merges, special_tokens)

    # --- Existing Methods (Save/Load/Encode/Decode) ---

    def save(self, file_prefix: str):
        """保存 vocab.json 和 merges.txt"""
        print(f"正在保存模型到 {file_prefix}_vocab.json 和 {file_prefix}_merges.txt ...")
        byte_encoder = self._bytes_to_unicode()

        vocab_output = {}
        for token_id, token_bytes in self._vocab.items():
            token_str = "".join([byte_encoder[b] for b in token_bytes])
            vocab_output[token_str] = token_id

        vocab_path = f"{file_prefix}_vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_output, f, ensure_ascii=False, indent=2)

        merges_path = f"{file_prefix}_merges.txt"
        with open(merges_path, "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n") 
            for p1_bytes, p2_bytes in self._merges:
                s1 = "".join([byte_encoder[b] for b in p1_bytes])
                s2 = "".join([byte_encoder[b] for b in p2_bytes])
                f.write(f"{s1} {s2}\n")
        print("保存完成。")

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
            parts = line.strip().split()
            if len(parts) != 2: continue
            p1_str, p2_str = parts
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
                if special_id is not None:
                    encoded_ids.append(special_id)
                else:
                    # Fallback if special token somehow not in map
                    encoded_ids.extend(self.encode_segment(segment))
            else:
                encoded_ids.extend(self.encode_segment(segment))

        return encoded_ids
    
    def encode_segment(self, text: str) -> List[int]:
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
        raw_ids = [self.tokens_to_id[bytes([b])] for b in word_bytes]
        n = len(raw_ids)
        if n < 2:
            return raw_ids
        
        ids = list(raw_ids)
        next_pos = list(range(1, n + 1))
        prev_pos = list(range(-1, n - 1))

        pq = []
        for i in range(n - 1):
            pair = (ids[i], ids[i + 1])
            rank = self.rank_map.get(pair)
            if rank is not None:
                heapq.heappush(pq, (rank, i))

        while pq:
            rank, left_idx = heapq.heappop(pq)
            
            if ids[left_idx] == -1: continue

            right_idx = next_pos[left_idx]
            if right_idx >= n: continue
            if ids[right_idx] == -1: continue

            current_pair = (ids[left_idx], ids[right_idx])
            if self.rank_map.get(current_pair) != rank:
                continue

            new_id = self.merge_to_vocab[current_pair]
            ids[left_idx] = new_id
            ids[right_idx] = -1

            next_neighbor = next_pos[right_idx]
            next_pos[left_idx] = next_neighbor
            if next_neighbor < n:
                prev_pos[next_neighbor] = left_idx

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
            token_bytes = self._vocab.get(token_id)
            if token_bytes is None:
                continue # Skip unknown tokens
            decoded_bytes.append(token_bytes)

        return b"".join(decoded_bytes).decode(encoding='utf-8', errors='replace')