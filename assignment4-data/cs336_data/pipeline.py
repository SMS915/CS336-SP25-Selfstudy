# cs336_data/pipeline.py

import argparse
import multiprocessing
import os
import glob
import yaml
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional, Tuple, List, Any
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from fastwarc.warc import ArchiveIterator, WarcRecordType
from .extraction import extract_text
from .filter import (identify_language, 
                     gopher_quality_filter, 
                     classify_nsfw, 
                     classify_toxic_speech)

from .deduplication import minhash_deduplication, exact_line_deduplication
from .quality_classifier import QualityClassifier


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
    

_worker_context = {}

def worker_initializer(config):
    """
    这个函数会在每个子进程启动时只执行一次。
    用于加载模型并存入全局变量。
    """
    print(f"Worker {os.getpid()} initializing models...")
    
    # 1. 加载 QualityClassifier
    model_path = config['paths']['models']['quality_classifier']
    label_map = config['paths']['models']['label_mapping']
    _worker_context['quality_classifier'] = QualityClassifier(model_path, label_map)

def process_single_wet_file(wet_file_path: str | os.PathLike,
                            output_dir: str | os.PathLike,
                            config: Dict[str, Any]
                            ) -> Tuple[List[str | os.PathLike], Dict[str, int]]:
    
    stats = defaultdict(int)
    model_path = config['paths']['models']['quality_classifier']
    label_map = config['paths']['models']['label_mapping']
    quality_classifier = _worker_context['quality_classifier']

    web_base_name = os.path.basename(wet_file_path).replace('.warc.wet.gz', '')
    output_subdir = os.path.join(output_dir, web_base_name)
    os.makedirs(output_subdir, exist_ok=True)
    output_paths = []

    min_len = config['filter']['min_doc_length']
    lang_target = config['filter']['language']['target']
    lang_min_score = config['filter']['language']['min_score']
    nsfw_thresh = config['filter']['safety']['nsfw_threshold']
    toxic_thresh = config['filter']['safety']['toxic_threshold']

    with open(wet_file_path, 'rb') as f_in:
        for i, record in enumerate(ArchiveIterator(f_in)):
            if record.record_type != WarcRecordType.conversion :
                stats['other_record_type'] += 1
                continue
            
            stats['total'] += 1
            text = record.reader.read().decode('utf-8', errors='ignore')

            # 长度预检，高效过滤
            if not text or text.isspace() or len(text) < min_len:
                stats['short_count'] += 1
                continue
            
            lang_code, lang_score = identify_language(text)
            if lang_code != lang_target or lang_score <= lang_min_score:
                stats['lang_failed'] += 1
                continue
                
            nsfw_label, nsfw_score = classify_nsfw(text)
            transformed_nsfw_score = nsfw_score if nsfw_label == 'nsfw' else (1 - nsfw_score)
            if transformed_nsfw_score > nsfw_thresh:
                stats['nsfw_failed'] += 1
                continue

            toxic_label, toxic_score = classify_toxic_speech(text)
            transformed_toxic_score = toxic_score if toxic_label == 'toxic' else (1 - toxic_score)
            if transformed_toxic_score > toxic_thresh:
                stats['toxic_failed'] += 1
                continue

            if not gopher_quality_filter(text):
                stats['gopher_failed'] += 1
                continue

            quality_label, _ = quality_classifier.predict(text)
            if quality_label == 'cc':
                stats['quality_failed'] += 1
                continue

            stats['kept'] += 1
            doc_id = f"doc_{i}.txt"
            doc_output_path = os.path.join(output_subdir, doc_id)
            output_paths.append(doc_output_path)
            with open(doc_output_path, 'w', encoding='utf-8') as f_doc:
                f_doc.write(text)

    return output_paths, stats

def filter_wet_files(config: Dict[str, Any]) -> Tuple[List[os.PathLike], Dict[str, int]]:
    """
    使用多进程并行处理输入目录中的所有WET文件。

    Args:
        input_dir (str): 包含 .warc.wet.gz 文件的输入目录。
        output_dir (str): 用于存放过滤后文档的根输出目录。
        quality_classifier_path (str): 训练好的质量分类器模型路径。
        max_workers (Optional[int]): 使用的并行进程数。如果为 None，则使用所有可用CPU核心。
        max_files (Optional[int]): 要处理的最大文件数量，用于快速测试。
    """
    input_dir = config['paths']['input_dir']
    # 动态生成子目录，防止不同配置覆盖
    output_dir = os.path.join(config['paths']['base_output_dir'], 'filtered')
    
    max_workers = config['processing']['max_workers']
    max_files = config['processing']['max_files_limit']

    wet_files = sorted(glob.glob(f"{input_dir}/*.warc.wet.gz"))
    if not wet_files:
        print(f"错误! 在{input_dir}中未找到任何.warc.wet.gz文件")
        return [], defaultdict(int)
    
    if max_files:
        wet_files = wet_files[:max_files]

    print(f"共找到{len(wet_files)}个.warc.wet.gz文件")
    os.makedirs(output_dir, exist_ok=True)

    all_output_paths = []
    total_stats = defaultdict(int)

    with ProcessPoolExecutor(max_workers=max_workers,
                             initializer=worker_initializer,
                             initargs=(config,)) as executor:
        print(f"启动ProcessPoolExecutor，最大工作进程数: {max_workers}")

        process_func = partial(process_single_wet_file, output_dir=output_dir, config=config)

        results = tqdm(executor.map(process_func, wet_files),
                       total=len(wet_files),
                       desc="并行处理wet文件")
        
        for output_paths, stats in results:
            all_output_paths.extend(output_paths)
            for k, v in stats.items():
                total_stats[k] += v

    print("初步过滤完成")
    print("并行过滤统计情况如下")
    for k, v in total_stats.items():
        print(f" - {k}: {v}, {v/total_stats['total']:.2%}")
    
    return all_output_paths, total_stats

def exact_deduplicate(input_files: List[os.PathLike], input_base_dir: str | os.PathLike, config: Dict[str, Any]):
    output_dir = os.path.join(config['paths']['base_output_dir'], 'exact_dedup')
    output_paths, total_lines_before, total_lines_after = exact_line_deduplication(input_files=input_files,
                                                                                   input_base_dir=input_base_dir,
                                                                                   output_directory=output_dir)
    return output_paths, output_dir


def fuzzy_deduplicate(input_files: List[os.PathLike], input_base_dir: str | os.PathLike, config: Dict[str, int]):
    output_dir = os.path.join(config['paths']['base_output_dir'], 'fuzzy_dedup') # type: ignore
    dedup_conf = config['deduplication']['fuzzy'] # type: ignore
    output_paths, before_count, after_count = minhash_deduplication(
        input_files=input_files,
        input_base_dir=input_base_dir,
        num_hashes=dedup_conf['num_hashes'],
        num_bands=dedup_conf['bands'],
        n=dedup_conf['n_gram'],
        output_dir=output_dir,
        jaccard_threshold=dedup_conf['threshold']
    )
    return output_paths, before_count, after_count
    

def write_shard(file_paths: List[Path], output_file: Path, separator: str):
    with open(output_file, 'w', encoding='utf-8') as fout:
        for fp in file_paths:
            with open(fp, 'r', encoding='utf-8') as fin:
                content = fin.read().strip()
                # 清洗异常行终止符
                # 将 Unicode 行分隔符(LS)和段落分隔符(PS) 统一替换为标准换行符
                content = content.replace('\u2028', '\n').replace('\u2029', '\n')
                # 只有内容非空才写入
                if content:
                    fout.write(content)
                    fout.write(separator) # 拼接分隔符

def build_dataset_parallel(kept_files: List[Path], output_dir: Path, num_shards: int = 16):
    os.makedirs(output_dir, exist_ok=True)
    
    # 将文件列表切分为 num_shards 份
    chunk_size = len(kept_files) // num_shards + 1
    chunks = [kept_files[i:i + chunk_size] for i in range(0, len(kept_files), chunk_size)]
    
    tasks = []
    for i, chunk in enumerate(chunks):
        shard_path = output_dir / f"corpus_shard_{i:03d}.txt"
        tasks.append((chunk, shard_path, "<|endoftext|>\n"))
    
    # 并行写入
    with multiprocessing.Pool(processes=15) as pool:
        pool.starmap(write_shard, tasks)

if __name__ == "__main__":
    # input_dir = 'data/crawls/wet'
    # filtered_output_dir = 'data/filtered'
    # exact_output_dir = 'data/exact_deduplicated'
    # fuzzy_output_dir = 'data/fuzzy_deduplicated'
    parser = argparse.ArgumentParser(description="Run the data processing pipeline.")
    parser.add_argument("--config", type=str, default="test_pipeline.yaml", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # 1. 加载配置
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    config = load_config(args.config)
    print(f"成功加载配置: {args.config}")
    print(f"项目名称: {config['project_name']}")

    os.makedirs(config['paths']['base_output_dir'], exist_ok=True)

    filtered_doc_paths, total_stats = filter_wet_files(config)
    if filtered_doc_paths:
        filtered_base_dir = os.path.join(config['paths']['base_output_dir'], 'filtered')

        print("\n--- 开始精确去重 ---")
        exact_output_paths, exact_base_dir = exact_deduplicate(filtered_doc_paths, filtered_base_dir, config)

        print("\n--- 开始模糊去重 (MinHash) ---")
        fuzzy_output_paths, doc_count_before, doc_count_after = fuzzy_deduplicate(exact_output_paths, exact_base_dir, config)
        fuzzy_output_paths = [
        Path(p) for p in fuzzy_output_paths 
        if os.path.getsize(p) > 0
        ]
        print(f"模糊去重结束: {doc_count_before} -> {doc_count_after} 文档")
        
        print("\n--- 构建最终分片数据集 ---")
        final_shard_dir = os.path.join(config['paths']['base_output_dir'], 'shards')
        build_dataset_parallel(fuzzy_output_paths, Path(final_shard_dir))

        empty_docs = sum(1 for p in fuzzy_output_paths if os.path.getsize(p) == 0)
        print(f"警告：结果中包含 {empty_docs} 个空文件")
