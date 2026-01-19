# cs336_data/pipeline.py
import argparse
import multiprocessing
import fasttext
import os
import glob
import yaml
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional, Tuple, List, Any
from collections import defaultdict
from functools import partial

from fastwarc.warc import ArchiveIterator, WarcRecordType
from .filter import (gopher_quality_filter, 
                     classify_text,
                     mask_all_pii)

from .deduplication import minhash_deduplication, exact_line_deduplication
from .quality_classifier import QualityClassifier


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

_MODELS_LOADED = False
LANG_MODEL = None
NSFW_MODEL = None
TOXIC_MODEL = None
QUALITY_CLASSIFIER = None

def init_models_globally(config: Dict[str, Any]):
    """在主进程中加载模型到全局变量"""
    global LANG_MODEL, NSFW_MODEL, TOXIC_MODEL, QUALITY_CLASSIFIER, _MODELS_LOADED
    
    if _MODELS_LOADED:
        return
        
    print(f"Parent process {os.getpid()} loading models for COW...")
    m = config['models']
    try:
        LANG_MODEL = fasttext.load_model(m['lang_path'])
        NSFW_MODEL = fasttext.load_model(m['nsfw_path'])
        TOXIC_MODEL = fasttext.load_model(m['toxic_path'])
        TOXIC_MODEL = fasttext.load_model(m['toxic_path']) # 补上刚才 Traceback 里报错的这个
        QUALITY_CLASSIFIER = QualityClassifier(m['quality_classifier'], m['label_mapping'])
        _MODELS_LOADED = True
        print(f"Process {os.getpid()} models loaded successfully.")
    except Exception as e:
        print(f"Error loading models in process {os.getpid()}: {e}")
        raise e


def _worker_initializer(config: Dict[str, Any]):
    """
    这个函数会在每个子进程启动时只执行一次。
    用于加载模型并存入全局变量。
    """
    init_models_globally(config)

def process_single_wet_file(
    wet_file_path: str | os.PathLike,
    output_dir: str | os.PathLike,
    config: Dict[str, Any]
    ) -> Tuple[List[str | os.PathLike], Dict[str, int]]:
    
    stats = defaultdict(int)
    masked_dict = defaultdict(int)
    # 为每个wet文件创立对应的二级目录
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
            if record.record_type != WarcRecordType.conversion:
                stats['other_record_type'] += 1
                continue
            
            stats['total'] += 1
            text = record.reader.read().decode('utf-8', errors='ignore')

            # 长度预检，高效过滤
            if not text or text.isspace() or len(text) < min_len:
                stats['short_count'] += 1
                continue

            text_for_filter = text.replace('\n', ' ').strip()

            # lang_code, lang_score = classify_text(lang_model, text_for_filter)
            lang_code, lang_score = classify_text(LANG_MODEL, text_for_filter)
            if lang_code != lang_target or lang_score <= lang_min_score:
                stats['lang_failed'] += 1
                continue
                
            # nsfw_label, nsfw_score = classify_text(nsfw_model, text_for_filter)
            nsfw_label, nsfw_score = classify_text(NSFW_MODEL, text_for_filter)
            transformed_nsfw_score = nsfw_score if nsfw_label == 'nsfw' else (1 - nsfw_score)
            if transformed_nsfw_score > nsfw_thresh:
                stats['nsfw_failed'] += 1
                continue

            # toxic_label, toxic_score = classify_text(toxic_model, text_for_filter)
            toxic_label, toxic_score = classify_text(TOXIC_MODEL, text_for_filter)
            transformed_toxic_score = toxic_score if toxic_label == 'toxic' else (1 - toxic_score)
            if transformed_toxic_score > toxic_thresh:
                stats['toxic_failed'] += 1
                continue

            if not gopher_quality_filter(text):
                stats['gopher_failed'] += 1
                continue

            # quality_label, _ = quality_classifier.predict(text)
            quality_label, _ = QUALITY_CLASSIFIER.predict(text)
            if quality_label == 'cc':
                stats['quality_failed'] += 1
                continue

            stats['kept'] += 1
            doc_id = f"doc_{i}.txt"
            doc_output_path = os.path.join(output_subdir, doc_id)
            output_paths.append(doc_output_path)
            text_masked, masked_count = mask_all_pii(text)
            for k, v in masked_count.items():
                masked_dict[k] += v
            with open(doc_output_path, 'w', encoding='utf-8') as f_doc:
                f_doc.write(text_masked)

    return output_paths, stats, masked_dict

def filter_wet_files(config: Dict[str, Any], config_path: str, max_cpu_workers: int = 30, worker_ttl: int = 50) -> Tuple[List[os.PathLike], Dict[str, int]]:
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

    max_files = config['processing']['max_files_limit']

    print(f"启动 multiprocessing.Pool (Workers={max_cpu_workers}, TTL={worker_ttl})...")

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
    total_masked_stats = defaultdict(int)

    with multiprocessing.Pool(processes=max_cpu_workers, 
                              initializer=_worker_initializer,
                              initargs=(config,), # 进程启动时加载模型
                              maxtasksperchild=worker_ttl) as pool: # 防内存泄漏

    # processpool写法
    # with ProcessPoolExecutor(max_workers=max_workers,
    #                          initializer=_worker_initializer,
    #                          ) as executor:

    #   results = tqdm(executor.map(process_func, wet_files),
    #                total=len(wet_files),
    #                desc="并行处理wet文件")

        print(f"启动ProcessPoolExecutor，最大工作进程数: {max_cpu_workers}")

        process_func = partial(process_single_wet_file, output_dir=output_dir, config=config)
        results = tqdm(pool.imap_unordered(process_func, wet_files, chunksize=1),
                       total=len(wet_files),
                       desc="并行过滤 WET 文件")
        
        for output_paths, stats, masked_dicts in results:
            all_output_paths.extend(output_paths)
            for k, v in stats.items():
                total_stats[k] += v
            for k, v in masked_dicts.items():
                total_masked_stats[k] += v

    print("初步过滤完成")
    print("并行过滤统计情况如下")
    print(f"过滤后的文件共{len(all_output_paths)}个")
    for k, v in total_stats.items():
        print(f" - {k}: {v}, {v/total_stats['total']:.2%}")
    for k, v in total_masked_stats.items():
        print(f" - {k}: {v}个")
    
    return all_output_paths

def exact_deduplicate(input_files: List[os.PathLike], input_base_dir: str | os.PathLike, config: Dict[str, Any], max_workers: int = 20):
    output_dir = os.path.join(config['paths']['base_output_dir'], 'exact_dedup')
    output_paths, total_lines_before, total_lines_after = exact_line_deduplication(input_files=input_files,
                                                                                   input_base_dir=input_base_dir,
                                                                                   output_directory=output_dir,
                                                                                   num_workers = max_workers)
    return output_paths, output_dir


def fuzzy_deduplicate(input_files: List[os.PathLike], input_base_dir: str | os.PathLike, config: Dict[str, int], max_workers: int = 20):
    output_dir = os.path.join(config['paths']['base_output_dir'], 'fuzzy_dedup') # type: ignore
    dedup_conf = config['deduplication']['fuzzy'] # type: ignore
    output_paths, before_count, after_count = minhash_deduplication(
        input_files=input_files,
        input_base_dir=input_base_dir,
        num_hashes=dedup_conf['num_hashes'],
        num_bands=dedup_conf['bands'],
        n=dedup_conf['n_gram'],
        output_dir=output_dir,
        jaccard_threshold=dedup_conf['threshold'],
        max_workers = max_workers
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

def build_dataset_parallel(kept_files: List[Path], output_dir: Path, num_workers: int = 20):
    os.makedirs(output_dir, exist_ok=True)
    
    num_shards = num_workers

    # 将文件列表切分为 num_shards 份
    chunk_size = len(kept_files) // num_shards + 1
    chunks = [kept_files[i:i + chunk_size] for i in range(0, len(kept_files), chunk_size)]
    
    tasks = []
    for i, chunk in enumerate(chunks):
        shard_path = output_dir / f"corpus_shard_{i:03d}.txt"
        tasks.append((chunk, shard_path, "<|endoftext|>\n"))
    
    # 并行写入
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(write_shard, tasks)


# 辅助函数：统计直接子目录的数量
def count_subdirectories(path: str | os.PathLike) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    # 只统计目录，不递归
    return sum(1 for x in p.iterdir() if x.is_dir())


# 辅助函数：递归获取目录下所有txt文件路径
def get_all_txt_files(path: str | os.PathLike) -> List[Path]:
    return list(Path(path).rglob("*.txt"))


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass
    parser = argparse.ArgumentParser(description="Run the data processing pipeline.")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, # 改为必填或提供更合理的默认值
        help="Path to the YAML config"
    )
    args = parser.parse_args()

    # 1. 加载配置
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    config = load_config(args.config)

    init_models_globally(config)


    print(f"成功加载配置: {args.config}")
    print(f"项目名称: {config['project_name']}")

    resume_mode = config.get('pipeline', {}).get('resume', False)
    base_output = config['paths']['base_output_dir']
    os.makedirs(base_output, exist_ok=True)

    input_dir = config['paths']['input_dir']
    max_files_limit = config['processing'].get('max_files_limit')

    max_cpu_workers = config['processing'].get('max_cpu_workers', 30)
    max_io_workers  = config['processing'].get('max_io_workers', 20)
    worker_ttl = config['processing'].get('worker_ttl', 50)

    process_chunk_size = config['processing'].get('chunk_size', 10)

    # 断点续传检查, 首先获取当前运行预期的中间产出文件夹数量
    expected_limit = 0
    if max_files_limit is not None:
        expected_limit = max_files_limit
        print(f"[Plan] 配置限制最大处理文件数: {expected_limit}")
    else:
        # 如果是 None，则扫描源目录下的 .warc.wet.gz 总数
        print(f"[Plan] 配置为全量处理，正在统计源文件...")
        source_files = glob.glob(os.path.join(input_dir, "*.warc.wet.gz"))
        expected_limit = len(source_files)
        print(f"[Plan] 源目录实际文件数: {expected_limit}")

    if expected_limit == 0:
        print("错误: 预期处理数量为0，请检查输入目录或配置。")
        exit(1)

    filtered_dir = os.path.join(base_output, 'filtered')
    exact_dedup_dir = os.path.join(base_output, 'exact_dedup')
    
    # 状态变量初始化
    exact_input_files = [] # 传给 Exact Dedup 的输入
    exact_input_base = ""
    
    fuzzy_input_files = [] # 传给 Fuzzy Dedup 的输入
    fuzzy_input_base = ""
    
    skip_exact_dedupe = False # 标记位

    # ---------------------------------------------------------
    # 首先检查倒数第二个阶段 Exact Dedupe 输出是否完整
    # ---------------------------------------------------------
    if resume_mode:
        count = count_subdirectories(exact_dedup_dir)
        print(f"[Check] Exact Dedupe 目录 ({exact_dedup_dir}) 包含子目录数: {count}")
        
        # 只要子文件夹数量 >= 预期的 WET 文件数量，就认为该阶段已完成
        if count >= expected_limit:
            print(f">>> [Resume] Exact Dedupe 阶段已完成 ({count} >= {expected_limit})。")
            print(">>> 跳过 Filter 和 Exact Dedupe，加载数据准备进入 Fuzzy Dedupe...")
            
            # 加载该目录下的所有 .txt 文件路径作为下一阶段输入
            fuzzy_input_files = get_all_txt_files(exact_dedup_dir)
            fuzzy_input_base = exact_dedup_dir
            skip_exact_dedupe = True
        else:
            if count > 0:
                print(f"[Resume] Exact Dedupe 不完整 ({count}/{expected_limit})，尝试回退到上一阶段。")
            else:
                print(f"[Resume] Exact Dedupe 目录不存在或为空。")


    # ---------------------------------------------------------
    # 检查一阶段 Filtered 输出是否完整 (如果不能直接跳过 Exact)
    # ---------------------------------------------------------
    if not skip_exact_dedupe:
        if resume_mode:
            count = count_subdirectories(filtered_dir)
            print(f"[Check] Filtered 目录 ({filtered_dir}) 包含子目录数: {count}")
            
            if count >= expected_limit:
                print(f">>> [Resume] Filter 阶段已完成 ({count} >= {expected_limit})。")
                print(">>> 跳过 Filter 阶段，加载数据准备进入 Exact Dedupe...")
                
                exact_input_files = get_all_txt_files(filtered_dir)
                exact_input_base = filtered_dir
            else:
                if count > 0:
                    print(f"[Resume] Filtered 不完整 ({count}/{expected_limit})，将从头开始运行。")

        # ---------------------------------------------------------
        # 执行 Filter (如果需要)
        # ---------------------------------------------------------
        if not exact_input_files:
            print("\n=== 执行 Stage 1: WET 过滤 ===")
            exact_input_files = filter_wet_files(config, args.config, max_cpu_workers,)
            exact_input_base = filtered_dir
            if not exact_input_files:
                print("错误：过滤阶段未产生任何文件，程序终止。")
                exit(1)

        print("\n=== 执行 Stage 2: 精确去重 ===")
        # 注意：exact_deduplicate 会根据 relative_to 保持子目录结构，这符合后续检查子目录数量的逻辑
        fuzzy_input_files, fuzzy_input_base = exact_deduplicate(exact_input_files, exact_input_base, config)


    if fuzzy_input_files:
        print("\n=== 执行 Stage 3: 模糊去重 (MinHash) ===")
        
        # 过滤可能存在的空文件或无效路径
        valid_fuzzy_inputs = [p for p in fuzzy_input_files if os.path.exists(p) and os.path.getsize(p) > 0]
        
        # 再次简单的防御性检查：如果有输入文件，就开始跑
        if len(valid_fuzzy_inputs) == 0:
             print("警告: 没有任何有效文件输入到模糊去重阶段，流程结束。")
             exit(0)

        fuzzy_output_paths, doc_count_before, doc_count_after = fuzzy_deduplicate(
            input_files = valid_fuzzy_inputs, 
            input_base_dir = fuzzy_input_base, 
            config = config,
            max_workers = max_io_workers
        )

        print("\n=== 构建分片数据集 ===")
        fuzzy_output_paths = [Path(p) for p in fuzzy_output_paths if os.path.exists(p) and os.path.getsize(p) > 0]
        
        final_shard_dir = os.path.join(base_output, 'shards')
        build_dataset_parallel(fuzzy_output_paths, Path(final_shard_dir))

        print(f"\nPipeline 完成！最终分片保存在: {final_shard_dir}")
    else:
        print("流程异常：没有文件传递给模糊去重阶段。")



    # os.makedirs(config['paths']['base_output_dir'], exist_ok=True)

    # filtered_doc_paths, total_stats = filter_wet_files(config)
    # if filtered_doc_paths:
    #     filtered_base_dir = os.path.join(config['paths']['base_output_dir'], 'filtered')

    #     print("\n--- 开始精确去重 ---")
    #     exact_output_paths, exact_base_dir = exact_deduplicate(filtered_doc_paths, filtered_base_dir, config)

    #     print("\n--- 开始模糊去重 (MinHash) ---")
    #     fuzzy_output_paths, doc_count_before, doc_count_after = fuzzy_deduplicate(exact_output_paths, exact_base_dir, config)
    #     fuzzy_output_paths = [
    #     Path(p) for p in fuzzy_output_paths 
    #     if os.path.getsize(p) > 0
    #     ]
    #     print(f"模糊去重结束: {doc_count_before} -> {doc_count_after} 文档")
        
    #     print("\n--- 构建最终分片数据集 ---")
    #     final_shard_dir = os.path.join(config['paths']['base_output_dir'], 'shards')
    #     build_dataset_parallel(fuzzy_output_paths, Path(final_shard_dir))

    #     empty_docs = sum(1 for p in fuzzy_output_paths if os.path.getsize(p) == 0)
    #     print(f"警告：结果中包含 {empty_docs} 个空文件")
