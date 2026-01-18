import json
import os
import re
import glob
import multiprocessing
from functools import partial
from tqdm import tqdm
import pandas as pd
import numpy as np
from datasets import load_dataset, snapshot_download
from huggingface_hub import login
from transformers import AutoTokenizer

# --- Overall Configuration ---
# Set a single base directory for all data
BASE_DATA_DIR = "data"
# Name of the dataset on Hugging Face Hub
DATASET_REPO_ID = "bespokelabs/Bespoke-Stratos-17k"
# Subdirectory for the downloaded raw data
RAW_DATA_SUBDIR = "Bespoke-Stratos-17k"
# Subdirectory for processed outputs (CoT stands for Chain-of-Thought)
COT_DATA_SUBDIR = "CoT"

# --- Filter Configurations ---
# Config for the first filtering pass (long/wrong format)
MODEL_PATH = "models/Qwen2.5-Math-1.5B"  # Model for tokenizer
MAX_LEN = 2560  # Maximum token length
MIN_LEN = 500  # Minimum token length
NUM_PROCESSES = multiprocessing.cpu_count()  # Use all available CPU cores

# Config for the second filtering pass (purity/platinum)
MAX_ANSWER_LEN = 100  # Max character length for the <answer> content

# --- Global Tokenizer ---
# This will be initialized in each worker process for multiprocessing
tokenizer = None


def run_complete_platinum_pipeline():
    """
    Executes the full pipeline from downloading the dataset to generating
    the final "platinum" quality data file.
    """
    # --- 1. Download the Dataset ---
    print("--- Step 1: Downloading Dataset ---")
    raw_data_path = os.path.join(BASE_DATA_DIR, RAW_DATA_SUBDIR)
    download_bespoke_stratos_dataset(raw_data_path)

    # --- 2. Convert to Standard Format ---
    print("\n--- Step 2: Converting to Standard SFT Format ---")
    sft_interim_path = os.path.join(BASE_DATA_DIR, COT_DATA_SUBDIR, "sft_interim_v1.jsonl")
    convert_bespoke_stratos_to_jsonl(raw_data_path, sft_interim_path)

    # --- 3. Filter by Length and Format ---
    print("\n--- Step 3: Filtering by Length and Format ---")
    sft_v4_path = os.path.join(BASE_DATA_DIR, COT_DATA_SUBDIR, "sft_v4_formatted.jsonl")
    filter_long_wrong_format_cot(sft_interim_path, sft_v4_path)

    # --- 4. Filter for Purity (Platinum Version) ---
    print("\n--- Step 4: Filtering for 'Platinum' Purity ---")
    platinum_output_path = os.path.join(BASE_DATA_DIR, COT_DATA_SUBDIR, "sft_v5_platinum.jsonl")
    rejected_output_path = os.path.join(BASE_DATA_DIR, COT_DATA_SUBDIR, "sft_v5_rejected.jsonl")
    filter_pure_cot(sft_v4_path, platinum_output_path, rejected_output_path)

    print("\n--- Pipeline Complete! ---")
    print(f"Final 'platinum' data is available at: {platinum_output_path}")


def download_bespoke_stratos_dataset(target_dir):
    """
    Downloads the Bespoke-Stratos-17k dataset from the Hugging Face Hub.
    """
    os.makedirs(target_dir, exist_ok=True)
    # Set the HF_ENDPOINT, useful for users in regions with restricted access
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    print(f"Downloading dataset '{DATASET_REPO_ID}' to '{target_dir}'...")
    snapshot_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("Download complete.")


def convert_bespoke_stratos_to_jsonl(local_data_dir, output_path):
    """
    Reads local Parquet files, processes them, and saves them in a JSONL format.
    """
    print(f"Reading Parquet files from: {os.path.join(local_data_dir, 'data')} ...")

    parquet_files = glob.glob(os.path.join(local_data_dir, "data", "train-*.parquet"))
    if not parquet_files:
        print("Error: No Parquet files found.")
        return

    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting to JSONL"):
            messages = row.get("conversations")

            if messages is None:
                continue
            if hasattr(messages, '__len__') and len(messages) == 0:
                continue

            if isinstance(messages, np.ndarray):
                messages = messages.tolist()

            if len(messages) > 0 and isinstance(messages[0], list):
                messages = messages[0]

            prompt = ""
            raw_response = ""

            for msg in messages:
                if not isinstance(msg, dict):
                    continue

                role = msg.get("from") or msg.get("role")
                content = msg.get("value") or msg.get("content")

                if role == "user":
                    prompt = content
                elif role == "assistant":
                    raw_response = content

            if not prompt or not raw_response:
                continue

            # Standardize tags
            response = raw_response.replace("<|begin_of_thought|>\n\n", "<think>")
            response = response.replace("\n\n<|end_of_thought|>\n\n", "</think>")
            response = response.replace("<|begin_of_solution|>\n\n", " <answer>")
            response = response.replace("\n\n<|end_of_solution|>", "</answer>").strip()

            entry = {"prompt": prompt, "response": response}
            f.write(json.dumps(entry) + "\n")
            count += 1

    print(f"Successfully converted {count} data entries.")
    print(f"Intermediate file saved to: {output_path}")


# --- Helper functions for multiprocessing in filter_long_wrong_format_cot ---
def init_worker(model_path):
    """Initializes the tokenizer for each worker process."""
    global tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def check_quality_worker(line):
    """Worker function to check a single line of data for quality."""
    global tokenizer
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return False, "json_error", None

    prompt = data.get('prompt', "")
    response = data.get('response', "")

    if not prompt or not response:
        return False, "empty_data", None

    full_text = prompt + response

    try:
        tokens = tokenizer.encode(full_text)
        if not (MIN_LEN <= len(tokens) <= MAX_LEN):
            return False, "length_out_of_range", None
    except Exception:
        return False, "tokenization_error", None

    tags = ["<think>", "</think>", "<answer>", "</answer>"]
    for tag in tags:
        if response.count(tag) != 1:
            return False, f"tag_count_error_{tag}", None

    t_start = response.find("<think>")
    t_end = response.find("</think>")
    a_start = response.find("<answer>")
    a_end = response.find("</answer>")

    if not (t_start < t_end < a_start < a_end):
        return False, "wrong_tag_order", None

    answer_content = response[a_start + 8: a_end]
    if "\\boxed" not in answer_content:
        return False, "no_boxed_in_answer", None

    think_content = response[t_start + 7: t_end].strip()
    if not think_content or not answer_content.strip():
        return False, "empty_content", None

    return True, "pass", json.dumps(data, ensure_ascii=False)


def filter_long_wrong_format_cot(input_file, output_file):
    """
    Filters the dataset based on token length and correct CoT format using multiprocessing.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    valid_count = 0
    reject_stats = {}

    with open(input_file, 'r', encoding='utf-8') as fin, \
            open(output_file, 'w', encoding='utf-8') as fout:

        with multiprocessing.Pool(processes=NUM_PROCESSES, initializer=init_worker, initargs=(MODEL_PATH,)) as pool:

            iterator = pool.imap(check_quality_worker, fin, chunksize=100)

            for is_valid, reason, result_str in tqdm(iterator, total=total_lines, desc="Filtering by length/format"):
                if is_valid:
                    # Ensure newline between tags for consistency
                    fout.write(result_str.replace("</think><answer>", "</think>\n<answer>") + "\n")
                    valid_count += 1
                else:
                    reject_stats[reason] = reject_stats.get(reason, 0) + 1

    print("\n--- Filtering (Length/Format) Report ---")
    print(f"Original data: {total_lines}")
    print(f"Retained data: {valid_count} ({valid_count / total_lines:.2%})")
    print("Rejection reasons:")
    for reason, count in sorted(reject_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason:<25}: {count}")


def filter_pure_cot(input_file, output_file, bad_file):
    """
    Performs the final purity filtering to create the 'platinum' dataset.
    """
    stats = {}
    valid_data = []

    with open(input_file, 'r', encoding='utf-8') as fin, \
            open(bad_file, 'w', encoding='utf-8') as fbad:
        for line in tqdm(fin, desc="Filtering for 'Platinum'"):
            data = json.loads(line)
            is_ok, reason = _clean_and_filter_platinum(data)

            stats[reason] = stats.get(reason, 0) + 1

            if is_ok:
                valid_data.append(data)
            else:
                fbad.write(json.dumps(data, ensure_ascii=False) + "\n")

    with open(output_file, 'w', encoding='utf-8') as fout:
        for data in valid_data:
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

    print("\n--- 'Platinum' Filtering Report ---")
    print(f"Original data: {sum(stats.values())}")
    print(f"Platinum data: {len(valid_data)} ({len(valid_data) / sum(stats.values()):.2%})")
    print("Disposition statistics:")
    for k, v in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k}: {v}")


def _clean_and_filter_platinum(data):
    """Helper function for the final purity check."""
    response = data.get('response', '')

    try:
        think_start = response.find("<think>") + 7
        think_end = response.find("</think>")
        answer_start = response.find("<answer>") + 8
        answer_end = response.find("</answer>")

        if think_start == 6 or think_end == -1 or answer_start == 7 or answer_end == -1:
            return False, "broken_tags"

        think_content = response[think_start:think_end]
        answer_content = response[answer_start:answer_end]
    except Exception:
        return False, "parse_error"

    # Rule 1: No `\boxed` in the thinking process
    if "\\boxed" in think_content:
        return False, "boxed_in_think"

    # Rule 2: Answer must be concise
    if len(answer_content) > MAX_ANSWER_LEN:
        boxed_match = re.search(r"(\\boxed\{.*?\})", answer_content)
        if boxed_match:
            # Attempt to fix by extracting only the boxed part
            clean_answer = boxed_match.group(1)
            data['response'] = f"<think>{think_content}</think>\n<answer>{clean_answer}</answer>"
            return True, "fixed_verbose_answer"
        else:
            return False, "verbose_answer_no_boxed"

    # Rule 3: No newlines in the answer
    if "\n" in answer_content.strip():
        # Attempt to fix by removing newlines
        clean_answer = answer_content.replace("\n", " ").strip()
        data['response'] = f"<think>{think_content}</think>\n<answer>{clean_answer}</answer>"
        return True, "fixed_newline_answer"

    return True, "clean"


if __name__ == "__main__":
    # To run this, you will need to have a model tokenizer locally,
    # for example, at 'models/Qwen2.5-Math-1.5B'
    # If you do not have the model, you can change the MODEL_PATH to a
    # valid Hugging Face model name that has a fast tokenizer.
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Tokenizer model path not found at '{MODEL_PATH}'.")
        print("Please download the model or update the MODEL_PATH variable.")
    else:
        run_complete_platinum_pipeline()