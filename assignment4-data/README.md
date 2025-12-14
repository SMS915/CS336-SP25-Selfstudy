[![中文](https://img.shields.io/badge/lang-中文-red.svg)](README_zh-CN.md) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Course](https://img.shields.io/badge/Course-Stanford_CS336-cardinal)

# CS336 Assignment 4: Language Model Data Filtering Pipeline



## 1. Project Overview

This project aims to implement a complete, end-to-end data processing pipeline designed to clean and filter high-quality datasets suitable for training Large Language Models (LLMs) from raw web data in the Common Crawl corpus. The pipeline adopts a modular design, covering key steps such as text extraction, multi-stage content and quality filtering, and large-scale data deduplication.

The **architectural design** of this pipeline is optimized for processing massive datasets, building a **highly concurrent system based on multi-processing**. Core improvements include:

1.  **Parallel Architecture**: Utilizes `ProcessPoolExecutor` to achieve full-link parallel acceleration for computationally intensive tasks such as text extraction, filtering, signature calculation, and deduplication verification.
2.  **Memory Optimization**: Addressing the massive N-gram data in the deduplication stage, a **Lazy Loading** mechanism and the Copy-On-Write (COW) feature of Linux are introduced. This significantly reduces peak memory usage while ensuring verification precision.
3.  **Modularity & Decoupling**: Core algorithms (such as MinHash, LSH, Union-Find clustering) are encapsulated as independent libraries. This supports efficient single-machine operation while reserving interfaces for future migration to distributed clusters (e.g., Slurm/Submitit).

## 2. Core Concepts and Pipeline Design

The core of this project is to build a modular data processing pipeline aimed at distilling high-quality corpora from massive Common Crawl raw texts. The design follows the efficiency principle of "cheap filtering first, expensive processing later" and employs various engineering optimization means during the deduplication stage.

### 2.1 Data Source Strategy

The pipeline processes two main data sources, playing different roles in the flow:

**Primary Processing Object: Common Crawl (CC)**: This constitutes the main body of the final language model training corpus. The pipeline is designed to process large-scale CC WET files.

**High-Quality Reference: Wiki-cited Pages**: This data serves as the "Gold Standard" (positive samples) for training the **High-Quality Text Classifier**, used to define and screen for content with high information density and credibility on the internet.

### 2.2 Key Strategy: Contrastive Learning Quality Classifier

To go beyond simple heuristic rules, **the key strategy of this project lies in** training a custom quality classifier. The training aims to learn to distinguish between two types of text that are both considered "good," rather than just distinguishing "good" from "bad":

*   **Positive Samples (`__label__wiki`)**: Derived from Wikipedia external links that have **passed** a series of strict preliminary screenings (language, Gopher rules, content safety, etc.).
*   **Negative Samples (`__label__cc`)**: Derived from general Common Crawl data, but which **have also passed** the same strict screenings mentioned above.

This **"Elite vs. Good"** training strategy (implemented by `dataset_builder.py` for sample selection and `train_quality_classifier.py` for training) forces the model to learn the deeper stylistic, structural, and vocabulary patterns that distinguish "citation-quality" text from ordinary high-quality web pages. This classifier thus becomes the core of the entire pipeline, embodying **data quality preferences** and **filtering goals**.

### 2.3 Main Filtering Pipeline

The pipeline logic is orchestrated by `pipeline.py`, executing the following core steps through a highly parallel strategy:

1.  **Parallel Preliminary Filtering**:
    *   Uses `ProcessPoolExecutor` to process WET files concurrently.
    *   Each document independently undergoes **Language Identification**, **NSFW/Toxic Content Detection**, **Gopher Rules**, and the **Custom Quality Classifier**.
    *   Uses `fastwarc` and `xxhash` libraries to accelerate basic I/O and hash calculations, ensuring high throughput in the "cheap filtering" stage.

2.  **Global Exact Line Deduplication**:
    *   **Goal**: Remove "Boilerplate" text appearing across documents, such as navigation bars, footer copyright information, and generic advertising slogans.
    *   **Strategy**: Adopts a two-stage algorithm. The first stage scans all documents in parallel to build a global line frequency counter; the second stage rewrites documents, retaining only lines with a global frequency of 1. This means any sentence appearing more than once is eliminated, significantly increasing the uniqueness density of the corpus.

3.  **Fuzzy Deduplication (MinHash + LSH)**:
    *   Targets documents with highly similar but not identical content (e.g., news reprints with different timestamps).
    *   **Signature Generation**: Parallel calculation of document n-gram MinHash signatures.
    *   **LSH Candidate Discovery**: Uses Locality Sensitive Hashing (LSH) to map high-dimensional signatures into hash buckets, reducing $O(N^2)$ full comparisons to only comparing conflicting elements within buckets.
    *   **Lazy Jaccard Verification**: This is **key to memory optimization**. When verifying candidates produced by LSH, the huge N-gram mapping table is not preloaded into memory. Instead, file paths are passed. Child processes re-read files on demand (Lazy Load) and calculate Jaccard similarity on the spot. This "computation for memory" strategy eliminates memory bottlenecks during large-scale deduplication.
    *   **Clustering & Filtering**: Uses Union-Find to cluster duplicate documents and filters out unique documents based on deterministic rules (keeping the one with the smallest ID).

4.  **Parallel Sharding Output**:
    *   Instead of generating a single huge text file, the final cleaned data is **written in parallel** to multiple shard files (e.g., `corpus_shard_001.txt`).
    *   Documents are joined using the `<|endoftext|>` separator, directly adapting to most LLM training frameworks (such as GPT-NeoX, Megatron-LM) and the data loading requirements for the assignment 1 training model.

## 3. Pipeline Results and Data Analysis

This section presents key statistics from running the data processing pipeline on sample datasets and analyzes them. These figures quantify the filtering effect of each stage on raw Common Crawl data and verify the effectiveness of the pipeline design.

### 3.1 Observations from Small-Scale Filtering

The following data are statistical results obtained after running **Parallel Preliminary Filtering** on a Common Crawl WET file sample containing **545,744** documents:

| Filtering Step                            | Removed Docs | Remaining Docs | % Removed (of remaining) | % of Total  | Note                                       |
| :---------------------------------------- | :----------: | :------------: | :----------------------: | :---------: | :----------------------------------------- |
| **Short Text** (`short_count < 100`)      |    11,928    |    533,816     |          ~2.19%          |   ~2.19%    | Removes pages with too little content      |
| **Language** (`lang_failed`)              |   494,663    |     39,153     |         ~92.67%          | **~90.64%** | Filters non-English or low-confidence text |
| **NSFW Content** (`nsfw_failed`)          |     129      |     39,024     |          ~0.33%          |   ~0.023%   | Removes NSFW content                       |
| **Toxic Speech** (`toxic_failed`)         |     264      |     38,760     |          ~0.67%          |   ~0.048%   | Removes Toxic content                      |
| **Gopher Rules** (`gopher_failed`)        |    3,573     |     35,187     |          ~9.22%          |   ~0.65%    | Removes structurally low-quality text      |
| **Quality Classifier** (`quality_failed`) |    29,362    |     5,825      |         ~83.44%          |   ~5.38%    | Removes "Good" but not "Excellent" text    |
| **Final Kept** (`kept`)                   |  **5,825**   |       -        |            -             | **~1.07%**  | -                                          |

#### Analysis

1.  **Extremely Low Signal-to-Noise Ratio in Raw CC Data**
    Experimental data clearly indicates that over **98%** of the content in raw Common Crawl data does not meet high-quality standards. The final retention rate of about **1%** is highly consistent with industry findings when building large-scale high-quality corpora (such as C4, RefinedWeb), proving the necessity and effectiveness of this pipeline's strict screening.

2.  **Clear Division of Labor for Filters**
    The **Language Filter** is the most efficient "coarse sieve," removing the vast majority of irrelevant documents (~**90.64%**) on its own, which aligns with the raw distribution of Common Crawl (CC). As data scraped from the entire web, CC contains a large amount of non-English, garbled, or unrecognizable encoding web pages. Other filters (Gopher, Toxic content, etc.) serve as important supplements, capturing specific types of low-quality content.

    The **Custom Quality Classifier** is the final but most critical "fine sieve." Among all documents that passed the basic checks, it removed over 80% (~83.44%) of content judged as "non-citation quality," perfectly reflecting the core "Elite vs. Good" screening strategy of this project.

### 3.2 Scalability Performance Verification

To verify the throughput and stability of the pipeline when processing actual production-level batches, a **subset of 5,000 files** from Common Crawl was selected based on official data scales for a large-scale stress test. This aims to establish a performance baseline for a single node under full load.

#### 3.3.1 Test Environment and Configuration

*   **Input Data**: 5,000 Common Crawl WET files (containing approx. 134 million raw web records, raw compressed size ~375GB).
*   **Hardware Environment**:
    *   CPU: 32 vCPU (Intel Xeon Platinum)
    *   RAM: 120GB
    *   Concurrent Processes: 30
*   **Processing Strategy (Strict Mode)**:
    *   Language Filter (FastText): Threshold **0.9** (Retains only high-confidence English)
    *   Content Safety: NSFW/Toxic Threshold **0.99**
    *   Deduplication Params: Jaccard **0.8**, N-grams **13**, Hash Functions **128**, Bands **20**

#### 3.3.2 Processing Flow and Time Analysis

The entire processing pipeline took approximately **3 hours 46 minutes**. Detailed performance data for each stage is as follows:

**1. Parallel Filtering Stage**
This stage involves streaming reading of raw WET data, language identification, and quality classification.
*   **Time Taken**: 1 hour 35 minutes 05 seconds
*   **Peak Memory**: ~14 GB
*   **Data Flow**: 134,078,978 (Raw) → 1,464,040 (Retained)
*   **Retention Rate**: **1.09%**
*   **Filtering Details**:
    *   **Language Filter (LangID)**: Removed 120 million records (89.61%) — *Note: High threshold caused extensive dropping of imperfect English.*
    *   **Quality Filter (Quality/Short)**: Removed ~11.5 million records (8.59%).
    *   **Safety Filter (Gopher/NSFW/Toxic)**: Removed ~960k records (0.7%).

**2. Exact Line Deduplication Stage**
This stage builds a global line-level hash table to remove all duplicate lines appearing more than once in the corpus (e.g., nav bars, footers).
*   **Time Taken**: 14 minutes 37 seconds (Statistics 43s + Rewrite 13m 54s)
*   **Peak Memory**: 27 GB
*   **Processing Efficiency**:
    *   Raw Lines: 197 million lines
    *   Retained Lines: 57.65 million lines
    *   **Removal Rate**: **70.79%** — *Note: Extremely high removal rate indicates web pages contain massive amounts of templated noise.*

**3. Fuzzy Deduplication Stage (MinHash + LSH)**
Targeting documents after line deduplication for Jaccard similarity-based fuzzy deduplication.
*   **Total Time**: 1 hour 56 minutes 27 seconds
    *   Signature Generation: 1 hour 08 minutes 16 seconds (Memory: 21GB → 31GB)
    *   Candidate Screening: 25 minutes (Peak Memory: **53GB**)
    *   Exact Verification: 23 minutes 11 seconds
*   **Deduplication Results**:
    *   Input Documents: 1,411,697
    *   Candidate Pairs: 194 million pairs (Candidates Explosion)
    *   Final Retained: 1,411,349
    *   **Removal Rate**: **0.01%**

#### 3.3.3 Results Analysis and Discussion

This round of experiments adopted an extremely strict screening strategy, reflecting the following characteristics:

1.  **Extremely High Data Purity and Sparsity**: The combined effect of the language filter (threshold 0.9) and line-level deduplication (removing 70% of content) resulted in data entering the MinHash stage being "atomic" unique fragments, hence the final MinHash deduplication rate was only 0.01%.
2.  **LSH Parameter Sensitivity**: With `Bands=20`, 190 million candidate pairs were generated (avg 138 candidates per doc), causing memory to spike to 53GB during the candidate screening stage. This suggests that with documents already highly cleaned by line deduplication, loose LSH parameters introduce significant invalid low-similarity candidate calculations. The Bands parameter needs adjustment in future optimizations to balance computational overhead.
3.  **System Bottleneck**: Under the current configuration, peak memory occurs during the LSH candidate generation stage, which is the main bottleneck limiting single-node processing of larger-scale data.

## 4. Downstream Model Performance Evaluation

#### **4.1 Experimental Setup**

Adopted the same parameter configuration as the official assignment:

*   **Model Architecture**: LLaMA-like Decoder-Only Language Model (SwiGLU, RoPE, RMSNorm).
    *   n_layers: 12, n_heads: 12, d_model: 768, d_ff: 2048, context_length: 512, num_params: ~122M.
*   **Training Config**: 11K iterations, 1 GPU (5090), batch size 128.
*   **Baseline**: OpenWebText (Representing manually curated high-quality Reddit external link data).
*   **Experimental Group (Ours)**: Rigorously selected dataset based on Common Crawl (Self-Dataset).
*   **Evaluation Metric**: Validation Loss / Perplexity on **C4 (en) validation set**.

#### **4.2 Training Dynamics Comparison**

![dataset comparison loss curve](asset/loss_curve.png)

As shown in the figure, maintaining the exact same model architecture (LLaMA-like) and hyperparameters, we compared the training dynamics of the Self-Dataset (Cyan curve) against the OpenWebText Baseline (OWT, Yellow curve) over the first 11k steps. The experiment presents the following significant statistical phenomena:

1.  **Superior Out-of-Distribution Generalization**:
    On the C4 Validation Set, the Self-Dataset's Loss curve (Cyan dashed line) consistently remains lower than the OWT baseline (Yellow dashed line) from the early stages of training (around 2k steps). Although neither has fully converged, this continuous performance gap indicates that data cleaned by the automated pipeline is distributionally closer to high-quality general corpora and possesses stronger generalization capabilities.
2.  **The "Scissors Gap" Phenomenon between Training and Validation**:
    It is worth noting a counter-intuitive but highly valuable phenomenon: **The Self-Dataset's Training Loss (solid line) is consistently higher than OWT, but the Validation Loss (dashed line) is consistently lower than OWT.**
    *   **High Training Loss** indicates the training data is "harder to learn." This is attributed to MinHash and exact line deduplication strategies thoroughly removing simple repetitive patterns (Boilerplate) and redundant fragments, forcing the model not to rely on "rote memorization" to lower Loss, but to learn deep linguistic laws.
    *   **Low Validation Loss** indicates the model "learned correctly." This proves that the features learned by the model while "chewing on hard bones" have stronger transferability.
3.  **Extremely High Data Efficiency**:
    The performance gap is established in the very early stages of training (first 10% progress, approx. 2,000 steps) and gradually widens. This implies the Self-Dataset has higher **Information Density**. The model can achieve performance levels surpassing the manually curated dataset (OWT) using less Compute Budget and fewer Token consumptions.

<p align='center'>Table: Training Dynamics Comparison - Self-Dataset (Self) vs. OpenWebText (OWT)</p>

| Training Steps (Steps) | **Training Loss**  |                  | **Validation Loss (C4)** |                      | Perplexity (C4)    |                      |
| ---------------------- | ------------------ | ---------------- | ------------------------ | -------------------- | ------------------ | -------------------- |
|                        | **OWT** (Baseline) | **Self** (Ours)  | **OWT** (Baseline)       | **Self** (Ours)      | **OWT** (Baseline) | **Self** (Ours)      |
| **2,000** (Early)      | 3.94               | 4.15             | 4.62                     | **4.40**             | 101.6              | 82.2                 |
| **5,000** (Mid)        | 3.68               | 3.83             | 4.29                     | **4.10**             | 79.7               | 60.7                 |
| **8,000**              | 3.52               | 3.63             | 4.26                     | **4.00**             | 71.0               | 54.7                 |
| **11,000** (Final)     | 3.47               | 3.58             | 4.23                     | **3.94**             | 68.5               | 51.5                 |
| **`Δ` (Final)**        | -                  | *+0.11 (Higher)* | -                        | ***-0.10 (Better)*** |                    | ***-17.0 (Better)*** |

## 5. Project Structure

The core logic of this project is unified under the `cs336_data` Python package, achieving separation between reusable library code and executable scripts.

```
·
├── cs336_data/                                # Core Python package, containing all implementations
│   ├── __init__.py                            # Package initialization
│   ├── dataset_builder.py                     # [Executable] Core script to build quality classifier dataset
│   ├── deduplication.py                       # [Library] Core logic for Exact Line Dedup & MinHash+LSH
│   ├── extraction.py                          # [Library] Robust text extraction from HTML
│   ├── download_c4_subset.py                  # [Executable] Streams allenai/c4 en validation set
│   ├── filter.py                              # [Library] Contains all filter components: Lang, Gopher, NSFW/Toxic, Quality
│   ├── pipeline.py                            # [Executable] Final end-to-end data processing pipeline main script
│   ├── prepare_wiki_data.py                   # [Executable] Samples from Wiki URL list and generates download scripts
│   ├── quality_classifier.py                  # [Library] QualityClassifier class for loading models and prediction
│   ├── sample_cc_paths.py                     # [Executable] Samples from Common Crawl path list and generates download scripts
│   ├── sample_data_from_warc.py               # [Executable] Exploratory script to generate sample CSVs for analysis
│   ├── train_quality_classifier.py            # [Executable] Main script to train quality classifier using YAML config
│   ├── UF.py                                  # [Library] Implementation of Union-Find data structure
│   └── utils.py                               # [Library] General utility functions, e.g., text normalization
│
├── data/                                      # (Ignored by .gitignore) Stores all data
│   ├── cc_path/                               # Stores Common Crawl path list files
│   ├── classifiers/                           # Stores pre-trained fastText classifier models
│   ├── classifiers_dataset/                   # Stores generated datasets for training quality classifier
│   ├── crawls/                                # Stores downloaded WARC/WET sample files
│   ├── dataset/                               # Stores cleaned data files
│   ├── my_classifiers/                        # Stores self-trained quality classifier models
│   ├── wiki/                                  # Stores downloaded Wikipedia page WARC files
│   └── wiki_links/                            # Stores Wikipedia URL list files
│
├── scripts/                                   # Convenient Bash execution scripts
│   ├── build_fasttext_dataset.sh              # Wrapper script for dataset_builder.py
│   ├── download_requirings.sh                 # One-click download for all required lists and pre-trained Fasttext models
│   ├── download_wet_file.sh                   # (Generated by sample_cc_paths.py) Script to download WET files
│   ├── download_wiki_pages.sh                 # (Generated by prepare_wiki_data.py) Script to download Wiki pages
│   └── train_fasttext_classifier.sh           # Wrapper script for train_quality_classifier.py
│
├── classifier_config.yaml                     # YAML config file for fasttext tokenizer training
│
├── configs/
│   ├── test_pipeline.yaml                     # Small batch config for testing pipeline functionality
│   └── 5000_scale_config.yaml                 # Configuration for 5000 wet file sample scale
│
├── tests/
│   ├── adapters.py                            # Official test interface adapters
│   └── ...                                    # Official test cases
│
├── cs336_spring2025_assignment4_data.pdf      # Official handout
├── [Bilingual]cs336_spring2025_assignment4_data.pdf # Original and translated mixed file
└── uv.lock                                    # Official dependency lock file, may not adapt to newer GPUs (e.g., Blackwell)
```

## 6. Usage Workflow

The workflow of this project is designed as a series of clear, independent steps. It is recommended to execute the scripts in the following order to complete the full process from data preparation to model training.

### **Step 1: Data Preparation**

The goal of this step is to sample and download the web data (WARC/WET format) required for subsequent steps from massive raw list files.

**A. Prepare Wikipedia Pages (For Positive Samples)**

First, sample from the Wikipedia URL list and generate a download script.

1.  **Sample URLs**:
    Run `prepare_wiki_data.py` to sample URLs. Use the outputted txt link list with `wget` to download corresponding WARC files.

    ```bash
    # Ensure enwiki-20240420-extracted_urls.txt.gz is downloaded to data/wiki/ directory
    python -m cs336_data.prepare_wiki_data --num-samples 15000 # Specify number of samples
    ```

    The command will generate a text file named `data/wiki/subsampled_positive_{num_samples}_urls.txt` containing the specified number of URLs to download.

2.  **Create and Enter `tmux` Session**:
    Create a new session named `download` for the download task.

    ```bash
    tmux new -s download_wiki_pages
    ```

3.  **Execute Download in `tmux` Session**:
    In the popped-up new `tmux` window, run the bash script for downloading links.

    ```bash
    chmod +x scripts/download_wiki_pages.sh
    scripts/download_wiki_pages.sh
    # <ctrl + b> + d to detach from session
    ```

    If you need to re-attach to the session, run the following command:

    ```bash
    # Re-attach to session
    tmux attach -t download_wiki_pages
    ```

    After the download is complete, kill the session:

    ```bash
    # Kill session
    tmux kill-session -t download_wiki_pages
    ```

**B. Prepare Common Crawl Pages (For Negative Samples and Pipeline Testing)**

The `sample_cc_paths.py` script is used to sample from the Common Crawl path list. Since the sampling script uses a fixed random seed, it supports **incremental downloading** to avoid downloading large amounts of duplicate wet files across multiple runs.

1.  **Initial Sampling**:
    ```bash
    # Sample 20 WET file links from wet.paths.gz list and generate download script download_cc_batch_1.sh
    # Defaults to downloading only WET files, add "--download_warc" to download corresponding warc.gz file
    python sample_cc_path.py data/cc_path/wet.paths.gz -n 100 --output-script scripts/download_cc_batch_1.sh --download_warc 
    ```

2.  **Subsequent Incremental Sampling**:
    If more non-duplicate samples are needed, use the `--skip` parameter.

    ```bash
    # Based on the already sampled 20, sample another 100 brand new file links
    python -m cs336_data.sample_cc_paths data/cc_path/wet.paths.gz -n 100 --skip 20  --output-script scripts/download_cc_batch_2.sh
    ```

3.  **Execute Download**:
    Execute the background download task for each generated script.
    
    ```bash
    chmod +x scripts/download_cc_batch_1.sh
    nohup scripts/download_cc_batch_1.sh > cc_download_1.log 2>&1 &
    ```

### **Step 2: Build and Train Quality Classifier**

1.  **Build Training Set**:
    This script samples from the specified WARC files and wiki external page files according to the "Elite vs. Good" strategy to generate `.train` and `.valid` datasets.

    ```bash
    # The script looks for wiki and warc data sources under data/wiki and data/crawls; modify script config based on specific paths
    chmod +x scripts/build_fasttext_dataset.sh
    scripts/build_fasttext_dataset.sh
    ```

2.  **Train Model**:
    Use the YAML configuration file to train the classifier.
    ```bash
    python -m cs336_data.train_quality_classifier --config classifier_config.yaml
    ```

### **Step 3: Run Full Data Filtering Pipeline**

After the quality classifier training is complete, execute the main pipeline script to process the downloaded **WET** files.
```bash
# The script will process all WET files in the data/crawls/wet directory
python -m cs336_data.pipeline --config configs/{config_name}.yaml
```

## 7. Environment Setup and Installation

1.  **Create and Sync Virtual Environment using `uv`:**
    ```bash
    uv venv
    source .venv/bin/activate
    
    uv sync # Newer GPUs, such as Blackwell architecture, may not support the official torch version; manual upgrade of torch and dependencies might be needed.
    
    uv pip install xxhash  # Exact line removal uses hash functions from xxhash for maximum efficiency
    
    uv pip install datasets # Used to download c4 validation set
    
    sudo apt-get install tmux # Install tmux for long-running sessions
    ```

2.  Download Required Data
    ```bash
    # All external data required for this project (CC samples, pre-trained classifiers, Wikipedia URL lists) can be downloaded via one script.
    chmod +x scripts/download_requirings.sh
    scripts/download_requirings.sh
    ```

## 7. Key Dependencies

*   `fastwarc`: For efficient reading of WARC/WET files.
*   `fasttext`: For training and running text classifiers.
*   `numpy`: For numerical calculations, especially in MinHash signatures.
*   `mmh3`: For high-performance non-cryptographic hashing (MurmurHash3).
*   `nltk`: For text processing tools like `ngrams`.
*   `tqdm`: For generating user-friendly progress bars.
*   `pyyaml`: For parsing YAML configuration files.