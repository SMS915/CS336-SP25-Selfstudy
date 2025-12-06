# CS336 作业4：语言模型数据过滤流水线

## 1. 项目概述

本项目旨在实现一个完整的、端到端的数据处理流水线，用于从 Common Crawl（通用爬取）语料库的原始网页数据中，清洗和筛选出适用于训练大型语言模型（LLM）的高质量数据集。该流水线采用模块化设计，涵盖了文本提取、多阶段内容与质量过滤，以及大规模数据去重等关键步骤。

项目的核心目标是将海量的、充满噪声的原始网页文本，转化为一个高质量、干净且内容唯一的预训练语料库。本项目的一个关键亮点是自定义训练了一个文本质量分类器，该分类器专门用于区分高信息密度的“引用级”网页内容与普通的、质量尚可的通用网页。



本流水线的**架构设计**考虑了对大规模数据集（如斯坦福CS336课程提供的5000个WET文件）的处理需求。因此，项目采用了**模块化的结构**，将各个处理步骤（过滤、去重等）解耦，**为未来的并行化和分布式部署奠定了基础**。当前实现是一个功能完备的单机版本，重点在于验证和实现每个核心算法的正确性。

## 2. 项目结构

本项目的核心逻辑被统一组织在 `cs336_data` 这个Python包中，实现了可复用“库”代码与可执行“脚本”的分离。

```
cs336_data/
├── __init__.py                  # 包初始化文件
├── dataset_builder.py           # 【脚本】构建质量分类器的训练数据集，支持多种负采样策略
├── deduplication.py             # 【库】  精确行去重与MinHash+LSH近似去重的核心逻辑
├── extraction.py                # 【库】  从原始HTML中进行稳健的文本提取
├── filter.py                    # 【库】  包含所有过滤组件：语言识别、Gopher规则、NSFW/Toxic分类器，以及核心的 `judge_high_quality` 函数
├── pipeline.py                  # 【脚本】最终的端到端流水线主脚本，负责编排所有处理阶段
├── prepare_wiki_data.py         # 【脚本】用于从维基百科URL源文件采样，并生成wget下载命令的辅助脚本
├── quality_classifier.py        # 【库】  包含 `QualityClassifier` 类，用于加载训练好的模型并执行预测
├── sample_data_from_warc.py     # 【脚本】用于生成样本CSV文件以供分析和确定阈值的探索性脚本
├── train_quality_classifier.py  # 【脚本】使用YAML配置文件来训练fastText质量分类器
├── UF.py                        # 【库】  并查集（Disjoint Set Union）数据结构的实现，用于聚类
└── utils.py                     # 【库】  包含通用辅助函数，如文本标准化 `normalize_text_for_duplication`
```

## 3. 核心概念与流水线设计

本项目的核心是构建一个模块化的数据处理流水线，旨在从海量的Common Crawl原始文本中，提炼出适用于语言模型训练的高质量语料。整个设计遵循“廉价过滤先行，昂贵处理置后”的效率原则。



### 3.1 数据源

本管道处理两种主要的数据源，它们扮演不同的角色：

- **主要处理对象：Common Crawl (CC)**: 这是构成最终语言模型训练语料的主体。本项目的流水线 (pipeline.py) 被设计用于处理大规模的CC WET文件（例如，斯坦福CS336课程提供的5000个文件），这些文件包含了未经筛选的、多样化的互联网文本。
- **高质量参照系：维基百科外部链接 (Wiki-cited Pages)**: 这部分数据并不直接进入最终的训练语料，而是作为训练**高质量文本分类器**的“黄金标准”（正样本）。这些由维基百科编辑“策选”出的页面，被假定代表了互联网上信息密度和可信度较高的内容。



### 3.2 关键组件：对比学习策略的质量分类器

为了超越简单的启发式规则，**本项目的关键策略在于**训练了一个自定义的质量分类器。该分类器的训练旨在学习区分两种都算“良好”的文本，而不仅仅是“好”与“坏”：

1.  **数据源准备**: 整个流程始于从维基百科出站链接数据集中采样URL（`prepare_wiki_data.py`），这些URL指向的页面被视为高质量正样本的来源。

2.  **分类器训练策略**: 本项目的一大特色是其质量分类器的训练策略。该分类器旨在区分两种都算“好”的文本，而不仅仅是“好”与“坏”：
    *   **正样本 (`__label__wiki`)**: 来源于维基百科外部链接，且**通过了**一系列严格的初步筛选。
    *   **负样本 (`__label__cc`)**: 来源于通用的Common Crawl，但**同样也通过了**上述严格的筛选。


​	这种 **“精英 vs. 良好”** 的训练策略（由 `dataset_builder.py` 实现，`train_quality_classifier.py` 执行训练）迫使模型去学习那些真正能区分“引用级”文本和普通高质量网页之间的，更深层次的文体、结构和词汇模式。这个分类器因此成为了整个流水线中**体现数据质量偏好和筛选目标的核心**。



## 3.3 主过滤流水线

这是一个**编排脚本**，负责按正确的逻辑顺序调用所有已实现的库函数，来处理一个或多个WET文件。它展示了如何将各个独立的组件组合成一个端到端的处理流程。



*   **处理流程**: 脚本首先对输入的Common Crawl文档应用一个过滤器链，然后将通过筛选的文档送入去重模块，最终生成聚合后的语料库。其核心步骤如下：

    1. **初步过滤**: 对每个文档独立应用**语言识别**、**Gopher规则**、**有害内容过滤**，以及最重要的**自定义质量分类器**。
    2. **全局去重**: 对所有通过了初步过滤的文档，进行两阶段去重。首先用exact_line_deduplication（精确行去重）移除样板文字，然后用minhash + LSH deduplication（模糊去重）来移除内容高度相似的文档。
    3. **聚合输出**: 将最终干净、唯一的文档聚合成一个单一的语料库文件。

*   **架构与扩展性**: 当前的`pipeline.py`实现是一个功能完备的**单进程串行版本**，其重点在于验证和展示整个流程的逻辑正确性。由于其模块化的设计（每个WET文件被独立处理），其核心处理逻辑可以被轻松地封装并部署到并行计算框架（如 Python 的 concurrent.futures 或集群调度工具 submitit）中，以适应未来更大规模的处理任务。

    

## 4. 环境设置与安装



1.  **使用 `uv` 创建并同步虚拟环境:**
    ```bash
    uv venv
    source .venv/bin/activate
    
    uv sync # 较新的GPU，如Blackwell架构可能不支持官方给出的torch版本，需要手动升级torch与相关依赖
    ```

    

2.  创建数据文件夹

    ```
    # 2和以后的下载逻辑都被封装到了该脚本中
    ./data_downloading.sh
    ```

    

    ```bash
    # 创建数据父目录
    mkdir -p data
    cd data
    
    # 创建存放Common Crawls文件的目录
    mkdir crawls
    
    # 创建存放预训练的fasttext分类器的目录
    mkdir classifiers
    
    # 创建存放wiki外部链接和数据的目录
    mkdir wiki
    
    cd ..
    ```

    

3.  **下载所需数据**:

    下载Common Crawl的样本文件（WARC/WET），并放置在 `data/crawls/` 目录下。

    ```bash
    cd data/crawls
    
    # 官方给出的示例数据下载代码
    wget https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-18/segments/1744889135610.12/warc/CC-MAIN-20250417135010-20250417165010-00065.warc.gz
    wget https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-18/segments/1744889135610.12/wet/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz
    
    cd ../..
    ```

4.  **下载预训练 Fasttext 分类器**

    下载预训练好的分类器模型（`lid.176.bin` 等），并放置在 `data/classifiers/` 目录下。

    ```bash
    cd data/classifiers
    
    # 下载语言识别模型 lid.176.bin
    wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
    
    # 下载NSFW内容分类器
    wget https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_nsfw_final.bin
    
    # 下载有害言论分类器
    wget https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_hatespeech_final.bin
    
    cd ../..
    ```

    

5.  **下载维基百科外部链接列表文件**

    下载维基百科的URL列表文件（`enwiki-...-urls.txt.gz`），并放置在 `data/wiki/` 目录下。

    ```bash
    cd data/wiki
    
    # 下载斯坦福官方给出的url压缩包，包含43.5M个维基百科外部链接
    wget https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment4/enwiki-20240420-extracted_urls.txt.gz
    
    cd ../..
    ```

    

## 5. 使用工作流

本项目设计为按以下顺序执行：

1.  **准备正样本URL** (如尚未完成):
    运行 `prepare_wiki_data.py` 采样URL，并使用其输出配合 `wget` 下载对应的WARC文件。

2.  **构建分类器训练集**:
    运行数据集构建脚本，以生成用于训练质量分类器的 `.train` 和 `.valid` 文件。
    ```bash
    python -m cs336_data.dataset_builder
    ```

3.  **训练质量分类器**:
    根据需要修改 `configs/quality_classifier_config.yaml` 配置文件，然后运行训练脚本。
    ```bash
    python -m cs336_data.train_quality_classifier --config classifier_config.yaml
    ```

4.  **运行完整数据过滤流水线**:
    在质量分类器训练完成并更新其路径后，执行主流水线脚本来处理大规模的WET文件。
    ```bash
    python -m cs336_data.pipeline
    ```

## 6. 关键依赖

*   `fastwarc`: 用于高效地读取WARC/WET文件。
*   `fasttext`: 用于训练和运行文本分类器。
*   `numpy`: 用于数值计算，尤其是在MinHash签名中。
*   `mmh3`: 用于高性能的非加密哈希算法 (MurmurHash3)。
*   `nltk`: 用于 `ngrams` 等文本处理工具。
*   `tqdm`: 用于生成友好的进度条。
*   `pyyaml`: 用于解析YAML配置文件。