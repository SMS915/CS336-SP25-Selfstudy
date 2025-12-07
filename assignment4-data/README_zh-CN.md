# CS336 作业4：语言模型数据过滤流水线

## 1. 项目概述

本项目旨在实现一个完整的、端到端的数据处理流水线，用于从 Common Crawl（通用爬取）语料库的原始网页数据中，清洗和筛选出适用于训练大型语言模型（LLM）的高质量数据集。该流水线采用模块化设计，涵盖了文本提取、多阶段内容与质量过滤，以及大规模数据去重等关键步骤。

本流水线的**架构设计**考虑了对大规模数据集（如斯坦福CS336课程提供的5000个WET文件）的处理需求。因此，项目采用了**模块化的结构**，将各个处理步骤（过滤、去重等）解耦，**为未来的并行化和分布式部署奠定了基础**。当前实现是一个功能完备的单机版本，重点在于验证和实现每个核心算法的正确性。



## 2. 核心概念与流水线设计

本项目的核心是构建一个模块化的数据处理流水线，旨在从海量的Common Crawl原始文本中，提炼出高质量语料。整个设计遵循“廉价过滤先行，昂贵处理置后”的效率原则。



### 2.1 数据源策略 

本管道处理两种主要数据源，它们在流程中扮演不同角色：

**主要处理对象：Common Crawl (CC)**: 这是构成最终语言模型训练语料的主体。本项目的流水线被设计用于处理大规模的CC WET文件。

**高质量参照系：维基百科外部链接 (Wiki-cited Pages)**: 这部分数据作为训练**高质量文本分类器**的“黄金标准”（正样本），用于定义和筛选出互联网上信息密度和可信度较高的内容。



### 2.2 关键策略：对比学习的质量分类器

为了超越简单的启发式规则，**本项目的关键策略在于**训练了一个自定义的质量分类器。该分类器的训练旨在学习区分两种都算“良好”的文本，而不仅仅是“好”与“坏”：

*   **正样本 (`__label__wiki`)**: 来源于维基百科外部链接，且**通过了**一系列严格的初步筛选（语言、Gopher规则、内容安全等）。
*   **负样本 (`__label__cc`)**: 来源于通用的Common Crawl，但**同样也通过了**上述严格的筛选。

​	这种 **“精英 vs. 良好”** 的训练策略（由 `dataset_builder.py` 实现样本筛选，`train_quality_classifier.py` 执行训练）迫使模型去学习区分“引用级”文本和普通高质量网页之间，那些更深层次的文体、结构和词汇模式。这个分类器因此成为了整个流水线中体现**数据质量偏好**和**筛选目标**的核心。



### 2.3 主过滤流水线

这是一个**编排脚本**，负责按正确的逻辑顺序调用所有已实现的库函数，负责按正确的逻辑顺序调用所有已实现的库函数，其核心步骤如下：

1. **初步过滤**: 对每个文档独立应用**语言识别**、**Gopher规则**、**有害内容过滤**，以及最重要的**自定义质量分类器**。
2. **全局去重**: 对所有通过了初步过滤的文档，进行两阶段去重。首先用exact_line_deduplication（精确行去重）移除样板文字，然后用minhash + LSH deduplication（模糊去重）来移除内容高度相似的文档。
3. **聚合输出**: 将最终干净、唯一的文档聚合成一个单一的语料库文件。

* **架构与扩展性**: 当前的`pipeline.py`实现是一个功能完备的**单进程串行版本**，其重点在于验证和展示整个流程的逻辑正确性。由于其模块化的设计（每个WET文件被独立处理），其核心处理逻辑可以被轻松地封装并部署到并行计算框架（如 Python 的 concurrent.futures 或集群调度工具 submitit）中，以适应未来更大规模的处理任务。

  
  
  

## 3. 流水线成果与数据分析

本节展示了数据处理流水线在样本数据集上运行后的关键统计数据，并对其进行分析。这些数据量化了每个过滤阶段对原始Common Crawl数据的筛选效果，并验证了本管道设计的有效性。



### 3.1 初步过滤阶段统计

以下数据是在一个包含 **545,744** 个文档的Common Crawl WET文件样本上，运行**并行初步过滤**后得到的统计结果：

| 过滤步骤                                | 移除的文档数 | 剩余文档数 | (移除)占余量的百分比 | 占总数的百分比 | 备注                       |
| :-------------------------------------- | :----------: | :--------: | :------------------: | :------------: | :------------------------- |
| **短文本过滤** (`short_count < 100`)    |    11,928    |  533,816   |        ~2.19%        |     ~2.19%     | 移除内容过少的页面         |
| **语言过滤** (`lang_failed`)            |   494,663    |   39,153   |       ~92.67%        |  **~90.64%**   | 过滤非英语或低置信度文本   |
| **NSFW内容过滤** (`nsfw_failed`)        |     129      |   39,024   |        ~0.33%        |    ~0.023%     | 移除NSFW内容               |
| **有害言论过滤**(`toxic_failed`)        |     264      |   38,760   |        ~0.67%        |    ~0.048%     | 移除Toxic内容              |
| **Gopher规则** (`gopher_failed`)        |    3,573     |   35,187   |        ~9.22%        |     ~0.65%     | 移除结构性低质量文本       |
| **自定义质量分类器** (`quality_failed`) |    29,362    |   5,825    |       ~83.44%        |     ~5.38%     | 移除“良好”但非“卓越”的文本 |
| **最终保留** (`kept`)                   |  **5,825**   |     -      |          -           |   **~1.07%**   | -                          |

#### 分析与洞见

1. CC原始数据的信噪比极低

   实验数据明确表明，原始Common Crawl数据中超过 **98%** 的内容不符合高质量标准。约 **1%** 的最终保留率与业界在构建大规模高质量语料库（如C4, RefinedWeb）时的发现高度一致，证明了本管道严格筛选的必要性和有效性。

2. 过滤器的作用分工

    **语言过滤器**是最高效的“粗筛”工具，仅此一项就剔除了绝大部分不相关的文档（~**90.64%**），符合 Common Crawl (CC) 原始分布。作为全网抓取的数据，CC包含了大量非英语、乱码、或者无法识别编码的网页。。

    其他过滤器（Gopher、有害内容等）作为重要的补充，捕捉了特定类型的低质量内容。

   **自定义质量分类器**则是最后但最关键的“精筛”工具。在所有通过了基础检查的文档中，它移除了超过八成（~83.44%）被判定为“非引用级”质量的内容，完美体现了本项目“精英 vs. 良好”的核心筛选策略。

   

## 4. 项目结构

本项目的核心逻辑被统一组织在 `cs336_data` 这个Python包中，实现了可复用“库”代码与可执行“脚本”的分离。

```
·
├── cs336_data/                                # 核心Python包，包含所有功能实现
│   ├── __init__.py                            # 包初始化文件
│   ├── dataset_builder.py                     # [可执行] 构建质量分类器训练集的核心脚本
│   ├── deduplication.py                       # [库] 精确行去重与MinHash+LSH近似去重的核心逻辑
│   ├── extraction.py                          # [库] 从HTML中进行稳健的文本提取
│   ├── filter.py                              # [库] 包含所有过滤组件：语言、Gopher、NSFW/Toxic及高质量判断函数
│   ├── pipeline.py                            # [可执行] 最终的端到端数据处理流水线主脚本
│   ├── prepare_wiki_data.py                   # [可执行] 从维基百科URL清单中抽样，并生成下载脚本
│   ├── quality_classifier.py                  # [库] QualityClassifier类，用于加载模型并执行预测
│   ├── sample_cc_paths.py                     # [可执行] 从Common Crawl路径清单中抽样，并生成下载脚本
│   ├── sample_data_from_warc.py               # [可执行] 用于生成样本CSV以供分析和确定阈值的探索性脚本
│   ├── train_quality_classifier.py            # [可执行] 使用YAML配置训练质量分类器的主脚本
│   ├── UF.py                                  # [库] 并查集（Union-Find）数据结构的实现
│   └── utils.py                               # [库] 通用辅助函数，如文本标准化
│
├── data/                                      # (此目录被.gitignore忽略) 存放所有数据
│   ├── cc_path/                               # 存放Common Crawl的路径清单文件
│   ├── classifiers/                           # 存放预训练的fastText分类器模型
│   ├── classifiers_dataset/                   # 存放生成的用于训练质量分类器的数据集
│   ├── crawls/                                # 存放下载的WARC/WET样本文件
│   ├── dataset/                               # 存放清洗后的数据文件	
│   ├── my_classifiers/                        # 存放自己训练好的质量分类器模型
│   ├── wiki/                                  # 存放下载的维基百科页面WARC文件
│   └── wiki_links/                            # 存放维基百科的URL清单文件
│
├── scripts/                                   # 便捷的Bash执行脚本
│   ├── build_fasttext_dataset.sh              # 调用 dataset_builder.py 的封装脚本
│   ├── download_requirings.sh                 # 一键下载所有必需的清单文件和预训练Fasttext模型
│   ├── download_wet_file.sh                   # (由sample_cc_paths.py生成) 下载WET文件的脚本
│   ├── download_wiki_pages.sh                 # (由prepare_wiki_data.py生成) 下载维基百科页面的脚本
│   └── train_fasttext_classifier.sh           # 调用 train_quality_classifier.py 的封装脚本
│
├── classifier_config.yaml                     # fasttext分词器训练YAML配置文件
│
├── tests/
│   ├── adapters.py                            # 官方测试接口适配器
│   └── ...                                    # 官方测试用例
│
├── cs336_spring2025_assignment4_data.pdf      # 官方handout
├── [双语]cs336_spring2025_assignment4_data.pdf # 原文与翻译混合文件
└── uv.lock                                    # 官方环境依赖锁定文件,在较新的 (如Blackwell架构) GPU上不适配
```

## 5. 使用工作流

本项目的工作流被设计为一系列清晰、独立的步骤。推荐按以下顺序执行脚本，以完成从数据准备到模型训练的完整流程。

### **步骤一：准备网页数据 (Data Preparation)**

此步骤的目标是从海量的原始清单文件中，抽样并下载后续步骤所需的网页数据（WARC/WET格式）。

**A. 准备维基百科页面 (用于正样本)**

首先，从维基百科的URL清单中抽样，并生成下载脚本。

1. **抽样URL**:
   运行 `prepare_wiki_data.py` 采样URL，并使用其输出的txt链接列表配合 `wget` 下载对应的WARC文件。

   ```bash
   # 确保 enwiki-20240420-extracted_urls.txt.gz 文件已下载至 data/wiki/ 目录
   python -m cs336_data.prepare_wiki_data --num-samples 15000 # 指定采样数量
   ```

   命令会生成一个名为 `data/wiki/subsampled_positive_{num_samples}_urls.txt` 的文本文件，其中包含了指定数量个待下载的URL。

   

2. **创建并进入 `tmux` 会话**:

   为下载任务创建一个名为 `download` 的新会话。

   ```bash
   tmux new -s download_wiki_pages
   ```

   

3. **在 `tmux` 会话中执行下载**:

   

   在弹出的新`tmux`窗口中，运行下载连接的bash脚本

   ```bash
   chmod +x scripts/download_wiki_pages.sh
   scripts/download_wiki_pages.sh
   # <ctrl + b> + d 退出会话
   ```

   

   如果需要重新接入会话，运行以下命令

   ```bash
   # 重新接入会话
   tmux attach -t download_wiki_pages
   ```

   

   下载完成后，杀死会话

   ```bash
   # 杀死会话
   tmux kill-session -t download_wiki_pages
   ```

​	

**B. 准备Common Crawl页面 (用于负样本和管道测试)**

`sample_cc_paths.py` 脚本用于从Common Crawl的路径清单中抽样，由于抽样脚本的随机种子固定，脚本支持**增量下载**，以避免多次下载时下载大量重复的wet文件。

1.  **首次抽样**:
    ```bash
    # 从 wet.paths.gz 清单中抽样20个WET文件链接，并生成下载脚本 download_cc_batch_1.sh
    # 默认只下载WET文件
    python sample_cc_path.py data/cc_path/wet.paths.gz -n 100 --output-script scripts/download_cc_batch_1.sh
    ```
2.  **后续增量抽样**:
    如果需要更多不重复的样本，可以使用 `--skip` 参数。
    
    ```bash
    # 在已抽样20个的基础上，再抽样100个全新的文件链接，并同时下载对应的WARC文件
    python -m cs336_data.sample_cc_paths data/manifests/wet.paths.gz -n 100 --skip 20 --download-warc --output-script scripts/download_cc_batch_2.sh
    ```
3.  **执行下载**:
    对每个生成的脚本执行后台下载任务。
    ```bash
    chmod +x scripts/download_cc_batch_1.sh
    nohup scripts/download_cc_batch_1.sh > cc_download_1.log 2>&1 &
    ```

### **步骤二：构建与训练质量分类器**

1.  **构建训练集**:
    此脚本会根据“精英 vs. 良好”策略，从指定的WARC文件与wiki外部页面文件中采样并生成 `.train` 和 `.valid` 数据集。
    
    ```bash
    # 脚本会寻找data/wiki和data/crawls下的wiki和warc数据源，需要根据具体路径修改脚本配置
    chmod +x scripts/build_fasttext_dataset.sh
    scripts/build_fasttext_dataset.sh
    ```
2.  **训练模型**:
    使用YAML配置文件来训练分类器。
    ```bash
    python -m cs336_data.train_quality_classifier --config classifier_config.yaml
    ```

### **步骤三：运行完整数据过滤流水线**

在质量分类器训练完成后，执行主流水线脚本来处理下载好的**WET**文件。
```bash
# 脚本将处理 data/crawls/ 目录下的所有WET文件
python -m cs336_data.pipeline --config configs/{config_name}.yaml
```

## 6. 环境设置与安装

1.  **使用 `uv` 创建并同步虚拟环境:**
    ```bash
    uv venv
    source .venv/bin/activate
    
    uv sync # 较新的GPU，如Blackwell架构可能不支持官方给出的torch版本，需要手动升级torch与相关依赖
    
    uv pip install xxhash  # 精确行去除使用了xxhash中的哈希函数，以最大化效率
    ```

    
    
2.  下载所需数据

    ```bash
    # 本项目所需的所有外部数据（CC样本、预训练分类器、维基百科URL列表）均可通过一个脚本下载。
    chmod +x scripts/download_requirings.sh
    scripts/data_downloading.sh
    ```
    

## 7. 关键依赖

*   `fastwarc`: 用于高效地读取WARC/WET文件。
*   `fasttext`: 用于训练和运行文本分类器。
*   `numpy`: 用于数值计算，尤其是在MinHash签名中。
*   `mmh3`: 用于高性能的非加密哈希算法 (MurmurHash3)。
*   `nltk`: 用于 `ngrams` 等文本处理工具。
*   `tqdm`: 用于生成友好的进度条。
*   `pyyaml`: 用于解析YAML配置文件。