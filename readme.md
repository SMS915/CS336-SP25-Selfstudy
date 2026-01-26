# **R1-Lite & LLM-Sys: High-Performance Reasoning & Data Pipeline**

*An Extended Implementation based on Stanford CS336 (Spring 2025)*

![Course Status](https://img.shields.io/badge/Course-CS336_Spring_2025-cardinal)
![Language](https://img.shields.io/badge/Python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/PyTorch-2.x-orange)

本仓库包含我在学习斯坦福 CS336 课程中的核心实现。

不仅仅是完成作业，本项目旨在扩展课程内容。在完成标准教学要求的基础上，重点进行了系统的性能优化、关键架构的**消融实验**，并在消费级显卡上成功复现了 **GRPO** 与其部分变体。

**课程网站：** [https://stanford-cs336.github.io/spring2025/](https://stanford-cs336.github.io/spring2025/)



## 核心亮点

### 1. 复现 DeepSeek R1-Zero：单卡训练推理模型

> **关键词：** SFT, GRPO, 单卡优化, 数学推理
>
> 🔗 **[浏览项目详情](/assignment5-alignment)**

在**单张 RTX 4090 (48GB)** 上成功复现了 **DeepSeek R1** 的 SFT冷启动 + RL 的两阶段后训练流程，将 Qwen-2.5-Math-1.5B 模型的数学推理能力（MATH-500 Pass@1）从 2.8% 提升至 **60.6%**, **Pass@64**提升至**93.6%**，与官方模型的93.8%近乎持平。

**核心贡献与洞察：**

- **GRPO 算法实现：**摒弃了传统的 Value Model， 实现了**组相对策略优化 (GRPO)**，大幅节省了显存开销，并比对了**"Dr. GRPO"** 改进策略（去除 Advantage 标准化和响应长度偏差），

- **“白金数据”定律：** 发现 **3,000 条高质量/格式清洗数据** 的训练效果远超 25,000 条含噪数据。

- **思维链 (CoT) 演化观测：** 观测并记录了模型思维链长度的“倒 U 型”进化过程——从Baseline的直觉式解答，到SFT“机械模仿”导致的冗长重复，最终通过RL收敛为“内化推理”的高效精简。

- **数据泄露观测**：观测到基座模型在AIME24和AIME25数据集上的反常表现，结合具体推理链证实了模型预训练阶段存在数据泄露现象。

  

### 2. 工业级数据流水线：高并发、去重与质量过滤

> **关键词：** 数据工程, 分布式系统, MinHash LSH
>
> 🔗 **[浏览项目详情](/assignment4-data)**

构建了一个鲁棒的高并发数据处理流水线，旨在从大量原始 Common Crawl WET 文件中清洗出高质量的预训练语料。

**核心贡献与洞察：**

- **高并发架构设计：** 基于 multiprocessing 实现全链路并行，引入 **惰性加载 (Lazy Loading)** 和 Linux **写时复制 (COW)** 机制，有效解决了 N-gram 生成阶段的内存爆炸 (OOM) 问题。

- **大规模模糊去重：** 实现了 **MinHash + LSH (局部敏感哈希)** 算法，通过调整 Banding 参数与丢弃超出预设阈值大小的桶解决了“候选对爆炸”问题，实现了高效的文档级去重。

- **“精英 vs. 良好”分类器：** 训练了一个以维基百科引用页为“黄金标准”的 FastText 分类器，成功过滤掉了 83% “及格但平庸”的网页内容。

- **结果验证：** 经本流水线清洗的数据集 (Self-Dataset) 在 C4 验证集上的 Loss 和 Perplexity **均优于 OpenWebText 基线**，证明了“廉价过滤先行，昂贵处理置后”策略的有效性。

  

### 3. 从零构建 Transformer：架构演进与 BPE 高度优化

> **关键词：** 架构分析, 高性能预处理, BPE 算法优化
>
> 🔗 **[浏览项目详情](/assignment1-basics)**

使用 PyTorch 原生组件（不依赖高级封装）构建了 Decoder-only Transformer，并进行了深度的架构演进分析。

**核心贡献与洞察：**

- **BPE 分词器工程级重构：** 在朴素BPE分词器实现的基础逻辑上，基于 **倒排索引**、**最小堆** 和 **多进程并行** 重写了 BPE 算法，实现了从“玩具级”到“工业级”的跨越：
  - **训练效率 (Training):** 在 **10GB+ 全量 OpenWebText** 语料上，仅耗时 **8分30秒** 即完成 32k 词表的构建（相比之下，朴素算法处理 100MB 语料需 50分钟，在大规模数据上实现了数量级的性能飞跃）。
  - **编码吞吐 (Encoding):** 在 15 vCPU 环境下，对 **27亿 Token (11GB)** 的文本实现了 **15.5M Tokens/s** 的超高吞吐量。
  
- **架构演进消融实验：**深度对比 RoPE vs. Sinusoidal、SwiGLU vs. ReLU、RMSNorm vs. LayerNorm，验证了现代架构（Llama-style）的收敛优势。

  

## 项目架构

虽然底层基于课程框架，但本项目已重构为三个核心独立子系统：

*   **Core / Transformer-From-Scratch** (`./assignment1-basics`):
    *   手写 Decoder-only Transformer (RoPE, RMSNorm, SwiGLU)
    *   高性能 BPE 分词器 
*   **Data / Pipeline** (`./assignment4-data`):
    *   TB级 Common Crawl 处理流水线
    *   MinHash LSH 去重与质量过滤
*   **Alignment / R1-Lite-Reasoning** (`./assignment5-alignment`):
    *   DeepSeek-R1 复现 (SFT + GRPO)
    *   Math Reasoning 强化学习与评估

---

## 设置与使用

本项目使用 `uv` 进行现代化的 Python 依赖管理。

```bash
# 1. 安装环境与依赖
pip install uv

# 在对应文件夹同步环境与运行特定作业的测试
cd ./assignment1-data
uv sync
uv run pytest # 运行全部测试
uv run pytest -k tests/test_assignment_name.py # 运行单个测试
```