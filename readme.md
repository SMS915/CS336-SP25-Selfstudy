# CS336：语言模型 (2025春季)

![Course Status](https://img.shields.io/badge/Course-CS336_Spring_2025-cardinal)
![Language](https://img.shields.io/badge/Python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/PyTorch-2.x-orange)

本仓库包含我在自学 **Stanford CS336: Language Modeling** 课程中的作业实现代码。该课程涵盖了大型语言模型 (LLM) 的全生命周期，包括从零构建架构、系统优化、数据工程、扩展定律 (Scaling Laws) 以及后训练对齐 (Post-training Alignment)。

**课程网站：** [https://stanford-cs336.github.io/spring2025/](https://stanford-cs336.github.io/spring2025/)

## 项目进度

| 作业                                           | 主题                          |   状态   |
| :--------------------------------------------- | :---------------------------- | :------: |
| **Assignment 1** (整理中)                      | 基础：构建 Transformer LM     | &#x2705; |
| **Assignment 2**                               | 系统：优化与并行计算          | &#9744;  |
| **Assignment 3 **(无法完成，需要斯坦福专属API) | 扩展：Scaling Laws 与计算预算 | &cross;  |
| **Assignment 4** (整理中)                      | 数据：过滤与去重              | &#x2705; |
| **Assignment 5 **(整理中)                      | 对齐：推理与强化学习 (GRPO)   | &#x2705; |

---

## 作业详情

### 1. Assignment 1: Basics (构建 Transformer)
**目标：** 完全使用 PyTorch 从零开始构建一个现代的 Decoder-only Transformer 语言模型（不使用 `nn.Linear` 或 `nn.Transformer` 等高级层）

* **核心组件：**
    * 在 TinyStories 数据集上实现 **BPE 分词器**（训练与编码/解码）。
    * 构建现代 **Transformer 架构**组件：Embeddings, RMSNorm, SwiGLU 激活函数, RoPE (旋转位置编码), 以及因果多头自注意力机制 (Causal Multi-Head Self-Attention) 
    * 实现训练基础设施：交叉熵损失函数, **AdamW 优化器**, 以及学习率调度器。
    * **文本生成：** 实现解码策略，包括温度缩放 (Temperature scaling) 和 Top-p (Nucleus) 采样。

### 2. Assignment 2: Systems (优化与并行)
**目标：** 优化 Transformer 的单 GPU 性能并将训练扩展到多 GPU 环境。

* **核心组件：**
    * **性能分析：** 使用 Python 工具和 **NVIDIA Nsight Systems** 对前向/后向传播进行基准测试，分析计算和内存瓶颈。
    * **FlashAttention-2：** 实现自定义 **Triton kernel** 以实现内存高效的注意力机制（分块 tiling, 在线 softmax, 重计算）。
    * **分布式训练：** 从零实现 **分布式数据并行 (DDP)**，处理梯度同步、分桶通信 (bucketed communication) 以及计算/通信重叠。
    * **优化器分片：** 实现优化器状态分区 (Optimizer Sharding) 以减少跨节点的内存冗余。

### 3. Assignment 3: Scaling (扩展定律)
**目标：** 理解并应用经验扩展定律 (Scaling Laws)，在计算约束下预测最佳模型配置。

* **核心组件：**
    * IsoFLOPs 曲线：复现 Chinchilla 扩展定律方法，分析模型大小与数据规模之间的权衡。
    * **外推预测：** 拟合小规模实验数据的幂律曲线，以预测更大计算预算（$10^{23}$ - $10^{24}$ FLOPs）下的损失。
    * **最优分配：** 使用 Training API 搜索超参数空间，确定特定预算（$10^{19}$ FLOPs）下的计算最优模型大小和超参数。

### 4. Assignment 4: Data (工程与过滤)
**目标：** 构建健壮的数据管道，将原始网页抓取数据（Common Crawl）处理为高质量训练数据。

* **核心组件：**
    * **提取：** 使用 Resiliparse 将原始 **WARC/WET** 文件从 HTML 转换为纯文本 。
    * **过滤管道：** 实现语言识别 (FastText)、**PII 掩码**（邮箱、IP、电话）以及有害内容检测 (NSFW/Toxic 分类器) 。
    * **质量控制：** 应用Gopher启发式规则并训练基于模型的质量分类器。
    * **去重：** 实现 **MinHash + LSH** (局部敏感哈希) 以进行大规模模糊文档级去重。

### 5. Assignment 5: Alignment (推理与 RL)
**目标：** 使用监督微调 (SFT) 和强化学习 (RL) 训练模型执行数学推理任务。

* **核心组件：**
    * **评估：** 使用 zero-shot CoT 提示在 MATH 数据集上对 Qwen 2.5 Math 1.5B 进行基准测试。
    * **SFT：** 使用 DeepSeek R1 生成的推理轨迹 (reasoning traces) 对模型进行微调。
    * **专家迭代 (Expert Iteration)：** 实现自举 (bootstrapping)，通过学习模型自身生成的正确解来提升推理能力。
    * **GRPO：** 实现 **组相对策略优化 (Group Relative Policy Optimization)**，这是一种用于推理的 PPO 变体，使用基于组的优势估计而无需价值函数网络 。

---

## 设置与使用

本项目使用 `uv` 进行依赖管理。

```bash
# 安装依赖
uv sync

# 运行特定作业的测试
uv run pytest tests/test_assignment_name.py