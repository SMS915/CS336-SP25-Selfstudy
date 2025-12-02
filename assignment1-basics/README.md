# CS336 作业 1：从零构建 Transformer 语言模型

本项目实现了一个 GPT 风格的 Decoder-only Transformer 语言模型，完全使用 PyTorch 原生构建（避免使用 `nn.Transformer` 模块）。该项目旨在探索大语言模型（LLM）架构的演变，通过对比 2017 年最初的 Transformer 设计与现代 Llama 风格的改进。

## 🚀 核心特性

- **分词器 (Tokenizer)**：在 TinyStories/OpenWebText 数据集上训练的自定义字节对编码（BPE）分词器。
- **模型架构**：
  - **现代组件**：RoPE（旋转位置编码）、RMSNorm、SwiGLU 激活函数。
  - **传统支持**：可配置支持绝对可学习位置编码/正余弦位置编码、LayerNorm 和 ReLU。
- **优化**：自定义实现的 `AdamW` 优化器和带有数值稳定性技巧的交叉熵损失函数（Cross-Entropy Loss）。
- **效率**：支持 `torch.compile` 和混合精度训练（AMP）。

## 🔬 消融实验与洞察

我们在 TinyStories 数据集上进行了广泛的消融实验，以理解现代架构选择的影响。

### 1. 训练稳定性：Post-Norm 与 Pre-Norm 对比

**假设**：与 Pre-Norm（GPT-2/Llama 采用）相比，Post-Norm 架构（原始 Transformer 采用）在早期训练阶段容易出现梯度不稳定的问题。

观察：

*上图展示了训练过程中的梯度范数。**橙色线条（原生/Post-Norm）在预热（warmup）结束后表现出剧烈的不稳定性和梯度尖峰，而蓝色线条（现代/Pre-Norm）*则保持稳定。

### 2. 位置编码：“冷启动”问题

**假设**：与可学习的绝对位置编码（Learned Absolute Embeddings）相比，RoPE 为相对位置提供了更好的归纳偏置（Inductive Bias），从而导致更快的收敛速度。

**观察**：在有限步数的训练场景中（例如 20k 步），**Learned Absolute PE** 明显落后于 **RoPE**。这是因为 Learned PE 初始化为随机噪声，需要消耗早期的训练步数来“学习”顺序的概念，而 RoPE 从第 0 步开始就注入了几何相对距离信息。

### 3. 权重绑定 (Weight Tying) 的影响

**分析**：将嵌入层（Embedding layer）和语言模型头（LM Head/输出层）的权重绑定可以显著减少参数量，但也可能限制表示能力。我们的实验表明... [在此添加您的发现]

## 🛠️ 使用方法 (Usage)

### 1. 环境设置 (Setup)

官方推荐使用 `uv` 进行环境管理以确保可复现性。

```
# 安装 uv (如果尚未安装)
pip install uv

# 使用 uv 运行代码 (自动管理依赖)
uv run Train.py --config config.yaml
```

### 2. 数据准备与解压 (Download Data)

在开始训练之前，需要下载 TinyStories 和 OpenWebText 数据集。请运行以下命令：

```
mkdir -p data
cd data

# 下载 TinyStories (用于快速实验)
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# 下载 OpenWebText 样本 (用于进阶实验)
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

### 3. 分词器训练 (Train BPE)



为了提高训练效率，需要将原始文本文件（`.txt`）转换为二进制格式（`.bin`），其中包含 token ID。

### 运行测试 (Run Unit Tests)

本项目包含完整的单元测试，用于验证各个组件（Attention, RMSNorm 等）的正确性：

```
uv run pytest
# 注意，windows环境下需要注释掉tests/test_tokenizer.py 的 import resource语句
# 另外，由于windows与linux的多进程运行方式不同，windows上由于多进程分发原因无法通过官方speed test, 在linux环境中测试时长可以在0.4-0.6秒左右。
```

### 训练与配置 (Training & Config)

模型可以通过 YAML 文件进行完全配置。详细参数说明请参见 `config.yaml`。

**训练一个现代 Llama 风格的模型：**

```
python Train.py --config config.yaml
```

**训练一个“复古”的 2017 风格模型（Post-Norm, ReLU, Absolute PE）：**

```
python Train.py --config vanilia_transformer_config.yaml
```

## 📂 项目结构

- `cs336_basics/`：包含模型实现的核心库。
  - `model.py`：Transformer 组件（Attention, MLP, RoPE 等）。
  - `optimizer.py`：AdamW 的手动实现。
- `Train.py`：包含检查点保存和 wandb 日志记录的主训练循环。
- `config.yaml`：超参数和架构切换的集中配置文件。

## 📜 参考文献

- Vaswani et al., "Attention Is All You Need" (2017)
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- Touvron et al., "LLaMA: Open and Efficient Foundation Language Models" (2023)
