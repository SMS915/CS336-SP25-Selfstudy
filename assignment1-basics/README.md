[![中文](https://img.shields.io/badge/lang-中文-red.svg)](README_zh-CN.md) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-orange)![License](https://img.shields.io/badge/license-MIT-green) ![Course](https://img.shields.io/badge/Course-Stanford_CS336-cardinal)

# CS336 Assignment 1: Building a Transformer Language Model from Scratch

This project implements a GPT-style Decoder-only Transformer language model, built using native PyTorch (avoiding wrapper modules like `nn.Transformer`). The project aims to explore the evolution of Large Language Model (LLM) architectures by comparing the original 2017 Transformer design with modern Llama-style improvements.

## Core Features

- **Tokenizer**：Custom Byte Pair Encoding (BPE) tokenizer trained on TinyStories/OpenWebText datasets.

- **Model Architecture**：
  
  - **Modern Components**：RoPE (Rotary Positional Embeddings), RMSNorm, and SwiGLU activation function.
  - **Legacy Support**：Configurable support for absolute learnable/sinusoidal positional embeddings, LayerNorm, and ReLU.
  
- **Optimization**：Custom implementation of the `AdamW` optimizer and Cross-Entropy Loss with numerical stability techniques.

- **Efficiency**：Supports `torch.compile` and Automatic Mixed Precision (AMP) training.

  

## Ablation Studies & Insights

Several ablation experiments were conducted on the OpenWebText dataset based on the Transformer architecture, aiming to understand the impact of modern architectural choices.

###  Experimental Setup

To ensure fair comparison, all ablation experiments were conducted based on the following baseline hyperparameters (unless otherwise stated in a specific experiment):

- **Dataset**: OpenWebText (Subset)

- **Model Scale**: ~16M Parameters

  - $d_{model} = 256$, $n_{layers} = 4$, $n_{heads} = 4$

- **Training Config**:

  - Context Length: 256 (for rapid iteration) / 1024 (for long-sequence comparison)

  - Max Steps: 40,000

  - Batch Size: 64

  - Weight_Tying: True

  - Max Learning Rate: 3e-4

  - Min Learning Rate:  3e-5
  
    

### Architectural Evolution Analysis

Since its proposal in 2017, the Transformer architecture has undergone significant evolution from the original design (Vanilla) to the modern mainstream (Modern, e.g., Llama). To visually demonstrate the value of this evolution, we first compared the performance of the fully realized "Modern Architecture" against the "Original Architecture" at a small parameter scale:

![ablation_modern&vanilia_valloss](asset/ablation_modern&vanilla_val_loss.png)

![ablation_modern&vanilia_gradnorm](asset/ablation_modern&vanilla_gradnorm.png)

<p align="center">(The figures above compare the Validation Loss and Gradient Norm of the Modern architecture vs. the Vanilla architecture. It can be seen that the Modern architecture has an overwhelming advantage in convergence speed and training stability.)<p>




This evolution is not accidental but addresses two core pain points: **Training Stability** and **Model Expressiveness**. The following experiments aim to deconstruct this process:

1. **The Stability Trade-off: Post-Norm vs. Pre-Norm** 

   The original Transformer used a **Post-Norm** structure (`Norm(x + attn(x))`), which theoretically preserved stronger gradient flow but was prone to gradient explosion or vanishing during the initialization phase of deep networks, making training extremely sensitive to hyperparameters (like learning rate, Warmup). Modern architectures have universally shifted to **Pre-Norm** (`x + attn(Norm(x))`). Combined with **RMSNorm**, placing normalization within the residual branch creates a "gradient superhighway," significantly improving stability in the early stages of training.

2. **Evolution of Positional Awareness: Absolute vs. Relative** 

   The shift from **Absolute Positional Embeddings** (Sinusoidal/Learned) to **Rotary Positional Embeddings (RoPE)** is another major leap. Absolute PE forces positional information onto the Token Embedding, suffering from "cold start" issues and difficulty in capturing long-distance relative relationships. RoPE injects relative positional information into the Attention mechanism via rotation operations, endowing the model with stronger length extrapolation capabilities and faster convergence speeds.

3. **Efficacy of Activation Functions** 

   The transition from **ReLU** to **SwiGLU** introduces a gating mechanism. This increases the richness of non-linear transformations and, when combined with adjusted hidden layer dimensions, yields better convergence with only a small increase in parameter count.

   

### 1. Training Stability: Post-Norm vs. Pre-Norm

**Hypothesis**：Compared to Pre-Norm (used by GPT-2/Llama), Post-Norm architecture (used by the original Transformer) is prone to gradient instability during the early training stages.

![abletion_norm_train_loss](asset/ablation_norm_train_loss.png)

![abletion_norm_gradnorm](asset/ablation_norm_gradnorm.png)

<center>The figures above show the loss and gradient norm during training.</center>

**Observation**：The orange line **(Native/Post-Norm)** shows severe instability and gradient spikes after warmup ends, while the blue line **(Modern/Pre-Norm)** remains stable. This confirms the instability of Post-Norm in the early training of deep networks.



### 2. Positional Encoding Strategy Analysis: RoPE, NoPE, Sinusoidal & Learned

**Background**：Based on the original Transformer (Post-Norm) and keeping other variables constant, we compared the effects of four positional encoding configurations: RoPE (Rotary), NoPE (No Positional Encoding), Sinusoidal (Absolute), and Learned (Absolute).

![ablation_pe_trainloss](asset/ablation_pe_train_loss.png)

![ablation_pe_valloss](asset/ablation_pe_val_loss.png)

![ablation_pe_gradnorm](asset/ablation_pe_gradnorm.png)

<p align="center">From top to bottom: Training Loss, Validation Loss, Gradient Norm.</p>



**Observation**： 

**A. Loss Curve Analysis (Training & Validation Loss)** 

Observing the validation Loss curve, we see distinct performance stratification:

1. **RoPE (Green)**: Fastest convergence, lowest final Loss. Proves the effectiveness of injecting relative positional information into the attention mechanism.

2. **NoPE (Dark Blue)**: Surprisingly, in the current setting (short context length of 256), the model without any explicit positional encoding outperforms traditional absolute PEs. This suggests the Causal Mask itself leaks enough implicit positional information for the model to learn.

3. **Sinusoidal (Orange)**: As a baseline, performs averagely.

4. **Learned Absolute (Light Blue)**: Worst performance, slowest convergence. This reflects the "cold start" problem—the model must spend early training steps learning the sequential relationship of position vectors from scratch, slowing down semantic learning.

   

**B. Gradient Norm Analysis** 

Observing the gradient norm curves reveals a polarization:

1. **Unstable Group (Additive PE)**: 

   **Sinusoidal (Orange)** and **Learned (Light Blue)** both use additive positional encoding (`x + pos_emb(x)`). They both show severe gradient oscillations and spikes in early training (end of Warmup). This suggests that directly **adding** position vectors to word embeddings significantly increases the variance of the residual stream in Post-Norm architectures, leading to optimization difficulties.

2. **Stable Group (Non-Additive)：** 

   **RoPE (Green)** and **NoPE (Dark Blue)** have extremely smooth gradient norms. RoPE uses rotation (multiplicative) operations without changing vector magnitude; NoPE introduces no extra positional vector interference at all.

   

**Conclusion**：RoPE combines "training stability of multiplicative operations" with "explicit relative geometric information", thus achieving the best comprehensive performance in this comparison.



### 3. Impact of Weight Tying

**Analysis**：Tying the weights of the embedding layer and the language model head (Output layer) can significantly reduce the number of parameters, but may limit representation capability. 

Our experiments show ...



### 4. Impact of Activation Functions

Based on the original Transformer, keeping parameter count and other factors constant, we compared the impact of different activation functions on performance.

**Hypothesis**：SwiGLU (SiLU + Gating), compared to traditional ReLU, increases non-linear expressive power by introducing a Gated Linear Unit, usually leading to better convergence.

![ablation_activation_train_loss](asset/ablation_activation_train_loss.png)

![ablation_activation_val_loss](asset/ablation_activation_val_loss.png)

![ablation_activation_grad_norm](asset/ablation_activation_grad_norm.png)

<p align="center">The figure above compares Training & Validation Loss for SwiGLU (Blue) vs. Original ReLU (Orange)</p>

**Observation**：

1. **Convergence Advantage**：The Loss curves show that **SwiGLU (Blue)** has faster convergence and lower final Loss compared to **ReLU (Orange)**. This verifies the effectiveness of the gating mechanism in improving model performance.

2. **Scale Effect**: Notably, while SwiGLU performs better, at the current small parameter scale (~17M), the magnitude of improvement is smaller compared to improvements from positional encoding (RoPE).
   
   - This may be because, in small models, parameter limitations make it difficult to fully utilize the extra expressive power brought by SwiGLU. Typically, in larger models (like Llama-70B), the advantage of SwiGLU becomes more significant.
   
     

## Usage

### 1. Environment Setup

It is officially recommended to use `uv` for environment management to ensure reproducibility.

```bash
# Install uv (if not already installed)
pip install uv

# Sync environment from uv.lock. If using a Blackwell GPU, you need to manually install the latest torch and corresponding libraries.
uv sync
```

### 2. Run Unit Tests

Unit tests were provided officially by Stanford to verify the correctness of each component (Attention, RMSNorm, etc.). Adjust the interfaces in `adapter.py` and run the command below to test the correctness of different modules.

In a Windows environment, you need to comment out the `import resource` statement in `tests/test_tokenizer.py`. Additionally, due to differences in default file reading encodings, the tokenizer may fail specific tests on Windows, whereas they pass successfully in Linux environments.

Furthermore, due to different multiprocessing implementations between Windows and Linux, the official 1.5s speed test cannot be passed on Windows due to multiprocessing dispatch overhead. In a Linux environment, the `speed_test` takes about 0.4-0.6 seconds. 

```bash
uv run pytest # Run all tests at once
uv run pytest -k test_{test_name}.py # Test a single module/function
```



### 3. Download Data

Before starting training, you need to download the TinyStories and OpenWebText datasets. Run the following commands:

```bash
mkdir -p data
cd data

# Download TinyStories (for fast experimentation)
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# Download OpenWebText sample (for advanced experimentation)
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```



### 4. Tokenizer Training (Train BPE)

Two versions of BPE were implemented: a Naive version and an Optimized version.

Environment: 15 vCPU Intel(R) Xeon(R) Platinum 8474C, using 14 processes, 80GB RAM, ubuntu22.04

#### Encoding

| **Dataset File**                 | **Total Tokens** | **Version**   | **Total Time (s)** | **Throughput (k tokens/s)** | **Speedup (vs Naive)** |
| -------------------------------- | ---------------- | ------------- | ------------------ | --------------------------- | ---------------------- |
| **TinyStoriesV2-GPT4-train.txt** | 547,725,817      | Naive         | 219.56             | 2,494.62                    | -                      |
|                                  |                  | **Optimized** | **31.42**          | **17,430.98**               | **~6.99x**             |
|                                  |                  |               |                    |                             |                        |
| **TinyStoriesV2-GPT4-valid.txt** | 5,532,654        | Naive         | 4.84               | 1,142.19                    | -                      |
|                                  |                  | **Optimized** | **3.12**           | **1,773.08**                | **~1.55x**             |
|                                  |                  |               |                    |                             |                        |
| **owt_train.txt**                | 2,704,046,552    | Naive         | N/A                | N/A                         | -                      |
|                                  |                  | **Optimized** | **174.79**         | **15,470.62**               | -                      |
|                                  |                  |               |                    |                             |                        |
| **owt_valid.txt**                | 65,853,560       | Naive         | N/A                | N/A                         | -                      |
|                                  |                  | **Optimized** | **11.66**          | **5,649.66**                | -                      |

#### Training

##### Naive Version Test Results

upcoming...



##### Optimized Version Test Results

Training Corpus: TinyStoriesV2-GPT4-train.txt (59,933 distinct words/pre-tokenized fragments)

Vocab Size: 10000 ,

Runtime: **13s**, Peak Memory: **6.5GB**  vs  (Official Expectation <30min, <30GB RAM)



Training Corpus: owt_train.txt (6,601,892 distinct words/pre-tokenized fragments)

Vocab Size: 32000

Runtime: **8m 31s**, Peak Memory: **26.13GB**  vs  (Official Expectation <12hours , <100GB RAM)



How to train: Specify the corpus file and vocab size in `run_train_bpe.sh`, then run the bash file.

```bash
chmod +x ./run_train_bpe.sh # Grant execution permissions
./run_train_bpe.sh
```



### 5. Data Preprocessing

To improve training efficiency, raw text files (`.txt`) need to be converted into binary format (`.bin`) containing the sequence of token IDs encoded by the tokenizer.

Run the script below to preprocess data using multiprocessing. Specify the tokenizer read file prefix and process count in the bash script, and modify the list of files to encode in `preprocess_training_data.py`.

```
./preprocess_corpus.sh
```



### 6. Training & Config

The model is fully configurable via YAML files stored in the `configs` folder. For detailed parameter descriptions, see `base_modern.yaml`.

**Train a modern Llama-style model (PreNorm, RMSNorm, RoPE):**

```
python Train.py --config base_modern.yaml
```

**Train a "Retro" 2017-style model (Post-Norm, ReLU, Sinusoidal PE):**

```
python Train.py --config base_vanilla.yaml
```

**Run ablation experiments:**

```
python Train.py --config experiments/ablation_{}.yaml
```



## Project Structure

```text
.
├── asset/                                          # Experiment records & visualizations (WandB curves, etc.)
├── configs/                                        # Model & Training configurations
│   ├── base_modern.yaml                            # Modern architecture baseline (Llama-style, RoPE, SwiGLU)
│   ├── base_vanilla.yaml                           # Original architecture config (Post-Norm, Sinusoidal PE, ReLU)
│   └── experiments/
│       └── ablation_{ablation_name}.yaml           # Independent configs for ablation studies
├── cs336_basics/                                   # Core source code library
│   ├── model.py                                    # Core Transformer components (Attention, RoPE, RMSNorm)
│   ├── optimizer.py                                # Hand-written AdamW optimizer
│   ├── utils.py                                    # Basic utilities (Softmax, CrossEntropy, LR Schedule)
│   ├── checkpointing.py                            # Model checkpoint saving & loading
│   ├── train.py                                    # Main training loop (Integrated with WandB)
│   ├── bpe_baseline.py                             # BPE Tokenizer (Naive implementation)
│   ├── bpe_fast.py                                 # BPE Tokenizer (Optimized implementation)
│   ├── profile_bpe.py                              # Efficiency profiling for original or optimized tokenizers
│   ├── train_bpe.py                                # BPE training entry code
│   ├── data.py                                     # Single-process Dataloader (Memory Mapping)
│   ├── fast_data.py                                # Multi-process Dataloader
│   ├── preprocess_training_data.py                 # Corpus preprocessing & binarization script
│   ├── pretokenization_example.py                  # Official code for multi-process text chunk boundary handling
│   ├── generation_utils.py                         # Core text generation logic (Top-k, Temp)
│   └── generate.py                                 # Inference generation entry code
├── tests/                                          # Unit tests directory
│   ├── adapter.py                                  # Official test interface adapter
│   └── ...                                         # Official test cases
├── cs336_spring2025_assignment1_basics.pdf         # Official Assignment Handout (English)
├── [翻译]cs336_spring2025_assignment1_basics.pdf    # Assignment Handout (Chinese Translation)
├── run_train.sh                                    # Launch script: Model Training
├── run_generate.sh                                 # Launch script: Text Generation
├── run_train_bpe.sh                                # Launch script: Tokenizer Training
└── uv.lock                                         # Python environment lock file (not adapted for newer GPUs like Blackwell)
```



## References

- Vaswani et al., "Attention Is All You Need" (2017)
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- Touvron et al., "LLaMA: Open and Efficient Foundation Language Models" (2023)
