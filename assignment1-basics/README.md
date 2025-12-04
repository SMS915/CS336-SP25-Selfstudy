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

<p align="center">Figure: Comparison between Modern architecture and Vanilla architecture on Validation Loss and Gradient Norm.</p><p align="center">
It can be seen that the Modern architecture has an overwhelming advantage in convergence speed and training stability.
</p>



This evolution is not accidental but addresses two core pain points: **Training Stability** and **Model Expressiveness**. The following experiments aim to deconstruct this process:

1. **The Stability Trade-off: Post-Norm vs. Pre-Norm** 

   The original Transformer used a **Post-Norm** structure (`Norm(x + attn(x))`), which theoretically preserved stronger gradient flow but was prone to gradient explosion or vanishing during the initialization phase of deep networks, making training extremely sensitive to hyperparameters (like learning rate, Warmup). Modern architectures have universally shifted to **Pre-Norm** (`x + attn(Norm(x))`). Combined with **RMSNorm**, placing normalization within the residual branch creates a "gradient superhighway," significantly improving stability in the early stages of training.

2. **Evolution of Positional Awareness: Absolute vs. Relative** 

   The shift from **Absolute Positional Embeddings** (Sinusoidal/Learned) to **Rotary Positional Embeddings (RoPE)** is another major leap. Absolute PE forces positional information onto the Token Embedding, suffering from "cold start" issues and difficulty in capturing long-distance relative relationships. RoPE injects relative positional information into the Attention mechanism via rotation operations, endowing the model with stronger length extrapolation capabilities and faster convergence speeds.

3. **Efficacy of Activation Functions** 

   The transition from **ReLU** to **SwiGLU** introduces a gating mechanism. This increases the richness of non-linear transformations and, when combined with adjusted hidden layer dimensions, yields better convergence with only a small increase in parameter count.



___



### 1. Training Stability: Post-Norm vs. Pre-Norm

**Hypothesis**：Compared to Pre-Norm (used by GPT-2/Llama), Post-Norm architecture (used by the original Transformer) is prone to gradient instability during the early training stages.

![abletion_norm_train_loss](asset/ablation_norm_train_loss.png)

![abletion_norm_gradnorm](asset/ablation_norm_gradnorm.png)

<center>The figures above show the loss and gradient norm during training.</center>

**Observation**：The orange line **(Native/Post-Norm)** shows severe instability and gradient spikes after warmup ends, while the blue line **(Modern/Pre-Norm)** remains stable. 



**Conclusion**：This confirms the instability of Post-Norm in the early training of deep networks.



___



### 2. Positional Encoding Strategy Analysis: RoPE, NoPE, Sinusoidal & Learned

**Experimental Setup**：Based on the original Transformer (Post-Norm) and keeping other variables constant, we compared the effects of four positional encoding configurations: RoPE (Rotary), NoPE (No Positional Encoding), Sinusoidal (Absolute), and Learned (Absolute).

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



___



### 3. Impact of Weight Tying

**Experimental Setup:** Building upon the Vanilla (Post-Norm) architecture, this experiment compares two embedding strategies: **Tied** (yellow), where the input embedding and the output LM head share parameters, and **Untied** (red), where the input and output utilize two independent matrices. The Tied configuration contains 28.9M parameters, while the Untied configuration contains 16M.



![ablation_WeightTying_train_loss](D:/CSDIY/CS336/assignment1-basics/asset/ablation_WeightTying_train_loss.png)

![ablation_WeightTying_val_loss](D:/CSDIY/CS336/assignment1-basics/asset/ablation_WeightTying_val_loss.png)

![ablation_WeightTying_gradnorm](D:/CSDIY/CS336/assignment1-basics/asset/ablation_WeightTying_gradnorm.png)

**Observations:**

**1. Generalization Victory on Tied** 

As shown in the `val/loss` curve, the **Tied Model (Yellow)** achieves a significantly lower terminal loss (**4.67**) compared to the **Untied Model (Red)** (**4.79**). This demonstrates that weight tying effectively functions as a regularizer. By reducing the parameter count, it mitigates overfitting and enhances the model's performance on the validation set.

**2. The Cost of Optimization Stability** 

The `grad_norm` plot reveals a counter-intuitive phenomenon. The **Untied Model (Red)** exhibits a remarkably stable gradient norm with almost no fluctuation throughout the entire training process. In contrast, the **Tied Model (Yellow)** undergoes severe **Gradient Spikes** during the early training phase (Warmup), indicating significant optimization instability despite its superior final performance.



**Analysis:** 

Under the current experimental conditions—characterized by a small model scale and limited data diversity—the results exhibit a distinct trade-off: the **Untied configuration demonstrates superior gradient stability**, whereas the **Tied configuration yields better final performance**. We analyze this phenomenon through the lenses of gradient propagation and modular functional responsibilities.



**1. Gradient Propagation Perspective: Structural Decoupling and Buffering** 

In the **Untied** configuration, the input embedding layer and the output classification layer are structurally decoupled. During backpropagation, error gradients originating from the output prediction logits do not directly impinge upon the input representations. This physical isolation acts as a natural **optimization buffer**, granting the model superior stability during the early training phase. Conversely, in the **Tied** configuration, a single set of parameters must simultaneously handle divergent gradient flows from both the network's "head" (feature extraction) and "tail" (logit prediction). This direct feedback loop induces high variance and severe gradient spikes during the initial training stages.



**2. Functional Role Perspective: Friction in Multi-Objective Optimization** 

From a functional standpoint, the tied embedding matrix faces a **Dual-Role Constraint**: it must serve as the input layer to construct a robust, continuous semantic space while simultaneously acting as the output layer to provide high-resolution, discriminative classification boundaries. In the early stages of training, the gradient directions required to satisfy these two distinct objectives often exhibit orthogonal or even conflicting trends. This conflict results in significant **"Optimization Friction,"** which manifests as the observed gradient oscillations.



**3. Small-Scale Regime Analysis: The Dividend of Inductive Bias** 

In the current experimental setting characterized by a smaller model size and limited data diversity, the weight tying strategy—despite its initial instability—introduces a critical **Inductive Bias**: the assumption that the input semantic space and the output semantic space are isomorphic. This strong constraint acts as an effective regularizer, forcing the model to learn more compact and essential semantic representations within a restricted parameter space. Consequently, this translates into superior generalization capabilities within the current data scope.



**4. Large-Scale Scenario Extrapolation: The Necessity of Semantic Heterogeneity** 

However, when extrapolating to large-scale model training driven by massive, diverse corpora, subtle **"Semantic Deviations"** may emerge between the contextual semantics of the input and the predictive distributional semantics of the output. In such regimes, the **Untied** strategy is released from the isomorphic constraint and possesses greater parameter capacity to independently model this input-output asymmetry. Given abundant data, this higher degree of expressive freedom is expected to capture broader semantic nuances, potentially surpassing the performance ceiling of the tied strategy in long-run training.



**Conclusion**：

In summary, the decision to employ Weight Tying should not be viewed merely as a binary architectural preference, but as a strategic trade-off contingent upon the specific **"Data-Parameter Regime."** Fundamentally, Weight Tying serves as a **strong regularizer**. By enforcing isomorphism between input and output spaces, it sacrifices early-stage optimization stability in exchange for enhanced generalization in resource-constrained settings. Conversely, in resource-abundant regimes characteristic of large-scale pre-training, decoupling these weights (Untying) may be the necessary path to unlock the model's full expressive potential.



___



### 4. Impact of Activation Functions

**Experimental Setup**：In order to isolate the effect of activation functions, we evaluated the performance of SwiGLU (SiLU + gating mechanism) versus standard ReLU on both Modern and Vanilla architectures. We strictly controlled the parameter counts to ensure a fair comparison (~77M/16M for SwiGLU vs. ~76M/16M for ReLU).



**Hypothesis**：SwiGLU (SiLU + Gating), compared to traditional ReLU, increases non-linear expressive power by introducing a Gated Linear Unit, usually leading to better convergence.

![ablation_activation_train_loss](asset/ablation_activation_train_loss.png)

![ablation_activation_val_loss](asset/ablation_activation_val_loss.png)

![ablation_activation_grad_norm](asset/ablation_activation_gradnorm.png)

![ablation_activation_gradnorm_modern](asset/ablation_activation_gradnorm_modern.png)

<p align="center">The figure above compares Training & Validation Loss for SwiGLU (Blue) vs. Original ReLU (Orange)</p>

**Observations**：

1. **Performance Superiority**

   **Performance Improvement under Equal Parameters**: At nearly identical parameter scales, models equipped with **SwiGLU** (green/blue curves) consistently achieve significantly lower validation loss compared to **ReLU** models (orange/yellow curves).

   **Consistency of the Gap**: This performance advantage is particularly pronounced within the Modern architecture. **Modern + SwiGLU (green)** exhibits the steepest descent slope and the lowest convergence value, demonstrating that Gated Linear Units (GLU) possess higher **information encoding efficiency** under equivalent computational budgets.

   

2. **Gradient Stability Advantage**

   **Baseline Role of Architecture**: First, the `grad_norm` chart confirms that the Modern (Pre-Norm) architecture (green/orange) successfully eliminates the severe gradient spikes observed in the Vanilla (Post-Norm) architecture (blue/yellow) during the early stages of training. This validates the necessity of Pre-Norm for optimizing deep networks.

   

   **SwiGLU’s "Stabilizer" Effect**: More notably, even within the already stable Modern architecture, the gradient norm of **SwiGLU (green)** remains lower and smoother than that of **ReLU (orange)**.

   - While it is often assumed that introducing complex gating structures might increase optimization complexity, the experiment indicate that the multiplicative interactions in SwiGLU appear to create a smoother optimization landscape, rendering the model more robust during parameter updates.

   

**Conclusion**：

The experiments compellingly demonstrate that the superiority of SwiGLU stems from its intrinsic structure rather than an increase in parameter count (which was strictly controlled at ~76M vs. ~77M and 16M vs. 16M). SwiGLU not only enhances the model's **nonlinear expressive capability** through its feature gating mechanism but also unexpectedly yields **additional benefits in gradient stability**. This positions it as the indisputable activation function of choice for modern LLM architectures.



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
