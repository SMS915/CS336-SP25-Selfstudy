#!/bin/bash

# 配置部分
MODEL_BASE_DIR="checkpoints/grpo_v6_5e-6" # 你的 Checkpoint 根目录
OUTPUT_DIR="results/scan_grpo_"
LOG_FILE="${OUTPUT_DIR}/scan_grpo_summary.log"
mkdir -p $OUTPUT_DIR

# 要测试的 Checkpoints (根据你实际保存的文件夹名修改)
# 例如: checkpoint-50, checkpoint-100...
CHECKPOINT_NUMS=("50" "100" "150" "200" "250" "300" "350" "400")

echo "=================================================="
echo "开始 Dr.GRPO 训练动态扫描"
echo "=================================================="

for ckpt in "${CHECKPOINT_NUMS[@]}"; do
    MODEL_PATH="${MODEL_BASE_DIR}/checkpoint-step-${ckpt}"
    
    if [ ! -d "$MODEL_PATH" ]; then
        echo "⚠️ 跳过: 找不到 $MODEL_PATH"
        continue
    fi
    
    echo ""
    echo ">>> 正在评估: $ckpt" | tee -a "$LOG_FILE"
    
    # 1. 测 MATH-500 (Pass@1, Greedy) - 测基础能力/格式
    echo "    [Task 1] MATH-500..." | tee -a "$LOG_FILE"
    python cs336_alignment/evaluate_passk.py \
        --model_path "$MODEL_PATH" \
        --example_path "data/MATH/math500-test.jsonl" \
        --prompt_path "cs336_alignment/prompts/r1_zero.prompt" \
        --output_path "${OUTPUT_DIR}/math500_${ckpt}.jsonl" \
        --max_tokens 4096 \
        --temperature 0 \
        --pass_k 1 2>&1 | tee -a "$LOG_FILE"
        
    # 2. 测 AIME 2024 (Pass@8, Sampling) - 测智力上限
    # 选 Pass@8 是为了快，同时又能看到一点搜索能力
    echo "    [Task 2] AIME 2024 Pass@8..." | tee -a "$LOG_FILE"
    python cs336_alignment/evaluate_passk.py \
        --model_path "$MODEL_PATH" \
        --example_path "data/AIME/aime2024_test.jsonl" \
        --prompt_path "cs336_alignment/prompts/r1_zero.prompt" \
        --output_path "${OUTPUT_DIR}/aime2024_${ckpt}.jsonl" \
        --max_tokens 4096 \
        --temperature 0.6 \
        --pass_k 8 2>&1 | tee -a "$LOG_FILE"

    echo ">>> $ckpt 完成" | tee -a "$LOG_FILE"
done

echo "==================================================" | tee -a "$LOG_FILE"
echo "扫描结束。请查看 $OUTPUT_DIR" | tee -a "$LOG_FILE"