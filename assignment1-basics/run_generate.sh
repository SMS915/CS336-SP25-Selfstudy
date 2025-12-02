#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:."

# 默认参数
CONFIG_PATH="configs/config.yaml"
DEFAULT_PROMPT="The future of artificial intelligence is"

# 获取命令行传入的 Prompt，如果没有则使用默认值
PROMPT="${1:-$DEFAULT_PROMPT}"

echo "========================================================"
echo "Generating text with Prompt: '$PROMPT'"
echo "Using Config: $CONFIG_PATH"
echo "========================================================"

# 运行生成脚本
# 注意：你需要确保你的 GenerateText.py 或 main 生成脚本支持这些参数
# 这里的 GenerateText 指向包含 main 函数的文件名 (不带 .py)
python cs336_basics/generate.py \
    --config "$CONFIG_PATH" \
    --prompt "$PROMPT" \
    --max_new_tokens 200 \
    --temperature 0.8 \
    --top_k 40

echo -e "\n========================================================"