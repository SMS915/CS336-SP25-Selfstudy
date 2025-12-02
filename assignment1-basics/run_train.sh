#!/bin/bash

# --- 配置区域 ---
# 指定你要使用的配置文件路径
CONFIG_PATH="configs/vanilla_based_config.yaml"

# 指定使用的 GPU
export CUDA_VISIBLE_DEVICES=0

# 设置 PYTHONPATH 为当前目录，确保 python 能找到 cs336_basics 包
export PYTHONPATH="${PYTHONPATH}:."

# --- 执行训练 ---
echo "========================================================"
echo "Starting Training with config: $CONFIG_PATH"
echo "Device: CUDA $CUDA_VISIBLE_DEVICES"
echo "========================================================"

python cs336_basics/Train.py --config "$CONFIG_PATH"

# 如果训练成功，打印提示
if [ $? -eq 0 ]; then
    echo "✅ 训练成功"
else
    echo "❌ 训练失败"
fi