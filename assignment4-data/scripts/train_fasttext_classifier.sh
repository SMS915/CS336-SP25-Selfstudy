#!/bin/bash
# -----------------------------------------------------------------------------
# CS336 作业4 - 质量分类器训练启动脚本
#
# 本脚本是 cs336_data/train_quality_classifier.py 的一个便捷封装。
# 通过修改开头的 "配置变量" 部分，可以轻松地运行不同的训练实验。
# -----------------------------------------------------------------------------

set -e # 如果任何命令失败，脚本将立即退出

# --- 1. 配置变量 (在此处修改你的训练实验设置) ---

# --- 核心配置文件 ---
# 指定用于本次训练的YAML配置文件路径
CONFIG_FILE="classifier_config.yaml"

# --- 命令行覆盖参数 (可选) ---
# 如果你想临时覆盖YAML文件中的设置，请在这里指定。
# 如果留空 ("")，则会使用YAML文件中的默认值。

# 覆盖训练轮数
EPOCHS="150"
# 覆盖训练文件路径
# INPUT_FILE="data/classifiers_dataset/my_special_dataset.train"
INPUT_FILE=""
# 覆盖模型输出目录
# OUTPUT_DIR="data/my_classifiers/special_experiment"
OUTPUT_DIR=""


# --- 2. 脚本执行逻辑 (通常无需修改) ---

echo "=================================================="
echo "          开始质量分类器训练"
echo "=================================================="
echo "实验配置:"
echo "  - 配置文件:    $CONFIG_FILE"
echo "  - Epochs (覆盖): ${EPOCHS:-使用配置文件默认值}"
echo "  - 输入文件 (覆盖): ${INPUT_FILE:-使用配置文件默认值}"
echo "  - 输出目录 (覆盖): ${OUTPUT_DIR:-使用配置文件默认值}"
echo "--------------------------------------------------"

# --- 动态构建命令行参数 ---
# 定义基础命令
CMD="python -m cs336_data.train_quality_classifier --config \"$CONFIG_FILE\""

# 根据条件添加可选的覆盖参数
if [ -n "$EPOCHS" ]; then
    CMD="$CMD --epoch $EPOCHS"
fi

if [ -n "$INPUT_FILE" ]; then
    CMD="$CMD --input_file \"$INPUT_FILE\""
fi

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir \"$OUTPUT_DIR\""
fi

# 打印将要执行的完整命令，便于调试
echo "即将执行命令:"
echo "$CMD"
echo "--------------------------------------------------"

# 执行命令
eval $CMD

echo "=================================================="
echo "          训练流程执行完毕！"
echo "=================================================="