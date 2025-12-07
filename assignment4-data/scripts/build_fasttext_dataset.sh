#!/bin/bash
# -----------------------------------------------------------------------------
# CS336 作业4 - 质量分类器数据集构建脚本
#
# 本脚本是 cs336_data/dataset_builder.py 的一个便捷封装。
# 通过修改开头的 "配置变量" 部分，可以灵活地生成不同策略和规模的数据集。
# -----------------------------------------------------------------------------

set -e # 如果任何命令失败，脚本将立即退出

# --- 1. 配置变量  ---

# --- 输入数据源路径 ---
WIKI_WARC_PATH="data/wiki/subsampled_positive_15000_pages.warc.gz"
CC_WARC_PATH="data/crawls/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"

# --- 输出配置 ---
OUTPUT_DIR="data/classifiers_dataset"
# 输出文件名的基础部分，脚本会自动添加策略和样本数等信息
OUTPUT_BASE_NAME="test_dataset"

# --- 核心策略与参数 ---
# 负样本策略: 
#   'filtered'   -> "精英 vs. 良好"
#   'unfiltered' -> "干净 vs. 混沌"
STRATEGY="filtered"

# 正样本最大数量 (留空 "" 表示不限制，即处理所有可用样本)
MAX_SAMPLES=""

# 是否将数据集分割为训练集和验证集 (true / false)
SPLIT_DATASET=true
# 训练集比例 (仅当 SPLIT_DATASET=true 时生效)
TRAIN_RATIO=0.9


# --- 2. 脚本执行逻辑 (通常无需修改) ---

echo "=================================================="
echo "          开始构建分类器数据集"
echo "=================================================="
echo "构建配置:"
echo "  - 正样本源 (Wiki): $WIKI_WARC_PATH"
echo "  - 负样本源 (CC):   $CC_WARC_PATH"
echo "  - 负样本策略:      $STRATEGY"
echo "  - 正样本上限:      ${MAX_SAMPLES:-无限制}"
echo "  - 分割数据集:      $SPLIT_DATASET"
if [ "$SPLIT_DATASET" = true ]; then
    echo "  - 训练集比例:      $TRAIN_RATIO"
fi
echo "--------------------------------------------------"

# --- 动态构建命令行参数 ---
# 定义基础命令
CMD="python -m cs336_data.dataset_builder \
    --wiki_path \"$WIKI_WARC_PATH\" \
    --cc_path \"$CC_WARC_PATH\" \
    --output_dir \"$OUTPUT_DIR\" \
    --output_base \"$OUTPUT_BASE_NAME\" \
    --strategy \"$STRATEGY\""

# 根据条件添加可选参数
if [ "$SPLIT_DATASET" = true ]; then
    CMD="$CMD --split --train_ratio $TRAIN_RATIO"
fi

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_sample $MAX_SAMPLES"
fi

# 打印将要执行的完整命令，便于调试
echo "即将执行命令:"
echo "$CMD"
echo "--------------------------------------------------"

# 执行命令
eval $CMD

echo "=================================================="
echo "          数据集构建完成！"
echo "=================================================="
echo "生成的数据集位于: $OUTPUT_DIR"