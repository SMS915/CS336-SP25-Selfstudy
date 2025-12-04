#!/bin/bash

# ================= 配置区域 =================
# Python 脚本的文件名

PYTHON_SCRIPT="cs336_basics/preprocess_training_data.py"

# 模型文件所在的目录 (根据你的 ls 输出)
MODEL_DIR="BPE_File"

# 默认使用的 CPU 核心数
WORKERS=8
# ===========================================

# 检查是否传入了前缀参数
if [ -z "$1" ]; then
    echo "错误: 未指定分词器前缀。"
    echo "用法: ./run_encoder.sh <prefix> [input_files...]"
    echo ""
    echo "可用前缀示例:"
    echo "  gpt2          -> 使用 gpt2_vocab.json"
    echo "  self_gpt      -> 使用 self_gpt_vocab.json"
    echo "  self_gpt_owt  -> 使用 self_gpt_owt_vocab.json"
    echo "  base          -> 使用 vocab.json (无前缀)"
    exit 1
fi

PREFIX=$1
shift # 移除第一个参数(prefix)，剩下的参数($@)将作为输入文件传递

# 构造文件名逻辑
if [ "$PREFIX" == "base" ]; then
    # 特殊情况：处理没有前缀的 vocab.json / merges.txt
    VOCAB_FILE="vocab.json"
    MERGES_FILE="merges.txt"
else
    # 标准情况：prefix_vocab.json
    VOCAB_FILE="${PREFIX}_vocab.json"
    MERGES_FILE="${PREFIX}_merges.txt"
fi

# 检查文件是否存在于 MODEL_DIR 中
if [ ! -f "$MODEL_DIR/$VOCAB_FILE" ] || [ ! -f "$MODEL_DIR/$MERGES_FILE" ]; then
    echo "错误: 在 $MODEL_DIR 中找不到对应的前缀文件。"
    echo "检查路径: $MODEL_DIR/$VOCAB_FILE"
    exit 1
fi

echo "=========================================="
echo " 正在运行 BPE 编码任务"
echo " ------------------------------------------"
echo " 脚本: $PYTHON_SCRIPT"
echo " 词表: $VOCAB_FILE"
echo " 合并: $MERGES_FILE"
echo " 核心: $WORKERS"
if [ $# -gt 0 ]; then
    echo " 输入: $@"
else
    echo " 输入: (使用 Python 脚本中的默认列表)"
fi
echo "=========================================="

# 构建命令
CMD="python $PYTHON_SCRIPT \
    --model_dir $MODEL_DIR \
    --vocab_file $VOCAB_FILE \
    --merges_file $MERGES_FILE \
    --workers $WORKERS"

# 如果有额外的输入文件参数，则追加到命令后面
if [ $# -gt 0 ]; then
    CMD="$CMD --inputs $@"
fi

# 执行命令
$CMD