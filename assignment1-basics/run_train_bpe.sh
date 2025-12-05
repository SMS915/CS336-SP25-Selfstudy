#!/bin/bash

# ================= 配置区域 =================

INPUT_DATA="data/OpenWebText-Sample-100MB.txt"

# 2. 设置目标词表大小 (GPT-2 标准是 50257)
VOCAB_SIZE=32000

# 3. 设置保存的文件名前缀 (会生成 ..._vocab.json 和 ..._merges.txt)
SAVE_PREFIX="BPE_File/naive_test_100mb"

BPE_VERSION="naive"

# 4. 设置特殊 Token (用空格分隔)
SPECIAL_TOKENS="<|endoftext|>"

# ================= 环境设置 =================

# 获取当前脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 设置 Python 路径，确保能找到 cs336_basics 包
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# 如果使用 uv 管理环境，确保已激活
source .venv/bin/activate

# ================= 执行命令 =================

echo "🚀开始训练"
echo "Data: $INPUT_DATA"
echo "Vocab: $VOCAB_SIZE"

# 确保输出目录存在
mkdir -p "$(dirname "$SAVE_PREFIX")"

python cs336_basics/train_bpe.py \
    --input_path "$INPUT_DATA" \
    --vocab_size "$VOCAB_SIZE" \
    --save_name "$SAVE_PREFIX" \
    --bpe_version "$BPE_VERSION" \
    --special_tokens $SPECIAL_TOKENS

# 检查退出状态
if [ $? -eq 0 ]; then
    echo "✅ 分词器训练成功!"
    echo "保存到 ${SAVE_PREFIX}_vocab.json"
else
    echo "❌ 训练失败"
fi