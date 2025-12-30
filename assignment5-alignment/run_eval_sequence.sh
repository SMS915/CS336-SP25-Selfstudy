#!/bin/bash
# 定义日志文件
LOG_FILE="evaluate-drpo-350.log"

exec > >(tee -a "$LOG_FILE") 2>&1

CONFIG_DIR="configs/eval/drgrpo-350"
EVAL_SCRIPT="cs336_alignment/evaluate_passk.py"
# EVAL_SCRIPT="cs336_alignment/evaluate_instruct_passk.py"
tasks=(
    "evaluate_drgrpo_gsm8k_pass1.yaml"
    "evaluate_drgrpo_math500_pass64.yaml"
    "evaluate_drgrpo_amc_pass64.yaml"
    "evaluate_drgrpo_aime24_pass64.yaml"
    "evaluate_drgrpo_aime25_pass64.yaml"
    "evaluate_drgrpo_MathTest_pass8.yaml"
)

echo "========== 开始批量评估任务: $(date) =========="
total=${#tasks[@]}
for i in "${!tasks[@]}"; do
    current=$((i+1))
    conf="${tasks[$i]}"
    
    echo ">>> [$current/$total] Running $conf ..."
    
    # 执行命令
    python "$EVAL_SCRIPT" --config "$CONFIG_DIR/$conf"
    
    # 检查上一个命令的退出状态码
    if [ $? -ne 0 ]; then
        echo "!!! 任务 [$current/$total] 失败，正在停止脚本。"
        exit 1
    fi
done

echo "========== 所有任务已完成: $(date) =========="

# evaluate_drgrpo_MathTest_pass8.yaml
# evaluate_drgrpo_aime24_pass64.yaml
# evaluate_drgrpo_amc_pass64.yaml
# evaluate_drgrpo_gsm8k_pass1.yaml
# evaluate_drgrpo_math500_pass64.yaml