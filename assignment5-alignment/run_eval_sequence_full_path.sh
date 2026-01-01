#!/bin/bash

# 1. 基础环境配置
LOG_FILE="logs/evaluate-fixed-math500-configs.log"
mkdir -p logs
exec > >(tee -a "$LOG_FILE") 2>&1

# 定义两个评估脚本
DEFAULT_EVAL_SCRIPT="cs336_alignment/evaluate_passk.py"
INSTRUCT_EVAL_SCRIPT="cs336_alignment/evaluate_instruct_passk.py"

# 2. 配置文件列表
configs=(
    "configs/eval/sft/evaluate_sft_math500_pass64.yaml"
    "configs/eval/grpo/evaluate_grpo_math500_pass64.yaml"
    "configs/eval/grpo_no_std_norm/evaluate_grpo_no_std_norm_math500_pass64.yaml"
    "configs/eval/instruct/evaluate_instruct_math500_pass64.yaml"
)

echo "========== 开始评估任务: $(date) =========="

total=${#configs[@]}

for i in "${!configs[@]}"; do
    current=$((i+1))
    conf_path="${configs[$i]}"
    
    echo "------------------------------------------------------------"
    
    # 3. 运行前的物理检查
    if [ ! -f "$conf_path" ]; then
        echo "!!! 错误: 配置文件不存在 -> $conf_path"
        continue
    fi

    # 4. 动态逻辑：判断是否为 instruct 模型任务
    # 使用 bash 的 [[ $string == *substring* ]] 进行模糊匹配
    if [[ "$conf_path" == *"instruct"* ]]; then
        CURRENT_SCRIPT="$INSTRUCT_EVAL_SCRIPT"
        echo ">>> [$current/$total] 检测到 Instruct 任务，使用脚本: $CURRENT_SCRIPT"
    else
        CURRENT_SCRIPT="$DEFAULT_EVAL_SCRIPT"
        echo ">>> [$current/$total] 普通任务，使用脚本: $CURRENT_SCRIPT"
    fi

    echo ">>> 运行配置: $conf_path"

    # 5. 执行评估
    # python -u 确保输出不被缓冲
    python -u "$CURRENT_SCRIPT" --config "$conf_path"
    
    # 检查退出码
    if [ $? -ne 0 ]; then
        echo "!!! 任务 [$current/$total] 失败: $conf_path"
        echo "正在停止脚本以防后续任务链式失败。"
        exit 1
    fi
    echo ">>> [$current/$total] 任务完成。"
done

echo "============================================================"
echo "所有任务已完成: $(date)"
echo "日志已记录至: $LOG_FILE"