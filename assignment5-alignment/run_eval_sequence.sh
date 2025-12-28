#!/bin/bash

echo "========== 开始批量评估任务: $(date) =========="

echo ">>> [1/3] Running Baseline evaluation..."
python cs336_alignment/evaluate_passk.py --config configs/eval/evaluate_baseline_amc_pass64.yaml

echo ">>> [2/3] Running SFT evaluation..."
python cs336_alignment/evaluate_passk.py --config configs/eval/evaluate_sft_amc_pass64.yaml

echo ">>> [3/3] Running Dr.GRPO evaluation..."
python cs336_alignment/evaluate_passk.py --config configs/eval/evaluate_drgrpo_amc_pass64.yaml

echo "========== 所有任务已完成: $(date) =========="
