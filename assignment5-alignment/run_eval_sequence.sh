#!/bin/bash

echo "========== 开始批量评估任务: $(date) =========="

echo ">>> [1/5] Running gsm8k pass@1 evaluation..."
python cs336_alignment/evaluate_passk.py --config configs/eval/evaluate_drgrpo_gsm8k_pass1.yaml

echo ">>> [2/5] Running MATH-500 pass@64 evaluation..."
python cs336_alignment/evaluate_passk.py --config configs/eval/evaluate_drgrpo_math500_pass64.yaml

echo ">>> [3/5] Running AMC pass@64 evaluation..."
python cs336_alignment/evaluate_passk.py --config configs/eval/evaluate_drgrpo_amc_pass64.yaml

echo ">>> [4/5] Running AIME24 pass@64 evaluation..."
python cs336_alignment/evaluate_passk.py --config configs/eval/evaluate_drgrpo_aime24_pass64.yaml

echo ">>> [5/5] Running Math-Test pass@8 evaluation..."
python cs336_alignment/evaluate_passk.py --config configs/eval/evaluate_drgrpo_MathTest_pass8.yaml

echo "========== 所有任务已完成: $(date) =========="

# evaluate_drgrpo_MathTest_pass8.yaml
# evaluate_drgrpo_aime24_pass64.yaml
# evaluate_drgrpo_amc_pass64.yaml
# evaluate_drgrpo_gsm8k_pass1.yaml
# evaluate_drgrpo_math500_pass64.yaml