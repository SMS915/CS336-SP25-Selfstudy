#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_PATH="${OUTPUT_PATH:-data/RL/qwen3_rl_math_mix_15k.jsonl}"
MATH_COUNT="${MATH_COUNT:-7500}"
GSM8K_COUNT="${GSM8K_COUNT:-3000}"
NUMINA_COUNT="${NUMINA_COUNT:-4500}"
CURRICULUM_COUNT="${CURRICULUM_COUNT:-0}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

"$PYTHON_BIN" cs336_alignment/data_preparation/build_qwen3_rl_dataset.py \
  --output_path "$OUTPUT_PATH" \
  --math_count "$MATH_COUNT" \
  --gsm8k_count "$GSM8K_COUNT" \
  --numina_count "$NUMINA_COUNT" \
  --curriculum_count "$CURRICULUM_COUNT" \
  --seed "$SEED" \
  --shuffle \
  "$@"
