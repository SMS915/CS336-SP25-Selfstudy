#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

INPUT_PATH="${INPUT_PATH:-data/CoT/Bespoke-Stratos-17k-formatted.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-data/CoT/Bespoke-Stratos-17k-qwen3-sft.jsonl}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

"$PYTHON_BIN" cs336_alignment/data_preparation/build_qwen3_sft_dataset.py \
  --input_path "$INPUT_PATH" \
  --output_path "$OUTPUT_PATH" \
  "$@"
