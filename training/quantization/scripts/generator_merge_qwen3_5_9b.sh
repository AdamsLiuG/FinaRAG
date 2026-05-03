#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/../../.." && pwd)

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/media/main/lgd/llm/models/Qwen/Qwen3.5-9B}"
ADAPTER_PATH="${ADAPTER_PATH:-${REPO_ROOT}/training/generator_sft/saves/qwen3.5_9b_qlora_sft_2x4090}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/training/quantization/artifacts/qwen3.5_9b_merged}"
TEMPLATE="${TEMPLATE:-default}"

cd "${REPO_ROOT}"

exec "${PYTHON_BIN}" -m training.quantization.merge_lora \
  --base-model-path "${BASE_MODEL_PATH}" \
  --adapter-path "${ADAPTER_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --template "${TEMPLATE}" \
  --trust-remote-code \
  --overwrite-output-dir \
  "$@"
