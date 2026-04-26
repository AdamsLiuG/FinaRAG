#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/../../.." && pwd)

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/media/main/lgd/llm/models/Qwen/Qwen3-Reranker-0.6B}"
ADAPTER_PATH="${ADAPTER_PATH:-${REPO_ROOT}/training/reranker_distill/saves/qwen3_reranker_0.6b_sft_lora_20260420_173247}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/training/quantization/artifacts/qwen3_reranker_0.6b_merged}"
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
