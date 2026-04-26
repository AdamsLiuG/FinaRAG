#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/../../.." && pwd)

PYTHON_BIN="${PYTHON_BIN:-python}"
BACKEND="${BACKEND:-llmcompressor}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/training/quantization/artifacts/qwen3_reranker_0.6b_merged}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/training/quantization/artifacts/qwen3_reranker_0.6b_awq_int4}"
CALIB_PATH="${CALIB_PATH:-${REPO_ROOT}/training/reranker_distill/processed/qwen3_reranker_sft_train.jsonl}"
DATASET_FORMAT="${DATASET_FORMAT:-auto}"
MAX_CALIB_SAMPLES="${MAX_CALIB_SAMPLES:-128}"
MAX_CALIB_LENGTH="${MAX_CALIB_LENGTH:-3072}"
W_BIT="${W_BIT:-4}"
Q_GROUP_SIZE="${Q_GROUP_SIZE:-128}"
VERSION="${VERSION:-GEMM}"

cd "${REPO_ROOT}"

exec "${PYTHON_BIN}" -m training.quantization.quantize_awq \
  --backend "${BACKEND}" \
  --task-type reranker \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --calib-path "${CALIB_PATH}" \
  --dataset-format "${DATASET_FORMAT}" \
  --max-calib-samples "${MAX_CALIB_SAMPLES}" \
  --max-calib-length "${MAX_CALIB_LENGTH}" \
  --w-bit "${W_BIT}" \
  --q-group-size "${Q_GROUP_SIZE}" \
  --zero-point \
  --version "${VERSION}" \
  --overwrite-output-dir \
  "$@"
