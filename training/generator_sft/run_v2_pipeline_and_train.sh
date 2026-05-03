#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/media/main/lgd/llm/FinaRAG"
CONDA_ROOT="/media/main/lgd/miniconda3"
BUILD_VENV_PATH="${BUILD_VENV_PATH:-/media/main/lgd/llm/FinaRAG/.venv}"
TRAIN_CONDA_ENV_PATH="${TRAIN_CONDA_ENV_PATH:-/media/main/lgd/miniconda3/envs/ecollm}"

PIPELINE_MODE="${PIPELINE_MODE:-minimal}"
RUN_TRAIN="${RUN_TRAIN:-1}"
TRAIN_CONFIG="${TRAIN_CONFIG:-training/generator_sft/configs/train.2x4090.qwen35_4b.yaml}"

ANCHOR_CONFIG="${ANCHOR_CONFIG:-training/generator_sft/configs/anchor_clean_positive.example.yaml}"
FILTER_CONFIG="${FILTER_CONFIG:-training/generator_sft/configs/filter.strict_v2.example.yaml}"
HARD_CONTEXT_CONFIG="${HARD_CONTEXT_CONFIG:-training/generator_sft/configs/hard_context.example.yaml}"
HARD_CONTEXT_RECHECK_CONFIG="${HARD_CONTEXT_RECHECK_CONFIG:-training/generator_sft/configs/hard_context_recheck.example.yaml}"
TRUNCATED_REFUSAL_CONFIG="${TRUNCATED_REFUSAL_CONFIG:-training/generator_sft/configs/truncated_refusal.example.yaml}"
WRONG_CONTEXT_CONFIG="${WRONG_CONTEXT_CONFIG:-training/generator_sft/configs/wrong_context.example.yaml}"
SPLIT_CONFIG="${SPLIT_CONFIG:-training/generator_sft/configs/split.doc_holdout_v2.example.yaml}"

PROCESSED_DIR="${PROCESSED_DIR:-training/generator_sft/processed}"
MAIN_CHAT_PATH="${MAIN_CHAT_PATH:-${PROCESSED_DIR}/all.chat.v2.jsonl}"
CORE_ONLY_CHAT_BACKUP_PATH="${CORE_ONLY_CHAT_BACKUP_PATH:-${PROCESSED_DIR}/all.core_only.chat.v2.jsonl}"
HARD_RECHECK_CHAT_PATH="${HARD_RECHECK_CHAT_PATH:-${PROCESSED_DIR}/hard_context_rechecked.chat.v1.jsonl}"
TRUNCATED_CHAT_PATH="${TRUNCATED_CHAT_PATH:-${PROCESSED_DIR}/truncated_refusal.chat.v1.jsonl}"
WRONG_CHAT_PATH="${WRONG_CHAT_PATH:-${PROCESSED_DIR}/wrong_context_refusal.chat.v1.jsonl}"

usage() {
  cat <<'EOF'
Usage:
  bash training/generator_sft/run_v2_pipeline_and_train.sh

Environment variables:
  BUILD_VENV_PATH=/media/main/lgd/llm/FinaRAG/.venv
    Python venv used for data-building steps before training

  TRAIN_CONDA_ENV_PATH=/media/main/lgd/miniconda3/envs/ecollm
    Conda env used only for LLaMA-Factory SFT training

  PIPELINE_MODE=minimal|full
    minimal: anchor -> filter -> convert -> split -> train
    full:    anchor -> filter -> hard_context -> recheck -> truncated_refusal
             -> wrong_context_refusal -> convert -> merge -> split -> train

  RUN_TRAIN=1|0
    1: continue to LLaMA-Factory training
    0: stop after split/export

  TRAIN_CONFIG=training/generator_sft/configs/train.2x4090.qwen35_4b.yaml
    Training config passed to llamafactory-cli train

Examples:
  PIPELINE_MODE=minimal RUN_TRAIN=0 bash training/generator_sft/run_v2_pipeline_and_train.sh
  PIPELINE_MODE=full TRAIN_CONFIG=training/generator_sft/configs/train.example.yaml \
    bash training/generator_sft/run_v2_pipeline_and_train.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${PIPELINE_MODE}" != "minimal" && "${PIPELINE_MODE}" != "full" ]]; then
  echo "[error] PIPELINE_MODE must be minimal or full, got: ${PIPELINE_MODE}" >&2
  exit 1
fi

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "[error] Missing conda init script: ${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
  exit 1
fi

if [[ ! -x "${BUILD_VENV_PATH}/bin/python" ]]; then
  echo "[error] Missing build venv python: ${BUILD_VENV_PATH}/bin/python" >&2
  exit 1
fi

cd "${REPO_ROOT}"

echo "[info] repo_root=${REPO_ROOT}"
echo "[info] build_venv=${BUILD_VENV_PATH}"
echo "[info] train_conda_env=${TRAIN_CONDA_ENV_PATH}"
echo "[info] pipeline_mode=${PIPELINE_MODE}"
echo "[info] run_train=${RUN_TRAIN}"
echo "[info] train_config=${TRAIN_CONFIG}"

run_step() {
  local title="$1"
  shift
  echo
  echo "[step] ${title}"
  echo "[cmd] $*"
  "$@"
}

run_build_python_step() {
  local title="$1"
  shift
  run_step "${title}" "${BUILD_VENV_PATH}/bin/python" "$@"
}

merge_chat_jsonl() {
  mkdir -p "$(dirname "${MAIN_CHAT_PATH}")"
  if [[ -f "${MAIN_CHAT_PATH}" ]]; then
    cp "${MAIN_CHAT_PATH}" "${CORE_ONLY_CHAT_BACKUP_PATH}"
    echo "[info] backed up core-only chat to ${CORE_ONLY_CHAT_BACKUP_PATH}"
  fi

  "${BUILD_VENV_PATH}/bin/python" - "${MAIN_CHAT_PATH}" "${CORE_ONLY_CHAT_BACKUP_PATH}" "${HARD_RECHECK_CHAT_PATH}" "${TRUNCATED_CHAT_PATH}" "${WRONG_CHAT_PATH}" <<'PY'
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
candidate_paths = [Path(value) for value in sys.argv[2:]]
merged = []
seen_keys = set()
used_inputs = []

for path in candidate_paths:
    if not path.exists():
        continue
    used_inputs.append(str(path))
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        meta = record.get("meta") or {}
        sample_id = meta.get("sample_id") or ""
        query_id = meta.get("query_id") or ""
        variant_type = meta.get("variant_type") or ""
        key = (sample_id, query_id, variant_type)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged.append(record)

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as handle:
    for record in merged:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"[info] merged_chat_count={len(merged)}")
print(f"[info] merged_chat_output={output_path}")
print(f"[info] merged_chat_inputs={used_inputs}")
PY
}

run_build_python_step "Build anchor_clean_positive" \
  training/generator_sft/scripts/build_anchor_clean_positive.py \
  --config-path "${ANCHOR_CONFIG}"

run_build_python_step "Filter teacher answers with strict validators" \
  training/generator_sft/scripts/filter_sft_samples.py \
  --config-path "${FILTER_CONFIG}"

if [[ "${PIPELINE_MODE}" == "full" ]]; then
  run_build_python_step "Build hard-context positives" \
    training/generator_sft/scripts/build_hard_context_samples.py \
    --config-path "${HARD_CONTEXT_CONFIG}"

  run_build_python_step "Recheck hard-context positives" \
    training/generator_sft/scripts/recheck_hard_context_samples.py \
    --config-path "${HARD_CONTEXT_RECHECK_CONFIG}"

  run_build_python_step "Build truncated refusal" \
    training/generator_sft/scripts/build_truncated_refusal.py \
    --config-path "${TRUNCATED_REFUSAL_CONFIG}"

  run_build_python_step "Build wrong-context refusal" \
    training/generator_sft/scripts/build_wrong_context_refusal.py \
    --config-path "${WRONG_CONTEXT_CONFIG}"
fi

run_build_python_step "Convert filtered samples to chat SFT" \
  training/generator_sft/scripts/convert_to_chat_sft.py \
  --config-path "${FILTER_CONFIG}"

if [[ "${PIPELINE_MODE}" == "full" ]]; then
  echo
  echo "[step] Merge chat JSONL for split/training"
  merge_chat_jsonl
fi

run_build_python_step "Split train/dev/test and export LLaMA-Factory datasets" \
  training/generator_sft/scripts/split_train_dev_test.py \
  --config-path "${SPLIT_CONFIG}"

if [[ "${RUN_TRAIN}" == "1" ]]; then
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${TRAIN_CONDA_ENV_PATH}"
  echo
  echo "[info] switched to training conda env: ${TRAIN_CONDA_ENV_PATH}"
  run_step "Start LLaMA-Factory SFT training" \
    llamafactory-cli train "${TRAIN_CONFIG}"
else
  echo
  echo "[info] RUN_TRAIN=0, skipped training."
fi

echo
echo "[done] pipeline completed successfully."
