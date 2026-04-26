#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/media/main/lgd/llm/FinaRAG}"
BUILD_VENV_PATH="${BUILD_VENV_PATH:-${REPO_ROOT}/.venv}"
PY="${PY:-${BUILD_VENV_PATH}/bin/python}"

TAG="${TAG:-gen120_text_v1}"
RET="${RET:-config/qwen_zh_finance_colbert_cascade_qwen_hyde_fallback.yaml}"
DATASET_ROOT_PATH="${DATASET_ROOT_PATH:-data/top10_industries_2024_20each}"

QWEN_MODEL_VALUE="${QWEN_MODEL_VALUE:-Qwen3.5-9B}"
QWEN_BASE_URL_VALUE="${QWEN_BASE_URL_VALUE:-http://192.168.1.158:8081/v1}"
RERANKING_MODEL_VALUE="${RERANKING_MODEL_VALUE:-Qwen3-Reranker-0.6B}"
RERANKING_BASE_URL_VALUE="${RERANKING_BASE_URL_VALUE:-http://192.168.1.158:8084/v1}"
SUB2API_MODEL_VALUE="${SUB2API_MODEL_VALUE:-gpt-5.4}"
SUB2API_BASE_URL_VALUE="${SUB2API_BASE_URL_VALUE:-https://yybb.codes/v1}"

EMBEDDING_DEVICE_VALUE="${EMBEDDING_DEVICE_VALUE:-cuda:0,cuda:1}"
EMBEDDING_MAX_DEVICES_VALUE="${EMBEDDING_MAX_DEVICES_VALUE:-2}"
EMBEDDING_SPARSE_DEVICE_VALUE="${EMBEDDING_SPARSE_DEVICE_VALUE:-cuda:0,cuda:1}"
EMBEDDING_SPARSE_MAX_DEVICES_VALUE="${EMBEDDING_SPARSE_MAX_DEVICES_VALUE:-2}"
COLBERT_DEVICE_VALUE="${COLBERT_DEVICE_VALUE:-cuda:0}"
COLBERT_MAX_DEVICES_VALUE="${COLBERT_MAX_DEVICES_VALUE:-1}"

CANDIDATE_PARALLEL_REQUESTS="${CANDIDATE_PARALLEL_REQUESTS:-4}"
TEACHER_PARALLEL_REQUESTS="${TEACHER_PARALLEL_REQUESTS:-4}"
CANDIDATE_POOL_SIZE="${CANDIDATE_POOL_SIZE:-16}"
PER_QUERY_RETRIEVAL_TOP_K="${PER_QUERY_RETRIEVAL_TOP_K:-12}"
MAX_CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-7000}"
MAX_DOC_CHARS="${MAX_DOC_CHARS:-1200}"

RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
LOG_DIR_REL="${LOG_DIR_REL:-training/generator_sft/logs/gen120_text_postprocess/${TAG}/${RUN_ID}}"
LOG_DIR="${REPO_ROOT}/${LOG_DIR_REL}"
BACKUP_DIR="${LOG_DIR}/backups"
SUMMARY_PATH="${LOG_DIR}/summary.txt"
TRUNCATED_STATUS="not_run"

FILTERED_INPUT_REL="training/generator_sft/processed/teacher_answers_filtered.${TAG}.jsonl"
RETRIEVED_CACHE_REL="training/generator_sft/raw/retrieved_cache.${TAG}.jsonl"
CANDIDATE_POOL_REL="training/generator_sft/raw/candidate_pool.generator.${TAG}.jsonl"
CANDIDATE_STATS_REL="training/generator_sft/manifests/candidate_pool.${TAG}.stats.json"

HARD_CONTEXT_CONFIG_REL="training/generator_sft/configs/hard_context.example.yaml"
HARD_CONTEXT_OUTPUT_REL="training/generator_sft/processed/hard_context_samples.${TAG}.jsonl"
HARD_CONTEXT_CHAT_REL="training/generator_sft/processed/hard_context.chat.${TAG}.jsonl"
HARD_CONTEXT_REJECTED_REL="training/generator_sft/manifests/hard_context.${TAG}.rejected.jsonl"
HARD_CONTEXT_STATS_REL="training/generator_sft/manifests/hard_context.${TAG}.stats.json"

HARD_RECHECK_CONFIG_REL="training/generator_sft/configs/hard_context_recheck.example.yaml"
HARD_RECHECK_OUTPUT_REL="training/generator_sft/processed/hard_context_rechecked.${TAG}.jsonl"
HARD_RECHECK_CHAT_REL="training/generator_sft/processed/hard_context_rechecked.chat.${TAG}.jsonl"
HARD_RECHECK_REJECTED_REL="training/generator_sft/manifests/hard_context_rechecked.${TAG}.rejected.jsonl"
HARD_RECHECK_STATS_REL="training/generator_sft/manifests/hard_context_rechecked.${TAG}.stats.json"

ANCHOR_INPUT_REL="training/generator_sft/processed/anchor_clean_positive.${TAG}.jsonl"
TRUNCATED_CONFIG_REL="training/generator_sft/configs/truncated_refusal.example.yaml"
TRUNCATED_OUTPUT_REL="training/generator_sft/processed/truncated_refusal.${TAG}.jsonl"
TRUNCATED_CHAT_REL="training/generator_sft/processed/truncated_refusal.chat.${TAG}.jsonl"
TRUNCATED_REJECTED_REL="training/generator_sft/manifests/truncated_refusal.${TAG}.rejected.jsonl"
TRUNCATED_STATS_REL="training/generator_sft/manifests/truncated_refusal.${TAG}.stats.json"

WRONG_CONTEXT_CONFIG_REL="training/generator_sft/configs/wrong_context.example.yaml"
WRONG_CONTEXT_OUTPUT_REL="training/generator_sft/processed/wrong_context_refusal.${TAG}.jsonl"
WRONG_CONTEXT_CHAT_REL="training/generator_sft/processed/wrong_context_refusal.chat.${TAG}.jsonl"
WRONG_CONTEXT_REJECTED_REL="training/generator_sft/manifests/wrong_context_refusal.${TAG}.rejected.jsonl"
WRONG_CONTEXT_STATS_REL="training/generator_sft/manifests/wrong_context_refusal.${TAG}.stats.json"

usage() {
  cat <<'EOF'
Usage:
  bash training/generator_sft/run_gen120_text_postprocess.sh

This script runs the post-processing pipeline for the gen120 text mix:
  1. collect_candidate_pool.py           (resume)
  2. build_hard_context_samples.py       (rebuild)
  3. recheck_hard_context_samples.py     (rebuild)
  4. build_truncated_refusal.py          (auto-skip when anchor count is 0)
  5. build_wrong_context_refusal.py      (resume)

Environment variables you may override:
  TAG=gen120_text_v1
  RET=config/qwen_zh_finance_colbert_cascade_qwen_hyde_fallback.yaml
  CANDIDATE_PARALLEL_REQUESTS=4
  TEACHER_PARALLEL_REQUESTS=4
  CANDIDATE_POOL_SIZE=16
  PER_QUERY_RETRIEVAL_TOP_K=12
  MAX_CONTEXT_CHARS=7000
  MAX_DOC_CHARS=1200
  RUN_ID=custom_id
  LOG_DIR_REL=training/generator_sft/logs/gen120_text_postprocess/<tag>/<run_id>

Examples:
  bash training/generator_sft/run_gen120_text_postprocess.sh
  TAG=gen120_text_v1 TEACHER_PARALLEL_REQUESTS=6 \
    bash training/generator_sft/run_gen120_text_postprocess.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

mkdir -p "${LOG_DIR}" "${BACKUP_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  log "[error] $*" >&2
  exit 1
}

on_error() {
  local exit_code="$1"
  local line_no="$2"
  log "[error] command failed at line ${line_no} with exit code ${exit_code}" >&2
  log "[error] see logs under ${LOG_DIR_REL}" >&2
  exit "${exit_code}"
}

trap 'on_error "$?" "$LINENO"' ERR

count_jsonl() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo 0
    return
  fi
  wc -l < "${path}" | tr -d '[:space:]'
}

backup_if_exists() {
  local rel_path="$1"
  local abs_path="${REPO_ROOT}/${rel_path}"
  if [[ -f "${abs_path}" ]]; then
    cp -f "${abs_path}" "${BACKUP_DIR}/$(basename "${abs_path}")"
  fi
}

run_logged_step() {
  local step_name="$1"
  shift
  local step_log="${LOG_DIR}/${step_name}.log"
  {
    log "[step] ${step_name}"
    log "[cmd] $*"
    "$@"
  } 2>&1 | tee -a "${step_log}"
}

write_summary() {
  {
    echo "run_id=${RUN_ID}"
    echo "log_dir=${LOG_DIR_REL}"
    echo "tag=${TAG}"
    echo "ret=${RET}"
    echo "qwen_model=${QWEN_MODEL}"
    echo "qwen_base_url=${QWEN_BASE_URL}"
    echo "reranking_model=${RERANKING_MODEL}"
    echo "reranking_base_url=${RERANKING_BASE_URL}"
    echo "sub2api_model=${SUB2API_MODEL}"
    echo "sub2api_base_url=${SUB2API_BASE_URL}"
    echo "filtered_input_lines=$(count_jsonl "${REPO_ROOT}/${FILTERED_INPUT_REL}")"
    echo "candidate_pool_lines=$(count_jsonl "${REPO_ROOT}/${CANDIDATE_POOL_REL}")"
    echo "hard_context_lines=$(count_jsonl "${REPO_ROOT}/${HARD_CONTEXT_OUTPUT_REL}")"
    echo "hard_context_rechecked_lines=$(count_jsonl "${REPO_ROOT}/${HARD_RECHECK_OUTPUT_REL}")"
    echo "truncated_refusal_status=${TRUNCATED_STATUS}"
    echo "truncated_refusal_lines=$(count_jsonl "${REPO_ROOT}/${TRUNCATED_OUTPUT_REL}")"
    echo "wrong_context_refusal_lines=$(count_jsonl "${REPO_ROOT}/${WRONG_CONTEXT_OUTPUT_REL}")"
    echo "anchor_lines=$(count_jsonl "${REPO_ROOT}/${ANCHOR_INPUT_REL}")"
    echo "candidate_pool_stats=${CANDIDATE_STATS_REL}"
    echo "hard_context_stats=${HARD_CONTEXT_STATS_REL}"
    echo "hard_context_rechecked_stats=${HARD_RECHECK_STATS_REL}"
    echo "truncated_refusal_stats=${TRUNCATED_STATS_REL}"
    echo "wrong_context_stats=${WRONG_CONTEXT_STATS_REL}"
  } | tee "${SUMMARY_PATH}"
}

[[ -d "${REPO_ROOT}" ]] || die "missing repo root: ${REPO_ROOT}"
[[ -x "${PY}" ]] || die "missing build venv python: ${PY}"
[[ -f "${REPO_ROOT}/${FILTERED_INPUT_REL}" ]] || die "missing filtered input: ${FILTERED_INPUT_REL}"
[[ -f "${REPO_ROOT}/${RETRIEVED_CACHE_REL}" ]] || die "missing retrieved cache: ${RETRIEVED_CACHE_REL}"

cd "${REPO_ROOT}"

export QWEN_MODEL="${QWEN_MODEL:-${QWEN_MODEL_VALUE}}"
export QWEN_BASE_URL="${QWEN_BASE_URL:-${QWEN_BASE_URL_VALUE}}"
export RERANKING_MODEL="${RERANKING_MODEL:-${RERANKING_MODEL_VALUE}}"
export RERANKING_BASE_URL="${RERANKING_BASE_URL:-${RERANKING_BASE_URL_VALUE}}"
export SUB2API_MODEL="${SUB2API_MODEL:-${SUB2API_MODEL_VALUE}}"
export SUB2API_BASE_URL="${SUB2API_BASE_URL:-${SUB2API_BASE_URL_VALUE}}"

export EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-${EMBEDDING_DEVICE_VALUE}}"
export EMBEDDING_MAX_DEVICES="${EMBEDDING_MAX_DEVICES:-${EMBEDDING_MAX_DEVICES_VALUE}}"
export EMBEDDING_SPARSE_DEVICE="${EMBEDDING_SPARSE_DEVICE:-${EMBEDDING_SPARSE_DEVICE_VALUE}}"
export EMBEDDING_SPARSE_MAX_DEVICES="${EMBEDDING_SPARSE_MAX_DEVICES:-${EMBEDDING_SPARSE_MAX_DEVICES_VALUE}}"
export COLBERT_DEVICE="${COLBERT_DEVICE:-${COLBERT_DEVICE_VALUE}}"
export COLBERT_MAX_DEVICES="${COLBERT_MAX_DEVICES:-${COLBERT_MAX_DEVICES_VALUE}}"

log "[info] repo_root=${REPO_ROOT}" | tee -a "${SUMMARY_PATH}"
log "[info] log_dir=${LOG_DIR_REL}" | tee -a "${SUMMARY_PATH}"
log "[info] tag=${TAG}" | tee -a "${SUMMARY_PATH}"
log "[info] filtered_input_lines=$(count_jsonl "${REPO_ROOT}/${FILTERED_INPUT_REL}")" | tee -a "${SUMMARY_PATH}"
log "[info] existing_candidate_pool_lines=$(count_jsonl "${REPO_ROOT}/${CANDIDATE_POOL_REL}")" | tee -a "${SUMMARY_PATH}"
log "[info] existing_anchor_lines=$(count_jsonl "${REPO_ROOT}/${ANCHOR_INPUT_REL}")" | tee -a "${SUMMARY_PATH}"

backup_if_exists "${HARD_CONTEXT_OUTPUT_REL}"
backup_if_exists "${HARD_CONTEXT_CHAT_REL}"
backup_if_exists "${HARD_CONTEXT_REJECTED_REL}"
backup_if_exists "${HARD_CONTEXT_STATS_REL}"
backup_if_exists "${HARD_RECHECK_OUTPUT_REL}"
backup_if_exists "${HARD_RECHECK_CHAT_REL}"
backup_if_exists "${HARD_RECHECK_REJECTED_REL}"
backup_if_exists "${HARD_RECHECK_STATS_REL}"
backup_if_exists "${TRUNCATED_OUTPUT_REL}"
backup_if_exists "${TRUNCATED_CHAT_REL}"
backup_if_exists "${TRUNCATED_REJECTED_REL}"
backup_if_exists "${TRUNCATED_STATS_REL}"

run_logged_step "01_collect_candidate_pool" \
  "${PY}" training/reranker_distill/scripts/collect_candidate_pool.py \
  --query-input-path "${FILTERED_INPUT_REL}" \
  --candidate-output-path "${CANDIDATE_POOL_REL}" \
  --stats-output-path "${CANDIDATE_STATS_REL}" \
  --retrieval-config-path "${RET}" \
  --dataset-root-path "${DATASET_ROOT_PATH}" \
  --candidate-pool-size "${CANDIDATE_POOL_SIZE}" \
  --per-query-retrieval-top-k "${PER_QUERY_RETRIEVAL_TOP_K}" \
  --parallel-requests "${CANDIDATE_PARALLEL_REQUESTS}" \
  --resume

run_logged_step "02_build_hard_context" \
  "${PY}" training/generator_sft/scripts/build_hard_context_samples.py \
  --config-path "${HARD_CONTEXT_CONFIG_REL}" \
  --input-path "${FILTERED_INPUT_REL}" \
  --retrieved-cache-input-path "${RETRIEVED_CACHE_REL}" \
  --candidate-pool-input-path "${CANDIDATE_POOL_REL}" \
  --candidate-pool-output-path "${CANDIDATE_POOL_REL}" \
  --output-path "${HARD_CONTEXT_OUTPUT_REL}" \
  --chat-output-path "${HARD_CONTEXT_CHAT_REL}" \
  --rejected-output-path "${HARD_CONTEXT_REJECTED_REL}" \
  --stats-output-path "${HARD_CONTEXT_STATS_REL}" \
  --retrieval-config-path "${RET}" \
  --max-context-chars "${MAX_CONTEXT_CHARS}" \
  --max-doc-chars "${MAX_DOC_CHARS}"

run_logged_step "03_recheck_hard_context" \
  "${PY}" training/generator_sft/scripts/recheck_hard_context_samples.py \
  --config-path "${HARD_RECHECK_CONFIG_REL}" \
  --input-path "${HARD_CONTEXT_OUTPUT_REL}" \
  --retrieved-cache-input-path "${RETRIEVED_CACHE_REL}" \
  --output-path "${HARD_RECHECK_OUTPUT_REL}" \
  --chat-output-path "${HARD_RECHECK_CHAT_REL}" \
  --rejected-output-path "${HARD_RECHECK_REJECTED_REL}" \
  --stats-output-path "${HARD_RECHECK_STATS_REL}" \
  --teacher-answer-provider sub2api \
  --teacher-answer-model "${SUB2API_MODEL}" \
  --parallel-requests "${TEACHER_PARALLEL_REQUESTS}"

ANCHOR_COUNT="$(count_jsonl "${REPO_ROOT}/${ANCHOR_INPUT_REL}")"
if [[ "${ANCHOR_COUNT}" == "0" ]]; then
  TRUNCATED_STATUS="skipped_anchor_zero"
  log "[info] skipping truncated_refusal because anchor count is 0 for ${ANCHOR_INPUT_REL}" | tee -a "${LOG_DIR}/04_truncated_refusal.log"
  rm -f "${TRUNCATED_OUTPUT_REL}" "${TRUNCATED_CHAT_REL}" "${TRUNCATED_REJECTED_REL}" "${TRUNCATED_STATS_REL}"
else
  TRUNCATED_STATUS="completed"
  run_logged_step "04_truncated_refusal" \
    "${PY}" training/generator_sft/scripts/build_truncated_refusal.py \
    --config-path "${TRUNCATED_CONFIG_REL}" \
    --input-path "${FILTERED_INPUT_REL}" \
    --retrieved-cache-input-path "${RETRIEVED_CACHE_REL}" \
    --anchor-input-path "${ANCHOR_INPUT_REL}" \
    --output-path "${TRUNCATED_OUTPUT_REL}" \
    --chat-output-path "${TRUNCATED_CHAT_REL}" \
    --rejected-output-path "${TRUNCATED_REJECTED_REL}" \
    --stats-output-path "${TRUNCATED_STATS_REL}" \
    --teacher-answer-provider sub2api \
    --teacher-answer-model "${SUB2API_MODEL}" \
    --max-context-chars "${MAX_CONTEXT_CHARS}" \
    --max-doc-chars "${MAX_DOC_CHARS}"
fi

run_logged_step "05_wrong_context_refusal" \
  "${PY}" training/generator_sft/scripts/build_wrong_context_refusal.py \
  --config-path "${WRONG_CONTEXT_CONFIG_REL}" \
  --input-path "${FILTERED_INPUT_REL}" \
  --retrieved-cache-input-path "${RETRIEVED_CACHE_REL}" \
  --candidate-pool-input-path "${CANDIDATE_POOL_REL}" \
  --candidate-pool-output-path "${CANDIDATE_POOL_REL}" \
  --output-path "${WRONG_CONTEXT_OUTPUT_REL}" \
  --chat-output-path "${WRONG_CONTEXT_CHAT_REL}" \
  --rejected-output-path "${WRONG_CONTEXT_REJECTED_REL}" \
  --stats-output-path "${WRONG_CONTEXT_STATS_REL}" \
  --retrieval-config-path "${RET}" \
  --dataset-root-path "${DATASET_ROOT_PATH}" \
  --teacher-answer-provider sub2api \
  --teacher-answer-model "${SUB2API_MODEL}" \
  --candidate-pool-size "${CANDIDATE_POOL_SIZE}" \
  --per-query-retrieval-top-k "${PER_QUERY_RETRIEVAL_TOP_K}" \
  --parallel-requests "${TEACHER_PARALLEL_REQUESTS}" \
  --max-context-chars "${MAX_CONTEXT_CHARS}" \
  --max-doc-chars "${MAX_DOC_CHARS}" \
  --resume

write_summary
log "[done] gen120 text postprocess pipeline completed" | tee -a "${SUMMARY_PATH}"
log "[done] summary=${SUMMARY_PATH}" | tee -a "${SUMMARY_PATH}"
