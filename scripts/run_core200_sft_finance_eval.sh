#!/bin/sh
set -eu

# Edit these defaults, or override any of them from the command line:
#   RUN_ID=my_run QWEN_BASE_URL_VALUE=http://127.0.0.1:8081/v1 bash scripts/run_core200_sft_finance_eval.sh

ROOT="${ROOT:-/media/main/lgd/llm/FinaRAG}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"

# New SFT model service. Start vLLM separately before running this script.
QWEN_BASE_URL_VALUE="${QWEN_BASE_URL_VALUE:-http://127.0.0.1:8081/v1}"
QWEN_API_KEY_VALUE="${QWEN_API_KEY_VALUE:-EMPTY}"
QWEN_MODEL_VALUE="${QWEN_MODEL_VALUE:-qwen3.5_9b_gen120_mix_v2_topup_awq_int4}"

# Benchmark inputs.
PIPELINE_DATASET_DIR="${PIPELINE_DATASET_DIR:-$ROOT/data/top10_industries_2024_20each}"
QUESTIONS_FILE="${QUESTIONS_FILE:-$ROOT/data/finance_eval_benchmark_v2/core200/questions.json}"
GOLD_ANSWERS_FILE="${GOLD_ANSWERS_FILE:-$ROOT/data/finance_eval_benchmark_v2/core200/answers_gold.json}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT/config/qwen_zh_finance_colbert_cascade_bge.yaml}"

# Output naming.
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs}"
RUN_ID="${RUN_ID:-sft_gen120_mix_v2_topup_awq_$(date '+%Y%m%d_%H%M%S')}"
RAW_ANSWERS_FILE="${RAW_ANSWERS_FILE:-$OUTPUT_DIR/benchmark_core200_${RUN_ID}.raw_answers.json}"
PRED_ANSWERS_OUT="${PRED_ANSWERS_OUT:-$OUTPUT_DIR/benchmark_core200_${RUN_ID}.finance_eval.json}"
EVAL_DEBUG_OUT="${EVAL_DEBUG_OUT:-$OUTPUT_DIR/benchmark_core200_${RUN_ID}.finance_eval_debug.json}"
REPORT_OUT="${REPORT_OUT:-$OUTPUT_DIR/core200.${RUN_ID}.ragas.report.json}"

# Evaluation switches.
RAGAS_CONTEXT_LIMIT="${RAGAS_CONTEXT_LIMIT:-5}"
NO_CASE_DETAILS="${NO_CASE_DETAILS:-1}"
DISABLE_RAGAS="${DISABLE_RAGAS:-0}"

# Set RUN_PIPELINE=0 to skip generation and only evaluate existing finance_eval files.
RUN_PIPELINE="${RUN_PIPELINE:-1}"
PRED_ANSWERS_FILE="${PRED_ANSWERS_FILE:-$PRED_ANSWERS_OUT}"
DEBUG_FILE="${DEBUG_FILE:-$EVAL_DEBUG_OUT}"

# Set RESUME=1 to resume into RAW_ANSWERS_FILE when running the full pipeline.
RESUME="${RESUME:-0}"

# Set DRY_RUN=1 to print resolved paths without executing Python.
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$OUTPUT_DIR"
cd "$ROOT"

echo "RUN_ID: $RUN_ID"
echo "QWEN_BASE_URL: $QWEN_BASE_URL_VALUE"
echo "QWEN_MODEL: $QWEN_MODEL_VALUE"
echo "Questions: $QUESTIONS_FILE"
echo "Gold answers: $GOLD_ANSWERS_FILE"
echo "Report: $REPORT_OUT"

if [ "$RUN_PIPELINE" = "0" ]; then
  echo "Mode: eval only"
  echo "Pred answers: $PRED_ANSWERS_FILE"
  echo "Debug file: $DEBUG_FILE"

  set -- -m eval.finance_eval \
    --questions-file "$QUESTIONS_FILE" \
    --gold-answers-file "$GOLD_ANSWERS_FILE" \
    --pred-answers-file "$PRED_ANSWERS_FILE" \
    --debug-file "$DEBUG_FILE" \
    --output "$REPORT_OUT" \
    --ragas-context-limit "$RAGAS_CONTEXT_LIMIT"
else
  echo "Mode: generate + export + eval"
  echo "Raw answers: $RAW_ANSWERS_FILE"
  echo "Pred answers: $PRED_ANSWERS_OUT"
  echo "Eval debug: $EVAL_DEBUG_OUT"

  set -- -m eval.run_finance_benchmark \
    --pipeline-dataset-dir "$PIPELINE_DATASET_DIR" \
    --questions-file "$QUESTIONS_FILE" \
    --gold-answers-file "$GOLD_ANSWERS_FILE" \
    --config-path "$CONFIG_PATH" \
    --resume-file "$RAW_ANSWERS_FILE" \
    --pred-answers-out "$PRED_ANSWERS_OUT" \
    --eval-debug-out "$EVAL_DEBUG_OUT" \
    --report-out "$REPORT_OUT" \
    --ragas-context-limit "$RAGAS_CONTEXT_LIMIT"

  if [ "$RESUME" = "1" ]; then
    set -- "$@" --resume
  fi
fi

if [ "$NO_CASE_DETAILS" = "1" ]; then
  set -- "$@" --no-case-details
fi

if [ "$DISABLE_RAGAS" = "1" ]; then
  set -- "$@" --disable-ragas
fi

if [ "$DRY_RUN" = "1" ]; then
  echo "Dry run command:"
  printf 'QWEN_BASE_URL=%s QWEN_API_KEY=%s QWEN_MODEL=%s %s' \
    "$QWEN_BASE_URL_VALUE" "$QWEN_API_KEY_VALUE" "$QWEN_MODEL_VALUE" "$PYTHON_BIN"
  for arg in "$@"; do
    printf ' %s' "$arg"
  done
  printf '\n'
  exit 0
fi

exec env \
  QWEN_BASE_URL="$QWEN_BASE_URL_VALUE" \
  QWEN_API_KEY="$QWEN_API_KEY_VALUE" \
  QWEN_MODEL="$QWEN_MODEL_VALUE" \
  "$PYTHON_BIN" "$@"
