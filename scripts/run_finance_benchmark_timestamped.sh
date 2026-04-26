#!/bin/sh
set -eu

ROOT="/media/main/lgd/llm/FinaRAG"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"

PIPELINE_DATASET_DIR="${PIPELINE_DATASET_DIR:-$ROOT/data/top10_industries_2024_20each}"
QUESTIONS_FILE="${QUESTIONS_FILE:-$ROOT/data/finance_eval_benchmark_v2/core200/questions.json}"
GOLD_ANSWERS_FILE="${GOLD_ANSWERS_FILE:-$ROOT/data/finance_eval_benchmark_v2/core200/answers_gold.json}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT/config/qwen_zh_finance_colbert_cascade_bge.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-benchmark_core200_qwen_colbert_bge}"

TIMESTAMP="${TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
RAW_ANSWERS_FILE="$OUTPUT_DIR/${OUTPUT_PREFIX}.${TIMESTAMP}.raw_answers.json"
PRED_ANSWERS_OUT="$OUTPUT_DIR/${OUTPUT_PREFIX}.${TIMESTAMP}.finance_eval.json"
EVAL_DEBUG_OUT="$OUTPUT_DIR/${OUTPUT_PREFIX}.${TIMESTAMP}.finance_eval_debug.json"
REPORT_OUT="$OUTPUT_DIR/${OUTPUT_PREFIX}.${TIMESTAMP}.finance_eval.report.json"

mkdir -p "$OUTPUT_DIR"

echo "Running benchmark with timestamp: $TIMESTAMP"
echo "Raw answers: $RAW_ANSWERS_FILE"
echo "Pred answers: $PRED_ANSWERS_OUT"
echo "Eval debug: $EVAL_DEBUG_OUT"
echo "Report: $REPORT_OUT"

exec "$PYTHON_BIN" -m eval.run_finance_benchmark \
  --pipeline-dataset-dir "$PIPELINE_DATASET_DIR" \
  --questions-file "$QUESTIONS_FILE" \
  --gold-answers-file "$GOLD_ANSWERS_FILE" \
  --config-path "$CONFIG_PATH" \
  --resume-file "$RAW_ANSWERS_FILE" \
  --pred-answers-out "$PRED_ANSWERS_OUT" \
  --eval-debug-out "$EVAL_DEBUG_OUT" \
  --report-out "$REPORT_OUT" \
  "$@"
