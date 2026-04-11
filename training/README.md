# Training Workspace

This directory defines the offline training workflows that sit beside the
online FinaRAG inference pipeline.

It is intentionally split into two tracks:

- `generator_sft/`: build retrieval-grounded SFT data for the answer model
- `reranker_distill/`: build pointwise distillation data for the reranker

Recommended execution order:

1. Build and train the reranker student first.
2. Re-run offline retrieval with the improved reranker.
3. Build cleaner generator SFT data.
4. Train the generator student.

Design docs:

- `generator_sft/README.md`
- `reranker_distill/README.md`

## 2x4090 Quickstart

Use the repo-local virtualenv for the data build steps:

```bash
cd /media/main/lgd/llm/FinaRAG
PYTHON_BIN=/media/main/lgd/llm/FinaRAG/.venv/bin/python
```

The generator teacher configs now read model names from `.env`:

- `training/generator_sft/configs/data_build.example.yaml` -> `${SUB2API_MODEL}`
- `training/generator_sft/configs/data_build.local_vllm_qwen35.example.yaml` -> `${QWEN_VLLM_MODEL}`

Recommended execution order:

```bash
# 1. Build generator seed queries from the current financial-report dataset.
$PYTHON_BIN training/generator_sft/scripts/build_seed_queries.py \
  --config-path training/generator_sft/configs/build_seed.example.yaml

# 2. Mine teacher answers with your stronger online API teacher.
$PYTHON_BIN training/generator_sft/scripts/mine_teacher_answers.py \
  --config-path training/generator_sft/configs/data_build.example.yaml

# Or switch to a larger local vLLM-served Qwen3.5 teacher.
$PYTHON_BIN training/generator_sft/scripts/mine_teacher_answers.py \
  --config-path training/generator_sft/configs/data_build.local_vllm_qwen35.example.yaml

# 3. Filter the raw teacher answers into high-quality SFT samples.
$PYTHON_BIN training/generator_sft/scripts/filter_sft_samples.py \
  --config-path training/generator_sft/configs/filter.example.yaml

# 4. Collect reranker candidate pools from the filtered query set.
$PYTHON_BIN training/reranker_distill/scripts/collect_candidate_pool.py \
  --config-path training/reranker_distill/configs/data_build.example.yaml

# 5. Score those candidates with the stronger reranker teacher API.
$PYTHON_BIN training/reranker_distill/scripts/score_with_teacher_reranker.py \
  --config-path training/reranker_distill/configs/data_build.example.yaml

# 6. Build pointwise reranker labels.
$PYTHON_BIN training/reranker_distill/scripts/build_pointwise_labels.py \
  --config-path training/reranker_distill/configs/data_build.example.yaml

# 7. Split reranker data.
$PYTHON_BIN training/reranker_distill/scripts/split_train_dev_test.py \
  --config-path training/reranker_distill/configs/split.example.yaml

# 8. Export reranker train/dev/test files for the trainer.
$PYTHON_BIN training/reranker_distill/scripts/export_for_trainer.py \
  --config-path training/reranker_distill/configs/export.example.yaml
$PYTHON_BIN training/reranker_distill/scripts/export_for_trainer.py \
  --config-path training/reranker_distill/configs/export.example.yaml \
  --input-path training/reranker_distill/processed/pointwise_dev_raw.jsonl \
  --output-path training/reranker_distill/processed/pointwise_dev.jsonl \
  --stats-output-path training/reranker_distill/manifests/export_dev_stats.json
$PYTHON_BIN training/reranker_distill/scripts/export_for_trainer.py \
  --config-path training/reranker_distill/configs/export.example.yaml \
  --input-path training/reranker_distill/processed/pointwise_test_raw.jsonl \
  --output-path training/reranker_distill/processed/pointwise_test.jsonl \
  --stats-output-path training/reranker_distill/manifests/export_test_stats.json

# 9. Train the reranker student (LoRA by default, switch quantization_bits to 4 for QLoRA).
torchrun --nproc_per_node=2 training/reranker_distill/scripts/train_reranker_lora.py \
  --config-path training/reranker_distill/configs/train.example.yaml

# 10. Convert filtered generator samples into chat SFT format.
$PYTHON_BIN training/generator_sft/scripts/convert_to_chat_sft.py \
  --config-path training/generator_sft/configs/filter.example.yaml

# 11. Split generator chat data and export LLaMA Factory dataset_info.json.
$PYTHON_BIN training/generator_sft/scripts/split_train_dev_test.py \
  --config-path training/generator_sft/configs/split.example.yaml

# 12. Train the generator student with QLoRA on Qwen3.5-9B.
llamafactory-cli train training/generator_sft/configs/train.example.yaml
```
