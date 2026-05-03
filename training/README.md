# 训练工作区

这个目录定义了与在线 FinaRAG 推理流水线配套的离线训练流程。

它被有意拆分为两条主线：

- `generator_sft/`：为答案生成模型构建基于检索上下文的 SFT 数据
- `reranker_distill/`：为重排器构建 pointwise 蒸馏数据

推荐执行顺序：

1. 先构建并训练学生重排器。
2. 使用改进后的重排器重新运行离线检索。
3. 构建更干净的 generator SFT 数据。
4. 训练生成器学生模型。

设计文档：

- `generator_sft/README.md`
- `reranker_distill/README.md`

## 2x4090 快速开始

数据构建步骤建议使用仓库本地的虚拟环境：

```bash
cd /media/main/lgd/llm/FinaRAG
PYTHON_BIN=/media/main/lgd/llm/FinaRAG/.venv/bin/python
```

生成器 teacher 配置现在会从 `.env` 读取模型名：

- `training/generator_sft/configs/data_build.example.yaml` -> `${SUB2API_MODEL}`
- `training/generator_sft/configs/data_build.local_vllm_qwen35.example.yaml` -> `${QWEN_VLLM_MODEL}`

重排器蒸馏的本地 vLLM 配置也会从 `.env` 读取：

- `training/reranker_distill/configs/data_build.local_vllm_reranker.example.yaml`
  -> `${RERANKING_BASE_URL}` + `${RERANKING_MODEL}`

推荐执行顺序：

```bash
# 1. 基于当前财报数据集构建 generator 的种子查询。
$PYTHON_BIN training/generator_sft/scripts/build_seed_queries.py \
  --config-path training/generator_sft/configs/build_seed.example.yaml

# 2. 使用更强的在线 API teacher 挖掘 teacher 答案。
$PYTHON_BIN training/generator_sft/scripts/mine_teacher_answers.py \
  --config-path training/generator_sft/configs/data_build.example.yaml

# 或者切换到由本地 vLLM 提供服务的更大 Qwen3.5 teacher。
$PYTHON_BIN training/generator_sft/scripts/mine_teacher_answers.py \
  --config-path training/generator_sft/configs/data_build.local_vllm_qwen35.example.yaml

# 3. 将原始 teacher 答案过滤为高质量 SFT 样本。
$PYTHON_BIN training/generator_sft/scripts/filter_sft_samples.py \
  --config-path training/generator_sft/configs/filter.example.yaml

# 4. 从过滤后的查询集合中收集 reranker 候选池。
$PYTHON_BIN training/reranker_distill/scripts/collect_candidate_pool.py \
  --config-path training/reranker_distill/configs/data_build.example.yaml

# 或者直接针对本地 vLLM teacher 运行完整的 reranker 蒸馏数据集构建流程。
$PYTHON_BIN training/reranker_distill/scripts/build_distill_dataset.py \
  --data-config-path training/reranker_distill/configs/data_build.local_vllm_reranker.example.yaml \
  --split-config-path training/reranker_distill/configs/split.example.yaml \
  --export-config-path training/reranker_distill/configs/export.example.yaml

# 5. 使用更强的 reranker teacher API 给这些候选打分。
$PYTHON_BIN training/reranker_distill/scripts/score_with_teacher_reranker.py \
  --config-path training/reranker_distill/configs/data_build.example.yaml

# 6. 构建 pointwise reranker 标签。
$PYTHON_BIN training/reranker_distill/scripts/build_pointwise_labels.py \
  --config-path training/reranker_distill/configs/data_build.example.yaml

# 7. 切分 reranker 数据。
$PYTHON_BIN training/reranker_distill/scripts/split_train_dev_test.py \
  --config-path training/reranker_distill/configs/split.example.yaml

# 8. 为训练器导出 reranker 的 train/dev/test 文件。
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

# 9. 将 pointwise 蒸馏记录导出为原生的 Qwen3-Reranker yes/no SFT 数据。
$PYTHON_BIN training/reranker_distill/scripts/export_to_qwen3_reranker_sft.py \
  --config-path training/reranker_distill/configs/sft_export.example.yaml

# 10. 使用 LoRA/QLoRA 训练 Qwen3-Reranker-0.6B 学生模型。
torchrun --nproc_per_node=2 training/reranker_distill/scripts/train_qwen3_reranker_sft.py \
  --config-path training/reranker_distill/configs/sft_train.example.yaml

# 11. 将过滤后的 generator 样本转换为 chat SFT 格式。
$PYTHON_BIN training/generator_sft/scripts/convert_to_chat_sft.py \
  --config-path training/generator_sft/configs/filter.example.yaml

# 12. 切分 generator chat 数据，并导出 LLaMA Factory 的 dataset_info.json。
$PYTHON_BIN training/generator_sft/scripts/split_train_dev_test.py \
  --config-path training/generator_sft/configs/split.example.yaml

# 13. 在 Qwen3.5-9B 上使用 QLoRA 训练生成器学生模型。
llamafactory-cli train training/generator_sft/configs/train.example.yaml
```
