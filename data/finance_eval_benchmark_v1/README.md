# finance_eval_benchmark_v1

`finance_eval_benchmark_v1` 是基于 `top10_industries_2024_20each` 构建的正式金融 RAG 评测基准目录，面向 `eval.finance_eval` 的结构化评测管线。

## 目录结构

- `questions.json`: Core60 正式题集
- `answers_gold.json`: Core60 金标答案
- `dataset_manifest.json`: Core60 元信息
- `review_checklist.csv`: 人工复核清单
- `splits/dev/`: 30 题开发集
- `splits/test/`: 30 题保留集

## 数据概览

- 总题量: 60
- 行业覆盖: 10 个一级行业
- 每行业题量: 6
- 能力覆盖:
  - `single_doc_fact`
  - `single_doc_boolean`
  - `section_filter`
  - `metadata_tag_retrieval`
  - `cross_doc_compare`
- 答案形态覆盖:
  - `name`
  - `number`
  - `boolean`
  - `names`

## Split 策略

为保证 `dev/test` 都覆盖所有行业与主要能力，本 benchmark 使用行业奇偶交错切分：

- 奇数行业组:
  - `dev`: slot `1/3/5`
  - `test`: slot `2/4/6`
- 偶数行业组:
  - `dev`: slot `2/4/6`
  - `test`: slot `1/3/5`

这样得到：

- `dev`: 30 题
- `test`: 30 题
- 两个 split 都包含 10 个行业
- 两个 split 都保留 `name/number/boolean/names` 四类答案与五类能力

## 质量说明

这套 benchmark 的标签来源于仓库内已有的 `top10_industries_2024_20each` 自动填充结果，并已补齐为 `finance_eval_v1` 正式 schema。

当前状态适合：

- 正式对比不同检索/生成配置
- 作为人工复核前的 benchmark 候选集
- 作为后续 120/200 题扩展的核心骨架

当前状态不建议直接用于对外宣称最终分数，因为：

- `annotation_status` 仍以 `auto_filled` 为主
- `metadata_tag_retrieval` 与 `cross_doc_compare` 题型仍建议人工复核
- 数值题仍应复核单位换算与页码

建议优先使用 `review_checklist.csv` 完成一轮人工核对，再将关键题目标记为 `human_verified`。

## 构建方式

如需重新生成 benchmark：

```bash
.venv/bin/python -m eval.build_finance_eval_benchmark
```

## 校验方式

校验 Core60:

```bash
.venv/bin/python -m eval.validate_dataset \
  --questions-file data/finance_eval_benchmark_v1/questions.json \
  --gold-answers-file data/finance_eval_benchmark_v1/answers_gold.json \
  --manifest-file data/finance_eval_benchmark_v1/dataset_manifest.json
```

校验 Dev split:

```bash
.venv/bin/python -m eval.validate_dataset \
  --questions-file data/finance_eval_benchmark_v1/splits/dev/questions.json \
  --gold-answers-file data/finance_eval_benchmark_v1/splits/dev/answers_gold.json \
  --manifest-file data/finance_eval_benchmark_v1/splits/dev/dataset_manifest.json
```

## 评测方式

如果你希望把下面三步串成一次执行：

- 跑 FinaRAG 主流程
- 导出正式 `pred/debug` 评测文件
- 调用 `finance_eval` 输出最终报告

可以直接使用一键脚本：

```bash
.venv/bin/python -m eval.run_finance_benchmark \
  --pipeline-dataset-dir data/top10_industries_2024_20each \
  --questions-file data/finance_eval_benchmark_v1/questions.json \
  --gold-answers-file data/finance_eval_benchmark_v1/answers_gold.json \
  --config qwen_base
```

说明：

- `--pipeline-dataset-dir` 必须指向已经准备好检索索引和文档的真实运行数据目录
- `finance_eval_benchmark_v1` 本身只存正式评测题与金标，不是可直接跑 Pipeline 的全文语料目录
- 如果你已经有原生 `answers.json`，也可以跳过 Pipeline，直接传 `--answers-file`

如果你先运行的是 FinaRAG 主流程，拿到的是原生 `answers*.json` / `answers*_debug.json`，建议先做一次正式评测导出：

```bash
.venv/bin/python -m eval.export_finance_eval_bundle \
  --questions-file data/finance_eval_benchmark_v1/questions.json \
  --answers-file path/to/answers.json
```

默认会在原答案文件旁边生成：

- `answers.finance_eval.json`
- `answers.finance_eval_debug.json`

导出器会自动：

- 按正式题集补齐 `question_id`
- 保持问题顺序与 benchmark 对齐
- 为缺失预测补 `N/A` 占位，避免漏答题在汇总时被忽略
- 在缺少原始 debug 时，根据 citation 合成最小可用 `debug` 文件

然后再运行正式评分。

运行 Core60:

```bash
.venv/bin/python -m eval.finance_eval \
  --questions-file data/finance_eval_benchmark_v1/questions.json \
  --gold-answers-file data/finance_eval_benchmark_v1/answers_gold.json \
  --pred-answers-file path/to/answers.finance_eval.json \
  --debug-file path/to/answers.finance_eval_debug.json
```
