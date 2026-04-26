# finance_eval_benchmark_v2

`finance_eval_benchmark_v2` 是在 `finance_eval_benchmark_v1` Core60 的基础上扩展得到的正式评测集，提供两套规模：

- `core120/`: 120 题精简正式集
- `core200/`: 200 题完整正式集

## 扩展来源

- 保留 `benchmark_v1` 的 Core60 原题
- 基于原始数值题派生单位换算与阈值判断题
- 基于原始 metadata 名单题派生成员统计题
- 基于原始比较题派生布尔比较题
- 引入不与 Core60 冲突的 `teacher_answers_filtered` 拒答样本

## 重点新增场景

- 多跳/多文档:
  - `metadata_count_aggregation`
  - `compare_boolean_variant`
- 表格数值:
  - `numeric_unit_wanyuan`
  - `numeric_unit_yiyuan`
  - `numeric_unit_baiwan`
  - `numeric_threshold_boolean`
- 拒答:
  - `refusal_teacher_verified`

## 使用方式

校验 `core120`:

```bash
.venv/bin/python -m eval.validate_dataset   --questions-file data/finance_eval_benchmark_v2/core120/questions.json   --gold-answers-file data/finance_eval_benchmark_v2/core120/answers_gold.json   --manifest-file data/finance_eval_benchmark_v2/core120/dataset_manifest.json
```

校验 `core200`:

```bash
.venv/bin/python -m eval.validate_dataset   --questions-file data/finance_eval_benchmark_v2/core200/questions.json   --gold-answers-file data/finance_eval_benchmark_v2/core200/answers_gold.json   --manifest-file data/finance_eval_benchmark_v2/core200/dataset_manifest.json
```

运行 `core200` 正式评测:

```bash
.venv/bin/python -m eval.finance_eval   --questions-file data/finance_eval_benchmark_v2/core200/questions.json   --gold-answers-file data/finance_eval_benchmark_v2/core200/answers_gold.json   --pred-answers-file path/to/answers.finance_eval.json   --debug-file path/to/answers.finance_eval_debug.json
```

## 质量说明

这套 v2 benchmark 比 v1 更完备，但仍属于内部正式 benchmark 候选集，不建议直接对外宣称最终分数。特别是：

- 单位换算题需复核换算精度与保留位数
- metadata 聚合题需复核名单口径与计数口径
- refusal 样本当前以“法定代表人证据不足”类为主，后续仍值得继续补充更多类型

建议优先使用每个子目录下的 `review_checklist.csv` 做一轮人工复核。
