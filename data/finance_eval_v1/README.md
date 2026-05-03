# finance_eval_v1

`finance_eval_v1` 是 FinaRAG 的第一版金融问答评测数据 schema 与样例目录，目标是把现有页码级评测升级成“检索 + 答案 + 引用”三层闭环。

## 目录说明

- `questions.sample.json`: 金融评测问题集样例
- `answers_gold.sample.json`: 对应的金标答案样例
- `pred_answers.sample.json`: 用于 smoke test 的预测结果样例
- `pred_answers_debug.sample.json`: 对应的 debug bundle 样例
- `dataset_manifest.sample.json`: 数据集元信息样例

## 核心字段

- `question_id` / `question_text`: 题目唯一标识与问题文本
- `kind`: 题型，建议使用 `name` / `number` / `boolean` / `comparative` / `names`
- `doc_ids`: 标准答案依赖的文档 ID 列表
- `gold_value`: 问题级金标答案
- `gold_pages`: 问题级金标页码，使用 1-based 页码
- `metric_name` / `currency` / `unit` / `period`: 金融实体与口径字段
- `references`: 答案级引用，使用 `pdf_sha1 + page_index(0-based)` 表示
- `should_refuse`: 是否应该拒答

## 使用方式

校验 schema:

```bash
.venv/bin/python -m eval.validate_dataset \
  --questions-file data/finance_eval_v1/questions.sample.json \
  --gold-answers-file data/finance_eval_v1/answers_gold.sample.json \
  --manifest-file data/finance_eval_v1/dataset_manifest.sample.json
```

运行第一版金融评测:

```bash
.venv/bin/python -m eval.finance_eval \
  --questions-file data/finance_eval_v1/questions.sample.json \
  --gold-answers-file data/finance_eval_v1/answers_gold.sample.json \
  --pred-answers-file data/finance_eval_v1/pred_answers.sample.json \
  --debug-file data/finance_eval_v1/pred_answers_debug.sample.json
```

启用 RAGAS 真正打分:

```bash
.venv/bin/python -m eval.finance_eval \
  --questions-file data/finance_eval_v1/questions.sample.json \
  --gold-answers-file data/finance_eval_v1/answers_gold.sample.json \
  --pred-answers-file data/finance_eval_v1/pred_answers.sample.json \
  --debug-file data/finance_eval_v1/pred_answers_debug.sample.json \
  --ragas-llm-base-url "$RAGAS_LLM_BASE_URL" \
  --ragas-llm-api-key "$RAGAS_LLM_API_KEY" \
  --ragas-llm-model "${RAGAS_LLM_MODEL:-Qwen3.5-35B-A3B-AWQ-4bit}" \
  --ragas-embedding-provider huggingface \
  --ragas-embedding-model "${RAGAS_EMBEDDING_MODEL:-BAAI/bge-m3}"
```

如果已经在 `.env` 或 shell 里配置了 `RAGAS_LLM_BASE_URL / RAGAS_LLM_API_KEY / RAGAS_LLM_MODEL / RAGAS_EMBEDDING_MODEL`，则可以直接运行，不必额外传参。RAGAS evaluator 现在与线上答题模型配置解耦，不会再复用 `QWEN_*`。若只想跑原有评测而跳过 RAGAS，可追加 `--disable-ragas`。

## 设计原则

- 先兼容你当前仓库里的 `question_text / value / references / citations` 结构
- 对 `company_name / stock_code / report_year / currency / unit / doc_ids` 做结构化实体校验
- RAGAS 使用 `ragas 0.4.x` 的 `AnswerCorrectness / Faithfulness / AnswerRelevancy` 三项指标，并把均值写入 `ragas_score`
- 默认走 OpenAI 兼容的 LLM endpoint 做 evaluator，embedding 默认走本地 HuggingFace 模型，兼顾现有 Qwen 配置与离线 embedding 能力
