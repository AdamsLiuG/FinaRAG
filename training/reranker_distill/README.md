# Reranker 蒸馏流水线

这个目录是从更强的 teacher（例如 `Qwen3-Reranker-4B`）蒸馏出紧凑型
reranker（例如 `Qwen3-Reranker-0.6B`）的设计基线。

目标训练形式是 pointwise 蒸馏，并可选加入 hard label。

相关在线代码：

- `src/reranking.py`
- `src/retrieval.py`
- `src/questions_processing.py`
- `config/qwen_zh_finance_colbert_cascade_qwen.yaml`

## 1. 目标

训练一个小型 reranker 学生模型，使其能够：

- 将真正包含答案的段落排在相邻干扰项之前
- 保留与年份、公司、指标、章节和单位相关的金融精度
- 能通过现有兼容 `/v1/rerank` 的路径部署
- 改善下游检索指标和端到端回答质量

## 2. 建议的目录结构

```text
training/reranker_distill/
├── README.md
├── examples/
│   └── pointwise_record.json
├── configs/
│   ├── data_build.example.yaml
│   ├── data_build.local_vllm_reranker.example.yaml
│   └── train.example.yaml
├── scripts/
│   ├── build_distill_dataset.py
│   ├── collect_candidate_pool.py
│   ├── score_with_teacher_reranker.py
│   ├── build_pointwise_labels.py
│   ├── split_train_dev_test.py
│   └── export_for_trainer.py
├── manifests/
│   ├── build_stats.json
│   ├── rejected_pairs.jsonl
│   └── teacher_score_histogram.json
├── raw/
│   ├── candidate_pool.jsonl
│   ├── teacher_scores.jsonl
│   └── pointwise_labels_raw.jsonl
├── processed/
│   ├── pointwise_train.jsonl
│   ├── pointwise_dev.jsonl
│   └── pointwise_test.jsonl
└── cache/
    ├── rerank_requests/
    └── retrieval_candidates/
```

## 3. 流水线阶段

### 本地 vLLM Teacher 快速开始

如果你更强的 teacher reranker 已经通过本地、兼容 OpenAI 的
`/v1/rerank` 接口提供服务，可以用一条命令跑完整个数据集构建流程：

```bash
cd /media/main/lgd/llm/FinaRAG
PYTHON_BIN=/media/main/lgd/llm/FinaRAG/.venv/bin/python

$PYTHON_BIN training/reranker_distill/scripts/build_distill_dataset.py \
  --data-config-path training/reranker_distill/configs/data_build.local_vllm_reranker.example.yaml \
  --split-config-path training/reranker_distill/configs/split.example.yaml \
  --export-config-path training/reranker_distill/configs/export.example.yaml
```

本地 vLLM 示例配置会从 `.env` 读取以下值：

- `RERANKING_BASE_URL`
- `RERANKING_MODEL`
- `RERANKING_API_KEY`

### 阶段 A：收集候选池

`collect_candidate_pool.py`

对每个高质量查询：

1. 在最终 top-k 截断之前运行混合检索器。
2. 保留更大的候选集，例如 20 到 50 个段落。
3. 保留候选来源信息，包括检索来源和原始分数。

注意事项：

- 候选收集应发生在最后一次 reranking 截断之前
- 使用与生产环境一致的检索家族
- 尽量保留来自同一公司、同一份报告的 hard negative

### 阶段 B：使用 Teacher Reranker 打分

`score_with_teacher_reranker.py`

对于每个 `(query, passage)` 对：

1. 调用 `Qwen3-Reranker-4B`
2. 保存 teacher 分数
3. 保存 teacher 排序产生的名次

Teacher 应通过兼容 `/v1/rerank` 的接口进行配置，这样就可以复用
`src/reranking.py` 中现有的 API 风格。

### 阶段 C：构建 Pointwise 标签

`build_pointwise_labels.py`

仅有软分数还不够，需要构建混合标签：

- `label = 2`：被直接引用，或位于已验证的相关页上
- `label = 1`：语义相关，或是同页邻近证据
- `label = 0`：hard negative

最佳实践是组合以下信号：

- teacher reranker 分数
- 答案 teacher 的页码引用
- 引文命中
- 数字类问题的表格 grounding 命中

### 阶段 D：切分数据集

`split_train_dev_test.py`

建议的切分策略：

- 按 `report_id` 或 `company_name` 切分
- 不要把同一问题的 pair 随机拆到不同集合
- 为完整 RAG 评估保留一个单独的下游基准集

### 阶段 E：导出给训练器

`export_for_trainer.py`

将 pointwise 记录导出为所选训练器期望的格式。

建议的训练目标：

- 对通用 reranker，保留 pointwise teacher 分数和 hard label
- 对 `Qwen3-Reranker-0.6B`，把 pointwise 记录转换为原生 `yes/no`
  监督，并将其作为 causal-LM reranker 训练

## 4. 记录定义

### 4.1 `candidate_pool.jsonl`

每行表示 teacher reranking 之前的一条查询记录。

```json
{
  "query_id": "seed-000001",
  "question_text": "华胜天成2024年年报中的营业收入是多少元？",
  "schema": "number",
  "doc_ids": ["600410_2024_20250426"],
  "candidates": [
    {
      "candidate_id": "cand-0001",
      "doc_id": "600410_2024_20250426",
      "page": 9,
      "chunk_id": 321,
      "text": "......",
      "retrieval_sources": ["vector", "bm25"],
      "base_score": 0.8421,
      "section_name": "公司简介和主要财务指标"
    }
  ]
}
```

字段定义：

| 字段 | 类型 | 必填 | 含义 |
| --- | --- | --- | --- |
| `query_id` | string | 是 | 稳定的查询标识 |
| `question_text` | string | 是 | 自然语言查询 |
| `schema` | string | 是 | 任务 schema |
| `doc_ids` | list[string] | 否 | 目标报告 ID |
| `candidates` | list[object] | 是 | teacher rerank 前的候选段落 |

候选对象字段：

| 字段 | 类型 | 必填 | 含义 |
| --- | --- | --- | --- |
| `candidate_id` | string | 是 | 查询内稳定的候选标识 |
| `doc_id` | string | 是 | 报告 ID |
| `page` | int | 是 | 在线 1-based 形式的页码索引 |
| `chunk_id` | int or null | 否 | 如果可用则填写 chunk ID |
| `text` | string | 是 | 发给 reranker 的段落文本 |
| `retrieval_sources` | list[string] | 是 | 候选来源，如 `vector`、`bm25`、`sparse`、`tag` |
| `base_score` | float | 是 | rerank 前分数 |
| `section_name` | string or null | 否 | 章节提示 |

### 4.2 `teacher_scores.jsonl`

每行表示一条候选在 teacher 打分之后的结果。

```json
{
  "query_id": "seed-000001",
  "candidate_id": "cand-0001",
  "question_text": "华胜天成2024年年报中的营业收入是多少元？",
  "teacher_reranker_model": "Qwen3-Reranker-4B",
  "teacher_score": 0.9382,
  "teacher_rank": 1,
  "doc_id": "600410_2024_20250426",
  "page": 9,
  "chunk_id": 321,
  "text": "......",
  "base_score": 0.8421,
  "retrieval_sources": ["vector", "bm25"]
}
```

字段定义：

| 字段 | 类型 | 必填 | 含义 |
| --- | --- | --- | --- |
| `query_id` | string | 是 | 查询标识 |
| `candidate_id` | string | 是 | 候选标识 |
| `question_text` | string | 是 | 原始查询 |
| `teacher_reranker_model` | string | 是 | teacher 模型名 |
| `teacher_score` | float | 是 | 归一化软目标 |
| `teacher_rank` | int | 是 | teacher 排序中的名次 |
| `doc_id` | string | 是 | 报告 ID |
| `page` | int | 是 | 页码 |
| `chunk_id` | int or null | 否 | chunk ID |
| `text` | string | 是 | 段落文本 |
| `base_score` | float | 是 | rerank 前分数 |
| `retrieval_sources` | list[string] | 是 | 候选来源信息 |

### 4.3 `pointwise_labels_raw.jsonl`

每行表示结合软监督与硬监督之后的一条 pair 记录。

```json
{
  "pair_id": "pair-000001",
  "query_id": "seed-000001",
  "candidate_id": "cand-0001",
  "query": "华胜天成2024年年报中的营业收入是多少元？",
  "passage": "......",
  "schema": "number",
  "teacher_score": 0.9382,
  "teacher_rank": 1,
  "hard_label": 2,
  "label_source": [
    "teacher_reranker",
    "answer_relevant_page",
    "citation_hit"
  ],
  "doc_id": "600410_2024_20250426",
  "page": 9,
  "chunk_id": 321,
  "base_score": 0.8421,
  "retrieval_sources": ["vector", "bm25"],
  "section_name": "公司简介和主要财务指标",
  "is_hard_negative": false
}
```

字段定义：

| 字段 | 类型 | 必填 | 含义 |
| --- | --- | --- | --- |
| `pair_id` | string | 是 | 唯一 pair 标识 |
| `query_id` | string | 是 | 父查询标识 |
| `candidate_id` | string | 是 | 候选标识 |
| `query` | string | 是 | 训练用查询文本 |
| `passage` | string | 是 | 训练用段落文本 |
| `schema` | string | 是 | 查询 schema |
| `teacher_score` | float | 是 | 来自 teacher reranker 的软目标 |
| `teacher_rank` | int | 是 | teacher 排名 |
| `hard_label` | int | 是 | `0`、`1` 或 `2` |
| `label_source` | list[string] | 是 | hard label 的赋值原因 |
| `doc_id` | string | 是 | 报告 ID |
| `page` | int | 是 | 页码 |
| `chunk_id` | int or null | 否 | chunk ID |
| `base_score` | float | 是 | rerank 前的检索分数 |
| `retrieval_sources` | list[string] | 是 | 候选来源信息 |
| `section_name` | string or null | 否 | 可选章节提示 |
| `is_hard_negative` | bool | 是 | 该 pair 是否是刻意构造的强负样本 |

### 4.4 `pointwise_train.jsonl`

这是 reranker 训练器消费的最终训练文件。

最小结构：

```json
{
  "query": "华胜天成2024年年报中的营业收入是多少元？",
  "passage": "......",
  "teacher_score": 0.9382,
  "hard_label": 2,
  "meta": {
    "pair_id": "pair-000001",
    "query_id": "seed-000001",
    "candidate_id": "cand-0001",
    "doc_id": "600410_2024_20250426",
    "page": 9,
    "schema": "number",
    "is_hard_negative": false
  }
}
```

## 5. Hard-Negative 策略

reranker 从困难负样本中获得的收益最大，而不是从随机负样本中。

建议的 hard-negative 类别：

- 同公司、同报告、错误页码
- 同公司、同章节家族、错误指标
- 同一张表、错误行
- 同一指标术语、错误年份或期间
- 同行业、不同公司
- 语义相关但并不回答问题的叙述性文本

最终 pair 集合中的建议比例：

- `hard_label=2` 的强正样本：20% 到 30%
- `hard_label=1` 的中等正样本：20% 到 30%
- `hard_label=0` 的 hard negative：40% 到 60%

## 6. 质量门槛

建议的验收检查：

- 每个查询都至少要有一个 `hard_label=2`
- 每个查询都至少要有三个负样本
- 分数分布不应塌缩到接近 0 或 1
- dev 和 test 切分中应包含未见过的报告
- 数字类问题的正样本在可能的情况下应与 grounded 页码对齐

## 7. 部署映射

训练完成后，学生 reranker 应通过 `VLLMApiReranker` 期望的同一套 API
形状暴露出来：

- endpoint：`/v1/rerank`
- input：`model`、`query`、`documents`、`top_n`
- output：`results[index, relevance_score]`

这样可以保持在线集成简单，并让 `src/reranking.py` 中现有的代码路径继续
工作。

## 8. Qwen3-Reranker-0.6B 原生 SFT

`Qwen3-Reranker-0.6B` 不是一个普通的 sequence-classification checkpoint。
它的原生推理路径是一个 causal-LM prompt，用来对 `Query` + `Document`
pair 预测 `yes` 或 `no`。因此，这个仓库里推荐的学生训练路径是：

1. 使用上面的脚本先构建 pointwise teacher-score 数据集。
2. 将这些 pointwise 记录转换成二分类 `yes/no` SFT 样本。
3. 以 causal LM 的方式使用 LoRA/QLoRA 训练模型。

示例命令：

```bash
cd /media/main/lgd/llm/FinaRAG
PYTHON_BIN=/media/main/lgd/llm/FinaRAG/.venv/bin/python

$PYTHON_BIN training/reranker_distill/scripts/export_to_qwen3_reranker_sft.py \
  --config-path training/reranker_distill/configs/sft_export.example.yaml

torchrun --nproc_per_node=2 training/reranker_distill/scripts/train_qwen3_reranker_sft.py \
  --config-path training/reranker_distill/configs/sft_train.example.yaml
```

标签转换默认规则：

- 正样本：`hard_label=2`
- 负样本：`hard_label=0`
- 仅当 `hard_label` 缺失或含义不明确时，才回退为只使用 teacher 分数

这样既能让上游数据构建兼容现有 `RAG-Retrieval` 风格的 pointwise 挖掘流，
也能与 Qwen3 reranker 模型本身文档中的原生 prompt 格式保持一致。
