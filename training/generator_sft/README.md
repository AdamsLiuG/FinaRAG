# Generator SFT 数据流水线

这个目录是为 FinaRAG 使用的答案生成模型构建监督微调数据的设计基线。

目标并不是训练一个自由对话的聊天机器人，而是训练一个能够消费基于检索
构建的上下文，并输出与当前在线流水线一致的结构化 schema 的模型。

相关在线代码：

- `src/questions_processing.py`
- `src/api_requests.py`
- `src/prompts.py`
- `src/table_grounding.py`
- `src/answer_validation.py`

## 1. 目标

使用 LoRA 训练一个类似 `Qwen3.5-9B` 的学生答案模型，使其能够：

- 稳定遵循现有 JSON schema
- 仅基于检索到的证据作答
- 在证据不足时拒答
- 改善在财报场景下对数字、名称、布尔和比较类问题的表现
- 与当前在线 prompt 和校验链保持兼容

## 2. 建议的目录结构

```text
training/generator_sft/
├── README.md
├── examples/
│   └── sft_chat_record.json
├── configs/
│   ├── data_build.example.yaml
│   └── train.example.yaml
├── scripts/
│   ├── build_seed_queries.py
│   ├── mine_teacher_answers.py
│   ├── filter_sft_samples.py
│   ├── convert_to_chat_sft.py
│   └── split_train_dev_test.py
├── manifests/
│   ├── query_sources.json
│   ├── build_stats.json
│   └── rejected_samples.jsonl
├── raw/
│   ├── seed_queries.jsonl
│   ├── teacher_answers_raw.jsonl
│   └── teacher_answers_debug.jsonl
├── processed/
│   ├── teacher_answers_filtered.jsonl
│   ├── train.chat.jsonl
│   ├── dev.chat.jsonl
│   └── test.chat.jsonl
└── cache/
    ├── retrieval_runs/
    └── prompt_cache/
```

## 3. 流水线阶段

### 阶段 A：构建种子查询

`build_seed_queries.py`

输入来源：

- `data/top10_industries_2024_20each/questions.json` 中现有的留出集问题风格
- `metadata_store/chunk_metadata.jsonl` 中的 chunk 元数据
- `metadata_store/company_master.jsonl` 中的公司元数据
- 可选的人工整理模板

输出：

- `raw/seed_queries.jsonl`

建议覆盖的查询类型：

- 单文档事实型
- 限定章节的事实型
- 元数据或标签检索型
- 跨文档比较型
- 布尔确认型
- 拒答场景

建议的初始比例：

- `number`：40%
- `name` 和 `names`：25%
- `boolean`：20%
- `comparative`：10%
- 拒答与超出范围：5% 到 10%

### 阶段 B：挖掘 Teacher 答案

`mine_teacher_answers.py`

每个查询的高层流程：

1. 运行当前最强的检索流水线。
2. 以与在线系统相同的格式构建 RAG 上下文。
3. 使用与在线一致的 schema prompt 家族调用答案 teacher。
4. 同时保存结构化答案和检索调试载荷。

Teacher 模型应该通过配置指定，而不是硬编码。

建议的配置键：

```yaml
teacher_answer_provider: qwen
teacher_answer_model: Qwen3.5-32B
teacher_verify_provider: qwen
teacher_verify_model: Qwen3.5-32B
retrieval_config_path: config/qwen_zh_finance_colbert_cascade_qwen.yaml
max_queries: 5000
parallel_requests: 4
```

### 阶段 C：过滤与校验

`filter_sft_samples.py`

过滤规则应尽可能复用项目里已有的逻辑：

- `relevant_pages` 在校验后必须是检索页的子集
- 答案必须能按预期 schema 成功解析
- 数字类问题应优先保留表格 grounding 成功的样本
- 非拒答的正样本至少应带有一个引用或参考依据
- `answer_validation.py` 里的校验标志不应显示严重不匹配
- 删除重复或近重复问题
- 对完全相同的上下文去重

建议的拒绝桶：

- schema_parse_failed
- no_retrieval
- hallucinated_pages
- weak_number_grounding
- severe_validation_flags
- duplicate_query
- duplicate_context
- empty_final_answer

### 阶段 D：转换为 Chat SFT

`convert_to_chat_sft.py`

输出格式应与 Qwen 风格模型的 chat SFT 训练格式保持一致。

每条样本应保留：

- system prompt
- 带检索上下文的 user prompt
- assistant 的 JSON 答案
- 便于追踪的轻量元数据

### 阶段 E：切分数据集

`split_train_dev_test.py`

建议的切分策略：

- 按 `report_id` 或 `company_name` 切分，而不只是随机按行切
- 保留当前的 `top10_industries_2024_20each` 作为留出基准
- 如果比较类问题数量有限，优先保留在 dev/test 中

## 4. 记录定义

### 4.1 `seed_queries.jsonl`

每行表示一条在生成 teacher 答案之前的候选查询。

```json
{
  "query_id": "seed-000001",
  "question_text": "华胜天成2024年年报中的营业收入是多少元？",
  "schema": "number",
  "task_type": "single_doc_fact",
  "company_name": "华胜天成",
  "mentioned_companies": [],
  "doc_ids": ["600410_2024_20250426"],
  "expected_filters": {
    "company_name": "华胜天成",
    "report_year": 2024
  },
  "source": "template_from_metadata",
  "difficulty": "medium",
  "should_refuse": false
}
```

字段定义：

| 字段 | 类型 | 必填 | 含义 |
| --- | --- | --- | --- |
| `query_id` | string | 是 | 用于追踪的稳定 ID |
| `question_text` | string | 是 | 最终自然语言查询 |
| `schema` | string | 是 | `name`、`number`、`boolean`、`names`、`comparative` 之一 |
| `task_type` | string | 是 | 查询家族标签 |
| `company_name` | string or null | 否 | 单文档任务的主公司 |
| `mentioned_companies` | list[string] | 是 | 比较任务中提到的公司 |
| `doc_ids` | list[string] | 否 | 若已知则写目标报告 ID |
| `expected_filters` | object | 否 | 基于元数据推导出的检索提示 |
| `source` | string | 是 | 查询来源 |
| `difficulty` | string | 否 | 可选的难度桶 |
| `should_refuse` | bool | 是 | 理想答案是否应当拒答 |

### 4.2 `teacher_answers_raw.jsonl`

每行表示过滤前的一次 teacher 运行结果。

```json
{
  "query_id": "seed-000001",
  "question_text": "华胜天成2024年年报中的营业收入是多少元？",
  "schema": "number",
  "teacher_answer_model": "Qwen3.5-32B",
  "retrieval_config": "config/qwen_zh_finance_colbert_cascade_qwen.yaml",
  "rag_context": "Text retrieved from page 9 ...",
  "retrieval_pages": [9, 8, 10],
  "retrieval_results": [],
  "answer": {
    "step_by_step_analysis": "...",
    "reasoning_summary": "...",
    "relevant_pages": [9],
    "final_answer": 4270629476.42
  },
  "response_data": {
    "model": "Qwen3.5-32B",
    "input_tokens": 5321,
    "output_tokens": 244
  },
  "table_grounding_result": null,
  "validation_result": null,
  "build_timestamp": "2026-04-08T00:00:00Z"
}
```

字段定义：

| 字段 | 类型 | 必填 | 含义 |
| --- | --- | --- | --- |
| `query_id` | string | 是 | 可关联回种子查询 |
| `question_text` | string | 是 | 原始查询 |
| `schema` | string | 是 | 任务 schema |
| `teacher_answer_model` | string | 是 | 使用的答案 teacher |
| `retrieval_config` | string | 是 | 挖掘时使用的检索栈 |
| `rag_context` | string | 是 | 提供给 teacher 的完整上下文 |
| `retrieval_pages` | list[int] | 是 | 按检索顺序排列的页码 |
| `retrieval_results` | list[object] | 是 | 序列化后的检索证据 |
| `answer` | object | 是 | 在线 schema 下的原始 teacher 答案 |
| `response_data` | object | 否 | token 用量和模型信息 |
| `table_grounding_result` | object or null | 否 | 数值 grounding 结果 |
| `validation_result` | object or null | 否 | 答案校验结果 |
| `build_timestamp` | string | 是 | UTC 时间戳 |

### 4.3 `teacher_answers_filtered.jsonl`

每行表示过滤后被接受的一条 SFT 样本。

```json
{
  "sample_id": "sft-000001",
  "query_id": "seed-000001",
  "question_text": "华胜天成2024年年报中的营业收入是多少元？",
  "schema": "number",
  "system_prompt": "...",
  "user_prompt": "...",
  "assistant_response_json": {
    "step_by_step_analysis": "...",
    "reasoning_summary": "...",
    "relevant_pages": [9],
    "final_answer": 4270629476.42
  },
  "doc_ids": ["600410_2024_20250426"],
  "company_name": "华胜天成",
  "retrieval_pages": [9, 8, 10],
  "accepted_checks": [
    "schema_ok",
    "page_alignment_ok",
    "number_grounding_ok"
  ],
  "source": "teacher_filtered"
}
```

字段定义：

| 字段 | 类型 | 必填 | 含义 |
| --- | --- | --- | --- |
| `sample_id` | string | 是 | 稳定的训练样本 ID |
| `query_id` | string | 是 | 来源查询 ID |
| `question_text` | string | 是 | 原始查询文本 |
| `schema` | string | 是 | prompt 家族 |
| `system_prompt` | string | 是 | 用于 SFT 的完整 system prompt |
| `user_prompt` | string | 是 | 带上下文的完整 user prompt |
| `assistant_response_json` | object | 是 | 目标结构化答案 |
| `doc_ids` | list[string] | 否 | 源报告 ID |
| `company_name` | string or null | 否 | 主公司 |
| `retrieval_pages` | list[int] | 是 | 展示给模型的页码 |
| `accepted_checks` | list[string] | 是 | 通过的质量检查 |
| `source` | string | 是 | 来源标签 |

### 4.4 `train.chat.jsonl` 与 `dev.chat.jsonl`

最终的 SFT 训练格式。

最小必需结构：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "..."
    },
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "{\"step_by_step_analysis\":\"...\",\"reasoning_summary\":\"...\",\"relevant_pages\":[9],\"final_answer\":4270629476.42}"
    }
  ],
  "meta": {
    "sample_id": "sft-000001",
    "schema": "number",
    "company_name": "华胜天成",
    "doc_ids": ["600410_2024_20250426"],
    "source": "teacher_filtered"
  }
}
```

## 5. 质量门槛

建议的硬性验收标准：

- schema 解析成功率高于 98%
- 正样本页码对齐率高于 95%
- 数字类样本的 grounding 成功率高于 85%
- 拒答样本的人工抽检精度高于 90%
- 重复查询比例低于 5%

## 6. FinaRAG 特定说明

- 将 `top10_industries_2024_20each` 保留为留出集，而不是主训练集。
- 优先复用 `src/prompts.py` 中的在线 prompt 家族。
- 优先保存来自 `QuestionsProcessor` 的序列化检索结果，以便后续审计
  每一条 SFT 样本。
- 对数字类问题来说，如果出现冲突，应认为 grounding 后的值比纯
  teacher 文本更可信。
