# Generator SFT Data Pipeline

This directory is the design anchor for building supervised fine-tuning data
for the answer generation model used by FinaRAG.

The goal is not to train a free-form chatbot. The goal is to train a model
that consumes retrieval-grounded context and emits the same structured schema
already used in the online pipeline.

Relevant online code:

- `src/questions_processing.py`
- `src/api_requests.py`
- `src/prompts.py`
- `src/table_grounding.py`
- `src/answer_validation.py`

## 1. Target Objective

Train a student answer model such as `Qwen3.5-9B` with LoRA so that it:

- follows the existing JSON schema reliably
- answers using retrieved evidence only
- refuses when evidence is insufficient
- improves number/name/boolean/comparative behavior in financial reports
- remains compatible with the current online prompt and validation chain

## 2. Proposed Directory Layout

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

## 3. Pipeline Stages

### Stage A. Build Seed Queries

`build_seed_queries.py`

Input sources:

- existing holdout question styles from `data/top10_industries_2024_20each/questions.json`
- chunk metadata from `metadata_store/chunk_metadata.jsonl`
- company metadata from `metadata_store/company_master.jsonl`
- optional manually curated templates

Output:

- `raw/seed_queries.jsonl`

Query families to include:

- single-document fact
- section-constrained fact
- metadata/tag retrieval
- cross-document comparison
- boolean confirmation
- refusal cases

Recommended initial ratio:

- `number`: 40%
- `name` and `names`: 25%
- `boolean`: 20%
- `comparative`: 10%
- refusal and out-of-scope: 5% to 10%

### Stage B. Mine Teacher Answers

`mine_teacher_answers.py`

High-level flow for each query:

1. Run the strongest available retrieval pipeline.
2. Build RAG context in the same format as the online system.
3. Call the answer teacher with the same schema prompt family used online.
4. Save both the structured answer and the retrieval debug payload.

Teacher model should be configurable, not hard-coded.

Suggested config keys:

```yaml
teacher_answer_provider: qwen
teacher_answer_model: Qwen3.5-32B
teacher_verify_provider: qwen
teacher_verify_model: Qwen3.5-32B
retrieval_config_path: config/qwen_zh_finance_colbert_cascade_qwen.yaml
max_queries: 5000
parallel_requests: 4
```

### Stage C. Filter and Validate

`filter_sft_samples.py`

Filtering rules should reuse the existing project logic as much as possible:

- `relevant_pages` must be a subset of retrieved pages after validation
- answer must parse under the expected schema
- number questions should prefer samples with successful table grounding
- non-refusal positive samples should carry at least one citation or reference
- validation flags from `answer_validation.py` should not indicate severe mismatch
- duplicate or near-duplicate questions should be removed
- exact repeated contexts should be deduplicated

Suggested rejection buckets:

- schema_parse_failed
- no_retrieval
- hallucinated_pages
- weak_number_grounding
- severe_validation_flags
- duplicate_query
- duplicate_context
- empty_final_answer

### Stage D. Convert to Chat SFT

`convert_to_chat_sft.py`

Output format should match chat SFT training for Qwen-style models.

Each sample should preserve:

- system prompt
- user prompt with retrieval context
- assistant JSON answer
- lightweight metadata for tracking

### Stage E. Split Dataset

`split_train_dev_test.py`

Recommended split strategy:

- split by `report_id` or `company_name`, not by random row only
- keep the current `top10_industries_2024_20each` as a holdout benchmark
- keep comparison questions in dev/test if count is limited

## 4. Record Definitions

### 4.1 `seed_queries.jsonl`

One line per candidate query before teacher generation.

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

Field definitions:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `query_id` | string | yes | stable ID for traceability |
| `question_text` | string | yes | final natural-language query |
| `schema` | string | yes | one of `name`, `number`, `boolean`, `names`, `comparative` |
| `task_type` | string | yes | query family label |
| `company_name` | string or null | no | primary company for single-doc tasks |
| `mentioned_companies` | list[string] | yes | company mentions for comparison tasks |
| `doc_ids` | list[string] | no | intended target reports if known |
| `expected_filters` | object | no | retrieval hints derived from metadata |
| `source` | string | yes | origin of the query |
| `difficulty` | string | no | optional difficulty bucket |
| `should_refuse` | bool | yes | whether the ideal answer should refuse |

### 4.2 `teacher_answers_raw.jsonl`

One line per teacher run before filtering.

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

Field definitions:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `query_id` | string | yes | joins back to the seed query |
| `question_text` | string | yes | original query |
| `schema` | string | yes | task schema |
| `teacher_answer_model` | string | yes | answer teacher used |
| `retrieval_config` | string | yes | retrieval stack used during mining |
| `rag_context` | string | yes | exact context given to the teacher |
| `retrieval_pages` | list[int] | yes | pages in retrieved order |
| `retrieval_results` | list[object] | yes | serialized retrieval evidence |
| `answer` | object | yes | raw teacher answer in online schema |
| `response_data` | object | no | token usage and model info |
| `table_grounding_result` | object or null | no | numeric grounding outcome |
| `validation_result` | object or null | no | answer validation outcome |
| `build_timestamp` | string | yes | UTC timestamp |

### 4.3 `teacher_answers_filtered.jsonl`

One line per accepted SFT sample after filtering.

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

Field definitions:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `sample_id` | string | yes | stable training sample ID |
| `query_id` | string | yes | source query |
| `question_text` | string | yes | original query text |
| `schema` | string | yes | prompt family |
| `system_prompt` | string | yes | exact system prompt used for SFT |
| `user_prompt` | string | yes | exact user prompt with context |
| `assistant_response_json` | object | yes | target structured answer |
| `doc_ids` | list[string] | no | source reports |
| `company_name` | string or null | no | primary company |
| `retrieval_pages` | list[int] | yes | pages shown to the model |
| `accepted_checks` | list[string] | yes | passed quality checks |
| `source` | string | yes | provenance label |

### 4.4 `train.chat.jsonl` and `dev.chat.jsonl`

Final SFT training format.

Minimal required structure:

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

## 5. Quality Gates

Recommended hard acceptance criteria:

- schema parse success rate above 98%
- positive sample page alignment above 95%
- number sample grounding success above 85%
- refusal sample manual spot-check precision above 90%
- duplicate query ratio below 5%

## 6. Notes Specific to FinaRAG

- Keep `top10_industries_2024_20each` as holdout, not as the main training set.
- Prefer reusing the online prompt family from `src/prompts.py`.
- Prefer storing serialized retrieval results from `QuestionsProcessor` so that
  every SFT sample can be audited later.
- For number questions, treat grounded values as more trustworthy than pure
  teacher text when conflicts appear.
