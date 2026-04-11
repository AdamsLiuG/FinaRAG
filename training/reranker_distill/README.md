# Reranker Distillation Pipeline

This directory is the design anchor for distilling a compact reranker such as
`Qwen3-Reranker-0.6B` from a stronger teacher such as `Qwen3-Reranker-4B`.

The target training style is pointwise distillation with optional hard labels.

Relevant online code:

- `src/reranking.py`
- `src/retrieval.py`
- `src/questions_processing.py`
- `config/qwen_zh_finance_colbert_cascade_qwen.yaml`

## 1. Target Objective

Train a small reranker student that:

- ranks answer-bearing passages above nearby distractors
- preserves financial precision around year, company, metric, section, and unit
- can be deployed through the existing `/v1/rerank` compatible path
- improves downstream retrieval metrics and end-to-end answer quality

## 2. Proposed Directory Layout

```text
training/reranker_distill/
├── README.md
├── examples/
│   └── pointwise_record.json
├── configs/
│   ├── data_build.example.yaml
│   └── train.example.yaml
├── scripts/
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

## 3. Pipeline Stages

### Stage A. Collect Candidate Pool

`collect_candidate_pool.py`

For each high-quality query:

1. Run the hybrid retriever before final top-k truncation.
2. Preserve a larger candidate set, such as 20 to 50 passages.
3. Keep candidate provenance including retrieval source and original score.

Important:

- candidate collection should happen before the last reranking cut
- use the same retrieval family intended for production
- preserve hard negatives from the same company and same report whenever possible

### Stage B. Score with Teacher Reranker

`score_with_teacher_reranker.py`

For each `(query, passage)` pair:

1. Call `Qwen3-Reranker-4B`
2. Save the teacher score
3. Save the rank position induced by teacher sorting

Teacher should be configurable through a `/v1/rerank` compatible endpoint so
that it can reuse the existing API style in `src/reranking.py`.

### Stage C. Build Pointwise Labels

`build_pointwise_labels.py`

Soft scores alone are not enough. Build hybrid labels:

- `label = 2`: directly cited or on validated relevant pages
- `label = 1`: semantically relevant or same-page neighboring evidence
- `label = 0`: hard negative

Best practice is to combine:

- teacher reranker score
- answer teacher page references
- citation hits
- table grounding hits for number questions

### Stage D. Split the Dataset

`split_train_dev_test.py`

Recommended split strategy:

- split by `report_id` or `company_name`
- do not randomly split pairs from the same question into different sets
- keep a separate downstream benchmark for full RAG evaluation

### Stage E. Export for Trainer

`export_for_trainer.py`

Export the pointwise records into the format expected by the chosen trainer.

Recommended training target:

- main target: teacher score regression or KL-style soft target
- auxiliary target: hard label classification

## 4. Record Definitions

### 4.1 `candidate_pool.jsonl`

One line per query before teacher reranking.

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

Field definitions:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `query_id` | string | yes | stable query identifier |
| `question_text` | string | yes | natural-language query |
| `schema` | string | yes | task schema |
| `doc_ids` | list[string] | no | intended reports |
| `candidates` | list[object] | yes | candidate passages before teacher rerank |

Candidate object fields:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `candidate_id` | string | yes | stable candidate identifier within the query |
| `doc_id` | string | yes | report identifier |
| `page` | int | yes | page index in 1-based online form |
| `chunk_id` | int or null | no | chunk identifier if available |
| `text` | string | yes | passage text sent to reranker |
| `retrieval_sources` | list[string] | yes | provenance such as `vector`, `bm25`, `sparse`, `tag` |
| `base_score` | float | yes | pre-rerank score |
| `section_name` | string or null | no | section hint |

### 4.2 `teacher_scores.jsonl`

One line per candidate after teacher scoring.

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

Field definitions:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `query_id` | string | yes | query identifier |
| `candidate_id` | string | yes | candidate identifier |
| `question_text` | string | yes | original query |
| `teacher_reranker_model` | string | yes | teacher model name |
| `teacher_score` | float | yes | normalized soft target |
| `teacher_rank` | int | yes | rank under teacher ordering |
| `doc_id` | string | yes | report identifier |
| `page` | int | yes | page number |
| `chunk_id` | int or null | no | chunk identifier |
| `text` | string | yes | passage text |
| `base_score` | float | yes | pre-rerank score |
| `retrieval_sources` | list[string] | yes | candidate provenance |

### 4.3 `pointwise_labels_raw.jsonl`

One line per pair after combining soft and hard supervision.

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

Field definitions:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `pair_id` | string | yes | unique pair identifier |
| `query_id` | string | yes | parent query identifier |
| `candidate_id` | string | yes | candidate identifier |
| `query` | string | yes | query text for training |
| `passage` | string | yes | passage text for training |
| `schema` | string | yes | query schema |
| `teacher_score` | float | yes | soft target from teacher reranker |
| `teacher_rank` | int | yes | teacher order |
| `hard_label` | int | yes | `0`, `1`, or `2` |
| `label_source` | list[string] | yes | why the hard label was assigned |
| `doc_id` | string | yes | report ID |
| `page` | int | yes | page |
| `chunk_id` | int or null | no | chunk ID |
| `base_score` | float | yes | retriever score before rerank |
| `retrieval_sources` | list[string] | yes | candidate provenance |
| `section_name` | string or null | no | optional section hint |
| `is_hard_negative` | bool | yes | whether this pair is a deliberately strong negative |

### 4.4 `pointwise_train.jsonl`

This is the final training file consumed by the reranker trainer.

Minimal structure:

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

## 5. Hard-Negative Policy

The reranker gains most value from difficult negatives, not random negatives.

Recommended hard-negative categories:

- same company, same report, wrong page
- same company, same section family, wrong metric
- same table, wrong row
- same metric term, wrong year or period
- same industry, different company
- semantically related narrative that does not answer the question

Recommended ratio inside the final pair set:

- strong positives with `hard_label=2`: 20% to 30%
- medium positives with `hard_label=1`: 20% to 30%
- hard negatives with `hard_label=0`: 40% to 60%

## 6. Quality Gates

Recommended acceptance checks:

- every query should have at least one `hard_label=2`
- every query should have at least three negatives
- score distribution should not collapse near 0 or 1
- dev and test splits should contain unseen reports
- number-question positives should align with grounded pages when possible

## 7. Deployment Mapping

After training, the student reranker should be exposed through the same API
shape expected by `VLLMApiReranker`:

- endpoint: `/v1/rerank`
- input: `model`, `query`, `documents`, `top_n`
- output: `results[index, relevance_score]`

This keeps the online integration simple and lets the existing code path in
`src/reranking.py` continue to work.
