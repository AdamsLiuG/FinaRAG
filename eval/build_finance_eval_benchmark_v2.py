from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from eval.dataset_schema import (
    FinanceEvalManifest,
    FinanceEvalQuestionSet,
    FinanceGoldAnswerSet,
    load_gold_answer_set,
    load_question_set,
    validate_dataset_alignment,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_BENCHMARK_DIR = ROOT / "data" / "finance_eval_benchmark_v1"
DOCUMENT_MANIFEST_FILE = ROOT / "data" / "top10_industries_2024_20each" / "document_manifest.csv"
SEED_QUERY_FILE = ROOT / "training" / "generator_sft" / "raw" / "seed_queries.jsonl"
REFUSAL_SAMPLE_FILE = ROOT / "training" / "generator_sft" / "processed" / "teacher_answers_filtered.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "finance_eval_benchmark_v2"


SCENARIO_PRIORITY = {
    "base_core": 0,
    "numeric_unit_wanyuan": 1,
    "numeric_unit_yiyuan": 2,
    "numeric_threshold_boolean": 3,
    "metadata_count_aggregation": 4,
    "compare_boolean_variant": 5,
    "refusal_teacher_verified": 6,
    "numeric_unit_baiwan": 7,
    "section_boolean_variant": 8,
    "metadata_membership_positive": 9,
    "metadata_membership_negative": 10,
}


REVIEW_PRIORITY = {
    "base_core": "high",
    "numeric_unit_wanyuan": "high",
    "numeric_unit_yiyuan": "high",
    "numeric_unit_baiwan": "high",
    "numeric_threshold_boolean": "high",
    "metadata_count_aggregation": "high",
    "compare_boolean_variant": "high",
    "section_boolean_variant": "medium",
    "metadata_membership_positive": "high",
    "metadata_membership_negative": "high",
    "refusal_teacher_verified": "high",
}


REVIEW_FOCUS = {
    "base_core": "复核原始 gold_value、页码与文档范围是否一致。",
    "numeric_unit_wanyuan": "复核从元到万元的单位换算与保留小数位。",
    "numeric_unit_yiyuan": "复核从元到亿元的单位换算与保留小数位。",
    "numeric_unit_baiwan": "复核从元到百万元的单位换算与保留小数位。",
    "numeric_threshold_boolean": "复核阈值是否来自原始数值，并确认判断方向 true/false 正确。",
    "metadata_count_aggregation": "复核名单长度、去重口径与统计口径是否一致。",
    "compare_boolean_variant": "复核比较赢家与布尔判断方向是否一致。",
    "section_boolean_variant": "复核章节限制与布尔答案是否仍然成立。",
    "metadata_membership_positive": "复核候选公司确属名单成员。",
    "metadata_membership_negative": "复核候选公司属于候选文档范围但不在名单中。",
    "refusal_teacher_verified": "复核拒答是否合理，确认上下文确实不足以直接回答。",
}


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def _load_document_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            indexed[row["doc_id"]] = row
    return indexed


def _deep_copy(payload: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(payload, ensure_ascii=False))


def _metadata_with(source: Dict[str, Any] | None, **extra: Any) -> Dict[str, Any]:
    metadata = dict(source or {})
    metadata.update(extra)
    return metadata


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _quantize(value: Decimal, places: int) -> float:
    exponent = Decimal("1").scaleb(-places)
    return float(value.quantize(exponent, rounding=ROUND_HALF_UP))


def _round_robin(groups: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    ordered_keys = sorted(groups.keys())
    pools = {key: list(values) for key, values in groups.items()}
    merged: List[Dict[str, Any]] = []
    while True:
        progressed = False
        for key in ordered_keys:
            bucket = pools.get(key) or []
            if not bucket:
                continue
            merged.append(bucket.pop(0))
            progressed = True
        if not progressed:
            break
    return merged


def _build_wrapped_sets(
    *,
    dataset_name: str,
    questions: List[Dict[str, Any]],
    answers: List[Dict[str, Any]],
    split_name: str | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    metadata = {
        "owner": "FinaRAG",
        "quality_tier": "provisional_gold",
        "review_required": True,
        "split_name": split_name,
    }
    question_set = FinanceEvalQuestionSet(
        schema_version="finance_eval_v1",
        dataset_name=dataset_name,
        questions=questions,
        metadata=metadata,
    )
    answer_set = FinanceGoldAnswerSet(
        schema_version="finance_eval_v1",
        dataset_name=dataset_name,
        answers=answers,
        metadata=metadata,
    )
    return question_set.model_dump(by_alias=True), answer_set.model_dump()


def _build_strata_summary(questions: Iterable[Dict[str, Any]], split_name: str | None = None) -> Dict[str, int]:
    summary: Dict[str, int] = {}

    def _bump(key: str) -> None:
        summary[key] = summary.get(key, 0) + 1

    for question in questions:
        _bump("all")
        _bump(f"kind/{question['kind']}")
        _bump(f"capability/{question['capability']}")
        _bump(f"difficulty/{question['difficulty']}")
        _bump(f"scenario/{question.get('metadata', {}).get('scenario_family', 'unknown')}")
        if question.get("group_name"):
            _bump(f"industry/{question['group_name']}")
        if question.get("should_refuse"):
            _bump("refusal/true")
        else:
            _bump("refusal/false")
        if split_name is not None:
            _bump(f"split/{split_name}")
    return dict(sorted(summary.items()))


def _build_manifest(
    *,
    dataset_name: str,
    description: str,
    source_corpora: List[str],
    questions: List[Dict[str, Any]],
    answers: List[Dict[str, Any]],
    split_name: str | None = None,
) -> Dict[str, Any]:
    scenario_summary: Dict[str, int] = {}
    for question in questions:
        scenario = question.get("metadata", {}).get("scenario_family", "unknown")
        scenario_summary[scenario] = scenario_summary.get(scenario, 0) + 1
    metadata = {
        "owner": "FinaRAG",
        "quality_tier": "provisional_gold",
        "review_required": True,
        "split_name": split_name,
        "annotation_policy": "core60_gold_plus_verified_derivations_plus_teacher_refusal_nonconflict",
        "split_strategy": "derived variants inherit parent split; refusal samples are alternated to keep split sizes balanced",
        "scenario_summary": dict(sorted(scenario_summary.items())),
    }
    manifest = FinanceEvalManifest(
        schema_version="finance_eval_v1",
        dataset_name=dataset_name,
        description=description,
        source_corpora=source_corpora,
        question_count=len(questions),
        answer_count=len(answers),
        scoring_profile="finance_rag_v2",
        strata_summary=_build_strata_summary(questions, split_name=split_name),
        metadata=metadata,
    )
    return manifest.model_dump()


def _write_review_checklist(path: Path, questions: List[Dict[str, Any]], split_by_id: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question_id",
                "split",
                "scenario_family",
                "parent_question_id",
                "kind",
                "capability",
                "difficulty",
                "group_name",
                "company_name",
                "mentioned_companies",
                "doc_ids",
                "gold_value",
                "gold_pages",
                "should_refuse",
                "annotation_status",
                "review_priority",
                "review_focus",
                "notes",
            ],
        )
        writer.writeheader()
        for question in questions:
            metadata = question.get("metadata", {})
            scenario = metadata.get("scenario_family", "unknown")
            writer.writerow(
                {
                    "question_id": question["id"],
                    "split": split_by_id[question["id"]],
                    "scenario_family": scenario,
                    "parent_question_id": metadata.get("parent_question_id"),
                    "kind": question.get("kind"),
                    "capability": question.get("capability"),
                    "difficulty": question.get("difficulty"),
                    "group_name": question.get("group_name"),
                    "company_name": question.get("company_name"),
                    "mentioned_companies": "|".join(question.get("mentioned_companies") or []),
                    "doc_ids": "|".join(question.get("doc_ids") or []),
                    "gold_value": json.dumps(question.get("gold_value"), ensure_ascii=False),
                    "gold_pages": "|".join(str(page) for page in question.get("gold_pages") or []),
                    "should_refuse": question.get("should_refuse"),
                    "annotation_status": question.get("annotation_status"),
                    "review_priority": REVIEW_PRIORITY.get(scenario, "medium"),
                    "review_focus": REVIEW_FOCUS.get(scenario, "复核问题、答案、页码与来源是否一致。"),
                    "notes": question.get("notes"),
                }
            )


def _write_readme(path: Path) -> None:
    content = """# finance_eval_benchmark_v2

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
.venv/bin/python -m eval.validate_dataset \
  --questions-file data/finance_eval_benchmark_v2/core120/questions.json \
  --gold-answers-file data/finance_eval_benchmark_v2/core120/answers_gold.json \
  --manifest-file data/finance_eval_benchmark_v2/core120/dataset_manifest.json
```

校验 `core200`:

```bash
.venv/bin/python -m eval.validate_dataset \
  --questions-file data/finance_eval_benchmark_v2/core200/questions.json \
  --gold-answers-file data/finance_eval_benchmark_v2/core200/answers_gold.json \
  --manifest-file data/finance_eval_benchmark_v2/core200/dataset_manifest.json
```

运行 `core200` 正式评测:

```bash
.venv/bin/python -m eval.finance_eval \
  --questions-file data/finance_eval_benchmark_v2/core200/questions.json \
  --gold-answers-file data/finance_eval_benchmark_v2/core200/answers_gold.json \
  --pred-answers-file path/to/answers.finance_eval.json \
  --debug-file path/to/answers.finance_eval_debug.json
```

## 质量说明

这套 v2 benchmark 比 v1 更完备，但仍属于内部正式 benchmark 候选集，不建议直接对外宣称最终分数。特别是：

- 单位换算题需复核换算精度与保留位数
- metadata 聚合题需复核名单口径与计数口径
- refusal 样本当前以“法定代表人证据不足”类为主，后续仍值得继续补充更多类型

建议优先使用每个子目录下的 `review_checklist.csv` 做一轮人工复核。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _question_dump(path: Path) -> List[Dict[str, Any]]:
    return [question.model_dump(by_alias=True) for question in load_question_set(path).questions]


def _answer_dump(path: Path) -> List[Dict[str, Any]]:
    return [answer.model_dump() for answer in load_gold_answer_set(path).answers]


def _metric_name(question: Dict[str, Any]) -> str | None:
    if question.get("metric_name"):
        return question["metric_name"]
    text = str(question.get("text") or "")
    if "法定代表人" in text:
        return "法定代表人"
    if "营业收入" in text:
        return "营业收入"
    if "现金分红" in text:
        return "现金分红"
    if "国产替代" in text:
        return "国产替代"
    return None


def _scenario_sort_key(question: Dict[str, Any]) -> Tuple[int, str]:
    scenario = question.get("metadata", {}).get("scenario_family", "unknown")
    return (SCENARIO_PRIORITY.get(scenario, 999), question["id"])


def _derive_group_name_from_doc(doc_meta: Dict[str, Dict[str, Any]], industry_map: Dict[str, str], doc_id: str) -> str | None:
    row = doc_meta.get(doc_id) or {}
    industry_l1 = row.get("industry_l1")
    if not industry_l1:
        return None
    return industry_map.get(industry_l1, industry_l1)


def _round_robin_name_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for candidate in sorted(candidates, key=lambda item: (item["group_name"] or "", item["id"])):
        grouped.setdefault(candidate["group_name"] or "unknown", []).append(candidate)
    return _round_robin(grouped)


def _make_answer_reference_list(doc_id: str | None, pages: Sequence[int]) -> List[Dict[str, Any]]:
    if not doc_id:
        return []
    return [{"pdf_sha1": doc_id, "page_index": max(int(page) - 1, 0)} for page in pages if int(page) > 0]


def _make_pair(question: Dict[str, Any], answer: Dict[str, Any], *, split: str) -> Dict[str, Any]:
    return {
        "question": question,
        "answer": answer,
        "split": split,
    }


def _numeric_value(answer: Dict[str, Any]) -> Decimal:
    return _decimal(answer["value"])


def _copy_base_question(question: Dict[str, Any]) -> Dict[str, Any]:
    copied = _deep_copy(question)
    copied["metadata"] = dict(copied.get("metadata") or {})
    return copied


def _copy_base_answer(answer: Dict[str, Any]) -> Dict[str, Any]:
    copied = _deep_copy(answer)
    copied["metadata"] = dict(copied.get("metadata") or {})
    return copied


def _make_numeric_unit_variant(
    question: Dict[str, Any],
    answer: Dict[str, Any],
    *,
    suffix: str,
    unit_text: str,
    divisor: Decimal,
    decimals: int,
    scenario_family: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    new_question = _copy_base_question(question)
    new_answer = _copy_base_answer(answer)
    company_name = question["company_name"]
    report_year = question["report_year"]
    converted = _quantize(_numeric_value(answer) / divisor, decimals)
    new_question["id"] = f"{question['id']}--{suffix}"
    if unit_text == "百万元":
        new_question["text"] = f"{company_name}{report_year}年年报中的营业收入折合多少百万元？"
    else:
        new_question["text"] = f"{company_name}{report_year}年年报中的营业收入是多少{unit_text}？"
    new_question["unit"] = unit_text
    new_question["difficulty"] = "medium"
    new_question["capability"] = "table_numeric_unit_normalization"
    new_question["gold_value"] = converted
    new_question["notes"] = f"由 {question['id']} 从元换算得到，目标单位为{unit_text}。"
    new_question["metadata"] = _metadata_with(
        new_question.get("metadata"),
        scenario_family=scenario_family,
        parent_question_id=question["id"],
        generation_method="deterministic_unit_conversion",
        source_dataset="finance_eval_benchmark_v1_core60",
    )

    new_answer["question_id"] = new_question["id"]
    new_answer["question_text"] = new_question["text"]
    new_answer["value"] = converted
    new_answer["unit"] = unit_text
    new_answer["difficulty"] = "medium"
    new_answer["capability"] = "table_numeric_unit_normalization"
    new_answer["notes"] = new_question["notes"]
    new_answer["metadata"] = _metadata_with(
        new_answer.get("metadata"),
        scenario_family=scenario_family,
        parent_question_id=question["id"],
        generation_method="deterministic_unit_conversion",
    )
    return new_question, new_answer


def _make_numeric_threshold_variant(
    question: Dict[str, Any],
    answer: Dict[str, Any],
    *,
    variant_index: int,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    new_question = _copy_base_question(question)
    new_answer = _copy_base_answer(answer)
    company_name = question["company_name"]
    report_year = question["report_year"]
    amount_yi = _numeric_value(answer) / Decimal("100000000")
    floor_value = int(amount_yi.to_integral_value(rounding="ROUND_FLOOR"))
    ceil_value = int(amount_yi.to_integral_value(rounding="ROUND_CEILING"))
    if variant_index % 2 == 0:
        threshold = max(1, floor_value)
        if Decimal(threshold) >= amount_yi:
            threshold = max(1, threshold - 1)
        verdict = True
    else:
        threshold = max(1, ceil_value)
        if Decimal(threshold) <= amount_yi:
            threshold += 1
        verdict = False

    new_question["id"] = f"{question['id']}--th"
    new_question["text"] = f"{company_name}{report_year}年年报中的营业收入是否超过{threshold}亿元？"
    new_question["kind"] = "boolean"
    new_question["capability"] = "table_numeric_threshold_reasoning"
    new_question["difficulty"] = "hard"
    new_question["unit"] = None
    new_question["gold_value"] = verdict
    new_question["notes"] = f"由 {question['id']} 的原始营业收入换算为亿元后，与阈值 {threshold} 进行直接比较。"
    new_question["metadata"] = _metadata_with(
        new_question.get("metadata"),
        scenario_family="numeric_threshold_boolean",
        parent_question_id=question["id"],
        generation_method="deterministic_threshold_comparison",
        source_dataset="finance_eval_benchmark_v1_core60",
        threshold_yi=threshold,
    )

    new_answer["question_id"] = new_question["id"]
    new_answer["question_text"] = new_question["text"]
    new_answer["kind"] = "boolean"
    new_answer["value"] = verdict
    new_answer["capability"] = "table_numeric_threshold_reasoning"
    new_answer["difficulty"] = "hard"
    new_answer["unit"] = None
    new_answer["notes"] = new_question["notes"]
    new_answer["metadata"] = _metadata_with(
        new_answer.get("metadata"),
        scenario_family="numeric_threshold_boolean",
        parent_question_id=question["id"],
        generation_method="deterministic_threshold_comparison",
        threshold_yi=threshold,
    )
    return new_question, new_answer


def _make_section_boolean_variant(question: Dict[str, Any], answer: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    new_question = _copy_base_question(question)
    new_answer = _copy_base_answer(answer)
    company_name = question["company_name"]
    report_year = question["report_year"]
    section_name = question.get("expected_filters", {}).get("section_name") or "重要事项"
    new_question["id"] = f"{question['id']}--secbool"
    new_question["text"] = f"在{company_name}{report_year}年年报《{section_name}》章节中，是否提到现金分红？"
    new_question["capability"] = "section_filter_boolean"
    new_question["difficulty"] = "medium"
    new_question["notes"] = f"由 {question['id']} 增加章节过滤条件得到。"
    new_question["metadata"] = _metadata_with(
        new_question.get("metadata"),
        scenario_family="section_boolean_variant",
        parent_question_id=question["id"],
        generation_method="section_condition_rewrite",
        source_dataset="finance_eval_benchmark_v1_core60",
    )

    new_answer["question_id"] = new_question["id"]
    new_answer["question_text"] = new_question["text"]
    new_answer["capability"] = "section_filter_boolean"
    new_answer["difficulty"] = "medium"
    new_answer["notes"] = new_question["notes"]
    new_answer["metadata"] = _metadata_with(
        new_answer.get("metadata"),
        scenario_family="section_boolean_variant",
        parent_question_id=question["id"],
        generation_method="section_condition_rewrite",
    )
    return new_question, new_answer


def _make_metadata_count_variant(question: Dict[str, Any], answer: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    new_question = _copy_base_question(question)
    new_answer = _copy_base_answer(answer)
    industry = question["group_name"]
    tag = question["metric_name"] or "国产替代"
    report_year = question["report_year"]
    count_value = len(answer["value"] or [])
    new_question["id"] = f"{question['id']}--count"
    new_question["text"] = f"在{industry}行业的科创板公司中，2024年年报里提到“{tag}”的公司共有几家？"
    new_question["kind"] = "number"
    new_question["capability"] = "metadata_count_aggregation"
    new_question["difficulty"] = "hard"
    new_question["unit"] = "家"
    new_question["gold_value"] = count_value
    new_question["notes"] = f"由 {question['id']} 的公司名单聚合统计得到。"
    new_question["metadata"] = _metadata_with(
        new_question.get("metadata"),
        scenario_family="metadata_count_aggregation",
        parent_question_id=question["id"],
        generation_method="list_to_count_aggregation",
        source_dataset="finance_eval_benchmark_v1_core60",
    )

    new_answer["question_id"] = new_question["id"]
    new_answer["question_text"] = new_question["text"]
    new_answer["kind"] = "number"
    new_answer["value"] = count_value
    new_answer["capability"] = "metadata_count_aggregation"
    new_answer["difficulty"] = "hard"
    new_answer["unit"] = "家"
    new_answer["notes"] = new_question["notes"]
    new_answer["metadata"] = _metadata_with(
        new_answer.get("metadata"),
        scenario_family="metadata_count_aggregation",
        parent_question_id=question["id"],
        generation_method="list_to_count_aggregation",
    )
    return new_question, new_answer


def _make_metadata_membership_variant(
    question: Dict[str, Any],
    answer: Dict[str, Any],
    *,
    candidate_name: str,
    verdict: bool,
    scenario_family: str,
    suffix: str,
    candidate_source: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    new_question = _copy_base_question(question)
    new_answer = _copy_base_answer(answer)
    industry = question["group_name"]
    tag = question["metric_name"] or "国产替代"
    new_question["id"] = f"{question['id']}--{suffix}"
    new_question["text"] = f"在{industry}行业的科创板公司中，{candidate_name}是否属于2024年年报提到“{tag}”的公司名单？"
    new_question["kind"] = "boolean"
    new_question["capability"] = "metadata_membership_boolean"
    new_question["difficulty"] = "hard"
    new_question["unit"] = None
    new_question["gold_value"] = verdict
    new_question["notes"] = f"由 {question['id']} 的原始名单派生成员判断题，候选公司为 {candidate_name}。"
    new_question["metadata"] = _metadata_with(
        new_question.get("metadata"),
        scenario_family=scenario_family,
        parent_question_id=question["id"],
        generation_method="list_membership_check",
        source_dataset="finance_eval_benchmark_v1_core60",
        membership_candidate=candidate_name,
        membership_candidate_source=candidate_source,
    )

    new_answer["question_id"] = new_question["id"]
    new_answer["question_text"] = new_question["text"]
    new_answer["kind"] = "boolean"
    new_answer["value"] = verdict
    new_answer["capability"] = "metadata_membership_boolean"
    new_answer["difficulty"] = "hard"
    new_answer["unit"] = None
    new_answer["notes"] = new_question["notes"]
    new_answer["metadata"] = _metadata_with(
        new_answer.get("metadata"),
        scenario_family=scenario_family,
        parent_question_id=question["id"],
        generation_method="list_membership_check",
        membership_candidate=candidate_name,
        membership_candidate_source=candidate_source,
    )
    return new_question, new_answer


def _make_compare_boolean_variant(question: Dict[str, Any], answer: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    new_question = _copy_base_question(question)
    new_answer = _copy_base_answer(answer)
    company_a, company_b = question["mentioned_companies"]
    winner = str(answer["value"])
    verdict = winner == company_a
    new_question["id"] = f"{question['id']}--cmpbool"
    new_question["text"] = f"在2024年年报中，{company_a}的营业收入是否高于{company_b}？"
    new_question["kind"] = "boolean"
    new_question["capability"] = "cross_doc_compare_boolean"
    new_question["difficulty"] = "hard"
    new_question["gold_value"] = verdict
    new_question["notes"] = f"由 {question['id']} 的比较赢家 {winner} 转换为布尔判断。"
    new_question["metadata"] = _metadata_with(
        new_question.get("metadata"),
        scenario_family="compare_boolean_variant",
        parent_question_id=question["id"],
        generation_method="winner_to_boolean_comparison",
        source_dataset="finance_eval_benchmark_v1_core60",
    )

    new_answer["question_id"] = new_question["id"]
    new_answer["question_text"] = new_question["text"]
    new_answer["kind"] = "boolean"
    new_answer["value"] = verdict
    new_answer["capability"] = "cross_doc_compare_boolean"
    new_answer["difficulty"] = "hard"
    new_answer["notes"] = new_question["notes"]
    new_answer["metadata"] = _metadata_with(
        new_answer.get("metadata"),
        scenario_family="compare_boolean_variant",
        parent_question_id=question["id"],
        generation_method="winner_to_boolean_comparison",
    )
    return new_question, new_answer


def _derive_capability(seed_record: Dict[str, Any], question_text: str, doc_ids: List[str], schema: str) -> str:
    task_type = seed_record.get("task_type")
    if task_type:
        return str(task_type)
    if len(doc_ids) > 1:
        return "cross_doc_compare"
    if schema == "boolean":
        return "single_doc_boolean"
    if schema == "names":
        return "metadata_tag_retrieval"
    if "章节" in question_text:
        return "section_filter"
    return "single_doc_fact"


def _build_refusal_candidates(
    *,
    core_question_texts: set[str],
    doc_manifest: Dict[str, Dict[str, Any]],
    seed_map: Dict[str, Dict[str, Any]],
    industry_name_map: Dict[str, str],
    group_id_by_name: Dict[str, str],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    records = _load_jsonl(REFUSAL_SAMPLE_FILE)
    for record in records:
        if not record.get("should_refuse"):
            continue
        if record.get("question_text") in core_question_texts:
            continue
        assistant_response = record.get("assistant_response") or {}
        if assistant_response.get("final_answer") != "N/A":
            continue
        accepted_checks = set(record.get("accepted_checks") or [])
        if "refusal_sample" not in accepted_checks:
            continue

        seed_record = seed_map.get(record.get("query_id"), {})
        doc_ids = list(record.get("doc_ids") or seed_record.get("doc_ids") or [])
        if not doc_ids:
            continue
        primary_doc_id = doc_ids[0]
        primary_doc = doc_manifest.get(primary_doc_id, {})
        industry_l1 = primary_doc.get("industry_l1")
        group_name = industry_name_map.get(industry_l1, industry_l1 or "未知行业")
        group_id = group_id_by_name.get(group_name, f"group-{len(group_id_by_name) + 1:02d}")
        capability = _derive_capability(seed_record, record["question_text"], doc_ids, record["schema"])
        pages = list(assistant_response.get("relevant_pages") or record.get("retrieval_pages") or [])
        pages = [int(page) for page in pages if isinstance(page, int) or str(page).isdigit()]
        stock_code = primary_doc.get("security_code") or primary_doc.get("stock_code")
        report_year_raw = seed_record.get("expected_filters", {}).get("report_year") or primary_doc.get("fiscal_year")
        try:
            report_year = int(report_year_raw) if report_year_raw is not None else 2024
        except (TypeError, ValueError):
            report_year = 2024
        question_id = f"refusal-{record['query_id'].replace('_', '-')}"
        notes = f"教师复核拒答样本；accepted_checks={','.join(sorted(accepted_checks)) or 'none'}"
        question = {
            "id": question_id,
            "text": record["question_text"],
            "kind": record["schema"],
            "capability": capability,
            "difficulty": "hard" if capability in {"cross_doc_compare", "metadata_tag_retrieval"} else "medium",
            "doc_ids": doc_ids,
            "company_name": record.get("company_name") or seed_record.get("company_name"),
            "mentioned_companies": list(seed_record.get("mentioned_companies") or []),
            "stock_code": stock_code,
            "report_year": report_year,
            "report_type": "annual_report",
            "period": None,
            "metric_name": _metric_name({"text": record["question_text"]}),
            "currency": None,
            "unit": None,
            "section_name": seed_record.get("expected_filters", {}).get("section_name"),
            "industry_l1": group_name,
            "group_id": group_id,
            "group_name": group_name,
            "group_slot": None,
            "evidence_type": "refusal",
            "gold_value": "N/A",
            "gold_pages": pages,
            "gold_chunk_ids": [],
            "should_refuse": True,
            "expected_filters": dict(seed_record.get("expected_filters") or {}),
            "annotation_status": "teacher_refusal_verified",
            "notes": notes,
            "metadata": {
                "scenario_family": "refusal_teacher_verified",
                "parent_question_id": None,
                "generation_method": "teacher_filtered_refusal_nonconflict",
                "source_dataset": "training_generator_sft_teacher_answers_filtered",
                "source_query_id": record.get("query_id"),
                "accepted_checks": sorted(accepted_checks),
            },
        }
        answer = {
            "question_id": question_id,
            "question_text": question["text"],
            "kind": question["kind"],
            "value": "N/A",
            "doc_ids": doc_ids,
            "gold_pages": pages,
            "gold_chunk_ids": [],
            "references": _make_answer_reference_list(primary_doc_id, pages) if len(doc_ids) == 1 else [],
            "company_name": question["company_name"],
            "stock_code": stock_code,
            "report_year": report_year,
            "report_type": "annual_report",
            "period": None,
            "metric_name": question["metric_name"],
            "currency": None,
            "unit": None,
            "evidence_type": "refusal",
            "capability": capability,
            "difficulty": question["difficulty"],
            "group_id": group_id,
            "group_name": group_name,
            "should_refuse": True,
            "annotation_status": "teacher_refusal_verified",
            "notes": notes,
            "metadata": {
                "scenario_family": "refusal_teacher_verified",
                "parent_question_id": None,
                "generation_method": "teacher_filtered_refusal_nonconflict",
                "source_query_id": record.get("query_id"),
                "accepted_checks": sorted(accepted_checks),
            },
        }
        candidates.append({"question": question, "answer": answer})

    name_like = [pair for pair in candidates if pair["question"]["capability"] == "single_doc_fact"]
    other = [pair for pair in candidates if pair["question"]["capability"] != "single_doc_fact"]
    name_like = _round_robin_name_candidates([pair["question"] | {"__answer__": pair["answer"]} for pair in name_like])
    reordered_name_pairs = [{"question": {k: v for k, v in item.items() if k != "__answer__"}, "answer": item["__answer__"]} for item in name_like]
    return other + reordered_name_pairs


def _select_refusal_pairs(refusal_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered = list(refusal_candidates)
    if len(ordered) < 50:
        raise ValueError(f"Need at least 50 refusal candidates, found {len(ordered)}")
    return ordered[:50]


def _assign_refusal_splits(refusal_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    assigned: List[Dict[str, Any]] = []
    for index, pair in enumerate(refusal_pairs):
        split = "dev" if index % 2 == 0 else "test"
        assigned.append(_make_pair(pair["question"], pair["answer"], split=split))
    return assigned


def _write_dataset(
    *,
    output_dir: Path,
    dataset_name: str,
    description: str,
    pairs: List[Dict[str, Any]],
) -> None:
    questions = [pair["question"] for pair in pairs]
    answers = [pair["answer"] for pair in pairs]
    split_by_id = {pair["question"]["id"]: pair["split"] for pair in pairs}
    dev_pairs = [pair for pair in pairs if pair["split"] == "dev"]
    test_pairs = [pair for pair in pairs if pair["split"] == "test"]
    dev_questions = [pair["question"] for pair in dev_pairs]
    dev_answers = [pair["answer"] for pair in dev_pairs]
    test_questions = [pair["question"] for pair in test_pairs]
    test_answers = [pair["answer"] for pair in test_pairs]

    full_questions_payload, full_answers_payload = _build_wrapped_sets(
        dataset_name=dataset_name,
        questions=questions,
        answers=answers,
    )
    dev_questions_payload, dev_answers_payload = _build_wrapped_sets(
        dataset_name=f"{dataset_name}_dev",
        questions=dev_questions,
        answers=dev_answers,
        split_name="dev",
    )
    test_questions_payload, test_answers_payload = _build_wrapped_sets(
        dataset_name=f"{dataset_name}_test",
        questions=test_questions,
        answers=test_answers,
        split_name="test",
    )

    full_manifest = _build_manifest(
        dataset_name=dataset_name,
        description=description,
        source_corpora=[
            "finance_eval_benchmark_v1",
            "top10_industries_2024_20each",
            "training/generator_sft/processed/teacher_answers_filtered.jsonl",
        ],
        questions=questions,
        answers=answers,
    )
    dev_manifest = _build_manifest(
        dataset_name=f"{dataset_name}_dev",
        description=f"Development split for {dataset_name}.",
        source_corpora=[
            "finance_eval_benchmark_v1",
            "top10_industries_2024_20each",
            "training/generator_sft/processed/teacher_answers_filtered.jsonl",
        ],
        questions=dev_questions,
        answers=dev_answers,
        split_name="dev",
    )
    test_manifest = _build_manifest(
        dataset_name=f"{dataset_name}_test",
        description=f"Holdout split for {dataset_name}.",
        source_corpora=[
            "finance_eval_benchmark_v1",
            "top10_industries_2024_20each",
            "training/generator_sft/processed/teacher_answers_filtered.jsonl",
        ],
        questions=test_questions,
        answers=test_answers,
        split_name="test",
    )

    _write_json(output_dir / "questions.json", full_questions_payload)
    _write_json(output_dir / "answers_gold.json", full_answers_payload)
    _write_json(output_dir / "dataset_manifest.json", full_manifest)
    _write_json(output_dir / "splits" / "dev" / "questions.json", dev_questions_payload)
    _write_json(output_dir / "splits" / "dev" / "answers_gold.json", dev_answers_payload)
    _write_json(output_dir / "splits" / "dev" / "dataset_manifest.json", dev_manifest)
    _write_json(output_dir / "splits" / "dev" / "question_ids.json", [question["id"] for question in dev_questions])
    _write_json(output_dir / "splits" / "test" / "questions.json", test_questions_payload)
    _write_json(output_dir / "splits" / "test" / "answers_gold.json", test_answers_payload)
    _write_json(output_dir / "splits" / "test" / "dataset_manifest.json", test_manifest)
    _write_json(output_dir / "splits" / "test" / "question_ids.json", [question["id"] for question in test_questions])
    _write_review_checklist(output_dir / "review_checklist.csv", questions, split_by_id)

    for questions_file, answers_file in [
        (output_dir / "questions.json", output_dir / "answers_gold.json"),
        (output_dir / "splits" / "dev" / "questions.json", output_dir / "splits" / "dev" / "answers_gold.json"),
        (output_dir / "splits" / "test" / "questions.json", output_dir / "splits" / "test" / "answers_gold.json"),
    ]:
        question_set = load_question_set(questions_file)
        answer_set = load_gold_answer_set(answers_file)
        report = validate_dataset_alignment(question_set, answer_set)
        if not report["valid"]:
            raise ValueError(f"Generated dataset is invalid for {questions_file}: {report}")


def build_benchmark_v2(output_dir: Path) -> None:
    core_questions = _question_dump(SOURCE_BENCHMARK_DIR / "questions.json")
    core_answers = _answer_dump(SOURCE_BENCHMARK_DIR / "answers_gold.json")
    core_question_by_id = {question["id"]: question for question in core_questions}
    core_answer_by_id = {answer["question_id"]: answer for answer in core_answers}
    core_split_ids = {
        "dev": set(_load_json(SOURCE_BENCHMARK_DIR / "splits" / "dev" / "question_ids.json")),
        "test": set(_load_json(SOURCE_BENCHMARK_DIR / "splits" / "test" / "question_ids.json")),
    }
    base_split_by_id = {
        question_id: ("dev" if question_id in core_split_ids["dev"] else "test")
        for question_id in core_question_by_id
    }

    doc_manifest = _load_document_manifest(DOCUMENT_MANIFEST_FILE)
    seed_map = {record["query_id"]: record for record in _load_jsonl(SEED_QUERY_FILE)}

    industry_name_map: Dict[str, str] = {}
    group_id_by_name: Dict[str, str] = {}
    doc_to_group_name: Dict[str, str] = {}
    for question in core_questions:
        group_id_by_name[question["group_name"]] = question["group_id"]
        for doc_id in question.get("doc_ids") or []:
            row = doc_manifest.get(doc_id) or {}
            industry_l1 = row.get("industry_l1")
            if industry_l1 and question.get("group_name"):
                industry_name_map[industry_l1] = question["group_name"]
                doc_to_group_name[doc_id] = question["group_name"]

    base_pairs: List[Dict[str, Any]] = []
    numeric_wanyuan_pairs: List[Dict[str, Any]] = []
    numeric_yiyuan_pairs: List[Dict[str, Any]] = []
    numeric_baiwan_pairs: List[Dict[str, Any]] = []
    numeric_threshold_pairs: List[Dict[str, Any]] = []
    section_boolean_pairs: List[Dict[str, Any]] = []
    metadata_count_pairs: List[Dict[str, Any]] = []
    metadata_membership_true_pairs: List[Dict[str, Any]] = []
    metadata_membership_false_pairs: List[Dict[str, Any]] = []
    compare_boolean_pairs: List[Dict[str, Any]] = []

    company_name_by_doc = {doc_id: row.get("company_name") for doc_id, row in doc_manifest.items()}

    numeric_questions = [question for question in core_questions if question["kind"] == "number" and question["capability"] == "single_doc_fact"]
    boolean_questions = [question for question in core_questions if question["kind"] == "boolean" and question["capability"] == "single_doc_boolean"]
    metadata_questions = [question for question in core_questions if question["kind"] == "names"]
    compare_questions = [question for question in core_questions if question["capability"] == "cross_doc_compare"]

    for question in core_questions:
        answer = core_answer_by_id[question["id"]]
        question["metadata"] = _metadata_with(
            question.get("metadata"),
            scenario_family="base_core",
            parent_question_id=None,
            generation_method="benchmark_v1_base",
            source_dataset="finance_eval_benchmark_v1_core60",
        )
        answer["metadata"] = _metadata_with(
            answer.get("metadata"),
            scenario_family="base_core",
            parent_question_id=None,
            generation_method="benchmark_v1_base",
            source_dataset="finance_eval_benchmark_v1_core60",
        )
        base_pairs.append(_make_pair(question, answer, split=base_split_by_id[question["id"]]))

    for index, question in enumerate(sorted(numeric_questions, key=lambda item: item["id"])):
        answer = core_answer_by_id[question["id"]]
        split = base_split_by_id[question["id"]]
        q_wan, a_wan = _make_numeric_unit_variant(
            question,
            answer,
            suffix="wanyuan",
            unit_text="万元",
            divisor=Decimal("10000"),
            decimals=4,
            scenario_family="numeric_unit_wanyuan",
        )
        q_yi, a_yi = _make_numeric_unit_variant(
            question,
            answer,
            suffix="yiyuan",
            unit_text="亿元",
            divisor=Decimal("100000000"),
            decimals=6,
            scenario_family="numeric_unit_yiyuan",
        )
        q_baiwan, a_baiwan = _make_numeric_unit_variant(
            question,
            answer,
            suffix="baiwan",
            unit_text="百万元",
            divisor=Decimal("1000000"),
            decimals=4,
            scenario_family="numeric_unit_baiwan",
        )
        q_threshold, a_threshold = _make_numeric_threshold_variant(question, answer, variant_index=index)
        numeric_wanyuan_pairs.append(_make_pair(q_wan, a_wan, split=split))
        numeric_yiyuan_pairs.append(_make_pair(q_yi, a_yi, split=split))
        numeric_baiwan_pairs.append(_make_pair(q_baiwan, a_baiwan, split=split))
        numeric_threshold_pairs.append(_make_pair(q_threshold, a_threshold, split=split))

    for question in sorted(boolean_questions, key=lambda item: item["id"]):
        answer = core_answer_by_id[question["id"]]
        q_variant, a_variant = _make_section_boolean_variant(question, answer)
        section_boolean_pairs.append(_make_pair(q_variant, a_variant, split=base_split_by_id[question["id"]]))

    for question in sorted(metadata_questions, key=lambda item: item["id"]):
        answer = core_answer_by_id[question["id"]]
        split = base_split_by_id[question["id"]]
        q_count, a_count = _make_metadata_count_variant(question, answer)
        metadata_count_pairs.append(_make_pair(q_count, a_count, split=split))

        gold_names = set(answer["value"] or [])
        positive_candidate = next((name for name in answer["value"] or [] if name), None)
        if positive_candidate is None:
            raise ValueError(f"No positive membership candidate for {question['id']}")
        mapped_companies = [company_name_by_doc.get(doc_id) for doc_id in question.get("doc_ids") or []]
        negative_candidate = next((name for name in mapped_companies if name and name not in gold_names), None)
        negative_source = "doc_pool"
        if negative_candidate is None:
            negative_candidate = next(
                (
                    company_name
                    for company_name in sorted(set(filter(None, company_name_by_doc.values())))
                    if company_name not in gold_names and company_name != positive_candidate
                ),
                None,
            )
            negative_source = "external_pool"
        if negative_candidate is None:
            raise ValueError(f"No negative membership candidate for {question['id']}")
        q_pos, a_pos = _make_metadata_membership_variant(
            question,
            answer,
            candidate_name=positive_candidate,
            verdict=True,
            scenario_family="metadata_membership_positive",
            suffix="member-pos",
            candidate_source="gold_member",
        )
        q_neg, a_neg = _make_metadata_membership_variant(
            question,
            answer,
            candidate_name=negative_candidate,
            verdict=False,
            scenario_family="metadata_membership_negative",
            suffix="member-neg",
            candidate_source=negative_source,
        )
        metadata_membership_true_pairs.append(_make_pair(q_pos, a_pos, split=split))
        metadata_membership_false_pairs.append(_make_pair(q_neg, a_neg, split=split))

    for question in sorted(compare_questions, key=lambda item: item["id"]):
        answer = core_answer_by_id[question["id"]]
        q_variant, a_variant = _make_compare_boolean_variant(question, answer)
        compare_boolean_pairs.append(_make_pair(q_variant, a_variant, split=base_split_by_id[question["id"]]))

    refusal_candidates = _build_refusal_candidates(
        core_question_texts={question["text"] for question in core_questions},
        doc_manifest=doc_manifest,
        seed_map=seed_map,
        industry_name_map=industry_name_map,
        group_id_by_name=group_id_by_name,
    )
    selected_refusals = _select_refusal_pairs(refusal_candidates)
    refusal_pairs = _assign_refusal_splits(selected_refusals)
    core_refusal_pairs = refusal_pairs[:10]
    extra_refusal_pairs = refusal_pairs[10:]

    core120_pairs = (
        base_pairs
        + numeric_wanyuan_pairs
        + numeric_yiyuan_pairs
        + numeric_threshold_pairs
        + metadata_count_pairs
        + compare_boolean_pairs
        + core_refusal_pairs
    )
    core200_pairs = (
        core120_pairs
        + numeric_baiwan_pairs
        + section_boolean_pairs
        + metadata_membership_true_pairs
        + metadata_membership_false_pairs
        + extra_refusal_pairs
    )

    core120_pairs = sorted(core120_pairs, key=lambda pair: _scenario_sort_key(pair["question"]))
    core200_pairs = sorted(core200_pairs, key=lambda pair: _scenario_sort_key(pair["question"]))

    if len(core120_pairs) != 120:
        raise ValueError(f"Expected core120 to contain 120 questions, found {len(core120_pairs)}")
    if len(core200_pairs) != 200:
        raise ValueError(f"Expected core200 to contain 200 questions, found {len(core200_pairs)}")
    if sum(1 for pair in core120_pairs if pair["split"] == "dev") != 60:
        raise ValueError("core120 dev split must contain 60 questions.")
    if sum(1 for pair in core200_pairs if pair["split"] == "dev") != 100:
        raise ValueError("core200 dev split must contain 100 questions.")

    _write_readme(output_dir / "README.md")
    _write_dataset(
        output_dir=output_dir / "core120",
        dataset_name="finance_eval_benchmark_v2_core120",
        description="120-question formal finance benchmark with unit normalization, threshold reasoning, aggregation, comparison, and refusal coverage.",
        pairs=core120_pairs,
    )
    _write_dataset(
        output_dir=output_dir / "core200",
        dataset_name="finance_eval_benchmark_v2_core200",
        description="200-question extended finance benchmark with stronger coverage of table numerics, multi-hop aggregation/comparison, and refusal behaviour.",
        pairs=core200_pairs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the v2 formal finance evaluation benchmark with 120/200-question subsets.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    build_benchmark_v2(args.output_dir)


if __name__ == "__main__":
    main()
