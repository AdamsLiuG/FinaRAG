from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

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
SOURCE_DIR = ROOT / "data" / "top10_industries_2024_20each"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "finance_eval_benchmark_v1"


QUESTION_SOURCE_FILE = SOURCE_DIR / "questions.json"
ANSWER_SOURCE_FILE = SOURCE_DIR / "answers_zh_gold_template_60.json"


CAPABILITY_DIFFICULTY = {
    "single_doc_fact": "easy",
    "single_doc_boolean": "medium",
    "section_filter": "medium",
    "metadata_tag_retrieval": "hard",
    "cross_doc_compare": "hard",
}


REVIEW_PRIORITY = {
    "single_doc_fact:name": "low",
    "single_doc_fact:number": "high",
    "single_doc_boolean:boolean": "medium",
    "section_filter:name": "medium",
    "metadata_tag_retrieval:names": "high",
    "cross_doc_compare:name": "high",
}


REVIEW_FOCUS = {
    "single_doc_fact:name": "核对公司/人员实体、页码、是否来自正确文档。",
    "single_doc_fact:number": "核对数值、单位换算、币种、页码和表格位置。",
    "single_doc_boolean:boolean": "核对是否存在明确文本证据，避免推断式标注。",
    "section_filter:name": "核对 section_name 约束、答案实体和章节页码是否一致。",
    "metadata_tag_retrieval:names": "核对名单完整性、去重规则、排序规则和标签口径。",
    "cross_doc_compare:name": "核对参与比较的所有文档、比较维度和赢家实体是否正确。",
}


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _infer_metric_name(question_text: str, capability: str, kind: str) -> str | None:
    text = question_text.strip()
    mappings = [
        ("法定代表人", "法定代表人"),
        ("营业收入", "营业收入"),
        ("现金分红", "现金分红"),
        ("国产替代", "国产替代"),
    ]
    for needle, metric_name in mappings:
        if needle in text:
            return metric_name
    if capability == "cross_doc_compare" and "谁的" in text and "更高" in text:
        return "比较题"
    if kind == "boolean":
        return "布尔判断"
    return None


def _infer_period(question_text: str, kind: str) -> str | None:
    if kind != "number":
        return None
    if "年末" in question_text or "期末" in question_text:
        return "期末"
    return "本期"


def _infer_currency(question_text: str, kind: str) -> str | None:
    if kind != "number":
        return None
    if "元" in question_text:
        return "CNY"
    return None


def _infer_unit(question_text: str, kind: str) -> str | None:
    if kind != "number":
        return None
    if "元" in question_text:
        return "元"
    return None


def _question_review_key(question: Dict[str, Any]) -> str:
    return f"{question.get('capability')}:{question.get('kind')}"


def _normalize_question(question: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(question)
    normalized["report_type"] = "annual_report"
    normalized["difficulty"] = normalized.get("difficulty") or CAPABILITY_DIFFICULTY.get(normalized.get("capability"), "medium")
    normalized["metric_name"] = normalized.get("metric_name") or _infer_metric_name(
        normalized.get("text", ""),
        normalized.get("capability", ""),
        normalized.get("kind", ""),
    )
    normalized["period"] = normalized.get("period") or _infer_period(normalized.get("text", ""), normalized.get("kind", ""))
    normalized["currency"] = normalized.get("currency") or _infer_currency(normalized.get("text", ""), normalized.get("kind", ""))
    normalized["unit"] = normalized.get("unit") or _infer_unit(normalized.get("text", ""), normalized.get("kind", ""))
    metadata = dict(normalized.get("metadata") or {})
    metadata.update(
        {
            "source_dataset": "top10_industries_2024_20each",
            "source_record_type": "question",
            "quality_tier": "provisional_gold",
            "review_required": True,
        }
    )
    normalized["metadata"] = metadata
    return normalized


def _normalize_answer(answer: Dict[str, Any], question: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(answer)
    normalized["report_type"] = normalized.get("report_type") or "annual_report"
    normalized["difficulty"] = normalized.get("difficulty") or question.get("difficulty")
    normalized["metric_name"] = normalized.get("metric_name") or question.get("metric_name")
    normalized["period"] = normalized.get("period") or question.get("period")
    normalized["currency"] = normalized.get("currency") or question.get("currency")
    normalized["unit"] = normalized.get("unit") or question.get("unit")
    normalized["company_name"] = normalized.get("company_name") or question.get("company_name")
    normalized["stock_code"] = normalized.get("stock_code") or question.get("stock_code")
    normalized["report_year"] = normalized.get("report_year") or question.get("report_year")
    normalized["evidence_type"] = normalized.get("evidence_type") or question.get("evidence_type")
    normalized["group_id"] = normalized.get("group_id") or question.get("group_id")
    normalized["group_name"] = normalized.get("group_name") or question.get("group_name")
    metadata = dict(normalized.get("metadata") or {})
    metadata.update(
        {
            "source_dataset": "top10_industries_2024_20each",
            "source_record_type": "answer",
            "quality_tier": "provisional_gold",
            "review_required": True,
        }
    )
    normalized["metadata"] = metadata
    return normalized


def _select_split(question: Dict[str, Any]) -> str:
    group_id = question.get("group_id", "")
    slot = int(question.get("group_slot") or 0)
    group_index = int(str(group_id).split("-")[-1])
    if group_index % 2 == 1:
        return "dev" if slot in {1, 3, 5} else "test"
    return "dev" if slot in {2, 4, 6} else "test"


def _build_strata_summary(questions: Iterable[Dict[str, Any]], split_name: str | None = None) -> Dict[str, int]:
    summary: Dict[str, int] = {}

    def _bump(key: str) -> None:
        summary[key] = summary.get(key, 0) + 1

    for question in questions:
        _bump("all")
        _bump(f"kind/{question['kind']}")
        _bump(f"capability/{question['capability']}")
        _bump(f"difficulty/{question['difficulty']}")
        if question.get("group_name"):
            _bump(f"industry/{question['group_name']}")
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
    metadata = {
        "owner": "FinaRAG",
        "quality_tier": "provisional_gold",
        "review_required": True,
        "split_name": split_name,
        "annotation_policy": "top10_auto_filled_needs_human_review",
        "split_strategy": (
            "odd industry groups -> dev slots {1,3,5}, test slots {2,4,6}; "
            "even industry groups -> dev slots {2,4,6}, test slots {1,3,5}"
        ),
    }
    manifest = FinanceEvalManifest(
        schema_version="finance_eval_v1",
        dataset_name=dataset_name,
        description=description,
        source_corpora=source_corpora,
        question_count=len(questions),
        answer_count=len(answers),
        scoring_profile="finance_rag_v1",
        strata_summary=_build_strata_summary(questions, split_name=split_name),
        metadata=metadata,
    )
    return manifest.model_dump()


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


def _write_review_checklist(path: Path, questions: List[Dict[str, Any]], split_by_id: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question_id",
                "split",
                "group_id",
                "group_name",
                "group_slot",
                "kind",
                "capability",
                "difficulty",
                "company_name",
                "mentioned_companies",
                "doc_ids",
                "gold_value",
                "gold_pages",
                "evidence_type",
                "annotation_status",
                "review_priority",
                "review_focus",
                "notes",
            ],
        )
        writer.writeheader()
        for question in questions:
            review_key = _question_review_key(question)
            writer.writerow(
                {
                    "question_id": question["id"],
                    "split": split_by_id[question["id"]],
                    "group_id": question.get("group_id"),
                    "group_name": question.get("group_name"),
                    "group_slot": question.get("group_slot"),
                    "kind": question.get("kind"),
                    "capability": question.get("capability"),
                    "difficulty": question.get("difficulty"),
                    "company_name": question.get("company_name"),
                    "mentioned_companies": "|".join(question.get("mentioned_companies") or []),
                    "doc_ids": "|".join(question.get("doc_ids") or []),
                    "gold_value": json.dumps(question.get("gold_value"), ensure_ascii=False),
                    "gold_pages": "|".join(str(page) for page in question.get("gold_pages") or []),
                    "evidence_type": question.get("evidence_type"),
                    "annotation_status": question.get("annotation_status"),
                    "review_priority": REVIEW_PRIORITY.get(review_key, "medium"),
                    "review_focus": REVIEW_FOCUS.get(review_key, "核对答案值、页码、文档范围与证据类型。"),
                    "notes": question.get("notes"),
                }
            )


def build_benchmark(output_dir: Path) -> None:
    raw_questions = _load_json(QUESTION_SOURCE_FILE)
    raw_answers = _load_json(ANSWER_SOURCE_FILE)["answers"]

    question_by_id = {question["id"]: _normalize_question(question) for question in raw_questions}
    answers = [_normalize_answer(answer, question_by_id[answer["question_id"]]) for answer in raw_answers]
    questions = [question_by_id[question["id"]] for question in raw_questions]

    split_by_id = {question["id"]: _select_split(question) for question in questions}
    dev_questions = [question for question in questions if split_by_id[question["id"]] == "dev"]
    test_questions = [question for question in questions if split_by_id[question["id"]] == "test"]
    dev_answers = [answer for answer in answers if split_by_id[answer["question_id"]] == "dev"]
    test_answers = [answer for answer in answers if split_by_id[answer["question_id"]] == "test"]

    full_dataset_name = "finance_eval_benchmark_v1_core60"
    dev_dataset_name = "finance_eval_benchmark_v1_dev30"
    test_dataset_name = "finance_eval_benchmark_v1_test30"

    full_questions_payload, full_answers_payload = _build_wrapped_sets(
        dataset_name=full_dataset_name,
        questions=questions,
        answers=answers,
    )
    dev_questions_payload, dev_answers_payload = _build_wrapped_sets(
        dataset_name=dev_dataset_name,
        questions=dev_questions,
        answers=dev_answers,
        split_name="dev",
    )
    test_questions_payload, test_answers_payload = _build_wrapped_sets(
        dataset_name=test_dataset_name,
        questions=test_questions,
        answers=test_answers,
        split_name="test",
    )

    full_manifest = _build_manifest(
        dataset_name=full_dataset_name,
        description="Formal 60-question finance RAG benchmark built from the 2024 top10 industries corpus.",
        source_corpora=["top10_industries_2024_20each", "chinese_annual_reports_2024_v1"],
        questions=questions,
        answers=answers,
    )
    dev_manifest = _build_manifest(
        dataset_name=dev_dataset_name,
        description="30-question development split for finance_eval_benchmark_v1_core60.",
        source_corpora=["top10_industries_2024_20each", "chinese_annual_reports_2024_v1"],
        questions=dev_questions,
        answers=dev_answers,
        split_name="dev",
    )
    test_manifest = _build_manifest(
        dataset_name=test_dataset_name,
        description="30-question holdout split for finance_eval_benchmark_v1_core60.",
        source_corpora=["top10_industries_2024_20each", "chinese_annual_reports_2024_v1"],
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a formal finance evaluation benchmark from the top10 industry dataset.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    build_benchmark(args.output_dir)


if __name__ == "__main__":
    main()
