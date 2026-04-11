from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


AnswerScalar = str | int | float | bool
AnswerValue = AnswerScalar | list[AnswerScalar] | None


class FinanceEvalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


def _dedupe_strings(values: List[str]) -> List[str]:
    seen: set[str] = set()
    deduped: List[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _dedupe_positive_pages(values: List[int]) -> List[int]:
    pages = sorted({int(value) for value in values if int(value) > 0})
    return pages


def _normalize_answer_value(value: AnswerValue) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return json.dumps(sorted(str(item) for item in value), ensure_ascii=False)
    return " ".join(str(value).strip().lower().split())


class FinanceGoldReference(FinanceEvalModel):
    pdf_sha1: str
    page_index: int = Field(ge=0, description="Zero-based page index in the original PDF.")
    chunk_id: int | str | None = None
    section_name: str | None = None
    evidence_type: str | None = None


class FinanceEvalQuestion(FinanceEvalModel):
    question_id: str = Field(alias="id")
    question_text: str = Field(alias="text")
    kind: str
    capability: str | None = None
    difficulty: str | None = None
    doc_ids: List[str] = Field(default_factory=list)
    company_name: str | None = None
    mentioned_companies: List[str] = Field(default_factory=list)
    stock_code: str | None = None
    report_year: int | None = None
    report_type: str | None = None
    period: str | None = None
    metric_name: str | None = None
    currency: str | None = None
    unit: str | None = None
    section_name: str | None = None
    industry_l1: str | None = None
    group_id: str | None = None
    group_name: str | None = None
    group_slot: int | None = None
    evidence_type: str | None = None
    gold_value: AnswerValue = None
    gold_pages: List[int] = Field(default_factory=list)
    gold_chunk_ids: List[int | str] = Field(default_factory=list)
    should_refuse: bool = False
    expected_filters: Dict[str, Any] = Field(default_factory=dict)
    annotation_status: str = "draft"
    notes: str | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("question_id", "question_text")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("Field cannot be empty.")
        return normalized

    @field_validator("doc_ids", "mentioned_companies")
    @classmethod
    def _normalize_string_lists(cls, values: List[str]) -> List[str]:
        return _dedupe_strings(values)

    @field_validator("gold_pages")
    @classmethod
    def _normalize_gold_pages(cls, values: List[int]) -> List[int]:
        return _dedupe_positive_pages(values)


class FinanceGoldAnswer(FinanceEvalModel):
    question_id: str
    question_text: str
    kind: str
    value: AnswerValue = None
    doc_ids: List[str] = Field(default_factory=list)
    gold_pages: List[int] = Field(default_factory=list)
    gold_chunk_ids: List[int | str] = Field(default_factory=list)
    references: List[FinanceGoldReference] = Field(default_factory=list)
    company_name: str | None = None
    stock_code: str | None = None
    report_year: int | None = None
    report_type: str | None = None
    period: str | None = None
    metric_name: str | None = None
    currency: str | None = None
    unit: str | None = None
    evidence_type: str | None = None
    capability: str | None = None
    difficulty: str | None = None
    group_id: str | None = None
    group_name: str | None = None
    should_refuse: bool = False
    annotation_status: str = "draft"
    notes: str | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("question_id", "question_text")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("Field cannot be empty.")
        return normalized

    @field_validator("doc_ids")
    @classmethod
    def _normalize_doc_ids(cls, values: List[str]) -> List[str]:
        return _dedupe_strings(values)

    @field_validator("gold_pages")
    @classmethod
    def _normalize_gold_pages(cls, values: List[int]) -> List[int]:
        return _dedupe_positive_pages(values)


class FinanceEvalQuestionSet(FinanceEvalModel):
    schema_version: str = "finance_eval_v1"
    dataset_name: str | None = None
    questions: List[FinanceEvalQuestion]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FinanceGoldAnswerSet(FinanceEvalModel):
    schema_version: str = "finance_eval_v1"
    dataset_name: str | None = None
    answers: List[FinanceGoldAnswer]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FinanceEvalManifest(FinanceEvalModel):
    schema_version: str = "finance_eval_v1"
    dataset_name: str
    description: str | None = None
    source_corpora: List[str] = Field(default_factory=list)
    question_count: int | None = None
    answer_count: int | None = None
    scoring_profile: str = "finance_rag_v1"
    strata_summary: Dict[str, int] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _wrap_question_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, list):
        return {
            "schema_version": "finance_eval_v1",
            "questions": payload,
            "metadata": {},
        }
    if isinstance(payload, dict) and "questions" in payload:
        return {
            "schema_version": payload.get("schema_version", "finance_eval_v1"),
            "dataset_name": payload.get("dataset_name"),
            "questions": payload["questions"],
            "metadata": payload.get("metadata", {}),
        }
    raise ValueError("Questions payload must be a list or a dict containing 'questions'.")


def _wrap_answer_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, list):
        return {
            "schema_version": "finance_eval_v1",
            "answers": payload,
            "metadata": {},
        }
    if isinstance(payload, dict) and "answers" in payload:
        return {
            "schema_version": payload.get("schema_version", "finance_eval_v1"),
            "dataset_name": payload.get("dataset_name"),
            "answers": payload["answers"],
            "metadata": payload.get("metadata", {}),
        }
    raise ValueError("Answers payload must be a list or a dict containing 'answers'.")


def load_question_set(path: Path) -> FinanceEvalQuestionSet:
    return FinanceEvalQuestionSet.model_validate(_wrap_question_payload(_load_json(path)))


def load_gold_answer_set(path: Path) -> FinanceGoldAnswerSet:
    return FinanceGoldAnswerSet.model_validate(_wrap_answer_payload(_load_json(path)))


def load_manifest(path: Path) -> FinanceEvalManifest:
    return FinanceEvalManifest.model_validate(_load_json(path))


def validate_dataset_alignment(
    question_set: FinanceEvalQuestionSet,
    answer_set: FinanceGoldAnswerSet,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    questions_by_id = {question.question_id: question for question in question_set.questions}
    answers_by_id = {answer.question_id: answer for answer in answer_set.answers}

    duplicate_question_ids = len(questions_by_id) != len(question_set.questions)
    duplicate_answer_ids = len(answers_by_id) != len(answer_set.answers)
    if duplicate_question_ids:
        errors.append("Duplicate question_id values detected in questions.")
    if duplicate_answer_ids:
        errors.append("Duplicate question_id values detected in answers.")

    for answer in answer_set.answers:
        question = questions_by_id.get(answer.question_id)
        if question is None:
            errors.append(f"Answer '{answer.question_id}' does not map to any question.")
            continue
        if answer.question_text != question.question_text:
            warnings.append(
                f"Question text mismatch for '{answer.question_id}': "
                f"question='{question.question_text}' answer='{answer.question_text}'."
            )
        if question.gold_value is not None and answer.value is not None:
            if _normalize_answer_value(question.gold_value) != _normalize_answer_value(answer.value):
                warnings.append(f"gold_value and answer.value differ for '{answer.question_id}'.")
        if question.gold_pages and answer.gold_pages and question.gold_pages != answer.gold_pages:
            warnings.append(f"gold_pages differ between question and answer for '{answer.question_id}'.")

    missing_answers = sorted(set(questions_by_id) - set(answers_by_id))
    if missing_answers:
        warnings.append(
            "Questions without gold answers: " + ", ".join(missing_answers[:10]) + ("..." if len(missing_answers) > 10 else "")
        )

    return {
        "valid": not errors,
        "question_count": len(question_set.questions),
        "answer_count": len(answer_set.answers),
        "errors": errors,
        "warnings": warnings,
    }


def export_json_schemas() -> Dict[str, Dict[str, Any]]:
    return {
        "finance_eval_question": FinanceEvalQuestion.model_json_schema(),
        "finance_gold_answer": FinanceGoldAnswer.model_json_schema(),
        "finance_eval_question_set": FinanceEvalQuestionSet.model_json_schema(),
        "finance_gold_answer_set": FinanceGoldAnswerSet.model_json_schema(),
        "finance_eval_manifest": FinanceEvalManifest.model_json_schema(),
    }


def validate_payload(model_cls: type[BaseModel], payload: Any) -> Dict[str, Any]:
    try:
        model_cls.model_validate(payload)
        return {"valid": True, "errors": []}
    except ValidationError as exc:
        return {"valid": False, "errors": exc.errors(include_url=False)}
