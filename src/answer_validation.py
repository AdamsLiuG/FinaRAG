from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.query_plan import QueryPlan


_CONFIDENCE_ORDER = ["low", "medium", "high"]


def _normalize_confidence(value: str | None) -> str:
    normalized = (value or "low").lower()
    return normalized if normalized in _CONFIDENCE_ORDER else "low"


def _downgrade_confidence(value: str, steps: int = 1) -> str:
    current = _CONFIDENCE_ORDER.index(_normalize_confidence(value))
    return _CONFIDENCE_ORDER[max(0, current - steps)]


def _union_metadata_flags(retrieval_results: List[Dict[str, Any]]) -> set[str]:
    topic_flags: set[str] = set()
    for result in retrieval_results:
        metadata = result.get("metadata") or {}
        topic_flags.update(metadata.get("topic_flags") or [])
    return topic_flags


def _period_matches(requested_period: str | None, observed_period: str | None) -> bool:
    if not requested_period or not observed_period:
        return True
    requested = str(requested_period).lower()
    observed = str(observed_period).lower()
    return requested in observed or observed in requested


@dataclass
class ValidatedAnswer:
    answer: Dict[str, Any]
    validation_flags: List[str]
    confidence: str
    confidence_reason: str


def validate_answer(answer_dict: Dict[str, Any], retrieval_results: List[Dict[str, Any]], query_plan: QueryPlan) -> ValidatedAnswer:
    answer = dict(answer_dict)
    validation_flags: List[str] = []
    confidence = _normalize_confidence(answer.get("confidence"))
    relevant_pages = answer.get("relevant_pages") or []
    citations = answer.get("citations") or []
    final_answer = answer.get("final_answer")

    retrieval_metadata = [result.get("metadata") or {} for result in retrieval_results]
    currencies = {metadata.get("currency") for metadata in retrieval_metadata if metadata.get("currency")}
    years = {metadata.get("report_year") for metadata in retrieval_metadata if metadata.get("report_year") is not None}
    doc_source_types = {metadata.get("doc_source_type") for metadata in retrieval_metadata if metadata.get("doc_source_type")}
    periods = {metadata.get("period") for metadata in retrieval_metadata if metadata.get("period")}
    topic_flags = _union_metadata_flags(retrieval_results)
    table_grounding_result = answer.get("table_grounding_result") or {}

    if final_answer not in (None, "N/A", []):
        if not citations:
            validation_flags.append("missing_citations")
            confidence = _downgrade_confidence(confidence)

        if not relevant_pages:
            validation_flags.append("missing_relevant_pages")
            confidence = _downgrade_confidence(confidence)

    if query_plan.filters.currency and currencies and query_plan.filters.currency not in currencies:
        validation_flags.append("currency_mismatch")
        answer["final_answer"] = "N/A"
        answer["references"] = []
        answer["citations"] = []
        answer["relevant_pages"] = []
        confidence = "low"

    if query_plan.filters.year is not None and years and query_plan.filters.year not in years:
        validation_flags.append("report_year_mismatch")
        answer["final_answer"] = "N/A"
        answer["references"] = []
        answer["citations"] = []
        answer["relevant_pages"] = []
        confidence = "low"

    if query_plan.filters.doc_source_type and doc_source_types and query_plan.filters.doc_source_type not in doc_source_types:
        validation_flags.append("doc_source_type_mismatch")
        confidence = _downgrade_confidence(confidence)

    if query_plan.filters.period and periods and not any(_period_matches(query_plan.filters.period, period) for period in periods):
        validation_flags.append("period_filter_weak_match")
        confidence = _downgrade_confidence(confidence)

    if query_plan.topic_flags and topic_flags and not (set(query_plan.topic_flags) & topic_flags):
        validation_flags.append("topic_filter_weak_match")
        confidence = _downgrade_confidence(confidence)

    if query_plan.expected_answer_type == "numeric" and final_answer not in (None, "N/A"):
        grounded_period = table_grounding_result.get("period")
        grounded_unit = str(table_grounding_result.get("unit") or "")

        if table_grounding_result:
            if table_grounding_result.get("normalized_value") is None:
                validation_flags.append("numeric_grounding_missing_value")
                answer["final_answer"] = "N/A"
                answer["references"] = []
                answer["citations"] = []
                answer["relevant_pages"] = []
                confidence = "low"
            if query_plan.filters.period and grounded_period and not _period_matches(query_plan.filters.period, grounded_period):
                validation_flags.append("numeric_grounding_period_mismatch")
                answer["final_answer"] = "N/A"
                answer["references"] = []
                answer["citations"] = []
                answer["relevant_pages"] = []
                confidence = "low"
            if query_plan.filters.currency and grounded_unit:
                normalized_unit = grounded_unit.upper()
                if query_plan.filters.currency not in normalized_unit and not (
                    query_plan.filters.currency == "CNY" and "人民币" in grounded_unit
                ):
                    validation_flags.append("numeric_grounding_currency_mismatch")
                    answer["final_answer"] = "N/A"
                    answer["references"] = []
                    answer["citations"] = []
                    answer["relevant_pages"] = []
                    confidence = "low"
        elif not any((citation.get("chunk_type") in {"serialized_table", "table", "table_grounding"}) for citation in citations):
            validation_flags.append("numeric_answer_without_table_grounding")
            confidence = _downgrade_confidence(confidence)

    if not retrieval_results:
        validation_flags.append("no_retrieval_results")
        confidence = "low"

    if answer.get("final_answer") in (None, "N/A"):
        confidence = "low"

    if not validation_flags:
        confidence_reason = "Grounded answer validated against retrieval metadata and citation coverage."
    else:
        confidence_reason = "Validation adjusted answer confidence due to: " + ", ".join(validation_flags)

    answer["validation_flags"] = validation_flags
    answer["confidence"] = confidence
    answer["confidence_reason"] = confidence_reason
    return ValidatedAnswer(
        answer=answer,
        validation_flags=validation_flags,
        confidence=confidence,
        confidence_reason=confidence_reason,
    )
