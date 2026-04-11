from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Set

from eval.dataset_schema import FinanceEvalQuestion, FinanceGoldAnswer


DEFAULT_ENTITY_WEIGHTS = {
    "doc_ids": 0.15,
    "company_name": 0.2,
    "stock_code": 0.1,
    "report_year": 0.15,
    "report_type": 0.1,
    "period": 0.1,
    "currency": 0.1,
    "unit": 0.1,
}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return "".join(str(value).strip().lower().split())


def _normalize_doc_id(value: Any) -> str:
    return _normalize_text(value)


def _normalize_stock_code(value: Any) -> str:
    if value is None:
        return ""
    digits = re.sub(r"\D+", "", str(value))
    if len(digits) >= 6:
        return digits[-6:]
    return digits


def _normalize_currency(value: Any) -> str:
    normalized = _normalize_text(value)
    currency_map = {
        "人民币": "cny",
        "rmb": "cny",
        "cny": "cny",
        "usd": "usd",
        "美元": "usd",
        "eur": "eur",
        "欧元": "eur",
    }
    return currency_map.get(normalized, normalized)


def _normalize_unit(value: Any) -> str:
    normalized = _normalize_text(value)
    normalized = normalized.replace("人民币", "")
    return normalized


def _extract_metadata_candidates(pred_answer: Dict, debug_detail: Dict | None = None) -> Dict[str, Set[Any]]:
    observed: Dict[str, Set[Any]] = defaultdict(set)

    for reference in pred_answer.get("references", []) or []:
        if reference.get("pdf_sha1"):
            observed["doc_ids"].add(reference["pdf_sha1"])

    for citation in pred_answer.get("citations", []) or []:
        if citation.get("source"):
            observed["doc_ids"].add(citation["source"])
        if citation.get("company_name"):
            observed["company_name"].add(citation["company_name"])
        if citation.get("stock_code"):
            observed["stock_code"].add(citation["stock_code"])
        if citation.get("security_code"):
            observed["stock_code"].add(citation["security_code"])
        if citation.get("report_year") is not None:
            observed["report_year"].add(citation["report_year"])
        if citation.get("report_type"):
            observed["report_type"].add(citation["report_type"])
        if citation.get("doc_source_type"):
            observed["report_type"].add(citation["doc_source_type"])
        if citation.get("currency"):
            observed["currency"].add(citation["currency"])
        if citation.get("unit"):
            observed["unit"].add(citation["unit"])
        if citation.get("period"):
            observed["period"].add(citation["period"])

    for result in (debug_detail or {}).get("retrieval_results", []) or []:
        metadata = result.get("metadata") or {}
        if metadata.get("sha1_name"):
            observed["doc_ids"].add(metadata["sha1_name"])
        if metadata.get("company_name"):
            observed["company_name"].add(metadata["company_name"])
        if metadata.get("stock_code"):
            observed["stock_code"].add(metadata["stock_code"])
        if metadata.get("security_code"):
            observed["stock_code"].add(metadata["security_code"])
        if metadata.get("report_year") is not None:
            observed["report_year"].add(metadata["report_year"])
        if metadata.get("report_type"):
            observed["report_type"].add(metadata["report_type"])
        if metadata.get("doc_source_type"):
            observed["report_type"].add(metadata["doc_source_type"])
        if metadata.get("currency"):
            observed["currency"].add(metadata["currency"])
        if metadata.get("unit_hint"):
            observed["unit"].add(metadata["unit_hint"])
        if metadata.get("period"):
            observed["period"].add(metadata["period"])

    return observed


def _build_expected_fields(question: FinanceEvalQuestion | None, gold_answer: FinanceGoldAnswer) -> Dict[str, Any]:
    return {
        "doc_ids": question.doc_ids if question and question.doc_ids else gold_answer.doc_ids,
        "company_name": question.company_name if question and question.company_name else gold_answer.company_name,
        "stock_code": question.stock_code if question and question.stock_code else gold_answer.stock_code,
        "report_year": question.report_year if question and question.report_year is not None else gold_answer.report_year,
        "report_type": question.report_type if question and question.report_type else gold_answer.report_type,
        "period": question.period if question and question.period else gold_answer.period,
        "currency": question.currency if question and question.currency else gold_answer.currency,
        "unit": question.unit if question and question.unit else gold_answer.unit,
    }


def _values_match(field_name: str, expected: Any, observed_values: Iterable[Any]) -> bool:
    observed_values = list(observed_values)
    if field_name == "doc_ids":
        expected_values = {_normalize_doc_id(item) for item in (expected or [])}
        observed_normalized = {_normalize_doc_id(item) for item in observed_values}
        return bool(expected_values and expected_values & observed_normalized)
    if field_name == "company_name":
        expected_value = _normalize_text(expected)
        observed_normalized = {_normalize_text(item) for item in observed_values}
        return any(
            observed == expected_value or observed in expected_value or expected_value in observed
            for observed in observed_normalized
            if observed and expected_value
        )
    if field_name == "stock_code":
        expected_value = _normalize_stock_code(expected)
        observed_normalized = {_normalize_stock_code(item) for item in observed_values}
        return bool(expected_value and expected_value in observed_normalized)
    if field_name == "report_year":
        return any(int(item) == int(expected) for item in observed_values if item is not None)
    if field_name == "currency":
        expected_value = _normalize_currency(expected)
        observed_normalized = {_normalize_currency(item) for item in observed_values}
        return bool(expected_value and expected_value in observed_normalized)
    if field_name == "unit":
        expected_value = _normalize_unit(expected)
        observed_normalized = {_normalize_unit(item) for item in observed_values}
        return any(
            observed == expected_value or observed in expected_value or expected_value in observed
            for observed in observed_normalized
            if observed and expected_value
        )
    expected_value = _normalize_text(expected)
    observed_normalized = {_normalize_text(item) for item in observed_values}
    return bool(expected_value and expected_value in observed_normalized)


def score_finance_entities(
    pred_answer: Dict,
    gold_answer: FinanceGoldAnswer,
    question: FinanceEvalQuestion | None = None,
    *,
    debug_detail: Dict | None = None,
    field_weights: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    if gold_answer.should_refuse:
        return {
            "entity_score": None,
            "active_field_count": 0,
            "matched_field_count": 0,
            "field_scores": {},
            "skipped": True,
            "skip_reason": "refusal_case",
        }

    weights = field_weights or DEFAULT_ENTITY_WEIGHTS
    observed = _extract_metadata_candidates(pred_answer, debug_detail=debug_detail)
    expected_fields = _build_expected_fields(question, gold_answer)

    field_scores: Dict[str, Dict[str, Any]] = {}
    weighted_sum = 0.0
    active_weight = 0.0
    matched_field_count = 0
    active_field_count = 0

    for field_name, expected_value in expected_fields.items():
        if expected_value in (None, "", []):
            continue

        active_field_count += 1
        observed_values = sorted(observed.get(field_name, set()), key=lambda item: str(item))
        weight = weights.get(field_name, 0.0)
        score = 1.0 if _values_match(field_name, expected_value, observed_values) else 0.0
        if score == 1.0:
            matched_field_count += 1

        field_scores[field_name] = {
            "expected": expected_value,
            "observed": observed_values,
            "score": score,
            "weight": weight,
            "status": "matched" if score == 1.0 else ("missing_observation" if not observed_values else "mismatch"),
        }
        weighted_sum += score * weight
        active_weight += weight

    entity_score = round(weighted_sum / active_weight, 4) if active_weight else None
    return {
        "entity_score": entity_score,
        "active_field_count": active_field_count,
        "matched_field_count": matched_field_count,
        "field_scores": field_scores,
        "skipped": False,
    }
