from __future__ import annotations

import re
from typing import Any, Dict, List

from eval.dataset_schema import FinanceEvalQuestion, FinanceGoldAnswer


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return " ".join(_normalize_text(item) for item in value)
    return " ".join(str(value).strip().lower().split())


def _normalize_stock_code(value: Any) -> str:
    if value is None:
        return ""
    digits = re.sub(r"\D+", "", str(value))
    if len(digits) >= 6:
        return digits[-6:]
    return digits


def _append_keyword(keywords: List[str], value: Any) -> None:
    normalized = _normalize_text(value)
    if not normalized or len(normalized) <= 1:
        return
    if normalized not in keywords:
        keywords.append(normalized)


def _report_type_aliases(value: str | None) -> List[str]:
    normalized = _normalize_text(value)
    if not normalized:
        return []
    alias_map = {
        "annual_report": ["annual report", "年度报告", "年报"],
        "q1_report": ["一季报", "第一季度报告"],
        "semi_annual_report": ["半年报", "中报", "半年度报告"],
        "q3_report": ["三季报", "第三季度报告"],
    }
    aliases = [normalized]
    aliases.extend(alias_map.get(normalized, []))
    deduped: List[str] = []
    for alias in aliases:
        normalized_alias = _normalize_text(alias)
        if normalized_alias and normalized_alias not in deduped:
            deduped.append(normalized_alias)
    return deduped


def _currency_aliases(value: str | None) -> List[str]:
    normalized = _normalize_text(value)
    if not normalized:
        return []
    alias_map = {
        "cny": ["人民币", "rmb", "元"],
        "usd": ["美元", "usd"],
        "eur": ["欧元", "eur"],
    }
    aliases = [normalized]
    aliases.extend(alias_map.get(normalized, []))
    deduped: List[str] = []
    for alias in aliases:
        normalized_alias = _normalize_text(alias)
        if normalized_alias and normalized_alias not in deduped:
            deduped.append(normalized_alias)
    return deduped


def _build_expected_keywords(question: FinanceEvalQuestion | None, gold_answer: FinanceGoldAnswer) -> List[str]:
    expected: List[str] = []

    company_name = question.company_name if question and question.company_name else gold_answer.company_name
    stock_code = question.stock_code if question and question.stock_code else gold_answer.stock_code
    report_year = question.report_year if question and question.report_year is not None else gold_answer.report_year
    report_type = question.report_type if question and question.report_type else gold_answer.report_type
    period = question.period if question and question.period else gold_answer.period
    metric_name = question.metric_name if question and question.metric_name else gold_answer.metric_name
    currency = question.currency if question and question.currency else gold_answer.currency
    unit = question.unit if question and question.unit else gold_answer.unit

    _append_keyword(expected, company_name)
    stock_code_normalized = _normalize_stock_code(stock_code)
    if stock_code_normalized:
        _append_keyword(expected, stock_code_normalized)
    _append_keyword(expected, report_year)
    _append_keyword(expected, period)
    _append_keyword(expected, metric_name)
    _append_keyword(expected, unit)

    for alias in _report_type_aliases(report_type):
        _append_keyword(expected, alias)
    for alias in _currency_aliases(currency):
        _append_keyword(expected, alias)

    question_keywords = (question.metadata or {}).get("keywords") if question else None
    gold_keywords = (gold_answer.metadata or {}).get("keywords")
    for value in (question_keywords or []):
        _append_keyword(expected, value)
    for value in (gold_keywords or []):
        _append_keyword(expected, value)

    return expected


def _build_matching_corpus(pred_answer: Dict[str, Any], debug_detail: Dict[str, Any] | None = None) -> str:
    parts: List[str] = []

    _append_keyword(parts, pred_answer.get("value"))
    for citation in pred_answer.get("citations", []) or []:
        _append_keyword(parts, citation.get("evidence_snippet"))
        _append_keyword(parts, citation.get("source"))
        _append_keyword(parts, citation.get("company_name"))
        _append_keyword(parts, citation.get("stock_code"))
        _append_keyword(parts, citation.get("security_code"))
        _append_keyword(parts, citation.get("report_year"))
        _append_keyword(parts, citation.get("period"))
        _append_keyword(parts, citation.get("unit"))
        for alias in _report_type_aliases(citation.get("report_type") or citation.get("doc_source_type")):
            _append_keyword(parts, alias)
        for alias in _currency_aliases(citation.get("currency")):
            _append_keyword(parts, alias)

    for result in (debug_detail or {}).get("retrieval_results", []) or []:
        _append_keyword(parts, result.get("text"))
        metadata = result.get("metadata") or {}
        _append_keyword(parts, metadata.get("company_name"))
        _append_keyword(parts, metadata.get("stock_code"))
        _append_keyword(parts, metadata.get("security_code"))
        _append_keyword(parts, metadata.get("report_year"))
        _append_keyword(parts, metadata.get("period"))
        _append_keyword(parts, metadata.get("unit_hint"))
        for alias in _report_type_aliases(metadata.get("report_type") or metadata.get("doc_source_type")):
            _append_keyword(parts, alias)
        for alias in _currency_aliases(metadata.get("currency")):
            _append_keyword(parts, alias)

    return " ".join(parts)


def score_answer_keywords(
    pred_answer: Dict[str, Any],
    gold_answer: FinanceGoldAnswer,
    question: FinanceEvalQuestion | None = None,
    *,
    debug_detail: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if gold_answer.should_refuse:
        return {
            "keyword_score": None,
            "keyword_hit_count": 0,
            "keyword_total": 0,
            "expected_keywords": [],
            "matched_keywords": [],
            "missing_keywords": [],
            "skipped": True,
            "skip_reason": "refusal_case",
        }

    expected_keywords = _build_expected_keywords(question, gold_answer)
    if not expected_keywords:
        return {
            "keyword_score": None,
            "keyword_hit_count": 0,
            "keyword_total": 0,
            "expected_keywords": [],
            "matched_keywords": [],
            "missing_keywords": [],
            "skipped": True,
            "skip_reason": "no_expected_keywords",
        }

    corpus = _build_matching_corpus(pred_answer, debug_detail=debug_detail)
    matched_keywords = [keyword for keyword in expected_keywords if keyword and keyword in corpus]
    missing_keywords = [keyword for keyword in expected_keywords if keyword not in matched_keywords]
    keyword_total = len(expected_keywords)
    keyword_hit_count = len(matched_keywords)
    keyword_score = round(keyword_hit_count / keyword_total, 4) if keyword_total else None

    return {
        "keyword_score": keyword_score,
        "keyword_hit_count": keyword_hit_count,
        "keyword_total": keyword_total,
        "expected_keywords": expected_keywords,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "skipped": False,
    }
