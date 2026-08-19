from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.query_plan import QueryPlan
from src.text_normalization import normalize_text


_CONFIDENCE_ORDER = ["low", "medium", "high"]
_VALIDATION_FLAG_LABELS = {
    "missing_citations": "缺少 citation 覆盖",
    "missing_relevant_pages": "缺少相关页码",
    "currency_mismatch": "币种与问题不一致",
    "report_year_mismatch": "报告年份不一致",
    "doc_source_type_mismatch": "文档类型不一致",
    "period_filter_weak_match": "期间匹配较弱",
    "topic_filter_weak_match": "主题匹配较弱",
    "numeric_grounding_missing_value": "数字 grounding 缺少值",
    "numeric_grounding_period_mismatch": "数字 grounding 的期间不匹配",
    "numeric_grounding_currency_mismatch": "数字 grounding 的币种不一致",
    "numeric_answer_without_table_grounding": "数字答案缺少表格 grounding",
    "numeric_answer_without_structured_grounding": "数字答案缺少结构化 grounding",
    "chart_grounding_low_confidence": "图表 grounding 置信度较低",
    "chart_unit_mismatch": "图表单位与问题不一致",
    "chart_year_mismatch": "图表年份与问题不一致",
    "implicit_intent_evidence_mismatch": "隐含意图缺少对应证据",
    "metric_evidence_mismatch": "指标证据与问题不匹配",
    "topk_result_count_mismatch": "top-k 返回数量不足",
    "no_retrieval_results": "未返回检索证据",
    "processing_error": "处理流程报错",
}


def _normalize_confidence(value: str | None) -> str:
    normalized = (value or "low").lower()
    return normalized if normalized in _CONFIDENCE_ORDER else "low"


def _downgrade_confidence(value: str, steps: int = 1) -> str:
    current = _CONFIDENCE_ORDER.index(_normalize_confidence(value))
    return _CONFIDENCE_ORDER[max(0, current - steps)]


def _humanize_validation_flag(flag: str) -> str:
    return _VALIDATION_FLAG_LABELS.get(flag, flag)


def _union_metadata_flags(retrieval_results: List[Dict[str, Any]]) -> set[str]:
    topic_flags: set[str] = set()
    for result in retrieval_results:
        metadata = result.get("metadata") or {}
        topic_flags.update(metadata.get("topic_flags") or [])
    return topic_flags


def _retrieval_evidence_blob(retrieval_results: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    metadata_fields = (
        "section_title",
        "section_name",
        "report_section",
        "chunk_type",
        "evidence_type",
        "table_id",
    )
    list_metadata_fields = (
        "topic_flags",
        "business_tags",
        "strategy_tags",
        "factor_tags",
        "matched_tags",
        "retrieval_sources",
        "matched_row_headers",
        "matched_col_headers",
    )
    for result in retrieval_results:
        if result.get("text"):
            parts.append(str(result.get("text")))
        metadata = result.get("metadata") or {}
        for field in metadata_fields:
            if metadata.get(field):
                parts.append(str(metadata.get(field)))
        for field in list_metadata_fields:
            values = metadata.get(field)
            if isinstance(values, list):
                parts.extend(str(value) for value in values if value not in (None, ""))
            elif values:
                parts.append(str(values))
        for field in ("matched_tags", "retrieval_sources"):
            values = result.get(field)
            if isinstance(values, list):
                parts.extend(str(value) for value in values if value not in (None, ""))
            elif values:
                parts.append(str(values))
    return normalize_text(" ".join(parts))


def _evidence_contains_any_target(retrieval_results: List[Dict[str, Any]], targets: List[str]) -> bool:
    normalized_blob = _retrieval_evidence_blob(retrieval_results)
    if not normalized_blob:
        return False
    for target in targets:
        normalized_target = normalize_text(str(target or ""))
        if len(normalized_target) >= 2 and normalized_target in normalized_blob:
            return True
    return False


def _playbook_metric_coverage(query_plan, retrieval_results: List[Dict[str, Any]]) -> Optional[float]:
    """剧本必需指标的证据覆盖率（无剧本或无必需指标时返回 None）。

    校验粒度用各原子完整 evidence_targets（宽松覆盖别名与报表名），与检索粒度区分。
    """
    payload = getattr(query_plan, "playbook", None) or {}
    intent = payload.get("intent")
    if not intent:
        return None
    try:
        from src.analysis_playbooks import get_default_playbook_store

        store = get_default_playbook_store()
    except Exception:
        return None
    playbook = store.get_by_intent(str(intent))
    if playbook is None or not playbook.required_metrics:
        return None
    hits, total = store.metric_coverage(playbook, _retrieval_evidence_blob(retrieval_results))
    if not total:
        return None
    return hits / total


def _topk_answer_count(final_answer: Any) -> int | None:
    if isinstance(final_answer, list):
        return len(final_answer)
    if isinstance(final_answer, dict):
        for key in ("items", "companies", "results", "top_k"):
            value = final_answer.get(key)
            if isinstance(value, list):
                return len(value)
    return None


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
    chart_grounding_result = answer.get("chart_grounding_result") or {}

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

    if final_answer not in (None, "N/A", []) and retrieval_results:
        evidence_targets = list(query_plan.evidence_targets or [])
        if query_plan.task_type == "implicit_intent_qa" and evidence_targets:
            if not _evidence_contains_any_target(retrieval_results, evidence_targets):
                validation_flags.append("implicit_intent_evidence_mismatch")
                confidence = _downgrade_confidence(confidence)
            else:
                # 剧本必需指标覆盖度：部分命中时不算完全不匹配，但据不完整证据给结论同样需降级
                coverage = _playbook_metric_coverage(query_plan, retrieval_results)
                if coverage is not None and coverage < 0.5:
                    validation_flags.append("implicit_intent_evidence_incomplete")
                    confidence = _downgrade_confidence(confidence)

        metric_targets = [query_plan.metric] if query_plan.metric else []
        metric_targets.extend(evidence_targets)
        if query_plan.task_type in {"single_metric_lookup", "metric_aggregation", "comparison"} and metric_targets:
            has_structured_grounding = bool(table_grounding_result or chart_grounding_result)
            if not has_structured_grounding and not _evidence_contains_any_target(retrieval_results, metric_targets):
                validation_flags.append("metric_evidence_mismatch")
                confidence = _downgrade_confidence(confidence)

        if query_plan.operation == "top_k" and query_plan.limit:
            answer_count = _topk_answer_count(final_answer)
            if answer_count is not None and answer_count < query_plan.limit:
                validation_flags.append("topk_result_count_mismatch")
                confidence = _downgrade_confidence(confidence)

    if query_plan.expected_answer_type == "numeric" and final_answer not in (None, "N/A"):
        grounded_period = table_grounding_result.get("period")
        grounded_unit = str(table_grounding_result.get("unit") or "")
        chart_period = str(chart_grounding_result.get("period") or chart_grounding_result.get("x_label") or "")
        chart_unit = str(chart_grounding_result.get("unit") or "")

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
        elif chart_grounding_result:
            if chart_grounding_result.get("normalized_value") is None:
                validation_flags.append("numeric_grounding_missing_value")
                answer["final_answer"] = "N/A"
                answer["references"] = []
                answer["citations"] = []
                answer["relevant_pages"] = []
                confidence = "low"
            chart_confidence = chart_grounding_result.get("confidence", chart_grounding_result.get("chart_confidence"))
            try:
                chart_confidence_value = float(chart_confidence)
            except (TypeError, ValueError):
                chart_confidence_value = 0.0
            if chart_confidence_value < 0.7:
                validation_flags.append("chart_grounding_low_confidence")
                confidence = _downgrade_confidence(confidence)
            if query_plan.filters.year is not None and chart_period and str(query_plan.filters.year) not in chart_period:
                validation_flags.append("chart_year_mismatch")
                answer["final_answer"] = "N/A"
                answer["references"] = []
                answer["citations"] = []
                answer["relevant_pages"] = []
                confidence = "low"
            if query_plan.filters.period and chart_period and not _period_matches(query_plan.filters.period, chart_period):
                validation_flags.append("chart_year_mismatch")
                answer["final_answer"] = "N/A"
                answer["references"] = []
                answer["citations"] = []
                answer["relevant_pages"] = []
                confidence = "low"
            question_text = query_plan.original_query or ""
            if ("%" in question_text or "占比" in question_text or "比例" in question_text) and "%" not in chart_unit:
                validation_flags.append("chart_unit_mismatch")
                confidence = _downgrade_confidence(confidence)
        elif not any(
            citation.get("chunk_type") in {"serialized_table", "table", "table_grounding", "chart_to_table", "chart_grounding"}
            for citation in citations
        ):
            validation_flags.append("numeric_answer_without_structured_grounding")
            confidence = _downgrade_confidence(confidence)

    if not retrieval_results:
        validation_flags.append("no_retrieval_results")
        confidence = "low"

    if answer.get("final_answer") in (None, "N/A"):
        confidence = "low"

    if not validation_flags:
        confidence_reason = "答案已通过检索元数据与 citation 覆盖校验。"
    else:
        confidence_reason = "答案置信度已因以下因素下调：" + "，".join(
            _humanize_validation_flag(flag) for flag in validation_flags
        )

    answer["validation_flags"] = validation_flags
    answer["confidence"] = confidence
    answer["confidence_reason"] = confidence_reason
    return ValidatedAnswer(
        answer=answer,
        validation_flags=validation_flags,
        confidence=confidence,
        confidence_reason=confidence_reason,
    )
