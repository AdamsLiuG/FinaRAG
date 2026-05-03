from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.text_normalization import normalize_currency_token, normalize_text


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _extract_topic_flags(metadata: Dict) -> List[str]:
    return sorted(
        key
        for key, value in metadata.items()
        if key.startswith(("has_", "mentions_")) and _truthy(value)
    )


@dataclass
class RetrievalFilters:
    company_name: Optional[str] = None
    currency: Optional[str] = None
    year: Optional[int] = None
    report_type: Optional[str] = None
    doc_source_type: Optional[str] = None
    major_industry: Optional[str] = None
    exchange: Optional[str] = None
    board: Optional[str] = None
    market_type: Optional[str] = None
    industry_l1: Optional[str] = None
    industry_l2: Optional[str] = None
    security_code: Optional[str] = None
    broker_name: Optional[str] = None
    period: Optional[str] = None
    section_name: Optional[str] = None
    business_tags: Optional[List[str]] = None
    strategy_tags: Optional[List[str]] = None
    factor_tags: Optional[List[str]] = None
    chain_position_major: Optional[str] = None
    chain_position_minor: Optional[List[str]] = None
    listing_tags: Optional[List[str]] = None
    ownership_tags: Optional[List[str]] = None
    status_tags: Optional[List[str]] = None
    style_tags: Optional[List[str]] = None
    required_topic_flags: Optional[List[str]] = None
    candidate_doc_ids: Optional[List[str]] = None
    question_kind: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "company_name": self.company_name,
            "currency": self.currency,
            "year": self.year,
            "report_type": self.report_type,
            "doc_source_type": self.doc_source_type,
            "major_industry": self.major_industry,
            "exchange": self.exchange,
            "board": self.board,
            "market_type": self.market_type,
            "industry_l1": self.industry_l1,
            "industry_l2": self.industry_l2,
            "security_code": self.security_code,
            "broker_name": self.broker_name,
            "period": self.period,
            "section_name": self.section_name,
            "business_tags": list(self.business_tags or []),
            "strategy_tags": list(self.strategy_tags or []),
            "factor_tags": list(self.factor_tags or []),
            "chain_position_major": self.chain_position_major,
            "chain_position_minor": list(self.chain_position_minor or []),
            "listing_tags": list(self.listing_tags or []),
            "ownership_tags": list(self.ownership_tags or []),
            "status_tags": list(self.status_tags or []),
            "style_tags": list(self.style_tags or []),
            "required_topic_flags": list(self.required_topic_flags or []),
            "candidate_doc_ids": list(self.candidate_doc_ids or []),
            "question_kind": self.question_kind,
        }


def build_result_metadata(document_meta: Dict, chunk: Dict | None = None) -> Dict:
    chunk = chunk or {}
    topic_flags = sorted(set(_extract_topic_flags(document_meta)) | set(chunk.get("topic_flags") or []))
    node_type = chunk.get("node_type")
    if not node_type:
        if chunk.get("chunk_type") == "page":
            node_type = "page"
        elif chunk.get("parent_chunk_id") is not None:
            node_type = "child"
        elif chunk.get("child_chunk_ids") is not None:
            node_type = "parent"
        else:
            node_type = "child"
    return {
        "company_name": document_meta.get("company_name"),
        "company_aliases": list(document_meta.get("company_aliases") or []),
        "security_code": document_meta.get("security_code"),
        "stock_code": chunk.get("stock_code", document_meta.get("stock_code", document_meta.get("security_code"))),
        "broker_name": document_meta.get("broker_name"),
        "currency": normalize_currency_token(document_meta.get("currency")),
        "major_industry": document_meta.get("major_industry"),
        "exchange": chunk.get("exchange", document_meta.get("exchange")),
        "board": chunk.get("board", document_meta.get("board")),
        "market_type": chunk.get("market_type", document_meta.get("market_type")),
        "industry_l1": chunk.get("industry_l1", document_meta.get("industry_l1")),
        "industry_l2": chunk.get("industry_l2", document_meta.get("industry_l2")),
        "report_year": chunk.get("report_year", document_meta.get("report_year")),
        "report_type": chunk.get("report_type", document_meta.get("report_type")),
        "doc_source_type": chunk.get("doc_source_type", document_meta.get("doc_source_type")),
        "report_date": chunk.get("report_date", document_meta.get("report_date")),
        "fiscal_year": chunk.get("fiscal_year", document_meta.get("fiscal_year")),
        "period": chunk.get("period", document_meta.get("period")),
        "unit_hint": chunk.get("unit_hint", document_meta.get("unit_hint")),
        "language": chunk.get("language", document_meta.get("language")),
        "topic_flags": topic_flags,
        "chunk_id": chunk.get("chunk_id", chunk.get("id")),
        "chunk_type": chunk.get("chunk_type", chunk.get("type", "content")),
        "section_title": chunk.get("section_title"),
        "section_name": chunk.get("section_name", chunk.get("report_section", chunk.get("section_title"))),
        "report_section": chunk.get("report_section", chunk.get("section_name", chunk.get("section_title"))),
        "section_l1": chunk.get("section_l1"),
        "section_l2": chunk.get("section_l2"),
        "section_l3": chunk.get("section_l3"),
        "section_path": chunk.get("section_path", chunk.get("report_section", chunk.get("section_name"))),
        "section_leaf": chunk.get("section_leaf", chunk.get("section_name", chunk.get("section_title"))),
        "table_id": chunk.get("table_id"),
        "chart_id": chunk.get("chart_id"),
        "picture_id": chunk.get("picture_id"),
        "chart_type": chunk.get("chart_type"),
        "series_name": chunk.get("series_name"),
        "x_label": chunk.get("x_label"),
        "chart_confidence": chunk.get("chart_confidence"),
        "parent_block_id": chunk.get("parent_block_id"),
        "parent_chunk_id": chunk.get("parent_chunk_id"),
        "child_chunk_ids": list(chunk.get("child_chunk_ids") or []),
        "node_type": node_type,
        "evidence_type": chunk.get("evidence_type"),
        "has_table_context": bool(chunk.get("has_table_context")),
        "has_chart_context": bool(chunk.get("has_chart_context")),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "business_tags": list(chunk.get("business_tags") or document_meta.get("business_tags") or []),
        "strategy_tags": list(chunk.get("strategy_tags") or document_meta.get("strategy_tags") or []),
        "factor_tags": list(chunk.get("factor_tags") or document_meta.get("factor_tags") or []),
        "chain_position_major": chunk.get("chain_position_major", document_meta.get("chain_position_major")),
        "chain_position_minor": list(chunk.get("chain_position_minor") or document_meta.get("chain_position_minor") or []),
        "listing_tags": list(chunk.get("listing_tags") or document_meta.get("listing_tags") or []),
        "ownership_tags": list(chunk.get("ownership_tags") or document_meta.get("ownership_tags") or []),
        "status_tags": list(chunk.get("status_tags") or document_meta.get("status_tags") or []),
        "style_tags": list(chunk.get("style_tags") or document_meta.get("style_tags") or []),
        "sha1_name": document_meta.get("sha1_name"),
    }


def _matches_text_filter(expected: Optional[str], observed: Optional[str]) -> bool:
    if not expected or not observed:
        return True
    expected_norm = normalize_text(expected)
    observed_norm = normalize_text(observed)
    return expected_norm in observed_norm or observed_norm in expected_norm


def _matches_section_filter(expected: Optional[str], metadata: Dict) -> bool:
    if not expected:
        return True

    section_candidates = [
        metadata.get("section_name"),
        metadata.get("section_title"),
        metadata.get("report_section"),
        metadata.get("section_leaf"),
        metadata.get("section_path"),
        metadata.get("section_l1"),
        metadata.get("section_l2"),
        metadata.get("section_l3"),
    ]
    observed_values = [value for value in section_candidates if value]
    if not observed_values:
        return False

    return any(_matches_text_filter(expected, observed) for observed in observed_values)


def _matches_list_filter(expected: Optional[List[str]], observed: Optional[List[str]]) -> bool:
    if not expected:
        return True
    observed_norm = {normalize_text(str(item)) for item in observed or [] if item not in (None, "")}
    if not observed_norm:
        return False
    for item in expected:
        normalized = normalize_text(str(item))
        if not normalized:
            continue
        if not any(normalized in candidate or candidate in normalized for candidate in observed_norm):
            return False
    return True


def _matches_filters(result: Dict, filters: RetrievalFilters | None) -> bool:
    if filters is None:
        return True

    metadata = result.get("metadata", {})
    if filters.company_name and metadata.get("company_name") and metadata.get("company_name") != filters.company_name:
        return False
    if filters.currency and metadata.get("currency") and metadata.get("currency") != normalize_currency_token(filters.currency):
        return False
    if filters.year is not None and metadata.get("report_year") is not None and metadata.get("report_year") != filters.year:
        return False
    if filters.report_type and metadata.get("report_type") and not _matches_text_filter(filters.report_type, metadata.get("report_type")):
        return False
    if filters.doc_source_type and metadata.get("doc_source_type") and not _matches_text_filter(filters.doc_source_type, metadata.get("doc_source_type")):
        return False
    if filters.major_industry and metadata.get("major_industry") and not _matches_text_filter(filters.major_industry, metadata.get("major_industry")):
        return False
    if filters.exchange and metadata.get("exchange") and not _matches_text_filter(filters.exchange, metadata.get("exchange")):
        return False
    if filters.board and metadata.get("board") and not _matches_text_filter(filters.board, metadata.get("board")):
        return False
    if filters.market_type and metadata.get("market_type") and not _matches_text_filter(filters.market_type, metadata.get("market_type")):
        return False
    if filters.industry_l1 and metadata.get("industry_l1") and not _matches_text_filter(filters.industry_l1, metadata.get("industry_l1")):
        return False
    if filters.industry_l2 and metadata.get("industry_l2") and not _matches_text_filter(filters.industry_l2, metadata.get("industry_l2")):
        return False
    if filters.security_code:
        observed_security_code = metadata.get("security_code") or metadata.get("stock_code")
        if observed_security_code and str(observed_security_code) != str(filters.security_code):
            return False
    if filters.broker_name and metadata.get("broker_name") and metadata.get("broker_name") != filters.broker_name:
        return False
    if filters.period and metadata.get("period") and metadata.get("period") != filters.period:
        return False
    if filters.section_name and not _matches_section_filter(filters.section_name, metadata):
        return False
    if filters.chain_position_major and metadata.get("chain_position_major") and not _matches_text_filter(filters.chain_position_major, metadata.get("chain_position_major")):
        return False
    if not _matches_list_filter(filters.business_tags, metadata.get("business_tags")):
        return False
    if not _matches_list_filter(filters.strategy_tags, metadata.get("strategy_tags")):
        return False
    if not _matches_list_filter(filters.factor_tags, metadata.get("factor_tags")):
        return False
    if not _matches_list_filter(filters.chain_position_minor, metadata.get("chain_position_minor")):
        return False
    if not _matches_list_filter(filters.listing_tags, metadata.get("listing_tags")):
        return False
    if not _matches_list_filter(filters.ownership_tags, metadata.get("ownership_tags")):
        return False
    if not _matches_list_filter(filters.status_tags, metadata.get("status_tags")):
        return False
    if not _matches_list_filter(filters.style_tags, metadata.get("style_tags")):
        return False
    if filters.required_topic_flags:
        available_flags = set(metadata.get("topic_flags") or [])
        if not set(filters.required_topic_flags).issubset(available_flags):
            return False
    if filters.candidate_doc_ids:
        if metadata.get("sha1_name") not in set(filters.candidate_doc_ids):
            return False
    return True


def _question_kind_bonus(result: Dict, filters: RetrievalFilters | None) -> float:
    if filters is None or not filters.question_kind:
        return 0.0

    chunk_type = (result.get("metadata") or {}).get("chunk_type")
    if filters.question_kind == "number":
        return {
            "serialized_table": 0.12,
            "chart_to_table": 0.08,
            "table": 0.1,
            "content": 0.0,
        }.get(chunk_type, 0.0)
    if filters.question_kind == "boolean":
        topic_flags = (result.get("metadata") or {}).get("topic_flags") or []
        if filters.required_topic_flags and set(filters.required_topic_flags) & set(topic_flags):
            return 0.08
        return 0.02 if chunk_type == "content" else 0.0
    if filters.question_kind in {"names", "name"}:
        return 0.04 if chunk_type == "content" else 0.0
    return 0.0


def apply_retrieval_filters(results: List[Dict], filters: RetrievalFilters | None) -> List[Dict]:
    filtered = [result for result in results if _matches_filters(result, filters)]
    for result in filtered:
        base_score = result.get("combined_score", result.get("distance", 0.0))
        result["filter_bonus"] = round(_question_kind_bonus(result, filters), 4)
        result["ranking_score"] = round(float(base_score) + result["filter_bonus"], 4)

    filtered.sort(key=lambda item: item.get("ranking_score", item.get("combined_score", item.get("distance", 0.0))), reverse=True)
    return filtered
