from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.text_normalization import normalize_currency_token


@dataclass
class RetrievalFilters:
    company_name: Optional[str] = None
    currency: Optional[str] = None
    year: Optional[int] = None
    question_kind: Optional[str] = None


def build_result_metadata(document_meta: Dict, chunk: Dict | None = None) -> Dict:
    chunk = chunk or {}
    return {
        "company_name": document_meta.get("company_name"),
        "currency": normalize_currency_token(document_meta.get("currency")),
        "major_industry": document_meta.get("major_industry"),
        "report_year": chunk.get("report_year", document_meta.get("report_year")),
        "chunk_id": chunk.get("chunk_id", chunk.get("id")),
        "chunk_type": chunk.get("chunk_type", chunk.get("type", "content")),
        "section_title": chunk.get("section_title"),
        "table_id": chunk.get("table_id"),
        "sha1_name": document_meta.get("sha1_name"),
    }


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
    return True


def _question_kind_bonus(result: Dict, filters: RetrievalFilters | None) -> float:
    if filters is None or not filters.question_kind:
        return 0.0

    chunk_type = (result.get("metadata") or {}).get("chunk_type")
    if filters.question_kind == "number":
        return {
            "serialized_table": 0.12,
            "table": 0.1,
            "content": 0.0,
        }.get(chunk_type, 0.0)
    if filters.question_kind == "names":
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
