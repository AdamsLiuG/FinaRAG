from __future__ import annotations

from typing import Dict, Iterable, List, Optional


def _build_evidence_snippet(text: str, limit: int = 220) -> str:
    if not text:
        return ""
    snippet = " ".join(text.split())
    return snippet[:limit].rstrip() + ("..." if len(snippet) > limit else "")


def dedupe_references(references: Iterable[Dict]) -> List[Dict]:
    deduped = {}
    for reference in references:
        key = (reference.get("pdf_sha1"), reference.get("page_index"))
        deduped[key] = reference
    return list(deduped.values())


def dedupe_citations(citations: Iterable[Dict]) -> List[Dict]:
    deduped = {}
    for citation in citations:
        key = (
            citation.get("source"),
            citation.get("page"),
            citation.get("chunk_id"),
            citation.get("chunk_type"),
            citation.get("table_id"),
            tuple(citation.get("matched_row_headers") or []),
            tuple(citation.get("matched_col_headers") or []),
        )
        deduped[key] = citation
    return list(deduped.values())


def _build_table_grounding_citation(table_grounding_result: Optional[Dict]) -> Optional[Dict]:
    if not table_grounding_result:
        return None
    return {
        "page": table_grounding_result.get("page"),
        "chunk_id": None,
        "chunk_type": "table_grounding",
        "node_type": "table",
        "parent_chunk_id": None,
        "matched_child_chunk_ids": [],
        "matched_tags": [],
        "section_title": None,
        "section_name": None,
        "report_section": None,
        "source": table_grounding_result.get("source_doc_id"),
        "company_name": None,
        "currency": None,
        "report_year": None,
        "report_type": None,
        "major_industry": None,
        "topic_flags": [],
        "table_id": table_grounding_result.get("table_id"),
        "row_idx": table_grounding_result.get("row_idx"),
        "col_idx": table_grounding_result.get("col_idx"),
        "matched_row_headers": table_grounding_result.get("matched_row_headers") or [],
        "matched_col_headers": table_grounding_result.get("matched_col_headers") or [],
        "unit": table_grounding_result.get("unit"),
        "footnote_refs": table_grounding_result.get("footnote_refs") or [],
        "parent_block_id": None,
        "evidence_type": "table",
        "has_table_context": True,
        "retrieval_sources": ["table_grounding"],
        "evidence_snippet": _build_evidence_snippet(table_grounding_result.get("table_snippet", "")),
        "score": round(float(table_grounding_result.get("match_score", 0.0)), 4),
    }


def build_citations(
    retrieval_results: List[Dict],
    relevant_pages: List[int],
    table_grounding_result: Optional[Dict] = None,
) -> List[Dict]:
    citations: List[Dict] = []
    relevant_pages_set = set(relevant_pages or [])
    for result in retrieval_results:
        if result.get("page") not in relevant_pages_set:
            continue
        metadata = result.get("metadata", {})
        citations.append(
            {
                "page": result.get("page"),
                "chunk_id": metadata.get("chunk_id"),
                "chunk_type": metadata.get("chunk_type"),
                "node_type": metadata.get("node_type"),
                "parent_chunk_id": metadata.get("parent_chunk_id"),
                "matched_child_chunk_ids": result.get("matched_child_chunk_ids", []),
                "matched_tags": result.get("matched_tags", []),
                "section_title": metadata.get("section_title"),
                "section_name": metadata.get("section_name"),
                "report_section": metadata.get("report_section"),
                "source": metadata.get("sha1_name"),
                "company_name": metadata.get("company_name"),
                "security_code": metadata.get("security_code"),
                "stock_code": metadata.get("stock_code"),
                "broker_name": metadata.get("broker_name"),
                "currency": metadata.get("currency"),
                "report_year": metadata.get("report_year"),
                "report_type": metadata.get("report_type"),
                "doc_source_type": metadata.get("doc_source_type"),
                "major_industry": metadata.get("major_industry"),
                "topic_flags": metadata.get("topic_flags", []),
                "table_id": metadata.get("table_id"),
                "row_idx": None,
                "col_idx": None,
                "matched_row_headers": [],
                "matched_col_headers": [],
                "unit": metadata.get("unit_hint"),
                "footnote_refs": [],
                "parent_block_id": metadata.get("parent_block_id"),
                "evidence_type": metadata.get("evidence_type"),
                "has_table_context": metadata.get("has_table_context", False),
                "retrieval_sources": result.get("retrieval_sources", []),
                "evidence_snippet": _build_evidence_snippet(result.get("text", "")),
                "score": round(float(result.get("combined_score", result.get("ranking_score", result.get("distance", 0.0)))), 4),
            }
        )
    table_citation = _build_table_grounding_citation(table_grounding_result)
    if table_citation and (
        not relevant_pages_set or table_citation.get("page") in relevant_pages_set
    ):
        citations.append(table_citation)
    return dedupe_citations(citations)


def compute_confidence(answer_dict: Dict, retrieval_results: List[Dict]) -> str:
    final_answer = answer_dict.get("final_answer")
    if final_answer in (None, "N/A"):
        return "low"
    if not retrieval_results:
        return "low"

    scores = [
        float(result.get("combined_score", result.get("ranking_score", result.get("distance", 0.0))))
        for result in retrieval_results
    ]
    top_score = scores[0]
    second_score = scores[1] if len(scores) > 1 else 0.0
    page_count = len(answer_dict.get("relevant_pages") or [])
    citations = answer_dict.get("citations") or []
    table_grounding_result = answer_dict.get("table_grounding_result")
    validation_flags = answer_dict.get("validation_flags") or []
    citation_coverage = len(citations) / max(page_count, 1)

    if table_grounding_result and top_score >= 0.45 and page_count >= 1 and not validation_flags:
        return "high"
    if top_score >= 0.8 and top_score - second_score >= 0.08 and page_count >= 1 and citation_coverage >= 1:
        return "high"
    if top_score >= 0.45 and page_count >= 1:
        return "medium"
    return "low"
