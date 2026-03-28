from __future__ import annotations

from typing import Dict, Iterable, List


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
        )
        deduped[key] = citation
    return list(deduped.values())


def build_citations(retrieval_results: List[Dict], relevant_pages: List[int]) -> List[Dict]:
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
                "section_title": metadata.get("section_title"),
                "report_section": metadata.get("report_section"),
                "source": metadata.get("sha1_name"),
                "company_name": metadata.get("company_name"),
                "currency": metadata.get("currency"),
                "report_year": metadata.get("report_year"),
                "report_type": metadata.get("report_type"),
                "major_industry": metadata.get("major_industry"),
                "topic_flags": metadata.get("topic_flags", []),
                "parent_block_id": metadata.get("parent_block_id"),
                "evidence_type": metadata.get("evidence_type"),
                "has_table_context": metadata.get("has_table_context", False),
                "retrieval_sources": result.get("retrieval_sources", []),
                "evidence_snippet": _build_evidence_snippet(result.get("text", "")),
                "score": round(float(result.get("combined_score", result.get("ranking_score", result.get("distance", 0.0)))), 4),
            }
        )
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

    if top_score >= 0.8 and top_score - second_score >= 0.08 and page_count >= 1:
        return "high"
    if top_score >= 0.45 and page_count >= 1:
        return "medium"
    return "low"
