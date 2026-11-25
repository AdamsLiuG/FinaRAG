"""Format retrieved evidence and project references for QA answers."""

from typing import Callable, Dict, List, Optional


def format_retrieval_results(processor, retrieval_results: List[Dict]) -> str:
    if not retrieval_results:
        return ""

    context_parts = []
    total_chars = 0
    for result in retrieval_results:
        page_number = result['page']
        text = result['text']
        metadata = result.get("metadata", {})
        section_name = metadata.get("section_name") or metadata.get("section_title")
        chunk_type = metadata.get("chunk_type")
        node_type = metadata.get("node_type")
        matched_tags = result.get("matched_tags") or []
        doc_id = metadata.get("sha1_name") or metadata.get("doc_id")
        company = metadata.get("company_name")
        report_year = metadata.get("report_year") or metadata.get("fiscal_year")

        if processor.max_doc_chars > 0 and len(text) > processor.max_doc_chars:
            text = text[: processor.max_doc_chars].rstrip() + "\n...[truncated]"

        label = f"Text retrieved from page {page_number}"
        if company:
            label += f" | company: {company}"
        if doc_id:
            label += f" | doc_id: {doc_id}"
        if report_year:
            label += f" | report_year: {report_year}"
        if section_name:
            label += f" | section: {section_name}"
        if chunk_type:
            label += f" | chunk_type: {chunk_type}"
        if node_type:
            label += f" | node_type: {node_type}"
        if matched_tags:
            label += f" | matched_tags: {', '.join(matched_tags)}"

        part = f'{label}: \n"""\n{text}\n"""'

        if processor.max_context_chars > 0 and total_chars + len(part) > processor.max_context_chars:
            remaining = processor.max_context_chars - total_chars
            if remaining <= 0:
                break
            part = part[:remaining].rstrip() + "\n...[truncated]"

        context_parts.append(part)
        total_chars += len(part)

        if processor.max_context_chars > 0 and total_chars >= processor.max_context_chars:
            break

    return "\n\n---\n\n".join(context_parts)


def extract_references(processor, pages_list: List[int], company_name: str, pdf_sha1: Optional[str] = None) -> List[Dict]:
    company_sha1 = pdf_sha1 or ""
    if not company_sha1:
        if processor.report_catalog is not None:
            report = processor.report_catalog.get_report_by_company_name(company_name)
            if report is not None:
                company_sha1 = report.sha1
        companies_df = processor._load_companies_df()
        if not company_sha1 and "company_name" in companies_df.columns:
            matching_rows = companies_df[companies_df["company_name"] == company_name]
            if not matching_rows.empty:
                for column in ("sha1", "sha1_name", "doc_id"):
                    if column in matching_rows.columns:
                        company_sha1 = matching_rows.iloc[0][column]
                        if company_sha1:
                            break
    return [{"pdf_sha1": company_sha1, "page": page} for page in pages_list]


def extract_references_from_results(pages_list: List[int], retrieval_results: List[Dict]) -> List[Dict]:
    pages = set(pages_list or [])
    references = []
    seen = set()
    for result in retrieval_results:
        page = result.get("page")
        if page not in pages:
            continue
        metadata = result.get("metadata") or {}
        pdf_sha1 = metadata.get("sha1_name") or metadata.get("doc_id")
        key = (pdf_sha1, page)
        if key in seen:
            continue
        seen.add(key)
        references.append(
            {
                "pdf_sha1": pdf_sha1,
                "page": page,
            }
        )
    return references


def serialize_retrieval_result(result: Dict, *, result_score: Callable) -> Dict:
    metadata = result.get("metadata", {})
    return {
        "page": result.get("page"),
        "sha1_name": metadata.get("sha1_name"),
        "chunk_id": metadata.get("chunk_id"),
        "chunk_type": metadata.get("chunk_type"),
        "node_type": metadata.get("node_type"),
        "parent_chunk_id": metadata.get("parent_chunk_id"),
        "section_title": metadata.get("section_title"),
        "section_name": metadata.get("section_name"),
        "report_section": metadata.get("report_section"),
        "company_name": metadata.get("company_name"),
        "company_aliases": metadata.get("company_aliases", []),
        "security_code": metadata.get("security_code"),
        "stock_code": metadata.get("stock_code"),
        "broker_name": metadata.get("broker_name"),
        "currency": metadata.get("currency"),
        "exchange": metadata.get("exchange"),
        "board": metadata.get("board"),
        "market_type": metadata.get("market_type"),
        "industry_l1": metadata.get("industry_l1"),
        "industry_l2": metadata.get("industry_l2"),
        "report_year": metadata.get("report_year"),
        "report_type": metadata.get("report_type"),
        "doc_source_type": metadata.get("doc_source_type"),
        "report_date": metadata.get("report_date"),
        "fiscal_year": metadata.get("fiscal_year"),
        "period": metadata.get("period"),
        "unit_hint": metadata.get("unit_hint"),
        "language": metadata.get("language"),
        "topic_flags": metadata.get("topic_flags", []),
        "business_tags": metadata.get("business_tags", []),
        "strategy_tags": metadata.get("strategy_tags", []),
        "factor_tags": metadata.get("factor_tags", []),
        "chain_position_major": metadata.get("chain_position_major"),
        "chain_position_minor": metadata.get("chain_position_minor", []),
        "listing_tags": metadata.get("listing_tags", []),
        "ownership_tags": metadata.get("ownership_tags", []),
        "status_tags": metadata.get("status_tags", []),
        "style_tags": metadata.get("style_tags", []),
        "table_id": metadata.get("table_id"),
        "chart_id": metadata.get("chart_id"),
        "picture_id": metadata.get("picture_id"),
        "chart_type": metadata.get("chart_type"),
        "series_name": metadata.get("series_name"),
        "x_label": metadata.get("x_label"),
        "chart_confidence": metadata.get("chart_confidence"),
        "matched_child_chunk_ids": result.get("matched_child_chunk_ids", []),
        "matched_tags": result.get("matched_tags", []),
        "matched_queries": result.get("matched_queries", []),
        "query_hit_count": int(result.get("query_hit_count", len(result.get("matched_queries", [])))),
        "result_scope": result.get("result_scope"),
        "retrieval_sources": result.get("retrieval_sources", []),
        "score": round(result_score(result), 4),
        "final_score": round(result_score(result), 4),
        "distance_rrf": result.get("distance_rrf"),
        "colbert_score": result.get("colbert_score"),
        "final_relevance_score": result.get("final_relevance_score"),
        "text": result.get("text"),
        "text_preview": " ".join((result.get("text") or "").split())[:220],
    }


def aggregate_retrieval_results_by_report(processor, retrieval_results: List[Dict], *, result_score: Callable) -> List[Dict]:
    grouped: Dict[tuple, Dict] = {}
    for result in retrieval_results:
        metadata = result.get("metadata") or {}
        key = (
            metadata.get("sha1_name"),
            metadata.get("company_name"),
            metadata.get("stock_code") or metadata.get("security_code"),
            metadata.get("report_year"),
        )
        existing = grouped.get(key)
        serialized = processor._serialize_retrieval_result(result)
        if existing is None:
            grouped[key] = {
                "doc_id": metadata.get("sha1_name"),
                "company_name": metadata.get("company_name"),
                "stock_code": metadata.get("stock_code") or metadata.get("security_code"),
                "report_year": metadata.get("report_year"),
                "final_score": round(result_score(result), 4),
                "evidence_count": 1,
                "matched_tags": list(result.get("matched_tags", [])),
                "evidence_chunks": [serialized],
            }
            continue

        existing["final_score"] = max(existing["final_score"], round(result_score(result), 4))
        existing["evidence_count"] += 1
        existing["matched_tags"] = sorted(
            set(existing.get("matched_tags", [])) | set(result.get("matched_tags", []))
        )
        if len(existing["evidence_chunks"]) < 3:
            existing["evidence_chunks"].append(serialized)

    aggregated = list(grouped.values())
    for item in aggregated:
        item["aggregation_score"] = round(float(item["final_score"]) + 0.02 * max(0, item["evidence_count"] - 1), 4)
    aggregated.sort(key=lambda item: item["aggregation_score"], reverse=True)
    return aggregated


def validate_page_references(claimed_pages: List[int], retrieval_results: List[Dict], min_pages: int = 1, max_pages: int = 8) -> List[int]:
    if claimed_pages is None:
        claimed_pages = []

    retrieved_pages = [result['page'] for result in retrieval_results]
    validated_pages = [page for page in claimed_pages if page in retrieved_pages]

    if len(validated_pages) < len(claimed_pages):
        removed_pages = set(claimed_pages) - set(validated_pages)
        print(f"Warning: Removed {len(removed_pages)} hallucinated page references: {removed_pages}")

    if len(validated_pages) < min_pages and retrieval_results:
        existing_pages = set(validated_pages)
        for result in retrieval_results:
            page = result['page']
            if page not in existing_pages:
                validated_pages.append(page)
                existing_pages.add(page)
            if len(validated_pages) >= min_pages:
                break

    if len(validated_pages) > max_pages:
        print(f"Trimming references from {len(validated_pages)} to {max_pages} pages")
        validated_pages = validated_pages[:max_pages]

    return validated_pages


def confidence_from_individual_answers(individual_answers: Dict[str, Dict]) -> str:
    levels = [answer.get("confidence", "low") for answer in individual_answers.values()]
    if levels and all(level == "high" for level in levels):
        return "high"
    if any(level in {"high", "medium"} for level in levels):
        return "medium"
    return "low"
