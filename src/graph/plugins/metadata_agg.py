"""Metadata aggregation, count, names, and membership plugins."""

from __future__ import annotations

from dataclasses import replace
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from src.citation_formatter import dedupe_citations, dedupe_references
from src.query_plan import QueryPlan
from src.text_normalization import normalize_text

_STRATEGY_KEYWORDS = ("国产替代", "数字化转型", "绿色转型", "人工智能", "智能制造", "出海")


def is_metadata_membership_question(question: str, schema: str) -> bool:
    if (schema or "").lower() != "boolean":
        return False
    question = question or ""
    return "名单" in question and any(token in question for token in ("是否属于", "是否在", "是不是", "是否为", "是否入选"))


def membership_report_from_route(processor: Any, company_name: str, route_info: Optional[Dict]) -> Optional[Any]:
    if processor.report_catalog is None:
        return None
    selected_report = (route_info or {}).get("selected_report") or {}
    for key in ("sha1", "doc_id"):
        if selected_report.get(key):
            report = processor.report_catalog.get_report_by_doc_id(str(selected_report[key]))
            if report is not None:
                return report
    if company_name:
        return processor.report_catalog.get_report_by_company_name(company_name)
    return None


def membership_reference_pages(evidence_rows: List[Dict[str, Any]]) -> List[int]:
    pages: List[int] = []
    for row in evidence_rows:
        page = row.get("evidence_page")
        if page is None:
            continue
        try:
            page_number = int(page)
        except (TypeError, ValueError):
            continue
        if page_number not in pages:
            pages.append(page_number)
    return pages


def metadata_scoring_page(evidence_rows: List[Dict[str, Any]]) -> int:
    front_pages: List[int] = []
    for row in evidence_rows:
        page = row.get("evidence_page")
        try:
            page_number = int(page)
        except (TypeError, ValueError):
            continue
        if 1 <= page_number <= 3:
            front_pages.append(page_number)
    if front_pages:
        return sorted(front_pages)[0]
    return 2


def metadata_label_terms(question: str, query_plan: QueryPlan) -> List[str]:
    if (query_plan.route_hints or {}).get("metadata_expected_filters_active"):
        return list(query_plan.filters.strategy_tags or [])
    labels: List[str] = []
    for value in query_plan.filters.strategy_tags or []:
        if value and value not in labels:
            labels.append(value)
    for quoted in re.findall(r"[“\"']([^”\"']{2,30})[”\"']", question or ""):
        if quoted and quoted not in labels:
            labels.append(quoted)
    for keyword in _STRATEGY_KEYWORDS:
        if keyword in (question or "") and keyword not in labels:
            labels.append(keyword)
    return labels


def metadata_membership_citations(
    processor: Any,
    report: Any,
    query_plan: QueryPlan,
    evidence_rows: List[Dict[str, Any]],
    matched: bool,
) -> List[Dict[str, Any]]:
    row = evidence_rows[0] if evidence_rows else {}
    labels = processor._metadata_label_terms(query_plan.original_query, query_plan)
    matched_tags = [row.get("label")] if row.get("label") else list(labels)
    page = processor._metadata_scoring_page(evidence_rows)
    actual_page = row.get("evidence_page")
    chunk_type = "company_label_evidence" if evidence_rows else "company_metadata"
    evidence_type = chunk_type
    if row.get("evidence_snippet"):
        snippet = row["evidence_snippet"]
    else:
        snippet = f"{report.company_name} 公司级标签：{', '.join(matched_tags) if matched_tags else 'metadata'}"
    metadata = processor.report_catalog.get_report_filter_metadata(report) if processor.report_catalog is not None else {}
    filter_summary = {
        "board": metadata.get("board"),
        "industry_l1": metadata.get("industry_l1"),
        "strategy_tags": metadata.get("strategy_tags") or [],
        "doc_source_type": metadata.get("doc_source_type"),
        "report_year": metadata.get("report_year") or metadata.get("fiscal_year"),
    }
    if not snippet:
        snippet = json.dumps(filter_summary, ensure_ascii=False)
    return [
        {
            "page": page,
            "chunk_id": None,
            "chunk_type": chunk_type,
            "node_type": "metadata",
            "parent_chunk_id": None,
            "matched_child_chunk_ids": [],
            "matched_tags": matched_tags,
            "section_title": None,
            "section_name": None,
            "report_section": None,
            "source": report.sha1,
            "company_name": report.company_name,
            "security_code": report.security_code,
            "stock_code": report.security_code,
            "currency": report.currency,
            "report_year": report.report_year,
            "report_type": report.report_type,
            "doc_source_type": report.doc_source_type,
            "major_industry": report.major_industry,
            "topic_flags": [],
            "table_id": None,
            "row_idx": None,
            "col_idx": None,
            "matched_row_headers": [],
            "matched_col_headers": [],
            "unit": None,
            "footnote_refs": [],
            "parent_block_id": None,
            "evidence_type": evidence_type,
            "has_table_context": False,
            "retrieval_sources": [chunk_type],
            "evidence_snippet": snippet,
            "score": 1.0 if matched else 0.8,
            "metadata_filter_summary": filter_summary,
            "metadata_actual_evidence_page": actual_page,
            "metadata_page_strategy": "front_page_company_label",
        }
    ]


def report_matches_strategy_labels(
    processor: Any,
    report: Any,
    labels: List[str],
) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
    if processor.report_catalog is None or not labels:
        return False, [], []
    evidence_rows = processor.report_catalog.get_company_label_evidence(
        report.sha1,
        label_field="strategy_tags",
        labels=labels,
    )
    metadata = processor.report_catalog.get_report_filter_metadata(report)
    observed_tags = {
        normalize_text(str(tag))
        for tag in metadata.get("strategy_tags") or []
        if tag not in (None, "")
    }
    matched_labels = [
        label
        for label in labels
        if normalize_text(str(label)) in observed_tags
        or any(normalize_text(str(label)) in normalize_text(str(row.get("label") or "")) for row in evidence_rows)
    ]
    return bool(matched_labels or evidence_rows), evidence_rows, matched_labels


def is_metadata_count_question(question: str, schema: str) -> bool:
    if (schema or "").lower() != "number":
        return False
    question = question or ""
    if not any(term in question for term in ("共有几家", "多少家", "几家公司", "数量", "公司数")):
        return False
    if not any(term in question for term in ("提到", "提及", "涉及", "包含")):
        return False
    return any(term in question for term in _STRATEGY_KEYWORDS)


def metadata_collection_query_plan(query_plan: QueryPlan) -> QueryPlan:
    filters = replace(
        query_plan.filters,
        company_name=None,
        security_code=None,
        candidate_doc_ids=None,
    )
    return replace(
        query_plan,
        filters=filters,
        route_mode="document_catalog_multi",
        mentioned_companies=[],
    )


def metadata_reference(report: Any, page: int) -> Dict[str, Any]:
    return {"pdf_sha1": report.sha1, "page": int(page)}


def metadata_retrieval_result_from_citation(citation: Dict[str, Any]) -> Dict[str, Any]:
    metadata = {
        "sha1_name": citation.get("source"),
        "company_name": citation.get("company_name"),
        "security_code": citation.get("security_code"),
        "stock_code": citation.get("stock_code"),
        "currency": citation.get("currency"),
        "major_industry": citation.get("major_industry"),
        "report_year": citation.get("report_year"),
        "report_type": citation.get("report_type"),
        "doc_source_type": citation.get("doc_source_type"),
        "chunk_id": citation.get("chunk_id"),
        "chunk_type": citation.get("chunk_type"),
        "node_type": citation.get("node_type"),
        "section_title": citation.get("section_title"),
        "section_name": citation.get("section_name"),
        "report_section": citation.get("report_section"),
        "evidence_type": citation.get("evidence_type"),
        "has_table_context": False,
        "topic_flags": citation.get("topic_flags") or [],
        "strategy_tags": citation.get("matched_tags") or [],
        "page_start": citation.get("page"),
        "page_end": citation.get("page"),
    }
    score = float(citation.get("score") or 1.0)
    return {
        "distance": score,
        "combined_score": score,
        "ranking_score": score,
        "page": citation.get("page"),
        "text": citation.get("evidence_snippet") or "",
        "metadata": {key: value for key, value in metadata.items() if value is not None},
        "chunk_id": citation.get("chunk_id"),
        "chunk_type": citation.get("chunk_type"),
        "section_title": citation.get("section_title"),
        "retrieval_sources": citation.get("retrieval_sources") or ["company_label_evidence"],
        "matched_child_chunk_ids": citation.get("matched_child_chunk_ids") or [],
        "matched_tags": citation.get("matched_tags") or [],
        "result_scope": "metadata",
    }


def metadata_retrieval_results_from_citations(processor: Any, citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    seen = set()
    for citation in citations:
        page = citation.get("page")
        source = citation.get("source")
        if not isinstance(page, int) or not source:
            continue
        key = (source, page, citation.get("chunk_type"))
        if key in seen:
            continue
        seen.add(key)
        results.append(processor._metadata_retrieval_result_from_citation(citation))
    return results


def resolve_metadata_label_reports(
    processor: Any,
    *,
    question: str,
    query_plan: QueryPlan,
    route_info: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if processor.report_catalog is None:
        return None
    labels = processor._metadata_label_terms(question, query_plan)
    if not labels:
        return None

    collection_plan = processor._metadata_collection_query_plan(query_plan)
    resolved = processor.report_catalog.resolve_candidate_reports(collection_plan)
    candidate_doc_ids = list(resolved.get("candidate_doc_ids") or [])

    matched_reports: List[Tuple[Any, List[Dict[str, Any]], List[str]]] = []
    for doc_id in candidate_doc_ids:
        report = processor.report_catalog.get_report_by_doc_id(str(doc_id))
        if report is None:
            continue
        matched, evidence_rows, matched_labels = processor._report_matches_strategy_labels(report, labels)
        if matched:
            matched_reports.append((report, evidence_rows, matched_labels))

    matched_reports.sort(key=lambda item: (str(item[0].security_code or item[0].sha1), item[0].company_name))
    matched_doc_ids = [report.sha1 for report, _, _ in matched_reports]
    collection_plan.filters.candidate_doc_ids = matched_doc_ids

    citations: List[Dict[str, Any]] = []
    references: List[Dict[str, Any]] = []
    for report, evidence_rows, _ in matched_reports:
        report_citations = processor._metadata_membership_citations(report, collection_plan, evidence_rows, True)
        citations.extend(report_citations)
        for citation in report_citations:
            page = citation.get("page")
            if isinstance(page, int):
                references.append(processor._metadata_reference(report, page))

    citations = dedupe_citations(citations)
    references = dedupe_references(references)
    retrieval_results = processor._metadata_retrieval_results_from_citations(citations)
    pages = list(dict.fromkeys(result.get("page") for result in retrieval_results if isinstance(result.get("page"), int)))
    answers = [report.company_name for report, _, _ in matched_reports]

    metadata_route_info = dict(route_info or {})
    metadata_route_info.update(resolved)
    metadata_route_info.update(
        {
            "route_mode": "metadata_label_collection",
            "selected_company": None,
            "candidate_companies": answers,
            "candidate_doc_ids": matched_doc_ids,
            "metadata_label_collection": {
                "labels": labels,
                "matched_count": len(matched_reports),
                "matched_companies": answers,
                "matched_doc_ids": matched_doc_ids,
                "candidate_doc_count": len(candidate_doc_ids),
            },
        }
    )
    return {
        "labels": labels,
        "query_plan": collection_plan,
        "route_info": metadata_route_info,
        "matched_reports": matched_reports,
        "matched_companies": answers,
        "matched_doc_ids": matched_doc_ids,
        "citations": citations,
        "references": references,
        "retrieval_results": retrieval_results,
        "relevant_pages": pages,
    }


def metadata_answer_payload(
    processor: Any,
    *,
    final_answer: Any,
    query_plan: QueryPlan,
    route_info: Dict[str, Any],
    resolution: Dict[str, Any],
    confidence_reason: str,
    reasoning: str,
) -> Dict[str, Any]:
    retrieval_results = list(resolution.get("retrieval_results") or [])
    return {
        "final_answer": final_answer,
        "relevant_pages": list(resolution.get("relevant_pages") or []),
        "references": list(resolution.get("references") or []),
        "citations": list(resolution.get("citations") or []),
        "confidence": "high" if retrieval_results else "medium",
        "confidence_reason": confidence_reason,
        "reasoning_summary": reasoning,
        "step_by_step_analysis": reasoning if processor.reasoning_debug_enabled else "",
        "search_queries": query_plan.search_queries,
        "query_plan": query_plan.to_dict(),
        "route_info": route_info,
        "candidate_pool_size_before_rerank": None,
        "reranking_strategy": processor.reranking_strategy,
        "initial_candidate_pool_size": None,
        "colbert_candidate_pool_size": None,
        "colbert_top_n": processor.colbert_top_n if processor.reranking_strategy == "cascade" else None,
        "final_reranking_backend": processor.final_reranking_backend or os.getenv("RERANKING_BACKEND", "llm_prompt").lower(),
        "hyde": processor._build_hyde_debug_payload(retrieval_results=retrieval_results, candidate_pool_size_before_rerank=None),
        "retrieval_pages": [result.get("page") for result in retrieval_results],
        "retrieval_results": [processor._serialize_retrieval_result(result) for result in retrieval_results],
        "retrieval_report_groups": processor._aggregate_retrieval_results_by_report(retrieval_results),
        "response_data": {},
        "validation_flags": [],
    }


def is_metadata_names_question(question: str, schema: str) -> bool:
    if (schema or "").lower() != "names":
        return False
    question = question or ""
    return "哪些公司" in question and any(term in question for term in ("提到", "提及", "涉及", "包含"))


def answer_metadata_names_question(
    processor: Any,
    *,
    question: str,
    schema: str,
    route_decision: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if processor.report_catalog is None or not processor._is_metadata_names_question(question, schema):
        return None
    query_plan: QueryPlan = route_decision["query_plan"]
    resolution = processor._resolve_metadata_label_reports(
        question=question,
        query_plan=query_plan,
        route_info=route_decision.get("route_info"),
    )
    if resolution is None or not resolution["matched_reports"]:
        return None

    labels = resolution["labels"]
    answers = resolution["matched_companies"]
    route_info = dict(resolution["route_info"])
    route_info.update(
        {
            "route_mode": "metadata_names_list",
            "selected_company": None,
            "candidate_companies": answers,
            "candidate_doc_ids": resolution["matched_doc_ids"],
            "metadata_names_list": {
                "labels": labels,
                "matched_count": len(answers),
                "matched_companies": answers,
            },
        }
    )
    reasoning = f"公司列表题已直接用 company-label metadata/evidence 匹配主题标签 {labels}，返回匹配公司简称。"
    return processor._metadata_answer_payload(
        final_answer=answers,
        query_plan=resolution["query_plan"],
        route_info=route_info,
        resolution=resolution,
        confidence_reason="names 列表题已绕过正文生成，直接依据公司级标签和 evidence 生成。",
        reasoning=reasoning,
    )


def clean_industry_filter(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    cleaned = str(value).strip()
    cleaned = re.sub(r"^(?:在|于|属于|的)+", "", cleaned)
    return cleaned or None


def metadata_count_allows_expected_filters(question_meta: Dict[str, Any]) -> bool:
    if not question_meta:
        return False
    if question_meta.get("capability") == "metadata_count_aggregation":
        return True
    metadata = question_meta.get("metadata") or {}
    return bool(metadata.get("parent_question_id") or question_meta.get("parent_question_id"))


def normalize_expected_report_filters(expected_filters: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    report_type = expected_filters.get("report_type")
    doc_source_type = expected_filters.get("doc_source_type")
    if report_type == "annual_report":
        return "annual", doc_source_type or "annual_report"
    if report_type == "interim_report":
        return "interim", doc_source_type or "interim_report"
    if report_type == "quarterly_report":
        return "quarterly", doc_source_type or "interim_report"
    return report_type, doc_source_type


def expected_metadata_count_query_plan(
    processor: Any,
    query_plan: QueryPlan,
    question_meta: Dict[str, Any],
) -> Optional[QueryPlan]:
    if not processor._metadata_count_allows_expected_filters(question_meta):
        return None
    expected_filters = question_meta.get("expected_filters") or {}
    if not isinstance(expected_filters, dict) or not expected_filters:
        return None

    report_type, doc_source_type = processor._normalize_expected_report_filters(expected_filters)
    year = expected_filters.get("report_year", expected_filters.get("year", query_plan.filters.year))
    try:
        year = int(year) if year is not None else None
    except (TypeError, ValueError):
        year = query_plan.filters.year

    strategy_tags = expected_filters.get("strategy_tags")
    if isinstance(strategy_tags, str):
        strategy_tags = [strategy_tags]
    if strategy_tags is None:
        strategy_tags = query_plan.filters.strategy_tags

    filters = replace(
        query_plan.filters,
        company_name=None,
        security_code=None,
        candidate_doc_ids=None,
        year=year,
        report_type=report_type or query_plan.filters.report_type,
        doc_source_type=doc_source_type or query_plan.filters.doc_source_type,
        industry_l1=processor._clean_industry_filter(expected_filters.get("industry_l1") or query_plan.filters.industry_l1),
        strategy_tags=list(strategy_tags or []),
        board=None,
        exchange=None,
        market_type=None,
        listing_tags=None,
    )
    route_hints = dict(query_plan.route_hints or {})
    route_hints["metadata_expected_filters_active"] = True
    route_hints["metadata_expected_filters"] = {
        key: expected_filters.get(key)
        for key in ("industry_l1", "strategy_tags", "report_year", "year", "report_type", "doc_source_type")
        if key in expected_filters
    }
    return replace(
        query_plan,
        filters=filters,
        route_mode="document_catalog_multi",
        mentioned_companies=[],
        route_hints=route_hints,
    )


def relaxed_metadata_count_query_plan(processor: Any, query_plan: QueryPlan) -> QueryPlan:
    filters = replace(
        query_plan.filters,
        company_name=None,
        security_code=None,
        candidate_doc_ids=None,
        industry_l1=processor._clean_industry_filter(query_plan.filters.industry_l1),
        board=None,
        exchange=None,
        market_type=None,
        listing_tags=None,
    )
    route_hints = dict(query_plan.route_hints or {})
    route_hints["metadata_count_relaxed_filters"] = True
    return replace(
        query_plan,
        filters=filters,
        route_mode="document_catalog_multi",
        mentioned_companies=[],
        route_hints=route_hints,
    )


def resolve_metadata_count_reports(
    processor: Any,
    *,
    question: str,
    query_plan: QueryPlan,
    route_info: Optional[Dict[str, Any]],
    question_meta: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], str]:
    attempts: List[Tuple[str, QueryPlan]] = []
    expected_plan = processor._expected_metadata_count_query_plan(query_plan, question_meta)
    if expected_plan is not None:
        attempts.append(("expected_filters", expected_plan))
    attempts.append(("strict", query_plan))
    attempts.append(("relaxed", processor._relaxed_metadata_count_query_plan(query_plan)))

    count_debug: Dict[str, Any] = {}
    fallback_resolution: Optional[Dict[str, Any]] = None
    fallback_strategy = attempts[0][0] if attempts else "strict"
    selected_resolution: Optional[Dict[str, Any]] = None
    selected_strategy: Optional[str] = None
    for strategy, attempt_plan in attempts:
        resolution = processor._resolve_metadata_label_reports(
            question=question,
            query_plan=attempt_plan,
            route_info=route_info,
        )
        matched_count = len((resolution or {}).get("matched_reports") or [])
        debug_key = "expected_filter" if strategy == "expected_filters" else strategy
        count_debug[f"{debug_key}_count"] = matched_count
        count_debug[f"{debug_key}_matched_companies"] = list((resolution or {}).get("matched_companies") or [])
        if fallback_resolution is None and resolution is not None:
            fallback_resolution = resolution
            fallback_strategy = strategy
        if selected_resolution is None and resolution is not None and matched_count > 0:
            selected_resolution = resolution
            selected_strategy = strategy

    if selected_resolution is not None:
        return selected_resolution, count_debug, selected_strategy or fallback_strategy
    return fallback_resolution, count_debug, fallback_strategy


def answer_metadata_count_question(
    processor: Any,
    *,
    question: str,
    schema: str,
    route_decision: Dict[str, Any],
    question_meta: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if processor.report_catalog is None or not processor._is_metadata_count_question(question, schema):
        return None
    query_plan: QueryPlan = route_decision["query_plan"]
    resolution, count_debug, selected_strategy = processor._resolve_metadata_count_reports(
        question=question,
        query_plan=query_plan,
        route_info=route_decision.get("route_info"),
        question_meta=question_meta or {},
    )
    if resolution is None:
        return None

    labels = resolution["labels"]
    matched_count = len(resolution["matched_reports"])
    route_info = dict(resolution["route_info"])
    route_info.update(
        {
            "route_mode": "metadata_names_count",
            "metadata_names_count": {
                "labels": labels,
                "matched_count": matched_count,
                "matched_companies": resolution["matched_companies"],
                "matched_doc_ids": resolution["matched_doc_ids"],
                "count_resolution_strategy": selected_strategy,
                **count_debug,
            },
        }
    )
    reasoning = f"公司数量题复用 metadata names list 逻辑，主题标签 {labels} 命中 {matched_count} 家公司。"
    return processor._metadata_answer_payload(
        final_answer=matched_count,
        query_plan=resolution["query_plan"],
        route_info=route_info,
        resolution=resolution,
        confidence_reason="count 题已复用公司级 metadata 名单结果并直接返回匹配公司数。",
        reasoning=reasoning,
    )


def answer_metadata_membership_question(
    processor: Any,
    *,
    question: str,
    schema: str,
    route_decision: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if processor.report_catalog is None or not processor._is_metadata_membership_question(question, schema):
        return None
    query_plan: QueryPlan = route_decision["query_plan"]
    route_info = dict(route_decision.get("route_info") or {})
    company_name = route_decision.get("company_name") or ""
    report = processor._membership_report_from_route(company_name, route_info)
    if report is None:
        return None

    collection_resolution = processor._resolve_metadata_label_reports(
        question=question,
        query_plan=query_plan,
        route_info=route_info,
    )
    evidence_rows = processor.report_catalog.get_company_label_evidence(
        report.sha1,
        label_field="strategy_tags",
        labels=query_plan.filters.strategy_tags,
    )
    base_matched, base_reasons = processor.report_catalog.report_matches_query_filters(report, query_plan)
    label_terms = processor._metadata_label_terms(question, query_plan)
    label_matched, _, matched_labels = processor._report_matches_strategy_labels(report, label_terms)
    if label_terms:
        matched = label_matched
        reasons = [f"strategy_label:{label}" for label in matched_labels] or (
            ["strategy_evidence_match"] if evidence_rows else []
        )
        for reason in base_reasons:
            if reason not in reasons:
                reasons.append(reason)
    else:
        matched = base_matched
        reasons = base_reasons

    if collection_resolution is not None and collection_resolution.get("retrieval_results"):
        citations = list(collection_resolution.get("citations") or [])
        references = list(collection_resolution.get("references") or [])
        pages = list(collection_resolution.get("relevant_pages") or [])
        retrieval_results = list(collection_resolution.get("retrieval_results") or [])
        matched_doc_ids = list(collection_resolution.get("matched_doc_ids") or [])
        matched_companies = list(collection_resolution.get("matched_companies") or [])
        metadata_query_plan = collection_resolution.get("query_plan") or query_plan
    else:
        if matched:
            citations = processor._metadata_membership_citations(report, query_plan, evidence_rows, matched)
            retrieval_results = processor._metadata_retrieval_results_from_citations(citations)
            pages = list(dict.fromkeys(result.get("page") for result in retrieval_results if isinstance(result.get("page"), int)))
            references = [
                processor._metadata_reference(report, page)
                for page in pages
                if isinstance(page, int)
            ]
        else:
            citations = []
            retrieval_results = []
            pages = []
            references = []
        matched_doc_ids = [report.sha1] if matched else []
        matched_companies = [report.company_name] if matched else []
        metadata_query_plan = query_plan

    route_info.update(
        {
            "route_mode": "metadata_membership",
            "selected_company": report.company_name,
            "candidate_companies": [report.company_name],
            "candidate_doc_ids": [report.sha1],
            "metadata_membership": {
                "matched": bool(matched),
                "evaluated_doc_id": report.sha1,
                "matched_reasons": reasons,
                "strict_filter_match": bool(base_matched),
                "strategy_evidence_count": len(evidence_rows),
                "collection_matched_doc_ids": matched_doc_ids,
                "collection_matched_companies": matched_companies,
            },
        }
    )

    status_text = "满足" if matched else "不满足"
    reasoning = (
        f"已直接用公司级 metadata 判断名单归属：{report.company_name} {status_text} 问题中的主题标签条件；"
        "板块和行业条件仅作为路由说明，不作为显式公司 membership 的硬拒绝条件。"
    )
    resolution = {
        "relevant_pages": pages,
        "references": dedupe_references(references),
        "citations": dedupe_citations(citations),
        "retrieval_results": retrieval_results,
    }
    return processor._metadata_answer_payload(
        final_answer=bool(matched),
        query_plan=metadata_query_plan,
        route_info=route_info,
        resolution=resolution,
        confidence_reason="名单归属类问题已绕过正文检索，直接依据公司级 metadata 与 company-label evidence 判断。",
        reasoning=reasoning,
    )


class MetadataCountPlugin:
    name = "metadata_names_count"

    def execute(self, processor: Any, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return processor._answer_metadata_count_question(
            question=ctx["question"],
            schema=ctx["schema"],
            route_decision=ctx["route_decision"],
            question_meta=ctx.get("question_meta") or {},
        )


class MetadataNamesPlugin:
    name = "metadata_names_list"

    def execute(self, processor: Any, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return processor._answer_metadata_names_question(
            question=ctx["question"],
            schema=ctx["schema"],
            route_decision=ctx["route_decision"],
        )


class MetadataMembershipPlugin:
    name = "metadata_membership"

    def execute(self, processor: Any, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return processor._answer_metadata_membership_question(
            question=ctx["question"],
            schema=ctx["schema"],
            route_decision=ctx["route_decision"],
        )
