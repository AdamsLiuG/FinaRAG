"""Document loading and page, table, chart, and numeric grounding for QA."""

from collections import defaultdict
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.citation_formatter import dedupe_citations
from src.query_plan import QueryPlan
from src.text_normalization import normalize_text, parse_numeric_value


def load_document_payload(processor, doc_id: Optional[str]) -> Optional[Dict[str, Any]]:
    doc_id = str(doc_id or "").strip()
    if not doc_id:
        return None
    # 整个 load 持锁（double-checked）：分离的读锁/写锁存在 check-then-act
    # 竞态——并发 cold miss 时多线程各自加载产生不同 payload 对象
    # （并发一致性测试抓到）。锁内 IO 串行化冷加载可接受（一次性成本），
    # 热路径缓存命中快速返回。
    with processor._document_cache_lock:
        cached = processor._document_cache.get(doc_id)
        if cached is not None:
            return cached

        try:
            from src.document_store import get_sqlite_doc_repository

            sqlite_repo = get_sqlite_doc_repository(processor.documents_dir)
            if sqlite_repo is not None:
                payload = sqlite_repo.get_document_payload(doc_id)
                if payload is not None:
                    processor._document_cache[doc_id] = payload
                    return payload
        except Exception:
            pass

        candidate_paths = []
        if doc_id.endswith(".json"):
            candidate_paths.append(processor.documents_dir / doc_id)
        else:
            candidate_paths.append(processor.documents_dir / f"{doc_id}.json")
            candidate_paths.append(processor.documents_dir / doc_id)

        for path in candidate_paths:
            if not path.exists() or not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            processor._document_cache[doc_id] = payload
            return payload

        for path in processor.documents_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            metainfo = payload.get("metainfo") or {}
            observed_doc_id = metainfo.get("sha1_name") or metainfo.get("doc_id") or path.stem
            if str(observed_doc_id) == doc_id:
                processor._document_cache[doc_id] = payload
                return payload
        return None


def selected_doc_id(
    processor,
    company_name: str,
    route_info: Optional[Dict[str, Any]],
    query_plan: Optional[QueryPlan] = None,
) -> Optional[str]:
    selected_report = (route_info or {}).get("selected_report") or {}
    if isinstance(selected_report, dict):
        for key in ("sha1", "doc_id"):
            if selected_report.get(key):
                return str(selected_report[key])

    candidate_doc_ids = list((route_info or {}).get("candidate_doc_ids") or [])
    if not candidate_doc_ids and query_plan is not None:
        candidate_doc_ids = list(query_plan.filters.candidate_doc_ids or [])
    if len(candidate_doc_ids) == 1:
        return str(candidate_doc_ids[0])

    if processor.report_catalog is not None and company_name:
        report = processor.report_catalog.get_report_by_company_name(company_name)
        if report is not None:
            return report.sha1
    return None


def document_pages(document: Dict[str, Any]) -> List[Tuple[int, str]]:
    content = document.get("content") or {}
    pages: List[Tuple[int, str]] = []
    for page in content.get("pages") or []:
        try:
            page_number = int(page.get("page"))
        except (TypeError, ValueError):
            continue
        text = str(page.get("text") or "")
        if text:
            pages.append((page_number, text))
    if pages:
        return pages

    grouped: Dict[int, List[str]] = defaultdict(list)
    for chunk in content.get("chunks") or []:
        try:
            page_number = int(chunk.get("page") or chunk.get("page_start"))
        except (TypeError, ValueError):
            continue
        text = str(chunk.get("text") or chunk.get("search_text") or "")
        if text:
            grouped[page_number].append(text)
    return [(page, "\n".join(texts)) for page, texts in sorted(grouped.items())]


def document_page_result(
    *,
    document: Dict[str, Any],
    page: int,
    text: str,
    chunk_type: str,
    source: str,
    score: float = 1.0,
    build_result_metadata: Callable,
) -> Dict[str, Any]:
    metainfo = dict(document.get("metainfo") or {})
    doc_id = str(metainfo.get("sha1_name") or metainfo.get("doc_id") or "")
    if doc_id:
        metainfo["sha1_name"] = doc_id
    chunk = {
        "chunk_id": f"{doc_id or 'document'}:{chunk_type}:{page}",
        "chunk_type": chunk_type,
        "type": chunk_type,
        "node_type": "page",
        "page": page,
        "page_start": page,
        "page_end": page,
        "text": text,
        "search_text": text,
        "evidence_type": chunk_type,
    }
    metadata = build_result_metadata(metainfo, chunk)
    if doc_id and not metadata.get("sha1_name"):
        metadata["sha1_name"] = doc_id
    return {
        "distance": score,
        "combined_score": score,
        "ranking_score": score,
        "page": page,
        "text": text,
        "metadata": metadata,
        "chunk_id": chunk["chunk_id"],
        "chunk_type": chunk_type,
        "section_title": metadata.get("section_title"),
        "retrieval_sources": [source],
        "matched_child_chunk_ids": [],
        "matched_tags": [],
        "result_scope": "page",
    }


def grounded_number_value(grounding_result: Dict[str, Any]) -> Any:
    value = grounding_result.get("answer_value")
    if value is None:
        value = grounding_result.get("normalized_value")
    target_unit = grounding_result.get("target_unit")
    decimal_places = {
        "元": 2,
        "万元": 4,
        "亿元": 6,
        "百万元": 4,
    }.get(target_unit)
    if decimal_places is not None and isinstance(value, (int, float)):
        try:
            quant = Decimal("1") if decimal_places == 0 else Decimal("1").scaleb(-decimal_places)
            value = float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))
        except (InvalidOperation, ValueError):
            pass
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def table_grounding_retrieval_result(
    processor,
    grounding_result: Dict[str, Any],
    *,
    chunk_type: str = "serialized_table",
    retrieval_source: str = "table_grounding",
) -> Dict[str, Any]:
    doc_id = str(grounding_result.get("source_doc_id") or "")
    document = processor.table_grounder._load_document(doc_id) if processor.table_grounder is not None and doc_id else None
    metainfo = (document or {}).get("metainfo") or {}
    metadata = {
        "sha1_name": doc_id,
        "company_name": metainfo.get("company_name"),
        "company_aliases": list(metainfo.get("company_aliases") or []),
        "security_code": metainfo.get("security_code"),
        "stock_code": metainfo.get("stock_code") or metainfo.get("security_code"),
        "currency": metainfo.get("currency"),
        "major_industry": metainfo.get("major_industry"),
        "report_year": metainfo.get("report_year") or metainfo.get("fiscal_year"),
        "report_type": metainfo.get("report_type"),
        "doc_source_type": metainfo.get("doc_source_type"),
        "period": grounding_result.get("period"),
        "unit_hint": grounding_result.get("unit"),
        "language": metainfo.get("language"),
        "chunk_id": None,
        "chunk_type": chunk_type,
        "section_title": None,
        "section_name": None,
        "report_section": None,
        "table_id": grounding_result.get("table_id"),
        "logical_table_id": grounding_result.get("logical_table_id"),
        "logical_role": None,
        "page_span": list(grounding_result.get("page_span") or []),
        "merge_confidence": None,
        "merge_state": "confirmed" if grounding_result.get("logical_table_id") else None,
        "row_idx": grounding_result.get("row_idx"),
        "col_idx": grounding_result.get("col_idx"),
        "matched_row_headers": grounding_result.get("matched_row_headers") or [],
        "matched_col_headers": grounding_result.get("matched_col_headers") or [],
        "parent_block_id": None,
        "parent_chunk_id": None,
        "child_chunk_ids": [],
        "node_type": "table",
        "evidence_type": "table",
        "has_table_context": True,
        "page_start": grounding_result.get("page"),
        "page_end": grounding_result.get("page"),
        "topic_flags": [],
    }
    return {
        "distance": 1.0,
        "combined_score": 1.0,
        "ranking_score": 1.0,
        "page": grounding_result.get("page"),
        "text": grounding_result.get("table_snippet") or "",
        "metadata": metadata,
        "chunk_id": None,
        "chunk_type": chunk_type,
        "section_title": None,
        "table_id": grounding_result.get("table_id"),
        "retrieval_sources": [retrieval_source],
        "matched_child_chunk_ids": [],
        "matched_tags": [],
        "result_scope": "table",
    }


def ensure_table_grounding_retrieval_result(
    processor,
    retrieval_results: List[Dict],
    grounding_result: Optional[Dict[str, Any]],
) -> List[Dict]:
    if not grounding_result or grounding_result.get("page") is None:
        return retrieval_results
    doc_id = str(grounding_result.get("source_doc_id") or "")
    page = grounding_result.get("page")
    for result in retrieval_results:
        metadata = result.get("metadata") or {}
        if str(metadata.get("sha1_name") or "") == doc_id and result.get("page") == page:
            return retrieval_results
    return [processor._table_grounding_retrieval_result(grounding_result)] + list(retrieval_results)


def ensure_table_support_retrieval_results(
    processor,
    retrieval_results: List[Dict],
    grounding_result: Optional[Dict[str, Any]],
) -> List[Dict]:
    if not grounding_result:
        return retrieval_results
    results = list(retrieval_results)
    seen = {
        (str((result.get("metadata") or {}).get("sha1_name") or ""), result.get("page"))
        for result in results
    }
    for support_result in processor._table_support_results(grounding_result):
        doc_id = str(support_result.get("source_doc_id") or "")
        page = support_result.get("page")
        if not doc_id or page is None or (doc_id, page) in seen:
            continue
        results.append(
            processor._table_grounding_retrieval_result(
                support_result,
                chunk_type="table_support",
                retrieval_source="table_support",
            )
        )
        seen.add((doc_id, page))
    return results


def table_support_results(grounding_result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not grounding_result:
        return []
    return list(grounding_result.get("supporting_matches") or []) + list(
        grounding_result.get("logical_context_matches") or []
    )


def chart_grounding_retrieval_result(
    processor,
    grounding_result: Dict[str, Any],
    *,
    chunk_type: str = "chart_to_table",
    retrieval_source: str = "chart_grounding",
) -> Dict[str, Any]:
    doc_id = str(grounding_result.get("source_doc_id") or "")
    document = processor.chart_grounder._load_document(doc_id) if processor.chart_grounder is not None and doc_id else None
    metainfo = (document or {}).get("metainfo") or {}
    metadata = {
        "sha1_name": doc_id,
        "company_name": metainfo.get("company_name"),
        "company_aliases": list(metainfo.get("company_aliases") or []),
        "security_code": metainfo.get("security_code"),
        "stock_code": metainfo.get("stock_code") or metainfo.get("security_code"),
        "currency": metainfo.get("currency"),
        "major_industry": metainfo.get("major_industry"),
        "report_year": metainfo.get("report_year") or metainfo.get("fiscal_year"),
        "report_type": metainfo.get("report_type"),
        "doc_source_type": metainfo.get("doc_source_type"),
        "period": grounding_result.get("period"),
        "unit_hint": grounding_result.get("unit"),
        "language": metainfo.get("language"),
        "chunk_id": None,
        "chunk_type": chunk_type,
        "section_title": None,
        "section_name": None,
        "report_section": None,
        "table_id": None,
        "chart_id": grounding_result.get("chart_id"),
        "picture_id": grounding_result.get("picture_id"),
        "series_name": grounding_result.get("series_name"),
        "x_label": grounding_result.get("x_label"),
        "chart_confidence": grounding_result.get("chart_confidence") or grounding_result.get("confidence"),
        "parent_block_id": None,
        "parent_chunk_id": None,
        "child_chunk_ids": [],
        "node_type": "chart",
        "evidence_type": "chart",
        "has_table_context": False,
        "has_chart_context": True,
        "page_start": grounding_result.get("page"),
        "page_end": grounding_result.get("page"),
        "topic_flags": [],
    }
    return {
        "distance": 1.0,
        "combined_score": 1.0,
        "ranking_score": 1.0,
        "page": grounding_result.get("page"),
        "text": grounding_result.get("chart_context") or "",
        "metadata": metadata,
        "chunk_id": None,
        "chunk_type": chunk_type,
        "section_title": None,
        "chart_id": grounding_result.get("chart_id"),
        "retrieval_sources": [retrieval_source],
        "matched_child_chunk_ids": [],
        "matched_tags": [],
        "result_scope": "chart",
    }


def ensure_chart_grounding_retrieval_result(
    processor,
    retrieval_results: List[Dict],
    grounding_result: Optional[Dict[str, Any]],
) -> List[Dict]:
    if not grounding_result or grounding_result.get("page") is None:
        return retrieval_results
    doc_id = str(grounding_result.get("source_doc_id") or "")
    page = grounding_result.get("page")
    chart_id = grounding_result.get("chart_id")
    for result in retrieval_results:
        metadata = result.get("metadata") or {}
        if (
            str(metadata.get("sha1_name") or "") == doc_id
            and result.get("page") == page
            and metadata.get("chart_id") == chart_id
        ):
            return retrieval_results
    return [processor._chart_grounding_retrieval_result(grounding_result)] + list(retrieval_results)


def build_table_grounded_number_answer(processor, grounding_result: Dict[str, Any]) -> Dict[str, Any]:
    grounded_value = processor._grounded_number_value(grounding_result)
    unit_note = f"，并按问题要求换算为{grounding_result['target_unit']}" if grounding_result.get("target_unit") else ""
    logical_note = ""
    if grounding_result.get("logical_table_id") and grounding_result.get("logical_table_materialized"):
        logical_note = f"，并结合跨页逻辑表 {grounding_result.get('logical_table_id')} 的继承表头/单位"
    page = grounding_result.get("page")
    pages = [page] if page is not None else []
    for logical_page in grounding_result.get("member_pages") or []:
        if logical_page is not None and logical_page not in pages:
            pages.append(logical_page)
    for support_result in grounding_result.get("supporting_matches") or []:
        support_page = support_result.get("page")
        if support_page is not None and support_page not in pages:
            pages.append(support_page)
        if len(pages) >= 3:
            break
    pages = sorted(pages)
    return {
        "final_answer": grounded_value,
        "relevant_pages": pages,
        "references": [],
        "citations": [],
        "confidence": "high",
        "reasoning_summary": (
            f"答案来自表格 grounding：表 {grounding_result.get('table_id')} 第 {page} 页的单元格"
            f"{logical_note}{unit_note}。"
        ),
        "step_by_step_analysis": (
            f"匹配行头：{grounding_result.get('matched_row_headers') or []}；"
            f"匹配列头：{grounding_result.get('matched_col_headers') or []}；"
            f"原始值：{grounding_result.get('raw_value')}；标准值：{grounding_result.get('normalized_value')}。"
        ),
        "table_grounding_result": grounding_result,
    }


def build_chart_grounded_number_answer(processor, grounding_result: Dict[str, Any]) -> Dict[str, Any]:
    grounded_value = processor._grounded_number_value(grounding_result)
    unit_note = f"，并按问题要求换算为{grounding_result['target_unit']}" if grounding_result.get("target_unit") else ""
    page = grounding_result.get("page")
    pages = [page] if page is not None else []
    return {
        "final_answer": grounded_value,
        "relevant_pages": pages,
        "references": [],
        "citations": [],
        "confidence": "medium",
        "reasoning_summary": (
            f"答案来自 chart-to-table grounding：图表 {grounding_result.get('chart_id')} "
            f"第 {page} 页的 {grounding_result.get('series_name')} / {grounding_result.get('x_label')}"
            f"{unit_note}。"
        ),
        "step_by_step_analysis": (
            f"图表ID：{grounding_result.get('chart_id')}；"
            f"图片ID：{grounding_result.get('picture_id')}；"
            f"指标：{grounding_result.get('series_name')}；"
            f"横轴：{grounding_result.get('x_label')}；"
            f"原始值：{grounding_result.get('raw_value')}；标准值：{grounding_result.get('normalized_value')}。"
        ),
        "chart_grounding_result": grounding_result,
    }


def is_revenue_metric_question(question: str) -> bool:
    normalized = normalize_text(question or "")
    return "营业收入" in normalized or "营收" in normalized


def companies_in_question_order(question: str, companies: List[str]) -> List[str]:
    normalized_question = normalize_text(question or "").replace(" ", "")
    indexed = []
    for index, company in enumerate(companies):
        normalized_company = normalize_text(company).replace(" ", "")
        position = normalized_question.find(normalized_company) if normalized_company else -1
        indexed.append((position if position >= 0 else 10_000 + index, index, company))
    indexed.sort()
    return [company for _, _, company in indexed]


def threshold_value_yuan(question: str) -> Optional[float]:
    match = re.search(
        r"(?:超过|高于|大于)\s*([0-9０-９][0-9０-９,，]*(?:\.\d+)?)\s*(百万元|千万元|亿元|万元|元)?",
        question or "",
    )
    if not match:
        return None
    raw_number = match.group(1)
    unit = match.group(2) or "元"
    return parse_numeric_value(f"{raw_number}{unit}")


def target_unit_from_question(question: str) -> Optional[str]:
    question = question or ""
    for unit in ("百万元", "千万元", "亿元", "万元"):
        if unit in question:
            return unit
    if "元" in question and not any(currency_unit in question for currency_unit in ("美元", "港元", "欧元", "日元")):
        return "元"
    return None


def convert_yuan_to_target(value: Optional[float], target_unit: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    factors = {
        "元": 1.0,
        "万元": 1e4,
        "亿元": 1e8,
        "百万元": 1e6,
        "千万元": 1e7,
    }
    factor = factors.get(target_unit or "")
    return float(value) / factor if factor else value


def apply_target_unit_to_grounding(processor, grounding_result: Dict[str, Any], question: str) -> Dict[str, Any]:
    result = dict(grounding_result)
    target_unit = processor._target_unit_from_question(question)
    normalized_value = result.get("normalized_value")
    result["target_unit"] = target_unit
    result["answer_value"] = processor._convert_yuan_to_target(normalized_value, target_unit)
    if target_unit:
        result["unit_conversion"] = {
            "from": result.get("unit"),
            "to": target_unit,
            "base_value": normalized_value,
            "converted_value": result["answer_value"],
        }
    return result


def number_query_plan_for_company(processor, company_name: str, question: str) -> Tuple[QueryPlan, Dict[str, Any]]:
    query_plan = processor._build_query_plan(
        question,
        schema="number",
        company_name=company_name,
        mentioned_companies=[company_name],
        route_mode="explicit_company",
    )
    query_plan.filters.company_name = company_name
    query_plan.filters.question_kind = "number"

    if processor.report_catalog is not None:
        reports = [
            report
            for report in processor.report_catalog.get_reports()
            if report.company_name == company_name
        ]
        if query_plan.filters.year is not None:
            reports = [report for report in reports if report.report_year in (None, query_plan.filters.year)]
        if query_plan.filters.doc_source_type:
            reports = [
                report
                for report in reports
                if not report.doc_source_type or report.doc_source_type == query_plan.filters.doc_source_type
            ]
        if reports:
            reports.sort(key=lambda report: (report.report_year or -1, report.sha1), reverse=True)
            report = reports[0]
            query_plan.filters.candidate_doc_ids = [report.sha1]
            return query_plan, {
                "route_mode": "explicit_company",
                "selected_company": report.company_name,
                "candidate_companies": [report.company_name],
                "candidate_doc_ids": [report.sha1],
                "selected_report": report.to_dict(),
                "selection_reasons": ["company_name_exact_for_numeric_rule"],
            }
        try:
            _, route_info = processor.report_catalog.resolve_single_company(query_plan, limit=processor.candidate_doc_top_k)
            route_info["route_mode"] = "explicit_company"
            candidate_doc_ids = route_info.get("candidate_doc_ids") or []
            if candidate_doc_ids:
                query_plan.filters.candidate_doc_ids = candidate_doc_ids
            return query_plan, route_info
        except Exception:
            pass

    return query_plan, {
        "route_mode": "explicit_company",
        "selected_company": company_name,
        "candidate_companies": [company_name],
        "candidate_doc_ids": list(query_plan.filters.candidate_doc_ids or []),
        "selection_reasons": ["company_argument"],
    }


def cached_revenue_grounding(
    processor,
    *,
    company_name: str,
    question: str,
    query_plan: QueryPlan,
    candidate_doc_ids: List[str],
    allowed_doc_ids: List[str],
    retrieval_results: Optional[List[Dict]] = None,
) -> Optional[Dict[str, Any]]:
    if processor.table_grounder is None:
        return None
    doc_id = str((allowed_doc_ids or candidate_doc_ids or [""])[0] or "")
    if not doc_id:
        return None
    cache_key = (doc_id, query_plan.filters.year, "营业收入")
    with processor._revenue_grounding_cache_lock:
        cached = processor._revenue_grounding_cache.get(cache_key)
    if cached is None and cache_key not in processor._revenue_grounding_cache:
        year_text = f"{query_plan.filters.year}年报" if query_plan.filters.year is not None else ""
        sub_question = f"{company_name}{year_text}营业收入是多少元？"
        computed = processor.table_grounder.ground_number_query(
            question=sub_question,
            retrieval_results=retrieval_results or [],
            filters=query_plan.filters,
            candidate_doc_ids=candidate_doc_ids,
            allowed_doc_ids=allowed_doc_ids,
        )
        with processor._revenue_grounding_cache_lock:
            processor._revenue_grounding_cache.setdefault(cache_key, computed)
            cached = processor._revenue_grounding_cache.get(cache_key)
    if cached is None:
        return None
    return processor._apply_target_unit_to_grounding(cached, question)


def ground_revenue_for_company(processor, company_name: str, question: str) -> Dict[str, Any]:
    if processor.table_grounder is None:
        return {"company_name": company_name, "missing_reason": "numeric_grounding_disabled"}

    query_plan, route_info = processor._number_query_plan_for_company(company_name, question)
    selected_doc_id = processor._selected_doc_id(company_name, route_info, query_plan)
    allowed_doc_ids = [selected_doc_id] if selected_doc_id else list(query_plan.filters.candidate_doc_ids or [])
    if selected_doc_id:
        query_plan.filters.candidate_doc_ids = [selected_doc_id]
    grounding_result = processor._cached_revenue_grounding(
        company_name=company_name,
        question=question,
        query_plan=query_plan,
        candidate_doc_ids=allowed_doc_ids,
        allowed_doc_ids=allowed_doc_ids,
        retrieval_results=[],
    )
    if grounding_result is None or grounding_result.get("normalized_value") is None:
        return {
            "company_name": company_name,
            "query_plan": query_plan,
            "route_info": route_info,
            "missing_reason": "revenue_grounding_not_found",
        }

    return {
        "company_name": company_name,
        "value_yuan": grounding_result.get("normalized_value"),
        "grounded_answer": processor._grounded_number_value(grounding_result),
        "grounding_result": grounding_result,
        "retrieval_result": processor._table_grounding_retrieval_result(grounding_result),
        "retrieval_results": [
            processor._table_grounding_retrieval_result(grounding_result),
            *[
                processor._table_grounding_retrieval_result(
                    support_result,
                    chunk_type="table_support",
                    retrieval_source="table_support",
                )
                for support_result in processor._table_support_results(grounding_result)
            ],
        ],
        "query_plan": query_plan,
        "route_info": route_info,
    }


def numeric_rule_citations(processor, grounded_items: List[Dict[str, Any]], pages: List[int], *, build_citations: Callable) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []
    for item in grounded_items:
        retrieval_results = item.get("retrieval_results") or ([item["retrieval_result"]] if item.get("retrieval_result") else [])
        grounding_result = item.get("grounding_result")
        if not retrieval_results or not grounding_result:
            continue
        page = grounding_result.get("page")
        if page not in pages:
            continue
        citations.extend(
            build_citations(
                retrieval_results,
                pages,
                table_grounding_result=grounding_result,
                table_support_results=processor._table_support_results(grounding_result),
            )
        )
    return dedupe_citations(citations)


def build_numeric_rule_answer(
    processor,
    *,
    question: str,
    final_answer: Any,
    query_plan: QueryPlan,
    route_info: Dict[str, Any],
    grounded_items: List[Dict[str, Any]],
    rule_name: str,
    reasoning: str,
) -> Dict[str, Any]:
    retrieval_results = []
    for item in grounded_items:
        retrieval_results.extend(item.get("retrieval_results") or ([item["retrieval_result"]] if item.get("retrieval_result") else []))
    pages = [
        result.get("page")
        for item in grounded_items
        for result in (item.get("retrieval_results") or ([item["retrieval_result"]] if item.get("retrieval_result") else []))
        if result.get("page") is not None
    ]
    pages = list(dict.fromkeys(pages))
    if final_answer in (None, "N/A"):
        references: List[Dict[str, Any]] = []
        citations: List[Dict[str, Any]] = []
        pages = []
    else:
        references = processor._extract_references_from_results(pages, retrieval_results)
        citations = processor._numeric_rule_citations(grounded_items, pages)

    answer = {
        "final_answer": final_answer if final_answer is not None else "N/A",
        "relevant_pages": pages,
        "references": references,
        "citations": citations,
        "confidence": "high" if final_answer not in (None, "N/A") else "low",
        "confidence_reason": f"{rule_name} 由表格 grounding 的结构化数值直接计算。" if final_answer not in (None, "N/A") else f"{rule_name} 缺少关键数值，返回 N/A。",
        "reasoning_summary": reasoning,
        "step_by_step_analysis": reasoning if processor.reasoning_debug_enabled else "",
        "search_queries": [question],
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
    if grounded_items and len(grounded_items) == 1 and grounded_items[0].get("grounding_result"):
        answer["table_grounding_result"] = grounded_items[0]["grounding_result"]
    return answer


def answer_structured_revenue_comparison(
    processor,
    question: str,
    companies: List[str],
    schema: str,
) -> Optional[Dict[str, Any]]:
    if processor.table_grounder is None or len(companies) < 2 or not processor._is_revenue_metric_question(question):
        return None
    if not any(term in (question or "") for term in ("谁", "更高", "高于", "超过", "大于")):
        return None

    ordered_companies = processor._companies_in_question_order(question, companies)[:2]
    grounded_items = [processor._ground_revenue_for_company(company, question) for company in ordered_companies]
    missing = [item for item in grounded_items if item.get("value_yuan") is None]
    query_plan = processor._build_query_plan(
        question,
        schema="comparative",
        mentioned_companies=ordered_companies,
        route_mode="numeric_comparison_rule",
    )
    route_info = {
        "route_mode": "numeric_comparison_rule",
        "selected_company": None,
        "candidate_companies": ordered_companies,
        "selection_reasons": ["structured_revenue_comparison"],
        "numeric_comparison_rule": {
            "metric": "营业收入",
            "company_values": [
                {
                    "company_name": item.get("company_name"),
                    "value_yuan": item.get("value_yuan"),
                    "source_doc_id": (item.get("grounding_result") or {}).get("source_doc_id"),
                    "page": (item.get("grounding_result") or {}).get("page"),
                    "missing_reason": item.get("missing_reason"),
                }
                for item in grounded_items
            ],
        },
    }

    if missing:
        reasoning = "结构化营业收入比较缺少至少一个公司的表格 grounding 数值，因此返回 N/A。"
        return processor._build_numeric_rule_answer(
            question=question,
            final_answer="N/A",
            query_plan=query_plan,
            route_info=route_info,
            grounded_items=grounded_items,
            rule_name="numeric_comparison_rule",
            reasoning=reasoning,
        )

    first, second = grounded_items[0], grounded_items[1]
    if "谁" in question or "哪" in question:
        winner = first if float(first["value_yuan"]) >= float(second["value_yuan"]) else second
        final_answer = winner["company_name"]
        reasoning = (
            f"结构化比较营业收入：{first['company_name']}={first['value_yuan']} 元，"
            f"{second['company_name']}={second['value_yuan']} 元，因此答案为 {final_answer}。"
        )
    elif "是否" in question and any(term in question for term in ("高于", "超过", "大于")):
        final_answer = bool(float(first["value_yuan"]) > float(second["value_yuan"]))
        reasoning = (
            f"结构化判断营业收入：{first['company_name']}={first['value_yuan']} 元，"
            f"{second['company_name']}={second['value_yuan']} 元，比较结果为 {final_answer}。"
        )
    else:
        return None

    return processor._build_numeric_rule_answer(
        question=question,
        final_answer=final_answer,
        query_plan=query_plan,
        route_info=route_info,
        grounded_items=grounded_items,
        rule_name="numeric_comparison_rule",
        reasoning=reasoning,
    )


def answer_structured_revenue_threshold(
    processor,
    *,
    question: str,
    schema: str,
    route_decision: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if processor.table_grounder is None or (schema or "").lower() != "boolean" or not processor._is_revenue_metric_question(question):
        return None
    if "是否" not in (question or "") or not any(term in question for term in ("超过", "高于", "大于")):
        return None
    threshold_yuan = processor._threshold_value_yuan(question)
    if threshold_yuan is None:
        return None

    company_name = route_decision.get("company_name") or ""
    if not company_name:
        return None
    grounded_item = processor._ground_revenue_for_company(company_name, question)
    query_plan: QueryPlan = route_decision["query_plan"]
    route_info = dict(route_decision.get("route_info") or {})
    route_info.update(
        {
            "route_mode": "numeric_threshold_rule",
            "numeric_threshold_rule": {
                "metric": "营业收入",
                "threshold_yuan": threshold_yuan,
                "company_value_yuan": grounded_item.get("value_yuan"),
                "source_doc_id": (grounded_item.get("grounding_result") or {}).get("source_doc_id"),
                "page": (grounded_item.get("grounding_result") or {}).get("page"),
                "missing_reason": grounded_item.get("missing_reason"),
            },
        }
    )
    if grounded_item.get("value_yuan") is None:
        reasoning = "结构化营业收入阈值判断缺少表格 grounding 数值，因此返回 N/A。"
        final_answer: Any = "N/A"
    else:
        final_answer = bool(float(grounded_item["value_yuan"]) > float(threshold_yuan))
        reasoning = (
            f"结构化阈值判断营业收入：{company_name}={grounded_item['value_yuan']} 元，"
            f"阈值={threshold_yuan} 元，比较结果为 {final_answer}。"
        )

    return processor._build_numeric_rule_answer(
        question=question,
        final_answer=final_answer,
        query_plan=query_plan,
        route_info=route_info,
        grounded_items=[grounded_item],
        rule_name="numeric_threshold_rule",
        reasoning=reasoning,
    )
