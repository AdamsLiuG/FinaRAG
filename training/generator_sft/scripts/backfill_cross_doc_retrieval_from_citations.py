from __future__ import annotations

import argparse
import copy
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import append_jsonl, display_path, load_records, resolve_repo_path, utc_now_iso, write_json  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill cross-doc retrieval_results/rag_context from teacher citations for existing raw/cache records."
    )
    parser.add_argument("--raw-input-path", type=Path, required=True, help="Input raw teacher answers JSONL path.")
    parser.add_argument("--raw-output-path", type=Path, default=None, help="Output raw teacher answers JSONL path.")
    parser.add_argument("--cache-input-path", type=Path, default=None, help="Optional retrieved_cache JSONL path.")
    parser.add_argument("--cache-output-path", type=Path, default=None, help="Optional output retrieved_cache JSONL path.")
    parser.add_argument("--stats-output-path", type=Path, default=None, help="Optional stats JSON path.")
    parser.add_argument(
        "--backup-suffix",
        default=".bak_before_cross_doc_backfill",
        help="Suffix used when output path equals input path and an automatic backup is created.",
    )
    return parser


def _normalize_pages(values: Any) -> List[int]:
    pages: List[int] = []
    for value in values or []:
        try:
            pages.append(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(set(pages))


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _dedupe_preserve_order(items: Iterable[Any]) -> List[Any]:
    seen = set()
    deduped: List[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _is_non_na_final_answer(value: Any) -> bool:
    return _normalize_text(value).upper() not in {"", "N/A"}


def _record_doc_ids(record: Dict[str, Any]) -> List[str]:
    return [str(item) for item in (record.get("doc_ids") or []) if str(item or "").strip()]


def _citations_from_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    answer = record.get("answer") if isinstance(record.get("answer"), dict) else {}
    citations = answer.get("citations")
    if isinstance(citations, list):
        return [item for item in citations if isinstance(item, dict)]
    return []


def _serialized_result_from_citation(citation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    evidence_snippet = _normalize_text(citation.get("evidence_snippet") or citation.get("snippet"))
    if not evidence_snippet:
        return None

    doc_id = _normalize_text(citation.get("source") or citation.get("pdf_sha1") or citation.get("doc_id"))
    if not doc_id:
        return None

    try:
        page = int(citation.get("page"))
    except (TypeError, ValueError):
        return None

    company_name = _normalize_text(citation.get("company_name"))
    stock_code = _normalize_text(citation.get("stock_code"))
    security_code = _normalize_text(citation.get("security_code") or stock_code)
    report_year = citation.get("report_year")
    try:
        report_year = int(report_year) if report_year not in (None, "") else None
    except (TypeError, ValueError):
        report_year = None

    company_aliases = [item for item in [company_name, stock_code, security_code] if item]
    company_aliases = _dedupe_preserve_order(company_aliases)
    score = citation.get("score")
    try:
        numeric_score = float(score) if score not in (None, "") else 0.0
    except (TypeError, ValueError):
        numeric_score = 0.0

    result = {
        "page": page,
        "sha1_name": doc_id,
        "chunk_id": citation.get("chunk_id"),
        "chunk_type": citation.get("chunk_type") or "content",
        "node_type": citation.get("node_type") or "parent",
        "parent_chunk_id": citation.get("parent_chunk_id"),
        "section_title": citation.get("section_title"),
        "section_name": citation.get("section_name") or citation.get("section_title"),
        "report_section": citation.get("report_section") or citation.get("section_name") or citation.get("section_title"),
        "company_name": company_name or None,
        "company_aliases": company_aliases,
        "security_code": security_code or None,
        "stock_code": stock_code or None,
        "broker_name": citation.get("broker_name"),
        "currency": citation.get("currency"),
        "exchange": citation.get("exchange"),
        "board": citation.get("board"),
        "market_type": citation.get("market_type"),
        "industry_l1": citation.get("industry_l1") or citation.get("major_industry"),
        "industry_l2": citation.get("industry_l2"),
        "report_year": report_year,
        "report_type": citation.get("report_type"),
        "doc_source_type": citation.get("doc_source_type"),
        "report_date": citation.get("report_date"),
        "fiscal_year": str(report_year) if report_year is not None else None,
        "period": citation.get("period"),
        "unit_hint": citation.get("unit"),
        "language": citation.get("language") or "zh",
        "topic_flags": list(citation.get("topic_flags", []) or []),
        "business_tags": list(citation.get("business_tags", []) or []),
        "strategy_tags": list(citation.get("strategy_tags", []) or []),
        "factor_tags": list(citation.get("factor_tags", []) or []),
        "chain_position_major": citation.get("chain_position_major"),
        "chain_position_minor": list(citation.get("chain_position_minor", []) or []),
        "listing_tags": list(citation.get("listing_tags", []) or []),
        "ownership_tags": list(citation.get("ownership_tags", []) or []),
        "status_tags": list(citation.get("status_tags", []) or []),
        "style_tags": list(citation.get("style_tags", []) or []),
        "table_id": citation.get("table_id"),
        "matched_child_chunk_ids": list(citation.get("matched_child_chunk_ids", []) or []),
        "matched_tags": list(citation.get("matched_tags", []) or []),
        "matched_queries": [],
        "query_hit_count": 0,
        "result_scope": "parent",
        "retrieval_sources": list(citation.get("retrieval_sources", []) or []),
        "score": numeric_score,
        "final_score": numeric_score,
        "distance_rrf": citation.get("distance_rrf"),
        "colbert_score": citation.get("colbert_score"),
        "final_relevance_score": citation.get("final_relevance_score", numeric_score),
        "text": evidence_snippet,
        "text_preview": evidence_snippet[:280],
    }
    return result


def _build_serialized_retrieval_results(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    doc_order = {doc_id: idx for idx, doc_id in enumerate(_record_doc_ids(record))}
    results: List[Dict[str, Any]] = []
    seen = set()
    for citation in _citations_from_record(record):
        serialized = _serialized_result_from_citation(citation)
        if serialized is None:
            continue
        key = (
            str(serialized.get("sha1_name") or ""),
            int(serialized.get("page") or 0),
            str(serialized.get("chunk_id") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        results.append(serialized)

    results.sort(
        key=lambda item: (
            doc_order.get(str(item.get("sha1_name") or ""), 10**9),
            int(item.get("page") or 0),
            str(item.get("chunk_id") or ""),
        )
    )
    return results


def _build_rag_context(retrieval_results: List[Dict[str, Any]]) -> str:
    context_parts: List[str] = []
    for result in retrieval_results:
        page_number = result.get("page")
        company = _normalize_text(result.get("company_name"))
        doc_id = _normalize_text(result.get("sha1_name"))
        report_year = result.get("report_year")
        section_name = _normalize_text(result.get("section_name") or result.get("section_title"))
        chunk_type = _normalize_text(result.get("chunk_type"))
        node_type = _normalize_text(result.get("node_type"))
        matched_tags = [str(item) for item in (result.get("matched_tags") or []) if str(item).strip()]
        text = _normalize_text(result.get("text"))
        if not text:
            continue

        label = f"Text retrieved from page {page_number}"
        if company:
            label += f" | company: {company}"
        if doc_id:
            label += f" | doc_id: {doc_id}"
        if report_year not in (None, ""):
            label += f" | report_year: {report_year}"
        if section_name:
            label += f" | section: {section_name}"
        if chunk_type:
            label += f" | chunk_type: {chunk_type}"
        if node_type:
            label += f" | node_type: {node_type}"
        if matched_tags:
            label += f" | matched_tags: {', '.join(matched_tags)}"
        context_parts.append(f'{label}: \n"""\n{text}\n"""')

    return "\n\n---\n\n".join(context_parts)


def _build_retrieval_report_groups(retrieval_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for result in retrieval_results:
        doc_id = _normalize_text(result.get("sha1_name"))
        if not doc_id:
            continue
        group = grouped.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "company_name": result.get("company_name"),
                "stock_code": result.get("stock_code"),
                "report_year": result.get("report_year"),
                "final_score": None,
                "evidence_count": 0,
                "matched_tags": [],
                "evidence_chunks": [],
                "aggregation_score": None,
            },
        )
        group["evidence_chunks"].append(result)
        group["evidence_count"] += 1
        group["matched_tags"] = _dedupe_preserve_order(list(group.get("matched_tags", [])) + list(result.get("matched_tags", []) or []))
        numeric_score = result.get("final_score", result.get("score"))
        try:
            numeric_score = float(numeric_score) if numeric_score not in (None, "") else None
        except (TypeError, ValueError):
            numeric_score = None
        if numeric_score is not None:
            best_score = group.get("final_score")
            if best_score is None or numeric_score > best_score:
                group["final_score"] = numeric_score
                group["aggregation_score"] = numeric_score

    return list(grouped.values())


def backfill_record_from_citations(record: Dict[str, Any]) -> Dict[str, Any]:
    updated = copy.deepcopy(record)
    retrieval_results = _build_serialized_retrieval_results(updated)
    if not retrieval_results:
        return updated

    answer = updated.get("answer") if isinstance(updated.get("answer"), dict) else {}
    relevant_pages = _normalize_pages(answer.get("relevant_pages"))
    citation_pages = _normalize_pages(citation.get("page") for citation in _citations_from_record(updated))
    retrieval_pages = _normalize_pages(
        list(updated.get("retrieval_pages") or [])
        + list(answer.get("retrieval_pages") or [])
        + relevant_pages
        + citation_pages
    )
    rag_context = _build_rag_context(retrieval_results)
    retrieval_report_groups = _build_retrieval_report_groups(retrieval_results)

    updated["rag_context"] = rag_context
    updated["retrieval_results"] = retrieval_results
    updated["retrieval_pages"] = retrieval_pages

    answer = dict(answer)
    answer["retrieval_results"] = retrieval_results
    answer["retrieval_pages"] = retrieval_pages
    answer["retrieval_report_groups"] = retrieval_report_groups
    updated["answer"] = answer
    return updated


def _should_backfill_record(record: Dict[str, Any]) -> bool:
    if len(_record_doc_ids(record)) <= 1:
        return False
    if not _is_non_na_final_answer(((record.get("answer") or {}).get("final_answer"))):
        return False
    if record.get("retrieval_results"):
        return False
    if _normalize_text(record.get("rag_context")):
        return False
    return bool(_citations_from_record(record))


def _backfill_cache_record(cache_record: Dict[str, Any], raw_record: Dict[str, Any]) -> Dict[str, Any]:
    updated = copy.deepcopy(cache_record)
    raw_answer = raw_record.get("answer") if isinstance(raw_record.get("answer"), dict) else {}
    retrieval_results = list(raw_record.get("retrieval_results", []) or [])
    retrieval_pages = list(raw_record.get("retrieval_pages", []) or [])
    rag_context = str(raw_record.get("rag_context") or "")
    retrieval_report_groups = list(raw_answer.get("retrieval_report_groups", []) or _build_retrieval_report_groups(retrieval_results))

    updated["rag_context"] = rag_context
    updated["retrieval_results"] = retrieval_results
    updated["retrieval_pages"] = retrieval_pages
    updated["retrieval_report_groups"] = retrieval_report_groups
    updated["retrieval_result_count"] = len(retrieval_results)
    updated["retrieval_doc_count"] = len(
        {
            group.get("doc_id")
            for group in retrieval_report_groups
            if isinstance(group, dict) and group.get("doc_id")
        }
    )
    teacher_signal = updated.get("teacher_signal") if isinstance(updated.get("teacher_signal"), dict) else {}
    if teacher_signal:
        teacher_signal = dict(teacher_signal)
        teacher_signal["relevant_pages"] = list(raw_answer.get("relevant_pages", []) or teacher_signal.get("relevant_pages", []) or [])
        updated["teacher_signal"] = teacher_signal
    return updated


def _ensure_parent_dir(path: Optional[Path]) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


def _maybe_backup(input_path: Path, output_path: Path, backup_suffix: str) -> Optional[Path]:
    if input_path.resolve() != output_path.resolve():
        return None
    backup_path = input_path.with_name(input_path.name + backup_suffix)
    shutil.copy2(input_path, backup_path)
    return backup_path


def main() -> None:
    args = build_arg_parser().parse_args()
    raw_input_path = resolve_repo_path(REPO_ROOT, args.raw_input_path)
    raw_output_path = resolve_repo_path(REPO_ROOT, args.raw_output_path or args.raw_input_path)
    cache_input_path = resolve_repo_path(REPO_ROOT, args.cache_input_path)
    cache_output_path = resolve_repo_path(REPO_ROOT, args.cache_output_path or args.cache_input_path)
    stats_output_path = resolve_repo_path(REPO_ROOT, args.stats_output_path)

    if raw_input_path is None or raw_output_path is None:
        raise ValueError("raw input/output paths are required.")
    if not raw_input_path.exists():
        raise FileNotFoundError(f"missing raw input path: {raw_input_path}")
    if cache_input_path is not None and not cache_input_path.exists():
        raise FileNotFoundError(f"missing cache input path: {cache_input_path}")

    _ensure_parent_dir(raw_output_path)
    _ensure_parent_dir(cache_output_path)
    _ensure_parent_dir(stats_output_path)

    raw_backup_path = _maybe_backup(raw_input_path, raw_output_path, args.backup_suffix)
    cache_backup_path = None
    if cache_input_path is not None and cache_output_path is not None:
        cache_backup_path = _maybe_backup(cache_input_path, cache_output_path, args.backup_suffix)

    raw_records = load_records(raw_input_path)
    patched_by_query_id: Dict[str, Dict[str, Any]] = {}
    raw_stats = Counter()

    raw_output_path.write_text("", encoding="utf-8")
    for record in raw_records:
        raw_stats["total_raw_records"] += 1
        query_id = str(record.get("query_id") or "").strip()
        if _should_backfill_record(record):
            patched_record = backfill_record_from_citations(record)
            if patched_record.get("retrieval_results") and patched_record.get("rag_context"):
                record = patched_record
                if query_id:
                    patched_by_query_id[query_id] = patched_record
                raw_stats["patched_raw_records"] += 1
            else:
                raw_stats["skipped_raw_records"] += 1
        append_jsonl(raw_output_path, record)

    cache_stats = Counter()
    if cache_input_path is not None and cache_output_path is not None:
        cache_records = load_records(cache_input_path)
        cache_output_path.write_text("", encoding="utf-8")
        for record in cache_records:
            cache_stats["total_cache_records"] += 1
            query_id = str(record.get("query_id") or "").strip()
            patched_raw = patched_by_query_id.get(query_id)
            if patched_raw is not None:
                record = _backfill_cache_record(record, patched_raw)
                cache_stats["patched_cache_records"] += 1
            append_jsonl(cache_output_path, record)

    stats = {
        "build_timestamp": utc_now_iso(),
        "raw_input_path": display_path(raw_input_path, REPO_ROOT),
        "raw_output_path": display_path(raw_output_path, REPO_ROOT),
        "cache_input_path": display_path(cache_input_path, REPO_ROOT),
        "cache_output_path": display_path(cache_output_path, REPO_ROOT),
        "raw_backup_path": display_path(raw_backup_path, REPO_ROOT),
        "cache_backup_path": display_path(cache_backup_path, REPO_ROOT),
        **dict(raw_stats),
        **dict(cache_stats),
    }
    if stats_output_path is not None:
        write_json(stats_output_path, stats)
    else:
        print(stats)


if __name__ == "__main__":
    main()
