from __future__ import annotations

import argparse
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api_requests import APIProcessor  # noqa: E402
from training.common import (  # noqa: E402
    append_jsonl,
    build_questions_processor,
    build_rag_prompt_bundle,
    compact_json_dumps,
    display_path,
    load_records,
    load_yaml_mapping,
    prune_answer_to_schema,
    resolve_dataset_root,
    resolve_repo_path,
    stable_hash_int,
    utc_now_iso,
    write_json,
)
from training.generator_sft.scripts.build_hard_context_samples import (  # noqa: E402
    _build_user_prompt,
    _candidate_identity,
    _candidate_to_context_item,
    _collect_candidate_record_inline,
    _format_context_items,
    _index_records_by_query_id,
    _select_support_items,
    _support_identity,
)
from training.generator_sft.scripts.convert_to_chat_sft import build_chat_record  # noqa: E402
from training.reranker_distill.scripts.collect_candidate_pool import collect_candidate_pool_record  # noqa: E402


_THREAD_LOCAL = threading.local()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build wrong-context refusal samples and keep only teacher-validated refusals.")
    parser.add_argument("--config-path", type=Path, default=None, help="YAML config path.")
    parser.add_argument("--input-path", type=Path, default=None, help="Filtered positive sample JSONL path.")
    parser.add_argument("--retrieved-cache-input-path", type=Path, default=None, help="retrieved_cache JSONL path.")
    parser.add_argument("--candidate-pool-input-path", type=Path, default=None, help="Optional prebuilt candidate pool JSONL path.")
    parser.add_argument("--candidate-pool-output-path", type=Path, default=None, help="Optional JSONL path to persist inline same-doc candidate pools.")
    parser.add_argument("--output-path", type=Path, default=None, help="Accepted wrong-context refusal JSONL path.")
    parser.add_argument("--chat-output-path", type=Path, default=None, help="Optional chat-format JSONL output path.")
    parser.add_argument("--rejected-output-path", type=Path, default=None, help="Rejected sample JSONL path.")
    parser.add_argument("--stats-output-path", type=Path, default=None, help="Stats JSON path.")
    parser.add_argument("--retrieval-config-path", type=Path, default=None, help="Retrieval config path for inline candidate collection.")
    parser.add_argument("--dataset-root-path", type=Path, default=None, help="Dataset root used for inline candidate collection.")
    parser.add_argument("--teacher-answer-provider", default=None, help="Teacher API provider used for refusal validation.")
    parser.add_argument("--teacher-answer-model", default=None, help="Teacher model used for refusal validation.")
    parser.add_argument("--answer-temperature", type=float, default=None, help="Teacher sampling temperature.")
    parser.add_argument("--candidate-pool-size", type=int, default=None, help="Final deduped candidate cap per query for inline candidate collection.")
    parser.add_argument("--per-query-retrieval-top-k", type=int, default=None, help="Per-query retrieval cap before multi-query merge.")
    parser.add_argument("--parallel-requests", type=int, default=None, help="Forwarded to inline candidate collection.")
    parser.add_argument("--source-priority", nargs="*", default=None, help="Preferred wrong-context source order.")
    parser.add_argument("--min-context-results", type=int, default=None, help="Minimum wrong-context chunks required to keep a sample.")
    parser.add_argument("--max-context-results", type=int, default=None, help="Maximum wrong-context chunks included in the prompt.")
    parser.add_argument("--cross-doc-candidate-doc-cap", type=int, default=None, help="Max same-year cross-doc reports to search.")
    parser.add_argument("--max-support-results", type=int, default=None, help="Maximum support chunks to inspect from retrieved_cache.")
    parser.add_argument("--max-context-chars", type=int, default=None, help="Maximum context characters for the regenerated wrong context.")
    parser.add_argument("--max-doc-chars", type=int, default=None, help="Maximum per-chunk characters for the regenerated wrong context.")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional cap over filtered input records.")
    parser.add_argument("--resume", action="store_true", help="Skip query_ids already accepted in output_path and retry the rest.")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume behavior and rebuild outputs from scratch.")
    return parser


def _coalesce(cli_value: Any, config_value: Any, default: Any = None) -> Any:
    return cli_value if cli_value is not None else (config_value if config_value is not None else default)


def _reset_output_file(path: Optional[Path]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _record_key(record: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(record.get("sample_id") or ""),
        str(record.get("query_id") or ""),
        str(record.get("variant_type") or ""),
    )


def _accepted_resume_state(path: Optional[Path]) -> Tuple[set[str], int, int]:
    if path is None or not path.exists():
        return set(), 0, 0

    query_ids: set[str] = set()
    record_count = 0
    max_sample_index = 0
    for record in load_records(path):
        query_id = str(record.get("query_id") or "").strip()
        if query_id:
            query_ids.add(query_id)
        sample_id = str(record.get("sample_id") or "")
        if sample_id.startswith("wrctx-"):
            try:
                max_sample_index = max(max_sample_index, int(sample_id.removeprefix("wrctx-")))
            except ValueError:
                pass
        record_count += 1
    return query_ids, record_count, max(record_count, max_sample_index)


def _backfill_chat_from_accepted(output_path: Optional[Path], chat_output_path: Optional[Path]) -> int:
    if output_path is None or chat_output_path is None or not output_path.exists():
        return 0

    existing_chat_keys = set()
    if chat_output_path.exists():
        for record in load_records(chat_output_path):
            meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
            existing_chat_keys.add(
                (
                    str(meta.get("sample_id") or ""),
                    str(meta.get("query_id") or ""),
                    str(meta.get("variant_type") or ""),
                )
            )

    backfilled = 0
    for record in load_records(output_path):
        key = _record_key(record)
        if key in existing_chat_keys:
            continue
        append_jsonl(chat_output_path, build_chat_record(record))
        existing_chat_keys.add(key)
        backfilled += 1
    return backfilled


def _normalize_pages(values: Any) -> List[int]:
    pages: List[int] = []
    for value in values or []:
        try:
            pages.append(int(value))
        except (TypeError, ValueError):
            continue
    return pages


def _normalize_section_name(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _non_empty_text(value: Any) -> bool:
    return bool(str(value or "").strip())


def _primary_year(record: Dict[str, Any], cache_record: Dict[str, Any]) -> Optional[int]:
    expected_filters = cache_record.get("expected_filters") if isinstance(cache_record.get("expected_filters"), dict) else {}
    for candidate in (
        expected_filters.get("report_year"),
        expected_filters.get("year"),
    ):
        try:
            return int(candidate)
        except (TypeError, ValueError):
            continue

    for item in cache_record.get("retrieval_results", []) or []:
        try:
            return int(item.get("report_year"))
        except (TypeError, ValueError):
            continue

    for item in cache_record.get("retrieval_results", []) or []:
        try:
            return int(item.get("fiscal_year"))
        except (TypeError, ValueError):
            continue

    return None


def _primary_doc_source_type(cache_record: Dict[str, Any]) -> Optional[str]:
    expected_filters = cache_record.get("expected_filters") if isinstance(cache_record.get("expected_filters"), dict) else {}
    doc_source_type = expected_filters.get("doc_source_type")
    if doc_source_type:
        return str(doc_source_type)
    for item in cache_record.get("retrieval_results", []) or []:
        value = item.get("doc_source_type")
        if value:
            return str(value)
    return None


def _support_section_names(support_items: List[Dict[str, Any]]) -> set[str]:
    names = set()
    for item in support_items:
        normalized = _normalize_section_name(item.get("section_name"))
        if normalized:
            names.add(normalized)
    return names


def _dedupe_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for candidate in candidates:
        key = _candidate_identity(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _partition_same_doc_candidates(
    candidate_record: Dict[str, Any],
    *,
    primary_doc_id: str,
    support_items: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    support_keys = {_support_identity(item, primary_doc_id) for item in support_items}
    support_sections = _support_section_names(support_items)
    groups = {
        "same_company_wrong_section": [],
        "same_metric_wrong_page": [],
        "support_replaced_high_similarity_non_support": [],
    }
    for candidate in candidate_record.get("candidates", []) or []:
        if not _non_empty_text(candidate.get("text")):
            continue
        if str(candidate.get("doc_id") or "") != primary_doc_id:
            continue
        if _candidate_identity(candidate) in support_keys:
            continue

        groups["support_replaced_high_similarity_non_support"].append(candidate)
        candidate_section = _normalize_section_name(candidate.get("section_name") or candidate.get("report_section"))
        if candidate_section and candidate_section in support_sections:
            groups["same_metric_wrong_page"].append(candidate)
        else:
            groups["same_company_wrong_section"].append(candidate)

    return {name: _dedupe_candidates(values) for name, values in groups.items()}


def _thread_local_questions_processor(
    *,
    retrieval_config_path: Path,
    dataset_root_path: Path,
    parallel_requests: int,
):
    processor = getattr(_THREAD_LOCAL, "wrong_context_processor", None)
    processor_key = getattr(_THREAD_LOCAL, "wrong_context_processor_key", None)
    expected_key = (str(retrieval_config_path), str(dataset_root_path), int(parallel_requests))
    if processor is None or processor_key != expected_key:
        processor = build_questions_processor(
            REPO_ROOT,
            retrieval_config_path,
            dataset_root=dataset_root_path,
            reasoning_debug_enabled=True,
            parallel_requests=parallel_requests,
        )
        _THREAD_LOCAL.wrong_context_processor = processor
        _THREAD_LOCAL.wrong_context_processor_key = expected_key
    return processor


def _thread_local_api_processor(provider: str) -> APIProcessor:
    processor = getattr(_THREAD_LOCAL, "wrong_context_api_processor", None)
    current_provider = getattr(_THREAD_LOCAL, "wrong_context_api_provider", None)
    if processor is None or current_provider != provider:
        processor = APIProcessor(provider=provider)
        _THREAD_LOCAL.wrong_context_api_processor = processor
        _THREAD_LOCAL.wrong_context_api_provider = provider
    return processor


def _build_same_year_cross_doc_candidate_record(
    record: Dict[str, Any],
    cache_record: Dict[str, Any],
    processor,
    *,
    primary_doc_id: str,
    candidate_pool_size: int,
    per_query_retrieval_top_k: int,
    parallel_requests: int,
    retrieval_config_path: Path,
    dataset_root_path: Path,
    cross_doc_candidate_doc_cap: int,
) -> Optional[Dict[str, Any]]:
    report_catalog = getattr(processor, "report_catalog", None)
    if report_catalog is None:
        return None

    primary_year = _primary_year(record, cache_record)
    if primary_year is None:
        return None
    primary_company_name = str(record.get("company_name") or "")
    primary_doc_source_type = _primary_doc_source_type(cache_record)

    candidate_doc_ids: List[str] = []
    for report in report_catalog.get_reports():
        if report.sha1 == primary_doc_id:
            continue
        if primary_company_name and report.company_name == primary_company_name:
            continue
        if report.report_year != primary_year:
            continue
        if primary_doc_source_type and report.doc_source_type and report.doc_source_type != primary_doc_source_type:
            continue
        candidate_doc_ids.append(report.sha1)
        if len(candidate_doc_ids) >= cross_doc_candidate_doc_cap:
            break

    if not candidate_doc_ids:
        return None

    expected_filters = dict(cache_record.get("expected_filters") or {})
    if primary_year is not None and expected_filters.get("report_year") is None and expected_filters.get("year") is None:
        expected_filters["report_year"] = primary_year
    synthetic_record = {
        "query_id": f"{record.get('query_id')}::same_year_wrong_company",
        "question_text": record.get("question_text"),
        "schema": record.get("schema"),
        "company_name": "",
        "mentioned_companies": [],
        "doc_ids": candidate_doc_ids,
        "expected_filters": expected_filters,
        "source": record.get("source"),
        "difficulty": cache_record.get("difficulty"),
        "should_refuse": False,
    }
    return collect_candidate_pool_record(
        synthetic_record,
        retrieval_config_path=retrieval_config_path,
        dataset_root_path=dataset_root_path,
        candidate_pool_size=candidate_pool_size,
        per_query_retrieval_top_k=per_query_retrieval_top_k,
        parallel_requests=parallel_requests,
    )


def _compose_source_candidates(
    primary_source: str,
    groups: Dict[str, List[Dict[str, Any]]],
    *,
    min_context_results: int,
    max_context_results: int,
    source_loader: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
) -> Optional[Tuple[List[Dict[str, Any]], List[str]]]:
    fallback_map = {
        "same_company_wrong_section": [
            "same_metric_wrong_page",
            "support_replaced_high_similarity_non_support",
            "same_year_wrong_company",
        ],
        "same_year_wrong_company": [
            "support_replaced_high_similarity_non_support",
            "same_company_wrong_section",
            "same_metric_wrong_page",
        ],
        "same_metric_wrong_page": [
            "same_company_wrong_section",
            "support_replaced_high_similarity_non_support",
            "same_year_wrong_company",
        ],
        "support_replaced_high_similarity_non_support": [
            "same_company_wrong_section",
            "same_metric_wrong_page",
            "same_year_wrong_company",
        ],
    }
    selected: List[Dict[str, Any]] = []
    seen = set()
    source_mix: List[str] = []

    for source_name in [primary_source, *fallback_map.get(primary_source, [])]:
        source_candidates = groups.get(source_name, []) or []
        if not source_candidates and source_loader is not None:
            loaded_candidates = _dedupe_candidates(source_loader(source_name) or [])
            if loaded_candidates:
                groups[source_name] = loaded_candidates
                source_candidates = loaded_candidates
        if not source_candidates:
            continue
        source_mix.append(source_name)
        for candidate in source_candidates:
            key = _candidate_identity(candidate)
            if key in seen:
                continue
            seen.add(key)
            selected.append(candidate)
            if len(selected) >= max_context_results:
                break
        if len(selected) >= min_context_results:
            return selected[:max_context_results], source_mix
        if len(selected) >= max_context_results:
            break
    return None


def _run_record_tasks(records: List[Any], worker_fn: Callable[[Any], Any], max_workers: int) -> List[Any]:
    if max_workers <= 1 or len(records) <= 1:
        return [worker_fn(record) for record in records]

    ordered_results: List[Any] = [None] * len(records)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(worker_fn, record): index
            for index, record in enumerate(records)
        }
        for future in as_completed(future_to_index):
            ordered_results[future_to_index[future]] = future.result()
    return ordered_results


def _teacher_refusal_answer(
    api_processor: APIProcessor,
    record: Dict[str, Any],
    *,
    provider: str,
    model: str,
    temperature: float,
) -> Dict[str, Any]:
    schema = str(record.get("schema") or "")
    response_format = build_rag_prompt_bundle(schema, provider=provider)[2]
    raw_answer = api_processor.send_message(
        model=model,
        temperature=temperature,
        system_content=str(record.get("system_prompt") or "").strip(),
        human_content=str(record.get("user_prompt") or "").strip(),
        is_structured=True,
        response_format=response_format,
    )
    return prune_answer_to_schema(raw_answer, schema=schema, provider=provider)


def _build_rejected_record(
    record: Dict[str, Any],
    *,
    reason: str,
    source_type: Optional[str],
    source_mix: Optional[List[str]],
    wrong_pages: Optional[List[int]] = None,
    wrong_doc_ids: Optional[List[str]] = None,
    teacher_answer: Optional[Dict[str, Any]] = None,
    candidate_pool_source: Optional[str] = None,
    attempts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    parent_answer = record.get("assistant_response") if isinstance(record.get("assistant_response"), dict) else {}
    rejected_record = {
        "sample_id": record.get("sample_id"),
        "query_id": record.get("query_id"),
        "schema": record.get("schema"),
        "company_name": record.get("company_name"),
        "doc_ids": list(record.get("doc_ids", []) or []),
        "parent_sample_id": record.get("sample_id"),
        "parent_query_id": record.get("query_id"),
        "parent_final_answer": parent_answer.get("final_answer"),
        "wrong_context_source_type": source_type,
        "wrong_context_source_mix": list(source_mix or []),
        "wrong_context_pages": list(wrong_pages or []),
        "wrong_context_doc_ids": list(wrong_doc_ids or []),
        "candidate_pool_source": candidate_pool_source,
        "teacher_answer": teacher_answer,
        "reason": reason,
        "build_timestamp": utc_now_iso(),
    }
    if attempts:
        rejected_record["attempts"] = attempts
    return rejected_record


def _resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    default_config_path = REPO_ROOT / "training/generator_sft/configs/wrong_context.example.yaml"
    config_path = args.config_path or (default_config_path if default_config_path.exists() else None)
    config = load_yaml_mapping(config_path)

    resume_config = config.get("resume")
    resume = True if args.resume else False if args.no_resume else bool(resume_config if resume_config is not None else True)

    input_path = resolve_repo_path(REPO_ROOT, _coalesce(args.input_path, config.get("input_path")))
    retrieved_cache_input_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(args.retrieved_cache_input_path, config.get("retrieved_cache_input_path")),
    )
    candidate_pool_input_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(args.candidate_pool_input_path, config.get("candidate_pool_input_path")),
    )
    candidate_pool_output_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(args.candidate_pool_output_path, config.get("candidate_pool_output_path")),
    )
    output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.output_path, config.get("output_path")))
    chat_output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.chat_output_path, config.get("chat_output_path")))
    rejected_output_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(args.rejected_output_path, config.get("rejected_output_path")),
    )
    stats_output_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(args.stats_output_path, config.get("stats_output_path")),
    )
    retrieval_config_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(args.retrieval_config_path, config.get("retrieval_config_path")),
    )
    dataset_root_path = resolve_dataset_root(
        REPO_ROOT,
        _coalesce(args.dataset_root_path, config.get("dataset_root_path")),
    )
    if candidate_pool_input_path is not None and not candidate_pool_input_path.exists():
        candidate_pool_input_path = None
    if input_path is None or retrieved_cache_input_path is None or output_path is None or rejected_output_path is None or stats_output_path is None:
        raise ValueError("input/retrieved-cache/output/rejected/stats paths are required.")
    if retrieval_config_path is None:
        raise ValueError("retrieval_config_path is required for wrong-context generation.")

    teacher_answer_provider = str(_coalesce(args.teacher_answer_provider, config.get("teacher_answer_provider"), "qwen"))
    teacher_answer_model = _coalesce(args.teacher_answer_model, config.get("teacher_answer_model"))
    if not teacher_answer_model:
        raise ValueError("teacher_answer_model is required for wrong-context refusal validation.")

    source_priority = [
        str(value)
        for value in (
            _coalesce(
                args.source_priority,
                config.get("source_priority"),
                [
                    "same_company_wrong_section",
                    "same_year_wrong_company",
                    "same_metric_wrong_page",
                    "support_replaced_high_similarity_non_support",
                ],
            )
            or []
        )
        if str(value).strip()
    ]

    return {
        "config_path": config_path,
        "input_path": input_path,
        "retrieved_cache_input_path": retrieved_cache_input_path,
        "candidate_pool_input_path": candidate_pool_input_path,
        "candidate_pool_output_path": candidate_pool_output_path,
        "output_path": output_path,
        "chat_output_path": chat_output_path,
        "rejected_output_path": rejected_output_path,
        "stats_output_path": stats_output_path,
        "retrieval_config_path": retrieval_config_path,
        "dataset_root_path": dataset_root_path,
        "teacher_answer_provider": teacher_answer_provider,
        "teacher_answer_model": teacher_answer_model,
        "answer_temperature": float(_coalesce(args.answer_temperature, config.get("answer_temperature"), 0.0)),
        "candidate_pool_size": max(1, int(_coalesce(args.candidate_pool_size, config.get("candidate_pool_size"), 32))),
        "per_query_retrieval_top_k": max(1, int(_coalesce(args.per_query_retrieval_top_k, config.get("per_query_retrieval_top_k"), 24))),
        "parallel_requests": max(1, int(_coalesce(args.parallel_requests, config.get("parallel_requests"), 1))),
        "source_priority": source_priority,
        "min_context_results": max(1, int(_coalesce(args.min_context_results, config.get("min_context_results"), 2))),
        "max_context_results": max(1, int(_coalesce(args.max_context_results, config.get("max_context_results"), 3))),
        "cross_doc_candidate_doc_cap": max(1, int(_coalesce(args.cross_doc_candidate_doc_cap, config.get("cross_doc_candidate_doc_cap"), 8))),
        "max_support_results": max(1, int(_coalesce(args.max_support_results, config.get("max_support_results"), 3))),
        "max_context_chars": max(1, int(_coalesce(args.max_context_chars, config.get("max_context_chars"), 12000))),
        "max_doc_chars": max(1, int(_coalesce(args.max_doc_chars, config.get("max_doc_chars"), 1800))),
        "max_queries": max(0, int(_coalesce(args.max_queries, config.get("max_queries"), 0) or 0)),
        "resume": resume,
    }


def _process_filtered_record(
    filtered_record: Dict[str, Any],
    *,
    settings: Dict[str, Any],
    cache_by_query_id: Dict[str, Dict[str, Any]],
    candidate_by_query_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    query_id = str(filtered_record.get("query_id") or "").strip()
    source_type: Optional[str] = None
    source_mix: List[str] = []
    wrong_pages: List[int] = []
    wrong_doc_ids: List[str] = []
    candidate_pool_source: Optional[str] = None
    teacher_answer: Optional[Dict[str, Any]] = None
    attempt_logs: List[Dict[str, Any]] = []
    inline_same_doc_candidate_collected = 0
    inline_candidate_record: Optional[Dict[str, Any]] = None
    same_year_candidate_record: Optional[Dict[str, Any]] = None
    same_year_loaded = False
    processor = _thread_local_questions_processor(
        retrieval_config_path=settings["retrieval_config_path"],
        dataset_root_path=settings["dataset_root_path"],
        parallel_requests=settings["parallel_requests"],
    )
    api_processor = _thread_local_api_processor(settings["teacher_answer_provider"])

    try:
        if bool(filtered_record.get("should_refuse", False)):
            raise ValueError("wrong_context_requires_positive_parent")

        parent_answer = filtered_record.get("assistant_response") if isinstance(filtered_record.get("assistant_response"), dict) else {}
        if parent_answer.get("final_answer") in (None, "", [], "N/A"):
            raise ValueError("wrong_context_requires_positive_answer")

        doc_ids = [str(value) for value in filtered_record.get("doc_ids", []) if value not in (None, "")]
        if len(doc_ids) != 1:
            raise ValueError("wrong_context_requires_single_doc")
        primary_doc_id = doc_ids[0]

        cache_record = cache_by_query_id.get(query_id)
        if cache_record is None:
            raise ValueError("missing_retrieved_cache")

        support_items = _select_support_items(
            cache_record,
            primary_doc_id=primary_doc_id,
            max_support_results=settings["max_support_results"],
        )
        if not support_items:
            raise ValueError("missing_support_results")

        candidate_record = candidate_by_query_id.get(query_id)
        candidate_pool_source = "prebuilt_same_doc"
        if candidate_record is None:
            candidate_record = _collect_candidate_record_inline(
                cache_record,
                retrieval_config_path=settings["retrieval_config_path"],
                dataset_root_path=settings["dataset_root_path"],
                candidate_pool_size=settings["candidate_pool_size"],
                per_query_retrieval_top_k=settings["per_query_retrieval_top_k"],
                parallel_requests=settings["parallel_requests"],
            )
            inline_candidate_record = candidate_record
            candidate_pool_source = "inline_same_doc"
            inline_same_doc_candidate_collected += 1

        groups = _partition_same_doc_candidates(
            candidate_record,
            primary_doc_id=primary_doc_id,
            support_items=support_items,
        )

        def source_loader(source_name: str) -> List[Dict[str, Any]]:
            nonlocal candidate_pool_source
            nonlocal same_year_candidate_record
            nonlocal same_year_loaded
            if source_name != "same_year_wrong_company" or same_year_loaded:
                return []
            same_year_loaded = True
            same_year_candidate_record = _build_same_year_cross_doc_candidate_record(
                filtered_record,
                cache_record,
                processor,
                primary_doc_id=primary_doc_id,
                candidate_pool_size=settings["candidate_pool_size"],
                per_query_retrieval_top_k=settings["per_query_retrieval_top_k"],
                parallel_requests=settings["parallel_requests"],
                retrieval_config_path=settings["retrieval_config_path"],
                dataset_root_path=settings["dataset_root_path"],
                cross_doc_candidate_doc_cap=settings["cross_doc_candidate_doc_cap"],
            )
            if same_year_candidate_record is None:
                return []
            candidate_pool_source = (
                f"{candidate_pool_source}+inline_same_year_cross_doc"
                if candidate_pool_source
                else "inline_same_year_cross_doc"
            )
            return [
                candidate
                for candidate in (same_year_candidate_record.get("candidates", []) or [])
                if _non_empty_text(candidate.get("text"))
            ]

        accepted_sample_record: Optional[Dict[str, Any]] = None
        source_attempted = False
        support_pages = sorted({int(item.get("page") or 0) for item in support_items if item.get("page") not in (None, "")})
        for preferred_source in settings["source_priority"]:
            composed = _compose_source_candidates(
                preferred_source,
                groups,
                min_context_results=settings["min_context_results"],
                max_context_results=settings["max_context_results"],
                source_loader=source_loader,
            )
            if composed is None:
                continue
            source_attempted = True
            current_candidates, current_source_mix = composed
            current_wrong_context_items = [
                _candidate_to_context_item(candidate, fallback_company_name=filtered_record.get("company_name"))
                for candidate in current_candidates
            ]
            current_wrong_rag_context = _format_context_items(
                current_wrong_context_items,
                max_doc_chars=settings["max_doc_chars"],
                max_context_chars=settings["max_context_chars"],
            )
            current_wrong_pages = sorted(
                {
                    int(item.get("page") or 0)
                    for item in current_wrong_context_items
                    if item.get("page") not in (None, "")
                }
            )
            current_wrong_doc_ids = sorted(
                {
                    str(item.get("doc_id") or "")
                    for item in current_wrong_context_items
                    if str(item.get("doc_id") or "").strip()
                }
            )
            current_teacher_answer: Optional[Dict[str, Any]] = None
            try:
                if not current_wrong_rag_context.strip():
                    raise ValueError("empty_wrong_context")
                if set(current_wrong_pages) & set(support_pages):
                    raise ValueError("wrong_context_support_overlap")

                system_prompt = str(filtered_record.get("system_prompt") or "").strip()
                if not system_prompt:
                    provider = settings["teacher_answer_provider"]
                    system_prompt = build_rag_prompt_bundle(str(filtered_record.get("schema") or ""), provider=provider)[0]
                user_prompt = _build_user_prompt(filtered_record, cache_record, current_wrong_rag_context)

                teacher_input_record = {
                    **filtered_record,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
                current_teacher_answer = _teacher_refusal_answer(
                    api_processor,
                    teacher_input_record,
                    provider=settings["teacher_answer_provider"],
                    model=settings["teacher_answer_model"],
                    temperature=settings["answer_temperature"],
                )

                final_answer = current_teacher_answer.get("final_answer")
                if final_answer != "N/A":
                    raise ValueError("teacher_answered_under_wrong_context")

                relevant_pages = _normalize_pages(current_teacher_answer.get("relevant_pages"))
                if relevant_pages and not set(relevant_pages).issubset(set(current_wrong_pages)):
                    raise ValueError("wrong_context_hallucinated_pages")

                accepted_checks = [
                    "retrieval_present",
                    "schema_pruned",
                    "refusal_sample",
                    "wrong_context_support_removed",
                    "wrong_context_teacher_refused",
                    "wrong_context_candidate_pool",
                ]
                if relevant_pages and "wrong_context_relevant_pages_subset" not in accepted_checks:
                    accepted_checks.append("wrong_context_relevant_pages_subset")

                source_type = preferred_source
                source_mix = current_source_mix
                wrong_pages = current_wrong_pages
                wrong_doc_ids = current_wrong_doc_ids
                teacher_answer = current_teacher_answer

                accepted_sample_record = {
                    **filtered_record,
                    "parent_sample_id": filtered_record.get("sample_id"),
                    "parent_query_id": filtered_record.get("query_id"),
                    "parent_assistant_response": parent_answer,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "assistant_response": current_teacher_answer,
                    "assistant_response_json": compact_json_dumps(current_teacher_answer),
                    "accepted_checks": accepted_checks,
                    "source": "wrong_context_refusal",
                    "variant_type": "wrong_context_refusal.teacher_validated.v1",
                    "should_refuse": True,
                    "retrieval_pages": current_wrong_pages,
                    "wrong_context_pages": current_wrong_pages,
                    "context_doc_ids": current_wrong_doc_ids,
                    "wrong_context_source_type": preferred_source,
                    "wrong_context_source_mix": current_source_mix,
                    "candidate_pool_source": candidate_pool_source,
                    "wrong_context_stats": {
                        "support_result_count": len(support_items),
                        "wrong_result_count": len(current_wrong_context_items),
                        "candidate_pool_size": len(candidate_record.get("candidates", []) or []),
                        "same_year_cross_doc_candidate_pool_size": len((same_year_candidate_record or {}).get("candidates", []) or []),
                        "support_pages": support_pages,
                        "wrong_pages": current_wrong_pages,
                        "support_doc_ids": [primary_doc_id],
                        "wrong_doc_ids": current_wrong_doc_ids,
                        "support_overlap_count": 0,
                        "support_removed": True,
                        "source_type": preferred_source,
                        "source_mix": current_source_mix,
                        "source_candidate_counts": {
                            group_name: len(group_candidates)
                            for group_name, group_candidates in groups.items()
                        },
                        "context_hash": stable_hash_int(current_wrong_rag_context),
                    },
                    "teacher_answer_provider": settings["teacher_answer_provider"],
                    "teacher_answer_model": settings["teacher_answer_model"],
                    "build_timestamp": utc_now_iso(),
                }
                break
            except Exception as attempt_exc:
                attempt_logs.append(
                    {
                        "source_type": preferred_source,
                        "source_mix": current_source_mix,
                        "wrong_context_pages": current_wrong_pages,
                        "wrong_context_doc_ids": current_wrong_doc_ids,
                        "teacher_answer": current_teacher_answer,
                        "reason": str(attempt_exc),
                    }
                )
                continue

        if not source_attempted:
            raise ValueError("insufficient_wrong_context_candidates")
        if accepted_sample_record is None:
            if attempt_logs:
                last_attempt = attempt_logs[-1]
                source_type = str(last_attempt.get("source_type") or "") or None
                source_mix = list(last_attempt.get("source_mix") or [])
                wrong_pages = list(last_attempt.get("wrong_context_pages") or [])
                wrong_doc_ids = list(last_attempt.get("wrong_context_doc_ids") or [])
                teacher_answer = last_attempt.get("teacher_answer") if isinstance(last_attempt.get("teacher_answer"), dict) else None
                raise ValueError("no_teacher_validated_wrong_context")
            raise ValueError("insufficient_wrong_context_candidates")

        return {
            "status": "accepted",
            "sample_record": accepted_sample_record,
            "source_type": str(source_type or ""),
            "source_mix": "+".join(source_mix),
            "candidate_pool_source": str(candidate_pool_source or ""),
            "schema": str(accepted_sample_record.get("schema") or ""),
            "support_removed_key": "zero_overlap",
            "inline_same_doc_candidate_collected": inline_same_doc_candidate_collected,
            "inline_candidate_record": inline_candidate_record,
        }
    except Exception as exc:
        reason = str(exc)
        return {
            "status": "rejected",
            "reason": reason,
            "rejected_record": _build_rejected_record(
                filtered_record,
                reason=reason,
                source_type=source_type,
                source_mix=source_mix,
                wrong_pages=wrong_pages,
                wrong_doc_ids=wrong_doc_ids,
                teacher_answer=teacher_answer,
                candidate_pool_source=candidate_pool_source,
                attempts=attempt_logs,
            ),
            "inline_same_doc_candidate_collected": inline_same_doc_candidate_collected,
            "inline_candidate_record": inline_candidate_record,
        }


def _prepare_filtered_record_for_teacher_validation(
    filtered_record: Dict[str, Any],
    *,
    settings: Dict[str, Any],
    cache_by_query_id: Dict[str, Dict[str, Any]],
    candidate_by_query_id: Dict[str, Dict[str, Any]],
    processor_factory: Callable[[], Any],
) -> Dict[str, Any]:
    query_id = str(filtered_record.get("query_id") or "").strip()
    candidate_pool_source: Optional[str] = None
    inline_same_doc_candidate_collected = 0
    inline_candidate_record: Optional[Dict[str, Any]] = None
    same_year_candidate_record: Optional[Dict[str, Any]] = None
    same_year_loaded = False

    try:
        if bool(filtered_record.get("should_refuse", False)):
            raise ValueError("wrong_context_requires_positive_parent")

        parent_answer = filtered_record.get("assistant_response") if isinstance(filtered_record.get("assistant_response"), dict) else {}
        if parent_answer.get("final_answer") in (None, "", [], "N/A"):
            raise ValueError("wrong_context_requires_positive_answer")

        doc_ids = [str(value) for value in filtered_record.get("doc_ids", []) if value not in (None, "")]
        if len(doc_ids) != 1:
            raise ValueError("wrong_context_requires_single_doc")
        primary_doc_id = doc_ids[0]

        cache_record = cache_by_query_id.get(query_id)
        if cache_record is None:
            raise ValueError("missing_retrieved_cache")

        support_items = _select_support_items(
            cache_record,
            primary_doc_id=primary_doc_id,
            max_support_results=settings["max_support_results"],
        )
        if not support_items:
            raise ValueError("missing_support_results")

        candidate_record = candidate_by_query_id.get(query_id)
        candidate_pool_source = "prebuilt_same_doc"
        if candidate_record is None:
            candidate_record = _collect_candidate_record_inline(
                cache_record,
                retrieval_config_path=settings["retrieval_config_path"],
                dataset_root_path=settings["dataset_root_path"],
                candidate_pool_size=settings["candidate_pool_size"],
                per_query_retrieval_top_k=settings["per_query_retrieval_top_k"],
                parallel_requests=settings["parallel_requests"],
            )
            inline_candidate_record = candidate_record
            candidate_pool_source = "inline_same_doc"
            inline_same_doc_candidate_collected += 1

        groups = _partition_same_doc_candidates(
            candidate_record,
            primary_doc_id=primary_doc_id,
            support_items=support_items,
        )

        def source_loader(source_name: str) -> List[Dict[str, Any]]:
            nonlocal candidate_pool_source
            nonlocal same_year_candidate_record
            nonlocal same_year_loaded
            if source_name != "same_year_wrong_company" or same_year_loaded:
                return []
            same_year_loaded = True
            processor = processor_factory()
            same_year_candidate_record = _build_same_year_cross_doc_candidate_record(
                filtered_record,
                cache_record,
                processor,
                primary_doc_id=primary_doc_id,
                candidate_pool_size=settings["candidate_pool_size"],
                per_query_retrieval_top_k=settings["per_query_retrieval_top_k"],
                parallel_requests=settings["parallel_requests"],
                retrieval_config_path=settings["retrieval_config_path"],
                dataset_root_path=settings["dataset_root_path"],
                cross_doc_candidate_doc_cap=settings["cross_doc_candidate_doc_cap"],
            )
            if same_year_candidate_record is None:
                return []
            candidate_pool_source = (
                f"{candidate_pool_source}+inline_same_year_cross_doc"
                if candidate_pool_source
                else "inline_same_year_cross_doc"
            )
            return [
                candidate
                for candidate in (same_year_candidate_record.get("candidates", []) or [])
                if _non_empty_text(candidate.get("text"))
            ]

        prepared_attempts: List[Dict[str, Any]] = []
        source_attempted = False
        support_pages = sorted({int(item.get("page") or 0) for item in support_items if item.get("page") not in (None, "")})
        for preferred_source in settings["source_priority"]:
            composed = _compose_source_candidates(
                preferred_source,
                groups,
                min_context_results=settings["min_context_results"],
                max_context_results=settings["max_context_results"],
                source_loader=source_loader,
            )
            if composed is None:
                continue
            source_attempted = True
            current_candidates, current_source_mix = composed
            current_wrong_context_items = [
                _candidate_to_context_item(candidate, fallback_company_name=filtered_record.get("company_name"))
                for candidate in current_candidates
            ]
            current_wrong_rag_context = _format_context_items(
                current_wrong_context_items,
                max_doc_chars=settings["max_doc_chars"],
                max_context_chars=settings["max_context_chars"],
            )
            current_wrong_pages = sorted(
                {
                    int(item.get("page") or 0)
                    for item in current_wrong_context_items
                    if item.get("page") not in (None, "")
                }
            )
            current_wrong_doc_ids = sorted(
                {
                    str(item.get("doc_id") or "")
                    for item in current_wrong_context_items
                    if str(item.get("doc_id") or "").strip()
                }
            )
            if not current_wrong_rag_context.strip():
                continue
            if set(current_wrong_pages) & set(support_pages):
                continue

            system_prompt = str(filtered_record.get("system_prompt") or "").strip()
            if not system_prompt:
                provider = settings["teacher_answer_provider"]
                system_prompt = build_rag_prompt_bundle(str(filtered_record.get("schema") or ""), provider=provider)[0]
            user_prompt = _build_user_prompt(filtered_record, cache_record, current_wrong_rag_context)

            prepared_attempts.append(
                {
                    "source_type": preferred_source,
                    "source_mix": current_source_mix,
                    "wrong_pages": current_wrong_pages,
                    "wrong_doc_ids": current_wrong_doc_ids,
                    "teacher_input_record": {
                        **filtered_record,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                    },
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "accepted_checks": [
                        "retrieval_present",
                        "schema_pruned",
                        "refusal_sample",
                        "wrong_context_support_removed",
                        "wrong_context_teacher_refused",
                        "wrong_context_candidate_pool",
                    ],
                    "wrong_context_stats": {
                        "support_result_count": len(support_items),
                        "wrong_result_count": len(current_wrong_context_items),
                        "candidate_pool_size": len(candidate_record.get("candidates", []) or []),
                        "same_year_cross_doc_candidate_pool_size": len((same_year_candidate_record or {}).get("candidates", []) or []),
                        "support_pages": support_pages,
                        "wrong_pages": current_wrong_pages,
                        "support_doc_ids": [primary_doc_id],
                        "wrong_doc_ids": current_wrong_doc_ids,
                        "support_overlap_count": 0,
                        "support_removed": True,
                        "source_type": preferred_source,
                        "source_mix": current_source_mix,
                        "source_candidate_counts": {
                            group_name: len(group_candidates)
                            for group_name, group_candidates in groups.items()
                        },
                        "context_hash": stable_hash_int(current_wrong_rag_context),
                    },
                }
            )

        if not source_attempted:
            raise ValueError("insufficient_wrong_context_candidates")

        return {
            "status": "prepared",
            "filtered_record": filtered_record,
            "parent_answer": parent_answer,
            "candidate_pool_source": str(candidate_pool_source or ""),
            "prepared_attempts": prepared_attempts,
            "inline_same_doc_candidate_collected": inline_same_doc_candidate_collected,
            "inline_candidate_record": inline_candidate_record,
        }
    except Exception as exc:
        reason = str(exc)
        return {
            "status": "rejected",
            "reason": reason,
            "rejected_record": _build_rejected_record(
                filtered_record,
                reason=reason,
                source_type=None,
                source_mix=[],
                candidate_pool_source=candidate_pool_source,
            ),
            "inline_same_doc_candidate_collected": inline_same_doc_candidate_collected,
            "inline_candidate_record": inline_candidate_record,
        }


def _validate_prepared_record_with_teacher(
    prepared_record: Dict[str, Any],
    *,
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    filtered_record = prepared_record["filtered_record"]
    parent_answer = prepared_record["parent_answer"]
    api_processor = _thread_local_api_processor(settings["teacher_answer_provider"])
    attempt_logs: List[Dict[str, Any]] = []
    teacher_answer: Optional[Dict[str, Any]] = None
    source_type: Optional[str] = None
    source_mix: List[str] = []
    wrong_pages: List[int] = []
    wrong_doc_ids: List[str] = []

    for attempt in prepared_record.get("prepared_attempts", []) or []:
        source_type = str(attempt.get("source_type") or "") or None
        source_mix = list(attempt.get("source_mix") or [])
        wrong_pages = list(attempt.get("wrong_pages") or [])
        wrong_doc_ids = list(attempt.get("wrong_doc_ids") or [])
        teacher_answer = None
        try:
            teacher_answer = _teacher_refusal_answer(
                api_processor,
                attempt["teacher_input_record"],
                provider=settings["teacher_answer_provider"],
                model=settings["teacher_answer_model"],
                temperature=settings["answer_temperature"],
            )

            final_answer = teacher_answer.get("final_answer")
            if final_answer != "N/A":
                raise ValueError("teacher_answered_under_wrong_context")

            relevant_pages = _normalize_pages(teacher_answer.get("relevant_pages"))
            if relevant_pages and not set(relevant_pages).issubset(set(wrong_pages)):
                raise ValueError("wrong_context_hallucinated_pages")

            accepted_checks = list(attempt.get("accepted_checks", []) or [])
            if relevant_pages and "wrong_context_relevant_pages_subset" not in accepted_checks:
                accepted_checks.append("wrong_context_relevant_pages_subset")

            sample_record = {
                **filtered_record,
                "parent_sample_id": filtered_record.get("sample_id"),
                "parent_query_id": filtered_record.get("query_id"),
                "parent_assistant_response": parent_answer,
                "system_prompt": attempt["system_prompt"],
                "user_prompt": attempt["user_prompt"],
                "assistant_response": teacher_answer,
                "assistant_response_json": compact_json_dumps(teacher_answer),
                "accepted_checks": accepted_checks,
                "source": "wrong_context_refusal",
                "variant_type": "wrong_context_refusal.teacher_validated.v1",
                "should_refuse": True,
                "retrieval_pages": wrong_pages,
                "wrong_context_pages": wrong_pages,
                "context_doc_ids": wrong_doc_ids,
                "wrong_context_source_type": source_type,
                "wrong_context_source_mix": source_mix,
                "candidate_pool_source": prepared_record.get("candidate_pool_source"),
                "wrong_context_stats": dict(attempt.get("wrong_context_stats") or {}),
                "teacher_answer_provider": settings["teacher_answer_provider"],
                "teacher_answer_model": settings["teacher_answer_model"],
                "build_timestamp": utc_now_iso(),
            }
            return {
                "status": "accepted",
                "sample_record": sample_record,
                "source_type": str(source_type or ""),
                "source_mix": "+".join(source_mix),
                "candidate_pool_source": str(prepared_record.get("candidate_pool_source") or ""),
                "schema": str(sample_record.get("schema") or ""),
                "support_removed_key": "zero_overlap",
            }
        except Exception as attempt_exc:
            attempt_logs.append(
                {
                    "source_type": source_type,
                    "source_mix": source_mix,
                    "wrong_context_pages": wrong_pages,
                    "wrong_context_doc_ids": wrong_doc_ids,
                    "teacher_answer": teacher_answer,
                    "reason": str(attempt_exc),
                }
            )

    reason = "no_teacher_validated_wrong_context" if attempt_logs else "insufficient_wrong_context_candidates"
    return {
        "status": "rejected",
        "reason": reason,
        "rejected_record": _build_rejected_record(
            filtered_record,
            reason=reason,
            source_type=source_type,
            source_mix=source_mix,
            wrong_pages=wrong_pages,
            wrong_doc_ids=wrong_doc_ids,
            teacher_answer=teacher_answer,
            candidate_pool_source=prepared_record.get("candidate_pool_source"),
            attempts=attempt_logs,
        ),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = _resolve_settings(args)

    filtered_records = load_records(settings["input_path"])
    if settings["max_queries"] > 0:
        filtered_records = filtered_records[: settings["max_queries"]]
    requested_records = len(filtered_records)
    existing_accepted_ids: set[str] = set()
    existing_accepted_records = 0
    next_sample_index = 0
    backfilled_chat_records = 0
    skipped_existing_accepted = 0
    if settings["resume"]:
        existing_accepted_ids, existing_accepted_records, next_sample_index = _accepted_resume_state(settings["output_path"])
        backfilled_chat_records = _backfill_chat_from_accepted(settings["output_path"], settings["chat_output_path"])
        pending_records = [
            record
            for record in filtered_records
            if str(record.get("query_id") or "").strip() not in existing_accepted_ids
        ]
        skipped_existing_accepted = requested_records - len(pending_records)
        filtered_records = pending_records
    cache_by_query_id = _index_records_by_query_id(load_records(settings["retrieved_cache_input_path"]))
    candidate_by_query_id = (
        _index_records_by_query_id(load_records(settings["candidate_pool_input_path"]))
        if settings["candidate_pool_input_path"] is not None
        else {}
    )
    processor_holder: Dict[str, Any] = {}

    def shared_processor() -> Any:
        processor = processor_holder.get("processor")
        if processor is None:
            processor = build_questions_processor(
                REPO_ROOT,
                settings["retrieval_config_path"],
                dataset_root=settings["dataset_root_path"],
                reasoning_debug_enabled=True,
                parallel_requests=settings["parallel_requests"],
            )
            processor_holder["processor"] = processor
        return processor

    if not settings["resume"]:
        _reset_output_file(settings["output_path"])
        _reset_output_file(settings["chat_output_path"])
    _reset_output_file(settings["rejected_output_path"])
    if not settings["resume"] and settings["candidate_pool_input_path"] is None:
        _reset_output_file(settings["candidate_pool_output_path"])

    accepted_this_run = 0
    rejected_this_run = 0
    inline_same_doc_candidate_collected = 0
    source_type_counter = Counter()
    source_mix_counter = Counter()
    candidate_pool_source_counter = Counter()
    schema_counter = Counter()
    rejection_counter = Counter()
    support_removed_counter = Counter()

    prepared_records: List[Dict[str, Any]] = []
    for filtered_record in filtered_records:
        result = _prepare_filtered_record_for_teacher_validation(
            filtered_record,
            settings=settings,
            cache_by_query_id=cache_by_query_id,
            candidate_by_query_id=candidate_by_query_id,
            processor_factory=shared_processor,
        )
        inline_same_doc_candidate_collected += int(result.get("inline_same_doc_candidate_collected", 0) or 0)
        inline_candidate_record = result.get("inline_candidate_record")
        if inline_candidate_record is not None and settings["candidate_pool_output_path"] is not None:
            append_jsonl(settings["candidate_pool_output_path"], inline_candidate_record)
        if result.get("status") == "prepared":
            if result.get("prepared_attempts"):
                prepared_records.append(result)
                continue
            result = {
                "status": "rejected",
                "reason": "insufficient_wrong_context_candidates",
                "rejected_record": _build_rejected_record(
                    filtered_record,
                    reason="insufficient_wrong_context_candidates",
                    source_type=None,
                    source_mix=[],
                    candidate_pool_source=result.get("candidate_pool_source"),
                ),
            }

        rejected_this_run += 1
        append_jsonl(settings["rejected_output_path"], result["rejected_record"])
        rejection_counter[str(result.get("reason") or "").split(":", maxsplit=1)[0]] += 1

    validation_results = _run_record_tasks(
        prepared_records,
        lambda record: _validate_prepared_record_with_teacher(
            record,
            settings=settings,
        ),
        max_workers=settings["parallel_requests"],
    )

    for result in validation_results:
        if result.get("status") == "accepted":
            accepted_this_run += 1
            sample_record = dict(result["sample_record"])
            sample_record["sample_id"] = f"wrctx-{next_sample_index + accepted_this_run:06d}"
            append_jsonl(settings["output_path"], sample_record)
            if settings["chat_output_path"] is not None:
                append_jsonl(settings["chat_output_path"], build_chat_record(sample_record))

            source_type_counter[str(result.get("source_type") or "")] += 1
            source_mix_counter[str(result.get("source_mix") or "")] += 1
            candidate_pool_source_counter[str(result.get("candidate_pool_source") or "")] += 1
            schema_counter[str(result.get("schema") or "")] += 1
            support_removed_counter[str(result.get("support_removed_key") or "zero_overlap")] += 1
            continue

        rejected_this_run += 1
        append_jsonl(settings["rejected_output_path"], result["rejected_record"])
        rejection_counter[str(result.get("reason") or "").split(":", maxsplit=1)[0]] += 1

    stats = {
        "build_timestamp": utc_now_iso(),
        "config_path": display_path(settings["config_path"], REPO_ROOT),
        "input_path": display_path(settings["input_path"], REPO_ROOT),
        "retrieved_cache_input_path": display_path(settings["retrieved_cache_input_path"], REPO_ROOT),
        "candidate_pool_input_path": display_path(settings["candidate_pool_input_path"], REPO_ROOT),
        "candidate_pool_output_path": display_path(settings["candidate_pool_output_path"], REPO_ROOT),
        "output_path": display_path(settings["output_path"], REPO_ROOT),
        "chat_output_path": display_path(settings["chat_output_path"], REPO_ROOT),
        "rejected_output_path": display_path(settings["rejected_output_path"], REPO_ROOT),
        "retrieval_config_path": display_path(settings["retrieval_config_path"], REPO_ROOT),
        "dataset_root_path": display_path(settings["dataset_root_path"], REPO_ROOT),
        "teacher_answer_provider": settings["teacher_answer_provider"],
        "teacher_answer_model": settings["teacher_answer_model"],
        "answer_temperature": settings["answer_temperature"],
        "candidate_pool_size": settings["candidate_pool_size"],
        "per_query_retrieval_top_k": settings["per_query_retrieval_top_k"],
        "source_priority": settings["source_priority"],
        "min_context_results": settings["min_context_results"],
        "max_context_results": settings["max_context_results"],
        "cross_doc_candidate_doc_cap": settings["cross_doc_candidate_doc_cap"],
        "max_support_results": settings["max_support_results"],
        "max_queries": settings["max_queries"],
        "resume": settings["resume"],
        "existing_accepted_records": existing_accepted_records,
        "skipped_existing_accepted": skipped_existing_accepted,
        "backfilled_chat_records": backfilled_chat_records,
        "pending_records": len(filtered_records),
        "requested_records": requested_records,
        "accepted_records": existing_accepted_records + accepted_this_run,
        "accepted_records_this_run": accepted_this_run,
        "rejected_records": rejected_this_run,
        "rejected_records_this_run": rejected_this_run,
        "inline_same_doc_candidate_collected": inline_same_doc_candidate_collected,
        "source_type_distribution": dict(source_type_counter),
        "source_mix_distribution": dict(source_mix_counter),
        "candidate_pool_source_distribution": dict(candidate_pool_source_counter),
        "schema_distribution": dict(schema_counter),
        "support_removed_distribution": dict(support_removed_counter),
        "rejection_distribution": dict(rejection_counter),
    }
    write_json(settings["stats_output_path"], stats)


if __name__ == "__main__":
    main()
