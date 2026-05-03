from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import (  # noqa: E402
    append_jsonl,
    build_rag_prompt_bundle,
    compact_json_dumps,
    display_path,
    load_records,
    load_yaml_mapping,
    resolve_dataset_root,
    resolve_repo_path,
    stable_hash_int,
    utc_now_iso,
    write_json,
)
from training.generator_sft.scripts.convert_to_chat_sft import build_chat_record  # noqa: E402
from training.reranker_distill.scripts.collect_candidate_pool import (  # noqa: E402
    collect_candidate_pool_record,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build hard-context generator SFT samples using candidate pools and retrieved_cache.")
    parser.add_argument("--config-path", type=Path, default=None, help="YAML config path.")
    parser.add_argument("--input-path", type=Path, default=None, help="Filtered positive sample JSONL path.")
    parser.add_argument("--retrieved-cache-input-path", type=Path, default=None, help="retrieved_cache JSONL path.")
    parser.add_argument("--candidate-pool-input-path", type=Path, default=None, help="Optional prebuilt candidate pool JSONL path.")
    parser.add_argument("--candidate-pool-output-path", type=Path, default=None, help="Optional JSONL path to persist inline-collected candidate pools.")
    parser.add_argument("--output-path", type=Path, default=None, help="Hard-context sample JSONL path.")
    parser.add_argument("--chat-output-path", type=Path, default=None, help="Optional chat-format JSONL output path.")
    parser.add_argument("--rejected-output-path", type=Path, default=None, help="Rejected hard-context sample JSONL path.")
    parser.add_argument("--stats-output-path", type=Path, default=None, help="Stats JSON path.")
    parser.add_argument("--retrieval-config-path", type=Path, default=None, help="Retrieval config path used when collecting candidate pools inline.")
    parser.add_argument("--dataset-root-path", type=Path, default=None, help="Dataset root used when collecting candidate pools inline.")
    parser.add_argument("--candidate-pool-size", type=int, default=None, help="Final deduped candidate cap per query for inline candidate collection.")
    parser.add_argument("--per-query-retrieval-top-k", type=int, default=None, help="Per-query retrieval cap before multi-query merge.")
    parser.add_argument("--parallel-requests", type=int, default=None, help="Forwarded to inline candidate collection.")
    parser.add_argument("--min-distractors", type=int, default=None, help="Minimum number of distractors required to keep a sample.")
    parser.add_argument("--max-distractors", type=int, default=None, help="Maximum number of distractors to inject.")
    parser.add_argument("--max-support-results", type=int, default=None, help="Maximum number of support chunks preserved from retrieved_cache.")
    parser.add_argument("--max-context-chars", type=int, default=None, help="Maximum context characters for the regenerated hard context.")
    parser.add_argument("--max-doc-chars", type=int, default=None, help="Maximum per-chunk characters for the regenerated hard context.")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional cap over filtered input records.")
    return parser


def _coalesce(cli_value: Any, config_value: Any, default: Any = None) -> Any:
    return cli_value if cli_value is not None else (config_value if config_value is not None else default)


def _reset_output_file(path: Optional[Path]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _index_records_by_query_id(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for record in records:
        query_id = str(record.get("query_id") or "").strip()
        if query_id:
            indexed[query_id] = record
    return indexed


def _pages_from_values(values: Any) -> List[int]:
    pages: List[int] = []
    for value in values or []:
        try:
            pages.append(int(value))
        except (TypeError, ValueError):
            continue
    return pages


def _support_identity(item: Dict[str, Any], primary_doc_id: str) -> Tuple[str, int, str]:
    return (
        primary_doc_id,
        int(item.get("page") or 0),
        str(item.get("chunk_id") or ""),
    )


def _candidate_identity(candidate: Dict[str, Any]) -> Tuple[str, int, str]:
    return (
        str(candidate.get("doc_id") or ""),
        int(candidate.get("page") or 0),
        str(candidate.get("chunk_id") or ""),
    )


def _serialized_result_to_context_item(item: Dict[str, Any], primary_doc_id: str) -> Dict[str, Any]:
    return {
        "doc_id": primary_doc_id,
        "page": item.get("page"),
        "chunk_id": item.get("chunk_id"),
        "section_name": item.get("section_name") or item.get("section_title"),
        "chunk_type": item.get("chunk_type"),
        "node_type": item.get("node_type"),
        "matched_tags": list(item.get("matched_tags", []) or []),
        "text": str(item.get("text") or ""),
        "retrieval_sources": list(item.get("retrieval_sources", []) or []),
    }


def _candidate_to_context_item(candidate: Dict[str, Any], fallback_company_name: Optional[str] = None) -> Dict[str, Any]:
    return {
        "doc_id": str(candidate.get("doc_id") or ""),
        "page": candidate.get("page"),
        "chunk_id": candidate.get("chunk_id"),
        "section_name": candidate.get("section_name"),
        "chunk_type": candidate.get("chunk_type"),
        "node_type": candidate.get("node_type"),
        "matched_tags": list(candidate.get("topic_flags", []) or []),
        "text": str(candidate.get("text") or ""),
        "retrieval_sources": list(candidate.get("retrieval_sources", []) or []),
        "company_name": candidate.get("company_name") or fallback_company_name,
    }


def _format_context_items(
    items: Iterable[Dict[str, Any]],
    *,
    max_doc_chars: int,
    max_context_chars: int,
) -> str:
    context_parts: List[str] = []
    total_chars = 0
    for item in items:
        text = str(item.get("text") or "")
        if not text.strip():
            continue

        page_number = item.get("page")
        section_name = item.get("section_name")
        chunk_type = item.get("chunk_type")
        node_type = item.get("node_type")
        matched_tags = list(item.get("matched_tags", []) or [])
        if max_doc_chars > 0 and len(text) > max_doc_chars:
            text = text[: max_doc_chars].rstrip() + "\n...[truncated]"

        label = f"Text retrieved from page {page_number}"
        if section_name:
            label += f" | section: {section_name}"
        if chunk_type:
            label += f" | chunk_type: {chunk_type}"
        if node_type:
            label += f" | node_type: {node_type}"
        if matched_tags:
            label += f" | matched_tags: {', '.join(matched_tags)}"
        part = f'{label}: \n"""\n{text}\n"""'
        if max_context_chars > 0 and total_chars + len(part) > max_context_chars:
            remaining = max_context_chars - total_chars
            if remaining <= 0:
                break
            part = part[:remaining].rstrip() + "\n...[truncated]"

        context_parts.append(part)
        total_chars += len(part)
        if max_context_chars > 0 and total_chars >= max_context_chars:
            break

    return "\n\n---\n\n".join(context_parts)


def _select_support_items(
    cache_record: Dict[str, Any],
    *,
    primary_doc_id: str,
    max_support_results: int,
) -> List[Dict[str, Any]]:
    retrieval_results = list(cache_record.get("retrieval_results", []) or [])
    if not retrieval_results:
        return []

    teacher_signal = cache_record.get("teacher_signal") if isinstance(cache_record.get("teacher_signal"), dict) else {}
    relevant_pages = set(_pages_from_values(teacher_signal.get("relevant_pages")))

    selected: List[Dict[str, Any]] = []
    for item in retrieval_results:
        page = item.get("page")
        if relevant_pages and page not in relevant_pages:
            continue
        selected.append(_serialized_result_to_context_item(item, primary_doc_id))

    if not selected:
        selected = [
            _serialized_result_to_context_item(item, primary_doc_id)
            for item in retrieval_results[:max(1, max_support_results)]
        ]

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in selected:
        key = _support_identity(item, primary_doc_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= max_support_results:
            break
    return deduped


def _candidate_priority(candidate: Dict[str, Any], primary_doc_id: str) -> Tuple[int, float, int]:
    doc_id = str(candidate.get("doc_id") or "")
    same_doc = 0 if doc_id == primary_doc_id else 1
    score = float(candidate.get("base_score") or 0.0)
    hit_count = int(candidate.get("query_hit_count") or 0)
    return (same_doc, -score, -hit_count)


def _select_distractors(
    candidate_record: Dict[str, Any],
    *,
    primary_doc_id: str,
    support_items: List[Dict[str, Any]],
    min_distractors: int,
    max_distractors: int,
    fallback_company_name: Optional[str],
) -> List[Dict[str, Any]]:
    support_keys = {_support_identity(item, primary_doc_id) for item in support_items}
    candidates = list(candidate_record.get("candidates", []) or [])
    candidates.sort(key=lambda item: _candidate_priority(item, primary_doc_id))

    distractors: List[Dict[str, Any]] = []
    seen = set()
    for candidate in candidates:
        key = _candidate_identity(candidate)
        if key in seen or key in support_keys:
            continue
        if not str(candidate.get("text") or "").strip():
            continue
        seen.add(key)
        distractors.append(_candidate_to_context_item(candidate, fallback_company_name=fallback_company_name))
        if len(distractors) >= max_distractors:
            break

    if len(distractors) < min_distractors:
        raise ValueError(f"insufficient_distractors:{len(distractors)}<{min_distractors}")
    return distractors


def _merge_support_and_distractors(
    support_items: List[Dict[str, Any]],
    distractors: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not distractors:
        return list(support_items)
    merged: List[Dict[str, Any]] = [distractors[0]]
    merged.extend(support_items)
    merged.extend(distractors[1:])
    return merged


def _collect_candidate_record_inline(
    cache_record: Dict[str, Any],
    *,
    retrieval_config_path: Path,
    dataset_root_path: Path,
    candidate_pool_size: int,
    per_query_retrieval_top_k: int,
    parallel_requests: int,
) -> Dict[str, Any]:
    return collect_candidate_pool_record(
        cache_record,
        retrieval_config_path=retrieval_config_path,
        dataset_root_path=dataset_root_path,
        candidate_pool_size=candidate_pool_size,
        per_query_retrieval_top_k=per_query_retrieval_top_k,
        parallel_requests=parallel_requests,
    )


def _build_rejected_record(record: Dict[str, Any], reason: str) -> Dict[str, Any]:
    assistant_response = record.get("assistant_response") if isinstance(record.get("assistant_response"), dict) else {}
    return {
        "sample_id": record.get("sample_id"),
        "query_id": record.get("query_id"),
        "schema": record.get("schema"),
        "company_name": record.get("company_name"),
        "doc_ids": list(record.get("doc_ids", []) or []),
        "final_answer": assistant_response.get("final_answer"),
        "reason": reason,
        "build_timestamp": utc_now_iso(),
    }


def _build_user_prompt(
    filtered_record: Dict[str, Any],
    cache_record: Dict[str, Any],
    hard_rag_context: str,
) -> str:
    original_user_prompt = str(filtered_record.get("user_prompt") or "")
    original_rag_context = str(cache_record.get("rag_context") or "")
    if original_user_prompt and original_rag_context and original_rag_context in original_user_prompt:
        return original_user_prompt.replace(original_rag_context, hard_rag_context, 1)

    provider = str(cache_record.get("teacher_answer_provider") or "qwen")
    schema = str(filtered_record.get("schema") or "")
    question_text = str(filtered_record.get("question_text") or "")
    try:
        _, user_prompt_template, _ = build_rag_prompt_bundle(schema, provider=provider)
        return user_prompt_template.format(
            context=hard_rag_context,
            question=question_text,
        )
    except Exception:
        return f"Context:\n{hard_rag_context}\n\nQuestion:\n{question_text}"


def _build_hard_context_sample(
    filtered_record: Dict[str, Any],
    cache_record: Dict[str, Any],
    candidate_record: Dict[str, Any],
    *,
    sample_index: int,
    min_distractors: int,
    max_distractors: int,
    max_support_results: int,
    max_doc_chars: int,
    max_context_chars: int,
) -> Dict[str, Any]:
    doc_ids = [str(item) for item in filtered_record.get("doc_ids", []) if item not in (None, "")]
    if len(doc_ids) != 1:
        raise ValueError("hard_context_requires_single_doc")
    if bool(filtered_record.get("should_refuse", False)):
        raise ValueError("hard_context_skips_refusal_samples")

    assistant_response = filtered_record.get("assistant_response") if isinstance(filtered_record.get("assistant_response"), dict) else {}
    final_answer = assistant_response.get("final_answer")
    if final_answer in (None, "", "N/A"):
        raise ValueError("hard_context_requires_positive_answer")

    primary_doc_id = doc_ids[0]
    support_items = _select_support_items(
        cache_record,
        primary_doc_id=primary_doc_id,
        max_support_results=max_support_results,
    )
    if not support_items:
        raise ValueError("missing_support_results")

    distractors = _select_distractors(
        candidate_record,
        primary_doc_id=primary_doc_id,
        support_items=support_items,
        min_distractors=min_distractors,
        max_distractors=max_distractors,
        fallback_company_name=filtered_record.get("company_name"),
    )
    merged_items = _merge_support_and_distractors(support_items, distractors)
    hard_rag_context = _format_context_items(
        merged_items,
        max_doc_chars=max_doc_chars,
        max_context_chars=max_context_chars,
    )
    if not hard_rag_context.strip():
        raise ValueError("empty_hard_context")

    system_prompt = str(filtered_record.get("system_prompt") or "").strip()
    if not system_prompt:
        try:
            provider = str(cache_record.get("teacher_answer_provider") or "qwen")
            system_prompt, _, _ = build_rag_prompt_bundle(str(filtered_record.get("schema") or ""), provider=provider)
        except Exception:
            system_prompt = "You are a financial annual-report QA assistant."
    user_prompt = _build_user_prompt(filtered_record, cache_record, hard_rag_context)

    support_pages = sorted({int(item.get("page") or 0) for item in support_items if item.get("page") not in (None, "")})
    distractor_pages = sorted({int(item.get("page") or 0) for item in distractors if item.get("page") not in (None, "")})
    distractor_doc_ids = sorted({str(item.get("doc_id") or "") for item in distractors if str(item.get("doc_id") or "").strip()})
    accepted_checks = list(filtered_record.get("accepted_checks", []) or [])
    for check in (
        "hard_context_support_preserved",
        "hard_context_distractors_added",
        "hard_context_candidate_pool",
    ):
        if check not in accepted_checks:
            accepted_checks.append(check)

    sample_record = {
        **filtered_record,
        "sample_id": f"hardctx-{sample_index:06d}",
        "parent_sample_id": filtered_record.get("sample_id"),
        "parent_query_id": filtered_record.get("query_id"),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "assistant_response_json": compact_json_dumps(assistant_response),
        "accepted_checks": accepted_checks,
        "source": "hard_context_positive",
        "variant_type": "hard_context_with_distractors.v1",
        "hard_context_stats": {
            "support_result_count": len(support_items),
            "distractor_result_count": len(distractors),
            "candidate_pool_size": len(candidate_record.get("candidates", []) or []),
            "support_pages": support_pages,
            "distractor_pages": distractor_pages,
            "distractor_doc_ids": distractor_doc_ids,
            "context_result_count": len(merged_items),
            "context_hash": stable_hash_int(hard_rag_context),
        },
        "retrieval_pages": support_pages,
        "hard_context_pages": sorted({int(item.get("page") or 0) for item in merged_items if item.get("page") not in (None, "")}),
        "context_doc_ids": sorted({str(item.get("doc_id") or "") for item in merged_items if str(item.get("doc_id") or "").strip()}),
        "build_timestamp": utc_now_iso(),
    }
    return sample_record


def _resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    default_config_path = REPO_ROOT / "training/generator_sft/configs/hard_context.example.yaml"
    config_path = args.config_path or (default_config_path if default_config_path.exists() else None)
    config = load_yaml_mapping(config_path)

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
        if retrieval_config_path is None:
            raise FileNotFoundError(f"Missing candidate pool file: {candidate_pool_input_path}")
        candidate_pool_input_path = None
    if input_path is None or retrieved_cache_input_path is None or output_path is None or rejected_output_path is None or stats_output_path is None:
        raise ValueError("input/retrieved-cache/output/rejected/stats paths are required.")
    if candidate_pool_input_path is None and retrieval_config_path is None:
        raise ValueError("retrieval_config_path is required when candidate_pool_input_path is not provided.")

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
        "candidate_pool_size": max(1, int(_coalesce(args.candidate_pool_size, config.get("candidate_pool_size"), 32))),
        "per_query_retrieval_top_k": max(1, int(_coalesce(args.per_query_retrieval_top_k, config.get("per_query_retrieval_top_k"), 24))),
        "parallel_requests": max(1, int(_coalesce(args.parallel_requests, config.get("parallel_requests"), 1))),
        "min_distractors": max(1, int(_coalesce(args.min_distractors, config.get("min_distractors"), 1))),
        "max_distractors": max(1, int(_coalesce(args.max_distractors, config.get("max_distractors"), 3))),
        "max_support_results": max(1, int(_coalesce(args.max_support_results, config.get("max_support_results"), 3))),
        "max_context_chars": max(1, int(_coalesce(args.max_context_chars, config.get("max_context_chars"), 12000))),
        "max_doc_chars": max(1, int(_coalesce(args.max_doc_chars, config.get("max_doc_chars"), 1800))),
        "max_queries": max(0, int(_coalesce(args.max_queries, config.get("max_queries"), 0) or 0)),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = _resolve_settings(args)
    filtered_records = load_records(settings["input_path"])
    if settings["max_queries"] > 0:
        filtered_records = filtered_records[: settings["max_queries"]]
    cache_records = load_records(settings["retrieved_cache_input_path"])
    candidate_records = load_records(settings["candidate_pool_input_path"]) if settings["candidate_pool_input_path"] else []

    cache_by_query_id = _index_records_by_query_id(cache_records)
    candidate_by_query_id = _index_records_by_query_id(candidate_records)

    _reset_output_file(settings["output_path"])
    _reset_output_file(settings["chat_output_path"])
    _reset_output_file(settings["rejected_output_path"])
    if settings["candidate_pool_input_path"] is None:
        _reset_output_file(settings["candidate_pool_output_path"])

    accepted = 0
    rejected = 0
    sample_index = 0
    inline_candidate_collected = 0
    candidate_source_counter = Counter()
    schema_counter = Counter()
    rejection_counter = Counter()
    distractor_counter = Counter()

    for filtered_record in filtered_records:
        query_id = str(filtered_record.get("query_id") or "").strip()
        try:
            cache_record = cache_by_query_id.get(query_id)
            if cache_record is None:
                raise ValueError("missing_retrieved_cache")

            candidate_record = candidate_by_query_id.get(query_id)
            candidate_source = "prebuilt"
            if candidate_record is None:
                if settings["retrieval_config_path"] is None:
                    raise ValueError("missing_candidate_pool")
                candidate_record = _collect_candidate_record_inline(
                    cache_record,
                    retrieval_config_path=settings["retrieval_config_path"],
                    dataset_root_path=settings["dataset_root_path"],
                    candidate_pool_size=settings["candidate_pool_size"],
                    per_query_retrieval_top_k=settings["per_query_retrieval_top_k"],
                    parallel_requests=settings["parallel_requests"],
                )
                candidate_source = "inline"
                inline_candidate_collected += 1
                if settings["candidate_pool_output_path"] is not None:
                    append_jsonl(settings["candidate_pool_output_path"], candidate_record)
                candidate_by_query_id[query_id] = candidate_record

            sample_record = _build_hard_context_sample(
                filtered_record,
                cache_record,
                candidate_record,
                sample_index=sample_index + 1,
                min_distractors=settings["min_distractors"],
                max_distractors=settings["max_distractors"],
                max_support_results=settings["max_support_results"],
                max_doc_chars=settings["max_doc_chars"],
                max_context_chars=settings["max_context_chars"],
            )
            sample_record["candidate_pool_source"] = candidate_source
            append_jsonl(settings["output_path"], sample_record)
            if settings["chat_output_path"] is not None:
                append_jsonl(settings["chat_output_path"], build_chat_record(sample_record))

            sample_index += 1
            accepted += 1
            candidate_source_counter[candidate_source] += 1
            schema_counter[str(sample_record.get("schema") or "")] += 1
            distractor_count = int((sample_record.get("hard_context_stats") or {}).get("distractor_result_count") or 0)
            distractor_counter[str(distractor_count)] += 1
        except Exception as exc:
            append_jsonl(settings["rejected_output_path"], _build_rejected_record(filtered_record, str(exc)))
            rejected += 1
            rejection_counter[str(exc).split(":", maxsplit=1)[0]] += 1

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
        "candidate_pool_size": settings["candidate_pool_size"],
        "per_query_retrieval_top_k": settings["per_query_retrieval_top_k"],
        "min_distractors": settings["min_distractors"],
        "max_distractors": settings["max_distractors"],
        "max_support_results": settings["max_support_results"],
        "max_queries": settings["max_queries"],
        "requested_records": len(filtered_records),
        "accepted_records": accepted,
        "rejected_records": rejected,
        "inline_candidate_collected": inline_candidate_collected,
        "candidate_pool_source_distribution": dict(candidate_source_counter),
        "schema_distribution": dict(schema_counter),
        "distractor_count_distribution": dict(distractor_counter),
        "rejection_distribution": dict(rejection_counter),
    }
    write_json(settings["stats_output_path"], stats)


if __name__ == "__main__":
    main()
