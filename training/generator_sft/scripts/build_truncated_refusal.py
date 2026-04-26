from __future__ import annotations

import argparse
import copy
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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
    prune_answer_to_schema,
    resolve_repo_path,
    stable_hash_int,
    utc_now_iso,
    write_json,
)
from training.generator_sft.scripts.build_hard_context_samples import (  # noqa: E402
    _build_user_prompt,
    _format_context_items,
)
from training.generator_sft.scripts.convert_to_chat_sft import build_chat_record  # noqa: E402
from training.generator_sft.validators import (  # noqa: E402
    build_reject_log,
    classify_boolean_context,
    classify_name_support,
    compact_snippet,
    get_answer_dict,
    normalize_pages,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build refusal samples by truncating key support from answerable parents.")
    parser.add_argument("--config-path", type=Path, default=None, help="YAML config path.")
    parser.add_argument("--input-path", type=Path, default=None, help="Filtered positive sample JSONL path.")
    parser.add_argument("--retrieved-cache-input-path", type=Path, default=None, help="retrieved_cache JSONL path.")
    parser.add_argument("--anchor-input-path", type=Path, default=None, help="anchor_clean_positive JSONL path.")
    parser.add_argument("--output-path", type=Path, default=None, help="Accepted truncated refusal JSONL path.")
    parser.add_argument("--chat-output-path", type=Path, default=None, help="Optional chat-format JSONL path.")
    parser.add_argument("--rejected-output-path", type=Path, default=None, help="Rejected truncated refusal JSONL path.")
    parser.add_argument("--stats-output-path", type=Path, default=None, help="Stats JSON path.")
    parser.add_argument("--teacher-answer-provider", default=None, help="Optional teacher provider for refusal validation.")
    parser.add_argument("--teacher-answer-model", default=None, help="Optional teacher model for refusal validation.")
    parser.add_argument("--answer-temperature", type=float, default=None, help="Teacher sampling temperature.")
    parser.add_argument("--source-priority", nargs="*", default=None, help="Variant order to attempt.")
    parser.add_argument("--max-context-results", type=int, default=None, help="Maximum results preserved in truncated context.")
    parser.add_argument("--max-context-chars", type=int, default=None, help="Maximum context chars.")
    parser.add_argument("--max-doc-chars", type=int, default=None, help="Maximum chars per chunk.")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional cap over input records.")
    return parser


def _coalesce(cli_value: Any, config_value: Any, default: Any = None) -> Any:
    return cli_value if cli_value is not None else (config_value if config_value is not None else default)


def _reset_output_file(path: Optional[Path]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _index_by_query_id(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for record in records:
        query_id = str(record.get("query_id") or "").strip()
        if query_id:
            indexed[query_id] = record
    return indexed


def _result_identity(item: Dict[str, Any]) -> Tuple[int, str]:
    return (int(item.get("page") or 0), str(item.get("chunk_id") or ""))


def _result_to_context_item(item: Dict[str, Any], fallback_doc_id: str) -> Dict[str, Any]:
    return {
        "doc_id": fallback_doc_id,
        "page": item.get("page"),
        "chunk_id": item.get("chunk_id"),
        "section_name": item.get("section_name") or item.get("section_title"),
        "chunk_type": item.get("chunk_type"),
        "node_type": item.get("node_type"),
        "matched_tags": list(item.get("matched_tags", []) or []),
        "text": str(item.get("text") or ""),
        "retrieval_sources": list(item.get("retrieval_sources", []) or []),
    }


def _anchor_chunk_keys(anchor_record: Dict[str, Any]) -> set[Tuple[int, str]]:
    keys = set()
    for evidence in anchor_record.get("anchor_evidence", []) or []:
        try:
            page = int(evidence.get("page") or 0)
        except (TypeError, ValueError):
            page = 0
        keys.add((page, str(evidence.get("chunk_id") or "")))
    return keys


def _anchor_pages(anchor_record: Dict[str, Any], cache_record: Dict[str, Any]) -> List[int]:
    pages = normalize_pages(anchor_record.get("anchor_pages"))
    if pages:
        return pages
    teacher_signal = cache_record.get("teacher_signal") if isinstance(cache_record.get("teacher_signal"), dict) else {}
    return normalize_pages(teacher_signal.get("relevant_pages"))


def _truncate_number_support_text(text: str, anchor_record: Dict[str, Any]) -> str:
    grounding = anchor_record.get("table_grounding") if isinstance(anchor_record.get("table_grounding"), dict) else {}
    normalized_value = str(grounding.get("normalized_value") or anchor_record.get("final_answer") or "").strip()
    table_id = str(grounding.get("table_id") or "")
    lines = []
    for line in str(text or "").splitlines():
        if normalized_value and normalized_value in line:
            continue
        if table_id and table_id in line:
            continue
        lines.append(line)
    truncated = "\n".join(line for line in lines if line.strip()).strip()
    if truncated == str(text or "").strip():
        truncated = "\n".join(lines[:2]).strip()
    return truncated


def _variant_top_k(
    retrieval_results: List[Dict[str, Any]],
    *,
    k: int,
    max_context_results: int,
) -> List[Dict[str, Any]]:
    return list(retrieval_results[: min(k, max_context_results)])


def _variant_key_page_removed(
    retrieval_results: List[Dict[str, Any]],
    *,
    anchor_record: Dict[str, Any],
    cache_record: Dict[str, Any],
    max_context_results: int,
) -> List[Dict[str, Any]]:
    support_pages = set(_anchor_pages(anchor_record, cache_record))
    support_keys = _anchor_chunk_keys(anchor_record)
    kept: List[Dict[str, Any]] = []
    for item in retrieval_results:
        key = _result_identity(item)
        if int(item.get("page") or 0) in support_pages:
            continue
        if key in support_keys:
            continue
        kept.append(item)
        if len(kept) >= max_context_results:
            break
    return kept


def _variant_table_support_truncated(
    retrieval_results: List[Dict[str, Any]],
    *,
    anchor_record: Dict[str, Any],
    cache_record: Dict[str, Any],
    max_context_results: int,
) -> List[Dict[str, Any]]:
    support_pages = set(_anchor_pages(anchor_record, cache_record))
    variant_results: List[Dict[str, Any]] = []
    for item in retrieval_results[: max_context_results]:
        cloned = copy.deepcopy(item)
        if int(cloned.get("page") or 0) in support_pages:
            cloned["text"] = _truncate_number_support_text(str(cloned.get("text") or ""), anchor_record)
        if str(cloned.get("text") or "").strip():
            variant_results.append(cloned)
    return variant_results


def _build_variant_results(
    source_type: str,
    retrieval_results: List[Dict[str, Any]],
    *,
    anchor_record: Dict[str, Any],
    cache_record: Dict[str, Any],
    max_context_results: int,
) -> List[Dict[str, Any]]:
    if source_type == "top1_truncation":
        return _variant_top_k(retrieval_results, k=1, max_context_results=max_context_results)
    if source_type == "top2_truncation":
        return _variant_top_k(retrieval_results, k=2, max_context_results=max_context_results)
    if source_type == "key_page_removed":
        return _variant_key_page_removed(
            retrieval_results,
            anchor_record=anchor_record,
            cache_record=cache_record,
            max_context_results=max_context_results,
        )
    if source_type == "table_support_truncated":
        return _variant_table_support_truncated(
            retrieval_results,
            anchor_record=anchor_record,
            cache_record=cache_record,
            max_context_results=max_context_results,
        )
    raise ValueError(f"unsupported_source_type:{source_type}")


def _local_validate_truncated_refusal(
    filtered_record: Dict[str, Any],
    anchor_record: Dict[str, Any],
    variant_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    schema = str(filtered_record.get("schema") or "")
    variant_pages = normalize_pages([item.get("page") for item in variant_results])
    synthetic_record = {
        "query_id": filtered_record.get("query_id"),
        "question_text": filtered_record.get("question_text"),
        "schema": schema,
        "doc_ids": list(filtered_record.get("doc_ids", []) or []),
        "retrieval_results": variant_results,
        "retrieval_pages": variant_pages,
        "answer": {
            "final_answer": "N/A",
            "relevant_pages": variant_pages,
        },
    }

    if schema == "number":
        anchor_value = str(anchor_record.get("final_answer") or "")
        remaining_text = "\n".join(str(item.get("text") or "") for item in variant_results)
        if anchor_value and anchor_value in remaining_text:
            return {
                "accepted": False,
                "reason": "number_anchor_value_still_present",
                "support_pages": variant_pages,
            }
        if set(variant_pages) & set(normalize_pages(anchor_record.get("anchor_pages"))):
            return {
                "accepted": False,
                "reason": "number_anchor_pages_still_present",
                "support_pages": variant_pages,
            }
        return {
            "accepted": True,
            "reason": "number_support_removed",
            "support_pages": variant_pages,
        }

    if schema == "name":
        support = classify_name_support(
            synthetic_record,
            answer_value=anchor_record.get("final_answer"),
            anchor_record=anchor_record,
        )
        if support["support_type"] != "none":
            return {
                "accepted": False,
                "reason": f"name_support_still_visible:{support['support_type']}",
                "support_pages": support["support_pages"],
            }
        return {
            "accepted": True,
            "reason": "name_support_removed",
            "support_pages": support["support_pages"],
        }

    if schema == "boolean":
        classification = classify_boolean_context(
            synthetic_record,
            anchor_record=anchor_record,
        )
        if classification["classification"] in {"explicit_positive", "explicit_negative"}:
            return {
                "accepted": False,
                "reason": f"boolean_still_answerable:{classification['classification']}",
                "support_pages": classification["support_pages"],
            }
        return {
            "accepted": True,
            "reason": f"boolean_{classification['classification']}",
            "support_pages": classification["support_pages"],
        }

    return {
        "accepted": False,
        "reason": "unsupported_schema_for_truncation",
        "support_pages": variant_pages,
    }


def _teacher_validate_refusal(
    api_processor: Any,
    filtered_record: Dict[str, Any],
    cache_record: Dict[str, Any],
    truncated_context: str,
    variant_pages: List[int],
    *,
    provider: str,
    model: str,
    temperature: float,
) -> Dict[str, Any]:
    schema = str(filtered_record.get("schema") or "")
    system_prompt = str(filtered_record.get("system_prompt") or "").strip()
    if not system_prompt:
        system_prompt = build_rag_prompt_bundle(schema, provider=provider)[0]
    user_prompt = _build_user_prompt(filtered_record, cache_record, truncated_context)
    response_format = build_rag_prompt_bundle(schema, provider=provider)[2]
    raw_answer = api_processor.send_message(
        model=model,
        temperature=temperature,
        system_content=system_prompt,
        human_content=user_prompt,
        is_structured=True,
        response_format=response_format,
    )
    answer = prune_answer_to_schema(raw_answer, schema=schema, provider=provider)
    if answer.get("final_answer") != "N/A":
        raise ValueError("teacher_answered_after_truncation")
    relevant_pages = normalize_pages(answer.get("relevant_pages"))
    if relevant_pages and not set(relevant_pages).issubset(set(variant_pages)):
        raise ValueError("teacher_refusal_hallucinated_pages")
    return {
        "assistant_response": answer,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }


def _build_refusal_answer(schema: str, pages: Sequence[int], source_type: str) -> Dict[str, Any]:
    page_list = list(pages or [])[:2]
    return {
        "step_by_step_analysis": "1. 当前上下文是截断后的片段。\n2. 原本用于回答问题的关键证据已被删除或截断。\n3. 现有内容不足以直接支持结论，因此应返回 N/A。",
        "reasoning_summary": f"截断后的上下文不再包含回答该问题所需的关键证据，{source_type} 变体应拒答。",
        "relevant_pages": page_list,
        "final_answer": "N/A",
    }


def _resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    default_config_path = REPO_ROOT / "training/generator_sft/configs/truncated_refusal.example.yaml"
    config_path = args.config_path or (default_config_path if default_config_path.exists() else None)
    config = load_yaml_mapping(config_path)

    input_path = resolve_repo_path(REPO_ROOT, _coalesce(args.input_path, config.get("input_path")))
    retrieved_cache_input_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(args.retrieved_cache_input_path, config.get("retrieved_cache_input_path")),
    )
    anchor_input_path = resolve_repo_path(REPO_ROOT, _coalesce(args.anchor_input_path, config.get("anchor_input_path")))
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
    if input_path is None or retrieved_cache_input_path is None or anchor_input_path is None or output_path is None or rejected_output_path is None or stats_output_path is None:
        raise ValueError("input/retrieved-cache/anchor/output/rejected/stats paths are required.")

    source_priority = [
        str(value)
        for value in (
            _coalesce(
                args.source_priority,
                config.get("source_priority"),
                ["key_page_removed", "top1_truncation", "top2_truncation", "table_support_truncated"],
            )
            or []
        )
        if str(value).strip()
    ]

    return {
        "config_path": config_path,
        "input_path": input_path,
        "retrieved_cache_input_path": retrieved_cache_input_path,
        "anchor_input_path": anchor_input_path,
        "output_path": output_path,
        "chat_output_path": chat_output_path,
        "rejected_output_path": rejected_output_path,
        "stats_output_path": stats_output_path,
        "teacher_answer_provider": _coalesce(args.teacher_answer_provider, config.get("teacher_answer_provider")),
        "teacher_answer_model": _coalesce(args.teacher_answer_model, config.get("teacher_answer_model")),
        "answer_temperature": float(_coalesce(args.answer_temperature, config.get("answer_temperature"), 0.0)),
        "source_priority": source_priority,
        "max_context_results": max(1, int(_coalesce(args.max_context_results, config.get("max_context_results"), 3))),
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
    cache_by_query_id = _index_by_query_id(load_records(settings["retrieved_cache_input_path"]))
    anchor_by_query_id = _index_by_query_id(load_records(settings["anchor_input_path"]))
    api_processor = (
        None
    )
    if settings["teacher_answer_provider"] and settings["teacher_answer_model"]:
        from src.api_requests import APIProcessor  # noqa: E402

        api_processor = APIProcessor(provider=str(settings["teacher_answer_provider"]))

    _reset_output_file(settings["output_path"])
    _reset_output_file(settings["chat_output_path"])
    _reset_output_file(settings["rejected_output_path"])

    accepted = 0
    rejected = 0
    source_counter = Counter()
    schema_counter = Counter()
    validation_counter = Counter()
    rejection_counter = Counter()

    for filtered_record in filtered_records:
        query_id = str(filtered_record.get("query_id") or "").strip()
        parent_answer = get_answer_dict(filtered_record)
        source_type = None
        local_validation = None
        teacher_validation = None
        try:
            if bool(filtered_record.get("should_refuse", False)):
                raise ValueError("truncated_requires_positive_parent")
            if parent_answer.get("final_answer") in (None, "", [], "N/A"):
                raise ValueError("truncated_requires_positive_answer")

            cache_record = cache_by_query_id.get(query_id)
            if cache_record is None:
                raise ValueError("missing_retrieved_cache")
            anchor_record = anchor_by_query_id.get(query_id)
            if anchor_record is None:
                raise ValueError("missing_anchor_record")

            retrieval_results = list(cache_record.get("retrieval_results", []) or [])
            if not retrieval_results:
                raise ValueError("missing_retrieval_results")

            doc_ids = [str(item) for item in filtered_record.get("doc_ids", []) if item not in (None, "")]
            if len(doc_ids) != 1:
                raise ValueError("truncated_requires_single_doc")
            primary_doc_id = doc_ids[0]

            accepted_sample = None
            attempt_logs = []
            for preferred_source in settings["source_priority"]:
                if preferred_source == "table_support_truncated" and str(filtered_record.get("schema") or "") != "number":
                    continue

                source_type = preferred_source
                variant_results = _build_variant_results(
                    preferred_source,
                    retrieval_results,
                    anchor_record=anchor_record,
                    cache_record=cache_record,
                    max_context_results=settings["max_context_results"],
                )
                if not variant_results:
                    attempt_logs.append({"source_type": preferred_source, "reason": "empty_variant_results"})
                    continue

                context_items = [_result_to_context_item(item, primary_doc_id) for item in variant_results]
                truncated_context = _format_context_items(
                    context_items,
                    max_doc_chars=settings["max_doc_chars"],
                    max_context_chars=settings["max_context_chars"],
                )
                if not truncated_context.strip():
                    attempt_logs.append({"source_type": preferred_source, "reason": "empty_truncated_context"})
                    continue

                variant_pages = normalize_pages([item.get("page") for item in variant_results])
                local_validation = _local_validate_truncated_refusal(
                    filtered_record,
                    anchor_record,
                    variant_results,
                )
                if not local_validation["accepted"]:
                    attempt_logs.append({"source_type": preferred_source, "reason": local_validation["reason"]})
                    continue

                assistant_response = _build_refusal_answer(
                    str(filtered_record.get("schema") or ""),
                    variant_pages,
                    preferred_source,
                )
                system_prompt = str(filtered_record.get("system_prompt") or "").strip()
                user_prompt = _build_user_prompt(filtered_record, cache_record, truncated_context)
                validation_mode = "validator_only"
                if api_processor is not None:
                    teacher_validation = _teacher_validate_refusal(
                        api_processor,
                        filtered_record,
                        cache_record,
                        truncated_context,
                        variant_pages,
                        provider=str(settings["teacher_answer_provider"]),
                        model=str(settings["teacher_answer_model"]),
                        temperature=settings["answer_temperature"],
                    )
                    assistant_response = teacher_validation["assistant_response"]
                    system_prompt = teacher_validation["system_prompt"]
                    user_prompt = teacher_validation["user_prompt"]
                    validation_mode = "teacher_validated"

                accepted += 1
                accepted_sample = {
                    **filtered_record,
                    "sample_id": f"truncref-{accepted:06d}",
                    "parent_sample_id": filtered_record.get("sample_id"),
                    "parent_query_id": filtered_record.get("query_id"),
                    "parent_assistant_response": parent_answer,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "assistant_response": assistant_response,
                    "assistant_response_json": compact_json_dumps(assistant_response),
                    "accepted_checks": [
                        "retrieval_present",
                        "schema_pruned",
                        "refusal_sample",
                        "truncated_support_removed",
                        f"truncated_refusal_{validation_mode}",
                    ],
                    "source": "truncated_refusal",
                    "variant_type": f"truncated_refusal.{validation_mode}.v1",
                    "should_refuse": True,
                    "retrieval_pages": variant_pages,
                    "context_doc_ids": [primary_doc_id],
                    "anchor_id": anchor_record.get("anchor_id"),
                    "anchor_source_bucket": anchor_record.get("source_bucket"),
                    "anchor_final_answer": anchor_record.get("final_answer"),
                    "anchor_pages": list(anchor_record.get("anchor_pages", []) or []),
                    "truncated_refusal_source_type": preferred_source,
                    "truncated_refusal_source_mix": [preferred_source],
                    "truncated_refusal_stats": {
                        "parent_result_count": len(retrieval_results),
                        "truncated_result_count": len(variant_results),
                        "parent_pages": normalize_pages(cache_record.get("retrieval_pages")),
                        "truncated_pages": variant_pages,
                        "anchor_pages": list(anchor_record.get("anchor_pages", []) or []),
                        "context_hash": stable_hash_int(truncated_context),
                        "source_type": preferred_source,
                    },
                    "truncated_refusal_validation": {
                        "mode": validation_mode,
                        "local_validation": local_validation,
                        "teacher_validation_used": api_processor is not None,
                    },
                    "build_timestamp": utc_now_iso(),
                }
                break

            if accepted_sample is None:
                raise ValueError("no_valid_truncated_variant")

            append_jsonl(settings["output_path"], accepted_sample)
            if settings["chat_output_path"] is not None:
                append_jsonl(settings["chat_output_path"], build_chat_record(accepted_sample))
            source_counter[str(source_type or "")] += 1
            schema_counter[str(filtered_record.get("schema") or "")] += 1
            validation_counter[str(accepted_sample["truncated_refusal_validation"]["mode"])] += 1
        except Exception as exc:
            rejected += 1
            reason = str(exc)
            append_jsonl(
                settings["rejected_output_path"],
                {
                    "query_id": query_id,
                    "sample_id": filtered_record.get("sample_id"),
                    "schema": filtered_record.get("schema"),
                    "question_text": filtered_record.get("question_text"),
                    "parent_final_answer": parent_answer.get("final_answer"),
                    "truncated_refusal_source_type": source_type,
                    "local_validation": local_validation,
                    "teacher_validation": teacher_validation,
                    "reject_log": build_reject_log(
                        stage="build_truncated_refusal",
                        reason_code=reason.split(":", maxsplit=1)[0],
                        schema=str(filtered_record.get("schema") or ""),
                        query_id=query_id,
                        sample_id=filtered_record.get("sample_id"),
                        message=reason,
                    ),
                    "build_timestamp": utc_now_iso(),
                },
            )
            rejection_counter[reason.split(":", maxsplit=1)[0]] += 1

    stats = {
        "build_timestamp": utc_now_iso(),
        "config_path": display_path(settings["config_path"], REPO_ROOT),
        "input_path": display_path(settings["input_path"], REPO_ROOT),
        "retrieved_cache_input_path": display_path(settings["retrieved_cache_input_path"], REPO_ROOT),
        "anchor_input_path": display_path(settings["anchor_input_path"], REPO_ROOT),
        "output_path": display_path(settings["output_path"], REPO_ROOT),
        "chat_output_path": display_path(settings["chat_output_path"], REPO_ROOT),
        "rejected_output_path": display_path(settings["rejected_output_path"], REPO_ROOT),
        "teacher_answer_provider": settings["teacher_answer_provider"],
        "teacher_answer_model": settings["teacher_answer_model"],
        "source_priority": settings["source_priority"],
        "requested_records": len(filtered_records),
        "accepted_records": accepted,
        "rejected_records": rejected,
        "source_type_distribution": dict(source_counter),
        "schema_distribution": dict(schema_counter),
        "validation_distribution": dict(validation_counter),
        "rejection_distribution": dict(rejection_counter),
    }
    write_json(settings["stats_output_path"], stats)


if __name__ == "__main__":
    main()
