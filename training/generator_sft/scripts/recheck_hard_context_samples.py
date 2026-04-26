from __future__ import annotations

import argparse
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api_requests import APIProcessor  # noqa: E402
from training.common import (  # noqa: E402
    append_jsonl,
    build_rag_prompt_bundle,
    compact_json_dumps,
    display_path,
    load_records,
    load_yaml_mapping,
    prune_answer_to_schema,
    resolve_repo_path,
    utc_now_iso,
    write_json,
)
from training.generator_sft.scripts.convert_to_chat_sft import build_chat_record  # noqa: E402


_THREAD_LOCAL = threading.local()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-run the teacher on hard-context samples and keep only consistent positives.")
    parser.add_argument("--config-path", type=Path, default=None, help="YAML config path.")
    parser.add_argument("--input-path", type=Path, default=None, help="Hard-context sample JSONL path.")
    parser.add_argument("--retrieved-cache-input-path", type=Path, default=None, help="retrieved_cache JSONL path.")
    parser.add_argument("--output-path", type=Path, default=None, help="Accepted rechecked sample JSONL path.")
    parser.add_argument("--chat-output-path", type=Path, default=None, help="Optional chat-format JSONL output path.")
    parser.add_argument("--rejected-output-path", type=Path, default=None, help="Rejected sample JSONL path.")
    parser.add_argument("--stats-output-path", type=Path, default=None, help="Stats JSON path.")
    parser.add_argument("--teacher-answer-provider", default=None, help="Teacher API provider used for re-check.")
    parser.add_argument("--teacher-answer-model", default=None, help="Teacher model used for re-check.")
    parser.add_argument("--answer-temperature", type=float, default=None, help="Teacher sampling temperature.")
    parser.add_argument("--parallel-requests", type=int, default=None, help="Parallel teacher re-check worker count.")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional cap over hard-context samples.")
    return parser


def _coalesce(cli_value: Any, config_value: Any, default: Any = None) -> Any:
    return cli_value if cli_value is not None else (config_value if config_value is not None else default)


def _reset_output_file(path: Optional[Path]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _index_records_by_query_id(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for record in records:
        query_id = str(record.get("query_id") or "").strip()
        if query_id:
            indexed[query_id] = record
    return indexed


def _normalize_pages(values: Any) -> List[int]:
    pages: List[int] = []
    for value in values or []:
        try:
            pages.append(int(value))
        except (TypeError, ValueError):
            continue
    return pages


def _coerce_decimal(value: Any) -> Decimal | None:
    if value in (None, "", []):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("，", "")
        if cleaned in {"", "N/A"}:
            return None
        try:
            return Decimal(cleaned)
        except InvalidOperation:
            return None
    return None


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_free_text_for_similarity(value: Any) -> str:
    text = _normalize_text(value).lower()
    if not text:
        return ""
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text)


def _character_ngrams(text: str, n: int = 2) -> set[str]:
    if not text:
        return set()
    if len(text) <= n:
        return {text}
    return {text[index : index + n] for index in range(len(text) - n + 1)}


def _semantic_segments(text: str) -> set[str]:
    if not text:
        return set()
    parts = re.split(r"[。；;！？!\?\n\r\-•]+", text)
    return {
        normalized
        for normalized in (_normalize_free_text_for_similarity(part) for part in parts)
        if len(normalized) >= 4
    }


def _text_answers_semantically_consistent(parent_answer: Any, rechecked_answer: Any) -> Tuple[bool, Dict[str, float]]:
    normalized_parent = _normalize_free_text_for_similarity(parent_answer)
    normalized_rechecked = _normalize_free_text_for_similarity(rechecked_answer)
    if not normalized_parent or not normalized_rechecked:
        return False, {
            "sequence_ratio": 0.0,
            "ngram_jaccard": 0.0,
            "ngram_containment": 0.0,
            "segment_overlap": 0.0,
        }

    if normalized_parent == normalized_rechecked:
        return True, {
            "sequence_ratio": 1.0,
            "ngram_jaccard": 1.0,
            "ngram_containment": 1.0,
            "segment_overlap": 1.0,
        }

    parent_ngrams = _character_ngrams(normalized_parent)
    rechecked_ngrams = _character_ngrams(normalized_rechecked)
    ngram_intersection = parent_ngrams & rechecked_ngrams
    ngram_union = parent_ngrams | rechecked_ngrams
    ngram_jaccard = len(ngram_intersection) / len(ngram_union) if ngram_union else 0.0
    min_ngram_size = min(len(parent_ngrams), len(rechecked_ngrams))
    ngram_containment = len(ngram_intersection) / min_ngram_size if min_ngram_size else 0.0

    parent_segments = _semantic_segments(str(parent_answer or ""))
    rechecked_segments = _semantic_segments(str(rechecked_answer or ""))
    segment_union = parent_segments | rechecked_segments
    segment_overlap = len(parent_segments & rechecked_segments) / len(segment_union) if segment_union else 0.0

    sequence_ratio = SequenceMatcher(None, normalized_parent, normalized_rechecked).ratio()
    metrics = {
        "sequence_ratio": round(sequence_ratio, 4),
        "ngram_jaccard": round(ngram_jaccard, 4),
        "ngram_containment": round(ngram_containment, 4),
        "segment_overlap": round(segment_overlap, 4),
    }
    consistent = (
        sequence_ratio >= 0.58
        or ngram_jaccard >= 0.38
        or (ngram_containment >= 0.62 and sequence_ratio >= 0.42)
        or (segment_overlap >= 0.34 and ngram_containment >= 0.45)
    )
    return consistent, metrics


def _normalize_answer_value(value: Any, schema: str) -> Any:
    if value == "N/A":
        return "N/A"
    if schema == "number":
        decimal_value = _coerce_decimal(value)
        return decimal_value.normalize() if decimal_value is not None else None
    if schema == "boolean":
        if isinstance(value, bool):
            return value
        lowered = str(value or "").strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return None
    if schema == "names":
        if not isinstance(value, list):
            return None
        normalized_items = sorted({_normalize_text(item) for item in value if _normalize_text(item)})
        return tuple(normalized_items)
    normalized = _normalize_text(value)
    return normalized or None


def _thread_local_api_processor(provider: str) -> APIProcessor:
    processor = getattr(_THREAD_LOCAL, "api_processor", None)
    current_provider = getattr(_THREAD_LOCAL, "api_provider", None)
    if processor is None or current_provider != provider:
        processor = APIProcessor(provider=provider)
        _THREAD_LOCAL.api_processor = processor
        _THREAD_LOCAL.api_provider = provider
    return processor


def _canonical_final_answer(record: Dict[str, Any], cache_record: Dict[str, Any]) -> Any:
    for field_name in ("canonical_final_answer", "anchor_final_answer"):
        value = record.get(field_name)
        if value not in (None, "", []):
            return value

    teacher_signal = cache_record.get("teacher_signal") if isinstance(cache_record.get("teacher_signal"), dict) else {}
    grounding = teacher_signal.get("table_grounding_result") if isinstance(teacher_signal.get("table_grounding_result"), dict) else {}
    normalized_value = grounding.get("normalized_value")
    if normalized_value is not None:
        return normalized_value
    return None


def _validate_rechecked_pages(record: Dict[str, Any], answer: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    checks: List[str] = []
    stats: Dict[str, Any] = {}

    final_answer = answer.get("final_answer")
    relevant_pages = _normalize_pages(answer.get("relevant_pages"))
    hard_context_pages = set(_normalize_pages(record.get("hard_context_pages")))
    hard_context_stats = record.get("hard_context_stats") if isinstance(record.get("hard_context_stats"), dict) else {}
    support_pages = set(_normalize_pages(hard_context_stats.get("support_pages") or record.get("retrieval_pages")))
    relevant_page_set = set(relevant_pages)

    stats["relevant_pages"] = relevant_pages
    stats["hard_context_pages"] = sorted(hard_context_pages)
    stats["support_pages"] = sorted(support_pages)
    stats["relevant_support_overlap_count"] = len(relevant_page_set & support_pages)

    if final_answer == "N/A":
        raise ValueError("recheck_teacher_refused")
    if not relevant_pages:
        raise ValueError("recheck_missing_relevant_pages")
    if hard_context_pages and not relevant_page_set.issubset(hard_context_pages):
        raise ValueError("recheck_hallucinated_pages")
    if support_pages and not (relevant_page_set & support_pages):
        raise ValueError("recheck_missing_support_pages")

    checks.append("recheck_relevant_pages_subset")
    checks.append("recheck_support_pages_cited")
    return checks, stats


def _build_rejected_record(
    record: Dict[str, Any],
    cache_record: Optional[Dict[str, Any]],
    *,
    reason: str,
    provider: str,
    model: str,
    rechecked_answer: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    parent_answer = record.get("assistant_response") if isinstance(record.get("assistant_response"), dict) else {}
    rejected = {
        "sample_id": record.get("sample_id"),
        "query_id": record.get("query_id"),
        "schema": record.get("schema"),
        "company_name": record.get("company_name"),
        "doc_ids": list(record.get("doc_ids", []) or []),
        "parent_sample_id": record.get("parent_sample_id"),
        "parent_query_id": record.get("parent_query_id"),
        "parent_final_answer": parent_answer.get("final_answer"),
        "canonical_final_answer": _canonical_final_answer(record, cache_record or {}),
        "rechecked_final_answer": (rechecked_answer or {}).get("final_answer"),
        "hard_context_pages": list(record.get("hard_context_pages", []) or []),
        "support_pages": list(((record.get("hard_context_stats") or {}) if isinstance(record.get("hard_context_stats"), dict) else {}).get("support_pages", []) or []),
        "distractor_pages": list(((record.get("hard_context_stats") or {}) if isinstance(record.get("hard_context_stats"), dict) else {}).get("distractor_pages", []) or []),
        "reason": reason,
        "teacher_answer_provider": provider,
        "teacher_answer_model": model,
        "build_timestamp": utc_now_iso(),
    }
    if rechecked_answer is not None:
        rejected["rechecked_answer"] = rechecked_answer
    return rejected


def _recheck_record(
    record: Dict[str, Any],
    cache_record: Dict[str, Any],
    *,
    provider: str,
    model: str,
    temperature: float,
) -> Dict[str, Any]:
    schema = str(record.get("schema") or "")
    if not schema:
        raise ValueError("missing_schema")
    if not str(record.get("system_prompt") or "").strip() or not str(record.get("user_prompt") or "").strip():
        raise ValueError("missing_prompt")

    response_format = build_rag_prompt_bundle(schema, provider=provider)[2]
    api_processor = _thread_local_api_processor(provider)
    raw_answer = api_processor.send_message(
        model=model,
        temperature=temperature,
        system_content=str(record.get("system_prompt") or "").strip(),
        human_content=str(record.get("user_prompt") or "").strip(),
        is_structured=True,
        response_format=response_format,
    )
    rechecked_answer = prune_answer_to_schema(raw_answer, schema=schema, provider=provider)

    parent_answer = record.get("assistant_response") if isinstance(record.get("assistant_response"), dict) else {}
    parent_final_answer = parent_answer.get("final_answer")
    if parent_final_answer in (None, "", [], "N/A"):
        raise ValueError("parent_positive_answer_missing")

    rechecked_final_answer = rechecked_answer.get("final_answer")
    if rechecked_final_answer in (None, "", []):
        raise ValueError("recheck_empty_final_answer")

    normalized_parent = _normalize_answer_value(parent_final_answer, schema)
    normalized_rechecked = _normalize_answer_value(rechecked_final_answer, schema)
    if normalized_parent is None or normalized_rechecked is None:
        raise ValueError("recheck_uncomparable_answer")
    answer_match_mode = "exact"
    text_match_metrics: Optional[Dict[str, float]] = None
    if normalized_parent != normalized_rechecked:
        if schema in {"text", "long_text"}:
            text_consistent, text_match_metrics = _text_answers_semantically_consistent(
                parent_final_answer,
                rechecked_final_answer,
            )
            if not text_consistent:
                raise ValueError("recheck_parent_answer_mismatch")
            answer_match_mode = "semantic"
        else:
            raise ValueError("recheck_parent_answer_mismatch")

    canonical_final_answer = _canonical_final_answer(record, cache_record)
    if canonical_final_answer not in (None, "", []):
        normalized_canonical = _normalize_answer_value(canonical_final_answer, schema)
        if normalized_canonical is None:
            raise ValueError("canonical_answer_uncomparable")
        if normalized_rechecked != normalized_canonical:
            raise ValueError("recheck_canonical_answer_mismatch")
    else:
        normalized_canonical = None

    page_checks, page_stats = _validate_rechecked_pages(record, rechecked_answer)

    accepted_checks = list(record.get("accepted_checks", []) or [])
    for check in (
        "teacher_rechecked_hard_context",
        "hard_context_answer_matches_parent",
        *page_checks,
    ):
        if check not in accepted_checks:
            accepted_checks.append(check)
    if answer_match_mode == "semantic" and "hard_context_answer_semantic_match" not in accepted_checks:
        accepted_checks.append("hard_context_answer_semantic_match")
    if normalized_canonical is not None and "hard_context_answer_matches_canonical" not in accepted_checks:
        accepted_checks.append("hard_context_answer_matches_canonical")

    accepted_record = dict(record)
    accepted_record["parent_assistant_response"] = parent_answer
    accepted_record["hard_context_parent_sample_id"] = record.get("sample_id")
    accepted_record["assistant_response"] = rechecked_answer
    accepted_record["assistant_response_json"] = compact_json_dumps(rechecked_answer)
    accepted_record["accepted_checks"] = accepted_checks
    accepted_record["source"] = "hard_context_positive_rechecked"
    accepted_record["variant_type"] = "hard_context_with_distractors.teacher_rechecked.v1"
    accepted_record["teacher_answer_provider"] = provider
    accepted_record["teacher_answer_model"] = model
    accepted_record["should_refuse"] = False
    accepted_record["hard_context_recheck"] = {
        "teacher_answer_provider": provider,
        "teacher_answer_model": model,
        "answer_temperature": temperature,
        "answer_match_mode": answer_match_mode,
        "parent_final_answer": parent_final_answer,
        "rechecked_final_answer": rechecked_final_answer,
        "canonical_final_answer": canonical_final_answer,
        "relevant_pages": page_stats["relevant_pages"],
        "relevant_support_overlap_count": page_stats["relevant_support_overlap_count"],
        "build_timestamp": utc_now_iso(),
    }
    if text_match_metrics is not None:
        accepted_record["hard_context_recheck"]["text_match_metrics"] = text_match_metrics
    return accepted_record


def _resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    default_config_path = REPO_ROOT / "training/generator_sft/configs/hard_context_recheck.example.yaml"
    config_path = args.config_path or (default_config_path if default_config_path.exists() else None)
    config = load_yaml_mapping(config_path)

    input_path = resolve_repo_path(REPO_ROOT, _coalesce(args.input_path, config.get("input_path")))
    retrieved_cache_input_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(args.retrieved_cache_input_path, config.get("retrieved_cache_input_path")),
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
    if input_path is None or retrieved_cache_input_path is None or output_path is None or rejected_output_path is None or stats_output_path is None:
        raise ValueError("input/retrieved-cache/output/rejected/stats paths are required.")

    teacher_answer_provider = str(_coalesce(args.teacher_answer_provider, config.get("teacher_answer_provider"), "qwen"))
    teacher_answer_model = _coalesce(args.teacher_answer_model, config.get("teacher_answer_model"))
    if not teacher_answer_model:
        raise ValueError("teacher_answer_model is required for re-check.")

    return {
        "config_path": config_path,
        "input_path": input_path,
        "retrieved_cache_input_path": retrieved_cache_input_path,
        "output_path": output_path,
        "chat_output_path": chat_output_path,
        "rejected_output_path": rejected_output_path,
        "stats_output_path": stats_output_path,
        "teacher_answer_provider": teacher_answer_provider,
        "teacher_answer_model": teacher_answer_model,
        "answer_temperature": float(_coalesce(args.answer_temperature, config.get("answer_temperature"), 0.0)),
        "parallel_requests": max(1, int(_coalesce(args.parallel_requests, config.get("parallel_requests"), 1))),
        "max_queries": max(0, int(_coalesce(args.max_queries, config.get("max_queries"), 0) or 0)),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = _resolve_settings(args)

    hard_context_records = load_records(settings["input_path"])
    if settings["max_queries"] > 0:
        hard_context_records = hard_context_records[: settings["max_queries"]]
    cache_by_query_id = _index_records_by_query_id(load_records(settings["retrieved_cache_input_path"]))

    _reset_output_file(settings["output_path"])
    _reset_output_file(settings["chat_output_path"])
    _reset_output_file(settings["rejected_output_path"])

    worker_results: List[Tuple[int, Dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=settings["parallel_requests"]) as executor:
        future_to_index = {}
        for index, record in enumerate(hard_context_records):
            query_id = str(record.get("query_id") or "").strip()
            cache_record = cache_by_query_id.get(query_id)
            if cache_record is None:
                worker_results.append(
                    (
                        index,
                        {
                            "status": "rejected",
                            "record": _build_rejected_record(
                                record,
                                None,
                                reason="missing_retrieved_cache",
                                provider=settings["teacher_answer_provider"],
                                model=settings["teacher_answer_model"],
                            ),
                        },
                    )
                )
                continue
            future = executor.submit(
                _recheck_record,
                record,
                cache_record,
                provider=settings["teacher_answer_provider"],
                model=settings["teacher_answer_model"],
                temperature=settings["answer_temperature"],
            )
            future_to_index[future] = (index, record, cache_record)

        for future in as_completed(future_to_index):
            index, record, cache_record = future_to_index[future]
            try:
                accepted_record = future.result()
            except Exception as exc:
                reason = str(exc)
                rejected_record = _build_rejected_record(
                    record,
                    cache_record,
                    reason=reason,
                    provider=settings["teacher_answer_provider"],
                    model=settings["teacher_answer_model"],
                )
                worker_results.append((index, {"status": "rejected", "record": rejected_record}))
                continue
            worker_results.append((index, {"status": "accepted", "record": accepted_record}))

    worker_results.sort(key=lambda item: item[0])

    accepted = 0
    rejected = 0
    schema_counter = Counter()
    rejection_counter = Counter()
    recheck_outcome_counter = Counter()

    for _, result in worker_results:
        record = result["record"]
        if result["status"] == "accepted":
            accepted += 1
            record["sample_id"] = f"hardctxr-{accepted:06d}"
            append_jsonl(settings["output_path"], record)
            if settings["chat_output_path"] is not None:
                append_jsonl(settings["chat_output_path"], build_chat_record(record))
            schema_counter[str(record.get("schema") or "")] += 1
            recheck_outcome_counter["accepted"] += 1
            continue

        rejected += 1
        append_jsonl(settings["rejected_output_path"], record)
        rejection_counter[str(record.get("reason") or "").split(":", maxsplit=1)[0]] += 1
        recheck_outcome_counter[str(record.get("reason") or "").split(":", maxsplit=1)[0]] += 1

    stats = {
        "build_timestamp": utc_now_iso(),
        "config_path": display_path(settings["config_path"], REPO_ROOT),
        "input_path": display_path(settings["input_path"], REPO_ROOT),
        "retrieved_cache_input_path": display_path(settings["retrieved_cache_input_path"], REPO_ROOT),
        "output_path": display_path(settings["output_path"], REPO_ROOT),
        "chat_output_path": display_path(settings["chat_output_path"], REPO_ROOT),
        "rejected_output_path": display_path(settings["rejected_output_path"], REPO_ROOT),
        "teacher_answer_provider": settings["teacher_answer_provider"],
        "teacher_answer_model": settings["teacher_answer_model"],
        "answer_temperature": settings["answer_temperature"],
        "parallel_requests": settings["parallel_requests"],
        "requested_records": len(hard_context_records),
        "accepted_records": accepted,
        "rejected_records": rejected,
        "schema_distribution": dict(schema_counter),
        "rejection_distribution": dict(rejection_counter),
        "recheck_outcome_distribution": dict(recheck_outcome_counter),
    }
    write_json(settings["stats_output_path"], stats)


if __name__ == "__main__":
    main()
