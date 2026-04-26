from __future__ import annotations

import argparse
import copy
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import (  # noqa: E402
    append_jsonl,
    build_query_context,
    build_questions_processor,
    collect_existing_ids,
    display_path,
    inflate_serialized_retrieval_results,
    load_records,
    load_yaml_mapping,
    normalize_training_query_record,
    resolve_dataset_root,
    resolve_repo_path,
    utc_now_iso,
    write_json,
)


_THREAD_LOCAL = threading.local()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine retrieval-grounded teacher answers for generator SFT.")
    parser.add_argument("--config-path", type=Path, default=None, help="YAML config path.")
    parser.add_argument("--input-path", type=Path, default=None, help="Input JSONL/JSON seed query path.")
    parser.add_argument("--output-path", type=Path, default=None, help="Output JSONL path for raw teacher answers.")
    parser.add_argument("--debug-output-path", type=Path, default=None, help="Output JSONL path for debug records.")
    parser.add_argument("--retrieved-cache-output-path", type=Path, default=None, help="Output JSONL path for reusable retrieved_cache records.")
    parser.add_argument("--stats-output-path", type=Path, default=None, help="Output JSON path for build stats.")
    parser.add_argument("--retrieval-config-path", type=Path, default=None, help="Run config used for retrieval.")
    parser.add_argument("--dataset-root-path", type=Path, default=None, help="Dataset root holding databases, metadata, and manifests.")
    parser.add_argument("--teacher-answer-provider", default=None, help="Teacher API provider.")
    parser.add_argument("--teacher-answer-model", default=None, help="Teacher answer model.")
    parser.add_argument("--teacher-verify-provider", default=None, help="Reserved metadata field for verifier provider.")
    parser.add_argument("--teacher-verify-model", default=None, help="Reserved metadata field for verifier model.")
    parser.add_argument("--answer-temperature", type=float, default=None, help="Teacher answer temperature.")
    parser.add_argument("--max-queries", type=int, default=None, help="Maximum number of queries to process.")
    parser.add_argument("--parallel-requests", type=int, default=None, help="Worker count for mining.")
    parser.add_argument("--resume", action="store_true", help="Skip query_ids already present in output_path.")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume behavior.")
    return parser


def _coalesce(cli_value: Any, config_value: Any, default: Any = None) -> Any:
    return cli_value if cli_value is not None else (config_value if config_value is not None else default)


def _thread_local_processor(
    *,
    retrieval_config_path: Path,
    dataset_root_path: Path,
    teacher_answer_provider: str,
    teacher_answer_model: str,
    answer_temperature: float,
    parallel_requests: int,
):
    processor = getattr(_THREAD_LOCAL, "processor", None)
    if processor is None:
        processor = build_questions_processor(
            REPO_ROOT,
            retrieval_config_path,
            dataset_root=dataset_root_path,
            api_provider=teacher_answer_provider,
            answering_model=teacher_answer_model,
            answer_temperature=answer_temperature,
            reasoning_debug_enabled=True,
            parallel_requests=parallel_requests,
        )
        _THREAD_LOCAL.processor = processor
    return processor


def _build_validation_result(answer: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "validation_flags": list(answer.get("validation_flags", [])),
        "confidence": answer.get("confidence"),
        "confidence_reason": answer.get("confidence_reason"),
    }


def _build_teacher_signal(answer: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "final_answer": answer.get("final_answer"),
        "relevant_pages": list(answer.get("relevant_pages", []) or []),
        "references": list(answer.get("references", []) or []),
        "confidence": answer.get("confidence"),
        "confidence_reason": answer.get("confidence_reason"),
        "validation_flags": list(answer.get("validation_flags", []) or []),
        "table_grounding_result": answer.get("table_grounding_result"),
    }


def _build_retrieved_cache_record(
    *,
    normalized: Dict[str, Any],
    answer: Dict[str, Any],
    query_plan_payload: Dict[str, Any],
    route_info_payload: Dict[str, Any],
    retrieval_config: str,
    rag_context: str,
    build_timestamp: str,
    teacher_answer_provider: Optional[str],
    teacher_answer_model: Optional[str],
    teacher_verify_provider: Optional[str],
    teacher_verify_model: Optional[str],
) -> Dict[str, Any]:
    retrieval_pages = list(answer.get("retrieval_pages", []) or [])
    retrieval_results = list(answer.get("retrieval_results", []) or [])
    retrieval_report_groups = list(answer.get("retrieval_report_groups", []) or [])
    search_queries = query_plan_payload.get("search_queries")
    if not isinstance(search_queries, list) or not search_queries:
        search_queries = list(answer.get("search_queries", []) or [])
    return {
        "cache_type": "retrieved_cache.v1",
        "query_id": normalized["query_id"],
        "question_text": normalized["question_text"],
        "schema": normalized["schema"],
        "task_type": normalized.get("task_type"),
        "company_name": normalized["company_name"],
        "mentioned_companies": normalized["mentioned_companies"],
        "doc_ids": normalized["doc_ids"],
        "expected_filters": normalized["expected_filters"],
        "source": normalized["source"],
        "difficulty": normalized["difficulty"],
        "should_refuse": normalized["should_refuse"],
        "template_id": normalized.get("template_id"),
        "template_family": normalized.get("template_family"),
        "template_version": normalized.get("template_version"),
        "target_key": normalized.get("target_key"),
        "surface_variant_id": normalized.get("surface_variant_id"),
        "split_pool": normalized.get("split_pool"),
        "answer_policy": normalized.get("answer_policy"),
        "validator_target": normalized.get("validator_target"),
        "teacher_answer_provider": teacher_answer_provider,
        "teacher_answer_model": teacher_answer_model,
        "teacher_verify_provider": teacher_verify_provider,
        "teacher_verify_model": teacher_verify_model,
        "retrieval_config": retrieval_config,
        "query_plan": copy.deepcopy(query_plan_payload),
        "route_info": copy.deepcopy(route_info_payload),
        "search_queries": list(search_queries or []),
        "rag_context": rag_context,
        "retrieval_pages": retrieval_pages,
        "retrieval_results": retrieval_results,
        "retrieval_report_groups": retrieval_report_groups,
        "retrieval_result_count": len(retrieval_results),
        "retrieval_doc_count": len({group.get("doc_id") for group in retrieval_report_groups if isinstance(group, dict) and group.get("doc_id")}),
        "candidate_pool_size_before_rerank": answer.get("candidate_pool_size_before_rerank"),
        "initial_candidate_pool_size": answer.get("initial_candidate_pool_size"),
        "colbert_candidate_pool_size": answer.get("colbert_candidate_pool_size"),
        "colbert_top_n": answer.get("colbert_top_n"),
        "reranking_strategy": answer.get("reranking_strategy"),
        "final_reranking_backend": answer.get("final_reranking_backend"),
        "hyde": copy.deepcopy(answer.get("hyde") or {}),
        "response_data": dict(answer.get("response_data", {}) or {}),
        "teacher_signal": _build_teacher_signal(answer),
        "build_timestamp": build_timestamp,
    }


def _run_teacher_answer(processor, query_context: Dict[str, Any]) -> Dict[str, Any]:
    normalized = query_context["normalized"]
    schema = normalized["schema"]
    task_type = str(normalized.get("task_type") or "")
    question_text = normalized["question_text"]
    doc_ids = query_context["doc_ids"]
    mentioned_companies = query_context["mentioned_companies"]
    company_name = query_context["company_name"]

    if len(mentioned_companies) > 1 and (schema == "comparative" or task_type == "cross_doc_compare"):
        return processor.process_comparative_question(question_text, mentioned_companies, schema)

    try:
        return processor.get_answer_for_company(
            company_name or "",
            question_text,
            schema,
            query_plan=query_context["query_plan"],
            route_info=query_context["route_info"],
        )
    except Exception:
        if len(mentioned_companies) > 1 and (schema == "comparative" or task_type == "cross_doc_compare"):
            return processor.process_comparative_question(question_text, mentioned_companies, schema)
        raise


def mine_teacher_answer_record(
    record: Dict[str, Any],
    *,
    retrieval_config_path: Path,
    dataset_root_path: Path,
    teacher_answer_provider: str,
    teacher_answer_model: str,
    teacher_verify_provider: Optional[str],
    teacher_verify_model: Optional[str],
    answer_temperature: float,
    parallel_requests: int,
) -> Dict[str, Any]:
    processor = _thread_local_processor(
        retrieval_config_path=retrieval_config_path,
        dataset_root_path=dataset_root_path,
        teacher_answer_provider=teacher_answer_provider,
        teacher_answer_model=teacher_answer_model,
        answer_temperature=answer_temperature,
        parallel_requests=parallel_requests,
    )
    query_context = build_query_context(processor, record)
    normalized = query_context["normalized"]

    answer = _run_teacher_answer(processor, query_context)
    answer_copy = copy.deepcopy(answer)
    rag_context = processor._format_retrieval_results(
        inflate_serialized_retrieval_results(answer_copy.get("retrieval_results", []))
    )
    build_timestamp = utc_now_iso()
    retrieval_config = str(retrieval_config_path.relative_to(REPO_ROOT))
    query_plan_payload = answer_copy.get("query_plan") or query_context["query_plan"].to_dict()
    route_info_payload = answer_copy.get("route_info") or query_context["route_info"]

    raw_record = {
        "query_id": normalized["query_id"],
        "question_text": normalized["question_text"],
        "schema": normalized["schema"],
        "task_type": normalized.get("task_type"),
        "company_name": normalized["company_name"],
        "mentioned_companies": normalized["mentioned_companies"],
        "doc_ids": normalized["doc_ids"],
        "expected_filters": normalized["expected_filters"],
        "source": normalized["source"],
        "difficulty": normalized["difficulty"],
        "should_refuse": normalized["should_refuse"],
        "template_id": normalized.get("template_id"),
        "template_family": normalized.get("template_family"),
        "template_version": normalized.get("template_version"),
        "target_key": normalized.get("target_key"),
        "surface_variant_id": normalized.get("surface_variant_id"),
        "split_pool": normalized.get("split_pool"),
        "answer_policy": normalized.get("answer_policy"),
        "validator_target": normalized.get("validator_target"),
        "teacher_answer_provider": teacher_answer_provider,
        "teacher_answer_model": teacher_answer_model,
        "teacher_verify_provider": teacher_verify_provider,
        "teacher_verify_model": teacher_verify_model,
        "retrieval_config": retrieval_config,
        "rag_context": rag_context,
        "retrieval_pages": list(answer_copy.get("retrieval_pages", [])),
        "retrieval_results": list(answer_copy.get("retrieval_results", [])),
        "answer": answer_copy,
        "response_data": dict(answer_copy.get("response_data", {}) or {}),
        "table_grounding_result": answer_copy.get("table_grounding_result"),
        "validation_result": _build_validation_result(answer_copy),
        "build_timestamp": build_timestamp,
    }

    debug_record = {
        "query_id": normalized["query_id"],
        "question_text": normalized["question_text"],
        "schema": normalized["schema"],
        "status": "ok",
        "query_plan": query_plan_payload,
        "route_info": route_info_payload,
        "retrieval_pages": list(answer_copy.get("retrieval_pages", [])),
        "validation_flags": list(answer_copy.get("validation_flags", [])),
        "candidate_pool_size_before_rerank": answer_copy.get("candidate_pool_size_before_rerank"),
        "reranking_strategy": answer_copy.get("reranking_strategy"),
        "initial_candidate_pool_size": answer_copy.get("initial_candidate_pool_size"),
        "colbert_candidate_pool_size": answer_copy.get("colbert_candidate_pool_size"),
        "colbert_top_n": answer_copy.get("colbert_top_n"),
        "final_reranking_backend": answer_copy.get("final_reranking_backend"),
        "hyde": answer_copy.get("hyde"),
        "response_data": dict(answer_copy.get("response_data", {}) or {}),
        "build_timestamp": raw_record["build_timestamp"],
    }
    retrieved_cache_record = _build_retrieved_cache_record(
        normalized=normalized,
        answer=answer_copy,
        query_plan_payload=query_plan_payload,
        route_info_payload=route_info_payload,
        retrieval_config=retrieval_config,
        rag_context=rag_context,
        build_timestamp=build_timestamp,
        teacher_answer_provider=teacher_answer_provider,
        teacher_answer_model=teacher_answer_model,
        teacher_verify_provider=teacher_verify_provider,
        teacher_verify_model=teacher_verify_model,
    )
    return {
        "raw_record": raw_record,
        "debug_record": debug_record,
        "retrieved_cache_record": retrieved_cache_record,
    }


def _build_retrieved_cache_from_raw_record(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_training_query_record(record)
    answer = record.get("answer") if isinstance(record.get("answer"), dict) else {}
    query_plan_payload = answer.get("query_plan") if isinstance(answer.get("query_plan"), dict) else {}
    route_info_payload = answer.get("route_info") if isinstance(answer.get("route_info"), dict) else {}
    return _build_retrieved_cache_record(
        normalized=normalized,
        answer=answer,
        query_plan_payload=query_plan_payload,
        route_info_payload=route_info_payload,
        retrieval_config=str(record.get("retrieval_config") or ""),
        rag_context=str(record.get("rag_context") or ""),
        build_timestamp=str(record.get("build_timestamp") or utc_now_iso()),
        teacher_answer_provider=record.get("teacher_answer_provider"),
        teacher_answer_model=record.get("teacher_answer_model"),
        teacher_verify_provider=record.get("teacher_verify_provider"),
        teacher_verify_model=record.get("teacher_verify_model"),
    )


def _error_debug_record(record: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
    return {
        "query_id": record.get("query_id"),
        "question_text": record.get("question_text") or record.get("query") or record.get("question"),
        "schema": record.get("schema") or record.get("kind"),
        "status": "error",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "build_timestamp": utc_now_iso(),
    }


def _cache_error_debug_record(record: Dict[str, Any], exc: Exception, *, stage: str) -> Dict[str, Any]:
    return {
        "query_id": record.get("query_id"),
        "question_text": record.get("question_text"),
        "schema": record.get("schema"),
        "status": "cache_error",
        "cache_stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "build_timestamp": utc_now_iso(),
    }


def _resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    default_config_path = REPO_ROOT / "training/generator_sft/configs/data_build.example.yaml"
    config_path = args.config_path or (default_config_path if default_config_path.exists() else None)
    config = load_yaml_mapping(config_path)

    resume_config = config.get("resume")
    resume = True if args.resume else False if args.no_resume else bool(resume_config if resume_config is not None else True)

    retrieval_config_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(args.retrieval_config_path, config.get("retrieval_config_path")),
    )
    if retrieval_config_path is None:
        raise ValueError("`retrieval_config_path` is required.")

    input_path = resolve_repo_path(REPO_ROOT, _coalesce(args.input_path, config.get("input_path")))
    output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.output_path, config.get("output_path")))
    debug_output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.debug_output_path, config.get("debug_output_path")))
    retrieved_cache_output_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(args.retrieved_cache_output_path, config.get("retrieved_cache_output_path")),
    )
    stats_output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.stats_output_path, config.get("stats_output_path")))
    dataset_root_path = resolve_dataset_root(REPO_ROOT, _coalesce(args.dataset_root_path, config.get("dataset_root_path")))
    if retrieved_cache_output_path is None and output_path is not None:
        retrieved_cache_output_path = output_path.with_name("retrieved_cache.jsonl")
    if input_path is None or output_path is None or debug_output_path is None or stats_output_path is None:
        raise ValueError("input/output/debug/stats paths are required.")

    return {
        "config_path": config_path,
        "input_path": input_path,
        "output_path": output_path,
        "debug_output_path": debug_output_path,
        "retrieved_cache_output_path": retrieved_cache_output_path,
        "stats_output_path": stats_output_path,
        "retrieval_config_path": retrieval_config_path,
        "dataset_root_path": dataset_root_path,
        "teacher_answer_provider": _coalesce(args.teacher_answer_provider, config.get("teacher_answer_provider"), "qwen"),
        "teacher_answer_model": _coalesce(args.teacher_answer_model, config.get("teacher_answer_model")),
        "teacher_verify_provider": _coalesce(args.teacher_verify_provider, config.get("teacher_verify_provider")),
        "teacher_verify_model": _coalesce(args.teacher_verify_model, config.get("teacher_verify_model")),
        "answer_temperature": float(_coalesce(args.answer_temperature, config.get("answer_temperature"), 0.0)),
        "max_queries": int(_coalesce(args.max_queries, config.get("max_queries"), 0) or 0),
        "parallel_requests": max(1, int(_coalesce(args.parallel_requests, config.get("parallel_requests"), 1))),
        "resume": resume,
    }


def _reset_output_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _backfill_retrieved_cache_from_raw(
    *,
    raw_output_path: Path,
    retrieved_cache_output_path: Optional[Path],
    debug_output_path: Path,
) -> int:
    if retrieved_cache_output_path is None or not raw_output_path.exists():
        return 0
    existing_cache_ids = collect_existing_ids(retrieved_cache_output_path)
    backfilled = 0
    for raw_record in load_records(raw_output_path):
        query_id = str(raw_record.get("query_id") or "")
        if not query_id or query_id in existing_cache_ids:
            continue
        try:
            append_jsonl(retrieved_cache_output_path, _build_retrieved_cache_from_raw_record(raw_record))
        except Exception as exc:
            append_jsonl(
                debug_output_path,
                _cache_error_debug_record(raw_record, exc, stage="backfill_from_raw"),
            )
            continue
        existing_cache_ids.add(query_id)
        backfilled += 1
    return backfilled


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = _resolve_settings(args)
    backfilled = 0
    if not settings["resume"]:
        _reset_output_file(settings["output_path"])
        _reset_output_file(settings["debug_output_path"])
        if settings["retrieved_cache_output_path"] is not None:
            _reset_output_file(settings["retrieved_cache_output_path"])
    else:
        backfilled = _backfill_retrieved_cache_from_raw(
            raw_output_path=settings["output_path"],
            retrieved_cache_output_path=settings["retrieved_cache_output_path"],
            debug_output_path=settings["debug_output_path"],
        )
    records = load_records(settings["input_path"])
    if settings["max_queries"] > 0:
        records = records[: settings["max_queries"]]

    existing_ids = collect_existing_ids(settings["output_path"]) if settings["resume"] else set()
    pending_records = [
        record
        for record in records
        if str(record.get("query_id") or "") not in existing_ids
    ]

    stats = {
        "build_timestamp": utc_now_iso(),
        "config_path": display_path(settings["config_path"], REPO_ROOT),
        "input_path": display_path(settings["input_path"], REPO_ROOT),
        "output_path": display_path(settings["output_path"], REPO_ROOT),
        "debug_output_path": display_path(settings["debug_output_path"], REPO_ROOT),
        "retrieved_cache_output_path": display_path(settings["retrieved_cache_output_path"], REPO_ROOT),
        "retrieval_config_path": display_path(settings["retrieval_config_path"], REPO_ROOT),
        "teacher_answer_provider": settings["teacher_answer_provider"],
        "teacher_answer_model": settings["teacher_answer_model"],
        "teacher_verify_provider": settings["teacher_verify_provider"],
        "teacher_verify_model": settings["teacher_verify_model"],
        "requested_queries": len(records),
        "existing_output_records": len(existing_ids),
        "skipped_existing": len(existing_ids & {str(record.get('query_id') or '') for record in records}),
        "pending_queries": len(pending_records),
        "processed_ok": 0,
        "processed_failed": 0,
        "processed_cache_ok": 0,
        "processed_cache_failed": 0,
        "backfilled_cache_records": backfilled if settings["resume"] else 0,
        "failed_query_ids": [],
        "cache_failed_query_ids": [],
        "schema_counts": {},
    }

    with ThreadPoolExecutor(max_workers=settings["parallel_requests"]) as executor:
        future_to_record = {
            executor.submit(
                mine_teacher_answer_record,
                record,
                retrieval_config_path=settings["retrieval_config_path"],
                dataset_root_path=settings["dataset_root_path"],
                teacher_answer_provider=settings["teacher_answer_provider"],
                teacher_answer_model=settings["teacher_answer_model"],
                teacher_verify_provider=settings["teacher_verify_provider"],
                teacher_verify_model=settings["teacher_verify_model"],
                answer_temperature=settings["answer_temperature"],
                parallel_requests=settings["parallel_requests"],
            ): record
            for record in pending_records
        }

        for future in as_completed(future_to_record):
            record = future_to_record[future]
            schema = record.get("schema") or record.get("kind") or "unknown"
            stats["schema_counts"][schema] = int(stats["schema_counts"].get(schema, 0)) + 1
            try:
                result = future.result()
            except Exception as exc:
                stats["processed_failed"] += 1
                if record.get("query_id"):
                    stats["failed_query_ids"].append(str(record["query_id"]))
                append_jsonl(settings["debug_output_path"], _error_debug_record(record, exc))
                continue

            append_jsonl(settings["output_path"], result["raw_record"])
            append_jsonl(settings["debug_output_path"], result["debug_record"])
            try:
                if settings["retrieved_cache_output_path"] is not None:
                    append_jsonl(settings["retrieved_cache_output_path"], result["retrieved_cache_record"])
                    stats["processed_cache_ok"] += 1
            except Exception as exc:
                stats["processed_cache_failed"] += 1
                if record.get("query_id"):
                    stats["cache_failed_query_ids"].append(str(record["query_id"]))
                append_jsonl(
                    settings["debug_output_path"],
                    _cache_error_debug_record(result["raw_record"], exc, stage="write_retrieved_cache"),
                )
            stats["processed_ok"] += 1

    write_json(settings["stats_output_path"], stats)
    if stats["processed_ok"] == 0 and not existing_ids:
        raise SystemExit(
            "No teacher answers were generated. "
            f"See debug records at {display_path(settings['debug_output_path'], REPO_ROOT)} "
            f"and stats at {display_path(settings['stats_output_path'], REPO_ROOT)}."
        )
    if stats["processed_ok"] == 0 and existing_ids:
        print(
            "No new teacher answers were generated in this run. "
            f"Existing raw records retained: {len(existing_ids)}. "
            f"Pending failures this run: {stats['processed_failed']}."
        )


if __name__ == "__main__":
    main()
