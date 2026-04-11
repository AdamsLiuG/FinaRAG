from __future__ import annotations

import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import (  # noqa: E402
    append_jsonl,
    build_query_context,
    build_questions_processor,
    collect_existing_ids,
    display_path,
    load_records,
    load_yaml_mapping,
    resolve_dataset_root,
    resolve_repo_path,
    utc_now_iso,
    write_json,
)


_THREAD_LOCAL = threading.local()


def _result_score(result: Dict[str, Any]) -> float:
    return float(result.get("combined_score", result.get("ranking_score", result.get("distance", 0.0))))


def build_candidate_payload(result: Dict[str, Any], rank_index: int) -> Dict[str, Any]:
    metadata = result.get("metadata") or {}
    return {
        "candidate_id": f"cand-{rank_index:04d}",
        "doc_id": str(metadata.get("sha1_name") or metadata.get("doc_id") or ""),
        "page": int(result.get("page") or 0),
        "chunk_id": metadata.get("chunk_id"),
        "text": result.get("text", ""),
        "retrieval_sources": list(result.get("retrieval_sources", [])),
        "base_score": round(_result_score(result), 4),
        "section_name": metadata.get("section_name") or metadata.get("section_title"),
        "matched_queries": list(result.get("matched_queries", [])),
        "query_hit_count": int(result.get("query_hit_count", len(result.get("matched_queries", [])))),
        "result_scope": result.get("result_scope"),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect pre-rerank candidate pools for reranker distillation.")
    parser.add_argument("--config-path", type=Path, default=None, help="YAML config path.")
    parser.add_argument("--query-input-path", type=Path, default=None, help="Input query JSONL/JSON path.")
    parser.add_argument("--candidate-output-path", type=Path, default=None, help="Output JSONL path.")
    parser.add_argument("--stats-output-path", type=Path, default=None, help="Build stats JSON path.")
    parser.add_argument("--retrieval-config-path", type=Path, default=None, help="Run config used for retrieval.")
    parser.add_argument("--dataset-root-path", type=Path, default=None, help="Dataset root holding databases, metadata, and manifests.")
    parser.add_argument("--candidate-pool-size", type=int, default=None, help="Final deduped candidate cap per query.")
    parser.add_argument("--per-query-retrieval-top-k", type=int, default=None, help="Candidate count before merging across search queries.")
    parser.add_argument("--parallel-requests", type=int, default=None, help="Worker count for collection.")
    parser.add_argument("--resume", action="store_true", help="Skip query_ids already present in output_path.")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume behavior.")
    return parser


def _coalesce(cli_value: Any, config_value: Any, default: Any = None) -> Any:
    return cli_value if cli_value is not None else (config_value if config_value is not None else default)


def _thread_local_processor(*, retrieval_config_path: Path, dataset_root_path: Path, parallel_requests: int):
    processor = getattr(_THREAD_LOCAL, "processor", None)
    if processor is None:
        processor = build_questions_processor(
            REPO_ROOT,
            retrieval_config_path,
            dataset_root=dataset_root_path,
            reasoning_debug_enabled=True,
            parallel_requests=parallel_requests,
        )
        _THREAD_LOCAL.processor = processor
    return processor


def collect_candidate_pool_record(
    record: Dict[str, Any],
    *,
    retrieval_config_path: Path,
    dataset_root_path: Path,
    candidate_pool_size: int,
    per_query_retrieval_top_k: int,
    parallel_requests: int,
) -> Dict[str, Any]:
    processor = _thread_local_processor(
        retrieval_config_path=retrieval_config_path,
        dataset_root_path=dataset_root_path,
        parallel_requests=parallel_requests,
    )
    query_context = build_query_context(processor, record)
    normalized = query_context["normalized"]
    retriever, mode = processor._build_retriever()
    if not hasattr(retriever, "retrieve_candidates_by_company_name"):
        raise ValueError(f"Configured retriever mode does not expose candidate pooling: {mode}")

    company_name = query_context["company_name"]
    doc_ids = query_context["doc_ids"]
    search_queries = list(query_context["query_plan"].search_queries or [normalized["question_text"]])
    if not company_name and not doc_ids:
        raise ValueError("Candidate pool collection requires either company_name or doc_ids in the query record.")

    if len(search_queries) > 1:
        retrieval_runs = processor._retrieve_multi_query_candidate_runs(
            retriever=retriever,
            company_name=company_name or "",
            search_queries=search_queries,
            filters=query_context["query_plan"].filters,
            candidate_doc_ids=doc_ids or None,
        )
        merged_candidates = processor.merge_multi_query_candidates(retrieval_runs, pool_cap=candidate_pool_size)
    else:
        merged_candidates = retriever.retrieve_candidates_by_company_name(
            company_name=company_name or "",
            query=search_queries[0],
            top_n=per_query_retrieval_top_k,
            parent_retrieval_mode=processor.parent_retrieval_mode,
            filters=query_context["query_plan"].filters,
            candidate_doc_ids=doc_ids or None,
        )[:candidate_pool_size]

    candidates = [
        build_candidate_payload(result, rank_index=index)
        for index, result in enumerate(merged_candidates, start=1)
    ]
    return {
        "query_id": normalized["query_id"],
        "question_text": normalized["question_text"],
        "schema": normalized["schema"],
        "company_name": normalized["company_name"],
        "mentioned_companies": normalized["mentioned_companies"],
        "doc_ids": normalized["doc_ids"],
        "expected_filters": normalized["expected_filters"],
        "search_queries": search_queries,
        "retrieval_config": str(retrieval_config_path.relative_to(REPO_ROOT)),
        "candidates": candidates,
        "build_timestamp": utc_now_iso(),
    }


def _resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    default_config_path = REPO_ROOT / "training/reranker_distill/configs/data_build.example.yaml"
    config_path = args.config_path or (default_config_path if default_config_path.exists() else None)
    config = load_yaml_mapping(config_path)

    resume_config = config.get("resume")
    resume = True if args.resume else False if args.no_resume else bool(resume_config if resume_config is not None else True)

    retrieval_config_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(args.retrieval_config_path, config.get("retrieval_config_path")),
    )
    query_input_path = resolve_repo_path(REPO_ROOT, _coalesce(args.query_input_path, config.get("query_input_path")))
    candidate_output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.candidate_output_path, config.get("candidate_output_path")))
    stats_output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.stats_output_path, config.get("stats_output_path")))
    dataset_root_path = resolve_dataset_root(REPO_ROOT, _coalesce(args.dataset_root_path, config.get("dataset_root_path")))
    if retrieval_config_path is None or query_input_path is None or candidate_output_path is None or stats_output_path is None:
        raise ValueError("retrieval/query-output/stats paths are required.")

    return {
        "config_path": config_path,
        "retrieval_config_path": retrieval_config_path,
        "dataset_root_path": dataset_root_path,
        "query_input_path": query_input_path,
        "candidate_output_path": candidate_output_path,
        "stats_output_path": stats_output_path,
        "candidate_pool_size": max(1, int(_coalesce(args.candidate_pool_size, config.get("candidate_pool_size"), 32))),
        "per_query_retrieval_top_k": max(1, int(_coalesce(args.per_query_retrieval_top_k, config.get("per_query_retrieval_top_k"), 24))),
        "parallel_requests": max(1, int(_coalesce(args.parallel_requests, config.get("parallel_requests"), 1))),
        "resume": resume,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = _resolve_settings(args)
    records = load_records(settings["query_input_path"])
    existing_ids = collect_existing_ids(settings["candidate_output_path"]) if settings["resume"] else set()
    pending_records = [
        record
        for record in records
        if str(record.get("query_id") or "") not in existing_ids
    ]

    stats = {
        "build_timestamp": utc_now_iso(),
        "config_path": display_path(settings["config_path"], REPO_ROOT),
        "query_input_path": display_path(settings["query_input_path"], REPO_ROOT),
        "candidate_output_path": display_path(settings["candidate_output_path"], REPO_ROOT),
        "retrieval_config_path": display_path(settings["retrieval_config_path"], REPO_ROOT),
        "candidate_pool_size": settings["candidate_pool_size"],
        "per_query_retrieval_top_k": settings["per_query_retrieval_top_k"],
        "requested_queries": len(records),
        "skipped_existing": len(existing_ids & {str(record.get('query_id') or '') for record in records}),
        "processed_ok": 0,
        "processed_failed": 0,
        "failed_query_ids": [],
    }

    with ThreadPoolExecutor(max_workers=settings["parallel_requests"]) as executor:
        future_to_record = {
            executor.submit(
                collect_candidate_pool_record,
                record,
                retrieval_config_path=settings["retrieval_config_path"],
                dataset_root_path=settings["dataset_root_path"],
                candidate_pool_size=settings["candidate_pool_size"],
                per_query_retrieval_top_k=settings["per_query_retrieval_top_k"],
                parallel_requests=settings["parallel_requests"],
            ): record
            for record in pending_records
        }

        for future in as_completed(future_to_record):
            record = future_to_record[future]
            try:
                result = future.result()
            except Exception:
                stats["processed_failed"] += 1
                if record.get("query_id"):
                    stats["failed_query_ids"].append(str(record["query_id"]))
                continue
            append_jsonl(settings["candidate_output_path"], result)
            stats["processed_ok"] += 1

    write_json(settings["stats_output_path"], stats)


if __name__ == "__main__":
    main()
