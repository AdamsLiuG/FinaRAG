from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List
from urllib import request


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import (  # noqa: E402
    append_jsonl,
    collect_existing_ids,
    display_path,
    load_records,
    load_yaml_mapping,
    reset_output_file,
    resolve_repo_path,
    utc_now_iso,
    write_json,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score candidate passages with a /v1/rerank-compatible teacher endpoint.")
    parser.add_argument("--config-path", type=Path, default=None, help="YAML config path.")
    parser.add_argument("--candidate-input-path", type=Path, default=None, help="candidate_pool.jsonl path.")
    parser.add_argument("--output-path", type=Path, default=None, help="teacher_scores.jsonl path.")
    parser.add_argument("--stats-output-path", type=Path, default=None, help="Stats JSON path.")
    parser.add_argument("--teacher-base-url", default=None, help="Teacher reranker base URL.")
    parser.add_argument("--teacher-model", default=None, help="Teacher reranker model name.")
    parser.add_argument("--teacher-api-key-env", default=None, help="Env var name that stores the teacher API key.")
    parser.add_argument("--teacher-timeout-seconds", type=float, default=None, help="HTTP timeout in seconds.")
    parser.add_argument("--parallel-requests", type=int, default=None, help="Parallel query workers.")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional max query count.")
    parser.add_argument("--resume", action="store_true", help="Skip already-scored query_ids.")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume behavior.")
    return parser


def _coalesce(cli_value: Any, config_value: Any, default: Any = None) -> Any:
    return cli_value if cli_value is not None else (config_value if config_value is not None else default)


def _request_url(base_url: str) -> str:
    normalized_base = str(base_url or "").rstrip("/")
    if normalized_base.endswith("/chat/completions"):
        normalized_base = normalized_base[: -len("/chat/completions")]
    if normalized_base.endswith(("/rerank", "/v1/rerank", "/v2/rerank")):
        return normalized_base
    if normalized_base.endswith("/v1") or normalized_base.endswith("/v2"):
        return f"{normalized_base}/rerank"
    return f"{normalized_base}/v1/rerank"


def _normalize_remote_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    # Preserve the teacher's original score scale. Query-level min-max
    # normalization destroys absolute calibration and turns training targets
    # into pool-relative ranks only.
    return [float(score) for score in scores]


def _extract_scores(response_json: Dict[str, Any], total_documents: int) -> List[float]:
    results = response_json.get("results")
    if results is None:
        results = response_json.get("data")
    if not isinstance(results, list):
        raise ValueError(f"Unexpected rerank response payload: {response_json!r}")

    raw_scores = [0.0] * total_documents
    for item in results:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        score = item.get("relevance_score", item.get("score"))
        if index is None or score is None:
            continue
        try:
            normalized_index = int(index)
        except (TypeError, ValueError):
            continue
        if 0 <= normalized_index < total_documents:
            raw_scores[normalized_index] = float(score)
    return _normalize_remote_scores(raw_scores)


def _post_rerank(
    *,
    teacher_base_url: str,
    teacher_model: str,
    teacher_api_key_env: str | None,
    teacher_timeout_seconds: float,
    query: str,
    documents: List[str],
) -> Dict[str, Any]:
    payload = {
        "model": teacher_model,
        "query": query,
        "documents": documents,
        "top_n": len(documents),
        "return_documents": False,
    }
    headers = {
        "Content-Type": "application/json",
    }
    if teacher_api_key_env:
        api_key = os.getenv(teacher_api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

    http_request = request.Request(
        url=_request_url(teacher_base_url),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with request.urlopen(http_request, timeout=teacher_timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def score_candidate_pool_record(
    record: Dict[str, Any],
    *,
    teacher_base_url: str,
    teacher_model: str,
    teacher_api_key_env: str | None,
    teacher_timeout_seconds: float,
) -> List[Dict[str, Any]]:
    candidates = list(record.get("candidates") or [])
    documents = [str(candidate.get("text") or "") for candidate in candidates]
    if not documents:
        raise ValueError("Candidate record is missing candidates/text.")

    response_json = _post_rerank(
        teacher_base_url=teacher_base_url,
        teacher_model=teacher_model,
        teacher_api_key_env=teacher_api_key_env,
        teacher_timeout_seconds=teacher_timeout_seconds,
        query=str(record.get("question_text") or ""),
        documents=documents,
    )
    scores = _extract_scores(response_json, len(documents))

    ranked_indices = sorted(
        range(len(candidates)),
        key=lambda index: (-float(scores[index]), str(candidates[index].get("candidate_id") or "")),
    )
    teacher_rank_by_index = {
        index: rank
        for rank, index in enumerate(ranked_indices, start=1)
    }

    score_records = []
    for index, candidate in enumerate(candidates):
        score_records.append(
            {
                "query_id": record.get("query_id"),
                "candidate_id": candidate.get("candidate_id"),
                "question_text": record.get("question_text"),
                "schema": record.get("schema"),
                "teacher_reranker_model": teacher_model,
                "teacher_score": round(float(scores[index]), 4),
                "teacher_rank": int(teacher_rank_by_index[index]),
                "doc_id": candidate.get("doc_id"),
                "company_name": candidate.get("company_name"),
                "page": candidate.get("page"),
                "chunk_id": candidate.get("chunk_id"),
                "text": candidate.get("text"),
                "base_score": round(float(candidate.get("base_score") or 0.0), 4),
                "retrieval_sources": list(candidate.get("retrieval_sources", [])),
                "section_name": candidate.get("section_name"),
                "build_timestamp": utc_now_iso(),
            }
        )
    return score_records


def _resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    default_config_path = REPO_ROOT / "training/reranker_distill/configs/data_build.example.yaml"
    config_path = args.config_path or (default_config_path if default_config_path.exists() else None)
    config = load_yaml_mapping(config_path)

    resume_config = config.get("resume")
    resume = True if args.resume else False if args.no_resume else bool(resume_config if resume_config is not None else True)

    candidate_input_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(args.candidate_input_path, config.get("candidate_output_path") or config.get("candidate_input_path")),
    )
    output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.output_path, config.get("teacher_scores_path")))
    stats_output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.stats_output_path, config.get("teacher_score_stats_output_path") or config.get("stats_output_path")))
    if candidate_input_path is None or output_path is None or stats_output_path is None:
        raise ValueError("candidate_input/output/stats paths are required.")

    teacher_base_url = _coalesce(args.teacher_base_url, config.get("teacher_base_url"))
    teacher_model = _coalesce(args.teacher_model, config.get("teacher_model"))
    if not teacher_base_url or not teacher_model:
        raise ValueError("teacher_base_url and teacher_model are required.")

    return {
        "config_path": config_path,
        "candidate_input_path": candidate_input_path,
        "output_path": output_path,
        "stats_output_path": stats_output_path,
        "teacher_base_url": str(teacher_base_url),
        "teacher_model": str(teacher_model),
        "teacher_api_key_env": _coalesce(args.teacher_api_key_env, config.get("teacher_api_key_env")),
        "teacher_timeout_seconds": float(_coalesce(args.teacher_timeout_seconds, config.get("teacher_timeout_seconds"), 120.0)),
        "parallel_requests": max(1, int(_coalesce(args.parallel_requests, config.get("parallel_requests"), 1))),
        "max_queries": max(0, int(_coalesce(args.max_queries, config.get("max_queries"), 0) or 0)),
        "resume": resume,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = _resolve_settings(args)
    candidate_records = load_records(settings["candidate_input_path"])
    if settings["max_queries"] > 0:
        candidate_records = candidate_records[: settings["max_queries"]]

    if not settings["resume"]:
        reset_output_file(settings["output_path"])
    existing_ids = collect_existing_ids(settings["output_path"]) if settings["resume"] else set()
    pending_records = [
        record
        for record in candidate_records
        if str(record.get("query_id") or "") not in existing_ids
    ]

    stats = {
        "build_timestamp": utc_now_iso(),
        "config_path": display_path(settings["config_path"], REPO_ROOT),
        "candidate_input_path": display_path(settings["candidate_input_path"], REPO_ROOT),
        "output_path": display_path(settings["output_path"], REPO_ROOT),
        "teacher_model": settings["teacher_model"],
        "teacher_base_url": settings["teacher_base_url"],
        "parallel_requests": settings["parallel_requests"],
        "requested_queries": len(candidate_records),
        "skipped_existing": len(existing_ids & {str(record.get('query_id') or '') for record in candidate_records}),
        "processed_ok": 0,
        "processed_failed": 0,
        "failed_query_ids": [],
    }

    with ThreadPoolExecutor(max_workers=settings["parallel_requests"]) as executor:
        future_to_record = {
            executor.submit(
                score_candidate_pool_record,
                record,
                teacher_base_url=settings["teacher_base_url"],
                teacher_model=settings["teacher_model"],
                teacher_api_key_env=settings["teacher_api_key_env"],
                teacher_timeout_seconds=settings["teacher_timeout_seconds"],
            ): record
            for record in pending_records
        }

        for future in as_completed(future_to_record):
            record = future_to_record[future]
            try:
                score_records = future.result()
            except Exception:
                stats["processed_failed"] += 1
                if record.get("query_id"):
                    stats["failed_query_ids"].append(str(record["query_id"]))
                continue

            for score_record in score_records:
                append_jsonl(settings["output_path"], score_record)
            stats["processed_ok"] += 1

    write_json(settings["stats_output_path"], stats)


if __name__ == "__main__":
    main()
