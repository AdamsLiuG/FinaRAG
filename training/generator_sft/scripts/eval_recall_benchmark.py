from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import (  # noqa: E402
    build_query_context,
    build_questions_processor,
    display_path,
    load_records,
    load_yaml_mapping,
    resolve_dataset_root,
    resolve_repo_path,
    utc_now_iso,
    write_json,
    write_jsonl,
)

from src.hyde import HYDE_QUERY_MARKER, HyDEGenerator, should_trigger_hyde  # noqa: E402


_LEVELS = ("doc", "page", "chunk", "table")
_LEXICAL_SOURCES = {"bm25", "sparse", "tag"}
_DENSE_SOURCES = {"vector"}
_ATTRIBUTION_LABELS = ("BM25_only", "Dense_only", "Both", "Reranker_rescue", "Not_recalled", "Unknown")


@dataclass(frozen=True)
class BenchmarkSetting:
    top_n_retrieval: int
    candidate_pool_cap: int
    hyde_enabled: bool
    rewrite_enabled: bool

    @property
    def setting_id(self) -> str:
        return (
            f"top{self.top_n_retrieval}"
            f"_pool{self.candidate_pool_cap}"
            f"_hyde{'on' if self.hyde_enabled else 'off'}"
            f"_rewrite{'on' if self.rewrite_enabled else 'off'}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate retrieval recall for generator SFT benchmark queries.")
    parser.add_argument("--config-path", type=Path, default=None, help="Optional YAML config path.")
    parser.add_argument("--input-path", type=Path, default=None, help="Benchmark query JSONL/JSON path.")
    parser.add_argument("--summary-output-path", type=Path, default=None, help="Output JSON path for aggregated benchmark summary.")
    parser.add_argument("--details-output-path", type=Path, default=None, help="Output JSONL path for per-query benchmark details.")
    parser.add_argument("--retrieval-config-path", type=Path, default=None, help="Retrieval config YAML used by the online pipeline.")
    parser.add_argument("--dataset-root-path", type=Path, default=None, help="Dataset root holding databases, manifests, and chunked reports.")
    parser.add_argument("--top-n-values", nargs="*", type=int, default=None, help="Grid values for top_n_retrieval.")
    parser.add_argument("--candidate-pool-cap-values", nargs="*", type=int, default=None, help="Grid values for candidate pool cap.")
    parser.add_argument("--hyde-options", nargs="*", default=None, help="Grid values for HyDE toggle: on/off/true/false.")
    parser.add_argument("--rewrite-options", nargs="*", default=None, help="Grid values for query rewrite toggle: on/off/true/false.")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional benchmark query cap.")
    parser.add_argument("--parallel-requests", type=int, default=None, help="Forwarded to the shared QuestionsProcessor.")
    return parser


def _coalesce(cli_value: Any, config_value: Any, default: Any = None) -> Any:
    return cli_value if cli_value is not None else (config_value if config_value is not None else default)


def _parse_toggle_values(values: Sequence[Any] | None, *, default: bool) -> List[bool]:
    if not values:
        return [bool(default)]

    parsed: List[bool] = []
    for raw_value in values:
        text = str(raw_value).strip().lower()
        if text in {"1", "true", "on", "yes", "y"}:
            parsed.append(True)
        elif text in {"0", "false", "off", "no", "n"}:
            parsed.append(False)
        else:
            raise ValueError(f"Unsupported toggle value: {raw_value!r}")
    deduped: List[bool] = []
    for item in parsed:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_yaml_mapping(args.config_path)
    input_path = resolve_repo_path(REPO_ROOT, _coalesce(args.input_path, config.get("input_path")))
    summary_output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.summary_output_path, config.get("summary_output_path")))
    details_output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.details_output_path, config.get("details_output_path")))
    retrieval_config_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(args.retrieval_config_path, config.get("retrieval_config_path")),
    )
    dataset_root_path = resolve_dataset_root(REPO_ROOT, _coalesce(args.dataset_root_path, config.get("dataset_root_path")))
    if input_path is None or summary_output_path is None or details_output_path is None or retrieval_config_path is None:
        raise ValueError("input/summary/details/retrieval-config paths are required.")

    retrieval_config = load_yaml_mapping(retrieval_config_path)
    top_n_default = int(retrieval_config.get("top_n_retrieval", 10))
    pool_cap_default = int(retrieval_config.get("cascade_candidate_pool_cap", retrieval_config.get("llm_reranking_sample_size", 50)))
    hyde_default = bool(retrieval_config.get("hyde_enabled", False))

    top_n_values = [max(1, int(value)) for value in (_coalesce(args.top_n_values, config.get("top_n_values"), [top_n_default]) or [top_n_default])]
    candidate_pool_cap_values = [
        max(1, int(value))
        for value in (_coalesce(args.candidate_pool_cap_values, config.get("candidate_pool_cap_values"), [pool_cap_default]) or [pool_cap_default])
    ]
    hyde_options = _parse_toggle_values(_coalesce(args.hyde_options, config.get("hyde_options")), default=hyde_default)
    rewrite_options = _parse_toggle_values(_coalesce(args.rewrite_options, config.get("rewrite_options")), default=True)

    return {
        "config_path": args.config_path,
        "input_path": input_path,
        "summary_output_path": summary_output_path,
        "details_output_path": details_output_path,
        "retrieval_config_path": retrieval_config_path,
        "dataset_root_path": dataset_root_path,
        "top_n_values": top_n_values,
        "candidate_pool_cap_values": candidate_pool_cap_values,
        "hyde_options": hyde_options,
        "rewrite_options": rewrite_options,
        "max_queries": max(0, int(_coalesce(args.max_queries, config.get("max_queries"), 0) or 0)),
        "parallel_requests": max(1, int(_coalesce(args.parallel_requests, config.get("parallel_requests"), 1))),
    }


def _record_meta(record: Dict[str, Any]) -> Dict[str, Any]:
    meta = record.get("meta")
    return meta if isinstance(meta, dict) else {}


def _record_value(record: Dict[str, Any], field_name: str) -> Any:
    if field_name in record and record[field_name] not in (None, "", []):
        return record[field_name]
    meta = _record_meta(record)
    return meta.get(field_name)


def _coerce_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _stringify_list(values: Iterable[Any]) -> List[str]:
    return [str(value) for value in values if value not in (None, "")]


def _default_doc_ids_for_gold(record: Dict[str, Any]) -> List[str]:
    explicit = _stringify_list(record.get("gold_doc_ids") or (record.get("gold") or {}).get("doc_ids"))
    if explicit:
        return explicit
    return _stringify_list(_record_value(record, "doc_ids"))


def _normalize_gold_doc_ids(record: Dict[str, Any]) -> List[str]:
    return _default_doc_ids_for_gold(record)


def _normalize_gold_pages(record: Dict[str, Any]) -> List[str]:
    gold_blob = record.get("gold") if isinstance(record.get("gold"), dict) else {}
    raw_values = record.get("gold_pages") or gold_blob.get("pages") or []
    default_doc_ids = _default_doc_ids_for_gold(record)
    normalized: List[str] = []
    for item in _coerce_list(raw_values):
        if isinstance(item, dict):
            doc_id = str(item.get("doc_id") or item.get("pdf_sha1") or item.get("source") or "").strip()
            page = item.get("page")
        else:
            doc_id = default_doc_ids[0] if len(default_doc_ids) == 1 else ""
            page = item
        if not doc_id:
            continue
        try:
            normalized.append(f"{doc_id}::page::{int(page)}")
        except (TypeError, ValueError):
            continue
    return sorted(set(normalized))


def _normalize_gold_chunks(record: Dict[str, Any]) -> List[str]:
    gold_blob = record.get("gold") if isinstance(record.get("gold"), dict) else {}
    raw_values = record.get("gold_chunk_ids") or gold_blob.get("chunk_ids") or []
    default_doc_ids = _default_doc_ids_for_gold(record)
    normalized: List[str] = []
    for item in _coerce_list(raw_values):
        if isinstance(item, dict):
            doc_id = str(item.get("doc_id") or item.get("pdf_sha1") or item.get("source") or "").strip()
            chunk_id = item.get("chunk_id")
        else:
            doc_id = default_doc_ids[0] if len(default_doc_ids) == 1 else ""
            chunk_id = item
        if not doc_id or chunk_id in (None, ""):
            continue
        normalized.append(f"{doc_id}::chunk::{chunk_id}")
    return sorted(set(normalized))


def _normalize_gold_tables(record: Dict[str, Any]) -> List[str]:
    gold_blob = record.get("gold") if isinstance(record.get("gold"), dict) else {}
    raw_values = record.get("gold_table_ids") or gold_blob.get("table_ids") or []
    default_doc_ids = _default_doc_ids_for_gold(record)
    normalized: List[str] = []
    for item in _coerce_list(raw_values):
        if isinstance(item, dict):
            doc_id = str(item.get("doc_id") or item.get("pdf_sha1") or item.get("source") or "").strip()
            table_id = item.get("table_id")
        else:
            doc_id = default_doc_ids[0] if len(default_doc_ids) == 1 else ""
            table_id = item
        if not doc_id or table_id in (None, ""):
            continue
        normalized.append(f"{doc_id}::table::{table_id}")
    return sorted(set(normalized))


def _build_gold_units(record: Dict[str, Any]) -> Dict[str, List[str]]:
    return {
        "doc": _normalize_gold_doc_ids(record),
        "page": _normalize_gold_pages(record),
        "chunk": _normalize_gold_chunks(record),
        "table": _normalize_gold_tables(record),
    }


def _result_metadata(result: Dict[str, Any]) -> Dict[str, Any]:
    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    return result


def _result_doc_id(result: Dict[str, Any]) -> str:
    metadata = _result_metadata(result)
    return str(metadata.get("sha1_name") or metadata.get("doc_id") or result.get("doc_id") or "").strip()


def _extract_units_from_results(results: Sequence[Dict[str, Any]]) -> Dict[str, set[str]]:
    units = {
        "doc": set(),
        "page": set(),
        "chunk": set(),
        "table": set(),
    }
    for result in results:
        doc_id = _result_doc_id(result)
        if not doc_id:
            continue
        units["doc"].add(doc_id)
        page = result.get("page")
        if page not in (None, ""):
            try:
                units["page"].add(f"{doc_id}::page::{int(page)}")
            except (TypeError, ValueError):
                pass
        metadata = _result_metadata(result)
        chunk_id = result.get("chunk_id") or metadata.get("chunk_id")
        if chunk_id not in (None, ""):
            units["chunk"].add(f"{doc_id}::chunk::{chunk_id}")
        table_id = result.get("table_id") or metadata.get("table_id")
        if table_id not in (None, ""):
            units["table"].add(f"{doc_id}::table::{table_id}")
    return units


def _build_unit_source_map(results: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, set[str]]]:
    source_map = {
        "doc": {},
        "page": {},
        "chunk": {},
        "table": {},
    }
    for result in results:
        sources = {str(source) for source in result.get("retrieval_sources", []) if source}
        if not sources:
            continue
        doc_id = _result_doc_id(result)
        if not doc_id:
            continue
        keys: List[Tuple[str, str]] = [("doc", doc_id)]
        page = result.get("page")
        if page not in (None, ""):
            try:
                keys.append(("page", f"{doc_id}::page::{int(page)}"))
            except (TypeError, ValueError):
                pass
        metadata = _result_metadata(result)
        chunk_id = result.get("chunk_id") or metadata.get("chunk_id")
        if chunk_id not in (None, ""):
            keys.append(("chunk", f"{doc_id}::chunk::{chunk_id}"))
        table_id = result.get("table_id") or metadata.get("table_id")
        if table_id not in (None, ""):
            keys.append(("table", f"{doc_id}::table::{table_id}"))
        for level, key in keys:
            bucket = source_map[level].setdefault(key, set())
            bucket.update(sources)
    return source_map


def _score_level(gold_units: Sequence[str], retrieved_units: set[str]) -> Dict[str, Any]:
    gold_set = set(gold_units)
    if not gold_set:
        return {
            "gold_count": 0,
            "hit_count": 0,
            "recall": None,
            "matched_units": [],
            "missed_units": [],
        }
    hits = sorted(gold_set & retrieved_units)
    missed = sorted(gold_set - retrieved_units)
    return {
        "gold_count": len(gold_set),
        "hit_count": len(hits),
        "recall": len(hits) / len(gold_set),
        "matched_units": hits,
        "missed_units": missed,
    }


def _preferred_attribution_level(gold_units: Dict[str, List[str]]) -> str:
    for level in ("table", "chunk", "page", "doc"):
        if gold_units[level]:
            return level
    return "doc"


def _categorize_sources(sources: set[str]) -> str:
    dense_hit = bool(sources & _DENSE_SOURCES)
    lexical_hit = bool(sources & _LEXICAL_SOURCES)
    if dense_hit and lexical_hit:
        return "Both"
    if lexical_hit:
        return "BM25_only"
    if dense_hit:
        return "Dense_only"
    return "Unknown"


def _build_attribution(
    *,
    gold_units: Dict[str, List[str]],
    candidate_units: Dict[str, set[str]],
    pre_rerank_top_units: Dict[str, set[str]],
    final_units: Dict[str, set[str]],
    candidate_source_map: Dict[str, Dict[str, set[str]]],
    reranking_strategy: Optional[str],
) -> Dict[str, Any]:
    level = _preferred_attribution_level(gold_units)
    counts = Counter({label: 0 for label in _ATTRIBUTION_LABELS})
    unit_labels: Dict[str, str] = {}
    for gold_unit in gold_units[level]:
        if gold_unit not in final_units[level]:
            counts["Not_recalled"] += 1
            unit_labels[gold_unit] = "Not_recalled"
            continue
        if gold_unit in candidate_units[level] and gold_unit not in pre_rerank_top_units[level] and reranking_strategy:
            counts["Reranker_rescue"] += 1
            unit_labels[gold_unit] = "Reranker_rescue"
            continue
        source_label = _categorize_sources(candidate_source_map[level].get(gold_unit, set()))
        counts[source_label] += 1
        unit_labels[gold_unit] = source_label
    return {
        "attribution_level": level,
        "counts": dict(counts),
        "unit_labels": unit_labels,
    }


def _apply_setting_overrides(processor, setting: BenchmarkSetting) -> None:
    processor.top_n_retrieval = int(setting.top_n_retrieval)
    processor.cascade_candidate_pool_cap = int(setting.candidate_pool_cap)
    processor.llm_reranking_sample_size = max(int(processor.llm_reranking_sample_size), int(setting.candidate_pool_cap))
    processor.hyde_enabled = bool(setting.hyde_enabled)
    processor.hyde_trigger_mode = "fallback" if setting.hyde_enabled else "off"
    if setting.hyde_enabled:
        if processor.hyde_generator is None:
            processor.hyde_generator = HyDEGenerator(
                provider=processor.api_provider,
                model=processor.hyde_generation_model or processor.answering_model,
                temperature=processor.hyde_generation_temperature,
                max_tokens=processor.hyde_max_tokens,
                document_language=processor.document_language,
            )
    else:
        processor.hyde_generator = None


def _build_query_context_for_setting(processor, record: Dict[str, Any], *, rewrite_enabled: bool) -> Dict[str, Any]:
    query_context = build_query_context(processor, record)
    if not rewrite_enabled:
        query_context["query_plan"].search_queries = [str(query_context["normalized"]["question_text"])]
    return query_context


def _run_retrieval_only(processor, query_context: Dict[str, Any]) -> Dict[str, Any]:
    normalized = query_context["normalized"]
    question_text = normalized["question_text"]
    schema = normalized["schema"]
    company_name = query_context["company_name"] or ""
    route_info = query_context["route_info"]
    query_plan = query_context["query_plan"]
    candidate_doc_ids = list(route_info.get("candidate_doc_ids") or query_plan.filters.candidate_doc_ids or [])
    if candidate_doc_ids:
        query_plan.filters.candidate_doc_ids = candidate_doc_ids

    retriever, mode = processor._build_retriever()
    candidate_pool_size_before_rerank: Optional[int] = None
    reranking_debug = processor._build_reranking_debug_payload(retriever, candidate_pool_size_before_rerank)
    hyde_debug = processor._build_hyde_debug_payload(
        retrieval_results=[],
        candidate_pool_size_before_rerank=candidate_pool_size_before_rerank,
    )
    retrieval_results: List[Dict[str, Any]] = []
    merged_candidates: List[Dict[str, Any]] = []

    if mode == "full_context":
        retrieval_results = processor._run_retrieval(
            retriever,
            mode,
            company_name,
            question_text,
            query_plan.filters,
            candidate_doc_ids,
        )
        merged_candidates = [result.copy() for result in retrieval_results]
        reranking_debug = processor._build_reranking_debug_payload(retriever, candidate_pool_size_before_rerank)
        hyde_debug = processor._build_hyde_debug_payload(
            retrieval_results=retrieval_results,
            candidate_pool_size_before_rerank=candidate_pool_size_before_rerank,
        )
    elif mode == "hybrid_rerank":
        retrieval_runs = processor._retrieve_multi_query_candidate_runs(
            retriever=retriever,
            company_name=company_name,
            search_queries=query_plan.search_queries,
            filters=query_plan.filters,
            candidate_doc_ids=candidate_doc_ids,
        )
        candidate_pool_cap = processor._candidate_pool_cap(len(query_plan.search_queries))
        retrieval_results, merged_candidates = processor._rerank_merged_candidate_pool(
            retriever=retriever,
            question=question_text,
            retrieval_runs=retrieval_runs,
            pool_cap=candidate_pool_cap,
        )
        candidate_pool_size_before_rerank = len(merged_candidates)
        reranking_debug = processor._build_reranking_debug_payload(retriever, candidate_pool_size_before_rerank)
        hyde_debug = processor._build_hyde_debug_payload(
            retrieval_results=retrieval_results,
            candidate_pool_size_before_rerank=candidate_pool_size_before_rerank,
        )
        if processor._should_attempt_hyde(
            mode=mode,
            schema=schema,
            search_queries=query_plan.search_queries,
        ):
            hyde_triggered, hyde_reasons = should_trigger_hyde(
                retrieval_results=retrieval_results,
                top_score_threshold=processor.hyde_top_score_threshold,
                margin_threshold=processor.hyde_margin_threshold,
            )
            hyde_debug["triggered"] = hyde_triggered
            hyde_debug["trigger_reasons"] = hyde_reasons
            if hyde_triggered and processor.hyde_generator is not None:
                generated_hyde = processor.hyde_generator.generate(
                    question=question_text,
                    schema=schema,
                    query_plan=query_plan,
                    route_info=route_info,
                    model=processor.hyde_generation_model or processor.answering_model,
                    provider=processor.api_provider,
                )
                hyde_debug["generated_text"] = generated_hyde
                if generated_hyde:
                    hyde_candidate_runs = processor._retrieve_multi_query_candidate_runs(
                        retriever=retriever,
                        company_name=company_name,
                        search_queries=[generated_hyde],
                        filters=query_plan.filters,
                        candidate_doc_ids=candidate_doc_ids,
                        backend_scope="vector_only",
                    )
                    hyde_candidate_runs = [
                        (HYDE_QUERY_MARKER, results)
                        for _, results in hyde_candidate_runs
                    ]
                    if hyde_candidate_runs:
                        retrieval_results, merged_candidates = processor._rerank_merged_candidate_pool(
                            retriever=retriever,
                            question=question_text,
                            retrieval_runs=retrieval_runs + hyde_candidate_runs,
                            pool_cap=processor._candidate_pool_cap(len(query_plan.search_queries) + 1),
                        )
                        candidate_pool_size_before_rerank = len(merged_candidates)
                        reranking_debug = processor._build_reranking_debug_payload(retriever, candidate_pool_size_before_rerank)
                        hyde_debug["final_candidate_pool_size"] = candidate_pool_size_before_rerank
    else:
        retrieval_runs: List[Tuple[str, List[Dict[str, Any]]]] = []
        for search_query in query_plan.search_queries:
            results = processor._run_retrieval(
                retriever,
                mode,
                company_name,
                search_query,
                query_plan.filters,
                candidate_doc_ids,
            )
            if results:
                retrieval_runs.append((search_query, results))
        merged_candidates = processor.merge_multi_query_candidates(
            retrieval_runs,
            pool_cap=max(processor.top_n_retrieval, processor._candidate_pool_cap(len(query_plan.search_queries))),
        ) if retrieval_runs else []
        retrieval_results = processor._merge_multi_query_results(
            [results for _, results in retrieval_runs],
            processor.top_n_retrieval,
        ) if retrieval_runs else []
        reranking_debug = processor._build_reranking_debug_payload(retriever, candidate_pool_size_before_rerank)
        hyde_debug = processor._build_hyde_debug_payload(
            retrieval_results=retrieval_results,
            candidate_pool_size_before_rerank=candidate_pool_size_before_rerank,
        )

    return {
        "mode": mode,
        "query_plan": query_plan.to_dict(),
        "route_info": route_info,
        "search_queries": list(query_plan.search_queries or []),
        "retrieval_results": retrieval_results,
        "candidate_pool": merged_candidates,
        "candidate_pool_size_before_rerank": candidate_pool_size_before_rerank,
        "reranking_debug": reranking_debug,
        "hyde_debug": hyde_debug,
    }


def _evaluate_query(processor, record: Dict[str, Any], setting: BenchmarkSetting) -> Dict[str, Any]:
    query_context = _build_query_context_for_setting(processor, record, rewrite_enabled=setting.rewrite_enabled)
    normalized = query_context["normalized"]
    retrieval_payload = _run_retrieval_only(processor, query_context)
    final_results = retrieval_payload["retrieval_results"]
    candidate_pool = retrieval_payload["candidate_pool"] or final_results
    pre_rerank_top = list(candidate_pool[: setting.top_n_retrieval])

    gold_units = _build_gold_units(record)
    final_units = _extract_units_from_results(final_results)
    candidate_units = _extract_units_from_results(candidate_pool)
    pre_rerank_top_units = _extract_units_from_results(pre_rerank_top)
    candidate_source_map = _build_unit_source_map(candidate_pool)

    recall_by_level = {
        level: _score_level(gold_units[level], final_units[level])
        for level in _LEVELS
    }
    attribution = _build_attribution(
        gold_units=gold_units,
        candidate_units=candidate_units,
        pre_rerank_top_units=pre_rerank_top_units,
        final_units=final_units,
        candidate_source_map=candidate_source_map,
        reranking_strategy=retrieval_payload["reranking_debug"].get("reranking_strategy"),
    )

    return {
        "setting_id": setting.setting_id,
        "config": {
            "top_n_retrieval": setting.top_n_retrieval,
            "candidate_pool_cap": setting.candidate_pool_cap,
            "hyde_enabled": setting.hyde_enabled,
            "rewrite_enabled": setting.rewrite_enabled,
        },
        "query_id": normalized["query_id"],
        "question_text": normalized["question_text"],
        "schema": normalized["schema"],
        "company_name": normalized["company_name"],
        "doc_ids": normalized["doc_ids"],
        "search_queries": retrieval_payload["search_queries"],
        "retriever_mode": retrieval_payload["mode"],
        "candidate_pool_size_before_rerank": retrieval_payload["candidate_pool_size_before_rerank"],
        "retrieved_result_count": len(final_results),
        "gold_units": gold_units,
        "recall": recall_by_level,
        "attribution": attribution,
        "reranking_debug": retrieval_payload["reranking_debug"],
        "hyde": retrieval_payload["hyde_debug"],
        "build_timestamp": utc_now_iso(),
    }


def _aggregate_setting_results(setting: BenchmarkSetting, results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    level_totals = {
        level: {
            "gold_count": 0,
            "hit_count": 0,
            "query_count_with_gold": 0,
            "query_hit_count": 0,
        }
        for level in _LEVELS
    }
    attribution_counter = Counter({label: 0 for label in _ATTRIBUTION_LABELS})
    reranking_counter = Counter()
    total_candidate_pool = 0
    candidate_pool_queries = 0
    total_search_queries = 0

    for result in results:
        total_search_queries += len(result.get("search_queries", []))
        if result.get("candidate_pool_size_before_rerank") is not None:
            total_candidate_pool += int(result["candidate_pool_size_before_rerank"])
            candidate_pool_queries += 1
        reranking_counter[str(result.get("reranking_debug", {}).get("reranking_strategy") or "unknown")] += 1
        attribution_counter.update(result.get("attribution", {}).get("counts", {}))
        for level in _LEVELS:
            payload = result["recall"][level]
            if payload["gold_count"] <= 0:
                continue
            level_totals[level]["gold_count"] += int(payload["gold_count"])
            level_totals[level]["hit_count"] += int(payload["hit_count"])
            level_totals[level]["query_count_with_gold"] += 1
            if payload["hit_count"] > 0:
                level_totals[level]["query_hit_count"] += 1

    level_summary: Dict[str, Any] = {}
    micro_scores: List[float] = []
    for level, totals in level_totals.items():
        micro_recall = None
        query_hit_rate = None
        if totals["gold_count"] > 0:
            micro_recall = totals["hit_count"] / totals["gold_count"]
            micro_scores.append(micro_recall)
        if totals["query_count_with_gold"] > 0:
            query_hit_rate = totals["query_hit_count"] / totals["query_count_with_gold"]
        level_summary[level] = {
            **totals,
            "micro_recall": micro_recall,
            "query_hit_rate": query_hit_rate,
        }

    return {
        "setting_id": setting.setting_id,
        "config": asdict(setting),
        "query_count": len(results),
        "levels": level_summary,
        "overall_micro_recall_avg": (sum(micro_scores) / len(micro_scores)) if micro_scores else None,
        "average_candidate_pool_size_before_rerank": (total_candidate_pool / candidate_pool_queries) if candidate_pool_queries else None,
        "average_search_query_count": (total_search_queries / len(results)) if results else 0.0,
        "attribution_counts": dict(attribution_counter),
        "reranking_strategy_distribution": dict(reranking_counter),
    }


def _build_search_space(settings: Dict[str, Any]) -> List[BenchmarkSetting]:
    search_space = [
        BenchmarkSetting(
            top_n_retrieval=int(top_n),
            candidate_pool_cap=int(candidate_pool_cap),
            hyde_enabled=bool(hyde_enabled),
            rewrite_enabled=bool(rewrite_enabled),
        )
        for top_n, candidate_pool_cap, hyde_enabled, rewrite_enabled in itertools.product(
            settings["top_n_values"],
            settings["candidate_pool_cap_values"],
            settings["hyde_options"],
            settings["rewrite_options"],
        )
    ]
    deduped: List[BenchmarkSetting] = []
    seen = set()
    for item in search_space:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = _resolve_settings(args)
    records = load_records(settings["input_path"])
    if settings["max_queries"] > 0:
        records = records[: settings["max_queries"]]

    search_space = _build_search_space(settings)
    detailed_rows: List[Dict[str, Any]] = []
    config_summaries: List[Dict[str, Any]] = []

    for setting in search_space:
        processor = build_questions_processor(
            REPO_ROOT,
            settings["retrieval_config_path"],
            dataset_root=settings["dataset_root_path"],
            reasoning_debug_enabled=True,
            parallel_requests=settings["parallel_requests"],
        )
        _apply_setting_overrides(processor, setting)
        per_query_rows = [
            _evaluate_query(processor, record, setting)
            for record in records
        ]
        detailed_rows.extend(per_query_rows)
        config_summaries.append(_aggregate_setting_results(setting, per_query_rows))

    config_summaries.sort(
        key=lambda item: (
            item["overall_micro_recall_avg"] is not None,
            item["overall_micro_recall_avg"] or -1.0,
            item["levels"]["page"]["micro_recall"] or -1.0,
            item["levels"]["chunk"]["micro_recall"] or -1.0,
        ),
        reverse=True,
    )

    summary_payload = {
        "build_timestamp": utc_now_iso(),
        "config_path": display_path(settings["config_path"], REPO_ROOT),
        "input_path": display_path(settings["input_path"], REPO_ROOT),
        "summary_output_path": display_path(settings["summary_output_path"], REPO_ROOT),
        "details_output_path": display_path(settings["details_output_path"], REPO_ROOT),
        "retrieval_config_path": display_path(settings["retrieval_config_path"], REPO_ROOT),
        "dataset_root_path": display_path(settings["dataset_root_path"], REPO_ROOT),
        "requested_queries": len(records),
        "evaluated_queries": len(records),
        "search_space": {
            "top_n_values": settings["top_n_values"],
            "candidate_pool_cap_values": settings["candidate_pool_cap_values"],
            "hyde_options": settings["hyde_options"],
            "rewrite_options": settings["rewrite_options"],
        },
        "config_count": len(config_summaries),
        "best_config": config_summaries[0] if config_summaries else None,
        "config_results": config_summaries,
    }

    write_jsonl(settings["details_output_path"], detailed_rows)
    write_json(settings["summary_output_path"], summary_payload)


if __name__ == "__main__":
    main()
