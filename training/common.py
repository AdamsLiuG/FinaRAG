from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional

import yaml
from dotenv import load_dotenv

if TYPE_CHECKING:
    from src.questions_processing import QuestionsProcessor
    from src.retrieval_filters import RetrievalFilters


_QUERY_RECORD_LIST_KEYS = ("questions", "records", "data")
_PROJECT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_RUN_CONFIG_DEFAULTS: Dict[str, Any] = {
    "use_serialized_tables": False,
    "parent_document_retrieval": False,
    "parent_retrieval_mode": "page",
    "use_vector_dbs": True,
    "vector_search_k": 0,
    "vector_ivf_nprobe": 8,
    "vector_hnsw_ef_search": 64,
    "retriever_cache_enabled": True,
    "use_bm25_db": False,
    "use_sparse_lexical_db": False,
    "use_tag_db": True,
    "llm_reranking": False,
    "llm_reranking_sample_size": 30,
    "top_n_retrieval": 10,
    "parallel_requests": 10,
    "full_context": False,
    "api_provider": "qwen",
    "answering_model": "Qwen3.5-35B-A3B-AWQ-4bit",
    "document_language": "en",
    "doc_router_enabled": False,
    "candidate_doc_top_k": 5,
    "numeric_grounding_enabled": False,
    "reasoning_debug_enabled": True,
    "hyde_enabled": False,
    "hyde_trigger_mode": "off",
    "hyde_generation_model": None,
    "hyde_generation_temperature": 0.2,
    "hyde_max_tokens": 192,
    "hyde_top_score_threshold": 0.55,
    "hyde_margin_threshold": 0.05,
    "reranking_strategy": "single",
    "cascade_candidate_pool_cap": 50,
    "colbert_top_n": 10,
    "colbert_model": None,
    "colbert_device": None,
    "colbert_batch_size": 16,
    "colbert_query_max_length": 128,
    "colbert_passage_max_length": 512,
    "final_reranking_backend": None,
    "final_reranking_model": None,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def resolve_repo_path(repo_root: Path, value: str | Path | None) -> Optional[Path]:
    if value in (None, ""):
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def resolve_dataset_root(repo_root: Path, value: str | Path | None) -> Path:
    resolved = resolve_repo_path(repo_root, value)
    return resolved if resolved is not None else repo_root


def display_path(path: Path | None, repo_root: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _expand_env_placeholders(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env_placeholders(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_placeholders(item) for item in value]
    if not isinstance(value, str) or "${" not in value:
        return value

    def _replace(match: re.Match[str]) -> str:
        env_name = match.group(1)
        default_value = match.group(2)
        resolved = os.getenv(env_name)
        if resolved is None:
            if default_value is None:
                raise ValueError(f"Missing environment variable '{env_name}' required by config placeholder.")
            resolved = default_value
        return resolved

    substituted = _ENV_VAR_PATTERN.sub(_replace, value)
    if _ENV_VAR_PATTERN.fullmatch(value.strip()):
        if substituted == "":
            return ""
        parsed = yaml.safe_load(substituted)
        return substituted if parsed is None else parsed
    return substituted


def load_yaml_mapping(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    load_dotenv(_PROJECT_ENV_PATH)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return _expand_env_placeholders(payload)


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSONL line {line_number} from {path}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object on line {line_number} in {path}.")
            yield payload


def load_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return list(read_jsonl(path))

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in _QUERY_RECORD_LIST_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported input structure in {path}. Expected JSONL or a JSON array.")


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_records(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    if path.suffix.lower() == ".json":
        payload = list(records)
        ensure_parent(path)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    write_jsonl(path, records)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect_existing_ids(path: Path, id_field: str = "query_id") -> set[str]:
    if not path.exists():
        return set()
    existing_ids = set()
    for record in read_jsonl(path):
        value = record.get(id_field)
        if value not in (None, ""):
            existing_ids.add(str(value))
    return existing_ids


def _dedupe_preserve_order(values: Iterable[Any]) -> List[Any]:
    deduped: List[Any] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _coerce_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_training_query_record(record: Dict[str, Any]) -> Dict[str, Any]:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    answer = record.get("answer") if isinstance(record.get("answer"), dict) else {}
    route_info = record.get("route_info") if isinstance(record.get("route_info"), dict) else {}
    if not route_info:
        route_info = answer.get("route_info") if isinstance(answer.get("route_info"), dict) else {}

    query_plan = record.get("query_plan") if isinstance(record.get("query_plan"), dict) else {}
    if not query_plan:
        query_plan = answer.get("query_plan") if isinstance(answer.get("query_plan"), dict) else {}

    question_text = (
        record.get("question_text")
        or record.get("query")
        or record.get("question")
        or record.get("text")
        or meta.get("question_text")
        or query_plan.get("original_query")
    )
    schema = record.get("schema") or record.get("kind") or meta.get("schema")
    query_id = record.get("query_id") or record.get("id") or meta.get("query_id")

    company_name = (
        record.get("company_name")
        or meta.get("company_name")
        or route_info.get("selected_company")
        or (query_plan.get("filters") or {}).get("company_name")
    )
    mentioned_companies = _coerce_list(record.get("mentioned_companies")) or _coerce_list(query_plan.get("mentioned_companies"))
    if not mentioned_companies and company_name:
        mentioned_companies = [company_name]

    doc_ids = _coerce_list(record.get("doc_ids")) or _coerce_list(meta.get("doc_ids"))
    if not doc_ids:
        references = _coerce_list(record.get("references")) or _coerce_list(answer.get("references"))
        doc_ids = [
            reference.get("pdf_sha1")
            for reference in references
            if isinstance(reference, dict) and reference.get("pdf_sha1")
        ]
    if not doc_ids:
        retrieval_groups = _coerce_list(record.get("retrieval_report_groups")) or _coerce_list(answer.get("retrieval_report_groups"))
        doc_ids = [
            group.get("doc_id")
            for group in retrieval_groups
            if isinstance(group, dict) and group.get("doc_id")
        ]

    return {
        "query_id": str(query_id) if query_id not in (None, "") else None,
        "question_text": question_text,
        "schema": schema,
        "company_name": company_name,
        "mentioned_companies": _dedupe_preserve_order(str(value) for value in mentioned_companies if value not in (None, "")),
        "doc_ids": _dedupe_preserve_order(str(value) for value in doc_ids if value not in (None, "")),
        "expected_filters": record.get("expected_filters") or {},
        "source": record.get("source") or meta.get("source"),
        "difficulty": record.get("difficulty"),
        "should_refuse": bool(record.get("should_refuse", False)),
        "route_info": route_info,
        "query_plan": query_plan,
        "original_record": record,
    }


def compact_json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def stable_hash_int(value: Any, *, salt: str = "") -> int:
    if isinstance(value, str):
        raw = value
    else:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(f"{salt}::{raw}".encode("utf-8")).hexdigest()
    return int(digest, 16)


def deterministic_split_for_key(
    key: Any,
    *,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    salt: str = "",
) -> str:
    if dev_ratio < 0 or test_ratio < 0 or (dev_ratio + test_ratio) >= 1:
        raise ValueError("`dev_ratio` and `test_ratio` must be non-negative and sum to less than 1.")

    bucket = (stable_hash_int(key, salt=salt) % 1_000_000) / 1_000_000.0
    if bucket < test_ratio:
        return "test"
    if bucket < (test_ratio + dev_ratio):
        return "dev"
    return "train"


def _record_field(record: Dict[str, Any], field_name: str) -> Any:
    if field_name in record and record[field_name] not in (None, "", []):
        return record[field_name]
    meta = record.get("meta")
    if isinstance(meta, dict):
        value = meta.get(field_name)
        if value not in (None, "", []):
            return value
    return None


def build_split_group_key(record: Dict[str, Any], preferred_fields: Iterable[str] | None = None) -> str:
    fields = list(preferred_fields or ("doc_ids", "doc_id", "company_name", "query_id", "sample_id"))
    for field_name in fields:
        value = _record_field(record, field_name)
        if value in (None, "", []):
            continue
        if isinstance(value, list):
            deduped = [str(item) for item in value if item not in (None, "")]
            if deduped:
                return f"{field_name}:{'|'.join(deduped)}"
            continue
        return f"{field_name}:{value}"

    fallback = compact_json_dumps(record)
    return f"fallback:{stable_hash_int(fallback)}"


def build_rag_prompt_bundle(schema: str, provider: str = "qwen") -> tuple[str, str, Any]:
    import src.prompts as prompts

    prompt_map = {
        "name": prompts.AnswerWithRAGContextNamePrompt,
        "number": prompts.AnswerWithRAGContextNumberPrompt,
        "boolean": prompts.AnswerWithRAGContextBooleanPrompt,
        "names": prompts.AnswerWithRAGContextNamesPrompt,
        "comparative": prompts.ComparativeAnswerPrompt,
    }
    prompt_cls = prompt_map.get(schema)
    if prompt_cls is None:
        raise ValueError(f"Unsupported schema for prompt bundle: {schema}")

    use_schema_prompt = provider in {"ibm", "gemini", "qwen"}
    system_prompt = prompt_cls.system_prompt_with_schema if use_schema_prompt else prompt_cls.system_prompt
    return system_prompt, prompt_cls.user_prompt, prompt_cls.AnswerSchema


def schema_field_names(model_cls: Any) -> List[str]:
    model_fields = getattr(model_cls, "model_fields", None)
    if isinstance(model_fields, dict):
        return list(model_fields.keys())

    legacy_fields = getattr(model_cls, "__fields__", None)
    if isinstance(legacy_fields, dict):
        return list(legacy_fields.keys())

    raise ValueError(f"Unsupported schema model: {model_cls!r}")


def prune_answer_to_schema(answer: Dict[str, Any], schema: str, provider: str = "qwen") -> Dict[str, Any]:
    _, _, response_format = build_rag_prompt_bundle(schema, provider=provider)
    fields = schema_field_names(response_format)
    return {field_name: answer.get(field_name) for field_name in fields if field_name in answer}


def _apply_expected_filters(filters: "RetrievalFilters", expected_filters: Dict[str, Any]) -> None:
    if not isinstance(expected_filters, dict):
        return

    alias_map = {
        "report_year": "year",
        "candidate_doc_ids": "candidate_doc_ids",
    }
    list_fields = {
        "business_tags",
        "strategy_tags",
        "factor_tags",
        "chain_position_minor",
        "listing_tags",
        "ownership_tags",
        "status_tags",
        "style_tags",
        "required_topic_flags",
        "candidate_doc_ids",
    }

    for key, value in expected_filters.items():
        field_name = alias_map.get(key, key)
        if not hasattr(filters, field_name):
            continue
        if field_name in list_fields:
            setattr(filters, field_name, [item for item in _coerce_list(value) if item not in (None, "")])
        else:
            setattr(filters, field_name, value)


def build_query_context(processor: "QuestionsProcessor", record: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_training_query_record(record)
    question_text = normalized["question_text"]
    schema = normalized["schema"]
    if not question_text:
        raise ValueError("Query record is missing `question_text` / `query`.")
    if not schema:
        raise ValueError("Query record is missing `schema` / `kind`.")

    route_info = dict(normalized["route_info"] or {})
    company_name = normalized["company_name"] or route_info.get("selected_company") or ""
    mentioned_companies = list(
        normalized["mentioned_companies"]
        or route_info.get("candidate_companies")
        or ([company_name] if company_name else [])
    )

    query_plan_blob = normalized["query_plan"] or {}
    route_mode = route_info.get("route_mode") or query_plan_blob.get("route_mode")
    if not route_mode:
        route_mode = "explicit_company" if company_name else "document_catalog"
        if normalized["doc_ids"] and not company_name:
            route_mode = "document_catalog_multi"

    query_plan = processor._build_query_plan(
        question_text,
        schema=schema,
        company_name=company_name or None,
        mentioned_companies=mentioned_companies or None,
        route_mode=route_mode,
    )

    if isinstance(query_plan_blob, dict):
        search_queries = query_plan_blob.get("search_queries")
        if isinstance(search_queries, list) and search_queries:
            query_plan.search_queries = [str(item) for item in search_queries if str(item).strip()]
        topic_flags = query_plan_blob.get("topic_flags")
        if isinstance(topic_flags, list):
            query_plan.topic_flags = [str(item) for item in topic_flags if str(item).strip()]
        route_hints = query_plan_blob.get("route_hints")
        if isinstance(route_hints, dict):
            query_plan.route_hints = dict(route_hints)
        if query_plan_blob.get("route_mode"):
            query_plan.route_mode = str(query_plan_blob["route_mode"])

    _apply_expected_filters(query_plan.filters, normalized["expected_filters"])
    if normalized["doc_ids"]:
        query_plan.filters.candidate_doc_ids = list(normalized["doc_ids"])
    if route_mode == "document_catalog_multi":
        query_plan.filters.company_name = None
    elif company_name:
        query_plan.filters.company_name = company_name

    if not route_info:
        route_info = {
            "route_mode": route_mode,
            "selected_company": company_name or None,
            "candidate_companies": mentioned_companies,
            "candidate_doc_ids": list(normalized["doc_ids"]),
            "selection_reasons": ["training_query_record"],
        }
        if len(normalized["doc_ids"]) == 1:
            route_info["selected_report"] = {"sha1": normalized["doc_ids"][0]}

    return {
        "normalized": normalized,
        "query_plan": query_plan,
        "route_info": route_info,
        "company_name": company_name,
        "mentioned_companies": mentioned_companies,
        "doc_ids": list(normalized["doc_ids"]),
    }


def inflate_serialized_retrieval_results(serialized_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    inflated_results: List[Dict[str, Any]] = []
    metadata_fields = {
        "chunk_id",
        "chunk_type",
        "node_type",
        "parent_chunk_id",
        "section_title",
        "section_name",
        "report_section",
        "company_name",
        "company_aliases",
        "security_code",
        "stock_code",
        "broker_name",
        "currency",
        "exchange",
        "board",
        "market_type",
        "industry_l1",
        "industry_l2",
        "report_year",
        "report_type",
        "doc_source_type",
        "report_date",
        "fiscal_year",
        "period",
        "unit_hint",
        "language",
        "topic_flags",
        "business_tags",
        "strategy_tags",
        "factor_tags",
        "chain_position_major",
        "chain_position_minor",
        "listing_tags",
        "ownership_tags",
        "status_tags",
        "style_tags",
        "table_id",
    }

    for item in serialized_results or []:
        if not isinstance(item, dict):
            continue
        metadata = {field: item.get(field) for field in metadata_fields if field in item}
        inflated_results.append(
            {
                "page": item.get("page"),
                "text": item.get("text", ""),
                "matched_child_chunk_ids": item.get("matched_child_chunk_ids", []),
                "matched_tags": item.get("matched_tags", []),
                "matched_queries": item.get("matched_queries", []),
                "query_hit_count": item.get("query_hit_count"),
                "result_scope": item.get("result_scope"),
                "retrieval_sources": item.get("retrieval_sources", []),
                "metadata": metadata,
            }
        )
    return inflated_results


def build_questions_processor(
    repo_root: Path,
    retrieval_config_path: Path,
    *,
    dataset_root: Optional[Path] = None,
    api_provider: Optional[str] = None,
    answering_model: Optional[str] = None,
    answer_temperature: float = 0.0,
    reasoning_debug_enabled: Optional[bool] = None,
    parallel_requests: Optional[int] = None,
) -> "QuestionsProcessor":
    from src.questions_processing import QuestionsProcessor

    run_config = dict(_RUN_CONFIG_DEFAULTS)
    run_config.update(load_yaml_mapping(retrieval_config_path))
    if api_provider:
        run_config["api_provider"] = api_provider
    if answering_model:
        run_config["answering_model"] = answering_model
    if reasoning_debug_enabled is not None:
        run_config["reasoning_debug_enabled"] = bool(reasoning_debug_enabled)
    if parallel_requests is not None:
        run_config["parallel_requests"] = int(parallel_requests)

    workspace_root = resolve_dataset_root(repo_root, dataset_root)
    databases_suffix = "_ser_tab" if run_config["use_serialized_tables"] else ""
    databases_path = workspace_root / f"databases{databases_suffix}"
    document_manifest_path = workspace_root / "document_manifest.csv"
    if not document_manifest_path.exists():
        json_manifest = workspace_root / "document_manifest.json"
        if json_manifest.exists():
            document_manifest_path = json_manifest
        else:
            document_manifest_path = workspace_root / "subset.csv"

    return QuestionsProcessor(
        vector_db_dir=databases_path / "vector_dbs",
        bm25_db_path=databases_path / "bm25_dbs",
        sparse_db_dir=databases_path / "sparse_dbs",
        tag_db_dir=databases_path / "tag_dbs",
        documents_dir=databases_path / "chunked_reports",
        subset_path=document_manifest_path,
        parent_document_retrieval=run_config["parent_document_retrieval"],
        parent_retrieval_mode=run_config["parent_retrieval_mode"],
        use_vector_dbs=run_config["use_vector_dbs"],
        use_bm25_db=run_config["use_bm25_db"],
        use_sparse_lexical_db=run_config["use_sparse_lexical_db"],
        use_tag_db=run_config["use_tag_db"],
        llm_reranking=run_config["llm_reranking"],
        llm_reranking_sample_size=run_config["llm_reranking_sample_size"],
        top_n_retrieval=run_config["top_n_retrieval"],
        vector_search_k=run_config["vector_search_k"],
        vector_ivf_nprobe=run_config["vector_ivf_nprobe"],
        vector_hnsw_ef_search=run_config["vector_hnsw_ef_search"],
        retriever_cache_enabled=run_config["retriever_cache_enabled"],
        parallel_requests=run_config["parallel_requests"],
        api_provider=run_config["api_provider"],
        answering_model=run_config["answering_model"],
        answer_temperature=answer_temperature,
        full_context=run_config["full_context"],
        document_language=run_config["document_language"],
        doc_router_enabled=run_config["doc_router_enabled"],
        candidate_doc_top_k=run_config["candidate_doc_top_k"],
        numeric_grounding_enabled=run_config["numeric_grounding_enabled"],
        reasoning_debug_enabled=run_config["reasoning_debug_enabled"],
        hyde_enabled=run_config["hyde_enabled"],
        hyde_trigger_mode=run_config["hyde_trigger_mode"],
        hyde_generation_model=run_config["hyde_generation_model"],
        hyde_generation_temperature=run_config["hyde_generation_temperature"],
        hyde_max_tokens=run_config["hyde_max_tokens"],
        hyde_top_score_threshold=run_config["hyde_top_score_threshold"],
        hyde_margin_threshold=run_config["hyde_margin_threshold"],
        reranking_strategy=run_config["reranking_strategy"],
        cascade_candidate_pool_cap=run_config["cascade_candidate_pool_cap"],
        colbert_top_n=run_config["colbert_top_n"],
        colbert_model=run_config["colbert_model"],
        colbert_device=run_config["colbert_device"],
        colbert_batch_size=run_config["colbert_batch_size"],
        colbert_query_max_length=run_config["colbert_query_max_length"],
        colbert_passage_max_length=run_config["colbert_passage_max_length"],
        final_reranking_backend=run_config["final_reranking_backend"],
        final_reranking_model=run_config["final_reranking_model"],
    )
