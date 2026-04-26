from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import (  # noqa: E402
    append_jsonl,
    display_path,
    load_records,
    load_yaml_mapping,
    normalize_training_query_record,
    resolve_repo_path,
    utc_now_iso,
    write_json,
)
from training.generator_sft.validators import (  # noqa: E402
    boolean_answer_to_value,
    build_name_aliases,
    build_reject_log,
    classify_boolean_context,
    classify_name_support,
    compact_snippet,
    get_answer_dict,
    get_teacher_signal,
    normalize_pages,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build independent high-confidence anchor_clean_positive records.")
    parser.add_argument("--config-path", type=Path, default=None, help="YAML config path.")
    parser.add_argument("--raw-input-path", type=Path, default=None, help="Raw teacher answer JSONL path.")
    parser.add_argument("--retrieved-cache-input-path", type=Path, default=None, help="Optional retrieved_cache JSONL path.")
    parser.add_argument("--filtered-input-path", type=Path, default=None, help="Optional filtered sample JSONL path to backfill sample_id links.")
    parser.add_argument("--manual-override-input-path", type=Path, default=None, help="Optional manual override JSONL path.")
    parser.add_argument("--output-path", type=Path, default=None, help="Anchor output JSONL path.")
    parser.add_argument("--rejected-output-path", type=Path, default=None, help="Rejected anchor JSONL path.")
    parser.add_argument("--stats-output-path", type=Path, default=None, help="Anchor stats JSON path.")
    parser.add_argument("--max-records", type=int, default=None, help="Optional cap over raw records.")
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


def _group_sample_ids(records: Iterable[Dict[str, Any]]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for record in records:
        query_id = str(record.get("query_id") or "").strip()
        sample_id = str(record.get("sample_id") or "").strip()
        if query_id and sample_id:
            grouped[query_id].append(sample_id)
    return dict(grouped)


def _primary_doc_id(normalized: Dict[str, Any]) -> Optional[str]:
    doc_ids = [str(item) for item in normalized.get("doc_ids", []) if item not in (None, "")]
    return doc_ids[0] if len(doc_ids) == 1 else None


def _extract_number_anchor(record: Dict[str, Any], cache_record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    answer = get_answer_dict(record)
    final_answer = answer.get("final_answer")
    if final_answer in (None, "", [], "N/A"):
        return None

    teacher_signal = get_teacher_signal(record, cache_record)
    grounding = (
        record.get("table_grounding_result")
        if isinstance(record.get("table_grounding_result"), dict)
        else teacher_signal.get("table_grounding_result")
    )
    if not isinstance(grounding, dict) or grounding.get("normalized_value") is None:
        return None

    normalized = normalize_training_query_record(record)
    doc_id = str(grounding.get("doc_id") or _primary_doc_id(normalized) or "")
    page = grounding.get("page")
    return {
        "query_id": normalized["query_id"],
        "schema": "number",
        "question_text": normalized["question_text"],
        "company_name": normalized["company_name"],
        "doc_id": doc_id,
        "doc_ids": normalized["doc_ids"],
        "final_answer": grounding.get("normalized_value"),
        "answer_source_value": final_answer,
        "source_bucket": "table_grounding",
        "source_detail": "table_grounding_result",
        "anchor_pages": normalize_pages([page]),
        "anchor_evidence": [
            {
                "page": page,
                "table_id": grounding.get("table_id"),
                "snippet": compact_snippet(grounding.get("table_snippet") or grounding.get("cell_text") or ""),
                "support_type": "table_grounding",
            }
        ],
        "table_grounding": dict(grounding),
        "anchor_quality": "high",
    }


def _extract_name_anchor(record: Dict[str, Any], cache_record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    answer = get_answer_dict(record)
    final_answer = answer.get("final_answer")
    if final_answer in (None, "", [], "N/A"):
        return None

    support = classify_name_support(record, cache_record=cache_record)
    if support["support_type"] != "explicit_field_hit":
        return None

    normalized = normalize_training_query_record(record)
    doc_id = _primary_doc_id(normalized)
    if not doc_id:
        return None

    return {
        "query_id": normalized["query_id"],
        "schema": "name",
        "question_text": normalized["question_text"],
        "company_name": normalized["company_name"],
        "doc_id": doc_id,
        "doc_ids": normalized["doc_ids"],
        "final_answer": final_answer,
        "source_bucket": "explicit_field",
        "source_detail": "name_explicit_field_hit",
        "anchor_pages": support["support_pages"],
        "anchor_evidence": [
            {
                "page": hit.get("page"),
                "chunk_id": hit.get("chunk_id"),
                "snippet": hit.get("snippet"),
                "matched_labels": list(hit.get("matched_labels", []) or []),
                "support_type": "explicit_field_hit",
            }
            for hit in support["explicit_hits"]
        ],
        "field_keys": list((support.get("question_profile") or {}).get("field_keys", []) or []),
        "aliases": build_name_aliases(final_answer),
        "anchor_quality": "high",
    }


def _extract_boolean_anchor(record: Dict[str, Any], cache_record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    classification = classify_boolean_context(record, cache_record=cache_record)
    if classification["classification"] not in {"explicit_positive", "explicit_negative"}:
        return None

    normalized = normalize_training_query_record(record)
    doc_id = _primary_doc_id(normalized)
    if not doc_id:
        return None

    final_answer = classification["classification"] == "explicit_positive"
    teacher_answer = boolean_answer_to_value(get_answer_dict(record).get("final_answer"))
    hits = classification["positive_hits"] if final_answer else classification["negative_hits"]
    return {
        "query_id": normalized["query_id"],
        "schema": "boolean",
        "question_text": normalized["question_text"],
        "company_name": normalized["company_name"],
        "doc_id": doc_id,
        "doc_ids": normalized["doc_ids"],
        "final_answer": final_answer,
        "teacher_answer_value": teacher_answer,
        "source_bucket": "explicit_field",
        "source_detail": classification["classification"],
        "anchor_pages": classification["support_pages"],
        "anchor_evidence": [
            {
                "page": hit.get("page"),
                "chunk_id": hit.get("chunk_id"),
                "snippet": hit.get("snippet"),
                "support_type": hit.get("classification"),
            }
            for hit in hits
        ],
        "boolean_label": classification["classification"],
        "question_profile": classification["profile"],
        "anchor_quality": "high",
    }


def _build_anchor_record(
    record: Dict[str, Any],
    *,
    cache_record: Optional[Dict[str, Any]],
    linked_sample_ids: Optional[List[str]],
    anchor_index: int,
) -> Optional[Dict[str, Any]]:
    normalized = normalize_training_query_record(record)
    schema = str(normalized.get("schema") or "")
    if not normalized.get("query_id") or not normalized.get("question_text"):
        return None

    if schema == "number":
        anchor = _extract_number_anchor(record, cache_record)
    elif schema == "name":
        anchor = _extract_name_anchor(record, cache_record)
    elif schema == "boolean":
        anchor = _extract_boolean_anchor(record, cache_record)
    else:
        anchor = None
    if anchor is None:
        return None

    anchor["anchor_id"] = f"anchor-{anchor_index:06d}"
    anchor["linked_sample_ids"] = list(linked_sample_ids or [])
    anchor["build_timestamp"] = utc_now_iso()
    return anchor


def _normalize_manual_override(record: Dict[str, Any], anchor_index: int, linked_sample_ids: Optional[List[str]]) -> Dict[str, Any]:
    query_id = str(record.get("query_id") or "").strip()
    doc_ids = [str(item) for item in (record.get("doc_ids") or ([record.get("doc_id")] if record.get("doc_id") else [])) if item]
    doc_id = str(record.get("doc_id") or (doc_ids[0] if len(doc_ids) == 1 else "")).strip()
    anchor_pages = normalize_pages(record.get("anchor_pages") or record.get("pages"))
    schema = str(record.get("schema") or "").strip()
    final_answer = record.get("final_answer")
    anchor_record = {
        "anchor_id": f"anchor-{anchor_index:06d}",
        "query_id": query_id,
        "schema": schema,
        "question_text": record.get("question_text"),
        "company_name": record.get("company_name"),
        "doc_id": doc_id,
        "doc_ids": doc_ids or ([doc_id] if doc_id else []),
        "final_answer": final_answer,
        "source_bucket": "manual_override",
        "source_detail": str(record.get("source_detail") or record.get("note") or "manual_override"),
        "anchor_pages": anchor_pages,
        "anchor_evidence": list(record.get("anchor_evidence") or []),
        "field_keys": list(record.get("field_keys") or []),
        "aliases": list(record.get("aliases") or build_name_aliases(final_answer)),
        "boolean_label": record.get("boolean_label"),
        "table_grounding": record.get("table_grounding"),
        "question_profile": record.get("question_profile"),
        "linked_sample_ids": list(linked_sample_ids or []),
        "anchor_quality": str(record.get("anchor_quality") or "manual"),
        "build_timestamp": utc_now_iso(),
    }
    return anchor_record


def _resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    default_config_path = REPO_ROOT / "training/generator_sft/configs/anchor_clean_positive.example.yaml"
    config_path = args.config_path or (default_config_path if default_config_path.exists() else None)
    config = load_yaml_mapping(config_path)

    raw_input_path = resolve_repo_path(REPO_ROOT, _coalesce(args.raw_input_path, config.get("raw_input_path")))
    retrieved_cache_input_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(args.retrieved_cache_input_path, config.get("retrieved_cache_input_path")),
    )
    filtered_input_path = resolve_repo_path(REPO_ROOT, _coalesce(args.filtered_input_path, config.get("filtered_input_path")))
    manual_override_input_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(args.manual_override_input_path, config.get("manual_override_input_path")),
    )
    output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.output_path, config.get("output_path")))
    rejected_output_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(args.rejected_output_path, config.get("rejected_output_path")),
    )
    stats_output_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(args.stats_output_path, config.get("stats_output_path")),
    )
    if raw_input_path is None or output_path is None or rejected_output_path is None or stats_output_path is None:
        raise ValueError("raw/output/rejected/stats paths are required.")

    if retrieved_cache_input_path is not None and not retrieved_cache_input_path.exists():
        retrieved_cache_input_path = None
    if filtered_input_path is not None and not filtered_input_path.exists():
        filtered_input_path = None
    if manual_override_input_path is not None and not manual_override_input_path.exists():
        manual_override_input_path = None

    return {
        "config_path": config_path,
        "raw_input_path": raw_input_path,
        "retrieved_cache_input_path": retrieved_cache_input_path,
        "filtered_input_path": filtered_input_path,
        "manual_override_input_path": manual_override_input_path,
        "output_path": output_path,
        "rejected_output_path": rejected_output_path,
        "stats_output_path": stats_output_path,
        "max_records": max(0, int(_coalesce(args.max_records, config.get("max_records"), 0) or 0)),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = _resolve_settings(args)

    raw_records = load_records(settings["raw_input_path"])
    if settings["max_records"] > 0:
        raw_records = raw_records[: settings["max_records"]]
    cache_by_query_id = (
        _index_by_query_id(load_records(settings["retrieved_cache_input_path"]))
        if settings["retrieved_cache_input_path"] is not None
        else {}
    )
    linked_sample_ids = (
        _group_sample_ids(load_records(settings["filtered_input_path"]))
        if settings["filtered_input_path"] is not None
        else {}
    )
    manual_overrides = (
        load_records(settings["manual_override_input_path"])
        if settings["manual_override_input_path"] is not None
        else []
    )

    _reset_output_file(settings["output_path"])
    _reset_output_file(settings["rejected_output_path"])

    accepted = 0
    rejected = 0
    source_counter = Counter()
    schema_counter = Counter()
    rejection_counter = Counter()
    linked_sample_counter = Counter()
    anchors_by_query_id: Dict[str, Dict[str, Any]] = {}

    for raw_record in raw_records:
        normalized = normalize_training_query_record(raw_record)
        query_id = str(normalized.get("query_id") or "")
        if not query_id:
            rejected += 1
            rejection_counter["missing_query_id"] += 1
            append_jsonl(
                settings["rejected_output_path"],
                {
                    "query_id": None,
                    "schema": normalized.get("schema"),
                    "question_text": normalized.get("question_text"),
                    "reject_log": build_reject_log(
                        stage="anchor_builder",
                        reason_code="missing_query_id",
                        schema=str(normalized.get("schema") or ""),
                        query_id=None,
                    ),
                    "build_timestamp": utc_now_iso(),
                },
            )
            continue

        anchor_record = _build_anchor_record(
            raw_record,
            cache_record=cache_by_query_id.get(query_id),
            linked_sample_ids=linked_sample_ids.get(query_id),
            anchor_index=accepted + 1,
        )
        if anchor_record is None:
            rejected += 1
            rejection_counter["no_high_conf_anchor"] += 1
            append_jsonl(
                settings["rejected_output_path"],
                {
                    "query_id": query_id,
                    "schema": normalized.get("schema"),
                    "question_text": normalized.get("question_text"),
                    "final_answer": get_answer_dict(raw_record).get("final_answer"),
                    "reject_log": build_reject_log(
                        stage="anchor_builder",
                        reason_code="no_high_conf_anchor",
                        schema=str(normalized.get("schema") or ""),
                        query_id=query_id,
                        details={"doc_ids": normalized.get("doc_ids")},
                    ),
                    "build_timestamp": utc_now_iso(),
                },
            )
            continue

        accepted += 1
        anchors_by_query_id[query_id] = anchor_record
        append_jsonl(settings["output_path"], anchor_record)
        source_counter[str(anchor_record.get("source_bucket") or "")] += 1
        schema_counter[str(anchor_record.get("schema") or "")] += 1
        if anchor_record.get("linked_sample_ids"):
            linked_sample_counter["linked"] += 1
        else:
            linked_sample_counter["unlinked"] += 1

    for manual_record in manual_overrides:
        query_id = str(manual_record.get("query_id") or "").strip()
        if not query_id:
            continue
        accepted += 1
        anchor_record = _normalize_manual_override(
            manual_record,
            anchor_index=accepted,
            linked_sample_ids=linked_sample_ids.get(query_id),
        )
        anchors_by_query_id[query_id] = anchor_record
        append_jsonl(settings["output_path"], anchor_record)
        source_counter["manual_override"] += 1
        schema_counter[str(anchor_record.get("schema") or "")] += 1
        if anchor_record.get("linked_sample_ids"):
            linked_sample_counter["linked"] += 1
        else:
            linked_sample_counter["unlinked"] += 1

    stats = {
        "build_timestamp": utc_now_iso(),
        "config_path": display_path(settings["config_path"], REPO_ROOT),
        "raw_input_path": display_path(settings["raw_input_path"], REPO_ROOT),
        "retrieved_cache_input_path": display_path(settings["retrieved_cache_input_path"], REPO_ROOT),
        "filtered_input_path": display_path(settings["filtered_input_path"], REPO_ROOT),
        "manual_override_input_path": display_path(settings["manual_override_input_path"], REPO_ROOT),
        "output_path": display_path(settings["output_path"], REPO_ROOT),
        "rejected_output_path": display_path(settings["rejected_output_path"], REPO_ROOT),
        "requested_records": len(raw_records),
        "accepted_records": accepted,
        "rejected_records": rejected,
        "source_bucket_distribution": dict(source_counter),
        "schema_distribution": dict(schema_counter),
        "linked_sample_distribution": dict(linked_sample_counter),
        "rejection_distribution": dict(rejection_counter),
    }
    write_json(settings["stats_output_path"], stats)


if __name__ == "__main__":
    main()
