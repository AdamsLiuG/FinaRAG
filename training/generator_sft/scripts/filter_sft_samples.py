from __future__ import annotations

import argparse
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Tuple


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
    normalize_training_query_record,
    prune_answer_to_schema,
    resolve_repo_path,
    stable_hash_int,
    utc_now_iso,
    write_json,
)
from training.generator_sft.validators import (  # noqa: E402
    normalize_pages,
    validate_boolean_answer,
    validate_name_answer,
)


_SEVERE_VALIDATION_FLAGS = {
    "no_retrieval_results",
    "currency_mismatch",
    "report_year_mismatch",
    "doc_source_type_mismatch",
    "numeric_grounding_missing_value",
    "numeric_grounding_period_mismatch",
    "numeric_grounding_currency_mismatch",
    "numeric_answer_without_table_grounding",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter raw teacher answers into generator SFT-ready records.")
    parser.add_argument("--config-path", type=Path, default=None, help="YAML config path.")
    parser.add_argument("--input-path", type=Path, default=None, help="teacher_answers_raw.jsonl path.")
    parser.add_argument("--retrieved-cache-input-path", type=Path, default=None, help="Optional retrieved_cache JSONL path used by validators.")
    parser.add_argument("--anchor-input-path", type=Path, default=None, help="Optional anchor_clean_positive JSONL path.")
    parser.add_argument("--output-path", type=Path, default=None, help="teacher_answers_filtered.jsonl path.")
    parser.add_argument("--rejected-output-path", type=Path, default=None, help="Rejected sample JSONL path.")
    parser.add_argument("--stats-output-path", type=Path, default=None, help="Stats JSON path.")
    parser.add_argument("--allow-missing-citations", action="store_true", help="Reserved switch; citations are ignored by default.")
    return parser


def _coalesce(cli_value: Any, config_value: Any, default: Any = None) -> Any:
    return cli_value if cli_value is not None else (config_value if config_value is not None else default)


def _is_empty_final_answer(value: Any) -> bool:
    return value is None or value == "" or value == []


def _required_fields_present(record: Dict[str, Any], normalized: Dict[str, Any]) -> bool:
    return bool(
        normalized["query_id"]
        and normalized["question_text"]
        and normalized["schema"]
        and isinstance(record.get("answer"), dict)
    )


def _extract_table_grounding_result(record: Dict[str, Any]) -> Dict[str, Any]:
    grounding = record.get("table_grounding_result")
    if isinstance(grounding, dict):
        return grounding
    answer = record.get("answer") if isinstance(record.get("answer"), dict) else {}
    grounding = answer.get("table_grounding_result")
    if isinstance(grounding, dict):
        return grounding
    return {}


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


def _enforce_number_answer_consistency(record: Dict[str, Any], cleaned_answer: Dict[str, Any]) -> Dict[str, Any]:
    final_answer = cleaned_answer.get("final_answer")
    if final_answer == "N/A":
        return cleaned_answer

    grounding = _extract_table_grounding_result(record)
    normalized_value = grounding.get("normalized_value")
    if normalized_value is None:
        raise ValueError("number_grounding_missing_normalized_value")

    answer_decimal = _coerce_decimal(final_answer)
    if answer_decimal is None:
        raise ValueError("number_final_answer_not_numeric")

    grounding_decimal = _coerce_decimal(normalized_value)
    if grounding_decimal is None:
        raise ValueError("number_grounding_non_numeric")
    if answer_decimal != grounding_decimal:
        raise ValueError(f"number_final_answer_grounding_mismatch:{final_answer}!={normalized_value}")

    canonical_answer = dict(cleaned_answer)
    canonical_answer["final_answer"] = normalized_value
    return canonical_answer


def _requires_explicit_legal_representative_grounding(question_text: str, final_answer: Any) -> bool:
    if final_answer == "N/A":
        return False
    lowered_question = question_text.lower()
    return "法定代表人" in question_text or "legal representative" in lowered_question


def _has_explicit_legal_representative_grounding(record: Dict[str, Any]) -> bool:
    rag_context = str(record.get("rag_context") or "")
    return "法定代表人" in rag_context or "legal representative" in rag_context.lower()


def _index_records_by_query_id(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for record in records:
        query_id = str(record.get("query_id") or "").strip()
        if query_id:
            indexed[query_id] = record
    return indexed


def _apply_boolean_validation(
    record: Dict[str, Any],
    cleaned_answer: Dict[str, Any],
    *,
    anchor_record: Dict[str, Any] | None,
    cache_record: Dict[str, Any] | None,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    working_record = dict(record)
    working_record["answer"] = dict(cleaned_answer)
    validation_result = validate_boolean_answer(
        working_record,
        anchor_record=anchor_record,
        cache_record=cache_record,
    )
    if validation_result["decision"] == "reject":
        raise ValueError(validation_result["decision_reason"])

    accepted_checks = list(validation_result.get("accepted_checks") or [])
    canonical_answer = dict(cleaned_answer)
    canonical_answer["final_answer"] = validation_result.get("normalized_final_answer")
    if validation_result.get("support_pages"):
        canonical_answer["relevant_pages"] = list(validation_result["support_pages"])
    if validation_result.get("decision") == "rewrite" and validation_result.get("normalized_final_answer") == "N/A":
        canonical_answer["relevant_pages"] = list(validation_result.get("support_pages") or canonical_answer.get("relevant_pages") or [])
    return canonical_answer, validation_result, accepted_checks


def _build_sample_record(
    record: Dict[str, Any],
    sample_index: int,
    *,
    anchor_record: Dict[str, Any] | None = None,
    cache_record: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], List[str]]:
    normalized = normalize_training_query_record(record)
    if not _required_fields_present(record, normalized):
        raise ValueError("missing_required_fields")

    rag_context = str(record.get("rag_context") or "").strip()
    retrieval_results = record.get("retrieval_results") or []
    retrieval_pages = normalize_pages(record.get("retrieval_pages"))
    if not rag_context or not retrieval_results:
        raise ValueError("no_retrieval")

    answer = record.get("answer") if isinstance(record.get("answer"), dict) else {}
    cleaned_answer = prune_answer_to_schema(
        answer,
        schema=str(normalized["schema"]),
        provider=str(record.get("teacher_answer_provider") or "qwen"),
    )
    if normalized["schema"] == "number":
        cleaned_answer = _enforce_number_answer_consistency(record, cleaned_answer)
    final_answer = cleaned_answer.get("final_answer")
    if _is_empty_final_answer(final_answer):
        raise ValueError("empty_final_answer")
    if _requires_explicit_legal_representative_grounding(normalized["question_text"], final_answer):
        if not _has_explicit_legal_representative_grounding(record):
            raise ValueError("weak_legal_representative_grounding")

    name_validation_result = None
    boolean_validation_result = None
    validator_checks: List[str] = []
    if normalized["schema"] == "name":
        working_record = dict(record)
        working_record["answer"] = dict(cleaned_answer)
        name_validation_result = validate_name_answer(
            working_record,
            anchor_record=anchor_record,
            cache_record=cache_record,
        )
        if name_validation_result["decision"] != "accept":
            raise ValueError(name_validation_result["decision_reason"])
        validator_checks.extend(list(name_validation_result.get("accepted_checks") or []))
    elif normalized["schema"] == "boolean":
        cleaned_answer, boolean_validation_result, validator_checks = _apply_boolean_validation(
            record,
            cleaned_answer,
            anchor_record=anchor_record,
            cache_record=cache_record,
        )
        final_answer = cleaned_answer.get("final_answer")

    relevant_pages = normalize_pages(cleaned_answer.get("relevant_pages"))
    if final_answer != "N/A":
        if not relevant_pages:
            raise ValueError("missing_relevant_pages")
        if retrieval_pages and not set(relevant_pages).issubset(set(retrieval_pages)):
            raise ValueError("hallucinated_pages")

    validation_result = record.get("validation_result") if isinstance(record.get("validation_result"), dict) else {}
    validation_flags = {
        str(flag)
        for flag in (
            validation_result.get("validation_flags")
            or answer.get("validation_flags")
            or []
        )
        if flag
    }
    severe_flags = sorted(validation_flags & _SEVERE_VALIDATION_FLAGS)
    if severe_flags:
        raise ValueError(f"severe_validation_flags:{','.join(severe_flags)}")

    system_prompt, user_prompt_template, _ = build_rag_prompt_bundle(
        str(normalized["schema"]),
        provider=str(record.get("teacher_answer_provider") or "qwen"),
    )
    user_prompt = user_prompt_template.format(
        context=rag_context,
        question=normalized["question_text"],
    )

    accepted_checks = ["retrieval_present", "schema_pruned"]
    if final_answer == "N/A":
        accepted_checks.append("refusal_sample")
    else:
        accepted_checks.append("relevant_pages_subset")
    if normalized["schema"] == "number":
        accepted_checks.append("table_grounded")
        if final_answer != "N/A":
            accepted_checks.append("number_grounding_answer_match")
    if validation_flags:
        accepted_checks.append("mild_validation_flags_only")
    else:
        accepted_checks.append("no_validation_flags")
    accepted_checks.extend(check for check in validator_checks if check not in accepted_checks)

    sample_record = {
        "sample_id": f"sft-{sample_index:06d}",
        "query_id": normalized["query_id"],
        "question_text": normalized["question_text"],
        "schema": normalized["schema"],
        "task_type": normalized.get("task_type"),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "assistant_response": cleaned_answer,
        "assistant_response_json": compact_json_dumps(cleaned_answer),
        "doc_ids": normalized["doc_ids"],
        "company_name": normalized["company_name"],
        "retrieval_pages": retrieval_pages,
        "accepted_checks": accepted_checks,
        "source": normalized["source"] or "teacher_filtered",
        "should_refuse": bool(normalized["should_refuse"] or final_answer == "N/A"),
        "teacher_answer_model": record.get("teacher_answer_model"),
    }
    for optional_field in (
        "template_id",
        "template_family",
        "template_version",
        "target_key",
        "surface_variant_id",
        "split_pool",
        "answer_policy",
        "validator_target",
    ):
        if normalized.get(optional_field) not in (None, ""):
            sample_record[optional_field] = normalized.get(optional_field)
    if anchor_record is not None:
        sample_record["anchor_id"] = anchor_record.get("anchor_id")
        sample_record["anchor_source_bucket"] = anchor_record.get("source_bucket")
        sample_record["anchor_final_answer"] = anchor_record.get("final_answer")
        sample_record["anchor_pages"] = list(anchor_record.get("anchor_pages", []) or [])
    if name_validation_result is not None:
        sample_record["name_validation_result"] = name_validation_result
    if boolean_validation_result is not None:
        sample_record["boolean_validation_result"] = boolean_validation_result
    return sample_record, sorted(validation_flags)


def _resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    default_config_path = REPO_ROOT / "training/generator_sft/configs/filter.example.yaml"
    config_path = args.config_path or (default_config_path if default_config_path.exists() else None)
    config = load_yaml_mapping(config_path)

    input_path = resolve_repo_path(REPO_ROOT, _coalesce(args.input_path, config.get("input_path")))
    retrieved_cache_input_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(args.retrieved_cache_input_path, config.get("retrieved_cache_input_path")),
    )
    anchor_input_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(args.anchor_input_path, config.get("anchor_input_path")),
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
    if input_path is None or output_path is None or rejected_output_path is None or stats_output_path is None:
        raise ValueError("input/output/rejected/stats paths are required.")
    if retrieved_cache_input_path is not None and not retrieved_cache_input_path.exists():
        retrieved_cache_input_path = None
    if anchor_input_path is not None and not anchor_input_path.exists():
        anchor_input_path = None

    return {
        "config_path": config_path,
        "input_path": input_path,
        "retrieved_cache_input_path": retrieved_cache_input_path,
        "anchor_input_path": anchor_input_path,
        "output_path": output_path,
        "rejected_output_path": rejected_output_path,
        "stats_output_path": stats_output_path,
    }


def _reset_output_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _build_rejected_record(record: Dict[str, Any], reason: str) -> Dict[str, Any]:
    answer = record.get("answer") if isinstance(record.get("answer"), dict) else {}
    grounding = _extract_table_grounding_result(record)
    validation_result = record.get("validation_result") if isinstance(record.get("validation_result"), dict) else {}
    return {
        "query_id": record.get("query_id") or record.get("id"),
        "question_text": record.get("question_text") or record.get("text") or record.get("query"),
        "schema": record.get("schema") or record.get("kind"),
        "sample_id": record.get("sample_id"),
        "reason": reason,
        "final_answer": answer.get("final_answer"),
        "relevant_pages": normalize_pages(answer.get("relevant_pages")),
        "grounding_normalized_value": grounding.get("normalized_value"),
        "retrieval_pages": normalize_pages(record.get("retrieval_pages")),
        "validation_flags": list(validation_result.get("validation_flags") or answer.get("validation_flags") or []),
        "reject_log": {
            "stage": "filter_sft_samples",
            "reason_code": reason.split(":", maxsplit=1)[0],
            "reason_message": reason,
            "query_id": record.get("query_id") or record.get("id"),
            "sample_id": record.get("sample_id"),
            "schema": record.get("schema") or record.get("kind"),
        },
        "build_timestamp": utc_now_iso(),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = _resolve_settings(args)
    if not settings["input_path"].exists():
        raise FileNotFoundError(
            "Missing raw teacher-answer file: "
            f"{settings['input_path']}. "
            "Run training/generator_sft/scripts/mine_teacher_answers.py first and make sure it produced at least one successful record."
        )
    _reset_output_file(settings["output_path"])
    _reset_output_file(settings["rejected_output_path"])
    raw_records = load_records(settings["input_path"])
    cache_by_query_id = (
        _index_records_by_query_id(load_records(settings["retrieved_cache_input_path"]))
        if settings["retrieved_cache_input_path"] is not None
        else {}
    )
    anchor_by_query_id = (
        _index_records_by_query_id(load_records(settings["anchor_input_path"]))
        if settings["anchor_input_path"] is not None
        else {}
    )

    query_seen = set()
    context_seen = set()
    accepted = 0
    rejected = 0
    sample_index = 0
    schema_counter = Counter()
    rejection_counter = Counter()

    for record in raw_records:
        try:
            query_id = str(record.get("query_id") or record.get("id") or "").strip()
            sample_record, validation_flags = _build_sample_record(
                record,
                sample_index + 1,
                anchor_record=anchor_by_query_id.get(query_id),
                cache_record=cache_by_query_id.get(query_id),
            )
            query_key = (
                str(sample_record.get("schema") or ""),
                str(sample_record.get("question_text") or "").strip(),
                tuple(str(item) for item in sample_record.get("doc_ids", [])),
            )
            context_key = (
                str(sample_record.get("schema") or ""),
                stable_hash_int(sample_record.get("user_prompt") or ""),
                stable_hash_int(sample_record.get("assistant_response_json") or ""),
            )
            if query_key in query_seen:
                raise ValueError("duplicate_query")
            if context_key in context_seen:
                raise ValueError("duplicate_context")

            query_seen.add(query_key)
            context_seen.add(context_key)
            sample_index += 1
            sample_record["sample_id"] = f"sft-{sample_index:06d}"
            if validation_flags:
                sample_record["validation_flags"] = validation_flags
            append_jsonl(settings["output_path"], sample_record)
            accepted += 1
            schema_counter[str(sample_record.get("schema") or "")] += 1
        except Exception as exc:
            reason = str(exc)
            append_jsonl(settings["rejected_output_path"], _build_rejected_record(record, reason))
            rejected += 1
            rejection_counter[reason.split(":", maxsplit=1)[0]] += 1

    stats = {
        "build_timestamp": utc_now_iso(),
        "config_path": display_path(settings["config_path"], REPO_ROOT),
        "input_path": display_path(settings["input_path"], REPO_ROOT),
        "retrieved_cache_input_path": display_path(settings["retrieved_cache_input_path"], REPO_ROOT),
        "anchor_input_path": display_path(settings["anchor_input_path"], REPO_ROOT),
        "output_path": display_path(settings["output_path"], REPO_ROOT),
        "rejected_output_path": display_path(settings["rejected_output_path"], REPO_ROOT),
        "total_raw_records": len(raw_records),
        "accepted_records": accepted,
        "rejected_records": rejected,
        "schema_distribution": dict(schema_counter),
        "rejection_distribution": dict(rejection_counter),
    }
    write_json(settings["stats_output_path"], stats)


if __name__ == "__main__":
    main()
