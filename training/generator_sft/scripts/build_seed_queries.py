from __future__ import annotations

import argparse
import copy
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.text_normalization import normalize_text as normalize_query_text  # noqa: E402
from training.common import (  # noqa: E402
    display_path,
    load_records,
    load_yaml_mapping,
    normalize_training_query_record,
    read_jsonl,
    resolve_dataset_root,
    resolve_repo_path,
    stable_hash_int,
    utc_now_iso,
    write_json,
    write_jsonl,
)
from training.generator_sft.validators import (  # noqa: E402
    _BOOLEAN_TARGET_REGISTRY,
    _NAME_FIELD_REGISTRY,
)


_TEMPLATE_LIBRARY = {
    "legal_representative": {
        "schema": "name",
        "task_type": "single_doc_fact",
        "difficulty": "easy",
        "question_template": "{company_name}{report_year}年年报中的法定代表人是谁？",
    },
    "revenue": {
        "schema": "number",
        "task_type": "single_doc_fact",
        "difficulty": "medium",
        "question_template": "{company_name}{report_year}年年报中的营业收入是多少元？",
    },
    "cash_dividend": {
        "schema": "boolean",
        "task_type": "single_doc_boolean",
        "difficulty": "medium",
        "question_template": "{company_name}{report_year}年年报中是否提到现金分红？",
    },
}

_LEGACY_QUESTION_MODES = {"none", "supplement_multidoc_only", "supplement_all"}
_NUMBER_TARGET_KEYS = {
    "revenue",
    "attributable_net_profit",
    "deducted_attributable_net_profit",
    "operating_cashflow_net",
    "basic_eps",
    "asset_liability_ratio",
}
_OPTIONAL_META_FIELDS = (
    "template_id",
    "template_family",
    "template_version",
    "target_key",
    "surface_variant_id",
    "split_pool",
    "answer_policy",
    "validator_target",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build normalized seed queries for generator SFT.")
    parser.add_argument("--config-path", type=Path, default=None, help="YAML config path.")
    parser.add_argument("--dataset-root-path", type=Path, default=None, help="Dataset root path.")
    parser.add_argument("--questions-path", type=Path, default=None, help="Existing questions JSON/JSONL path.")
    parser.add_argument("--annual-report-path", type=Path, default=None, help="Annual report metadata JSONL path.")
    parser.add_argument("--output-path", type=Path, default=None, help="Output seed_queries.jsonl path.")
    parser.add_argument("--stats-output-path", type=Path, default=None, help="Stats JSON path.")
    parser.add_argument("--max-existing-questions", type=int, default=None, help="Limit imported base questions.")
    parser.add_argument("--max-template-reports", type=int, default=None, help="Limit annual reports used for template bootstrap.")
    parser.add_argument("--disable-template-bootstrap", action="store_true", help="Only normalize the existing question set.")
    parser.add_argument("--template-catalog-path", type=Path, default=None, help="V2 query template catalog YAML path.")
    parser.add_argument("--template-version", type=str, default=None, help="Template version. Use `v2` for the new catalog flow.")
    parser.add_argument(
        "--legacy-questions-mode",
        type=str,
        default=None,
        help="Legacy question import mode: none | supplement_multidoc_only | supplement_all.",
    )
    return parser


def _coalesce(cli_value: Any, config_value: Any, default: Any = None) -> Any:
    return cli_value if cli_value is not None else (config_value if config_value is not None else default)


def _short_hash(value: Any, *, salt: str = "") -> str:
    return f"{stable_hash_int(value, salt=salt):x}"[-8:]


def _question_placeholders(text: str) -> List[str]:
    return re.findall(r"{([a-zA-Z_][a-zA-Z0-9_]*)}", str(text or ""))


def _supports_surface_form(surface_form: str, context: Dict[str, Any]) -> bool:
    for placeholder in _question_placeholders(surface_form):
        value = context.get(placeholder)
        if value in (None, ""):
            return False
    return True


def _render_surface_form(
    *,
    surface_forms: Sequence[str],
    context: Dict[str, Any],
    selection_key: Any,
    salt: str,
) -> Tuple[str, str]:
    valid_forms = [
        (index, form)
        for index, form in enumerate(surface_forms)
        if _supports_surface_form(form, context)
    ]
    if not valid_forms:
        raise ValueError(f"No valid surface form could be rendered with context keys: {sorted(context.keys())}")
    selected_index = stable_hash_int(selection_key, salt=salt) % len(valid_forms)
    surface_index, surface_form = valid_forms[selected_index]
    return surface_form.format(**context), f"{surface_index:02d}"


def _select_section_name(template: Dict[str, Any], selection_key: Any) -> Optional[str]:
    section_hints = [str(item) for item in template.get("section_hints", []) if str(item).strip()]
    if not section_hints:
        return None
    index = stable_hash_int(selection_key, salt=f"{template['template_id']}:section") % len(section_hints)
    return section_hints[index]


def _infer_task_type(record: Dict[str, Any], normalized: Dict[str, Any]) -> str:
    if record.get("task_type"):
        return str(record["task_type"])
    if record.get("capability"):
        return str(record["capability"])
    if normalized.get("task_type"):
        return str(normalized["task_type"])
    if normalized["schema"] == "comparative":
        return "cross_doc_compare"
    if record.get("section_name"):
        return "section_filter"
    if normalized["mentioned_companies"] and len(normalized["mentioned_companies"]) > 1:
        return "cross_doc_compare"
    if normalized["schema"] == "boolean":
        return "single_doc_boolean"
    if normalized["schema"] == "names":
        return "metadata_tag_retrieval"
    return "single_doc_fact"


def _infer_difficulty(record: Dict[str, Any], normalized: Dict[str, Any]) -> str:
    if record.get("difficulty"):
        return str(record["difficulty"])
    task_type = _infer_task_type(record, normalized)
    if task_type in {"cross_doc_compare", "metadata_tag_retrieval"}:
        return "hard"
    if task_type in {"section_filter", "single_doc_boolean"}:
        return "medium"
    return "easy"


def _validator_target_for_schema(schema: str) -> str:
    mapping = {
        "name": "name_strict_v1",
        "number": "number_grounding_match_v1",
        "boolean": "boolean_trinary_v1",
        "names": "names_bucket_v1",
        "text": "generic_query_v1",
        "long_text": "generic_query_v1",
        "comparative": "comparative_pair_v1",
    }
    return mapping.get(schema, "generic_query_v1")


def _answer_policy_for_schema(schema: str) -> str:
    mapping = {
        "name": "direct_extract_only",
        "number": "direct_extract_only",
        "boolean": "direct_extract_only",
        "names": "candidate_bucket_only",
        "text": "evidence_grounded_synthesis",
        "long_text": "evidence_grounded_synthesis",
        "comparative": "compare_teacher_answers",
    }
    return mapping.get(schema, "direct_extract_only")


def _infer_split_pool(schema: str, task_type: str, doc_ids: Sequence[str]) -> str:
    if schema in {"names", "comparative"}:
        return "aux_multidoc"
    if task_type == "cross_doc_compare" or len(doc_ids) > 1:
        return "aux_multidoc"
    return "core_single_doc"


def _infer_legacy_target_key(question_text: Any, schema: str, task_type: str) -> str:
    text = str(question_text or "")
    if schema == "comparative":
        if "归母净利润" in text or "归属于上市公司股东的净利润" in text:
            return "attributable_net_profit"
        if "营业收入" in text:
            return "revenue"
        return "legacy_comparative"
    if schema == "names":
        for strategy_tag in ("国产替代", "人工智能", "出海", "绿色转型", "数字化转型"):
            if strategy_tag in text:
                return strategy_tag
        return "legacy_names"
    if task_type == "section_filter":
        return "legacy_section_filter"
    return f"legacy_{schema or 'unknown'}"


def _normalize_base_question(
    record: Dict[str, Any],
    index: int,
    *,
    template_version: str = "v2",
) -> Dict[str, Any]:
    normalized = normalize_training_query_record(record)
    question_text = normalized["question_text"] or record.get("text")
    if not question_text:
        raise ValueError(f"Question record at index {index} is missing text.")

    schema = str(normalized["schema"] or "").strip()
    if not schema:
        raise ValueError(f"Question record at index {index} is missing schema/kind.")

    expected_filters = copy.deepcopy(normalized["expected_filters"])
    if not isinstance(expected_filters, dict):
        expected_filters = {}
    if record.get("report_year") is not None and "report_year" not in expected_filters:
        expected_filters["report_year"] = record["report_year"]
    if record.get("section_name") and "section_name" not in expected_filters:
        expected_filters["section_name"] = record["section_name"]
    if normalized["company_name"] and "company_name" not in expected_filters:
        expected_filters["company_name"] = normalized["company_name"]

    source = "questions_json"
    if record.get("annotation_status"):
        source = f"{source}:{record['annotation_status']}"

    raw_query_id = normalized["query_id"] or f"seed-base-{index:06d}"
    query_id = f"seed-{raw_query_id}" if not str(raw_query_id).startswith("seed-") else str(raw_query_id)
    mentioned_companies = record.get("mentioned_companies")
    if not isinstance(mentioned_companies, list):
        mentioned_companies = normalized["mentioned_companies"]

    task_type = _infer_task_type(record, normalized)
    split_pool = _infer_split_pool(schema, task_type, normalized["doc_ids"])
    target_key = _infer_legacy_target_key(question_text, schema, task_type)
    normalized_record = {
        "query_id": query_id,
        "question_text": question_text,
        "schema": schema,
        "task_type": task_type,
        "company_name": normalized["company_name"],
        "mentioned_companies": mentioned_companies,
        "doc_ids": normalized["doc_ids"],
        "expected_filters": expected_filters,
        "source": source,
        "difficulty": _infer_difficulty(record, normalized),
        "should_refuse": bool(normalized["should_refuse"]),
        "template_id": f"legacy_{target_key}",
        "template_family": "legacy_curated",
        "template_version": template_version,
        "target_key": target_key,
        "surface_variant_id": "legacy",
        "split_pool": split_pool,
        "answer_policy": _answer_policy_for_schema(schema),
        "validator_target": _validator_target_for_schema(schema),
    }
    return normalized_record


def _dedupe_records(records: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    skipped = 0
    for record in records:
        dedupe_key = (
            str(record.get("schema") or ""),
            str(record.get("target_key") or ""),
            normalize_query_text(str(record.get("question_text") or "")),
            tuple(str(item) for item in record.get("doc_ids", [])),
        )
        if dedupe_key in seen:
            skipped += 1
            continue
        seen.add(dedupe_key)
        deduped.append(record)
    return deduped, skipped


def _resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    default_config_path = REPO_ROOT / "training/generator_sft/configs/build_seed.example.yaml"
    config_path = args.config_path or (default_config_path if default_config_path.exists() else None)
    config = load_yaml_mapping(config_path)

    dataset_root_path = resolve_dataset_root(REPO_ROOT, _coalesce(args.dataset_root_path, config.get("dataset_root_path")))
    questions_path = resolve_repo_path(REPO_ROOT, _coalesce(args.questions_path, config.get("questions_path")))
    annual_report_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(args.annual_report_path, config.get("annual_report_path")),
    )
    output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.output_path, config.get("output_path")))
    stats_output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.stats_output_path, config.get("stats_output_path")))
    template_catalog_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(args.template_catalog_path, config.get("template_catalog_path")),
    )
    chunk_metadata_path = resolve_repo_path(REPO_ROOT, config.get("chunk_metadata_path"))
    if questions_path is None:
        questions_path = dataset_root_path / "questions.json"
    if annual_report_path is None:
        annual_report_path = dataset_root_path / "metadata_store/annual_report.jsonl"
    if chunk_metadata_path is None:
        chunk_metadata_path = dataset_root_path / "metadata_store/chunk_metadata.jsonl"
    if output_path is None or stats_output_path is None:
        raise ValueError("`output_path` and `stats_output_path` are required.")

    legacy_questions_mode = str(
        _coalesce(args.legacy_questions_mode, config.get("legacy_questions_mode"), "supplement_multidoc_only")
    )
    if legacy_questions_mode not in _LEGACY_QUESTION_MODES:
        raise ValueError(f"Unsupported legacy_questions_mode: {legacy_questions_mode}")

    template_version = str(_coalesce(args.template_version, config.get("template_version"), "v2"))
    if template_catalog_path is None and template_version == "v2":
        template_catalog_path = REPO_ROOT / "training/generator_sft/configs/query_templates.v2.yaml"

    return {
        "config_path": config_path,
        "dataset_root_path": dataset_root_path,
        "questions_path": questions_path,
        "annual_report_path": annual_report_path,
        "chunk_metadata_path": chunk_metadata_path,
        "output_path": output_path,
        "stats_output_path": stats_output_path,
        "template_catalog_path": template_catalog_path,
        "template_version": template_version,
        "legacy_questions_mode": legacy_questions_mode,
        "max_existing_questions": max(0, int(_coalesce(args.max_existing_questions, config.get("max_existing_questions"), 0) or 0)),
        "max_template_reports": max(0, int(_coalesce(args.max_template_reports, config.get("max_template_reports"), 0) or 0)),
        "include_template_bootstrap": not bool(
            args.disable_template_bootstrap or not _coalesce(None, config.get("include_template_bootstrap"), True)
        ),
        "template_ids": [
            str(template_id)
            for template_id in (_coalesce(None, config.get("template_ids"), []) or [])
            if str(template_id).strip()
        ],
    }


def _load_template_catalog(path: Path) -> Dict[str, Any]:
    payload = load_yaml_mapping(path)
    templates = payload.get("templates")
    if not isinstance(templates, list) or not templates:
        raise ValueError(f"Template catalog must contain a non-empty `templates` list: {path}")
    return payload


def _validate_v2_catalog(templates: Sequence[Dict[str, Any]]) -> None:
    required_fields = {
        "template_id",
        "template_family",
        "schema",
        "task_type",
        "generation_mode",
        "target_key",
        "surface_forms",
        "section_hints",
        "split_pool",
        "answer_policy",
        "validator_target",
        "max_per_doc",
    }
    seen_ids = set()
    name_target_keys = {str(item["field_key"]) for item in _NAME_FIELD_REGISTRY}
    boolean_target_keys = {str(item["target_key"]) for item in _BOOLEAN_TARGET_REGISTRY}

    for template in templates:
        missing = sorted(field for field in required_fields if field not in template)
        if missing:
            raise ValueError(f"Template {template.get('template_id')!r} is missing fields: {missing}")
        template_id = str(template["template_id"])
        if template_id in seen_ids:
            raise ValueError(f"Duplicate template_id in catalog: {template_id}")
        seen_ids.add(template_id)

        if template["generation_mode"] not in {"per_doc", "per_bucket", "per_pair"}:
            raise ValueError(f"Unsupported generation_mode in {template_id}: {template['generation_mode']}")
        if not isinstance(template["surface_forms"], list) or not template["surface_forms"]:
            raise ValueError(f"Template {template_id} must define non-empty surface_forms.")
        if template["schema"] == "name" and template["target_key"] not in name_target_keys:
            raise ValueError(f"Name template target_key not found in validators registry: {template['target_key']}")
        if template["schema"] == "boolean" and template["target_key"] not in boolean_target_keys:
            raise ValueError(f"Boolean template target_key not found in validators registry: {template['target_key']}")
        if template["schema"] == "number" and template["target_key"] not in _NUMBER_TARGET_KEYS:
            raise ValueError(f"Number template target_key is outside the supported whitelist: {template['target_key']}")


def _select_templates(catalog: Dict[str, Any], requested_template_ids: Sequence[str]) -> List[Dict[str, Any]]:
    templates = [dict(item) for item in catalog.get("templates", []) if isinstance(item, dict)]
    if not requested_template_ids:
        return templates
    requested = {str(item) for item in requested_template_ids}
    selected = [template for template in templates if str(template.get("template_id")) in requested]
    if not selected:
        raise ValueError(f"No templates matched requested template_ids: {sorted(requested)}")
    return selected


def _should_keep_legacy_record(record: Dict[str, Any], mode: str) -> bool:
    if mode == "none":
        return False
    if mode == "supplement_all":
        return True

    task_type = str(record.get("task_type") or "")
    schema = str(record.get("schema") or "")
    if task_type in {"section_filter", "metadata_tag_retrieval", "cross_doc_compare"}:
        return True
    if schema in {"names", "comparative"}:
        return True
    return False


def _rotated_take(items: Sequence[Dict[str, Any]], count: int, *, key: Any, salt: str) -> List[Dict[str, Any]]:
    if not items or count <= 0:
        return []
    start_index = stable_hash_int(key, salt=salt) % len(items)
    ordered = [items[(start_index + offset) % len(items)] for offset in range(len(items))]
    return ordered[: min(count, len(ordered))]


def _build_per_doc_record(
    *,
    report_record: Dict[str, Any],
    template: Dict[str, Any],
    template_version: str,
) -> Dict[str, Any]:
    company_name = str(report_record.get("company_name") or "").strip()
    report_year = int(report_record.get("report_year") or 0)
    doc_id = str(report_record.get("doc_id") or report_record.get("report_id") or "").strip()
    if not company_name or not report_year or not doc_id:
        raise ValueError(f"Annual report record is missing company_name/report_year/doc_id: {report_record!r}")

    section_name = _select_section_name(template, doc_id)
    context = {
        "company_name": company_name,
        "report_year": report_year,
        "section_name": section_name,
        "field_label": template.get("field_label"),
        "metric_label": template.get("metric_label"),
        "question_suffix": template.get("question_suffix"),
    }
    question_text, surface_variant_id = _render_surface_form(
        surface_forms=template["surface_forms"],
        context=context,
        selection_key=doc_id,
        salt=f"{template['template_id']}:surface",
    )

    expected_filters = {
        "company_name": company_name,
        "report_year": report_year,
    }
    uses_section_name = "{section_name}" in str(template["surface_forms"][int(surface_variant_id)])
    if uses_section_name and section_name:
        expected_filters["section_name"] = section_name

    task_type = str(template["task_type"])
    if uses_section_name:
        task_type = "section_filter"

    schema = str(template["schema"])
    return {
        "query_id": f"seed-v2-{schema}-{template['target_key']}-{surface_variant_id}-{doc_id}",
        "question_text": question_text,
        "schema": schema,
        "task_type": task_type,
        "company_name": company_name,
        "mentioned_companies": [],
        "doc_ids": [doc_id],
        "expected_filters": expected_filters,
        "source": "template_from_annual_report_v2",
        "difficulty": str(template.get("difficulty") or "medium"),
        "should_refuse": False,
        "template_id": str(template["template_id"]),
        "template_family": str(template["template_family"]),
        "template_version": template_version,
        "target_key": str(template["target_key"]),
        "surface_variant_id": surface_variant_id,
        "split_pool": str(template["split_pool"]),
        "answer_policy": str(template["answer_policy"]),
        "validator_target": str(template["validator_target"]),
    }


def _build_per_doc_records(
    annual_reports: Sequence[Dict[str, Any]],
    templates: Sequence[Dict[str, Any]],
    *,
    template_version: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    templates_by_schema: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for template in templates:
        if template.get("generation_mode") != "per_doc":
            continue
        templates_by_schema[str(template["schema"])].append(template)

    for schema in templates_by_schema:
        templates_by_schema[schema] = sorted(templates_by_schema[schema], key=lambda item: str(item["template_id"]))

    for report_record in annual_reports:
        doc_id = str(report_record.get("doc_id") or report_record.get("report_id") or "").strip()
        if not doc_id:
            continue
        for schema, schema_templates in sorted(templates_by_schema.items()):
            max_per_doc = max(int(template.get("max_per_doc") or 1) for template in schema_templates)
            selected_templates = _rotated_take(
                schema_templates,
                max_per_doc,
                key=doc_id,
                salt=f"{schema}:{template_version}",
            )
            for template in selected_templates:
                records.append(
                    _build_per_doc_record(
                        report_record=report_record,
                        template=template,
                        template_version=template_version,
                    )
                )
    return records


def _build_bucket_entries(
    chunk_records: Sequence[Dict[str, Any]],
    *,
    strategy_tag: str,
    min_docs: int,
    max_docs: int,
) -> List[Dict[str, Any]]:
    base_buckets: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
    for record in chunk_records:
        doc_id = str(record.get("doc_id") or "").strip()
        report_year = int(record.get("report_year") or 0)
        industry = str(record.get("industry_l1") or "").strip()
        board = str(record.get("board") or "").strip()
        strategy_tags = {str(item).strip() for item in (record.get("strategy_tags") or []) if str(item).strip()}
        if not doc_id or not report_year or not industry or strategy_tag not in strategy_tags:
            continue

        bucket_key = (report_year, industry, strategy_tag)
        bucket = base_buckets.setdefault(
            bucket_key,
            {
                "report_year": report_year,
                "industry_l1": industry,
                "strategy_tag": strategy_tag,
                "doc_ids": set(),
                "boards": defaultdict(set),
            },
        )
        bucket["doc_ids"].add(doc_id)
        if board:
            bucket["boards"][board].add(doc_id)

    materialized: List[Dict[str, Any]] = []
    for bucket_key in sorted(base_buckets):
        bucket = base_buckets[bucket_key]
        doc_ids = sorted(bucket["doc_ids"])
        if len(doc_ids) < min_docs:
            continue
        if len(doc_ids) <= max_docs:
            materialized.append(
                {
                    "report_year": bucket["report_year"],
                    "industry_l1": bucket["industry_l1"],
                    "board": None,
                    "strategy_tag": bucket["strategy_tag"],
                    "doc_ids": doc_ids,
                }
            )
            continue

        for board_name in sorted(bucket["boards"]):
            board_doc_ids = sorted(bucket["boards"][board_name])
            if len(board_doc_ids) < min_docs or len(board_doc_ids) > max_docs:
                continue
            materialized.append(
                {
                    "report_year": bucket["report_year"],
                    "industry_l1": bucket["industry_l1"],
                    "board": board_name,
                    "strategy_tag": bucket["strategy_tag"],
                    "doc_ids": board_doc_ids,
                }
            )
    return materialized


def _build_names_bucket_records(
    chunk_records: Sequence[Dict[str, Any]],
    templates: Sequence[Dict[str, Any]],
    *,
    template_version: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for template in sorted(templates, key=lambda item: str(item["template_id"])):
        if template.get("generation_mode") != "per_bucket":
            continue

        strategy_tag = str(template.get("strategy_tag") or "").strip()
        if not strategy_tag:
            continue
        bucket_entries = _build_bucket_entries(
            chunk_records,
            strategy_tag=strategy_tag,
            min_docs=int(template.get("min_docs_per_bucket") or 3),
            max_docs=int(template.get("max_docs_per_bucket") or 20),
        )
        for bucket in bucket_entries:
            doc_ids = list(bucket["doc_ids"])
            context = {
                "year": bucket["report_year"],
                "industry": bucket["industry_l1"],
                "board": bucket["board"],
                "tag": strategy_tag,
            }
            question_text, surface_variant_id = _render_surface_form(
                surface_forms=template["surface_forms"],
                context=context,
                selection_key={"template_id": template["template_id"], "doc_ids": doc_ids},
                salt=f"{template['template_id']}:surface",
            )
            bucket_hash = _short_hash({"template_id": template["template_id"], "doc_ids": doc_ids}, salt=template_version)
            expected_filters = {
                "report_year": bucket["report_year"],
                "industry_l1": bucket["industry_l1"],
                "strategy_tags": [strategy_tag],
                "candidate_doc_ids": doc_ids,
            }
            if bucket["board"]:
                expected_filters["board"] = bucket["board"]
            records.append(
                {
                    "query_id": f"seed-v2-names-{template['target_key']}-{bucket_hash}",
                    "question_text": question_text,
                    "schema": "names",
                    "task_type": "metadata_tag_retrieval",
                    "company_name": None,
                    "mentioned_companies": [],
                    "doc_ids": doc_ids,
                    "expected_filters": expected_filters,
                    "source": "template_from_chunk_metadata_v2",
                    "difficulty": str(template.get("difficulty") or "hard"),
                    "should_refuse": False,
                    "template_id": str(template["template_id"]),
                    "template_family": str(template["template_family"]),
                    "template_version": template_version,
                    "target_key": str(template["target_key"]),
                    "surface_variant_id": surface_variant_id,
                    "split_pool": str(template["split_pool"]),
                    "answer_policy": str(template["answer_policy"]),
                    "validator_target": str(template["validator_target"]),
                }
            )
    return records


def _sorted_report_key(report_record: Dict[str, Any]) -> Tuple[str, str]:
    return (
        str(report_record.get("stock_code") or "").strip(),
        str(report_record.get("doc_id") or report_record.get("report_id") or "").strip(),
    )


def _build_report_pairs(annual_reports: Sequence[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    grouped_by_board: Dict[Tuple[int, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for report_record in annual_reports:
        report_year = int(report_record.get("report_year") or 0)
        industry = str(report_record.get("industry_l1") or "").strip()
        board = str(report_record.get("board") or "").strip()
        doc_id = str(report_record.get("doc_id") or report_record.get("report_id") or "").strip()
        if not report_year or not industry or not doc_id:
            continue
        grouped_by_board[(report_year, industry, board)].append(report_record)

    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    leftovers: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)

    for group_key in sorted(grouped_by_board):
        group_reports = sorted(grouped_by_board[group_key], key=_sorted_report_key)
        pairable_count = len(group_reports) - (len(group_reports) % 2)
        for index in range(0, pairable_count, 2):
            pairs.append((group_reports[index], group_reports[index + 1]))
        if len(group_reports) % 2 == 1:
            leftovers[(group_key[0], group_key[1])].append(group_reports[-1])

    for leftover_key in sorted(leftovers):
        group_reports = sorted(leftovers[leftover_key], key=_sorted_report_key)
        pairable_count = len(group_reports) - (len(group_reports) % 2)
        for index in range(0, pairable_count, 2):
            pairs.append((group_reports[index], group_reports[index + 1]))

    return pairs


def _build_comparative_records(
    annual_reports: Sequence[Dict[str, Any]],
    templates: Sequence[Dict[str, Any]],
    *,
    template_version: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    pairs = _build_report_pairs(annual_reports)
    for left_report, right_report in pairs:
        left_doc_id = str(left_report.get("doc_id") or left_report.get("report_id") or "").strip()
        right_doc_id = str(right_report.get("doc_id") or right_report.get("report_id") or "").strip()
        left_company = str(left_report.get("company_name") or "").strip()
        right_company = str(right_report.get("company_name") or "").strip()
        report_year = int(left_report.get("report_year") or right_report.get("report_year") or 0)
        industry = str(left_report.get("industry_l1") or right_report.get("industry_l1") or "").strip()
        board = str(left_report.get("board") or right_report.get("board") or "").strip()
        if not left_doc_id or not right_doc_id or not left_company or not right_company or not report_year:
            continue

        for template in sorted(templates, key=lambda item: str(item["template_id"])):
            if template.get("generation_mode") != "per_pair":
                continue
            context = {
                "year": report_year,
                "company_a": left_company,
                "company_b": right_company,
                "metric_label": template.get("metric_label"),
            }
            question_text, surface_variant_id = _render_surface_form(
                surface_forms=template["surface_forms"],
                context=context,
                selection_key={"template_id": template["template_id"], "doc_ids": [left_doc_id, right_doc_id]},
                salt=f"{template['template_id']}:surface",
            )
            expected_filters = {
                "report_year": report_year,
                "industry_l1": industry,
                "candidate_doc_ids": [left_doc_id, right_doc_id],
            }
            if board:
                expected_filters["board"] = board

            schema = str(template["schema"])
            records.append(
                {
                    "query_id": f"seed-v2-{schema}-{template['target_key']}-{left_doc_id}-{right_doc_id}",
                    "question_text": question_text,
                    "schema": schema,
                    "task_type": "cross_doc_compare",
                    "company_name": None,
                    "mentioned_companies": [left_company, right_company],
                    "doc_ids": [left_doc_id, right_doc_id],
                    "expected_filters": expected_filters,
                    "source": "template_from_annual_report_pair_v2",
                    "difficulty": str(template.get("difficulty") or "hard"),
                    "should_refuse": False,
                    "template_id": str(template["template_id"]),
                    "template_family": str(template["template_family"]),
                    "template_version": template_version,
                    "target_key": str(template["target_key"]),
                    "surface_variant_id": surface_variant_id,
                    "split_pool": str(template["split_pool"]),
                    "answer_policy": str(template["answer_policy"]),
                    "validator_target": str(template["validator_target"]),
                }
            )
    return records


def _legacy_template_seed_record(template_id: str, report_record: Dict[str, Any]) -> Dict[str, Any]:
    template = _TEMPLATE_LIBRARY[template_id]
    company_name = str(report_record.get("company_name") or "").strip()
    report_year = int(report_record.get("report_year") or 0)
    doc_id = str(report_record.get("doc_id") or report_record.get("report_id") or "").strip()
    if not company_name or not report_year or not doc_id:
        raise ValueError(f"Annual report record is missing company_name/report_year/doc_id: {report_record!r}")

    schema = str(template["schema"])
    return {
        "query_id": f"seed-auto-{template_id}-{doc_id}",
        "question_text": template["question_template"].format(company_name=company_name, report_year=report_year),
        "schema": schema,
        "task_type": template["task_type"],
        "company_name": company_name,
        "mentioned_companies": [],
        "doc_ids": [doc_id],
        "expected_filters": {
            "company_name": company_name,
            "report_year": report_year,
        },
        "source": "template_from_annual_report",
        "difficulty": template["difficulty"],
        "should_refuse": False,
        "template_id": f"legacy_catalog_{template_id}",
        "template_family": "legacy_bootstrap",
        "template_version": "legacy",
        "target_key": template_id,
        "surface_variant_id": "00",
        "split_pool": _infer_split_pool(schema, template["task_type"], [doc_id]),
        "answer_policy": _answer_policy_for_schema(schema),
        "validator_target": _validator_target_for_schema(schema),
    }


def _build_warnings(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    warnings: List[Dict[str, Any]] = []
    by_schema_text: Dict[str, Counter] = defaultdict(Counter)
    by_schema_target: Dict[str, Counter] = defaultdict(Counter)
    totals_by_schema: Counter = Counter()

    for record in records:
        schema = str(record.get("schema") or "")
        question_text = str(record.get("question_text") or "")
        target_key = str(record.get("target_key") or "")
        totals_by_schema[schema] += 1
        by_schema_text[schema][question_text] += 1
        by_schema_target[schema][target_key] += 1

    for schema, text_counter in sorted(by_schema_text.items()):
        total = totals_by_schema[schema]
        if total <= 0:
            continue
        question_text, count = text_counter.most_common(1)[0]
        share = count / float(total)
        if share > 0.12:
            warnings.append(
                {
                    "warning_type": "question_text_dominance",
                    "schema": schema,
                    "question_text": question_text,
                    "count": count,
                    "share": round(share, 4),
                }
            )

    dominance_thresholds = {
        "name": 0.40,
        "boolean": 0.45,
        "number": 0.35,
    }
    for schema, threshold in dominance_thresholds.items():
        total = totals_by_schema[schema]
        if total <= 0 or not by_schema_target[schema]:
            continue
        target_key, count = by_schema_target[schema].most_common(1)[0]
        share = count / float(total)
        if share > threshold:
            warnings.append(
                {
                    "warning_type": "target_key_imbalance",
                    "schema": schema,
                    "target_key": target_key,
                    "count": count,
                    "share": round(share, 4),
                    "threshold": threshold,
                }
            )
    return warnings


def _build_stats(
    *,
    settings: Dict[str, Any],
    question_records: Sequence[Dict[str, Any]],
    legacy_records: Sequence[Dict[str, Any]],
    template_records: Sequence[Dict[str, Any]],
    deduped_records: Sequence[Dict[str, Any]],
    deduped_count: int,
    template_report_count: int,
    legacy_question_dropped_count: int,
) -> Dict[str, Any]:
    schema_counter = Counter(str(record.get("schema") or "") for record in deduped_records)
    source_counter = Counter(str(record.get("source") or "") for record in deduped_records)
    task_counter = Counter(str(record.get("task_type") or "") for record in deduped_records)
    template_family_counter = Counter(str(record.get("template_family") or "") for record in deduped_records)
    target_key_counter = Counter(str(record.get("target_key") or "") for record in deduped_records)
    split_pool_counter = Counter(str(record.get("split_pool") or "") for record in deduped_records)
    surface_variant_counter = Counter(str(record.get("surface_variant_id") or "") for record in deduped_records)
    warnings = _build_warnings(deduped_records)

    return {
        "build_timestamp": utc_now_iso(),
        "config_path": display_path(settings["config_path"], REPO_ROOT),
        "dataset_root_path": display_path(settings["dataset_root_path"], REPO_ROOT),
        "questions_path": display_path(settings["questions_path"], REPO_ROOT),
        "annual_report_path": display_path(settings["annual_report_path"], REPO_ROOT),
        "chunk_metadata_path": display_path(settings["chunk_metadata_path"], REPO_ROOT),
        "template_catalog_path": display_path(settings["template_catalog_path"], REPO_ROOT),
        "output_path": display_path(settings["output_path"], REPO_ROOT),
        "template_version": settings["template_version"],
        "legacy_questions_mode": settings["legacy_questions_mode"],
        "max_existing_questions": settings["max_existing_questions"],
        "max_template_reports": settings["max_template_reports"],
        "template_ids": settings["template_ids"],
        "base_question_count": len(question_records),
        "legacy_question_kept_count": len(legacy_records),
        "legacy_question_dropped_count": legacy_question_dropped_count,
        "template_report_count": template_report_count,
        "template_record_count": len(template_records),
        "deduped_records": deduped_count,
        "total_seed_records": len(deduped_records),
        "schema_counts": dict(schema_counter),
        "schema_distribution": dict(schema_counter),
        "source_distribution": dict(source_counter),
        "task_distribution": dict(task_counter),
        "template_family_counts": dict(template_family_counter),
        "target_key_counts": dict(target_key_counter),
        "split_pool_counts": dict(split_pool_counter),
        "surface_variant_counts": dict(surface_variant_counter),
        "warnings": warnings,
    }


def _build_v2_seed_records(settings: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    catalog = _load_template_catalog(settings["template_catalog_path"])
    templates = _select_templates(catalog, settings["template_ids"])
    _validate_v2_catalog(templates)

    question_records = load_records(settings["questions_path"]) if settings["questions_path"].exists() else []
    if settings["max_existing_questions"] > 0:
        question_records = question_records[: settings["max_existing_questions"]]

    legacy_records: List[Dict[str, Any]] = []
    legacy_question_dropped_count = 0
    for index, record in enumerate(question_records, start=1):
        normalized_record = _normalize_base_question(
            record,
            index,
            template_version=settings["template_version"],
        )
        if _should_keep_legacy_record(normalized_record, settings["legacy_questions_mode"]):
            legacy_records.append(normalized_record)
        else:
            legacy_question_dropped_count += 1

    template_records: List[Dict[str, Any]] = []
    template_report_count = 0
    if settings["include_template_bootstrap"]:
        if not settings["annual_report_path"].exists():
            raise FileNotFoundError(f"Missing annual_report metadata: {settings['annual_report_path']}")
        annual_reports = list(read_jsonl(settings["annual_report_path"]))
        annual_reports = sorted(
            annual_reports,
            key=lambda item: str(item.get("doc_id") or item.get("report_id") or ""),
        )
        if settings["max_template_reports"] > 0:
            annual_reports = annual_reports[: settings["max_template_reports"]]
        template_report_count = len(annual_reports)

        allowed_doc_ids = {
            str(item.get("doc_id") or item.get("report_id") or "").strip()
            for item in annual_reports
            if str(item.get("doc_id") or item.get("report_id") or "").strip()
        }
        chunk_records = []
        if any(str(template.get("generation_mode")) == "per_bucket" for template in templates):
            if not settings["chunk_metadata_path"].exists():
                raise FileNotFoundError(f"Missing chunk metadata for per_bucket templates: {settings['chunk_metadata_path']}")
            chunk_records = [
                record
                for record in read_jsonl(settings["chunk_metadata_path"])
                if str(record.get("doc_id") or "").strip() in allowed_doc_ids
            ]

        template_records.extend(
            _build_per_doc_records(
                annual_reports,
                [template for template in templates if str(template.get("generation_mode")) == "per_doc"],
                template_version=settings["template_version"],
            )
        )
        template_records.extend(
            _build_names_bucket_records(
                chunk_records,
                [template for template in templates if str(template.get("generation_mode")) == "per_bucket"],
                template_version=settings["template_version"],
            )
        )
        template_records.extend(
            _build_comparative_records(
                annual_reports,
                [template for template in templates if str(template.get("generation_mode")) == "per_pair"],
                template_version=settings["template_version"],
            )
        )

    all_records, deduped_count = _dedupe_records([*legacy_records, *template_records])
    stats = _build_stats(
        settings=settings,
        question_records=question_records,
        legacy_records=legacy_records,
        template_records=template_records,
        deduped_records=all_records,
        deduped_count=deduped_count,
        template_report_count=template_report_count,
        legacy_question_dropped_count=legacy_question_dropped_count,
    )
    return all_records, stats


def _build_legacy_seed_records(settings: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    question_records = load_records(settings["questions_path"]) if settings["questions_path"].exists() else []
    if settings["max_existing_questions"] > 0:
        question_records = question_records[: settings["max_existing_questions"]]

    seed_records = [
        _normalize_base_question(record, index, template_version="legacy")
        for index, record in enumerate(question_records, start=1)
    ]

    template_records: List[Dict[str, Any]] = []
    template_report_count = 0
    if settings["include_template_bootstrap"]:
        annual_reports = list(read_jsonl(settings["annual_report_path"]))
        if settings["max_template_reports"] > 0:
            annual_reports = annual_reports[: settings["max_template_reports"]]
        template_ids = settings["template_ids"] or list(_TEMPLATE_LIBRARY.keys())
        for report_record in annual_reports:
            for template_id in template_ids:
                if template_id in _TEMPLATE_LIBRARY:
                    template_records.append(_legacy_template_seed_record(template_id, report_record))
        template_report_count = len(annual_reports)

    all_records, deduped_count = _dedupe_records([*seed_records, *template_records])
    stats = {
        "build_timestamp": utc_now_iso(),
        "config_path": display_path(settings["config_path"], REPO_ROOT),
        "dataset_root_path": display_path(settings["dataset_root_path"], REPO_ROOT),
        "questions_path": display_path(settings["questions_path"], REPO_ROOT),
        "annual_report_path": display_path(settings["annual_report_path"], REPO_ROOT),
        "output_path": display_path(settings["output_path"], REPO_ROOT),
        "template_version": "legacy",
        "max_existing_questions": settings["max_existing_questions"],
        "max_template_reports": settings["max_template_reports"],
        "template_ids": settings["template_ids"] or list(_TEMPLATE_LIBRARY.keys()),
        "base_question_count": len(question_records),
        "legacy_question_kept_count": len(seed_records),
        "legacy_question_dropped_count": 0,
        "template_report_count": template_report_count,
        "template_record_count": len(template_records),
        "deduped_records": deduped_count,
        "total_seed_records": len(all_records),
        "schema_counts": dict(Counter(str(record.get("schema") or "") for record in all_records)),
        "schema_distribution": dict(Counter(str(record.get("schema") or "") for record in all_records)),
        "source_distribution": dict(Counter(str(record.get("source") or "") for record in all_records)),
        "task_distribution": dict(Counter(str(record.get("task_type") or "") for record in all_records)),
        "template_family_counts": dict(Counter(str(record.get("template_family") or "") for record in all_records)),
        "target_key_counts": dict(Counter(str(record.get("target_key") or "") for record in all_records)),
        "split_pool_counts": dict(Counter(str(record.get("split_pool") or "") for record in all_records)),
        "surface_variant_counts": dict(Counter(str(record.get("surface_variant_id") or "") for record in all_records)),
        "warnings": [],
    }
    return all_records, stats


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = _resolve_settings(args)

    if settings["template_version"] == "v2":
        records, stats = _build_v2_seed_records(settings)
    else:
        records, stats = _build_legacy_seed_records(settings)

    write_jsonl(settings["output_path"], records)
    write_json(settings["stats_output_path"], stats)


if __name__ == "__main__":
    main()
