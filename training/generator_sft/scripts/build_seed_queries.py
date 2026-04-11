from __future__ import annotations

import argparse
import copy
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import (  # noqa: E402
    display_path,
    load_records,
    load_yaml_mapping,
    normalize_training_query_record,
    read_jsonl,
    resolve_dataset_root,
    resolve_repo_path,
    utc_now_iso,
    write_json,
    write_jsonl,
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
    return parser


def _coalesce(cli_value: Any, config_value: Any, default: Any = None) -> Any:
    return cli_value if cli_value is not None else (config_value if config_value is not None else default)


def _infer_task_type(record: Dict[str, Any], normalized: Dict[str, Any]) -> str:
    if record.get("task_type"):
        return str(record["task_type"])
    if record.get("capability"):
        return str(record["capability"])
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


def _normalize_base_question(record: Dict[str, Any], index: int) -> Dict[str, Any]:
    normalized = normalize_training_query_record(record)
    question_text = normalized["question_text"] or record.get("text")
    if not question_text:
        raise ValueError(f"Question record at index {index} is missing text.")

    schema = normalized["schema"]
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

    query_id = normalized["query_id"] or f"seed-base-{index:06d}"
    mentioned_companies = record.get("mentioned_companies")
    if not isinstance(mentioned_companies, list):
        mentioned_companies = normalized["mentioned_companies"]
    return {
        "query_id": f"seed-{query_id}" if not str(query_id).startswith("seed-") else str(query_id),
        "question_text": question_text,
        "schema": schema,
        "task_type": _infer_task_type(record, normalized),
        "company_name": normalized["company_name"],
        "mentioned_companies": mentioned_companies,
        "doc_ids": normalized["doc_ids"],
        "expected_filters": expected_filters,
        "source": source,
        "difficulty": _infer_difficulty(record, normalized),
        "should_refuse": bool(normalized["should_refuse"]),
    }


def _template_seed_record(template_id: str, report_record: Dict[str, Any]) -> Dict[str, Any]:
    template = _TEMPLATE_LIBRARY[template_id]
    company_name = str(report_record.get("company_name") or "").strip()
    report_year = int(report_record.get("report_year") or 0)
    doc_id = str(report_record.get("doc_id") or report_record.get("report_id") or "").strip()
    if not company_name or not report_year or not doc_id:
        raise ValueError(f"Annual report record is missing company_name/report_year/doc_id: {report_record!r}")

    return {
        "query_id": f"seed-auto-{template_id}-{doc_id}",
        "question_text": template["question_template"].format(
            company_name=company_name,
            report_year=report_year,
        ),
        "schema": template["schema"],
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
    }


def _dedupe_records(records: Iterable[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], int]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    skipped = 0
    for record in records:
        dedupe_key = (
            str(record.get("schema") or ""),
            str(record.get("question_text") or "").strip(),
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
    if questions_path is None:
        questions_path = dataset_root_path / "questions.json"
    if annual_report_path is None:
        annual_report_path = dataset_root_path / "metadata_store/annual_report.jsonl"
    if output_path is None or stats_output_path is None:
        raise ValueError("`output_path` and `stats_output_path` are required.")

    return {
        "config_path": config_path,
        "dataset_root_path": dataset_root_path,
        "questions_path": questions_path,
        "annual_report_path": annual_report_path,
        "output_path": output_path,
        "stats_output_path": stats_output_path,
        "max_existing_questions": max(0, int(_coalesce(args.max_existing_questions, config.get("max_existing_questions"), 0) or 0)),
        "max_template_reports": max(0, int(_coalesce(args.max_template_reports, config.get("max_template_reports"), 0) or 0)),
        "include_template_bootstrap": not bool(
            args.disable_template_bootstrap or not _coalesce(None, config.get("include_template_bootstrap"), True)
        ),
        "template_ids": [
            str(template_id)
            for template_id in (_coalesce(None, config.get("template_ids"), list(_TEMPLATE_LIBRARY.keys())) or [])
            if str(template_id) in _TEMPLATE_LIBRARY
        ],
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = _resolve_settings(args)

    question_records = load_records(settings["questions_path"])
    if settings["max_existing_questions"] > 0:
        question_records = question_records[: settings["max_existing_questions"]]

    seed_records = [
        _normalize_base_question(record, index)
        for index, record in enumerate(question_records, start=1)
    ]

    template_records: List[Dict[str, Any]] = []
    template_source_count = 0
    if settings["include_template_bootstrap"]:
        annual_reports = list(read_jsonl(settings["annual_report_path"]))
        if settings["max_template_reports"] > 0:
            annual_reports = annual_reports[: settings["max_template_reports"]]

        for report_record in annual_reports:
            for template_id in settings["template_ids"]:
                template_records.append(_template_seed_record(template_id, report_record))
        template_source_count = len(annual_reports)

    all_records, deduped_count = _dedupe_records([*seed_records, *template_records])
    write_jsonl(settings["output_path"], all_records)

    schema_counter = Counter(str(record.get("schema") or "") for record in all_records)
    source_counter = Counter(str(record.get("source") or "") for record in all_records)
    task_counter = Counter(str(record.get("task_type") or "") for record in all_records)
    stats = {
        "build_timestamp": utc_now_iso(),
        "config_path": display_path(settings["config_path"], REPO_ROOT),
        "dataset_root_path": display_path(settings["dataset_root_path"], REPO_ROOT),
        "questions_path": display_path(settings["questions_path"], REPO_ROOT),
        "annual_report_path": display_path(settings["annual_report_path"], REPO_ROOT),
        "output_path": display_path(settings["output_path"], REPO_ROOT),
        "max_existing_questions": settings["max_existing_questions"],
        "max_template_reports": settings["max_template_reports"],
        "template_ids": settings["template_ids"],
        "base_question_count": len(question_records),
        "template_report_count": template_source_count,
        "template_record_count": len(template_records),
        "deduped_records": deduped_count,
        "total_seed_records": len(all_records),
        "schema_distribution": dict(schema_counter),
        "source_distribution": dict(source_counter),
        "task_distribution": dict(task_counter),
    }
    write_json(settings["stats_output_path"], stats)


if __name__ == "__main__":
    main()
