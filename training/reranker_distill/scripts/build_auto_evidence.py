from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import display_path, read_jsonl, resolve_repo_path, utc_now_iso, write_json  # noqa: E402


_AUTO_PREFIX = "seed-auto-"
_DEFAULT_OUTPUT_PATH = "training/reranker_distill/raw/teacher_answers_with_auto_evidence.jsonl"
_DEFAULT_STATS_PATH = "training/reranker_distill/manifests/auto_evidence_stats.json"
_DEFAULT_BASE_TEACHER_ANSWERS_PATH = "training/generator_sft/raw/teacher_answers_raw.jsonl"
_DEFAULT_DEBUG_INPUT_PATH = "training/generator_sft/raw/teacher_answers_debug.jsonl"
_DEFAULT_TEACHER_SCORES_PATH = "training/reranker_distill/raw/teacher_scores.jsonl"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a combined teacher-answer evidence file with generated evidence for seed-auto-* reranker queries."
    )
    parser.add_argument("--base-teacher-answers-path", type=Path, default=None)
    parser.add_argument("--debug-input-path", type=Path, default=None)
    parser.add_argument("--teacher-scores-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--stats-output-path", type=Path, default=None)
    parser.add_argument("--max-positive-pages", type=int, default=2)
    parser.add_argument("--no-copy-base", action="store_true", help="Write only generated auto evidence records.")
    return parser


def _template_id_from_query_id(query_id: str) -> str:
    if not query_id.startswith(_AUTO_PREFIX):
        return ""
    suffix = query_id[len(_AUTO_PREFIX):]
    for template_id in ("legal_representative", "cash_dividend", "revenue"):
        if suffix.startswith(f"{template_id}-"):
            return template_id
    return suffix.split("-", 1)[0]


def _doc_id_from_debug_or_score(debug_record: Dict[str, Any], score_records: Sequence[Dict[str, Any]]) -> str:
    route_info = debug_record.get("route_info") if isinstance(debug_record.get("route_info"), dict) else {}
    selected_report = route_info.get("selected_report") if isinstance(route_info.get("selected_report"), dict) else {}
    selected_doc_id = selected_report.get("sha1")
    if selected_doc_id:
        return str(selected_doc_id)

    candidate_doc_ids = route_info.get("candidate_doc_ids")
    if isinstance(candidate_doc_ids, list) and candidate_doc_ids:
        return str(candidate_doc_ids[0])

    for record in score_records:
        if record.get("doc_id"):
            return str(record["doc_id"])
    return ""


def _combined_candidate_text(record: Dict[str, Any]) -> str:
    return f"{record.get('section_name') or ''}\n{record.get('text') or ''}"


def _has_current_year_marker(record: Dict[str, Any], year: str) -> bool:
    text = _combined_candidate_text(record)
    return bool(
        re.search(rf"{re.escape(year)}\s*年(?:度)?", text)
        or "本期" in text
        or "本年" in text
        or "报告期" in text
    )


def _looks_like_positive_evidence(query_id: str, record: Dict[str, Any]) -> bool:
    template_id = _template_id_from_query_id(query_id)
    text = _combined_candidate_text(record)
    if template_id == "legal_representative":
        return bool(
            re.search(r"(?:公司)?法定代表人\s*[：:|]", text)
            or "公司的法定代表人" in text
            or "单位负责人或法定代表人" in text
        )
    if template_id == "cash_dividend":
        return bool(
            re.search(r"现金分红|现金红利|现金股利|派送现金|派发现金|现金派息|利润分配|股利分配|分红", text)
        )
    if template_id == "revenue":
        doc_id = str(record.get("doc_id") or "")
        year_match = re.search(r"_(20\d{2})_", doc_id)
        year = year_match.group(1) if year_match else ""
        has_revenue_marker = "营业收入" in text or "营业总收入" in text
        return has_revenue_marker and (not year or _has_current_year_marker(record, year))
    return False


def select_positive_evidence_records(
    query_id: str,
    score_records: Sequence[Dict[str, Any]],
    *,
    max_positive_pages: int = 2,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    selected_pages: set[tuple[str, int]] = set()
    sorted_records = sorted(
        score_records,
        key=lambda record: (
            int(record.get("teacher_rank") or 10_000),
            -float(record.get("teacher_score") or 0.0),
            str(record.get("candidate_id") or ""),
        ),
    )
    for record in sorted_records:
        page = record.get("page")
        if page is None or not _looks_like_positive_evidence(query_id, record):
            continue
        page_key = (str(record.get("doc_id") or ""), int(page))
        if page_key in selected_pages:
            continue
        selected.append(record)
        selected_pages.add(page_key)
        if len(selected) >= max(1, int(max_positive_pages)):
            break
    return selected


def build_auto_evidence_record(
    debug_record: Dict[str, Any],
    selected_score_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if not selected_score_records:
        raise ValueError("selected_score_records must not be empty")

    query_id = str(debug_record.get("query_id") or selected_score_records[0].get("query_id") or "")
    doc_id = _doc_id_from_debug_or_score(debug_record, selected_score_records)
    pages = [int(record["page"]) for record in selected_score_records if record.get("page") is not None]
    unique_pages = list(dict.fromkeys(pages))
    references = [
        {
            "pdf_sha1": str(record.get("doc_id") or doc_id),
            "page_index": int(record["page"]),
            "source": "auto_evidence_from_teacher_score",
            "candidate_id": record.get("candidate_id"),
            "teacher_score": round(float(record.get("teacher_score") or 0.0), 4),
            "teacher_rank": int(record.get("teacher_rank") or 0),
        }
        for record in selected_score_records
        if record.get("page") is not None
    ]
    evidence_chunks = [
        {
            "page": int(record["page"]),
            "candidate_id": record.get("candidate_id"),
            "teacher_score": round(float(record.get("teacher_score") or 0.0), 4),
            "teacher_rank": int(record.get("teacher_rank") or 0),
            "text": str(record.get("text") or "")[:500],
        }
        for record in selected_score_records
        if record.get("page") is not None
    ]

    answer: Dict[str, Any] = {
        "relevant_pages": unique_pages,
        "references": references,
        "retrieval_report_groups": [
            {
                "doc_id": doc_id,
                "evidence_chunks": evidence_chunks,
            }
        ],
        "evidence_source": "auto_evidence_from_debug_and_teacher_scores",
    }
    if str(debug_record.get("schema") or selected_score_records[0].get("schema") or "") == "number" and unique_pages:
        answer["table_grounding_result"] = {
            "source_doc_id": doc_id,
            "page": unique_pages[0],
            "source": "auto_evidence_from_teacher_score",
        }

    return {
        "query_id": query_id,
        "question_text": debug_record.get("question_text") or selected_score_records[0].get("question_text"),
        "schema": debug_record.get("schema") or selected_score_records[0].get("schema"),
        "company_name": debug_record.get("company_name") or selected_score_records[0].get("company_name"),
        "doc_ids": [doc_id] if doc_id else [],
        "route_info": debug_record.get("route_info"),
        "query_plan": debug_record.get("query_plan"),
        "retrieval_pages": debug_record.get("retrieval_pages"),
        "validation_flags": debug_record.get("validation_flags") or [],
        "answer": answer,
        "auto_evidence": {
            "source": "teacher_answers_debug+teacher_scores",
            "selected_candidate_ids": [record.get("candidate_id") for record in selected_score_records],
            "selected_pages": unique_pages,
            "build_timestamp": utc_now_iso(),
        },
    }


def _index_records_by_query_id(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for record in records:
        query_id = str(record.get("query_id") or "").strip()
        if query_id:
            indexed[query_id] = record
    return indexed


def _group_scores_by_query_id(records: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        query_id = str(record.get("query_id") or "").strip()
        if query_id:
            grouped[query_id].append(record)
    return dict(grouped)


def _resolve_path(value: Path | None, default: str) -> Path:
    resolved = resolve_repo_path(REPO_ROOT, value or default)
    if resolved is None:
        raise ValueError(f"Failed to resolve path for {value or default}")
    return resolved


def main() -> None:
    args = build_arg_parser().parse_args()
    base_teacher_answers_path = _resolve_path(args.base_teacher_answers_path, _DEFAULT_BASE_TEACHER_ANSWERS_PATH)
    debug_input_path = _resolve_path(args.debug_input_path, _DEFAULT_DEBUG_INPUT_PATH)
    teacher_scores_path = _resolve_path(args.teacher_scores_path, _DEFAULT_TEACHER_SCORES_PATH)
    output_path = _resolve_path(args.output_path, _DEFAULT_OUTPUT_PATH)
    stats_output_path = _resolve_path(args.stats_output_path, _DEFAULT_STATS_PATH)

    base_records = list(read_jsonl(base_teacher_answers_path)) if base_teacher_answers_path.exists() else []
    base_query_ids = {str(record.get("query_id") or "") for record in base_records if record.get("query_id")}
    debug_by_query_id = _index_records_by_query_id(read_jsonl(debug_input_path))
    scores_by_query_id = _group_scores_by_query_id(read_jsonl(teacher_scores_path))

    generated_records: List[Dict[str, Any]] = []
    skipped_existing = 0
    skipped_missing_debug = 0
    skipped_no_positive_evidence = 0
    template_counts: Dict[str, int] = defaultdict(int)

    for query_id in sorted(scores_by_query_id):
        if not query_id.startswith(_AUTO_PREFIX):
            continue
        if query_id in base_query_ids:
            skipped_existing += 1
            continue
        debug_record = debug_by_query_id.get(query_id)
        if debug_record is None:
            skipped_missing_debug += 1
            continue
        selected = select_positive_evidence_records(
            query_id,
            scores_by_query_id[query_id],
            max_positive_pages=args.max_positive_pages,
        )
        if not selected:
            skipped_no_positive_evidence += 1
            continue
        generated_records.append(build_auto_evidence_record(debug_record, selected))
        template_counts[_template_id_from_query_id(query_id)] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        if not args.no_copy_base:
            for record in base_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        for record in generated_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    write_json(
        stats_output_path,
        {
            "build_timestamp": utc_now_iso(),
            "base_teacher_answers_path": display_path(base_teacher_answers_path, REPO_ROOT),
            "debug_input_path": display_path(debug_input_path, REPO_ROOT),
            "teacher_scores_path": display_path(teacher_scores_path, REPO_ROOT),
            "output_path": display_path(output_path, REPO_ROOT),
            "copied_base_record_count": 0 if args.no_copy_base else len(base_records),
            "generated_auto_evidence_count": len(generated_records),
            "skipped_existing_count": skipped_existing,
            "skipped_missing_debug_count": skipped_missing_debug,
            "skipped_no_positive_evidence_count": skipped_no_positive_evidence,
            "generated_by_template": dict(sorted(template_counts.items())),
            "max_positive_pages": int(args.max_positive_pages),
        },
    )


if __name__ == "__main__":
    main()
