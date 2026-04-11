from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import (  # noqa: E402
    append_jsonl,
    display_path,
    load_yaml_mapping,
    normalize_training_query_record,
    read_jsonl,
    resolve_repo_path,
    utc_now_iso,
    write_json,
)


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


@dataclass
class LabelConfig:
    positive_teacher_score: float = 0.85
    support_teacher_score: float = 0.55
    positive_rank_cutoff: int = 3
    neighbor_page_window: int = 1


@dataclass
class EvidenceSummary:
    query_id: str
    question_text: str
    schema: str
    query_doc_ids: set[str]
    positive_page_keys: set[tuple[str, int]]
    citation_page_keys: set[tuple[str, int]]
    table_grounding_keys: set[tuple[str, int]]
    positive_doc_ids: set[str]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build pointwise reranker labels from teacher scores and answer evidence.")
    parser.add_argument("--config-path", type=Path, default=None, help="YAML config path.")
    parser.add_argument("--teacher-scores-path", type=Path, default=None, help="Input teacher_scores.jsonl path.")
    parser.add_argument("--teacher-answers-path", type=Path, default=None, help="Optional teacher_answers_raw.jsonl path.")
    parser.add_argument("--output-path", type=Path, default=None, help="Output pointwise_labels_raw.jsonl path.")
    parser.add_argument("--rejected-pairs-path", type=Path, default=None, help="Rejected pair JSONL path.")
    parser.add_argument("--stats-output-path", type=Path, default=None, help="Build stats JSON path.")
    parser.add_argument("--label-positive-teacher-score", type=float, default=None, help="Score threshold for teacher-supported positives.")
    parser.add_argument("--label-support-teacher-score", type=float, default=None, help="Score threshold for teacher-supported medium labels.")
    parser.add_argument("--label-positive-rank-cutoff", type=int, default=None, help="Rank threshold for teacher-supported positives.")
    parser.add_argument("--neighbor-page-window", type=int, default=None, help="Nearby page window for weak positives.")
    return parser


def _coalesce(cli_value: Any, config_value: Any, default: Any = None) -> Any:
    return cli_value if cli_value is not None else (config_value if config_value is not None else default)


def summarize_answer_evidence(record: Dict[str, Any]) -> EvidenceSummary:
    normalized = normalize_training_query_record(record)
    answer = record.get("answer") if isinstance(record.get("answer"), dict) else {}
    relevant_pages = {
        int(page)
        for page in (answer.get("relevant_pages") or record.get("relevant_pages") or [])
        if page is not None
    }
    query_doc_ids = set(normalized["doc_ids"])
    positive_page_keys: set[tuple[str, int]] = set()
    citation_page_keys: set[tuple[str, int]] = set()
    table_grounding_keys: set[tuple[str, int]] = set()

    references = answer.get("references") or record.get("references") or []
    for reference in references:
        if not isinstance(reference, dict):
            continue
        pdf_sha1 = reference.get("pdf_sha1")
        page_index = reference.get("page_index")
        if pdf_sha1 and page_index is not None:
            citation_page_keys.add((str(pdf_sha1), int(page_index)))
            query_doc_ids.add(str(pdf_sha1))

    retrieval_groups = answer.get("retrieval_report_groups") or record.get("retrieval_report_groups") or []
    for group in retrieval_groups:
        if not isinstance(group, dict):
            continue
        doc_id = group.get("doc_id")
        if not doc_id:
            continue
        doc_id = str(doc_id)
        query_doc_ids.add(doc_id)
        for chunk in group.get("evidence_chunks") or []:
            if not isinstance(chunk, dict):
                continue
            page = chunk.get("page")
            if page is None:
                continue
            page_key = (doc_id, int(page))
            if int(page) in relevant_pages:
                positive_page_keys.add(page_key)

    if not positive_page_keys and relevant_pages and query_doc_ids:
        for doc_id in query_doc_ids:
            for page in relevant_pages:
                positive_page_keys.add((doc_id, int(page)))

    table_grounding = answer.get("table_grounding_result") or record.get("table_grounding_result") or {}
    grounding_doc_id = table_grounding.get("source_doc_id")
    grounding_page = table_grounding.get("page")
    if grounding_doc_id and grounding_page is not None:
        grounding_key = (str(grounding_doc_id), int(grounding_page))
        table_grounding_keys.add(grounding_key)
        positive_page_keys.add(grounding_key)
        query_doc_ids.add(str(grounding_doc_id))

    positive_doc_ids = {doc_id for doc_id, _ in positive_page_keys}
    return EvidenceSummary(
        query_id=str(normalized["query_id"] or ""),
        question_text=str(normalized["question_text"] or record.get("question_text") or record.get("query") or ""),
        schema=str(normalized["schema"] or record.get("schema") or ""),
        query_doc_ids=query_doc_ids,
        positive_page_keys=positive_page_keys,
        citation_page_keys=citation_page_keys,
        table_grounding_keys=table_grounding_keys,
        positive_doc_ids=positive_doc_ids,
    )


def assign_hard_label(
    score_record: Dict[str, Any],
    evidence: EvidenceSummary,
    config: LabelConfig,
) -> tuple[int, List[str], bool]:
    doc_id = str(score_record.get("doc_id") or "")
    page = score_record.get("page")
    teacher_score = float(score_record.get("teacher_score") or 0.0)
    teacher_rank = int(score_record.get("teacher_rank") or 0)
    schema = str(score_record.get("schema") or evidence.schema or "")
    if page is None:
        page = -1
    page = int(page)
    page_key = (doc_id, page)

    label = 0
    sources: List[str] = []

    if page_key in evidence.positive_page_keys:
        label = 2
        sources.append("answer_relevant_page")
    if page_key in evidence.citation_page_keys:
        label = 2
        sources.append("citation_hit")
    if schema == "number" and page_key in evidence.table_grounding_keys:
        label = 2
        sources.append("table_grounding_hit")

    if label < 2 and any(
        positive_doc_id == doc_id and abs(positive_page - page) <= config.neighbor_page_window
        for positive_doc_id, positive_page in evidence.positive_page_keys
    ):
        label = max(label, 1)
        sources.append("answer_neighbor_page")

    if label < 2 and teacher_score >= config.positive_teacher_score and teacher_rank <= config.positive_rank_cutoff:
        label = max(label, 1)
        sources.append("teacher_reranker")
    elif label < 1 and teacher_score >= config.support_teacher_score:
        label = 1
        sources.append("teacher_reranker")

    is_hard_negative = False
    if label == 0:
        if doc_id and (doc_id in evidence.query_doc_ids or doc_id in evidence.positive_doc_ids):
            is_hard_negative = True
            sources.append("hard_negative_same_report")
        sources.append("teacher_reranker_low")

    return label, _dedupe_preserve_order(sources), is_hard_negative


def _group_by_query_id(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        query_id = str(record.get("query_id") or "")
        grouped.setdefault(query_id, []).append(record)
    return grouped


def _load_teacher_answers(path: Path | None) -> Dict[str, EvidenceSummary]:
    if path is None or not path.exists():
        return {}
    evidence_by_query: Dict[str, EvidenceSummary] = {}
    for record in read_jsonl(path):
        summary = summarize_answer_evidence(record)
        if summary.query_id:
            evidence_by_query[summary.query_id] = summary
    return evidence_by_query


def _resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    default_config_path = REPO_ROOT / "training/reranker_distill/configs/data_build.example.yaml"
    config_path = args.config_path or (default_config_path if default_config_path.exists() else None)
    config = load_yaml_mapping(config_path)

    teacher_scores_path = resolve_repo_path(REPO_ROOT, _coalesce(args.teacher_scores_path, config.get("teacher_scores_path")))
    teacher_answers_path = resolve_repo_path(REPO_ROOT, _coalesce(args.teacher_answers_path, config.get("teacher_answers_path")))
    output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.output_path, config.get("pointwise_output_path")))
    rejected_pairs_path = resolve_repo_path(REPO_ROOT, _coalesce(args.rejected_pairs_path, config.get("rejected_pairs_path")))
    stats_output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.stats_output_path, config.get("stats_output_path")))
    if teacher_scores_path is None or output_path is None or rejected_pairs_path is None or stats_output_path is None:
        raise ValueError("teacher_scores/output/rejected/stats paths are required.")

    label_config = LabelConfig(
        positive_teacher_score=float(_coalesce(args.label_positive_teacher_score, config.get("label_positive_teacher_score"), 0.85)),
        support_teacher_score=float(_coalesce(args.label_support_teacher_score, config.get("label_support_teacher_score"), 0.55)),
        positive_rank_cutoff=max(1, int(_coalesce(args.label_positive_rank_cutoff, config.get("label_positive_rank_cutoff"), 3))),
        neighbor_page_window=max(0, int(_coalesce(args.neighbor_page_window, config.get("neighbor_page_window"), 1))),
    )
    return {
        "config_path": config_path,
        "teacher_scores_path": teacher_scores_path,
        "teacher_answers_path": teacher_answers_path,
        "output_path": output_path,
        "rejected_pairs_path": rejected_pairs_path,
        "stats_output_path": stats_output_path,
        "label_config": label_config,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = _resolve_settings(args)
    evidence_by_query = _load_teacher_answers(settings["teacher_answers_path"])
    score_records = list(read_jsonl(settings["teacher_scores_path"]))
    grouped_scores = _group_by_query_id(score_records)

    pair_counter = 0
    label_histogram = {"0": 0, "1": 0, "2": 0}
    queries_without_label2: List[str] = []
    queries_without_three_negatives: List[str] = []
    rejected_count = 0

    for query_id in sorted(grouped_scores):
        query_records = sorted(
            grouped_scores[query_id],
            key=lambda item: (
                int(item.get("teacher_rank") or 10_000),
                -float(item.get("teacher_score") or 0.0),
                str(item.get("candidate_id") or ""),
            ),
        )

        fallback_evidence = EvidenceSummary(
            query_id=query_id,
            question_text=str(query_records[0].get("question_text") or ""),
            schema=str(query_records[0].get("schema") or ""),
            query_doc_ids={str(item.get("doc_id")) for item in query_records if item.get("doc_id")},
            positive_page_keys=set(),
            citation_page_keys=set(),
            table_grounding_keys=set(),
            positive_doc_ids=set(),
        )
        evidence = evidence_by_query.get(query_id, fallback_evidence)
        query_label2 = 0
        query_negatives = 0

        for record in query_records:
            if not record.get("candidate_id") or not record.get("query_id") or not record.get("text"):
                append_jsonl(
                    settings["rejected_pairs_path"],
                    {
                        "query_id": record.get("query_id"),
                        "candidate_id": record.get("candidate_id"),
                        "reason": "missing_required_fields",
                        "record": record,
                        "build_timestamp": utc_now_iso(),
                    },
                )
                rejected_count += 1
                continue

            hard_label, label_source, is_hard_negative = assign_hard_label(
                record,
                evidence,
                settings["label_config"],
            )
            pair_counter += 1
            label_histogram[str(hard_label)] += 1
            if hard_label == 2:
                query_label2 += 1
            if hard_label == 0:
                query_negatives += 1

            output_record = {
                "pair_id": f"pair-{pair_counter:06d}",
                "query_id": str(record.get("query_id")),
                "candidate_id": str(record.get("candidate_id")),
                "query": record.get("question_text") or evidence.question_text,
                "passage": record.get("text"),
                "schema": record.get("schema") or evidence.schema,
                "teacher_score": round(float(record.get("teacher_score") or 0.0), 4),
                "teacher_rank": int(record.get("teacher_rank") or 0),
                "hard_label": hard_label,
                "label_source": label_source,
                "doc_id": record.get("doc_id"),
                "page": record.get("page"),
                "chunk_id": record.get("chunk_id"),
                "base_score": round(float(record.get("base_score") or 0.0), 4),
                "retrieval_sources": list(record.get("retrieval_sources", [])),
                "section_name": record.get("section_name"),
                "is_hard_negative": bool(is_hard_negative),
            }
            append_jsonl(settings["output_path"], output_record)

        if query_label2 == 0:
            queries_without_label2.append(query_id)
        if query_negatives < 3:
            queries_without_three_negatives.append(query_id)

    stats = {
        "build_timestamp": utc_now_iso(),
        "config_path": display_path(settings["config_path"], REPO_ROOT),
        "teacher_scores_path": display_path(settings["teacher_scores_path"], REPO_ROOT),
        "teacher_answers_path": (
            display_path(settings["teacher_answers_path"], REPO_ROOT)
            if settings["teacher_answers_path"] is not None and settings["teacher_answers_path"].exists()
            else None
        ),
        "output_path": display_path(settings["output_path"], REPO_ROOT),
        "rejected_pairs_path": display_path(settings["rejected_pairs_path"], REPO_ROOT),
        "label_config": json.loads(json.dumps(settings["label_config"].__dict__, ensure_ascii=False)),
        "total_queries": len(grouped_scores),
        "total_pairs_written": pair_counter,
        "rejected_pairs": rejected_count,
        "label_histogram": label_histogram,
        "queries_without_label2": queries_without_label2,
        "queries_without_three_negatives": queries_without_three_negatives,
    }
    write_json(settings["stats_output_path"], stats)


if __name__ == "__main__":
    main()
