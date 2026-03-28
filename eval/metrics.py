from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _normalize_scalar(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return json.dumps(sorted(str(item) for item in value), ensure_ascii=False)
    return " ".join(str(value).strip().lower().split())


def _normalize_page_set(answer: Dict) -> set[int]:
    pages: set[int] = set()
    for reference in answer.get("references", []) or []:
        page_index = reference.get("page_index")
        if isinstance(page_index, int):
            pages.add(page_index + 1)
    for citation in answer.get("citations", []) or []:
        page = citation.get("page")
        if isinstance(page, int):
            pages.add(page)
    return pages


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if not denominator:
        return None
    return round(float(numerator) / float(denominator), 4)


def load_answers_bundle(path: Path) -> Tuple[List[Dict], Dict]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict) and "answers" in payload:
        return payload["answers"], payload
    if isinstance(payload, dict) and "questions" in payload:
        return payload["questions"], payload
    if isinstance(payload, list):
        return payload, {"answers": payload}
    raise ValueError(f"Unsupported answers format in {path}")


def build_debug_index(debug_payload: Optional[Dict]) -> Dict[str, Dict]:
    if not isinstance(debug_payload, dict):
        return {}

    questions = debug_payload.get("questions") or []
    answer_details = debug_payload.get("answer_details") or []
    indexed: Dict[str, Dict] = {}

    for index, question in enumerate(questions):
        question_text = question.get("question_text") or question.get("question")
        if not question_text:
            continue
        detail = answer_details[index] if index < len(answer_details) and answer_details[index] else {}
        indexed[question_text] = detail
    return indexed


def summarize_answers(answers: Iterable[Dict]) -> Dict:
    answers = list(answers)
    total = len(answers)
    na_count = sum(1 for answer in answers if answer.get("value") == "N/A")
    answer_count = sum(1 for answer in answers if answer.get("value") not in (None, "N/A"))
    citation_count = sum(1 for answer in answers if answer.get("citations"))
    reference_count = sum(len(answer.get("references", [])) for answer in answers)
    confidence_counter = Counter(answer.get("confidence", "unknown") for answer in answers)

    by_kind: Dict[str, Dict[str, float | int]] = {}
    grouped_answers: Dict[str, List[Dict]] = defaultdict(list)
    for answer in answers:
        grouped_answers[answer.get("kind", "unknown")].append(answer)

    for kind, kind_answers in grouped_answers.items():
        kind_total = len(kind_answers)
        kind_answered = sum(1 for answer in kind_answers if answer.get("value") not in (None, "N/A"))
        kind_na = sum(1 for answer in kind_answers if answer.get("value") == "N/A")
        kind_citations = sum(1 for answer in kind_answers if answer.get("citations"))
        kind_references = sum(len(answer.get("references", [])) for answer in kind_answers)
        by_kind[kind] = {
            "total": kind_total,
            "answer_rate": round(kind_answered / kind_total, 4) if kind_total else 0.0,
            "na_rate": round(kind_na / kind_total, 4) if kind_total else 0.0,
            "citation_coverage": round(kind_citations / kind_total, 4) if kind_total else 0.0,
            "avg_references_per_answer": round(kind_references / kind_total, 4) if kind_total else 0.0,
        }

    return {
        "total_questions": total,
        "answer_rate": round(answer_count / total, 4) if total else 0.0,
        "na_rate": round(na_count / total, 4) if total else 0.0,
        "citation_coverage": round(citation_count / total, 4) if total else 0.0,
        "avg_references_per_answer": round(reference_count / total, 4) if total else 0.0,
        "confidence_distribution": dict(confidence_counter),
        "question_type_breakdown": by_kind,
    }


def compare_answers(pred_answers: List[Dict], ref_answers: List[Dict], debug_payload: Optional[Dict] = None) -> Dict:
    ref_by_question = {answer["question_text"]: answer for answer in ref_answers if answer.get("question_text")}
    debug_index = build_debug_index(debug_payload)

    matched = 0
    total = 0
    reference_page_hit = 0
    citation_page_hit = 0
    retrieval_hit = 0
    reference_precision_sum = 0.0
    citation_precision_sum = 0.0
    confidence_groups: Dict[str, List[int]] = defaultdict(list)
    by_kind: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "matched": 0})

    for pred in pred_answers:
        question_text = pred.get("question_text")
        if question_text not in ref_by_question:
            continue

        ref = ref_by_question[question_text]
        kind = pred.get("kind", "unknown")
        total += 1
        by_kind[kind]["total"] += 1

        is_match = _normalize_scalar(pred.get("value")) == _normalize_scalar(ref.get("value"))
        if is_match:
            matched += 1
            by_kind[kind]["matched"] += 1

        confidence_groups[pred.get("confidence", "unknown")].append(1 if is_match else 0)

        ref_pages = _normalize_page_set(ref)
        pred_pages = {reference.get("page_index", -1) + 1 for reference in pred.get("references", []) if isinstance(reference.get("page_index"), int)}
        citation_pages = {citation.get("page") for citation in pred.get("citations", []) if isinstance(citation.get("page"), int)}

        if ref_pages:
            if pred_pages & ref_pages:
                reference_page_hit += 1
            if citation_pages & ref_pages:
                citation_page_hit += 1
            if pred_pages:
                reference_precision_sum += len(pred_pages & ref_pages) / len(pred_pages)
            if citation_pages:
                citation_precision_sum += len(citation_pages & ref_pages) / len(citation_pages)

            detail = debug_index.get(question_text) or {}
            retrieval_pages = {
                page
                for page in (detail.get("retrieval_pages") or [])
                if isinstance(page, int)
            }
            if not retrieval_pages and detail.get("retrieval_results"):
                retrieval_pages = {
                    result.get("page")
                    for result in detail.get("retrieval_results", [])
                    if isinstance(result.get("page"), int)
                }
            if retrieval_pages & ref_pages:
                retrieval_hit += 1

    confidence_calibration = {
        confidence: {
            "count": len(values),
            "reference_exact_match": round(sum(values) / len(values), 4) if values else None,
        }
        for confidence, values in confidence_groups.items()
    }

    return {
        "reference_match_count": matched,
        "reference_match_total": total,
        "reference_exact_match": _safe_ratio(matched, total),
        "reference_page_hit": _safe_ratio(reference_page_hit, total),
        "citation_page_hit": _safe_ratio(citation_page_hit, total),
        "retrieval_hit_at_k": _safe_ratio(retrieval_hit, total),
        "avg_reference_page_precision": round(reference_precision_sum / total, 4) if total else None,
        "avg_citation_page_precision": round(citation_precision_sum / total, 4) if total else None,
        "confidence_calibration": confidence_calibration,
        "reference_exact_match_by_kind": {
            kind: _safe_ratio(values["matched"], values["total"])
            for kind, values in by_kind.items()
        },
    }
