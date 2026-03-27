from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


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


def load_answers_bundle(path: Path) -> Tuple[List[Dict], Dict]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict) and "answers" in payload:
        return payload["answers"], payload
    if isinstance(payload, list):
        return payload, {"answers": payload}
    raise ValueError(f"Unsupported answers format in {path}")


def compare_answers(pred_answers: List[Dict], ref_answers: List[Dict]) -> Dict:
    ref_by_question = {answer["question_text"]: answer for answer in ref_answers}
    matched = 0
    total = 0

    for pred in pred_answers:
        question_text = pred.get("question_text")
        if question_text not in ref_by_question:
            continue
        total += 1
        if _normalize_scalar(pred.get("value")) == _normalize_scalar(ref_by_question[question_text].get("value")):
            matched += 1

    return {
        "reference_match_count": matched,
        "reference_match_total": total,
        "reference_exact_match": round(matched / total, 4) if total else None,
    }


def summarize_answers(answers: Iterable[Dict]) -> Dict:
    answers = list(answers)
    total = len(answers)
    na_count = sum(1 for answer in answers if answer.get("value") == "N/A")
    answer_count = sum(1 for answer in answers if answer.get("value") not in (None, "N/A"))
    citation_count = sum(1 for answer in answers if answer.get("citations"))
    reference_count = sum(len(answer.get("references", [])) for answer in answers)
    confidence_counter = Counter(answer.get("confidence", "unknown") for answer in answers)

    return {
        "total_questions": total,
        "answer_rate": round(answer_count / total, 4) if total else 0.0,
        "na_rate": round(na_count / total, 4) if total else 0.0,
        "citation_coverage": round(citation_count / total, 4) if total else 0.0,
        "avg_references_per_answer": round(reference_count / total, 4) if total else 0.0,
        "confidence_distribution": dict(confidence_counter),
    }
