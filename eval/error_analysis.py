from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Dict, List, Optional

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from eval.metrics import build_debug_index, load_answers_bundle


def _normalize_answer(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return json.dumps(sorted(str(item) for item in value), ensure_ascii=False)
    return str(value).strip().lower()


def _classify_case(pred: Dict, ref: Optional[Dict], detail: Dict) -> tuple[str, str]:
    if pred.get("error"):
        error_message = pred["error"]
        lowered = error_message.lower()
        if "company name found" in lowered or "ambiguous" in lowered:
            return "routing", error_message
        if "parse" in lowered or "docling" in lowered or "pdf" in lowered:
            return "parse", error_message
        return "generation", error_message

    validation_flags = detail.get("validation_flags") or pred.get("validation_flags") or []
    if any("mismatch" in flag or "validation" in flag for flag in validation_flags):
        return "validation", ", ".join(validation_flags)

    retrieval_pages = detail.get("retrieval_pages") or []
    citations = pred.get("citations") or []
    if not retrieval_pages or not citations:
        return "retrieval", "Weak retrieval support or missing citations"

    if ref and _normalize_answer(pred.get("value")) != _normalize_answer(ref.get("value")):
        return "generation", "Reference answer mismatch despite grounded retrieval"

    return "correct", "Matched reference or no comparable reference"


def summarize_error_analysis(pred_answers: List[Dict], ref_answers: List[Dict], debug_payload: Optional[Dict] = None) -> Dict:
    ref_by_question = {answer.get("question_text"): answer for answer in ref_answers if answer.get("question_text")}
    debug_index = build_debug_index(debug_payload)

    stage_counter: Counter[str] = Counter()
    stage_examples: Dict[str, List[Dict]] = defaultdict(list)

    for pred in pred_answers:
        question_text = pred.get("question_text")
        detail = debug_index.get(question_text) or {}
        stage, reason = _classify_case(pred, ref_by_question.get(question_text), detail)
        stage_counter[stage] += 1

        if stage != "correct" and len(stage_examples[stage]) < 3:
            stage_examples[stage].append(
                {
                    "question_text": question_text,
                    "kind": pred.get("kind"),
                    "value": pred.get("value"),
                    "confidence": pred.get("confidence"),
                    "reason": reason,
                }
            )

    return {
        "stage_counts": dict(stage_counter),
        "stage_examples": dict(stage_examples),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize FinaRAG error stages from an answers bundle and optional debug bundle.")
    parser.add_argument("--answers-file", type=Path, required=True)
    parser.add_argument("--reference-answers", type=Path, default=None)
    parser.add_argument("--debug-file", type=Path, default=None)
    args = parser.parse_args()

    pred_answers, _ = load_answers_bundle(args.answers_file)
    ref_answers = []
    if args.reference_answers and args.reference_answers.exists():
        ref_answers, _ = load_answers_bundle(args.reference_answers)

    debug_payload = None
    if args.debug_file and args.debug_file.exists():
        _, debug_payload = load_answers_bundle(args.debug_file)

    report = summarize_error_analysis(pred_answers, ref_answers, debug_payload=debug_payload)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
