from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from eval.dataset_schema import FinanceEvalQuestion, load_question_set
from eval.metrics import build_debug_index, load_answers_bundle


def _normalize_question_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _default_debug_file(answers_file: Path) -> Path | None:
    candidate = answers_file.with_name(answers_file.stem + "_debug" + answers_file.suffix)
    return candidate if candidate.exists() else None


def _default_output_paths(answers_file: Path) -> Tuple[Path, Path]:
    pred_path = answers_file.with_name(answers_file.stem + ".finance_eval.json")
    debug_path = answers_file.with_name(answers_file.stem + ".finance_eval_debug.json")
    return pred_path, debug_path


def _normalize_reference(reference: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(reference)

    pdf_sha1 = normalized.get("pdf_sha1") or normalized.get("source") or normalized.get("doc_id")
    if pdf_sha1:
        normalized["pdf_sha1"] = str(pdf_sha1)

    page = normalized.get("page")
    if isinstance(page, int) and page > 0:
        normalized["page_index"] = page - 1
    else:
        page_index = normalized.get("page_index")
        if isinstance(page_index, int):
            normalized["page_index"] = page_index

    return {
        key: value
        for key, value in normalized.items()
        if key in {"pdf_sha1", "page_index", "chunk_id", "section_name", "evidence_type"}
    }


def _normalize_citation(citation: Dict[str, Any], references: List[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = dict(citation)
    if "source" not in normalized:
        for reference in references:
            if reference.get("pdf_sha1"):
                normalized["source"] = reference["pdf_sha1"]
                break
    return normalized


def _synthesize_debug_detail(answer: Dict[str, Any]) -> Dict[str, Any]:
    retrieval_results: List[Dict[str, Any]] = []
    for citation in answer.get("citations", []) or []:
        page = citation.get("page")
        snippet = " ".join(str(citation.get("evidence_snippet", "")).split())
        if not isinstance(page, int) or not snippet:
            continue
        metadata = {
            "sha1_name": citation.get("source"),
            "company_name": citation.get("company_name"),
            "security_code": citation.get("security_code"),
            "stock_code": citation.get("stock_code"),
            "report_year": citation.get("report_year"),
            "doc_source_type": citation.get("doc_source_type"),
            "currency": citation.get("currency"),
            "unit_hint": citation.get("unit"),
            "chunk_type": citation.get("chunk_type"),
        }
        retrieval_results.append(
            {
                "page": page,
                "text": snippet,
                "metadata": {key: value for key, value in metadata.items() if value is not None},
            }
        )

    return {
        "retrieval_results": retrieval_results,
        "retrieval_pages": [result["page"] for result in retrieval_results],
        "validation_flags": answer.get("validation_flags", []),
        "confidence": answer.get("confidence"),
        "confidence_reason": answer.get("confidence_reason", ""),
        "route_info": answer.get("route_info", {}),
    }


def _normalize_answer(
    raw_answer: Dict[str, Any],
    question: FinanceEvalQuestion,
) -> Dict[str, Any]:
    references = [_normalize_reference(reference) for reference in raw_answer.get("references", []) or []]
    references = [reference for reference in references if "pdf_sha1" in reference and isinstance(reference.get("page_index"), int)]

    normalized = {
        "question_id": question.question_id,
        "question_text": question.question_text,
        "kind": raw_answer.get("kind") or question.kind,
        "value": raw_answer.get("value", raw_answer.get("final_answer")),
        "references": references,
        "citations": [
            _normalize_citation(citation, references)
            for citation in (raw_answer.get("citations", []) or [])
        ],
        "confidence": raw_answer.get("confidence", "low"),
    }

    optional_fields = [
        "confidence_reason",
        "validation_flags",
        "route_info",
        "reasoning_process",
        "table_grounding_result",
    ]
    for field in optional_fields:
        if field in raw_answer:
            normalized[field] = raw_answer[field]
    return normalized


def export_finance_eval_bundle(
    *,
    questions_file: Path,
    answers_file: Path,
    pred_answers_out: Path,
    debug_out: Path,
    debug_file: Path | None = None,
    fill_missing: bool = True,
) -> Dict[str, Any]:
    question_set = load_question_set(questions_file)
    raw_answers, raw_payload = load_answers_bundle(answers_file)

    raw_debug_payload = None
    if debug_file is not None and debug_file.exists():
        raw_debug_payload = _load_json(debug_file)
    raw_debug_index = build_debug_index(raw_debug_payload)

    question_by_text = {
        _normalize_question_text(question.question_text): question
        for question in question_set.questions
    }
    answer_by_text = {
        _normalize_question_text(answer.get("question_text") or answer.get("question")): answer
        for answer in raw_answers
        if _normalize_question_text(answer.get("question_text") or answer.get("question"))
    }

    exported_answers: List[Dict[str, Any]] = []
    exported_debug_questions: List[Dict[str, Any]] = []
    exported_debug_details: List[Dict[str, Any]] = []
    matched_questions = 0
    missing_questions: List[str] = []

    for question in question_set.questions:
        question_text = _normalize_question_text(question.question_text)
        raw_answer = answer_by_text.get(question_text)
        exported_debug_questions.append(
            {
                "question_id": question.question_id,
                "question_text": question.question_text,
                "kind": question.kind,
            }
        )

        if raw_answer is None:
            missing_questions.append(question.question_id)
            if not fill_missing:
                exported_debug_details.append({})
                continue
            normalized_answer = {
                "question_id": question.question_id,
                "question_text": question.question_text,
                "kind": question.kind,
                "value": "N/A",
                "references": [],
                "citations": [],
                "confidence": "low",
                "confidence_reason": "missing_prediction_from_raw_answers_bundle",
                "validation_flags": ["missing_prediction"],
            }
            debug_detail = _synthesize_debug_detail(normalized_answer)
        else:
            matched_questions += 1
            normalized_answer = _normalize_answer(raw_answer, question)
            debug_detail = raw_debug_index.get(question.question_text) or _synthesize_debug_detail(normalized_answer)

        exported_answers.append(normalized_answer)
        exported_debug_details.append(debug_detail)

    unmatched_raw_answers = []
    for raw_answer in raw_answers:
        raw_question_text = _normalize_question_text(raw_answer.get("question_text") or raw_answer.get("question"))
        if raw_question_text and raw_question_text not in question_by_text:
            unmatched_raw_answers.append(raw_answer.get("question_text") or raw_answer.get("question"))

    pred_payload = {
        "answers": exported_answers,
        "details": {
            "source_answers_file": str(answers_file),
            "source_debug_file": str(debug_file) if debug_file is not None else None,
            "source_details": raw_payload.get("details"),
            "exporter": "finance_eval_bundle_v1",
            "matched_questions": matched_questions,
            "missing_questions": missing_questions,
            "unmatched_raw_answers": unmatched_raw_answers,
        },
    }
    debug_payload = {
        "questions": exported_debug_questions,
        "answer_details": exported_debug_details,
        "statistics": {
            "total_questions": len(question_set.questions),
            "matched_questions": matched_questions,
            "missing_questions": len(missing_questions),
            "unmatched_raw_answers": len(unmatched_raw_answers),
        },
        "details": {
            "source_answers_file": str(answers_file),
            "source_debug_file": str(debug_file) if debug_file is not None else None,
            "exporter": "finance_eval_bundle_v1",
        },
    }

    _write_json(pred_answers_out, pred_payload)
    _write_json(debug_out, debug_payload)

    return {
        "questions_file": str(questions_file),
        "answers_file": str(answers_file),
        "debug_file": str(debug_file) if debug_file is not None else None,
        "pred_answers_out": str(pred_answers_out),
        "debug_out": str(debug_out),
        "matched_questions": matched_questions,
        "missing_questions": missing_questions,
        "unmatched_raw_answers": unmatched_raw_answers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export current FinaRAG answer bundles into finance_eval-ready pred/debug files."
    )
    parser.add_argument("--questions-file", type=Path, required=True)
    parser.add_argument("--answers-file", type=Path, required=True)
    parser.add_argument("--debug-file", type=Path, default=None)
    parser.add_argument("--pred-answers-out", type=Path, default=None)
    parser.add_argument("--debug-out", type=Path, default=None)
    parser.add_argument("--no-fill-missing", action="store_true")
    args = parser.parse_args()

    debug_file = args.debug_file or _default_debug_file(args.answers_file)
    pred_answers_out, debug_out = _default_output_paths(args.answers_file)
    if args.pred_answers_out is not None:
        pred_answers_out = args.pred_answers_out
    if args.debug_out is not None:
        debug_out = args.debug_out

    report = export_finance_eval_bundle(
        questions_file=args.questions_file,
        answers_file=args.answers_file,
        pred_answers_out=pred_answers_out,
        debug_out=debug_out,
        debug_file=debug_file,
        fill_missing=not args.no_fill_missing,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
