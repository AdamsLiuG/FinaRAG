from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from eval.composite_score import (
    compute_citation_support,
    compute_finance_case_score,
    get_finance_scoring_profile,
    compute_retrieval_support,
)
from eval.dataset_schema import (
    FinanceEvalQuestion,
    FinanceGoldAnswer,
    load_gold_answer_set,
    load_question_set,
    validate_dataset_alignment,
)
from eval.entity_metrics import score_finance_entities
from eval.keyword_metrics import score_answer_keywords
from eval.metrics import build_debug_index, compare_answers, compare_ranked_retrieval, load_answers_bundle
from eval.ragas_adapter import RagasRuntime, RagasRuntimeConfig, collect_ragas_contexts, prepare_ragas_runtime, score_with_ragas
from eval.semantic_metrics import EmbeddingSimilarityScorer, score_semantic_similarity


NON_RETRYABLE_RAGAS_REASONS = {
    "no_contexts",
    "empty_question",
    "empty_answer",
    "empty_reference",
}


def _mean(values: List[float | None]) -> float | None:
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return None
    return round(sum(valid_values) / len(valid_values), 4)


def _rate_at_threshold(values: List[float | None], threshold: float) -> float | None:
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return None
    hit_count = sum(1 for value in valid_values if value >= threshold)
    return round(hit_count / len(valid_values), 4)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _normalize_prediction_answer(pred_answer: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(pred_answer)
    if "value" not in normalized and "final_answer" in normalized:
        normalized["value"] = normalized.get("final_answer")
    return normalized


def _build_question_lookup(questions: List[FinanceEvalQuestion]) -> tuple[Dict[str, FinanceEvalQuestion], Dict[str, FinanceEvalQuestion]]:
    by_id = {question.question_id: question for question in questions}
    by_text = {question.question_text: question for question in questions}
    return by_id, by_text


def _build_answer_lookup(answers: List[FinanceGoldAnswer]) -> tuple[Dict[str, FinanceGoldAnswer], Dict[str, FinanceGoldAnswer]]:
    by_id = {answer.question_id: answer for answer in answers}
    by_text = {answer.question_text: answer for answer in answers}
    return by_id, by_text


def _normalize_ragas_result(ragas_result: Dict[str, Any]) -> Dict[str, Any]:
    normalized = deepcopy(ragas_result)
    normalized.setdefault("available", False)
    normalized.setdefault("reason", None)
    normalized.setdefault("error", None)
    normalized.setdefault("errors", [])
    normalized.setdefault("contexts_used", 0)
    normalized.setdefault("answer_correctness", None)
    normalized.setdefault("faithfulness", None)
    normalized.setdefault("answer_relevancy", None)
    normalized.setdefault("context_recall", None)
    normalized.setdefault("context_precision", None)
    normalized.setdefault("ragas_score", None)
    return normalized


def _is_reusable_ragas_result(ragas_result: Dict[str, Any]) -> bool:
    if ragas_result.get("available") is True:
        return True
    return ragas_result.get("reason") in NON_RETRYABLE_RAGAS_REASONS


def _load_ragas_resume_cases(
    ragas_resume_report: Path | None,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    resume_state = {
        "enabled": ragas_resume_report is not None,
        "source_report": str(ragas_resume_report) if ragas_resume_report is not None else None,
        "source_exists": False,
        "source_cases": 0,
        "reusable_source_cases": 0,
        "reused_cases": 0,
        "retried_cases": 0,
        "missing_cases": 0,
    }
    if ragas_resume_report is None:
        return {}, {}, resume_state
    if not ragas_resume_report.exists():
        resume_state["reason"] = "source_report_not_found"
        return {}, {}, resume_state

    with open(ragas_resume_report, "r", encoding="utf-8") as file:
        payload = json.load(file)

    by_id: Dict[str, Dict[str, Any]] = {}
    by_text: Dict[str, Dict[str, Any]] = {}
    cases = payload.get("cases") or []
    resume_state["source_exists"] = True
    resume_state["source_cases"] = len(cases)

    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("ragas"), dict):
            continue
        ragas_result = _normalize_ragas_result(case["ragas"])
        if not _is_reusable_ragas_result(ragas_result):
            continue
        resume_state["reusable_source_cases"] += 1
        question_id = case.get("question_id")
        question_text = case.get("question_text")
        if question_id:
            by_id[str(question_id)] = ragas_result
        if question_text:
            by_text[str(question_text)] = ragas_result

    return by_id, by_text, resume_state


def _lookup_cached_ragas_result(
    *,
    question_id: str | None,
    question_text: str | None,
    resume_by_id: Dict[str, Dict[str, Any]],
    resume_by_text: Dict[str, Dict[str, Any]],
) -> Dict[str, Any] | None:
    if question_id and question_id in resume_by_id:
        return _normalize_ragas_result(resume_by_id[question_id])
    if question_text and question_text in resume_by_text:
        return _normalize_ragas_result(resume_by_text[question_text])
    return None


def _build_summary(case_reports: List[Dict[str, Any]], unmatched_predictions: List[str]) -> Dict[str, Any]:
    return {
        "matched_predictions": len(case_reports),
        "unmatched_predictions": unmatched_predictions,
        "mean_semantic_score": _mean([case["semantic"]["semantic_score"] for case in case_reports]),
        "mean_entity_score": _mean([case["entity"]["entity_score"] for case in case_reports]),
        "mean_keyword_score": _mean([case["keyword"]["keyword_score"] for case in case_reports]),
        "mean_type_aware_value_score": _mean([case["scores"]["type_aware_value_score"] for case in case_reports]),
        "mean_ragas_score": _mean([case["ragas"]["ragas_score"] for case in case_reports]),
        "mean_ragas_answer_correctness": _mean([case["ragas"]["answer_correctness"] for case in case_reports]),
        "mean_ragas_faithfulness": _mean([case["ragas"]["faithfulness"] for case in case_reports]),
        "mean_ragas_answer_relevancy": _mean([case["ragas"]["answer_relevancy"] for case in case_reports]),
        "mean_ragas_context_recall": _mean([case["ragas"]["context_recall"] for case in case_reports]),
        "mean_ragas_context_precision": _mean([case["ragas"]["context_precision"] for case in case_reports]),
        "mean_answer_score": _mean([case["scores"]["answer_score"] for case in case_reports]),
        "mean_retrieval_score": _mean([case["scores"]["retrieval_score"] for case in case_reports]),
        "mean_retrieval_rule_score": _mean([case["scores"]["retrieval_rule_score"] for case in case_reports]),
        "mean_citation_score": _mean([case["scores"]["citation_score"] for case in case_reports]),
        "mean_citation_rule_score": _mean([case["scores"]["citation_rule_score"] for case in case_reports]),
        "mean_final_quality_score": _mean([case["scores"]["final_quality_score"] for case in case_reports]),
        "final_quality_pass_rate_at_0_8": _rate_at_threshold(
            [case["scores"]["final_quality_score"] for case in case_reports],
            threshold=0.8,
        ),
        "answer_pass_rate_at_0_8": _rate_at_threshold(
            [case["scores"]["answer_score"] for case in case_reports],
            threshold=0.8,
        ),
        "ragas_ready_cases": sum(1 for case in case_reports if case["ragas"]["contexts_used"] > 0),
        "ragas_available_cases": sum(1 for case in case_reports if case["ragas"]["available"]),
    }


def _build_report_payload(
    *,
    questions_file: Path,
    gold_answers_file: Path,
    pred_answers_file: Path,
    debug_file: Path | None,
    alignment: Dict[str, Any],
    prediction_details: Any,
    runtime_config: RagasRuntimeConfig,
    ragas_runtime: RagasRuntime | None,
    ragas_unavailable_reason: str | None,
    ragas_unavailable_error: str | None,
    resume_state: Dict[str, Any],
    summary: Dict[str, Any],
    aggregate_metrics: Dict[str, Any] | None,
    ranked_retrieval_metrics: Dict[str, Any] | None,
    checkpoint_state: Dict[str, Any] | None = None,
    case_reports: List[Dict[str, Any]] | None = None,
    include_cases: bool = True,
) -> Dict[str, Any]:
    ragas_payload = {
        "enabled": runtime_config.enabled,
        "llm_provider": runtime_config.llm_provider,
        "llm_model": runtime_config.llm_model,
        "llm_adapter": runtime_config.llm_adapter,
        "llm_force_stream": runtime_config.llm_force_stream,
        "embedding_provider": runtime_config.embedding_provider,
        "embedding_model": runtime_config.embedding_model,
        "context_limit": runtime_config.context_limit,
        "runtime_ready": ragas_runtime is not None,
        "runtime_reason": ragas_unavailable_reason or "ok",
        "runtime_error": ragas_unavailable_error,
        "resume": resume_state,
    }
    if checkpoint_state is not None:
        ragas_payload["checkpoint"] = checkpoint_state

    report = {
        "dataset": {
            "questions_file": str(questions_file),
            "gold_answers_file": str(gold_answers_file),
            "pred_answers_file": str(pred_answers_file),
            "debug_file": str(debug_file) if debug_file is not None else None,
            "alignment": alignment,
            "prediction_details": prediction_details,
        },
        "ragas": ragas_payload,
        "scoring_profile": get_finance_scoring_profile(),
        "summary": summary,
        "aggregate_metrics": aggregate_metrics,
        "ranked_retrieval_metrics": ranked_retrieval_metrics,
    }
    if include_cases:
        report["cases"] = case_reports or []
    return report


def _emit_progress_event(progress_log: Path | None, payload: Dict[str, Any]) -> None:
    event_payload = dict(payload)
    event_payload.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    if progress_log is not None:
        _append_jsonl(progress_log, event_payload)
    print(json.dumps(event_payload, ensure_ascii=False), file=sys.stderr, flush=True)


def evaluate_finance_case(
    pred_answer: Dict[str, Any],
    gold_answer: FinanceGoldAnswer,
    question: FinanceEvalQuestion | None,
    *,
    debug_detail: Dict[str, Any] | None = None,
    embedding_scorer: EmbeddingSimilarityScorer | None = None,
    ragas_runtime: RagasRuntime | None = None,
    ragas_unavailable_reason: str | None = None,
    ragas_unavailable_error: str | None = None,
    ragas_context_limit: int = 5,
    cached_ragas_result: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    pred_answer = _normalize_prediction_answer(pred_answer)
    semantic_result = score_semantic_similarity(
        pred_answer.get("value"),
        gold_answer.value,
        kind=gold_answer.kind,
        embedding_scorer=embedding_scorer,
    )
    entity_result = score_finance_entities(
        pred_answer,
        gold_answer,
        question,
        debug_detail=debug_detail,
    )
    keyword_result = score_answer_keywords(
        pred_answer,
        gold_answer,
        question,
        debug_detail=debug_detail,
    )
    ragas_contexts = collect_ragas_contexts(pred_answer, debug_detail=debug_detail, limit=ragas_context_limit)
    if cached_ragas_result is not None:
        ragas_result = _normalize_ragas_result(cached_ragas_result)
    else:
        ragas_result = score_with_ragas(
            question_text=gold_answer.question_text,
            answer=pred_answer.get("value"),
            reference=gold_answer.value,
            contexts=ragas_contexts,
            runtime=ragas_runtime,
            unavailable_reason=ragas_unavailable_reason,
            unavailable_error=ragas_unavailable_error,
        )
    retrieval_result = compute_retrieval_support(pred_answer, gold_answer, debug_detail=debug_detail)
    citation_result = compute_citation_support(pred_answer, gold_answer)
    composite_result = compute_finance_case_score(
        semantic_result=semantic_result,
        entity_result=entity_result,
        keyword_result=keyword_result,
        ragas_result=ragas_result,
        retrieval_result=retrieval_result,
        citation_result=citation_result,
    )
    return {
        "question_id": gold_answer.question_id,
        "question_text": gold_answer.question_text,
        "kind": gold_answer.kind,
        "prediction_value": pred_answer.get("value"),
        "reference_value": gold_answer.value,
        "semantic": semantic_result,
        "entity": entity_result,
        "keyword": keyword_result,
        "ragas": ragas_result,
        "retrieval": retrieval_result,
        "citation": citation_result,
        "scores": composite_result,
    }


def evaluate_finance_answers(
    *,
    questions_file: Path,
    gold_answers_file: Path,
    pred_answers_file: Path,
    debug_file: Path | None = None,
    use_embedding_similarity: bool = False,
    include_cases: bool = True,
    ragas_config: RagasRuntimeConfig | None = None,
    ragas_resume_report: Path | None = None,
    ragas_progress_log: Path | None = None,
    ragas_checkpoint_report: Path | None = None,
    ragas_checkpoint_interval: int = 1,
) -> Dict[str, Any]:
    question_set = load_question_set(questions_file)
    gold_answer_set = load_gold_answer_set(gold_answers_file)
    alignment = validate_dataset_alignment(question_set, gold_answer_set)

    pred_answers, pred_payload = load_answers_bundle(pred_answers_file)
    debug_payload = None
    if debug_file is not None and debug_file.exists():
        _, debug_payload = load_answers_bundle(debug_file)
    debug_index = build_debug_index(debug_payload)

    question_by_id, question_by_text = _build_question_lookup(question_set.questions)
    answer_by_id, answer_by_text = _build_answer_lookup(gold_answer_set.answers)
    embedding_scorer = EmbeddingSimilarityScorer() if use_embedding_similarity else None
    runtime_config = ragas_config or RagasRuntimeConfig.from_env()
    ragas_runtime, ragas_unavailable_reason, ragas_unavailable_error = prepare_ragas_runtime(runtime_config)
    resume_by_id, resume_by_text, resume_state = _load_ragas_resume_cases(ragas_resume_report)
    progress_enabled = ragas_progress_log is not None or ragas_checkpoint_report is not None
    checkpoint_interval = max(ragas_checkpoint_interval, 1)
    start_time = time.monotonic()
    if ragas_progress_log is not None:
        ragas_progress_log.parent.mkdir(parents=True, exist_ok=True)
        ragas_progress_log.write_text("", encoding="utf-8")

    case_reports: List[Dict[str, Any]] = []
    unmatched_predictions: List[str] = []
    normalized_pred_answers: List[Dict[str, Any]] = []

    def write_checkpoint(*, complete: bool) -> None:
        if ragas_checkpoint_report is None:
            return
        checkpoint_summary = _build_summary(case_reports, unmatched_predictions)
        checkpoint_state = {
            "complete": complete,
            "processed_cases": len(case_reports),
            "total_predictions": len(pred_answers),
            "elapsed_seconds": round(time.monotonic() - start_time, 3),
        }
        checkpoint_payload = _build_report_payload(
            questions_file=questions_file,
            gold_answers_file=gold_answers_file,
            pred_answers_file=pred_answers_file,
            debug_file=debug_file,
            alignment=alignment,
            prediction_details=pred_payload.get("details"),
            runtime_config=runtime_config,
            ragas_runtime=ragas_runtime,
            ragas_unavailable_reason=ragas_unavailable_reason,
            ragas_unavailable_error=ragas_unavailable_error,
            resume_state=resume_state,
            summary=checkpoint_summary,
            aggregate_metrics=None,
            ranked_retrieval_metrics=None,
            checkpoint_state=checkpoint_state,
            case_reports=case_reports,
            include_cases=True,
        )
        _write_json(ragas_checkpoint_report, checkpoint_payload)

    if progress_enabled:
        _emit_progress_event(
            ragas_progress_log,
            {
                "event": "started",
                "total_predictions": len(pred_answers),
                "resume_enabled": ragas_resume_report is not None,
                "checkpoint_report": str(ragas_checkpoint_report) if ragas_checkpoint_report is not None else None,
            },
        )

    for pred_answer in pred_answers:
        pred_answer = _normalize_prediction_answer(pred_answer)
        normalized_pred_answers.append(pred_answer)

        question_id = pred_answer.get("question_id")
        question_text = pred_answer.get("question_text")
        gold_answer = answer_by_id.get(question_id) if question_id else None
        if gold_answer is None and question_text:
            gold_answer = answer_by_text.get(question_text)
        if gold_answer is None:
            unmatched_predictions.append(question_text or question_id or "<unknown>")
            if progress_enabled:
                _emit_progress_event(
                    ragas_progress_log,
                    {
                        "event": "unmatched_prediction",
                        "prediction_index": len(normalized_pred_answers),
                        "matched_cases": len(case_reports),
                        "question_id": question_id,
                        "question_text": question_text,
                    },
                )
            continue

        question = question_by_id.get(gold_answer.question_id) or question_by_text.get(gold_answer.question_text)
        detail = debug_index.get(gold_answer.question_text) or {}
        cached_ragas_result = _lookup_cached_ragas_result(
            question_id=gold_answer.question_id,
            question_text=gold_answer.question_text,
            resume_by_id=resume_by_id,
            resume_by_text=resume_by_text,
        )
        if ragas_resume_report is not None:
            if cached_ragas_result is None:
                resume_state["retried_cases"] += 1
                ragas_source = "scored"
            else:
                resume_state["reused_cases"] += 1
                ragas_source = "resume"
        else:
            ragas_source = "scored"
        case_report = evaluate_finance_case(
            pred_answer,
            gold_answer,
            question,
            debug_detail=detail,
            embedding_scorer=embedding_scorer,
            ragas_runtime=ragas_runtime,
            ragas_unavailable_reason=ragas_unavailable_reason,
            ragas_unavailable_error=ragas_unavailable_error,
            ragas_context_limit=runtime_config.context_limit,
            cached_ragas_result=cached_ragas_result,
        )
        case_reports.append(case_report)
        if progress_enabled:
            _emit_progress_event(
                ragas_progress_log,
                {
                    "event": "case_completed",
                    "prediction_index": len(normalized_pred_answers),
                    "matched_cases": len(case_reports),
                    "total_predictions": len(pred_answers),
                    "question_id": gold_answer.question_id,
                    "question_text": gold_answer.question_text,
                    "ragas_source": ragas_source,
                    "ragas_available": case_report["ragas"]["available"],
                    "ragas_reason": case_report["ragas"]["reason"],
                    "ragas_score": case_report["ragas"]["ragas_score"],
                    "elapsed_seconds": round(time.monotonic() - start_time, 3),
                },
            )
        if ragas_checkpoint_report is not None and len(case_reports) % checkpoint_interval == 0:
            write_checkpoint(complete=False)

    if ragas_resume_report is not None:
        resume_state["missing_cases"] = max(
            len(case_reports) - resume_state["reused_cases"] - resume_state["retried_cases"],
            0,
        )
    write_checkpoint(complete=True)
    if progress_enabled:
        _emit_progress_event(
            ragas_progress_log,
            {
                "event": "finished",
                "matched_cases": len(case_reports),
                "unmatched_predictions": len(unmatched_predictions),
                "elapsed_seconds": round(time.monotonic() - start_time, 3),
            },
        )

    reference_payload = {"answers": [answer.model_dump() for answer in gold_answer_set.answers]}
    aggregate_metrics = compare_answers(
        normalized_pred_answers,
        reference_payload["answers"],
        debug_payload=debug_payload,
    )
    ranked_retrieval_metrics = None
    if debug_payload is not None:
        ranked_retrieval_metrics = compare_ranked_retrieval(
            normalized_pred_answers,
            reference_payload["answers"],
            debug_payload=debug_payload,
        )

    summary = _build_summary(case_reports, unmatched_predictions)
    return _build_report_payload(
        questions_file=questions_file,
        gold_answers_file=gold_answers_file,
        pred_answers_file=pred_answers_file,
        debug_file=debug_file,
        alignment=alignment,
        prediction_details=pred_payload.get("details"),
        runtime_config=runtime_config,
        ragas_runtime=ragas_runtime,
        ragas_unavailable_reason=ragas_unavailable_reason,
        ragas_unavailable_error=ragas_unavailable_error,
        resume_state=resume_state,
        summary=summary,
        aggregate_metrics=aggregate_metrics,
        ranked_retrieval_metrics=ranked_retrieval_metrics,
        case_reports=case_reports,
        include_cases=include_cases,
    )


def main():
    parser = argparse.ArgumentParser(description="Run finance-oriented evaluation for FinaRAG answer bundles.")
    parser.add_argument("--questions-file", type=Path, required=True)
    parser.add_argument("--gold-answers-file", type=Path, required=True)
    parser.add_argument("--pred-answers-file", type=Path, required=True)
    parser.add_argument("--debug-file", type=Path, default=None)
    parser.add_argument("--use-embedding-similarity", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-case-details", action="store_true")
    parser.add_argument("--disable-ragas", action="store_true")
    parser.add_argument("--ragas-llm-provider", default=None)
    parser.add_argument("--ragas-llm-model", default=None)
    parser.add_argument("--ragas-llm-base-url", default=None)
    parser.add_argument("--ragas-llm-api-key", default=None)
    parser.add_argument("--ragas-llm-timeout", type=float, default=None)
    parser.add_argument("--ragas-llm-max-retries", type=int, default=None)
    parser.add_argument("--ragas-llm-adapter", default=None)
    parser.add_argument("--ragas-llm-force-stream", action="store_true")
    parser.add_argument("--ragas-embedding-provider", default=None)
    parser.add_argument("--ragas-embedding-model", default=None)
    parser.add_argument("--ragas-embedding-device", default=None)
    parser.add_argument("--ragas-embedding-base-url", default=None)
    parser.add_argument("--ragas-embedding-api-key", default=None)
    parser.add_argument("--ragas-context-limit", type=int, default=None)
    parser.add_argument(
        "--ragas-progress-log",
        type=Path,
        default=None,
        help="Append per-case JSONL progress events while RAGAS scoring runs.",
    )
    parser.add_argument(
        "--ragas-checkpoint-report",
        type=Path,
        default=None,
        help="Write a partial finance_eval report after completed RAGAS cases; usable with --ragas-resume-report.",
    )
    parser.add_argument(
        "--ragas-checkpoint-interval",
        type=int,
        default=1,
        help="Write the checkpoint report every N completed matched cases.",
    )
    parser.add_argument(
        "--ragas-resume-report",
        type=Path,
        default=None,
        help="Existing finance_eval report to reuse successful RAGAS case results from.",
    )
    parser.add_argument(
        "--resume-ragas",
        action="store_true",
        help="Reuse successful RAGAS case results from --ragas-resume-report, or from --output when no resume report is given.",
    )
    args = parser.parse_args()

    env_ragas_config = RagasRuntimeConfig.from_env()
    ragas_config = RagasRuntimeConfig(
        enabled=False if args.disable_ragas else env_ragas_config.enabled,
        llm_provider=args.ragas_llm_provider or env_ragas_config.llm_provider,
        llm_model=args.ragas_llm_model or env_ragas_config.llm_model,
        llm_base_url=args.ragas_llm_base_url or env_ragas_config.llm_base_url,
        llm_api_key=args.ragas_llm_api_key or env_ragas_config.llm_api_key,
        llm_timeout=args.ragas_llm_timeout if args.ragas_llm_timeout is not None else env_ragas_config.llm_timeout,
        llm_max_retries=args.ragas_llm_max_retries if args.ragas_llm_max_retries is not None else env_ragas_config.llm_max_retries,
        llm_adapter=args.ragas_llm_adapter or env_ragas_config.llm_adapter,
        llm_force_stream=True if args.ragas_llm_force_stream else env_ragas_config.llm_force_stream,
        embedding_provider=args.ragas_embedding_provider or env_ragas_config.embedding_provider,
        embedding_model=args.ragas_embedding_model or env_ragas_config.embedding_model,
        embedding_device=args.ragas_embedding_device or env_ragas_config.embedding_device,
        embedding_base_url=args.ragas_embedding_base_url or env_ragas_config.embedding_base_url,
        embedding_api_key=args.ragas_embedding_api_key or env_ragas_config.embedding_api_key,
        context_limit=args.ragas_context_limit if args.ragas_context_limit is not None else env_ragas_config.context_limit,
    )

    report = evaluate_finance_answers(
        questions_file=args.questions_file,
        gold_answers_file=args.gold_answers_file,
        pred_answers_file=args.pred_answers_file,
        debug_file=args.debug_file,
        use_embedding_similarity=args.use_embedding_similarity,
        include_cases=not args.no_case_details,
        ragas_config=ragas_config,
        ragas_resume_report=args.ragas_resume_report or (args.output if args.resume_ragas else None),
        ragas_progress_log=args.ragas_progress_log,
        ragas_checkpoint_report=args.ragas_checkpoint_report,
        ragas_checkpoint_interval=args.ragas_checkpoint_interval,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
