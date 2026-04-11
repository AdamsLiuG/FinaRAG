from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from eval.composite_score import (
    compute_citation_support,
    compute_finance_case_score,
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
from eval.metrics import build_debug_index, compare_answers, compare_ranked_retrieval, load_answers_bundle
from eval.ragas_adapter import RagasRuntime, RagasRuntimeConfig, collect_ragas_contexts, prepare_ragas_runtime, score_with_ragas
from eval.semantic_metrics import EmbeddingSimilarityScorer, score_semantic_similarity


def _mean(values: List[float | None]) -> float | None:
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return None
    return round(sum(valid_values) / len(valid_values), 4)


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
    ragas_contexts = collect_ragas_contexts(pred_answer, debug_detail=debug_detail, limit=ragas_context_limit)
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

    case_reports: List[Dict[str, Any]] = []
    unmatched_predictions: List[str] = []
    normalized_pred_answers: List[Dict[str, Any]] = []

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
            continue

        question = question_by_id.get(gold_answer.question_id) or question_by_text.get(gold_answer.question_text)
        detail = debug_index.get(gold_answer.question_text) or {}
        case_reports.append(
            evaluate_finance_case(
                pred_answer,
                gold_answer,
                question,
                debug_detail=detail,
                embedding_scorer=embedding_scorer,
                ragas_runtime=ragas_runtime,
                ragas_unavailable_reason=ragas_unavailable_reason,
                ragas_unavailable_error=ragas_unavailable_error,
                ragas_context_limit=runtime_config.context_limit,
            )
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

    summary = {
        "matched_predictions": len(case_reports),
        "unmatched_predictions": unmatched_predictions,
        "mean_semantic_score": _mean([case["semantic"]["semantic_score"] for case in case_reports]),
        "mean_entity_score": _mean([case["entity"]["entity_score"] for case in case_reports]),
        "mean_answer_score": _mean([case["scores"]["answer_score"] for case in case_reports]),
        "mean_retrieval_score": _mean([case["scores"]["retrieval_score"] for case in case_reports]),
        "mean_citation_score": _mean([case["scores"]["citation_score"] for case in case_reports]),
        "mean_final_quality_score": _mean([case["scores"]["final_quality_score"] for case in case_reports]),
        "ragas_ready_cases": sum(1 for case in case_reports if case["ragas"]["contexts_used"] > 0),
        "ragas_available_cases": sum(1 for case in case_reports if case["ragas"]["available"]),
    }

    report = {
        "dataset": {
            "questions_file": str(questions_file),
            "gold_answers_file": str(gold_answers_file),
            "pred_answers_file": str(pred_answers_file),
            "debug_file": str(debug_file) if debug_file is not None else None,
            "alignment": alignment,
            "prediction_details": pred_payload.get("details"),
        },
        "ragas": {
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
        },
        "summary": summary,
        "aggregate_metrics": aggregate_metrics,
        "ranked_retrieval_metrics": ranked_retrieval_metrics,
    }
    if include_cases:
        report["cases"] = case_reports
    return report


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
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
