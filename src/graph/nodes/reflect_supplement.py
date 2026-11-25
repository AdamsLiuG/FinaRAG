"""Post-validate adaptive reflection and convergence control nodes."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

from langgraph.runtime import Runtime

from src.graph.state import FinaRAGRunContext
from src.harness.trajectory_store import capture_active, capture_step
from src.adaptive_retrieval import (
    adaptive_max_rounds,
    adaptive_retrieval_enabled,
    detect_post_validate_gap,
    merge_supplement_results,
)


def decide_reflection_edge(state: Dict[str, Any], runtime: Runtime[FinaRAGRunContext] = None, *, replay_config=None) -> str:
    """Conditional edge after ground_generate: decide whether to supplement & re-run or finalize."""
    context = runtime.context if runtime is not None else None
    budget_ctx = context.budget_context if context else state.get("budget_context")
    if budget_ctx is not None and not budget_ctx.allow_adaptive_supplement:
        return "finalize"
    if not (adaptive_retrieval_enabled() if replay_config is None else replay_config.get("adaptive", False)):
        return "finalize"
    if state.get("retrieval_mode") == "full_context":
        return "finalize"
    if int(state.get("adaptive_rounds_used") or 0) >= adaptive_max_rounds():
        return "finalize"

    flags = state.get("validated_flags") or []
    gap = detect_post_validate_gap(
        flags,
        state["query_plan"],
        state.get("retrieval_results") or [],
    )
    if gap is None:
        return "finalize"

    query_sig = tuple(gap.supplement_queries)
    seen = list(state.get("adaptive_seen_queries") or [])
    if query_sig in seen:
        return "finalize"
    return "reflect"


def make_reflect_supplement_node(processor: Any):
    def reflect_supplement_node(state: Dict[str, Any], runtime: Runtime[FinaRAGRunContext] = None) -> Dict[str, Any]:
        context = runtime.context if runtime is not None else None
        started = time.perf_counter()
        question = state["question"]
        company_name = state.get("company_name") or ""
        rewrite_result = state["query_plan"]
        candidate_doc_ids = list(state.get("candidate_doc_ids") or [])
        current_results = list(state.get("retrieval_results") or [])
        flags = list(state.get("validated_flags") or [])

        gap = detect_post_validate_gap(flags, rewrite_result, current_results)
        if gap is None:
            return {}

        retriever = processor._build_retriever()[0] if context else state["retriever_ref"]
        query_sig: Tuple[str, ...] = tuple(gap.supplement_queries)
        seen = list(state.get("adaptive_seen_queries") or [])
        seen.append(query_sig)

        parent_override = "page" if getattr(gap, "strategy", None) == "parent_recall" else None
        if parent_override is not None:
            supplement = processor._collect_supplement(
                retriever=retriever,
                company_name=company_name,
                rewrite_result=rewrite_result,
                gap=gap,
                candidate_doc_ids=candidate_doc_ids,
                parent_mode_override=parent_override,
            )
        else:
            supplement = processor._collect_supplement(
                retriever=retriever,
                company_name=company_name,
                rewrite_result=rewrite_result,
                gap=gap,
                candidate_doc_ids=candidate_doc_ids,
            )

        observing = capture_active()
        if observing:
            retrieval_key = capture_step(
                "retrieval", operation="supplement", step_key="retrieval",
                inputs={"flags": flags, "candidate_doc_ids": candidate_doc_ids,
                        "search_queries": list(gap.supplement_queries), "trigger": gap.trigger},
                outputs={"results": supplement},
                costs={"elapsed_ms": (time.perf_counter() - started) * 1000},
            )

        rounds_used = int(state.get("adaptive_rounds_used") or 0) + 1
        if not supplement:
            return {
                "adaptive_rounds_used": rounds_used,
                "adaptive_seen_queries": seen,
                "stage_events": [
                    {
                        "stage": "reflect_supplement",
                        "trigger": gap.trigger,
                        "new_hits": 0,
                        "ms": round((time.perf_counter() - started) * 1000, 2),
                    }
                ],
            }

        merged = merge_supplement_results(current_results, supplement)
        rerank_started = time.perf_counter() if observing else None
        reranked = retriever.rerank_candidates(
            query=question,
            candidate_results=merged,
            documents_batch_size=processor.final_reranking_batch_size,
            top_n=processor.retrieval_debug_top_n,
            task_type=getattr(rewrite_result, "task_type", None),
        )
        if observing:
            capture_step(
                "rerank", operation="supplement_rank", step_key="rerank", inputs={"results": merged},
                outputs={"results": reranked}, parents=(retrieval_key,),
                costs={"elapsed_ms": (time.perf_counter() - rerank_started) * 1000},
            )
        trace_entry = {
            "question": question,
            "company_name": company_name,
            "trigger": gap.trigger,
            "strategy": getattr(gap, "strategy", "dense_recall"),
            "queries": list(gap.supplement_queries),
            "new_hits": len(supplement),
            "rounds": rounds_used,
            "atom_id": (getattr(rewrite_result, "metric_atom", None) or {}).get("atom_id"),
        }
        return {
            "retrieval_results": reranked,
            "adaptive_rounds_used": rounds_used,
            "adaptive_seen_queries": seen,
            "adaptive_traces": [trace_entry],
            "stage_events": [
                {
                    "stage": "reflect_supplement",
                    "trigger": gap.trigger,
                    "new_hits": len(supplement),
                    "rounds": rounds_used,
                    "ms": round((time.perf_counter() - started) * 1000, 2),
                }
            ],
        }

    return reflect_supplement_node


def make_finalize_single_node(processor: Any):
    def finalize_single_node(state: Dict[str, Any]) -> Dict[str, Any]:
        traces = list(state.get("adaptive_traces") or [])
        if traces:
            validated_flags = set(state.get("validated_flags") or [])
            final_traces: List[Dict[str, Any]] = []
            for idx, item in enumerate(traces):
                enriched = dict(item)
                if idx == len(traces) - 1 and "outcome" not in enriched:
                    enriched["outcome"] = (
                        "supplement_succeeded"
                        if "metric_caliber_conflict_refused" not in validated_flags
                        else "refused_maintained"
                    )
                    enriched["gap_closed"] = (
                        detect_post_validate_gap(
                            list(validated_flags),
                            state["query_plan"],
                            state.get("retrieval_results") or [],
                        )
                        is None
                    )
                final_traces.append(enriched)
            processor._record_adaptive_trace({"traces": final_traces})
        return {
            "stage_events": [
                {
                    "stage": "done",
                    "confidence": (state.get("final_answer_dict") or {}).get("confidence"),
                }
            ]
        }

    return finalize_single_node
