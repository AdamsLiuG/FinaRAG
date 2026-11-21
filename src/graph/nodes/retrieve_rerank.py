"""Retrieval, reranking, HyDE fallback, and pre-generation adaptive supplement node."""

from __future__ import annotations

import time
from typing import Any, Dict

from src.adaptive_retrieval import adaptive_retrieval_enabled
from src.hyde import HYDE_QUERY_MARKER, should_trigger_hyde


def make_retrieve_rerank_node(processor: Any):
    def retrieve_rerank_node(state: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        question = state["question"]
        schema = state["schema"]
        company_name = state.get("company_name") or ""
        rewrite_result = state["query_plan"]
        route_info = state.get("route_info")
        candidate_doc_ids = list(state.get("candidate_doc_ids") or [])

        retriever, mode = processor._build_retriever()
        candidate_pool_size_before_rerank = None
        reranking_debug = processor._build_reranking_debug_payload(retriever, candidate_pool_size_before_rerank)
        hyde_debug = processor._build_hyde_debug_payload(
            retrieval_results=[],
            candidate_pool_size_before_rerank=candidate_pool_size_before_rerank,
        )

        if mode == "full_context":
            retrieval_results = processor._run_retrieval(
                retriever,
                mode,
                company_name,
                question,
                rewrite_result.filters,
                candidate_doc_ids,
                task_type=getattr(rewrite_result, "task_type", None),
            )
            reranking_debug = processor._build_reranking_debug_payload(retriever, candidate_pool_size_before_rerank)
            hyde_debug = processor._build_hyde_debug_payload(
                retrieval_results=retrieval_results,
                candidate_pool_size_before_rerank=candidate_pool_size_before_rerank,
            )
        elif mode == "hybrid_rerank" and len(rewrite_result.search_queries) > 1:
            retrieval_results, candidate_pool_size_before_rerank, retrieval_runs, _ = processor._run_multi_query_global_rerank(
                retriever=retriever,
                company_name=company_name,
                question=question,
                search_queries=rewrite_result.search_queries,
                filters=rewrite_result.filters,
                candidate_doc_ids=candidate_doc_ids,
                task_type=getattr(rewrite_result, "task_type", None),
            )
            reranking_debug = processor._build_reranking_debug_payload(retriever, candidate_pool_size_before_rerank)
            hyde_debug = processor._build_hyde_debug_payload(
                retrieval_results=retrieval_results,
                candidate_pool_size_before_rerank=candidate_pool_size_before_rerank,
            )
            if processor._should_attempt_hyde(
                mode=mode,
                schema=schema,
                search_queries=rewrite_result.search_queries,
            ):
                hyde_triggered, hyde_reasons = should_trigger_hyde(
                    retrieval_results=retrieval_results,
                    top_score_threshold=processor.hyde_top_score_threshold,
                    margin_threshold=processor.hyde_margin_threshold,
                )
                hyde_debug["triggered"] = hyde_triggered
                hyde_debug["trigger_reasons"] = hyde_reasons
                if hyde_triggered:
                    try:
                        generated_hyde = processor.hyde_generator.generate(
                            question=question,
                            schema=schema,
                            query_plan=rewrite_result,
                            route_info=route_info,
                            model=processor.hyde_generation_model or processor.answering_model,
                            provider=processor.api_provider,
                        )
                        hyde_debug["generated_text"] = generated_hyde
                        if generated_hyde:
                            hyde_candidate_runs = processor._retrieve_multi_query_candidate_runs(
                                retriever=retriever,
                                company_name=company_name,
                                search_queries=[generated_hyde],
                                filters=rewrite_result.filters,
                                candidate_doc_ids=candidate_doc_ids,
                                backend_scope="vector_only",
                            )
                            hyde_candidate_runs = [
                                (HYDE_QUERY_MARKER, results)
                                for _, results in hyde_candidate_runs
                            ]
                            if hyde_candidate_runs:
                                retrieval_results, merged_candidates = processor._rerank_merged_candidate_pool(
                                    retriever=retriever,
                                    question=question,
                                    retrieval_runs=retrieval_runs + hyde_candidate_runs,
                                    pool_cap=processor._candidate_pool_cap(len(rewrite_result.search_queries) + 1),
                                    task_type=getattr(rewrite_result, "task_type", None),
                                )
                                candidate_pool_size_before_rerank = len(merged_candidates)
                                reranking_debug = processor._build_reranking_debug_payload(
                                    retriever, candidate_pool_size_before_rerank
                                )
                                hyde_debug["final_candidate_pool_size"] = candidate_pool_size_before_rerank
                    except Exception as exc:
                        print(f"Warning: HyDE fallback failed: {exc}")
        else:
            retrieval_runs = []
            for search_query in rewrite_result.search_queries:
                results = processor._run_retrieval(
                    retriever,
                    mode,
                    company_name,
                    search_query,
                    rewrite_result.filters,
                    candidate_doc_ids,
                    task_type=getattr(rewrite_result, "task_type", None),
                )
                if results:
                    retrieval_runs.append(results)
            retrieval_results = processor._merge_multi_query_results(retrieval_runs, processor.retrieval_debug_top_n)
            reranking_debug = processor._build_reranking_debug_payload(retriever, candidate_pool_size_before_rerank)
            hyde_debug = processor._build_hyde_debug_payload(
                retrieval_results=retrieval_results,
                candidate_pool_size_before_rerank=candidate_pool_size_before_rerank,
            )

        adaptive_state: Dict[str, Any] = {"rounds_used": 0, "traces": []}
        if adaptive_retrieval_enabled() and retrieval_results and mode != "full_context":
            supplemented = processor._run_adaptive_supplement(
                retriever=retriever,
                company_name=company_name,
                question=question,
                rewrite_result=rewrite_result,
                retrieval_results=retrieval_results,
                candidate_doc_ids=candidate_doc_ids,
                state=adaptive_state,
            )
            if supplemented is not None:
                retrieval_results = supplemented

        return {
            "retrieval_mode": mode,
            "retriever_ref": retriever,
            "retrieval_results": retrieval_results,
            "candidate_pool_size_before_rerank": candidate_pool_size_before_rerank,
            "reranking_debug": reranking_debug,
            "hyde_debug": hyde_debug,
            "adaptive_rounds_used": int(adaptive_state.get("rounds_used") or 0),
            "adaptive_traces": list(adaptive_state.get("traces") or []),
            "stage_events": [
                {
                    "stage": "retrieve_rerank",
                    "mode": mode,
                    "n_results": len(retrieval_results),
                    "pre_adaptive_rounds": int(adaptive_state.get("rounds_used") or 0),
                    "ms": round((time.perf_counter() - started) * 1000, 2),
                }
            ],
        }

    return retrieve_rerank_node
