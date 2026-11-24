"""Retrieval, reranking, HyDE fallback, and pre-generation adaptive supplement node."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict

from langgraph.runtime import Runtime

from src.graph.state import FinaRAGRunContext
from src.adaptive_retrieval import adaptive_retrieval_enabled
from src.hyde import HYDE_QUERY_MARKER, should_trigger_hyde


def make_retrieve_rerank_node(processor: Any):
    def retrieve_rerank_node(state: Dict[str, Any], runtime: Runtime[FinaRAGRunContext] = None) -> Dict[str, Any]:
        context = runtime.context if runtime is not None else None
        started = time.perf_counter()
        question = state["question"]
        schema = state["schema"]
        company_name = state.get("company_name") or ""
        rewrite_result = state["query_plan"]
        route_info = state.get("route_info")
        candidate_doc_ids = list(state.get("candidate_doc_ids") or [])
        budget_ctx = context.budget_context if context else state.get("budget_context")
        if budget_ctx is not None:
            budget_ctx.should_use_fast_path(rewrite_result)

        retriever, mode = processor._build_retriever()
        effective_mode = "hybrid" if (
            mode == "hybrid_rerank"
            and budget_ctx is not None
            and budget_ctx.skip_second_stage_rerank
        ) else mode
        candidate_pool_size_before_rerank = None
        reranking_debug = processor._build_reranking_debug_payload(retriever, candidate_pool_size_before_rerank)
        hyde_debug = processor._build_hyde_debug_payload(
            retrieval_results=[],
            candidate_pool_size_before_rerank=candidate_pool_size_before_rerank,
        )

        if effective_mode == "full_context":
            retrieval_results = processor._run_retrieval(
                retriever,
                effective_mode,
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
        elif effective_mode == "hybrid_rerank" and len(rewrite_result.search_queries) > 1:
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
            if (
                (budget_ctx is None or budget_ctx.allow_hyde)
                and processor._should_attempt_hyde(
                    mode=effective_mode,
                    schema=schema,
                    search_queries=rewrite_result.search_queries,
                )
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
                    effective_mode,
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
            if effective_mode != mode:
                reranking_debug["final_reranking_backend"] = "skipped_budget"
            hyde_debug = processor._build_hyde_debug_payload(
                retrieval_results=retrieval_results,
                candidate_pool_size_before_rerank=candidate_pool_size_before_rerank,
            )

        adaptive_state: Dict[str, Any] = {"rounds_used": 0, "traces": []}
        tree_nav_hits = 0
        kg_hits = 0
        if (
            os.getenv("FINARAG_TREE_NAV_ENABLED", "0").strip().lower() in {"1", "true", "yes"}
            and effective_mode != "full_context"
        ):
            try:
                from src.multimodal_nav.tree_navigator import (
                    TocTreeNavigator,
                    fuse_tree_and_hybrid_results,
                    should_trigger_tree_navigation,
                )
                from src.multimodal_nav.toc_tree_builder import TocTreeStore

                if should_trigger_tree_navigation(rewrite_result, question):
                    toc_db = Path(
                        os.getenv("FINARAG_TOC_TREE_DB")
                        or str(Path(processor.documents_dir).parent / "finarag_toc_trees.sqlite")
                    )
                    if toc_db.exists():
                        selected_doc_id = state.get("selected_doc_id")
                        toc_store = TocTreeStore(toc_db)
                        try:
                            tree = (
                                toc_store.get_tree(str(selected_doc_id))
                                if selected_doc_id
                                else None
                            )
                            if tree is None and company_name:
                                tree = toc_store.get_tree_by_company(company_name)
                        finally:
                            toc_store.close()
                        if tree is not None:
                            doc_payload = processor._load_document_payload(tree.doc_id) if hasattr(processor, "_load_document_payload") else None
                            pages_map = {
                                int(p["page"]): p
                                for p in ((doc_payload or {}).get("content") or {}).get("pages") or []
                                if isinstance(p, dict) and p.get("page") is not None
                            }
                            doc_meta = (doc_payload or {}).get("metainfo") or {
                                "sha1_name": tree.doc_id,
                                "doc_id": tree.doc_id,
                                "company_name": tree.company_name,
                                "report_year": tree.report_year,
                            }
                            nav = TocTreeNavigator()
                            tree_results = nav.retrieve_section_pages(
                                tree,
                                question,
                                lambda pno: pages_map.get(int(pno)),
                                doc_metainfo=doc_meta,
                            )
                            if tree_results:
                                tree_nav_hits = len(tree_results)
                                retrieval_results = fuse_tree_and_hybrid_results(
                                    retrieval_results,
                                    tree_results,
                                    top_n=max(len(retrieval_results), getattr(processor, "retrieval_debug_top_n", 10)),
                                )
            except Exception as exc:
                print(f"Warning: PageIndex tree navigation skipped: {exc}")

        if (
            os.getenv("FINARAG_KG_ENABLED", "0").strip().lower() in {"1", "true", "yes"}
            and effective_mode != "full_context"
        ):
            try:
                from src.knowledge_graph.kg_builder import FinancialKGStore
                from src.knowledge_graph.lightrag_retriever import (
                    LightRAGRetriever,
                    fuse_kg_and_hybrid_results,
                    should_trigger_kg_retrieval,
                )

                if should_trigger_kg_retrieval(rewrite_result, question):
                    kg_db = Path(
                        os.getenv("FINARAG_KG_DB")
                        or str(Path(processor.documents_dir).parent / "finarag_kg.sqlite")
                    )
                    if kg_db.exists():
                        kg_store = FinancialKGStore(kg_db)
                        payload_cache: Dict[str, Dict[int, Dict[str, Any]]] = {}

                        def _fetch_kg_page(did: str, pno: int) -> Optional[Dict[str, Any]]:
                            if did not in payload_cache:
                                payload = (
                                    processor._load_document_payload(did)
                                    if hasattr(processor, "_load_document_payload")
                                    else None
                                )
                                cname = str(((payload or {}).get("metainfo") or {}).get("company_name") or company_name or "")
                                payload_cache[did] = {
                                    int(p["page"]): {**p, "company_name": cname}
                                    for p in ((payload or {}).get("content") or {}).get("pages") or []
                                    if isinstance(p, dict) and p.get("page") is not None
                                }
                            return payload_cache[did].get(int(pno))

                        try:
                            kg_out = LightRAGRetriever(kg_store).retrieve(
                                question,
                                company_name=company_name or None,
                                page_fetcher=_fetch_kg_page,
                            )
                        finally:
                            kg_store.close()
                        kg_results = kg_out.get("results") or []
                        if kg_results:
                            kg_hits = len(kg_results)
                            retrieval_results = fuse_kg_and_hybrid_results(
                                retrieval_results,
                                kg_results,
                                top_n=max(len(retrieval_results), getattr(processor, "retrieval_debug_top_n", 10)),
                            )
            except Exception as exc:
                print(f"Warning: KG LightRAG retrieval skipped: {exc}")

        if (
            adaptive_retrieval_enabled()
            and retrieval_results
            and effective_mode != "full_context"
            and (budget_ctx is None or budget_ctx.allow_adaptive_supplement)
        ):
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
            "retrieval_mode": effective_mode,
            **({"retriever_ref": retriever} if context is None else {}),
            "retrieval_results": retrieval_results,
            "candidate_pool_size_before_rerank": candidate_pool_size_before_rerank,
            "reranking_debug": reranking_debug,
            "hyde_debug": hyde_debug,
            "adaptive_rounds_used": int(adaptive_state.get("rounds_used") or 0),
            "adaptive_traces": list(adaptive_state.get("traces") or []),
            "stage_events": [
                {
                    "stage": "retrieve_rerank",
                    "mode": effective_mode,
                    "n_results": len(retrieval_results),
                    "tree_nav_hits": tree_nav_hits,
                    "kg_hits": kg_hits,
                    "pre_adaptive_rounds": int(adaptive_state.get("rounds_used") or 0),
                    "degraded_level": budget_ctx.degradation_level if budget_ctx is not None else 0,
                    "ms": round((time.perf_counter() - started) * 1000, 2),
                }
            ],
        }

    return retrieve_rerank_node
