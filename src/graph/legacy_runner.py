"""Pure-Python non-LangGraph fallback runner (FINARAG_GRAPH_ENGINE=legacy)."""

from __future__ import annotations

import concurrent.futures
from typing import Any, Dict, List, Optional

from src.graph.nodes.comparative_map_reduce import make_comparative_reduce_node
from src.graph.nodes.ground_generate import make_ground_generate_node
from src.graph.nodes.plan_route import make_plan_route_node, make_prepare_single_company_node
from src.graph.nodes.reflect_supplement import (
    decide_reflection_edge,
    make_finalize_single_node,
    make_reflect_supplement_node,
)
from src.graph.nodes.retrieve_rerank import make_retrieve_rerank_node


def _merge_state(state: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(state)
    for key, value in (delta or {}).items():
        if key in {"adaptive_traces", "sub_answers", "stage_events"} and isinstance(value, list):
            merged[key] = list(merged.get(key) or []) + list(value)
        else:
            merged[key] = value
    return merged


def run_legacy_single_company(
    processor: Any,
    *,
    company_name: str,
    question: str,
    schema: str,
    query_plan: Optional[Any] = None,
    route_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "company_name": company_name,
        "question": question,
        "schema": schema,
        "query_plan": query_plan,
        "route_info": route_info,
    }
    state = _merge_state(state, make_prepare_single_company_node(processor)(state))
    if state.get("route_target") == "done":
        return dict(state.get("final_answer_dict") or {})

    return _run_single_rag_loop(processor, state)


def _run_single_rag_loop(processor: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    state = _merge_state(state, make_retrieve_rerank_node(processor)(state))
    ground_fn = make_ground_generate_node(processor)
    reflect_fn = make_reflect_supplement_node(processor)
    finalize_fn = make_finalize_single_node(processor)

    state = _merge_state(state, ground_fn(state))
    while decide_reflection_edge(state) == "reflect":
        state = _merge_state(state, reflect_fn(state))
        state = _merge_state(state, ground_fn(state))
    state = _merge_state(state, finalize_fn(state))
    return dict(state.get("final_answer_dict") or {})


def run_legacy_comparative(
    processor: Any,
    *,
    question: str,
    companies: List[str],
    schema: str,
) -> Dict[str, Any]:
    rephrased = processor.api_processor.get_rephrased_questions(
        original_question=question,
        companies=companies,
        model=processor.answering_model,
    )
    sub_answers: List[tuple[str, Dict[str, Any]]] = []

    def _work(company: str) -> tuple[str, Dict[str, Any]]:
        sub_q = rephrased.get(company)
        if not sub_q:
            raise ValueError(f"Could not generate sub-question for company: {company}")
        return company, processor.get_answer_for_company(
            company_name=company,
            question=sub_q,
            schema=schema,
        )

    max_workers = min(max(1, processor.parallel_requests), max(1, len(companies)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_work, company) for company in companies]
        for fut in concurrent.futures.as_completed(futures):
            sub_answers.append(fut.result())

    reduce_fn = make_comparative_reduce_node(processor)
    state = {
        "question": question,
        "schema": schema,
        "companies": companies,
        "sub_answers": sub_answers,
    }
    res = reduce_fn(state)
    return dict(res.get("final_answer_dict") or {})


def run_legacy_question(
    processor: Any,
    *,
    question: str,
    schema: str,
    question_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "question": question,
        "schema": schema,
        "question_meta": question_meta or {},
    }
    state = _merge_state(state, make_plan_route_node(processor)(state))
    target = state.get("route_target")
    if target == "done":
        return dict(state.get("final_answer_dict") or {})
    if target == "comparative_map":
        return run_legacy_comparative(
            processor,
            question=question,
            companies=list(state.get("companies") or []),
            schema=schema,
        )
    return _run_single_rag_loop(processor, state)
