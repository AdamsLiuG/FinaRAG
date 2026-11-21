"""LangGraph topology builders and sync/stream execution helpers for FinaRAG."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from langgraph.graph import END, START, StateGraph

from src.graph.nodes.comparative_map_reduce import (
    fanout_companies,
    make_company_worker_node,
    make_comparative_plan_node,
    make_comparative_reduce_node,
)
from src.graph.nodes.ground_generate import make_ground_generate_node
from src.graph.nodes.plan_route import make_plan_route_node, make_prepare_single_company_node
from src.graph.nodes.reflect_supplement import (
    decide_reflection_edge,
    make_finalize_single_node,
    make_reflect_supplement_node,
)
from src.graph.nodes.retrieve_rerank import make_retrieve_rerank_node
from src.graph.state import FinaRAGState


def _route_after_plan(state: Dict[str, Any]) -> str:
    target = state.get("route_target")
    if target == "comparative_map":
        return "comparative_plan"
    if target == "single_rag":
        return "retrieve_rerank"
    return "done"


def _route_after_prepare(state: Dict[str, Any]) -> str:
    if state.get("route_target") == "single_rag":
        return "retrieve_rerank"
    return "done"


def build_finarag_graph(processor: Any, checkpointer: Optional[Any] = None):
    """Build the full FinaRAG routing + RAG/reflection + comparative Map-Reduce graph."""
    builder = StateGraph(FinaRAGState)
    builder.add_node("plan_route", make_plan_route_node(processor))
    builder.add_node("retrieve_rerank", make_retrieve_rerank_node(processor))
    builder.add_node("ground_generate", make_ground_generate_node(processor))
    builder.add_node("reflect_supplement", make_reflect_supplement_node(processor))
    builder.add_node("finalize_single", make_finalize_single_node(processor))
    builder.add_node("comparative_plan", make_comparative_plan_node(processor))
    builder.add_node("company_worker", make_company_worker_node(processor))
    builder.add_node("comparative_reduce", make_comparative_reduce_node(processor))

    builder.add_edge(START, "plan_route")
    builder.add_conditional_edges(
        "plan_route",
        _route_after_plan,
        {
            "done": END,
            "single_rag": "retrieve_rerank",
            "comparative_map": "comparative_plan",
            "retrieve_rerank": "retrieve_rerank",
            "comparative_plan": "comparative_plan",
        },
    )
    builder.add_edge("retrieve_rerank", "ground_generate")
    builder.add_conditional_edges(
        "ground_generate",
        decide_reflection_edge,
        {
            "reflect": "reflect_supplement",
            "finalize": "finalize_single",
        },
    )
    builder.add_edge("reflect_supplement", "ground_generate")
    builder.add_edge("finalize_single", END)

    builder.add_conditional_edges("comparative_plan", fanout_companies, ["company_worker"])
    builder.add_edge("company_worker", "comparative_reduce")
    builder.add_edge("comparative_reduce", END)

    return builder.compile(checkpointer=checkpointer)


def build_single_company_graph(processor: Any, checkpointer: Optional[Any] = None):
    """Build single-company subgraph for direct get_answer_for_company calls."""
    builder = StateGraph(FinaRAGState)
    builder.add_node("prepare_single", make_prepare_single_company_node(processor))
    builder.add_node("retrieve_rerank", make_retrieve_rerank_node(processor))
    builder.add_node("ground_generate", make_ground_generate_node(processor))
    builder.add_node("reflect_supplement", make_reflect_supplement_node(processor))
    builder.add_node("finalize_single", make_finalize_single_node(processor))

    builder.add_edge(START, "prepare_single")
    builder.add_conditional_edges(
        "prepare_single",
        _route_after_prepare,
        {
            "done": END,
            "retrieve_rerank": "retrieve_rerank",
        },
    )
    builder.add_edge("retrieve_rerank", "ground_generate")
    builder.add_conditional_edges(
        "ground_generate",
        decide_reflection_edge,
        {
            "reflect": "reflect_supplement",
            "finalize": "finalize_single",
        },
    )
    builder.add_edge("reflect_supplement", "ground_generate")
    builder.add_edge("finalize_single", END)

    return builder.compile(checkpointer=checkpointer)


def build_comparative_graph(processor: Any, checkpointer: Optional[Any] = None):
    """Build comparative Map-Reduce subgraph for process_comparative_question calls."""
    builder = StateGraph(FinaRAGState)
    builder.add_node("comparative_plan", make_comparative_plan_node(processor))
    builder.add_node("company_worker", make_company_worker_node(processor))
    builder.add_node("comparative_reduce", make_comparative_reduce_node(processor))

    builder.add_edge(START, "comparative_plan")
    builder.add_conditional_edges("comparative_plan", fanout_companies, ["company_worker"])
    builder.add_edge("company_worker", "comparative_reduce")
    builder.add_edge("comparative_reduce", END)

    return builder.compile(checkpointer=checkpointer)


def run_finarag_graph(
    processor: Any,
    question: str,
    schema: str,
    question_meta: Optional[Dict[str, Any]] = None,
    *,
    config: Optional[Dict[str, Any]] = None,
    budget_context: Any = None,
) -> Dict[str, Any]:
    graph = build_finarag_graph(processor)
    initial: FinaRAGState = {
        "question": question,
        "schema": schema,
        "question_meta": question_meta or {},
    }
    if budget_context is not None:
        initial["budget_context"] = budget_context
    final_state = graph.invoke(initial, config=config)
    answer = dict(final_state.get("final_answer_dict") or {})
    if final_state.get("stage_events"):
        answer.setdefault("graph_stage_events", list(final_state["stage_events"]))
    if budget_context is not None and "budget_summary" not in answer:
        answer["budget_summary"] = budget_context.summary()
        answer["degraded_level"] = budget_context.degradation_level
    return answer


def stream_finarag_events(
    processor: Any,
    question: str,
    schema: str,
    question_meta: Optional[Dict[str, Any]] = None,
    *,
    config: Optional[Dict[str, Any]] = None,
    budget_context: Any = None,
) -> Iterator[Dict[str, Any]]:
    """Yield incremental stage events as each LangGraph node completes."""
    graph = build_finarag_graph(processor)
    initial: FinaRAGState = {
        "question": question,
        "schema": schema,
        "question_meta": question_meta or {},
    }
    if budget_context is not None:
        initial["budget_context"] = budget_context
    last_answer: Optional[Dict[str, Any]] = None
    for update in graph.stream(initial, config=config, stream_mode="updates"):
        for node_name, node_payload in update.items():
            if not isinstance(node_payload, dict):
                continue
            if node_payload.get("final_answer_dict"):
                last_answer = node_payload["final_answer_dict"]
            for event in node_payload.get("stage_events") or []:
                yield {"node": node_name, **event}
    if last_answer is not None:
        yield {
            "node": "end",
            "stage": "result",
            "final_answer": last_answer.get("final_answer"),
            "confidence": last_answer.get("confidence"),
            "validation_flags": last_answer.get("validation_flags") or [],
            "answer_dict": last_answer,
        }
