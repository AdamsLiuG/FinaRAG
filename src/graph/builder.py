"""LangGraph topology builders and sync/stream execution helpers for FinaRAG."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterator, List, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, Overwrite, interrupt

from src.graph.nodes.comparative_map_reduce import (
    fanout_companies,
    make_company_worker_node,
    make_comparative_plan_node,
    make_comparative_reduce_node,
)
from src.graph.nodes.ground_generate import approval_reasons, make_ground_generate_node
from src.graph.nodes.plan_route import make_plan_route_node, make_prepare_single_company_node
from src.graph.nodes.reflect_supplement import (
    decide_reflection_edge,
    make_finalize_single_node,
    make_reflect_supplement_node,
)
from src.graph.nodes.retrieve_rerank import make_retrieve_rerank_node
from src.graph.state import FinaRAGRunContext, FinaRAGState


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


def prepare_durable_turn(state: FinaRAGState) -> dict:
    """Reset the outputs for a new input. Continuations never cross START."""
    return {
        "route_decision": {}, "route_target": "", "company_name": "", "companies": [],
        "query_plan": None, "route_info": {}, "route_mode": "",
        "candidate_doc_ids": [], "allowed_grounding_doc_ids": [], "selected_doc_id": None,
        "retrieval_mode": "", "retrieval_results": [], "candidate_pool_size_before_rerank": None,
        "reranking_debug": {}, "hyde_debug": {}, "adaptive_rounds_used": 0,
        "adaptive_seen_queries": [], "adaptive_traces": Overwrite([]), "pending_gap": None,
        "rephrased_questions": {}, "sub_answers": Overwrite([]), "validated_flags": [],
        "final_answer_dict": {}, "stage_events": Overwrite([]), "approval_reasons": [],
        "approval_status": "not_required", "reviewed_candidate": None, "approval_decision": None,
    }


def _refusal(flag: str) -> dict:
    return {
        "final_answer": "N/A", "confidence": "low", "validation_flags": [flag],
        "references": [], "citations": [], "relevant_pages": [],
    }


def prepare_approval(state: FinaRAGState, runtime: Runtime[FinaRAGRunContext]) -> dict:
    candidate = state.get("final_answer_dict") or {}
    reasons = approval_reasons(candidate, schema=state["schema"])
    if (reasons and runtime.context and runtime.context.approval_enabled and
            "insufficient_context_budget" not in (candidate.get("validation_flags") or [])):
        return {
            "approval_reasons": reasons, "approval_status": "pending",
            "reviewed_candidate": candidate, "final_answer_dict": {},
        }
    return {"approval_reasons": reasons, "approval_status": "not_required"}


def approval_gate(state: FinaRAGState) -> dict:
    decision = interrupt({"turn_id": state["turn_id"], "approval_reasons": state["approval_reasons"]})
    if not isinstance(decision, dict) or decision.get("decision") not in {"approve", "reject"}:
        raise ValueError("Invalid approval decision")
    approved = decision["decision"] == "approve"
    status = "approved" if approved else "rejected"
    return {
        "approval_status": status,
        "approval_decision": decision,
        "reviewed_candidate": None,
        "final_answer_dict": state["reviewed_candidate"] if approved else _refusal("human_rejected"),
        "stage_events": [{"stage": "approval_decision", "approval_status": status}],
    }


_FOLLOW_UP = re.compile(
    r"^(?:那(?:么|它|这|该|同)?|同一|同年|同公司|该公司|这家|它的|其|接着|再问|此前|上[一述]|"
    r"and\b|what about\b|how about\b|for (?:that|the same)\b|it(?:s|\b)|their\b|same\b)",
    re.IGNORECASE,
)
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)|去年|今年|上年|明年|前年|last year|this year", re.IGNORECASE)
_PREVIOUS_METRIC = re.compile(r"该指标|这个指标|同一指标|上述指标|(?:that|same) metric", re.IGNORECASE)


def _follow_up_question(processor, question, memory):
    scope = memory.get("scope") or {}
    citation_ids = [citation for citation in memory.get("citation_ids") or [] if citation in question]
    if not scope or not (_FOLLOW_UP.search(question.strip()) or citation_ids):
        return question
    components = []
    explicit_companies = []
    extract_companies = getattr(processor, "_extract_companies_from_subset", None)
    company = scope.get("company")
    if extract_companies:
        explicit_companies = extract_companies(question)
        if company and not explicit_companies:
            components.append(company)
    if explicit_companies and explicit_companies != [company]:
        return question
    year = scope.get("year")
    if year and not _YEAR.search(question):
        components.append(f"{year}年")
    metric = scope.get("metric")
    if metric and (_PREVIOUS_METRIC.search(question) or citation_ids):
        from src.metric_atoms import get_default_store

        if get_default_store().find_metric(question) is None:
            components.append(metric)
    if not components:
        return question
    return " ".join([*components, question])


def _with_budget_snapshot(node):
    def run(state: FinaRAGState, runtime: Runtime[FinaRAGRunContext]) -> dict:
        updates = node(state, runtime)
        if runtime.context and runtime.context.budget_context:
            updates["budget_snapshot"] = runtime.context.budget_context.summary()
        return updates
    return run


def build_finarag_graph(processor: Any, checkpointer: Optional[Any] = None):
    """Build the full graph with a durable topology when a saver is supplied."""
    builder = StateGraph(FinaRAGState, context_schema=FinaRAGRunContext)
    durable = checkpointer is not None
    plan = make_plan_route_node(processor)
    if durable:
        def durable_plan(state, runtime):
            memory = state.get("working_context") or {}
            budget = runtime.context.budget_context if runtime.context else None
            if memory.get("over_budget") or (budget and memory.get("estimated_tokens", 0) > budget.remaining_tokens):
                return {"route_target": "done", "final_answer_dict": _refusal("insufficient_context_budget")}
            question = _follow_up_question(processor, state["question"], memory)
            return plan({**state, "question": question})

        builder.add_node("prepare_durable_turn", prepare_durable_turn)
        builder.add_node("plan_route", _with_budget_snapshot(durable_plan))
        builder.add_node("prepare_approval", _with_budget_snapshot(prepare_approval))
        builder.add_node("approval_gate", _with_budget_snapshot(lambda state, runtime: approval_gate(state)))
        builder.add_edge(START, "prepare_durable_turn")
        builder.add_edge("prepare_durable_turn", "plan_route")
        builder.add_conditional_edges(
            "prepare_approval", lambda state: state["approval_status"] == "pending",
            {True: "approval_gate", False: END},
        )
        builder.add_edge("approval_gate", END)
    else:
        builder.add_node("plan_route", plan)
        builder.add_edge(START, "plan_route")

    for name, factory in (
        ("retrieve_rerank", make_retrieve_rerank_node),
        ("ground_generate", make_ground_generate_node),
        ("reflect_supplement", make_reflect_supplement_node),
    ):
        node = factory(processor)
        builder.add_node(name, _with_budget_snapshot(node) if durable else node)

    builder.add_node("finalize_single", make_finalize_single_node(processor))
    builder.add_node("comparative_plan", make_comparative_plan_node(processor))
    builder.add_node("company_worker", make_company_worker_node(processor))
    builder.add_node("comparative_reduce", make_comparative_reduce_node(processor))

    terminal = "prepare_approval" if durable else END
    builder.add_conditional_edges(
        "plan_route", _route_after_plan,
        {"done": terminal, "retrieve_rerank": "retrieve_rerank", "comparative_plan": "comparative_plan"},
    )
    builder.add_edge("retrieve_rerank", "ground_generate")
    builder.add_conditional_edges(
        "ground_generate", decide_reflection_edge,
        {"reflect": "reflect_supplement", "finalize": "finalize_single"},
    )
    builder.add_edge("reflect_supplement", "ground_generate")
    builder.add_edge("finalize_single", terminal)
    builder.add_conditional_edges("comparative_plan", fanout_companies, ["company_worker"])
    builder.add_edge("company_worker", "comparative_reduce")
    builder.add_edge("comparative_reduce", terminal)
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


def _graph_input(question, schema, question_meta, config, context, budget_context):
    initial: FinaRAGState = {"question": question, "schema": schema, "question_meta": question_meta or {}}
    if context is not None:
        initial.update(
            thread_id=config["configurable"]["thread_id"], turn_id=context.turn_id,
            working_context=context.working_context or {},
            budget_snapshot=context.budget_context.summary() if context.budget_context else {},
        )
    elif budget_context is not None:
        initial["budget_context"] = budget_context
    return initial


def _answer_from_values(values, budget_context=None):
    answer = dict(values.get("final_answer_dict") or {})
    if values.get("stage_events"):
        answer.setdefault("graph_stage_events", list(values["stage_events"]))
    if budget_context is not None and "budget_summary" not in answer:
        answer["budget_summary"] = budget_context.summary()
        answer["degraded_level"] = budget_context.degradation_level
    return answer


def _durable_answer(snapshot, context):
    fields = snapshot.config["configurable"]
    metadata = {
        "turn_id": snapshot.values.get("turn_id"),
        "checkpoint_id": fields["checkpoint_id"],
        "approval_status": snapshot.values.get("approval_status", "not_required"),
    }
    if snapshot.interrupts:
        return {
            **metadata, "approval_status": "pending", "interrupt_id": snapshot.interrupts[0].id,
            "approval_reasons": list(snapshot.values.get("approval_reasons") or []),
        }
    answer = _answer_from_values(snapshot.values, context.budget_context)
    if budget := snapshot.values.get("budget_snapshot"):
        answer["budget_summary"] = budget
        answer["degraded_level"] = budget["degraded_level"]
    return {**answer, **metadata}


def _node_events(payload):
    # START resets the list reducer with Overwrite, which is not a stage event.
    events = payload.get("stage_events") if isinstance(payload, dict) else None
    return events if isinstance(events, list) else ()


def _durable_invoke(graph, graph_input, config, context):
    if context.on_event is None:
        graph.invoke(graph_input, config=config, context=context, durability="sync")
    else:
        for update in graph.stream(graph_input, config=config, context=context, stream_mode="updates", durability="sync"):
            for node, payload in update.items():
                for event in _node_events(payload):
                    context.on_event({"node": node, **event})
    return _durable_answer(graph.get_state(config), context)


def run_finarag_graph(
    processor: Any,
    question: str,
    schema: str,
    question_meta: Optional[Dict[str, Any]] = None,
    *,
    config: Optional[Dict[str, Any]] = None,
    budget_context: Any = None,
    graph=None,
    context: FinaRAGRunContext | None = None,
) -> Dict[str, Any]:
    graph = graph if graph is not None else build_finarag_graph(processor)
    if context is not None and context.budget_context is None:
        context.budget_context = budget_context
    initial = _graph_input(question, schema, question_meta, config, context, budget_context)
    if context is not None:
        return _durable_invoke(graph, initial, config, context)
    return _answer_from_values(graph.invoke(initial, config=config), budget_context)


def continue_finarag_graph(*, graph, config, context):
    return _durable_invoke(graph, None, config, context)


def resume_finarag_graph(processor, *, graph, config, context, interrupt_id, decision):
    fields = config["configurable"]
    latest_config = {"configurable": {"thread_id": fields["thread_id"]}}
    snapshot = graph.get_state(latest_config)
    if (("checkpoint_id" in fields and fields["checkpoint_id"] != snapshot.config["configurable"].get("checkpoint_id"))
            or len(snapshot.interrupts) != 1 or snapshot.interrupts[0].id != interrupt_id):
        raise ValueError("Stale approval interrupt")
    if decision not in {"approve", "reject"}:
        raise ValueError("Invalid approval decision")
    context.budget_context = FinaRAGRunContext.restore_budget(snapshot.values.get("budget_snapshot") or {})
    payload = {"decision": decision, "interrupt_id": interrupt_id, "reviewer": context.reviewer}
    return _durable_invoke(graph, Command(resume={interrupt_id: payload}), latest_config, context)


def stream_finarag_events(
    processor: Any,
    question: str,
    schema: str,
    question_meta: Optional[Dict[str, Any]] = None,
    *,
    config: Optional[Dict[str, Any]] = None,
    budget_context: Any = None,
    graph=None,
    context: FinaRAGRunContext | None = None,
) -> Iterator[Dict[str, Any]]:
    """Yield incremental stage events as each LangGraph node completes."""
    graph = graph if graph is not None else build_finarag_graph(processor)
    if context is not None and context.budget_context is None:
        context.budget_context = budget_context
    initial = _graph_input(question, schema, question_meta, config, context, budget_context)
    options = {"context": context, "durability": "sync"} if context is not None else {}
    last_answer: Optional[Dict[str, Any]] = None
    for update in graph.stream(initial, config=config, stream_mode="updates", **options):
        for node_name, node_payload in update.items():
            if not isinstance(node_payload, dict):
                continue
            if node_payload.get("final_answer_dict"):
                last_answer = node_payload["final_answer_dict"]
            for event in _node_events(node_payload):
                yield {"node": node_name, **event}
    if context is not None:
        last_answer = _durable_answer(graph.get_state(config), context)
        if last_answer.get("approval_status") == "pending":
            yield {"node": "approval_gate", "stage": "approval_pending", **last_answer}
            return
    if last_answer is not None:
        yield {
            "node": "end",
            "stage": "result",
            "final_answer": last_answer.get("final_answer"),
            "confidence": last_answer.get("confidence"),
            "validation_flags": last_answer.get("validation_flags") or [],
            "answer_dict": last_answer,
        }
