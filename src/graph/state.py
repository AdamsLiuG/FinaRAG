"""Typed state contracts for the FinaRAG LangGraph pipeline."""

from __future__ import annotations

import operator
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple
from typing_extensions import TypedDict


from src.service.budget_controller import BudgetContext


@dataclass
class FinaRAGRunContext:
    """Worker-local dependencies and policy for a durable invocation."""

    budget_context: BudgetContext | None = None
    turn_id: str = ""
    approval_enabled: bool = False
    working_context: Dict[str, Any] | None = None
    on_event: Callable[[Dict[str, Any]], None] | None = None
    reviewer: str | None = None

    @staticmethod
    def restore_budget(snapshot: Dict[str, Any]) -> BudgetContext:
        if not snapshot:
            return BudgetContext()
        latency = snapshot["max_latency_ms"]
        elapsed = min(max(snapshot["elapsed_ms"], 0), latency if latency > 0 else 120000)
        return BudgetContext(
            max_tokens=snapshot["max_tokens"],
            max_latency_ms=latency,
            spent_tokens=snapshot["spent_tokens"],
            start_ts=time.monotonic() - elapsed / 1000,
            fast_path=snapshot["fast_path"],
            degraded_events=list(snapshot["degraded_events"]),
        )


class FinaRAGState(TypedDict, total=False):
    # Request inputs
    question: str
    schema: str
    question_meta: Dict[str, Any]

    # Durable session inputs and review outcome
    thread_id: str
    turn_id: str
    working_context: Dict[str, Any]
    budget_snapshot: Dict[str, Any]
    approval_reasons: List[str]
    approval_status: Literal["not_required", "pending", "approved", "rejected"]
    reviewed_candidate: Dict[str, Any] | None
    approval_decision: Dict[str, Any] | None

    # Routing outputs
    route_decision: Dict[str, Any]
    route_target: str
    company_name: str
    companies: List[str]
    query_plan: Any
    route_info: Dict[str, Any]
    route_mode: str
    candidate_doc_ids: List[str]
    allowed_grounding_doc_ids: List[str]
    selected_doc_id: Optional[str]

    # Retrieval & reranking outputs
    retrieval_mode: str
    retriever_ref: Any
    retrieval_results: List[Dict[str, Any]]
    candidate_pool_size_before_rerank: Optional[int]
    reranking_debug: Dict[str, Any]
    hyde_debug: Dict[str, Any]

    # Adaptive reflection loop state
    adaptive_rounds_used: int
    adaptive_seen_queries: List[Tuple[str, ...]]
    adaptive_traces: Annotated[List[Dict[str, Any]], operator.add]
    pending_gap: Any

    # Comparative Map-Reduce state
    rephrased_questions: Dict[str, str]
    sub_answers: Annotated[List[Tuple[str, Dict[str, Any]]], operator.add]

    # Deep research outputs
    deep_research_plan: Dict[str, Any]
    evidence_ledger: Dict[str, Any]
    deep_research_status: Dict[str, Any]

    # Grounding, generation, validation & observability
    validated_flags: List[str]
    final_answer_dict: Dict[str, Any]
    stage_events: Annotated[List[Dict[str, Any]], operator.add]
    budget_context: Any


class CompanyBranchState(TypedDict):
    company: str
    sub_question: str
    schema: str
