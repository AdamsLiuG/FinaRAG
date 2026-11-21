"""Typed state contracts for the FinaRAG LangGraph pipeline."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, Tuple
from typing_extensions import TypedDict


class FinaRAGState(TypedDict, total=False):
    # Request inputs
    question: str
    schema: str
    question_meta: Dict[str, Any]

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

    # Grounding, generation, validation & observability
    validated_flags: List[str]
    final_answer_dict: Dict[str, Any]
    stage_events: Annotated[List[Dict[str, Any]], operator.add]
    budget_context: Any


class CompanyBranchState(TypedDict):
    company: str
    sub_question: str
    schema: str
