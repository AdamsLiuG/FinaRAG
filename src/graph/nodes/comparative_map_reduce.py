"""Comparative Map-Reduce nodes using LangGraph Send() parallel fan-out."""

from __future__ import annotations

import time
from typing import Any, Dict, List

from langgraph.types import Send

from src.citation_formatter import dedupe_citations, dedupe_references
from src.harness.trajectory_store import capture_active, capture_step, step_scope
from src.service.tracing import token_usage_of


def make_comparative_plan_node(processor: Any):
    def comparative_plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        question = state["question"]
        companies: List[str] = list(state.get("companies") or [])
        rephrased_questions = processor.api_processor.get_rephrased_questions(
            original_question=question,
            companies=companies,
            model=processor.answering_model,
        )
        if capture_active():
            capture_step(
                "generation", operation="comparative_rephrase", step_key="rephrase",
                inputs={"question": question, "companies": companies, "model": processor.answering_model},
                outputs={"search_queries": [rephrased_questions.get(company) for company in companies]},
                costs={"elapsed_ms": (time.perf_counter() - started) * 1000},
            )
        return {
            "rephrased_questions": rephrased_questions,
            "stage_events": [
                {
                    "stage": "comparative_plan",
                    "companies": companies,
                    "ms": round((time.perf_counter() - started) * 1000, 2),
                }
            ],
        }

    return comparative_plan_node


def fanout_companies(state: Dict[str, Any]) -> List[Send]:
    companies: List[str] = list(state.get("companies") or [])
    rephrased: Dict[str, str] = state.get("rephrased_questions") or {}
    schema = state["schema"]
    sends: List[Send] = []
    for index, company in enumerate(companies):
        sub_question = rephrased.get(company)
        if not sub_question:
            raise ValueError(f"Could not generate sub-question for company: {company}")
        sends.append(
            Send(
                "company_worker",
                {
                    "company": company,
                    "branch_index": index,
                    "sub_question": sub_question,
                    "schema": schema,
                },
            )
        )
    return sends


def make_company_worker_node(processor: Any):
    def company_worker_node(branch_state: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        company = branch_state["company"]
        sub_question = branch_state["sub_question"]
        schema = branch_state["schema"]
        observing = capture_active()
        if observing:
            parent = capture_step(
                "plan", operation="comparative_dispatch", step_key="dispatch",
                inputs={"company": company, "schema": schema}, outputs={"question": sub_question},
            )
            with step_scope("company", parents=(parent,)):
                answer_dict = processor.get_answer_for_company(company_name=company, question=sub_question, schema=schema)
            capture_step(
                "generation", operation="comparative_worker", step_key="worker",
                inputs={"company": company}, outputs={"answer": answer_dict},
                parents=(parent,), costs={"elapsed_ms": (time.perf_counter() - started) * 1000},
            )
        else:
            answer_dict = processor.get_answer_for_company(company_name=company, question=sub_question, schema=schema)
        return {
            "sub_answers": [(company, answer_dict)],
            "stage_events": [
                {
                    "stage": "company_worker",
                    "company": company,
                    "ms": round((time.perf_counter() - started) * 1000, 2),
                }
            ],
        }

    return company_worker_node


def make_comparative_reduce_node(processor: Any):
    def comparative_reduce_node(state: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        question = state["question"]
        schema = state["schema"]
        companies: List[str] = list(state.get("companies") or [])
        sub_map = dict(state.get("sub_answers") or [])

        individual_answers: Dict[str, Dict[str, Any]] = {}
        aggregated_references: List[Dict[str, Any]] = []
        aggregated_citations: List[Dict[str, Any]] = []
        for company in companies:
            if company not in sub_map:
                continue
            answer_dict = sub_map[company]
            individual_answers[company] = answer_dict
            aggregated_references.extend(answer_dict.get("references", []))
            aggregated_citations.extend(answer_dict.get("citations", []))

        final_schema = "text" if schema in {"text", "long_text"} else "comparative"
        comparative_answer = processor.api_processor.get_answer_from_rag_context(
            question=question,
            rag_context=individual_answers,
            schema=final_schema,
            model=processor.answering_model,
            temperature=processor.answer_temperature,
        )
        processor.response_data = dict(getattr(processor.api_processor, "response_data", {}) or {})
        observing = capture_active()
        if observing:
            generation_key = capture_step(
                "generation", operation="comparative_reduce", step_key="generation",
                inputs={"question": question, "schema": final_schema,
                        "answers": list(individual_answers.values())},
                outputs={"final_answer": comparative_answer.get("final_answer")},
                costs={
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                    **token_usage_of({"response_data": processor.response_data}),
                },
            )
        comparative_answer["references"] = dedupe_references(aggregated_references)
        comparative_answer["citations"] = dedupe_citations(aggregated_citations)
        comparative_answer["confidence"] = processor._confidence_from_individual_answers(individual_answers)
        comparative_answer["confidence_reason"] = "当前置信度由比较问答模式下的各公司答案聚合而来。"
        comparative_answer["validation_flags"] = []
        comparative_answer["search_queries"] = [question]
        comparative_answer["query_plan"] = processor._build_query_plan(
            question,
            schema="comparative",
            mentioned_companies=companies,
            route_mode="comparative_explicit",
        ).to_dict()
        comparative_answer["route_info"] = {
            "route_mode": "comparative_explicit",
            "selected_company": None,
            "candidate_companies": companies,
            "selection_reasons": ["multiple_companies_mentioned_in_question"],
        }
        comparative_answer["retrieval_pages"] = sorted(
            {citation.get("page") for citation in aggregated_citations if citation.get("page") is not None}
        )
        comparative_answer["retrieval_results"] = []
        comparative_answer["response_data"] = processor.response_data

        if observing:
            capture_step(
                "verification", operation="comparative_aggregate", step_key="verification",
                inputs={}, outputs={"answer": comparative_answer}, parents=(generation_key,),
            )
        return {
            "final_answer_dict": comparative_answer,
            "stage_events": [
                {
                    "stage": "comparative_reduce",
                    "companies": companies,
                    "ms": round((time.perf_counter() - started) * 1000, 2),
                }
            ],
        }

    return comparative_reduce_node
