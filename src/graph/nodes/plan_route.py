"""Routing & deterministic fast-path planning node."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from src.graph.plugins import (
    CashDividendPlugin,
    LegalRepresentativePlugin,
    MetadataCountPlugin,
    MetadataMembershipPlugin,
    MetadataNamesPlugin,
    PluginRegistry,
)

_METADATA_PLUGINS = PluginRegistry(
    [
        MetadataCountPlugin(),
        MetadataNamesPlugin(),
        MetadataMembershipPlugin(),
    ]
)
_SINGLE_COMPANY_PLUGINS = PluginRegistry(
    [
        LegalRepresentativePlugin(),
        CashDividendPlugin(),
    ]
)

_CROSS_DOC_SQL_HINTS = ("排名前", "前", "最高", "最大", "最低", "最小", "哪些公司", "多少家", "几家")


def structured_sql_enabled(processor: Any) -> bool:
    if getattr(processor, "structured_router", None) is None:
        return False
    flag = os.getenv("FINARAG_STRUCTURED_SQL_ENABLED", "0").strip().lower()
    explicit = getattr(processor, "_structured_sql_explicit", False)
    return explicit or flag in {"1", "true", "yes"}


def make_prepare_single_company_node(processor: Any):
    """Node used when entering the single-company subgraph directly."""

    def prepare_single_company_node(state: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        if (
            not processor.use_vector_dbs
            and not processor.use_bm25_db
            and not processor.use_sparse_lexical_db
            and not processor.use_tag_db
        ):
            raise ValueError("At least one retrieval backend must be enabled.")

        question = state["question"]
        schema = state["schema"]
        company_name = state.get("company_name") or ""
        route_info = state.get("route_info")
        query_plan = state.get("query_plan")

        rewrite_result = query_plan or processor._build_query_plan(
            question,
            schema=schema,
            company_name=company_name,
            mentioned_companies=[company_name] if company_name else [],
            route_mode="explicit_company",
        )
        route_mode = (route_info or {}).get("route_mode") or rewrite_result.route_mode
        if route_mode == "document_catalog_multi":
            rewrite_result.filters.company_name = None
        else:
            rewrite_result.filters.company_name = company_name
        candidate_doc_ids = list((route_info or {}).get("candidate_doc_ids") or rewrite_result.filters.candidate_doc_ids or [])
        if candidate_doc_ids:
            rewrite_result.filters.candidate_doc_ids = candidate_doc_ids
        selected_doc_id = processor._selected_doc_id(company_name, route_info, rewrite_result)
        allowed_grounding_doc_ids = (
            [selected_doc_id]
            if selected_doc_id and route_mode != "document_catalog_multi"
            else list(candidate_doc_ids)
        )

        if structured_sql_enabled(processor) and company_name:
            sql_answer = processor.structured_router.try_answer(
                question=question,
                schema=schema,
                query_plan=rewrite_result,
                companies=[company_name],
            )
            if sql_answer is not None:
                return {
                    "route_target": "done",
                    "final_answer_dict": sql_answer,
                    "stage_events": [
                        {
                            "stage": "route",
                            "route_target": "structured_sql",
                            "ms": round((time.perf_counter() - started) * 1000, 2),
                        }
                    ],
                }

        plugin_ctx = {
            "question": question,
            "schema": schema,
            "company_name": company_name,
            "query_plan": rewrite_result,
            "route_info": route_info,
        }
        plugin_answer = _SINGLE_COMPANY_PLUGINS.execute_first(processor, plugin_ctx)
        if plugin_answer is not None:
            return {
                "route_target": "done",
                "final_answer_dict": plugin_answer,
                "stage_events": [
                    {
                        "stage": "route",
                        "route_target": (plugin_answer.get("route_info") or {}).get("route_mode", "rule_plugin"),
                        "ms": round((time.perf_counter() - started) * 1000, 2),
                    }
                ],
            }

        return {
            "route_target": "single_rag",
            "company_name": company_name,
            "query_plan": rewrite_result,
            "route_info": route_info,
            "route_mode": route_mode,
            "candidate_doc_ids": candidate_doc_ids,
            "allowed_grounding_doc_ids": allowed_grounding_doc_ids,
            "selected_doc_id": selected_doc_id,
            "adaptive_rounds_used": 0,
            "adaptive_seen_queries": [],
            "stage_events": [
                {
                    "stage": "prepare_single",
                    "route_target": "single_rag",
                    "ms": round((time.perf_counter() - started) * 1000, 2),
                }
            ],
        }

    return prepare_single_company_node


def make_plan_route_node(processor: Any):
    """Full question routing node executed at the start of process_question."""

    single_prep = make_prepare_single_company_node(processor)

    def plan_route_node(state: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        question = state["question"]
        schema = state["schema"]
        question_meta = state.get("question_meta") or {}

        # Optional cross-report SQL fast-track when no specific company is named.
        if (
            structured_sql_enabled(processor)
            and getattr(processor, "report_catalog", None) is not None
            and any(tok in (question or "") for tok in _CROSS_DOC_SQL_HINTS)
        ):
            try:
                extracted = processor._extract_companies_from_subset(question)
            except Exception:
                extracted = []
            if not extracted:
                cross_plan = processor._build_query_plan(
                    question,
                    schema=schema,
                    mentioned_companies=[],
                    route_mode="document_catalog_multi",
                )
                sql_answer = processor.structured_router.try_answer(
                    question=question,
                    schema=schema,
                    query_plan=cross_plan,
                    companies=[],
                )
                if sql_answer is not None:
                    return {
                        "route_target": "done",
                        "final_answer_dict": sql_answer,
                        "stage_events": [
                            {
                                "stage": "route",
                                "route_target": "structured_sql_cross_doc",
                                "ms": round((time.perf_counter() - started) * 1000, 2),
                            }
                        ],
                    }

        route_decision = processor.route_question(question, schema)
        if route_decision.get("is_comparative"):
            companies: List[str] = list(route_decision.get("companies") or [])
            if structured_sql_enabled(processor):
                sql_answer = processor.structured_router.try_answer(
                    question=question,
                    schema=schema,
                    query_plan=route_decision.get("query_plan"),
                    companies=companies,
                )
                if sql_answer is not None:
                    return {
                        "route_target": "done",
                        "route_decision": route_decision,
                        "final_answer_dict": sql_answer,
                        "stage_events": [
                            {
                                "stage": "route",
                                "route_target": "structured_sql_compare",
                                "ms": round((time.perf_counter() - started) * 1000, 2),
                            }
                        ],
                    }

            structured_answer = processor._answer_structured_revenue_comparison(question, companies, schema)
            if structured_answer is not None:
                return {
                    "route_target": "done",
                    "route_decision": route_decision,
                    "final_answer_dict": structured_answer,
                    "stage_events": [
                        {
                            "stage": "route",
                            "route_target": "numeric_comparison_rule",
                            "ms": round((time.perf_counter() - started) * 1000, 2),
                        }
                    ],
                }
            if "process_comparative_question" in getattr(processor, "__dict__", {}):
                comp_answer = processor.process_comparative_question(question, companies, schema)
                return {
                    "route_target": "done",
                    "route_decision": route_decision,
                    "final_answer_dict": comp_answer,
                    "stage_events": [
                        {
                            "stage": "route",
                            "route_target": "comparative_override",
                            "ms": round((time.perf_counter() - started) * 1000, 2),
                        }
                    ],
                }
            return {
                "route_target": "comparative_map",
                "route_decision": route_decision,
                "companies": companies,
                "stage_events": [
                    {
                        "stage": "route",
                        "route_target": "comparative_map",
                        "companies": companies,
                        "ms": round((time.perf_counter() - started) * 1000, 2),
                    }
                ],
            }

        definition_answer = processor._answer_metric_definition_question(
            question=question,
            schema=schema,
            route_decision=route_decision,
        )
        if definition_answer is not None:
            return {
                "route_target": "done",
                "route_decision": route_decision,
                "final_answer_dict": definition_answer,
                "stage_events": [
                    {
                        "stage": "route",
                        "route_target": "metric_atom_store",
                        "ms": round((time.perf_counter() - started) * 1000, 2),
                    }
                ],
            }

        metadata_answer = _METADATA_PLUGINS.execute_first(
            processor,
            {
                "question": question,
                "schema": schema,
                "route_decision": route_decision,
                "question_meta": question_meta,
            },
        )
        if metadata_answer is not None:
            return {
                "route_target": "done",
                "route_decision": route_decision,
                "final_answer_dict": metadata_answer,
                "stage_events": [
                    {
                        "stage": "route",
                        "route_target": (metadata_answer.get("route_info") or {}).get("route_mode", "metadata"),
                        "ms": round((time.perf_counter() - started) * 1000, 2),
                    }
                ],
            }

        if structured_sql_enabled(processor):
            sql_answer = processor.structured_router.try_answer(
                question=question,
                schema=schema,
                query_plan=route_decision.get("query_plan"),
                companies=route_decision.get("companies") or ([route_decision["company_name"]] if route_decision.get("company_name") else []),
            )
            if sql_answer is not None:
                return {
                    "route_target": "done",
                    "route_decision": route_decision,
                    "final_answer_dict": sql_answer,
                    "stage_events": [
                        {
                            "stage": "route",
                            "route_target": (sql_answer.get("route_info") or {}).get("route_mode", "structured_sql"),
                            "ms": round((time.perf_counter() - started) * 1000, 2),
                        }
                    ],
                }

        threshold_answer = processor._answer_structured_revenue_threshold(
            question=question,
            schema=schema,
            route_decision=route_decision,
        )
        if threshold_answer is not None:
            return {
                "route_target": "done",
                "route_decision": route_decision,
                "final_answer_dict": threshold_answer,
                "stage_events": [
                    {
                        "stage": "route",
                        "route_target": "numeric_threshold_rule",
                        "ms": round((time.perf_counter() - started) * 1000, 2),
                    }
                ],
            }

        if "get_answer_for_company" in getattr(processor, "__dict__", {}):
            overridden_answer = processor.get_answer_for_company(
                company_name=route_decision["company_name"],
                question=question,
                schema=schema,
                query_plan=route_decision["query_plan"],
                route_info=route_decision["route_info"],
            )
            return {
                "route_target": "done",
                "route_decision": route_decision,
                "final_answer_dict": overridden_answer,
                "stage_events": [
                    {
                        "stage": "route",
                        "route_target": "single_company_override",
                        "ms": round((time.perf_counter() - started) * 1000, 2),
                    }
                ],
            }

        prepared = single_prep(
            {
                "question": question,
                "schema": schema,
                "company_name": route_decision.get("company_name") or "",
                "query_plan": route_decision.get("query_plan"),
                "route_info": route_decision.get("route_info"),
            }
        )
        prepared["route_decision"] = route_decision
        return prepared

    return plan_route_node
