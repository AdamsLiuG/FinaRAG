"""Grounding, generation, optional PoT execution, and validation node."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from src.answer_validation import validate_answer
from src.citation_formatter import build_citations, compute_confidence
from src.structured.pot_sandbox import PotCalculationError, PotSecurityError, extract_and_execute_pot


def make_ground_generate_node(processor: Any):
    def ground_generate_node(state: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        question = state["question"]
        schema = state["schema"]
        company_name = state.get("company_name") or ""
        rewrite_result = state["query_plan"]
        route_info = state.get("route_info")
        route_mode = state.get("route_mode") or getattr(rewrite_result, "route_mode", "explicit_company")
        candidate_doc_ids = list(state.get("candidate_doc_ids") or [])
        allowed_grounding_doc_ids = list(state.get("allowed_grounding_doc_ids") or [])
        candidate_pool_size_before_rerank = state.get("candidate_pool_size_before_rerank")
        reranking_debug = state["reranking_debug"]
        hyde_debug = state["hyde_debug"]
        current_results: List[Dict[str, Any]] = list(state.get("retrieval_results") or [])

        grounding_result = None
        chart_grounding_result = None
        if schema == "number" and processor.table_grounder is not None:
            if processor._is_revenue_metric_question(question) and allowed_grounding_doc_ids:
                grounding_result = processor._cached_revenue_grounding(
                    company_name=company_name,
                    question=question,
                    query_plan=rewrite_result,
                    candidate_doc_ids=candidate_doc_ids,
                    allowed_doc_ids=allowed_grounding_doc_ids,
                    retrieval_results=current_results,
                )
            if grounding_result is None:
                grounding_result = processor.table_grounder.ground_number_query(
                    question=question,
                    retrieval_results=current_results,
                    filters=rewrite_result.filters,
                    candidate_doc_ids=candidate_doc_ids,
                    allowed_doc_ids=allowed_grounding_doc_ids,
                )
            if grounding_result is not None and processor._grounded_number_value(grounding_result) is not None:
                current_results = processor._ensure_table_grounding_retrieval_result(current_results, grounding_result)
                current_results = processor._ensure_table_support_retrieval_results(current_results, grounding_result)

        if schema == "number" and grounding_result is None and processor.chart_grounder is not None:
            chart_grounding_result = processor.chart_grounder.ground_number_query(
                question=question,
                retrieval_results=current_results,
                filters=rewrite_result.filters,
                candidate_doc_ids=candidate_doc_ids,
                allowed_doc_ids=allowed_grounding_doc_ids,
            )
            if chart_grounding_result is not None and processor._grounded_number_value(chart_grounding_result) is not None:
                current_results = processor._ensure_chart_grounding_retrieval_result(current_results, chart_grounding_result)

        if not current_results:
            processor.response_data = {}
            no_context_answer = {
                "final_answer": "N/A",
                "relevant_pages": [],
                "references": [],
                "citations": [],
                "confidence": "low",
                "confidence_reason": "未检索到可支撑回答的上下文。",
                "reasoning_summary": "检索阶段没有返回相关证据，因此拒答。",
                "step_by_step_analysis": "检索阶段没有返回相关证据，因此拒答。",
                "search_queries": rewrite_result.search_queries,
                "query_plan": rewrite_result.to_dict(),
                "route_info": route_info or {
                    "route_mode": "explicit_company",
                    "selected_company": company_name,
                    "candidate_companies": [company_name] if company_name else [],
                    "selection_reasons": ["company_argument"],
                },
                "candidate_pool_size_before_rerank": candidate_pool_size_before_rerank,
                "reranking_strategy": reranking_debug["reranking_strategy"],
                "initial_candidate_pool_size": reranking_debug["initial_candidate_pool_size"],
                "colbert_candidate_pool_size": reranking_debug["colbert_candidate_pool_size"],
                "colbert_top_n": reranking_debug["colbert_top_n"],
                "final_reranking_backend": reranking_debug["final_reranking_backend"],
                "hyde": hyde_debug,
                "retrieval_pages": [],
                "retrieval_results": [],
                "retrieval_report_groups": [],
                "response_data": processor.response_data,
            }
            if not processor.reasoning_debug_enabled:
                no_context_answer["step_by_step_analysis"] = ""
            validated_answer = validate_answer(no_context_answer, [], rewrite_result)
            return {
                "retrieval_results": current_results,
                "validated_flags": list(validated_answer.validation_flags or []),
                "final_answer_dict": validated_answer.answer,
                "stage_events": [
                    {
                        "stage": "ground_generate",
                        "no_context": True,
                        "flags": list(validated_answer.validation_flags or []),
                        "confidence": validated_answer.confidence,
                        "ms": round((time.perf_counter() - started) * 1000, 2),
                    }
                ],
            }

        local_response_data: Dict[str, Any] = {}
        budget_ctx = state.get("budget_context")
        if schema == "number" and grounding_result is not None and processor._grounded_number_value(grounding_result) is not None:
            answer_dict = processor._build_table_grounded_number_answer(grounding_result)
            local_response_data = {}
            processor.response_data = local_response_data
        elif schema == "number" and chart_grounding_result is not None and processor._grounded_number_value(chart_grounding_result) is not None:
            answer_dict = processor._build_chart_grounded_number_answer(chart_grounding_result)
            local_response_data = {}
            processor.response_data = local_response_data
        else:
            if budget_ctx is not None:
                effective_top_n = budget_ctx.effective_top_n(processor.top_n_retrieval)
                answer_context_results = budget_ctx.compress_results(current_results[:effective_top_n])
                target_model = (
                    os.getenv("FAST_ANSWERING_MODEL", "").strip() or processor.answering_model
                    if budget_ctx.degradation_level >= 3
                    else processor.answering_model
                )
            else:
                answer_context_results = current_results[: processor.top_n_retrieval]
                target_model = processor.answering_model
            rag_context = processor._format_retrieval_results(answer_context_results)
            answer_dict = processor.api_processor.get_answer_from_rag_context(
                question=question,
                rag_context=rag_context,
                schema=schema,
                model=target_model,
                temperature=processor.answer_temperature,
                metric_caliber_hint=processor._build_metric_caliber_hint(rewrite_result),
            )
            local_response_data = dict(getattr(processor.api_processor, "response_data", {}) or {})
            processor.response_data = local_response_data
            if budget_ctx is not None:
                usage = local_response_data.get("usage") or {}
                budget_ctx.record_usage(
                    prompt_tokens=int(usage.get("prompt_tokens") or local_response_data.get("input_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or local_response_data.get("output_tokens") or 0),
                )

            # Optional PoT deterministic evaluation when LLM emits structured pot_calculation
            if isinstance(answer_dict, dict) and answer_dict.get("pot_calculation"):
                try:
                    pot_res = extract_and_execute_pot(answer_dict)
                    if pot_res is not None:
                        answer_dict["final_answer"] = pot_res.value if schema == "number" else f"{pot_res.value} {pot_res.unit}".strip()
                        answer_dict["pot_result"] = {
                            "expression": pot_res.expression,
                            "value": pot_res.value,
                            "unit": pot_res.unit,
                            "base_unit": pot_res.base_unit,
                        }
                except (PotSecurityError, PotCalculationError):
                    pass

        pages = answer_dict.get("relevant_pages", [])
        validated_pages = processor._validate_page_references(pages, current_results)
        selected_report = (route_info or {}).get("selected_report") or {}
        answer_dict["relevant_pages"] = validated_pages
        if route_mode == "document_catalog_multi":
            answer_dict["references"] = processor._extract_references_from_results(validated_pages, current_results)
        else:
            answer_dict["references"] = processor._extract_references(
                validated_pages,
                company_name,
                pdf_sha1=selected_report.get("sha1"),
            )
        answer_dict["citations"] = build_citations(
            current_results,
            validated_pages,
            table_grounding_result=answer_dict.get("table_grounding_result"),
            table_support_results=processor._table_support_results(answer_dict.get("table_grounding_result")),
            chart_grounding_result=answer_dict.get("chart_grounding_result"),
        )
        answer_dict["confidence"] = compute_confidence(answer_dict, current_results)
        answer_dict["search_queries"] = rewrite_result.search_queries
        answer_dict["query_plan"] = rewrite_result.to_dict()
        answer_dict["route_info"] = route_info or {
            "route_mode": rewrite_result.route_mode,
            "selected_company": company_name,
            "candidate_companies": [company_name],
        }
        answer_dict["candidate_pool_size_before_rerank"] = candidate_pool_size_before_rerank
        answer_dict["reranking_strategy"] = reranking_debug["reranking_strategy"]
        answer_dict["initial_candidate_pool_size"] = reranking_debug["initial_candidate_pool_size"]
        answer_dict["colbert_candidate_pool_size"] = reranking_debug["colbert_candidate_pool_size"]
        answer_dict["colbert_top_n"] = reranking_debug["colbert_top_n"]
        answer_dict["final_reranking_backend"] = reranking_debug["final_reranking_backend"]
        answer_dict["hyde"] = hyde_debug
        answer_dict["retrieval_pages"] = [result.get("page") for result in current_results]
        answer_dict["retrieval_results"] = [processor._serialize_retrieval_result(result) for result in current_results]
        answer_dict["retrieval_report_groups"] = processor._aggregate_retrieval_results_by_report(current_results)
        answer_dict["response_data"] = local_response_data
        if budget_ctx is not None:
            answer_dict["degraded_level"] = budget_ctx.degradation_level
            answer_dict["budget_summary"] = budget_ctx.summary()
        if not processor.reasoning_debug_enabled:
            answer_dict["step_by_step_analysis"] = ""

        validated_answer = validate_answer(answer_dict, current_results, rewrite_result)
        final_flags = list(validated_answer.validation_flags or [])
        final_dict = dict(validated_answer.answer)

        # Optional P2 VLM visual verification on chart low-confidence or numeric grounding conflicts
        vlm_reader = getattr(processor, "vlm_page_reader", None)
        needs_vlm = (
            "chart_grounding_low_confidence" in final_flags
            or any("grounding" in f and "mismatch" in f for f in final_flags)
            or (chart_grounding_result is not None and float(chart_grounding_result.get("confidence") or 1.0) < 0.7)
        )
        if (
            vlm_reader is not None
            and needs_vlm
            and os.getenv("FINARAG_VLM_VERIFY_ENABLED", "0").strip().lower() in {"1", "true", "yes"}
        ):
            try:
                from src.multimodal_nav.visual_index import render_pdf_pages

                target_page = (
                    (validated_pages[0] if validated_pages else None)
                    or (chart_grounding_result or grounding_result or {}).get("page")
                    or (current_results[0].get("page") if current_results else None)
                )
                doc_id = (
                    (selected_report or {}).get("sha1")
                    or (current_results[0].get("pdf_sha1") if current_results else None)
                )
                pdf_dir = getattr(processor, "pdf_reports_dir", None)
                if pdf_dir is None and hasattr(processor, "documents_dir"):
                    candidate_pdf_dir = Path(processor.documents_dir).parent.parent / "pdf_reports"
                    if candidate_pdf_dir.exists():
                        pdf_dir = candidate_pdf_dir
                if target_page and doc_id and pdf_dir is not None:
                    pdf_file = Path(pdf_dir) / f"{doc_id}.pdf"
                    if pdf_file.exists():
                        with tempfile.TemporaryDirectory() as vlm_tmp:
                            rendered = render_pdf_pages(pdf_file, [int(target_page)], vlm_tmp, dpi=150)
                            if int(target_page) in rendered:
                                vlm_res = vlm_reader.verify_grounding_with_vlm(
                                    rendered[int(target_page)],
                                    question=question,
                                    candidate_answer=final_dict.get("final_answer"),
                                    page=int(target_page),
                                    allowed_base_dir=vlm_tmp,
                                )
                                final_dict["vlm_verification"] = vlm_res.to_dict()
            except Exception:
                pass

        # Optional P3 Claim-Level Grounding Verifier on long/text answers
        claim_contradicted = 0
        if (
            os.getenv("FINARAG_CLAIM_VERIFY_ENABLED", "0").strip().lower() in {"1", "true", "yes"}
            and schema in {"text", "long_text"}
            and isinstance(final_dict.get("final_answer"), str)
            and final_dict["final_answer"].strip() not in {"", "N/A"}
        ):
            try:
                from src.answer_validation import _downgrade_confidence
                from src.claim_verifier import ClaimVerifier

                verifier = ClaimVerifier(
                    sql_store=getattr(processor, "sql_store", None),
                    nli_scorer_fn=getattr(processor, "claim_nli_scorer", None),
                )
                q_filters = getattr(rewrite_result, "filters", None)
                audit = verifier.verify(
                    final_dict["final_answer"],
                    citations=final_dict.get("citations") or [],
                    retrieval_results=current_results,
                    company_name=company_name or getattr(q_filters, "company_name", None),
                    report_year=getattr(q_filters, "year", None),
                )
                final_dict["claim_audit"] = audit.to_dict()
                claim_contradicted = audit.contradicted_count
                if audit.contradicted_count > 0:
                    if "claim_contradicted" not in final_flags:
                        final_flags.append("claim_contradicted")
                    final_dict["validation_flags"] = final_flags
                    final_dict["confidence"] = "low"
                elif audit.total_claims >= 2 and audit.unsupported_claim_rate > 0.5:
                    if "claim_high_unverified_rate" not in final_flags:
                        final_flags.append("claim_high_unverified_rate")
                    final_dict["validation_flags"] = final_flags
                    final_dict["confidence"] = _downgrade_confidence(final_dict.get("confidence"))
            except Exception:
                pass

        return {
            "retrieval_results": current_results,
            "validated_flags": final_flags,
            "final_answer_dict": final_dict,
            "stage_events": [
                {
                    "stage": "ground_generate",
                    "flags": final_flags,
                    "confidence": final_dict.get("confidence", validated_answer.confidence),
                    "claim_contradicted": claim_contradicted,
                    "degraded_level": budget_ctx.degradation_level if budget_ctx is not None else 0,
                    "ms": round((time.perf_counter() - started) * 1000, 2),
                }
            ],
        }

    return ground_generate_node
