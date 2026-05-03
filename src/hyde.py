from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import src.prompts as prompts
from src.api_requests import APIProcessor
from src.query_plan import QueryPlan


HYDE_QUERY_MARKER = "__hyde__"


def _result_score(result: Dict[str, Any]) -> float:
    return float(result.get("combined_score", result.get("ranking_score", result.get("distance", 0.0))))


def _format_metadata_block(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    for key, value in payload.items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            rendered = ", ".join(str(item) for item in value if item not in (None, ""))
            if not rendered:
                continue
        else:
            rendered = str(value)
        lines.append(f"- {key}: {rendered}")
    return "\n".join(lines) if lines else "- none"


def should_trigger_hyde(
    retrieval_results: List[Dict[str, Any]],
    top_score_threshold: float,
    margin_threshold: float,
) -> Tuple[bool, List[str]]:
    if not retrieval_results:
        return True, ["no_results"]

    reasons: List[str] = []
    top_score = _result_score(retrieval_results[0])
    if top_score < float(top_score_threshold):
        reasons.append("low_top_score")

    if len(retrieval_results) >= 2:
        second_score = _result_score(retrieval_results[1])
        score_margin = top_score - second_score
        max_query_hit_count = max(
            int(result.get("query_hit_count", len(result.get("matched_queries", [])) or 0))
            for result in retrieval_results
        )
        if score_margin < float(margin_threshold) and max_query_hit_count <= 1:
            reasons.append("weak_margin_no_consensus")

    return bool(reasons), reasons


class HyDEGenerator:
    def __init__(
        self,
        provider: str = "qwen",
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 192,
        document_language: str = "en",
    ):
        self.provider = provider
        self.default_model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.document_language = document_language
        self.api_processor = APIProcessor(provider=provider)

    def _language_label(self) -> str:
        return "Chinese" if str(self.document_language).lower().startswith("zh") else "English"

    def _build_prompt(
        self,
        question: str,
        schema: str,
        query_plan: QueryPlan,
        route_info: Optional[Dict[str, Any]],
    ) -> str:
        route_info = route_info or {}
        selected_report = route_info.get("selected_report") or {}
        route_hints = dict(query_plan.route_hints or {})
        filters = query_plan.filters
        security_codes = route_hints.get("security_codes") or ([filters.security_code] if filters.security_code else [])

        route_payload = {
            "currency": route_hints.get("currency") or filters.currency,
            "year": route_hints.get("year") or filters.year,
            "doc_source_type": route_hints.get("doc_source_type") or filters.doc_source_type,
            "period": route_hints.get("period") or filters.period,
            "section_name": route_hints.get("section_name") or filters.section_name,
            "exchange": route_hints.get("exchange") or filters.exchange,
            "board": route_hints.get("board") or filters.board,
            "market_type": route_hints.get("market_type") or filters.market_type,
            "industry_l1": route_hints.get("industry_l1") or filters.industry_l1,
            "industry_l2": route_hints.get("industry_l2") or filters.industry_l2,
            "security_codes": security_codes,
            "topic_flags": query_plan.topic_flags,
        }
        report_payload = {
            "company_name": selected_report.get("company_name") or filters.company_name,
            "company_aliases": selected_report.get("company_aliases") or [],
            "report_year": selected_report.get("report_year"),
            "report_type": selected_report.get("report_type"),
            "doc_source_type": selected_report.get("doc_source_type"),
            "currency": selected_report.get("currency") or filters.currency,
            "major_industry": selected_report.get("major_industry") or route_hints.get("industry_l1"),
            "security_code": selected_report.get("security_code") or filters.security_code,
            "broker_name": selected_report.get("broker_name"),
            "report_title": selected_report.get("report_title"),
            "mentioned_companies": query_plan.mentioned_companies,
        }

        return prompts.HyDEPrompt.user_prompt.format(
            question=question,
            schema=schema,
            language=self._language_label(),
            company_name=filters.company_name or selected_report.get("company_name") or "unknown",
            route_mode=route_info.get("route_mode") or query_plan.route_mode,
            route_hints=_format_metadata_block(route_payload),
            report_metadata=_format_metadata_block(report_payload),
        )

    def generate(
        self,
        question: str,
        schema: str,
        query_plan: QueryPlan,
        route_info: Optional[Dict[str, Any]],
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> str:
        processor = self.api_processor
        if provider and provider.lower() != self.provider.lower():
            processor = APIProcessor(provider=provider)

        prompt = self._build_prompt(
            question=question,
            schema=schema,
            query_plan=query_plan,
            route_info=route_info,
        )
        generated = processor.send_message(
            model=model or self.default_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system_content=prompts.HyDEPrompt.system_prompt,
            human_content=prompt,
            is_structured=False,
        )
        return str(generated or "").strip()
