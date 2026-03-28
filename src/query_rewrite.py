from __future__ import annotations

import re
from typing import List, Optional

from src.query_plan import QueryPlan, kind_to_expected_answer_type
from src.retrieval_filters import RetrievalFilters
from src.text_normalization import dedupe_preserve_order, normalize_currency_token, normalize_text


_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_CURRENCY_RE = re.compile(r"\b(USD|EUR|GBP|JPY|CNY|RMB|AUD|CAD|CHF)\b|[$€£]")

_TERM_EXPANSIONS = {
    "operating margin": ["operating profit margin"],
    "gross margin": ["gross profit margin"],
    "share buyback": ["share repurchase", "repurchase program"],
    "mergers or acquisitions": ["m&a", "acquisition", "merger"],
    "executive compensation": ["named executive officer compensation", "compensation table"],
    "dividend policy": ["dividend strategy", "dividend framework"],
    "leadership positions": ["executive officers", "management changes", "board changes"],
}

_TOPIC_PATTERNS = {
    "mentions_recent_mergers_and_acquisitions": [
        "mergers or acquisitions",
        "mergers and acquisitions",
        "m&a",
        "acquisition",
        "acquisitions",
        "merger",
        "mergers",
    ],
    "has_leadership_changes": [
        "leadership",
        "executive",
        "executives",
        "management changes",
        "board changes",
        "ceo",
        "cfo",
        "chief executive officer",
        "chief financial officer",
    ],
    "has_dividend_policy_changes": [
        "dividend policy",
        "dividend strategy",
        "dividend framework",
        "dividend changes",
    ],
    "has_executive_compensation": [
        "executive compensation",
        "named executive officer compensation",
        "compensation table",
    ],
    "has_new_product_launches": [
        "new product",
        "new products",
        "product launch",
        "launched products",
    ],
    "has_share_buyback_plans": [
        "share buyback",
        "share repurchase",
        "repurchase program",
        "buyback plan",
    ],
    "has_guidance_updates": [
        "guidance",
        "outlook",
        "forecast",
    ],
    "has_regulatory_or_litigation_issues": [
        "litigation",
        "lawsuit",
        "regulatory",
        "investigation",
    ],
    "has_supply_chain_disruptions": [
        "supply chain",
        "disruption",
        "logistics",
    ],
}


class QuestionRewriter:
    def rewrite(
        self,
        question: str,
        schema: str,
        company_name: Optional[str] = None,
        mentioned_companies: Optional[List[str]] = None,
    ) -> QueryPlan:
        normalized_query = normalize_text(question)
        topic_flags = self._extract_topic_flags(normalized_query)
        filters = RetrievalFilters(
            company_name=company_name,
            currency=self._extract_currency(question),
            year=self._extract_year(question),
            required_topic_flags=topic_flags or None,
            question_kind=schema,
        )

        expanded_queries = [question.strip()]
        expanded_queries.extend(self._expand_financial_terms(question))
        if normalized_query and normalized_query not in expanded_queries:
            expanded_queries.append(normalized_query)

        if filters.currency and filters.currency not in question:
            expanded_queries.append(f"{question.strip()} {filters.currency}")
        if filters.year is not None:
            expanded_queries.append(f"{question.strip()} {filters.year}")

        return QueryPlan(
            original_query=question,
            normalized_query=normalized_query,
            search_queries=dedupe_preserve_order(expanded_queries),
            filters=filters,
            route_mode="explicit_company" if company_name else "metadata_inference",
            expected_answer_type=kind_to_expected_answer_type(schema),
            topic_flags=topic_flags,
            mentioned_companies=list(mentioned_companies or ([company_name] if company_name else [])),
            route_hints={"currency": filters.currency, "year": filters.year},
        )

    def _extract_year(self, question: str) -> Optional[int]:
        years = [int(match.group(0)) for match in _YEAR_RE.finditer(question)]
        if not years:
            return None
        return max(years)

    def _extract_currency(self, question: str) -> Optional[str]:
        match = _CURRENCY_RE.search(question)
        if not match:
            return None
        token = match.group(0)
        return normalize_currency_token(token)

    def _expand_financial_terms(self, question: str) -> List[str]:
        lowered = question.lower()
        expansions: List[str] = []
        for term, synonyms in _TERM_EXPANSIONS.items():
            if term not in lowered:
                continue
            for synonym in synonyms:
                expansions.append(question.replace(term, synonym))
                expansions.append(question.replace(term.title(), synonym))
        return expansions

    def _extract_topic_flags(self, normalized_question: str) -> List[str]:
        topic_flags: List[str] = []
        for topic_flag, patterns in _TOPIC_PATTERNS.items():
            if any(pattern in normalized_question for pattern in patterns):
                topic_flags.append(topic_flag)
        return topic_flags


QueryRewriteResult = QueryPlan
