from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

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


@dataclass
class QueryRewriteResult:
    original_query: str
    normalized_query: str
    search_queries: List[str]
    filters: RetrievalFilters


class QuestionRewriter:
    def rewrite(self, question: str, schema: str, company_name: Optional[str] = None) -> QueryRewriteResult:
        normalized_query = normalize_text(question)
        filters = RetrievalFilters(
            company_name=company_name,
            currency=self._extract_currency(question),
            year=self._extract_year(question),
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

        return QueryRewriteResult(
            original_query=question,
            normalized_query=normalized_query,
            search_queries=dedupe_preserve_order(expanded_queries),
            filters=filters,
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
