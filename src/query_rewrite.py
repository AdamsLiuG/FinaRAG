from __future__ import annotations

import re
from typing import List, Optional

from src.query_plan import QueryPlan, kind_to_expected_answer_type
from src.retrieval_filters import RetrievalFilters
from src.text_normalization import (
    dedupe_preserve_order,
    extract_security_codes,
    normalize_currency_token,
    normalize_period_token,
    normalize_text,
)


_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_CURRENCY_RE = re.compile(r"\b(USD|EUR|GBP|JPY|CNY|RMB|AUD|CAD|CHF|HKD)\b|[$€£¥￥]")

_TERM_EXPANSIONS = {
    "operating margin": ["operating profit margin"],
    "gross margin": ["gross profit margin"],
    "share buyback": ["share repurchase", "repurchase program"],
    "mergers or acquisitions": ["m&a", "acquisition", "merger"],
    "营业收入": ["营收", "收入"],
    "归母净利润": ["母公司股东净利润", "归属于母公司股东的净利润", "归母"],
    "扣非归母净利润": ["扣非净利润", "归母扣非净利润"],
    "毛利率": ["综合毛利率"],
    "净利率": ["销售净利率"],
    "期间费用率": ["费用率"],
    "经营活动现金流净额": ["经营现金流净额", "经营性现金流净额"],
    "分红": ["派息", "股利", "现金分红"],
    "回购": ["股份回购", "股票回购"],
    "并购": ["收购", "兼并", "并表交易"],
    "定增": ["非公开发行", "再融资"],
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
        "并购",
        "收购",
        "兼并",
    ],
    "has_leadership_changes": [
        "leadership",
        "executive",
        "management changes",
        "board changes",
        "ceo",
        "cfo",
        "管理层变动",
        "董事变更",
        "高管变更",
    ],
    "has_dividend_policy_changes": [
        "dividend policy",
        "dividend strategy",
        "dividend framework",
        "dividend changes",
        "分红",
        "股利",
        "派息",
    ],
    "has_share_buyback_plans": [
        "share buyback",
        "share repurchase",
        "repurchase program",
        "buyback plan",
        "回购",
        "股份回购",
    ],
    "has_guidance_updates": [
        "guidance",
        "outlook",
        "forecast",
        "指引",
        "业绩预告",
        "展望",
    ],
    "has_regulatory_or_litigation_issues": [
        "litigation",
        "lawsuit",
        "regulatory",
        "investigation",
        "诉讼",
        "监管",
        "调查",
    ],
}


def _query_fingerprint(text: str) -> str:
    return re.sub(r"[\W_]+", "", normalize_text(text or ""))


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
        security_codes = extract_security_codes(question)
        doc_source_type = self._extract_doc_source_type(normalized_query)
        period = normalize_period_token(question)
        filters = RetrievalFilters(
            company_name=company_name,
            currency=self._extract_currency(question),
            year=self._extract_year(question),
            report_type=self._extract_report_type(normalized_query),
            doc_source_type=doc_source_type,
            security_code=security_codes[0] if security_codes else None,
            period=period,
            required_topic_flags=None,
            question_kind=schema,
        )

        expanded_queries: List[str] = []
        self._append_query(expanded_queries, question.strip())
        self._append_query(expanded_queries, normalized_query)

        for expanded_query in self._expand_financial_terms(question):
            self._append_query(expanded_queries, expanded_query)

        if filters.currency and filters.currency.lower() not in normalized_query:
            self._append_query(expanded_queries, f"{question.strip()} {filters.currency}")

        if filters.year is not None and str(filters.year) not in normalized_query:
            self._append_query(expanded_queries, f"{question.strip()} {filters.year}")

        if period and period.lower() not in normalized_query:
            self._append_query(expanded_queries, f"{question.strip()} {period}")

        if filters.doc_source_type == "research_report" and "研报" not in normalized_query and "券商" not in normalized_query:
            self._append_query(expanded_queries, f"{question.strip()} 券商研报")
        elif filters.doc_source_type == "annual_report" and "年报" not in normalized_query and "annual report" not in normalized_query:
            self._append_query(expanded_queries, f"{question.strip()} 年报")

        search_queries = self._limit_search_queries(
            expanded_queries,
            schema=schema,
            company_name=company_name,
        )

        return QueryPlan(
            original_query=question,
            normalized_query=normalized_query,
            search_queries=search_queries,
            filters=filters,
            route_mode="explicit_company" if company_name else "document_catalog",
            expected_answer_type=kind_to_expected_answer_type(schema),
            topic_flags=topic_flags,
            mentioned_companies=list(mentioned_companies or ([company_name] if company_name else [])),
            route_hints={
                "currency": filters.currency,
                "year": filters.year,
                "doc_source_type": filters.doc_source_type,
                "period": period,
                "security_codes": security_codes,
            },
        )

    def _extract_year(self, question: str) -> Optional[int]:
        years = [int(match.group(0)) for match in _YEAR_RE.finditer(question)]
        if not years:
            return None
        return max(years)

    def _extract_currency(self, question: str) -> Optional[str]:
        match = _CURRENCY_RE.search(question)
        if not match:
            if "人民币" in question:
                return "CNY"
            return None
        token = match.group(0)
        return normalize_currency_token(token)

    def _extract_doc_source_type(self, normalized_question: str) -> Optional[str]:
        if any(keyword in normalized_question for keyword in ("研报", "券商", "深度报告", "首次覆盖", "点评报告")):
            return "research_report"
        if any(keyword in normalized_question for keyword in ("年报", "年度报告", "annual report")):
            return "annual_report"
        if any(keyword in normalized_question for keyword in ("中报", "半年报", "季报", "q1", "q2", "q3", "q4")):
            return "interim_report"
        return None

    def _extract_report_type(self, normalized_question: str) -> Optional[str]:
        if "annual report" in normalized_question or "年报" in normalized_question or "年度报告" in normalized_question:
            return "annual"
        if "中报" in normalized_question or "半年报" in normalized_question:
            return "interim"
        if "季报" in normalized_question or any(token in normalized_question for token in ("q1", "q2", "q3", "q4")):
            return "quarterly"
        return None

    def _expand_financial_terms(self, question: str) -> List[str]:
        expansions: List[str] = []
        lowered = question.lower()
        for term, synonyms in _TERM_EXPANSIONS.items():
            if term.lower() not in lowered and term not in question:
                continue
            for synonym in synonyms:
                replaced = question.replace(term, synonym)
                if replaced != question:
                    expansions.append(replaced)
                replaced_lower = question.replace(term.lower(), synonym)
                if replaced_lower != question:
                    expansions.append(replaced_lower)
        return dedupe_preserve_order(expansions)

    def _extract_topic_flags(self, normalized_question: str) -> List[str]:
        topic_flags: List[str] = []
        for topic_flag, patterns in _TOPIC_PATTERNS.items():
            if any(pattern in normalized_question for pattern in patterns):
                topic_flags.append(topic_flag)
        return topic_flags

    def _append_query(self, expanded_queries: List[str], candidate: str | None) -> None:
        normalized_candidate = (candidate or "").strip()
        if not normalized_candidate:
            return

        candidate_fingerprint = _query_fingerprint(normalized_candidate)
        if not candidate_fingerprint:
            return

        for existing in expanded_queries:
            if _query_fingerprint(existing) == candidate_fingerprint:
                return
        expanded_queries.append(normalized_candidate)

    def _limit_search_queries(
        self,
        expanded_queries: List[str],
        schema: str,
        company_name: Optional[str] = None,
    ) -> List[str]:
        if not expanded_queries:
            return []

        if company_name:
            max_queries = 2 if schema in {"name", "boolean"} else 3
        else:
            max_queries = 4

        return expanded_queries[:max_queries]


QueryRewriteResult = QueryPlan
