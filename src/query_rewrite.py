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

_EXCHANGE_PATTERNS = {
    "上海证券交易所": ["上海证券交易所", "上交所", "sse"],
    "深圳证券交易所": ["深圳证券交易所", "深交所", "szse"],
    "北京证券交易所": ["北京证券交易所", "北交所", "bse"],
}

_BOARD_PATTERNS = {
    "科创板": ["科创板"],
    "创业板": ["创业板"],
    "沪主板": ["沪主板", "上海主板"],
    "深主板": ["深主板", "深圳主板"],
    "北交所": ["北交所", "北京证券交易所"],
}

_MARKET_TYPE_PATTERNS = {
    "A股": ["a股", "a 股"],
    "H股": ["h股", "h 股"],
    "港股": ["港股"],
}

_SECTION_HINTS = [
    "管理层讨论与分析",
    "公司简介和主要财务指标",
    "重要事项",
    "公司治理",
    "财务报告",
    "释义",
    "股份变动及股东情况",
    "风险提示",
]

_TAG_FILTER_PATTERNS = {
    "strategy_tags": {
        "出海": ["出海", "海外", "国际化", "境外"],
        "数字化转型": ["数字化", "数智化", "数字化转型"],
        "人工智能": ["人工智能", "ai", "大模型"],
        "绿色转型": ["绿色转型", "绿色低碳", "双碳", "碳中和"],
        "国产替代": ["国产替代", "自主可控", "国产化"],
    },
    "listing_tags": {
        "A股": ["a股", "a 股"],
        "科创板": ["科创板"],
        "创业板": ["创业板"],
        "沪主板": ["沪主板", "上海主板"],
        "深主板": ["深主板", "深圳主板"],
    },
    "status_tags": {
        "龙头": ["龙头", "龙头候选"],
    },
    "factor_tags": {
        "高资本开支": ["资本开支", "capex"],
    },
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
        tag_filters = self._extract_tag_filters(normalized_query)
        security_codes = extract_security_codes(question)
        doc_source_type = self._extract_doc_source_type(normalized_query)
        period = normalize_period_token(question)
        filters = RetrievalFilters(
            company_name=company_name,
            currency=self._extract_currency(question),
            year=self._extract_year(question),
            report_type=self._extract_report_type(normalized_query),
            doc_source_type=doc_source_type,
            exchange=self._extract_exchange(normalized_query),
            board=self._extract_board(normalized_query),
            market_type=self._extract_market_type(normalized_query),
            industry_l1=self._extract_industry_label(question),
            industry_l2=None,
            security_code=security_codes[0] if security_codes else None,
            period=period,
            section_name=self._extract_section_name(question),
            business_tags=tag_filters.get("business_tags"),
            strategy_tags=tag_filters.get("strategy_tags"),
            factor_tags=tag_filters.get("factor_tags"),
            chain_position_major=self._extract_chain_position(normalized_query),
            chain_position_minor=tag_filters.get("chain_position_minor"),
            listing_tags=tag_filters.get("listing_tags"),
            ownership_tags=tag_filters.get("ownership_tags"),
            status_tags=tag_filters.get("status_tags"),
            style_tags=tag_filters.get("style_tags"),
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

        if filters.section_name and filters.section_name not in question:
            self._append_query(expanded_queries, f"{question.strip()} {filters.section_name}")

        for filter_value in (
            filters.exchange,
            filters.board,
            filters.market_type,
            filters.industry_l1,
            filters.chain_position_major,
        ):
            if filter_value and normalize_text(str(filter_value)) not in normalized_query:
                self._append_query(expanded_queries, f"{question.strip()} {filter_value}")

        for list_filter in (
            filters.business_tags,
            filters.strategy_tags,
            filters.factor_tags,
            filters.chain_position_minor,
            filters.listing_tags,
            filters.ownership_tags,
            filters.status_tags,
            filters.style_tags,
        ):
            for value in list_filter or []:
                if normalize_text(str(value)) not in normalized_query:
                    self._append_query(expanded_queries, f"{question.strip()} {value}")

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
                "section_name": filters.section_name,
                "exchange": filters.exchange,
                "board": filters.board,
                "market_type": filters.market_type,
                "industry_l1": filters.industry_l1,
                "industry_l2": filters.industry_l2,
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

    def _extract_exchange(self, normalized_question: str) -> Optional[str]:
        for target, patterns in _EXCHANGE_PATTERNS.items():
            if any(pattern in normalized_question for pattern in patterns):
                return target
        return None

    def _extract_board(self, normalized_question: str) -> Optional[str]:
        for target, patterns in _BOARD_PATTERNS.items():
            if any(pattern in normalized_question for pattern in patterns):
                return target
        return None

    def _extract_market_type(self, normalized_question: str) -> Optional[str]:
        for target, patterns in _MARKET_TYPE_PATTERNS.items():
            if any(pattern in normalized_question for pattern in patterns):
                return target
        return None

    def _extract_section_name(self, question: str) -> Optional[str]:
        normalized_question = normalize_text(question)
        for hint in _SECTION_HINTS:
            if normalize_text(hint) in normalized_question:
                return hint
        return None

    def _extract_industry_label(self, question: str) -> Optional[str]:
        matches = re.findall(r"([A-Za-z0-9\u4e00-\u9fff·]{2,20})(?:行业|板块)", question)
        if matches:
            return matches[0]
        return None

    def _extract_chain_position(self, normalized_question: str) -> Optional[str]:
        if "上游" in normalized_question:
            return "上游资源"
        if "中游" in normalized_question:
            return "中游制造"
        if "下游" in normalized_question:
            return "下游应用"
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

    def _extract_tag_filters(self, normalized_question: str) -> dict[str, List[str]]:
        extracted: dict[str, List[str]] = {}
        for field, pattern_map in _TAG_FILTER_PATTERNS.items():
            values: List[str] = []
            for target, patterns in pattern_map.items():
                if any(pattern in normalized_question for pattern in patterns):
                    values.append(target)
            if values:
                extracted[field] = values
        return extracted

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
