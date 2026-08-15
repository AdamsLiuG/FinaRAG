from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional

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

_CHINESE_NUMERAL_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}

_METRIC_PATTERNS: Dict[str, Dict[str, List[str]]] = {
    "扣非归母净利润": {
        "aliases": ["扣非归母净利润", "扣非净利润", "归母扣非净利润"],
        "evidence_targets": ["扣非归母净利润", "扣除非经常性损益后的净利润", "合并利润表"],
    },
    "归母净利润": {
        "aliases": ["归母净利润", "母公司股东净利润", "归属于母公司股东的净利润", "归母"],
        "evidence_targets": ["归母净利润", "归属于母公司股东的净利润", "合并利润表"],
    },
    "经营活动现金流量净额": {
        "aliases": ["经营活动现金流量净额", "经营现金流净额", "经营性现金流净额", "经营现金流"],
        "evidence_targets": ["经营活动现金流量净额", "现金流量表", "经营现金流"],
    },
    "营业收入": {
        "aliases": ["营业收入", "营收", "主营业务收入"],
        "evidence_targets": ["营业收入", "主营业务收入", "合并利润表"],
    },
    "货币资金": {
        "aliases": ["货币资金", "现金及现金等价物"],
        "evidence_targets": ["货币资金", "现金及现金等价物", "资产负债表"],
    },
    "短期借款": {
        "aliases": ["短期借款", "短债"],
        "evidence_targets": ["短期借款", "一年内到期的非流动负债", "流动负债"],
    },
    "总资产": {
        "aliases": ["总资产", "资产总计"],
        "evidence_targets": ["总资产", "资产总计", "资产负债表"],
    },
    "总负债": {
        "aliases": ["总负债", "负债合计"],
        "evidence_targets": ["总负债", "负债合计", "资产负债表"],
    },
    "研发投入": {
        "aliases": ["研发投入", "研发费用"],
        "evidence_targets": ["研发投入", "研发费用", "研发人员"],
    },
    "毛利率": {
        "aliases": ["毛利率", "综合毛利率"],
        "evidence_targets": ["毛利率", "综合毛利率", "主营业务收入"],
    },
    "资产负债率": {
        "aliases": ["资产负债率", "负债率"],
        "evidence_targets": ["资产负债率", "总资产", "总负债"],
    },
}

_IMPLICIT_INTENT_PATTERNS: Dict[str, Dict[str, Any]] = {
    "shareholder_return": {
        "patterns": ["股东回报", "对股东怎么样", "回报股东", "分红怎么样", "回购怎么样"],
        "evidence_targets": ["现金分红", "利润分配预案", "股份回购", "每10股派发"],
    },
    "liquidity_pressure": {
        "patterns": ["现金压力", "流动性压力", "偿债压力", "短期偿债", "资金压力"],
        "evidence_targets": ["经营活动现金流量净额", "货币资金", "短期借款", "资产负债率", "偿债能力"],
    },
    "operating_performance": {
        "patterns": ["经营状况", "经营情况", "经营表现", "业绩表现"],
        "evidence_targets": ["营业收入", "归母净利润", "毛利率", "经营活动现金流量净额"],
    },
    "growth_quality": {
        "patterns": ["成长性", "增长质量", "增长怎么样", "发展质量"],
        "evidence_targets": ["营业收入增长率", "净利润增长率", "研发投入", "新增订单"],
    },
    "risk_exposure": {
        "patterns": ["风险大不大", "主要风险", "风险暴露", "合规风险"],
        "evidence_targets": ["风险因素", "诉讼", "监管", "资产负债率", "减值"],
    },
}

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


@dataclass(frozen=True)
class IntentParseResult:
    task_type: str = "evidence_qa"
    route: str = "general_rag"
    operation: Optional[str] = None
    metric: Optional[str] = None
    scope: Dict[str, Any] = field(default_factory=dict)
    limit: Optional[int] = None
    sort_order: Optional[str] = None
    intent: Optional[str] = None
    evidence_targets: List[str] = field(default_factory=list)
    fallback_route: Optional[str] = None
    intent_confidence: float = 0.5

    def normalized_scope(self) -> Dict[str, Any]:
        return dict(self.scope or {})

    def normalized_evidence_targets(self) -> List[str]:
        return list(self.evidence_targets or [])


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
        intent_result = self._parse_intent(
            question=question,
            normalized_question=normalized_query,
            schema=schema,
            filters=filters,
            tag_filters=tag_filters,
            company_name=company_name,
            mentioned_companies=mentioned_companies,
        )

        expanded_queries: List[str] = []
        self._append_query(expanded_queries, question.strip())
        self._append_query(expanded_queries, normalized_query)

        for intent_query in self._intent_search_queries(question, filters, intent_result):
            self._append_query(expanded_queries, intent_query)

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
            task_type=intent_result.task_type,
        )

        expected_answer_type = kind_to_expected_answer_type(schema)
        if intent_result.task_type == "metric_aggregation" and intent_result.operation == "top_k":
            expected_answer_type = "entity_list_with_values"

        scope = intent_result.normalized_scope()
        evidence_targets = intent_result.normalized_evidence_targets()
        route_hints = {
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
            "task_type": intent_result.task_type,
            "route": intent_result.route,
            "operation": intent_result.operation,
            "metric": intent_result.metric,
            "scope": scope,
            "limit": intent_result.limit,
            "sort_order": intent_result.sort_order,
            "implicit_intent": intent_result.intent,
            "evidence_targets": evidence_targets,
            "fallback_route": intent_result.fallback_route,
            "intent_confidence": intent_result.intent_confidence,
        }

        return QueryPlan(
            original_query=question,
            normalized_query=normalized_query,
            search_queries=search_queries,
            filters=filters,
            route_mode="explicit_company" if company_name else "document_catalog",
            expected_answer_type=expected_answer_type,
            topic_flags=topic_flags,
            mentioned_companies=list(mentioned_companies or ([company_name] if company_name else [])),
            route_hints=route_hints,
            task_type=intent_result.task_type,
            route=intent_result.route,
            operation=intent_result.operation,
            metric=intent_result.metric,
            scope=scope,
            limit=intent_result.limit,
            sort_order=intent_result.sort_order,
            intent=intent_result.intent,
            evidence_targets=evidence_targets,
            fallback_route=intent_result.fallback_route,
            intent_confidence=intent_result.intent_confidence,
        )

    def _parse_intent(
        self,
        *,
        question: str,
        normalized_question: str,
        schema: str,
        filters: RetrievalFilters,
        tag_filters: Dict[str, List[str]],
        company_name: Optional[str],
        mentioned_companies: Optional[List[str]],
    ) -> IntentParseResult:
        metric, metric_targets = self._extract_metric(normalized_question)
        implicit_intent, implicit_targets = self._extract_implicit_intent(normalized_question)
        limit = self._extract_limit(question)
        sort_order = self._extract_sort_order(normalized_question)
        operation = self._extract_operation(
            question=question,
            normalized_question=normalized_question,
            schema=schema,
            metric=metric,
            limit=limit,
            mentioned_companies=mentioned_companies,
        )
        scope = self._build_scope(filters, company_name=company_name, mentioned_companies=mentioned_companies)

        if metric and operation in {"top_k", "count", "filter"}:
            return IntentParseResult(
                task_type="metric_aggregation",
                route="metric_fact_store",
                operation=operation,
                metric=metric,
                scope=scope,
                limit=limit,
                sort_order=sort_order,
                evidence_targets=metric_targets,
                fallback_route="table_grounding_then_rag",
                intent_confidence=0.9 if operation == "top_k" else 0.84,
            )

        if self._is_metadata_aggregation_intent(question, normalized_question, schema, tag_filters):
            metadata_operation = self._metadata_operation(question, schema)
            labels = self._metadata_labels(tag_filters, question)
            return IntentParseResult(
                task_type="metadata_aggregation",
                route="metadata_store",
                operation=metadata_operation,
                scope=scope,
                evidence_targets=labels,
                fallback_route="general_rag",
                intent_confidence=0.84,
            )

        if operation == "compare":
            return IntentParseResult(
                task_type="comparison",
                route="metric_fact_store_plus_rag" if metric else "hybrid_multi_hop_retrieval",
                operation="compare",
                metric=metric,
                scope=scope,
                sort_order=sort_order,
                evidence_targets=metric_targets,
                fallback_route="general_rag",
                intent_confidence=0.82 if metric else 0.72,
            )

        if metric:
            return IntentParseResult(
                task_type="single_metric_lookup",
                route="table_grounding",
                operation="lookup",
                metric=metric,
                scope=scope,
                sort_order=sort_order,
                evidence_targets=metric_targets,
                fallback_route="general_rag",
                intent_confidence=0.82,
            )

        if implicit_intent:
            return IntentParseResult(
                task_type="implicit_intent_qa",
                route="metric_fact_store_plus_rag",
                operation="assess",
                scope=scope,
                intent=implicit_intent,
                evidence_targets=implicit_targets,
                fallback_route="general_rag",
                intent_confidence=0.86,
            )

        if self._is_summary_question(normalized_question):
            return IntentParseResult(
                task_type="summary",
                route="general_rag",
                operation="summarize",
                scope=scope,
                fallback_route=None,
                intent_confidence=0.68,
            )

        return IntentParseResult(
            task_type="evidence_qa",
            route="general_rag",
            operation=None,
            scope=scope,
            fallback_route=None,
            intent_confidence=0.55,
        )

    def _extract_metric(self, normalized_question: str) -> tuple[Optional[str], List[str]]:
        for metric, config in _METRIC_PATTERNS.items():
            aliases = config.get("aliases") or []
            if any(normalize_text(alias) in normalized_question for alias in aliases):
                return metric, list(config.get("evidence_targets") or [metric])
        return None, []

    def _extract_implicit_intent(self, normalized_question: str) -> tuple[Optional[str], List[str]]:
        for intent, config in _IMPLICIT_INTENT_PATTERNS.items():
            patterns = config.get("patterns") or []
            if any(normalize_text(pattern) in normalized_question for pattern in patterns):
                return intent, list(config.get("evidence_targets") or [])
        return None, []

    def _extract_operation(
        self,
        *,
        question: str,
        normalized_question: str,
        schema: str,
        metric: Optional[str],
        limit: Optional[int],
        mentioned_companies: Optional[List[str]],
    ) -> Optional[str]:
        ranking_terms = ("最高", "最大", "最多", "最低", "最小", "最少", "排名", "排行", "top")
        if limit is not None and any(token in normalized_question for token in ranking_terms):
            return "top_k"
        if self._looks_like_count_question(question, schema):
            return "count"
        comparison_terms = ("谁更", "哪家更", "哪个更", "比较", "对比", "高于", "低于", "超过", "大于", "小于")
        if (
            (mentioned_companies and len(mentioned_companies) > 1)
            or any(token in normalized_question for token in comparison_terms)
        ):
            return "compare"
        if metric and any(token in normalized_question for token in ("哪些公司", "公司有哪些", "筛选", "符合")):
            return "filter"
        return "lookup" if metric else None

    @staticmethod
    def _looks_like_count_question(question: str, schema: str) -> bool:
        if (schema or "").lower() != "number":
            return False
        return any(term in (question or "") for term in ("共有几家", "多少家", "几家公司", "数量", "公司数"))

    @staticmethod
    def _extract_sort_order(normalized_question: str) -> Optional[str]:
        if any(token in normalized_question for token in ("最低", "最小", "最少", "升序", "从低到高")):
            return "asc"
        if any(token in normalized_question for token in ("最高", "最大", "最多", "前", "top", "降序", "从高到低")):
            return "desc"
        return None

    def _extract_limit(self, question: str) -> Optional[int]:
        text = question or ""
        patterns = [
            r"(?:top|TOP)\s*([0-9]{1,3})",
            r"前\s*([0-9]{1,3}|[一二两三四五六七八九十]{1,3})\s*(?:家|名|个|位|只|项)?",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            value = self._parse_positive_int(match.group(1))
            if value is not None and value > 0:
                return value
        return None

    @staticmethod
    def _parse_positive_int(token: str) -> Optional[int]:
        token = (token or "").strip()
        if not token:
            return None
        if token.isdigit():
            return int(token)
        if token == "十":
            return 10
        if "十" in token:
            left, _, right = token.partition("十")
            tens = _CHINESE_NUMERAL_DIGITS.get(left, 1 if left == "" else 0)
            ones = _CHINESE_NUMERAL_DIGITS.get(right, 0) if right else 0
            value = tens * 10 + ones
            return value or None
        if len(token) == 1:
            return _CHINESE_NUMERAL_DIGITS.get(token)
        return None

    @staticmethod
    def _build_scope(
        filters: RetrievalFilters,
        *,
        company_name: Optional[str],
        mentioned_companies: Optional[List[str]],
    ) -> Dict[str, Any]:
        scope: Dict[str, Any] = {}
        if company_name:
            scope["company_name"] = company_name
        if mentioned_companies:
            scope["companies"] = list(mentioned_companies)
        if filters.industry_l1:
            scope["industry"] = filters.industry_l1
            scope["industry_l1"] = filters.industry_l1
        for key in ("exchange", "board", "market_type", "chain_position_major"):
            value = getattr(filters, key)
            if value:
                scope[key] = value
        for key in ("business_tags", "strategy_tags", "factor_tags", "listing_tags", "status_tags"):
            values = getattr(filters, key)
            if values:
                scope[key] = list(values)
        return scope

    @staticmethod
    def _metadata_labels(tag_filters: Dict[str, List[str]], question: str) -> List[str]:
        labels: List[str] = []
        for field in ("strategy_tags", "business_tags", "factor_tags", "listing_tags", "status_tags"):
            for value in tag_filters.get(field) or []:
                if value not in labels:
                    labels.append(value)
        for quoted in re.findall(r"[“\"']([^”\"']{2,30})[”\"']", question or ""):
            if quoted not in labels:
                labels.append(quoted)
        return labels

    def _is_metadata_aggregation_intent(
        self,
        question: str,
        normalized_question: str,
        schema: str,
        tag_filters: Dict[str, List[str]],
    ) -> bool:
        if not self._metadata_labels(tag_filters, question):
            return False
        if self._looks_like_count_question(question, schema):
            return True
        names_terms = ("哪些公司", "公司有哪些", "名单")
        if (schema or "").lower() == "names" and any(token in normalized_question for token in names_terms):
            return True
        if (schema or "").lower() == "boolean" and "名单" in normalized_question:
            return True
        return False

    @staticmethod
    def _metadata_operation(question: str, schema: str) -> str:
        if QuestionRewriter._looks_like_count_question(question, schema):
            return "count"
        if (schema or "").lower() == "boolean":
            return "membership"
        return "list"

    @staticmethod
    def _is_summary_question(normalized_question: str) -> bool:
        return any(token in normalized_question for token in ("总结", "概括", "归纳", "未来战略", "主要战略", "主要风险", "公司未来"))

    def _intent_search_queries(
        self,
        question: str,
        filters: RetrievalFilters,
        intent_result: IntentParseResult,
    ) -> List[str]:
        targets = intent_result.normalized_evidence_targets()
        if not targets:
            return []
        query_parts: List[str] = []
        if intent_result.metric:
            query_parts.append(intent_result.metric)
        query_parts.extend(targets)
        if filters.year is not None:
            query_parts.append(str(filters.year))
        if filters.period:
            query_parts.append(filters.period)
        evidence_query = " ".join(dedupe_preserve_order(query_parts))
        if intent_result.task_type == "implicit_intent_qa":
            return [evidence_query, f"{question.strip()} {evidence_query}"]
        return [evidence_query]

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
        matches = re.findall(r"([A-Za-z0-9\u4e00-\u9fff·、，,（）() -]{2,40}?)(?:行业|板块)", question)
        if matches:
            candidate = matches[0].strip(" ，,、")
            candidate = re.sub(r"^(?:在|于|属于|的)+", "", candidate).strip(" ，,、")
            candidate = re.sub(r"^(?:19|20)\d{2}年?", "", candidate).strip(" ，,、")
            return candidate or None
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
        task_type: str = "evidence_qa",
    ) -> List[str]:
        if not expanded_queries:
            return []

        if company_name:
            max_queries = 2 if schema in {"name", "boolean"} else 3
        else:
            max_queries = 4
        if task_type in {"implicit_intent_qa", "metric_aggregation", "comparison"}:
            max_queries = max(max_queries, 4)

        return expanded_queries[:max_queries]


QueryRewriteResult = QueryPlan
