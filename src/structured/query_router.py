"""Deterministic structured query router combining FinancialSQLStore and PotSandbox."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from src.citation_formatter import dedupe_citations, dedupe_references
from src.metric_atoms import MetricAtom, MetricAtomStore, get_default_store
from src.query_plan import QueryPlan
from src.structured.pot_sandbox import PotCalculationError, PotSandbox, PotSecurityError, PotVariable, normalize_monetary_unit
from src.structured.sql_store import FinancialSQLStore
from src.text_normalization import normalize_text

_YEAR_RE = re.compile(r"(20\d{2})")
_THRESHOLD_RE = re.compile(
    r"(?:超过|大于|高于|达到|突破|不少于|低于|少于|不足)\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*(亿元|千万元|百万元|万元|千元|元)"
)
_TOP_K_RE = re.compile(r"(?:排名前|前)\s*(\d+)\s*(?:名|家|位)?")
_CHINESE_NUMS = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _extract_target_year(question: str, query_plan: Optional[QueryPlan] = None) -> Optional[str]:
    if query_plan is not None and query_plan.filters.year is not None:
        return str(query_plan.filters.year)
    years = _YEAR_RE.findall(question or "")
    return years[-1] if years else None


def _extract_top_k(question: str) -> Optional[int]:
    q = question or ""
    match = _TOP_K_RE.search(q)
    if match:
        return max(1, min(20, int(match.group(1))))
    cn_match = re.search(r"前\s*([一二两三四五六七八九十])\s*(?:名|家|位)", q)
    if cn_match:
        return _CHINESE_NUMS.get(cn_match.group(1), 5)
    if any(tok in q for tok in ("最高", "最大", "最低", "最小")):
        return 1
    return None


def _format_cny_as_yi(value_cny: float) -> str:
    return f"{value_cny / 1e8:.2f} 亿元"


def _row_to_citation(row: Dict[str, Any]) -> Dict[str, Any]:
    snippet = (
        f"{row.get('company_name')} {row.get('period')}年 {row.get('metric_raw')}: "
        f"{row.get('raw_value')} {row.get('raw_unit')}（第{row.get('source_page')}页）"
    )
    return {
        "page": row.get("source_page"),
        "chunk_id": None,
        "chunk_type": "table",
        "node_type": "table",
        "parent_chunk_id": None,
        "matched_child_chunk_ids": [],
        "matched_tags": [],
        "section_title": row.get("statement_type"),
        "section_name": row.get("statement_type"),
        "report_section": row.get("statement_type"),
        "source": row.get("pdf_sha1") or row.get("doc_id"),
        "company_name": row.get("company_name"),
        "security_code": row.get("stock_code"),
        "stock_code": row.get("stock_code"),
        "currency": row.get("currency"),
        "report_year": row.get("report_year"),
        "table_id": row.get("table_id"),
        "row_idx": row.get("row_idx"),
        "matched_row_headers": [row.get("metric_raw")] if row.get("metric_raw") else [],
        "matched_col_headers": [row.get("col_header")] if row.get("col_header") else [],
        "unit": row.get("raw_unit"),
        "evidence_type": "table",
        "has_table_context": True,
        "retrieval_sources": ["structured_sql"],
        "evidence_snippet": snippet,
        "score": 1.0,
    }


def _row_to_retrieval_result(row: Dict[str, Any]) -> Dict[str, Any]:
    cit = _row_to_citation(row)
    return {
        "distance": 1.0,
        "combined_score": 1.0,
        "ranking_score": 1.0,
        "page": row.get("source_page"),
        "text": cit["evidence_snippet"],
        "metadata": {
            "sha1_name": row.get("pdf_sha1") or row.get("doc_id"),
            "company_name": row.get("company_name"),
            "stock_code": row.get("stock_code"),
            "report_year": row.get("report_year"),
            "currency": row.get("currency"),
            "chunk_type": "table",
            "node_type": "table",
            "table_id": row.get("table_id"),
        },
        "chunk_type": "table",
        "retrieval_sources": ["structured_sql"],
        "result_scope": "table",
    }


class StructuredQueryRouter:
    """Fail-closed SQL + PoT fast-track for deterministic financial aggregations."""

    def __init__(
        self,
        store: FinancialSQLStore,
        sandbox: Optional[PotSandbox] = None,
        atom_store: Optional[MetricAtomStore] = None,
    ):
        self.store = store
        self.sandbox = sandbox or PotSandbox()
        self.atom_store = atom_store or get_default_store()

    def _resolve_atom(self, question: str, query_plan: Optional[QueryPlan]) -> Optional[MetricAtom]:
        if query_plan is not None and isinstance(query_plan.metric_atom, dict):
            atom_id = query_plan.metric_atom.get("atom_id")
            if atom_id:
                found = self.atom_store.get(str(atom_id))
                if found is not None:
                    return found
        return self.atom_store.find_metric(normalize_text(question))

    @staticmethod
    def _build_answer_payload(
        *,
        question: str,
        final_answer: Any,
        rows: List[Dict[str, Any]],
        query_plan: Optional[QueryPlan],
        route_mode: str,
        reasoning: str,
        sql_debug: Dict[str, Any],
        pot_result_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        citations = dedupe_citations(_row_to_citation(r) for r in rows)
        references = dedupe_references(
            {"pdf_sha1": r.get("pdf_sha1") or r.get("doc_id"), "page": r.get("source_page")}
            for r in rows
            if r.get("source_page") is not None
        )
        retrieval_results = [_row_to_retrieval_result(r) for r in rows]
        pages = list(dict.fromkeys(int(r["source_page"]) for r in rows if r.get("source_page") is not None))
        route_info = {
            "route_mode": route_mode,
            "selected_company": rows[0].get("company_name") if len({r.get("company_name") for r in rows}) == 1 and rows else None,
            "candidate_companies": list(dict.fromkeys(str(r.get("company_name")) for r in rows if r.get("company_name"))),
            "structured_sql": sql_debug,
        }
        if pot_result_dict is not None:
            route_info["pot_result"] = pot_result_dict
        return {
            "final_answer": final_answer,
            "relevant_pages": pages,
            "references": references,
            "citations": citations,
            "confidence": "high",
            "confidence_reason": "结构化 SQL 事实库与 PoT 沙箱确定性计算生成。",
            "reasoning_summary": reasoning,
            "step_by_step_analysis": reasoning,
            "search_queries": query_plan.search_queries if query_plan else [question],
            "query_plan": query_plan.to_dict() if query_plan else {},
            "route_info": route_info,
            "structured_sql_result": sql_debug,
            "pot_result": pot_result_dict,
            "candidate_pool_size_before_rerank": None,
            "reranking_strategy": "none",
            "initial_candidate_pool_size": None,
            "colbert_candidate_pool_size": None,
            "colbert_top_n": None,
            "final_reranking_backend": "none",
            "hyde": {"enabled": False, "triggered": False},
            "retrieval_pages": pages,
            "retrieval_results": retrieval_results,
            "retrieval_report_groups": [],
            "response_data": {},
            "validation_flags": [],
        }

    def try_answer(
        self,
        *,
        question: str,
        schema: str,
        query_plan: Optional[QueryPlan] = None,
        companies: Optional[Sequence[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        atom = self._resolve_atom(question, query_plan)
        if atom is None:
            return None

        target_year = _extract_target_year(question, query_plan)
        period = target_year or "2024"
        company_list = [c for c in (companies or []) if c]
        q_text = question or ""
        schema_lower = (schema or "").lower()

        # 1. Cross-company comparison (>=2 companies)
        if len(company_list) >= 2:
            return self._try_compare(
                question=q_text,
                schema=schema_lower,
                companies=company_list,
                atom=atom,
                period=period,
                query_plan=query_plan,
            )

        # 2. Single-company YoY growth rate
        if len(company_list) == 1 and any(tok in q_text for tok in ("同比", "增长率", "增速", "增幅")):
            return self._try_yoy_growth(
                question=q_text,
                schema=schema_lower,
                company=company_list[0],
                atom=atom,
                period=period,
                query_plan=query_plan,
            )

        # 3. Single-company threshold check (boolean)
        if len(company_list) == 1 and schema_lower == "boolean":
            thresh_match = _THRESHOLD_RE.search(q_text)
            if thresh_match and "是否" in q_text:
                return self._try_single_threshold(
                    question=q_text,
                    company=company_list[0],
                    atom=atom,
                    period=period,
                    thresh_match=thresh_match,
                    query_plan=query_plan,
                )

        # 4. Cross-company ranking / threshold (when no single company is locked)
        if not company_list:
            top_k = _extract_top_k(q_text)
            if top_k is not None and any(tok in q_text for tok in ("排名", "最高", "最大", "最低", "最小", "前")):
                return self._try_ranking(
                    question=q_text,
                    schema=schema_lower,
                    atom=atom,
                    period=period,
                    top_k=top_k,
                    query_plan=query_plan,
                )
            thresh_match = _THRESHOLD_RE.search(q_text)
            if thresh_match and any(tok in q_text for tok in ("哪些公司", "多少家", "几家")):
                return self._try_cross_threshold(
                    question=q_text,
                    schema=schema_lower,
                    atom=atom,
                    period=period,
                    thresh_match=thresh_match,
                    query_plan=query_plan,
                )

        return None

    def _try_yoy_growth(
        self,
        *,
        question: str,
        schema: str,
        company: str,
        atom: MetricAtom,
        period: str,
        query_plan: Optional[QueryPlan],
    ) -> Optional[Dict[str, Any]]:
        try:
            prev_period = str(int(period) - 1)
        except ValueError:
            return None
        cur_rows = self.store.query_company_metric(
            company_name=company,
            metric_atom_id=atom.atom_id,
            period=period,
            canonical_name=atom.standard_name,
        )
        prev_rows = self.store.query_company_metric(
            company_name=company,
            metric_atom_id=atom.atom_id,
            period=prev_period,
            canonical_name=atom.standard_name,
        )
        if not cur_rows or not prev_rows:
            return None
        cur_row, prev_row = cur_rows[0], prev_rows[0]
        if prev_row["normalized_cny"] in (None, 0.0) or cur_row["normalized_cny"] is None:
            return None

        try:
            pot_res = self.sandbox.evaluate(
                variables={
                    "v_cur": PotVariable(
                        "v_cur",
                        float(cur_row["raw_value"]),
                        str(cur_row["raw_unit"]),
                        page=cur_row["source_page"],
                        doc_id=cur_row["doc_id"],
                    ),
                    "v_prev": PotVariable(
                        "v_prev",
                        float(prev_row["raw_value"]),
                        str(prev_row["raw_unit"]),
                        page=prev_row["source_page"],
                        doc_id=prev_row["doc_id"],
                    ),
                },
                expression="(v_cur - v_prev) / abs(v_prev) * 100",
                result_unit="%",
                ndigits=2,
            )
        except (PotSecurityError, PotCalculationError):
            return None

        final_val: Any = pot_res.value if schema == "number" else f"{pot_res.value:.2f}%"
        reasoning = (
            f"SQL 读取 {company} {period}年与{prev_period}年 {atom.standard_name}"
            f"（{cur_row['raw_value']} {cur_row['raw_unit']} vs {prev_row['raw_value']} {prev_row['raw_unit']}），"
            f"经 PoT 沙箱算得同比增速为 {pot_res.value:.2f}%。"
        )
        return self._build_answer_payload(
            question=question,
            final_answer=final_val,
            rows=[cur_row, prev_row],
            query_plan=query_plan,
            route_mode="structured_sql_pot_yoy",
            reasoning=reasoning,
            sql_debug={"mode": "yoy_growth", "atom_id": atom.atom_id, "periods": [period, prev_period]},
            pot_result_dict={"expression": pot_res.expression, "value": pot_res.value, "unit": pot_res.unit},
        )

    def _try_compare(
        self,
        *,
        question: str,
        schema: str,
        companies: Sequence[str],
        atom: MetricAtom,
        period: str,
        query_plan: Optional[QueryPlan],
    ) -> Optional[Dict[str, Any]]:
        if not any(tok in question for tok in ("谁", "哪", "更高", "高于", "超过", "大于", "差", "多多少")):
            return None
        ordered = list(companies)[:2]
        by_company = self.store.query_compare_companies(
            ordered,
            metric_atom_id=atom.atom_id,
            period=period,
            canonical_name=atom.standard_name,
        )
        rows = [by_company[c] for c in ordered]
        if any(r is None or r.get("normalized_cny") is None for r in rows):
            return None
        first, second = rows[0], rows[1]
        assert first is not None and second is not None

        try:
            diff_res = self.sandbox.evaluate(
                variables={
                    "a": PotVariable("a", float(first["raw_value"]), str(first["raw_unit"]), page=first["source_page"]),
                    "b": PotVariable("b", float(second["raw_value"]), str(second["raw_unit"]), page=second["source_page"]),
                },
                expression="a - b",
                result_unit="亿元",
                ndigits=2,
            )
        except (PotSecurityError, PotCalculationError):
            return None

        if "谁" in question or "哪" in question:
            final_answer: Any = ordered[0] if diff_res.value >= 0 else ordered[1]
        elif "是否" in question and any(tok in question for tok in ("高于", "超过", "大于")):
            final_answer = bool(diff_res.value > 0)
        elif schema == "number" and any(tok in question for tok in ("差", "多多少")):
            final_answer = abs(diff_res.value)
        else:
            winner = ordered[0] if diff_res.value >= 0 else ordered[1]
            final_answer = (
                f"{winner} 更高（{ordered[0]}: {_format_cny_as_yi(float(first['normalized_cny']))}，"
                f"{ordered[1]}: {_format_cny_as_yi(float(second['normalized_cny']))}，差值 {abs(diff_res.value):.2f} 亿元）"
            )

        reasoning = (
            f"SQL 结构化对比 {ordered[0]}（{first['raw_value']} {first['raw_unit']}）与 "
            f"{ordered[1]}（{second['raw_value']} {second['raw_unit']}）的 {period}年 {atom.standard_name}，"
            f"差值为 {diff_res.value:.2f} 亿元。"
        )
        return self._build_answer_payload(
            question=question,
            final_answer=final_answer,
            rows=[first, second],
            query_plan=query_plan,
            route_mode="structured_sql_compare",
            reasoning=reasoning,
            sql_debug={"mode": "compare", "atom_id": atom.atom_id, "period": period, "companies": ordered},
            pot_result_dict={"expression": diff_res.expression, "value": diff_res.value, "unit": diff_res.unit},
        )

    def _try_single_threshold(
        self,
        *,
        question: str,
        company: str,
        atom: MetricAtom,
        period: str,
        thresh_match: re.Match,
        query_plan: Optional[QueryPlan],
    ) -> Optional[Dict[str, Any]]:
        rows = self.store.query_company_metric(
            company_name=company,
            metric_atom_id=atom.atom_id,
            period=period,
            canonical_name=atom.standard_name,
        )
        if not rows or rows[0].get("normalized_cny") is None:
            return None
        row = rows[0]
        raw_thresh = float(thresh_match.group(1).replace(",", ""))
        thresh_unit = normalize_monetary_unit(thresh_match.group(2))
        is_lt = any(tok in thresh_match.group(0) for tok in ("低于", "少于", "不足"))

        try:
            pot_res = self.sandbox.evaluate(
                variables={
                    "val": PotVariable("val", float(row["raw_value"]), str(row["raw_unit"]), page=row["source_page"]),
                    "thresh": PotVariable("thresh", raw_thresh, thresh_unit),
                },
                expression="thresh - val" if is_lt else "val - thresh",
                result_unit="元",
            )
        except (PotSecurityError, PotCalculationError):
            return None

        final_answer = bool(pot_res.value > 0)
        reasoning = (
            f"SQL 读取 {company} {period}年 {atom.standard_name}={row['raw_value']} {row['raw_unit']}，"
            f"经 PoT 统一换算并与阈值 {raw_thresh} {thresh_unit} 比较，结果为 {final_answer}。"
        )
        return self._build_answer_payload(
            question=question,
            final_answer=final_answer,
            rows=[row],
            query_plan=query_plan,
            route_mode="structured_sql_threshold",
            reasoning=reasoning,
            sql_debug={"mode": "single_threshold", "atom_id": atom.atom_id, "period": period},
            pot_result_dict={"expression": pot_res.expression, "value": pot_res.value, "unit": pot_res.unit},
        )

    def _try_ranking(
        self,
        *,
        question: str,
        schema: str,
        atom: MetricAtom,
        period: str,
        top_k: int,
        query_plan: Optional[QueryPlan],
    ) -> Optional[Dict[str, Any]]:
        ascending = any(tok in question for tok in ("最低", "最小", "倒数"))
        industry = query_plan.filters.industry_l1 if query_plan is not None else None
        rows = self.store.query_company_ranking(
            metric_atom_id=atom.atom_id,
            period=period,
            top_k=top_k,
            ascending=ascending,
            industry_l1=industry,
            canonical_name=atom.standard_name,
        )
        if not rows:
            return None
        names = [str(r["company_name"]) for r in rows]
        if schema == "name":
            final_answer: Any = names[0]
        elif schema == "names":
            final_answer = names
        else:
            final_answer = "；".join(
                f"{idx}. {r['company_name']}（{_format_cny_as_yi(float(r['normalized_cny']))}）"
                for idx, r in enumerate(rows, start=1)
            )
        reasoning = f"SQL 按 {period}年 {atom.standard_name} {'升序' if ascending else '降序'} 排名返回前 {len(rows)} 家公司。"
        return self._build_answer_payload(
            question=question,
            final_answer=final_answer,
            rows=rows,
            query_plan=query_plan,
            route_mode="structured_sql_ranking",
            reasoning=reasoning,
            sql_debug={"mode": "ranking", "atom_id": atom.atom_id, "period": period, "top_k": top_k},
        )

    def _try_cross_threshold(
        self,
        *,
        question: str,
        schema: str,
        atom: MetricAtom,
        period: str,
        thresh_match: re.Match,
        query_plan: Optional[QueryPlan],
    ) -> Optional[Dict[str, Any]]:
        raw_thresh = float(thresh_match.group(1).replace(",", ""))
        thresh_unit = normalize_monetary_unit(thresh_match.group(2))
        op = "<" if any(tok in thresh_match.group(0) for tok in ("低于", "少于", "不足")) else ">"
        norm_thresh = self.sandbox.evaluate(
            variables={"t": PotVariable("t", raw_thresh, thresh_unit)},
            expression="t",
            result_unit="元",
        ).value
        industry = query_plan.filters.industry_l1 if query_plan is not None else None
        rows = self.store.query_threshold_companies(
            metric_atom_id=atom.atom_id,
            period=period,
            threshold_cny=norm_thresh,
            operator=op,
            industry_l1=industry,
            canonical_name=atom.standard_name,
        )
        names = [str(r["company_name"]) for r in rows]
        if schema == "number":
            final_answer: Any = len(names)
        elif schema == "names":
            final_answer = names
        else:
            final_answer = f"共 {len(names)} 家：{', '.join(names)}" if names else "共 0 家"
        reasoning = f"SQL 筛选 {period}年 {atom.standard_name} {op} {raw_thresh} {thresh_unit} 的公司，共命中 {len(names)} 家。"
        return self._build_answer_payload(
            question=question,
            final_answer=final_answer,
            rows=rows[:10],
            query_plan=query_plan,
            route_mode="structured_sql_cross_threshold",
            reasoning=reasoning,
            sql_debug={"mode": "cross_threshold", "atom_id": atom.atom_id, "period": period, "count": len(names)},
        )
