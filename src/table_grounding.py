from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from src.logical_table_merger import LogicalTableMerger
from src.metric_atoms import get_default_store
from src.retrieval_filters import RetrievalFilters
from src.text_normalization import normalize_text, parse_numeric_value, tokenize_for_bm25


_TARGET_UNIT_FACTORS = {
    "亿元": 1e8,
    "万元": 1e4,
    "百万元": 1e6,
    "千万元": 1e7,
    "元": 1.0,
}

_BAD_SEGMENT_COL_TERMS = (
    "公司业务",
    "个人业务",
    "资金业务",
    "公司银行业务",
    "零售银行业务",
    "金融市场业务",
    "未分配项目",
    "其他业务",
)
_NARRATIVE_VALUE_RE = re.compile(r"[\u4e00-\u9fff]{7,}")
_NUMERIC_LIKE_RE = re.compile(
    r"^\s*[\(（]?\s*[-+]?\d[\d,，]*(?:\.\d+)?\s*[\)）]?\s*(?:%|％|元|万元|亿元|千元|百万元|千万元)?\s*$"
)
_UNIT_PRIORITY = (
    "人民币百万元",
    "百万元",
    "人民币千万元",
    "千万元",
    "人民币千元",
    "千元",
    "人民币万元",
    "万元人民币",
    "万元",
    "亿元",
    "人民币元",
    "元",
    "%",
    "％",
)
_STRONG_CONTEXT_UNITS = (
    "人民币百万元",
    "百万元",
    "人民币千元",
    "千元",
)
_OCR_NORMALIZATION_MAP = str.maketrans(
    {
        "⼈": "人",
        "⼊": "入",
        "⾏": "行",
        "⾦": "金",
        "⽀": "支",
        "⼆": "二",
        "⼀": "一",
    }
)


def _score_overlap(query_tokens: List[str], text: str) -> float:
    if not query_tokens or not text:
        return 0.0
    normalized = normalize_text(text)
    return float(sum(1 for token in query_tokens if token and token in normalized))


def _target_unit_from_question(question: str) -> Optional[str]:
    question = question or ""
    for unit in ("百万元", "千万元", "亿元", "万元"):
        if unit in question:
            return unit
    if "元" in question and not any(currency_unit in question for currency_unit in ("美元", "港元", "欧元", "日元")):
        return "元"
    return None


def _compact_text(text: str) -> str:
    return normalize_text((text or "").translate(_OCR_NORMALIZATION_MAP)).replace(" ", "")


def _infer_unit_hint(raw_unit_hint: object, table_context: str) -> Optional[str]:
    raw_unit = str(raw_unit_hint or "").strip()
    if "%" in raw_unit or "％" in raw_unit:
        return "%"

    compact_context = _compact_text(table_context[:3000])
    raw_compact = _compact_text(raw_unit)
    strong_context_unit = next((unit for unit in _STRONG_CONTEXT_UNITS if unit in compact_context), None)
    if strong_context_unit and (
        not raw_compact
        or raw_compact in {"元", "万元", "人民币万元", "万元人民币"}
        or (raw_compact == "千元" and strong_context_unit in {"人民币千元", "千元"})
    ):
        return strong_context_unit.replace("％", "%")

    compact = _compact_text(f"{raw_unit} {table_context[:3000]}")
    for unit in _UNIT_PRIORITY:
        if unit in compact:
            return unit.replace("％", "%")
    return raw_unit or None


# 通用取证守卫词表（与指标无关，对所有指标生效）：衍生列/变动列
_UNIVERSAL_BAD_COL_TERMS = (
    "同比",
    "变动",
    "比例",
    "占比",
    "百分比",
    "本期比上年同期",
    "期末余额",
    "期初",
)
# 口径维度守卫：表格限定了某维度而问题未提及时否决（对称条件）
_SCOPE_DIMENSION_TERMS = ("分部", "地区", "产品", "关联方", "客户")


def _has_grounding_rule(question: str) -> bool:
    """问题是否命中带取证规则的指标原子（取代 _is_revenue_question 硬门控）。"""
    return get_default_store().grounding_for(question) is not None


def _universal_row_guard(question: str, row_text: str, col_text: str, unit_kind: str = "currency") -> bool:
    """通用取证守卫：期间粒度/主体范围/口径维度/单位/衍生列。

    全部为“表格限定了某条件而问题未提及则否决”的**对称条件**，不引入单边否决；
    原先被囚在营收专用函数内，其余指标一条都用不上。返回 False 表示否决。
    """
    normalized_question = normalize_text(question)
    row_normalized = normalize_text(row_text)
    col_normalized = normalize_text(col_text)
    combined = f"{row_normalized} {col_normalized}"
    has_year_col = bool(re.search(r"20\d{2}", col_normalized))

    if any(term in col_normalized for term in _UNIVERSAL_BAD_COL_TERMS):
        return False
    if "增减" in col_normalized and not has_year_col:
        return False
    if "季度" in col_normalized and "季度" not in normalized_question:
        return False
    if unit_kind == "currency" and "%" not in normalized_question and (
        "%" in col_normalized or "百分点" in col_normalized
    ):
        return False
    if "母公司" in combined and "母公司" not in normalized_question:
        return False
    if "本行" in combined and "本行" not in normalized_question:
        return False
    if any(term in combined for term in _SCOPE_DIMENSION_TERMS) and not any(
        term in normalized_question for term in _SCOPE_DIMENSION_TERMS
    ):
        return False
    return True


def _segment_col_guard(col_text: str) -> bool:
    """分部列守卫（按原子 segment_guard 开关启用）。"""
    col_normalized = normalize_text(col_text)
    return not (
        any(term in col_normalized for term in _BAD_SEGMENT_COL_TERMS) and "合计" not in col_normalized
    )


def _consolidated_bonus(col_text: str) -> float:
    """合并口径加成（通用）：列名明确为本集团/合计时加分。"""
    col_normalized = normalize_text(col_text)
    return 2.0 if ("本集团" in col_normalized or "合计" in col_normalized) else 0.0


def _atom_unit_kind(atom) -> str:
    unit = str(getattr(atom, "unit", "") or "")
    return "percent" if "%" in unit else "currency"


def _atom_row_score(question: str, row_text: str, col_text: str, raw_value: object = None) -> Optional[float]:
    """原子驱动的行打分（取代 _revenue_row_score）。

    指标专属词表来自 `atom.grounding`，通用守卫对所有指标生效；
    未命中带规则的原子时返回 0.0（与原门控行为一致）。
    """
    hit = get_default_store().grounding_for(question)
    if hit is None:
        return 0.0
    atom, rule = hit

    row_normalized = normalize_text(row_text)
    col_normalized = normalize_text(col_text)
    combined = f"{row_normalized} {col_normalized}"

    if rule.match_terms and not any(normalize_text(term) in combined for term in rule.match_terms):
        return None
    # bad_row_headers 查 combined（与原实现一致：衍生行名也可能出现在列上）
    if any(normalize_text(term) in combined for term in rule.bad_row_headers):
        return None
    if any(normalize_text(term) in col_normalized for term in rule.bad_col_headers):
        return None
    if not _universal_row_guard(question, row_text, col_text, unit_kind=_atom_unit_kind(atom)):
        return None
    if rule.segment_guard and not _segment_col_guard(col_text):
        return None

    # 分级给分：exact/secondary 为精确相等；low_priority/weak 为子串包含；
    # 合并口径加成仅作用于 exact/secondary（与原实现一致）。
    if row_normalized in {normalize_text(term) for term in rule.exact_row_headers}:
        return 8.0 + _consolidated_bonus(col_text)
    if row_normalized in {normalize_text(term) for term in rule.secondary_row_headers}:
        return 7.5 + _consolidated_bonus(col_text)
    if any(normalize_text(term) in row_normalized for term in rule.low_priority_row_headers):
        return 1.0
    if any(normalize_text(term) in row_normalized for term in rule.weak_row_headers):
        return 4.0
    return 1.5


def _atom_statement_score(question: str, table_context: str) -> float:
    """原子驱动的报表偏好打分（取代 _table_scope_score）。

    与 `_metric_statement_prior_score` 共存：后者按 statement_location 加分，两者对
    “合并利润表”存在**既有的重复加分**（原实现即如此）。在无端到端评测环境下
    保留该行为以保证等价性，待准确率可测后再调优权重。
    """
    hit = get_default_store().grounding_for(question)
    if hit is None:
        return 0.0
    _atom, rule = hit
    compact = _compact_text(table_context)
    score = 0.0
    for terms, points in rule.preferred_statements:
        if any(term in compact for term in terms):
            score += points
    for terms, points in rule.penalized_statements:
        if any(term in compact for term in terms):
            score += points
    return score


def _reject_unit_by_atom(question: str, unit_hint: Optional[str]) -> bool:
    """单位守卫（取代 _reject_revenue_unit）：按 atom.unit 判定，货币类指标拒百分比单位。"""
    hit = get_default_store().grounding_for(question)
    if hit is None:
        return False
    atom, _rule = hit
    if _atom_unit_kind(atom) != "currency":
        return False
    if "%" in (question or ""):
        return False
    unit_text = normalize_text(str(unit_hint or ""))
    return "%" in unit_text or "百分点" in unit_text


def _atom_nested_row_bonus(question: str, row_text: str, table_context: str) -> float:
    """嵌套行加成：指标作为上级科目子项（如“其中:营业收入”）且上下文含上级科目时加分。"""
    hit = get_default_store().grounding_for(question)
    if hit is None:
        return 0.0
    _atom, rule = hit
    if not rule.nested_row_bonus:
        return 0.0
    compact_row = _compact_text(row_text)
    compact_table = _compact_text(table_context)
    score = 0.0
    for row_prefix, requires_context, points in rule.nested_row_bonus:
        if compact_row.startswith(_compact_text(row_prefix)) and (
            not requires_context or _compact_text(requires_context) in compact_table
        ):
            score += points
    return score


def _atom_early_page_bonus(question: str, page: object) -> float:
    """页码先验：仅对显式配置了 early_page_bonus 的原子生效。

    未配置则不加分——否则给其他指标配上 grounding 后会误吃该加成（如货币资金在
    资产负债表而非前部摘要）。
    """
    hit = get_default_store().grounding_for(question)
    if hit is None or page is None:
        return 0.0
    _atom, rule = hit
    if not rule.early_page_bonus:
        return 0.0
    try:
        page_number = int(page)
    except (TypeError, ValueError):
        return 0.0
    for max_page, points in sorted(rule.early_page_bonus):
        if page_number <= max_page:
            return points
    return 0.0


def _atom_nested_row_bonus(question: str, row_text: str, table_context: str) -> float:
    """嵌套行加成：指标作为上级科目子项（如“其中:营业收入”）且上下文含上级科目时加分。"""
    hit = get_default_store().grounding_for(question)
    if hit is None:
        return 0.0
    _atom, rule = hit
    if not rule.nested_row_bonus:
        return 0.0
    compact_row = _compact_text(row_text)
    compact_table = _compact_text(table_context)
    score = 0.0
    for row_prefix, requires_context, points in rule.nested_row_bonus:
        if compact_row.startswith(_compact_text(row_prefix)) and (
            not requires_context or _compact_text(requires_context) in compact_table
        ):
            score += points
    return score


def _atom_early_page_bonus(question: str, page: object) -> float:
    """页码先验：仅对显式配置了 early_page_bonus 的原子生效。

    未配置则不加分——否则给其他指标配上 grounding 后会误吃该加成（如货币资金在
    资产负债表而非前部摘要）。
    """
    hit = get_default_store().grounding_for(question)
    if hit is None or page is None:
        return 0.0
    _atom, rule = hit
    if not rule.early_page_bonus:
        return 0.0
    try:
        page_number = int(page)
    except (TypeError, ValueError):
        return 0.0
    for max_page, points in sorted(rule.early_page_bonus):
        if page_number <= max_page:
            return points
    return 0.0


def _metric_statement_prior_score(question: str, table_context: str) -> float:
    """FinaWiki 取证先验：问题命中指标原子时，按其 statement_location 给匹配报表的表格加分。

    如归母净利润问题优先命中合并利润表/主要财务指标，而非分部信息表——
    从硬编码权重（_table_scope_score 仅覆盖营收）走向知识驱动的报表定位。
    """
    atom = get_default_store().find_metric(normalize_text(question))
    if atom is None or not atom.statement_location:
        return 0.0
    compact = _compact_text(table_context)
    keywords = [piece.strip() for piece in re.split(r"[/／]", atom.statement_location) if piece.strip()]
    for keyword in keywords:
        if keyword and keyword in compact:
            return 2.5
    # 弱命中：statement_location 为"合并利润表"时，"利润表"也视为报表类型匹配
    weak_keywords = [keyword.replace("合并", "") for keyword in keywords if keyword.startswith("合并")]
    for keyword in weak_keywords:
        if keyword and keyword in compact:
            return 1.5
    return 0.0


def _convert_to_target_unit(value: Optional[float], target_unit: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    factor = _TARGET_UNIT_FACTORS.get(target_unit or "")
    if not factor:
        return value
    return value / factor


def _is_valid_numeric_raw_value(raw_value: object, question: str) -> bool:
    text = str(raw_value or "").strip()
    if not text or not any(char.isdigit() for char in text):
        return False
    if len(text) > 48:
        return False
    if _NARRATIVE_VALUE_RE.search(text):
        return False
    if any(term in text for term in ("年", "年度", "报告期")):
        return False
    if "%" not in question and ("%" in text or "％" in text):
        return False
    parsed = parse_numeric_value(text)
    if parsed is None:
        return False
    return bool(_NUMERIC_LIKE_RE.match(text))


class TableGrounder:
    def __init__(self, documents_dir: Path):
        self.documents_dir = Path(documents_dir)
        self._documents_cache: Dict[str, Dict] = {}
        self.logical_table_merger = LogicalTableMerger()

    def _load_document(self, doc_id: str) -> Optional[Dict]:
        if doc_id in self._documents_cache:
            return self._documents_cache[doc_id]

        for path in self.documents_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            metainfo = payload.get("metainfo") or {}
            if str(metainfo.get("sha1_name") or metainfo.get("doc_id") or path.stem) == str(doc_id):
                self._documents_cache[doc_id] = payload
                return payload
        return None

    @staticmethod
    def _page_number(value: object) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _page_text(self, document: Dict, page: Optional[int]) -> str:
        if page is None:
            return ""
        for page_payload in (document.get("content") or {}).get("pages") or []:
            if self._page_number(page_payload.get("page")) == page:
                return str(page_payload.get("text") or "")
        return ""

    def _nearby_chunk_text(self, document: Dict, page: Optional[int], limit: int = 1600) -> str:
        if page is None:
            return ""
        pieces: List[str] = []
        for chunk in (document.get("content") or {}).get("chunks") or []:
            chunk_page = self._page_number(chunk.get("page") or chunk.get("page_start"))
            if chunk_page is None or abs(chunk_page - page) > 1:
                continue
            for key in ("section_title", "section_name", "report_section", "text"):
                value = chunk.get(key)
                if value:
                    pieces.append(str(value))
            if sum(len(piece) for piece in pieces) >= limit:
                break
        return "\n".join(pieces)[:limit]

    def _table_context(self, document: Dict, table: Dict, page: Optional[int]) -> str:
        pages = table.get("member_pages") or [page]
        parts = [
            str(table.get("markdown") or ""),
            str(table.get("caption") or ""),
            str(table.get("section_title") or ""),
            str(table.get("section_name") or ""),
            str(table.get("report_section") or ""),
        ]
        seen_pages = set()
        for member_page in pages:
            page_number = self._page_number(member_page)
            if page_number is None or page_number in seen_pages:
                continue
            seen_pages.add(page_number)
            parts.append(self._page_text(document, page_number))
            parts.append(self._nearby_chunk_text(document, page_number))
        return "\n".join(part for part in parts if part).translate(_OCR_NORMALIZATION_MAP)

    def _materialized_tables(self, document: Dict) -> List[Dict[str, Any]]:
        structured_tables = document.get("content", {}).get("structured_tables") or []
        logical_tables = document.get("content", {}).get("logical_tables") or []
        materialized: List[Dict[str, Any]] = []
        for logical_table in logical_tables:
            if str(logical_table.get("merge_state") or "") != "confirmed":
                continue
            if not bool(logical_table.get("materializable", False)):
                continue
            materialized_table = self.logical_table_merger.materialize_logical_table(
                structured_tables,
                logical_tables,
                logical_table.get("logical_table_id"),
            )
            if materialized_table is not None:
                materialized.append(materialized_table)
        return materialized

    @staticmethod
    def _same_value(left: Optional[float], right: Optional[float]) -> bool:
        if left is None or right is None:
            return False
        tolerance = max(1.0, abs(float(right)) * 1e-6)
        return abs(float(left) - float(right)) <= tolerance

    @staticmethod
    def _support_priority(match: Dict[str, Any]) -> Tuple[int, int, float]:
        compact_context = _compact_text(str(match.get("table_context") or match.get("table_snippet") or ""))
        page = match.get("page")
        try:
            page_number = int(page)
        except (TypeError, ValueError):
            page_number = 10_000
        priority = 0
        if any(term in compact_context for term in ("主要会计数据", "主要财务指标", "会计数据和财务指标摘要", "近三年主要会计数据和财务指标")):
            priority += 100
        if "公司简介和主要财务指标" in compact_context:
            priority += 20
        if "利润表" in compact_context:
            priority += 10
        if page_number <= 20:
            priority += 20
        return (-priority, page_number, -float(match.get("match_score") or 0.0))

    @staticmethod
    def _match_mentions_year(match: Dict[str, Any], year: Optional[int]) -> bool:
        if year is None:
            return True
        text = " ".join(
            str(piece or "")
            for piece in (
                " ".join(match.get("matched_row_headers") or []),
                " ".join(match.get("matched_col_headers") or []),
                match.get("period"),
                match.get("table_context"),
            )
        )
        return str(year) in normalize_text(text)

    def _build_match(
        self,
        *,
        doc_id: str,
        metainfo: Dict[str, Any],
        cell: Dict[str, Any],
        table_context: str,
        unit_hint: Optional[str],
        raw_value: object,
        score: float,
    ) -> Dict[str, Any]:
        normalized_value = parse_numeric_value(str(raw_value or ""), unit_hint=unit_hint)
        if normalized_value is None:
            normalized_value = cell.get("normalized_value")
        return {
            "source_doc_id": doc_id,
            "company_name": metainfo.get("company_name"),
            "security_code": metainfo.get("security_code") or metainfo.get("stock_code"),
            "currency": metainfo.get("currency"),
            "report_year": metainfo.get("report_year") or metainfo.get("fiscal_year"),
            "report_type": metainfo.get("report_type"),
            "doc_source_type": metainfo.get("doc_source_type"),
            "table_id": cell.get("table_id"),
            "source_table_id": cell.get("source_table_id") or cell.get("table_id"),
            "logical_table_id": cell.get("logical_table_id"),
            "logical_table_materialized": bool(cell.get("logical_table_materialized")),
            "materialized_from_tables": list(cell.get("materialized_from_tables") or []),
            "member_pages": list(cell.get("member_pages") or []),
            "page_span": list(cell.get("page_span") or []),
            "page": cell.get("page"),
            "source_page": cell.get("source_page") or cell.get("page"),
            "row_idx": cell.get("row_idx"),
            "source_row_idx": cell.get("source_row_idx") or cell.get("row_idx"),
            "col_idx": cell.get("col_idx"),
            "source_col_idx": cell.get("source_col_idx") or cell.get("col_idx"),
            "matched_row_headers": cell.get("matched_row_headers") or [],
            "matched_col_headers": cell.get("matched_col_headers") or [],
            "unit": unit_hint,
            "period": cell.get("period"),
            "raw_value": raw_value,
            "normalized_value": normalized_value,
            "footnote_refs": cell.get("footnote_refs") or [],
            "table_snippet": table_context[:320],
            "table_context": table_context[:2000],
            "match_score": round(score, 4),
        }

    def ground_number_query(
        self,
        question: str,
        retrieval_results: List[Dict],
        filters: RetrievalFilters,
        candidate_doc_ids: Optional[List[str]] = None,
        allowed_doc_ids: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        query_tokens = tokenize_for_bm25(question)
        if not query_tokens:
            return None

        candidate_pages = {result.get("page") for result in retrieval_results if result.get("page") is not None}
        target_unit = _target_unit_from_question(question)
        doc_ids = list(dict.fromkeys(str(doc_id) for doc_id in (allowed_doc_ids or candidate_doc_ids or []) if doc_id))
        if not doc_ids:
            doc_ids = list(
                dict.fromkeys(
                    str((result.get("metadata") or {}).get("sha1_name"))
                    for result in retrieval_results
                    if (result.get("metadata") or {}).get("sha1_name")
                )
            )

        best_match = None
        best_score = 0.0
        candidate_matches: List[Dict[str, Any]] = []
        for doc_id in doc_ids:
            document = self._load_document(doc_id)
            if not document:
                continue
            metainfo = document.get("metainfo") or {}
            if filters.company_name and metainfo.get("company_name"):
                if normalize_text(str(filters.company_name)) != normalize_text(str(metainfo.get("company_name"))):
                    continue
            if filters.security_code:
                observed_code = metainfo.get("security_code") or metainfo.get("stock_code")
                if observed_code and str(observed_code) != str(filters.security_code):
                    continue
            structured_tables = document.get("content", {}).get("structured_tables") or []
            grounding_tables = list(structured_tables) + self._materialized_tables(document)
            for table in grounding_tables:
                table_page = self._page_number(table.get("page"))
                table_context = self._table_context(document, table, table_page)
                for cell in table.get("cell_records") or []:
                    raw_value = cell.get("raw_value")
                    if not _is_valid_numeric_raw_value(raw_value, question):
                        continue
                    score = 0.0
                    row_text = " ".join(cell.get("matched_row_headers") or [])
                    col_text = " ".join(cell.get("matched_col_headers") or [])
                    unit_hint = _infer_unit_hint(cell.get("unit_hint"), table_context)
                    if _reject_unit_by_atom(question, unit_hint):
                        continue
                    row_score = _atom_row_score(question, row_text, col_text, raw_value=raw_value)
                    if row_score is None:
                        continue
                    score += row_score
                    score += _score_overlap(query_tokens, row_text) * 2.2
                    score += _score_overlap(query_tokens, col_text) * 1.8
                    score += _score_overlap(query_tokens, table_context) * 0.8
                    score += _atom_statement_score(question, table_context)
                    score += _metric_statement_prior_score(question, table_context)
                    score += _atom_nested_row_bonus(question, row_text, table_context)
                    score += _atom_early_page_bonus(question, cell.get("page"))

                    if cell.get("page") in candidate_pages:
                        score += 1.5
                    if filters.year is not None:
                        cell_period = normalize_text(str(cell.get("period") or table_context))
                        if str(filters.year) in cell_period:
                            score += 1.5
                    if filters.period:
                        cell_period = normalize_text(str(cell.get("period") or table_context))
                        if normalize_text(filters.period) in cell_period:
                            score += 1.0
                    if filters.currency:
                        unit_context = normalize_text(str(unit_hint or table_context))
                        if normalize_text(filters.currency) in unit_context or (
                            filters.currency == "CNY" and "人民币" in unit_context
                        ):
                            score += 0.6
                    if "%" in question and "%" in str(cell.get("raw_value")):
                        score += 0.5

                    match = self._build_match(
                        doc_id=doc_id,
                        metainfo=metainfo,
                        cell=cell,
                        table_context=table_context,
                        unit_hint=unit_hint,
                        raw_value=raw_value,
                        score=score,
                    )
                    candidate_matches.append(match)

                    if score > best_score:
                        best_score = score
                        best_match = match

        if not best_match or best_score < 2.2:
            return None

        normalized_value = best_match.get("normalized_value")
        best_match["normalized_value"] = normalized_value
        best_match["target_unit"] = target_unit
        best_match["answer_value"] = _convert_to_target_unit(normalized_value, target_unit)
        if target_unit:
            best_match["unit_conversion"] = {
                "from": best_match.get("unit"),
                "to": target_unit,
                "base_value": normalized_value,
                "converted_value": best_match["answer_value"],
            }
        if best_match.get("logical_table_materialized") and best_match.get("member_pages"):
            logical_context_matches: List[Dict[str, Any]] = []
            member_pages = list(best_match.get("member_pages") or [])
            member_table_ids = list(best_match.get("materialized_from_tables") or [])
            for index, member_page in enumerate(member_pages):
                if member_page == best_match.get("page"):
                    continue
                context_match = dict(best_match)
                context_match["page"] = member_page
                if index < len(member_table_ids):
                    context_match["table_id"] = member_table_ids[index]
                    context_match["source_table_id"] = member_table_ids[index]
                context_match["row_idx"] = None
                context_match["col_idx"] = None
                context_match["source_row_idx"] = None
                context_match["source_col_idx"] = None
                context_match["raw_value"] = None
                context_match["normalized_value"] = None
                context_match["answer_value"] = None
                context_match["logical_context_support"] = True
                logical_context_matches.append(context_match)
            best_match["logical_context_matches"] = logical_context_matches
        if _has_grounding_rule(question) and normalized_value is not None:
            support_candidates: List[Dict[str, Any]] = []
            best_key = (
                best_match.get("source_doc_id"),
                best_match.get("page"),
                best_match.get("table_id"),
                best_match.get("row_idx"),
                best_match.get("col_idx"),
            )
            for match in candidate_matches:
                key = (
                    match.get("source_doc_id"),
                    match.get("page"),
                    match.get("table_id"),
                    match.get("row_idx"),
                    match.get("col_idx"),
                )
                if key == best_key:
                    continue
                if match.get("source_doc_id") != best_match.get("source_doc_id"):
                    continue
                if not self._match_mentions_year(match, filters.year):
                    continue
                if not self._same_value(match.get("normalized_value"), normalized_value):
                    continue
                support_candidates.append(match)
            support_candidates.sort(key=self._support_priority)
            support_matches: List[Dict[str, Any]] = []
            seen_pages = {best_match.get("page")}
            for match in support_candidates:
                page = match.get("page")
                if page in seen_pages:
                    continue
                support_matches.append(match)
                seen_pages.add(page)
                if len(support_matches) >= 2:
                    break
            best_match["supporting_matches"] = support_matches
        return best_match
