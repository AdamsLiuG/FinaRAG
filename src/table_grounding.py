from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from src.retrieval_filters import RetrievalFilters
from src.text_normalization import normalize_text, parse_numeric_value, tokenize_for_bm25


_TARGET_UNIT_FACTORS = {
    "亿元": 1e8,
    "万元": 1e4,
    "百万元": 1e6,
    "千万元": 1e7,
    "元": 1.0,
}

_BAD_REVENUE_ROW_TERMS = (
    "占营业收入比例",
    "营业收入比例",
    "销售费用占",
    "研发投入占营业收入",
)
_BAD_REVENUE_COL_TERMS = (
    "同比",
    "变动",
    "比例",
    "占比",
    "百分比",
    "本期比上年同期",
    "期末余额",
    "期初",
    "数量合计",
    "分部收入",
    "关联方",
)
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
_LOW_PRIORITY_REVENUE_TERMS = (
    "扣除与主营业务无关",
    "不具备商业实质",
)
_CONSOLIDATED_REVENUE_TERMS = (
    "营业收入",
    "营业总收入",
    "一、营业总收入",
    "-、营业总收入",
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


def _is_revenue_question(question: str) -> bool:
    normalized_question = normalize_text(question)
    return "营业收入" in normalized_question or "营收" in normalized_question


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


def _reject_revenue_unit(question: str, unit_hint: Optional[str]) -> bool:
    if not _is_revenue_question(question):
        return False
    if "%" in (question or ""):
        return False
    unit_text = normalize_text(str(unit_hint or ""))
    return "%" in unit_text or "百分点" in unit_text


def _table_scope_score(question: str, table_context: str) -> float:
    if not _is_revenue_question(question):
        return 0.0
    compact = _compact_text(table_context)
    score = 0.0
    if "主要会计数据" in compact or "主要财务指标" in compact:
        score += 5.0
    if "公司简介和主要财务指标" in compact:
        score += 2.0
    if "合并利润表" in compact or "利润表" in compact:
        score += 2.5
    if any(term in compact for term in ("关联方", "分部信息", "按产品", "按地区", "母公司")):
        score -= 3.0
    return score


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


def _revenue_row_score(question: str, row_text: str, col_text: str, raw_value: object = None) -> Optional[float]:
    normalized_question = normalize_text(question)
    if "营业收入" not in normalized_question and "营收" not in normalized_question:
        return 0.0

    row_normalized = normalize_text(row_text)
    col_normalized = normalize_text(col_text)
    combined = f"{row_normalized} {col_normalized}"
    if not any(term in combined for term in ("营业收入", "营业总收入", "营收")):
        return None
    if any(term in combined for term in _BAD_REVENUE_ROW_TERMS):
        return None
    has_year_col = bool(re.search(r"20\d{2}", col_normalized))
    if any(term in col_normalized for term in _BAD_REVENUE_COL_TERMS):
        return None
    if "增减" in col_normalized and not has_year_col:
        return None
    if any(term in col_normalized for term in _BAD_SEGMENT_COL_TERMS) and "合计" not in col_normalized:
        return None
    if "季度" in col_normalized and "季度" not in normalized_question:
        return None
    if "%" not in normalized_question and ("%" in col_normalized or "百分点" in col_normalized):
        return None
    if "母公司" in combined and "母公司" not in normalized_question:
        return None
    if "本行" in combined and "本行" not in normalized_question:
        return None
    if any(term in combined for term in ("分部", "地区", "产品", "关联方", "客户")) and not any(
        term in normalized_question for term in ("分部", "地区", "产品", "关联方", "客户")
    ):
        return None

    if row_normalized in {"营业收入", "一、营业收入", "-、营业收入"}:
        score = 8.0
        if "本集团" in col_normalized or "合计" in col_normalized:
            score += 2.0
        return score
    if row_normalized in {"营业总收入", "一、营业总收入", "-、营业总收入"}:
        score = 7.5
        if "本集团" in col_normalized or "合计" in col_normalized:
            score += 2.0
        return score
    if any(term in row_normalized for term in _LOW_PRIORITY_REVENUE_TERMS):
        return 1.0
    if any(term in row_normalized for term in _CONSOLIDATED_REVENUE_TERMS):
        return 4.0
    return 1.5


class TableGrounder:
    def __init__(self, documents_dir: Path):
        self.documents_dir = Path(documents_dir)
        self._documents_cache: Dict[str, Dict] = {}

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
        parts = [
            str(table.get("markdown") or ""),
            str(table.get("caption") or ""),
            str(table.get("section_title") or ""),
            str(table.get("section_name") or ""),
            str(table.get("report_section") or ""),
            self._page_text(document, page),
            self._nearby_chunk_text(document, page),
        ]
        return "\n".join(part for part in parts if part).translate(_OCR_NORMALIZATION_MAP)

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
            "page": cell.get("page"),
            "row_idx": cell.get("row_idx"),
            "col_idx": cell.get("col_idx"),
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
            for table in structured_tables:
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
                    if _reject_revenue_unit(question, unit_hint):
                        continue
                    revenue_score = _revenue_row_score(question, row_text, col_text, raw_value=raw_value)
                    if revenue_score is None:
                        continue
                    score += revenue_score
                    score += _score_overlap(query_tokens, row_text) * 2.2
                    score += _score_overlap(query_tokens, col_text) * 1.8
                    score += _score_overlap(query_tokens, table_context) * 0.8
                    score += _table_scope_score(question, table_context)
                    if _is_revenue_question(question):
                        compact_row = _compact_text(row_text)
                        compact_table = _compact_text(table_context)
                        if compact_row.startswith("其中:营业收入") and "营业总收入" in compact_table:
                            score += 6.0
                    if _is_revenue_question(question) and cell.get("page") is not None:
                        try:
                            page_number = int(cell.get("page"))
                        except (TypeError, ValueError):
                            page_number = None
                        if page_number is not None:
                            if page_number <= 10:
                                score += 8.0
                            elif page_number <= 20:
                                score += 4.0

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
        if _is_revenue_question(question) and normalized_value is not None:
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
