"""Extract structured financial facts from FinaRAG parsed/chunked reports."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.metric_atoms import MetricAtom, MetricAtomStore, get_default_store
from src.structured.pot_sandbox import normalize_monetary_unit
from src.table_grounding import _TARGET_UNIT_FACTORS
from src.text_normalization import normalize_text

_UNIT_HEADER_RE = re.compile(r"单位[：:\s]*(?:人民币)?(亿元|千万元|百万元|万元|千元|元)")
_YEAR_RE = re.compile(r"(19\d{2}|20\d{2})")
_RATIO_OR_PER_SHARE_TOKENS = ("%", "％", "增减", "占比", "比例", "率", "每股")
_RATIO_ROW_RE = re.compile(r"(?:比营业收入|占营业收入|比总资产|占总资产|比重|比上年|比期初)")
_EXCLUDED_METRIC_ROW_TOKENS = (
    "比营业收入",
    "占营业收入",
    "扣除与主营业务无关",
    "营业收入扣除",
    "占总资产",
    "比总资产",
)
_EXTRA_TABLE_ATOM_ALIASES: Tuple[Tuple[str, str], ...] = (
    ("归属于上市公司股东的扣除非经常性损益的净利润", "MET-001"),
    ("归属于本行股东的扣除非经常性损益的净利润", "MET-001"),
    ("归属于母公司股东的扣除非经常性损益的净利润", "MET-001"),
    ("归属于母公司所有者的扣除非经常性损益的净利润", "MET-001"),
    ("归属于公司普通股股东的扣除非经常性损益的净利润", "MET-001"),
    ("扣除非经常性损益后的归属于公司普通股股东的净利润", "MET-001"),
    ("扣除非经常性损益后的净利润", "MET-001"),
    ("归属于上市公司股东的净利润", "MET-002"),
    ("归属于本行股东的净利润", "MET-002"),
    ("归属于母公司所有者的净利润", "MET-002"),
    ("归属于公司股东的净利润", "MET-002"),
    ("归属于公司普通股股东的净利润", "MET-002"),
    ("归属于母公司普通股股东的净利润", "MET-002"),
    ("归属于母公司普通股股东的合并净利润", "MET-002"),
    ("经营活动产生的现金流量净额", "MET-003"),
    ("经营活动产生的现金流净额", "MET-003"),
)
_MULTI_NUMBER_RE = re.compile(r"[\(（]?[-+]?\d[\d,]*(?:\.\d+)?[\)）]?")
_SEPARATOR_LINE_RE = re.compile(r"^\s*\|?(?:\s*:?-{2,}:?\s*\|)+\s*$")

_MONETARY_FACTORS: Dict[str, float] = {
    **_TARGET_UNIT_FACTORS,
    "千元": 1e3,
}


def _compact(text: object) -> str:
    return normalize_text(str(text or "")).replace(" ", "")


def _split_markdown_row(line: str) -> List[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _parse_numeric_cell(raw_value: object) -> Optional[float]:
    text = str(raw_value or "").strip()
    if not text or text in {"-", "--", "—", "——", "N/A", "n/a", "不适用", "/"}:
        return None
    # Fail closed when OCR merged multiple numbers into a single table cell.
    tokens = [tok for tok in re.split(r"\s{2,}", text) if tok.strip()]
    if len(tokens) > 1 and sum(1 for t in tokens if _MULTI_NUMBER_RE.search(t)) > 1:
        return None

    compact = text.replace(",", "").replace("，", "").replace(" ", "")
    if len(re.findall(r"[-+]?\d+(?:\.\d+)?", compact)) > 1:
        return None

    negative = False
    if (compact.startswith("(") and compact.endswith(")")) or (
        compact.startswith("（") and compact.endswith("）")
    ):
        negative = True
        compact = compact[1:-1]

    compact = compact.rstrip("%％")
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", compact):
        return None
    try:
        value = float(compact)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return -abs(value) if negative else value


def _infer_period(col_header: str, report_year: Optional[int]) -> Optional[str]:
    compact = _compact(col_header)
    if not compact:
        return None
    match = _YEAR_RE.search(compact)
    if match:
        return match.group(1)
    if report_year is not None:
        if any(tok in compact for tok in ("本报告期", "本期", "本期末", "本年度")):
            return str(report_year)
        if any(tok in compact for tok in ("上年同期", "上年度", "上期", "上年末", "上年")):
            return str(report_year - 1)
        if "上上年" in compact:
            return str(report_year - 2)
    return None


def _is_subheader_row(cells: List[str], report_year: Optional[int]) -> bool:
    if len(cells) < 2:
        return False
    period_hits = sum(1 for cell in cells[1:] if _infer_period(cell, report_year) is not None)
    numeric_only_hits = sum(1 for cell in cells[1:] if _parse_numeric_cell(cell) is not None)
    return period_hits >= 2 and numeric_only_hits == 0


def _classify_statement_and_scope(markdown: str, context_text: str) -> Tuple[str, str]:
    head_text = "\n".join((markdown or "").splitlines()[:8])
    combined = f"{context_text}\n{head_text}"
    compact = _compact(combined)

    if "主要会计数据" in compact or "主要财务指标" in compact:
        stmt = "key_indicators"
    elif "合并利润表" in compact or "母公司利润表" in compact or (
        "营业收入" in compact and "净利润" in compact and ("营业成本" in compact or "营业总成本" in compact)
    ):
        stmt = "income_statement"
    elif "资产负债表" in compact or ("资产总计" in compact and "负债合计" in compact):
        stmt = "balance_sheet"
    elif "现金流量表" in compact or "经营活动产生的现金流量净额" in compact:
        stmt = "cash_flow"
    else:
        stmt = "other"

    if any(tok in compact for tok in ("母公司资产负债表", "母公司利润表", "母公司现金流量表", "母公司财务报表")):
        scope = "parent"
    elif "合并" in compact or "归属于" in compact or stmt == "key_indicators":
        scope = "consolidated"
    else:
        scope = "unknown"
    return stmt, scope


def _preceding_page_context(page_text: object, markdown: str, window_lines: int = 8) -> str:
    if not page_text:
        return ""
    if isinstance(page_text, tuple):
        page_lines, compact_lines = page_text
    else:
        page_lines = str(page_text).splitlines()
        compact_lines = None
    if not page_lines:
        return ""
    first_md_line = ""
    for line in (markdown or "").splitlines():
        if line.strip():
            first_md_line = _compact(line)[:20]
            break
    if first_md_line:
        if compact_lines is not None:
            for idx, comp_line in enumerate(compact_lines):
                if first_md_line in comp_line:
                    start = max(0, idx - window_lines)
                    return "\n".join(page_lines[start:idx])
        else:
            for idx, line in enumerate(page_lines):
                if first_md_line in _compact(line):
                    start = max(0, idx - window_lines)
                    return "\n".join(page_lines[start:idx])
    return "\n".join(page_lines[:15])


class FinancialTableExtractor:
    """Normalize financial report tables into relational facts with explicit unit & provenance."""

    def __init__(self, atom_store: Optional[MetricAtomStore] = None):
        self.atom_store = atom_store or get_default_store()
        term_map: Dict[str, MetricAtom] = {}
        for atom in self.atom_store.all_atoms():
            for term in atom.match_terms:
                comp = _compact(term)
                if comp and comp not in term_map:
                    term_map[comp] = atom
        for alias, atom_id in _EXTRA_TABLE_ATOM_ALIASES:
            atom = self.atom_store.get(atom_id)
            comp = _compact(alias)
            if atom is not None and comp and comp not in term_map:
                term_map[comp] = atom
        self._compact_atom_index: List[Tuple[str, MetricAtom]] = sorted(
            term_map.items(),
            key=lambda pair: len(pair[0]),
            reverse=True,
        )
        self._atom_cache: Dict[str, Optional[str]] = {}

    def map_metric_atom(self, metric_raw: str) -> Optional[str]:
        compact = _compact(metric_raw)
        if not compact:
            return None
        if compact in self._atom_cache:
            return self._atom_cache[compact]
        if any(tok in compact for tok in _EXCLUDED_METRIC_ROW_TOKENS):
            self._atom_cache[compact] = None
            return None
        for term, atom in self._compact_atom_index:
            if term in compact:
                self._atom_cache[compact] = atom.atom_id
                return atom.atom_id
        self._atom_cache[compact] = None
        return None

    @staticmethod
    def detect_table_unit(table: Dict[str, Any], context_text: str) -> str:
        md_head = "\n".join((table.get("markdown") or "").splitlines()[:5])
        for candidate_text in (md_head, context_text):
            match = _UNIT_HEADER_RE.search(candidate_text or "")
            if match:
                return normalize_monetary_unit(match.group(1))

        explicit_hint = normalize_monetary_unit(table.get("unit_hint"))
        if explicit_hint in _MONETARY_FACTORS:
            return explicit_hint

        for cell in table.get("cell_records") or []:
            cell_unit = normalize_monetary_unit(cell.get("unit_hint"))
            if cell_unit in _MONETARY_FACTORS:
                return cell_unit

        return "元"

    @staticmethod
    def is_ratio_or_per_share(row_header: str, col_header: str) -> bool:
        combined = _compact(f"{row_header} {col_header}")
        if any(tok in combined for tok in _RATIO_OR_PER_SHARE_TOKENS):
            return True
        return bool(_RATIO_ROW_RE.search(_compact(row_header)))

    def extract_from_report_dict(
        self,
        report_data: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        metainfo = report_data.get("metainfo") or {}
        content = report_data.get("content") or {}
        doc_id = str(metainfo.get("doc_id") or metainfo.get("sha1_name") or "")
        pdf_sha1 = str(metainfo.get("sha1_name") or doc_id)
        company_name = str(metainfo.get("company_name") or "").strip()
        stock_code = str(metainfo.get("stock_code") or metainfo.get("security_code") or "").strip()
        industry_l1 = str(metainfo.get("industry_l1") or metainfo.get("major_industry") or "").strip()
        raw_year = metainfo.get("report_year") or metainfo.get("fiscal_year")
        try:
            report_year: Optional[int] = int(raw_year) if raw_year is not None else None
        except (TypeError, ValueError):
            report_year = None
        currency = str(metainfo.get("currency") or "CNY").strip()

        report_row = {
            "doc_id": doc_id,
            "pdf_sha1": pdf_sha1,
            "company_name": company_name,
            "stock_code": stock_code,
            "industry_l1": industry_l1,
            "report_year": report_year,
            "currency": currency,
        }

        pages_by_num: Dict[int, Tuple[List[str], List[str]]] = {}
        for p in content.get("pages") or []:
            if isinstance(p, dict) and p.get("page") is not None:
                raw_lines = str(p.get("text") or "").splitlines()
                comp_lines = [_compact(line) for line in raw_lines]
                pages_by_num[int(p["page"])] = (raw_lines, comp_lines)

        facts: List[Dict[str, Any]] = []
        seen_fact_ids: set[str] = set()
        for table in content.get("structured_tables") or []:
            if not isinstance(table, dict):
                continue
            table_id = str(table.get("table_id") or "")
            try:
                page = int(table.get("page") or 0)
            except (TypeError, ValueError):
                page = 0
            markdown = str(table.get("markdown") or "")
            page_entry = pages_by_num.get(page)
            context_text = _preceding_page_context(page_entry, markdown)
            table_unit = self.detect_table_unit(table, context_text)
            stmt_type, scope_type = _classify_statement_and_scope(markdown, context_text)

            extracted_cells = self._extract_markdown_cells(markdown, report_year)
            if not extracted_cells:
                extracted_cells = self._extract_fallback_cell_records(table, report_year)

            for row_idx, col_idx, row_header, col_header, raw_str, period in extracted_cells:
                numeric_val = _parse_numeric_cell(raw_str)
                if numeric_val is None or not row_header:
                    continue
                if self.is_ratio_or_per_share(row_header, col_header):
                    raw_unit = "%"
                    normalized_cny: Optional[float] = None
                else:
                    raw_unit = table_unit
                    factor = _MONETARY_FACTORS.get(table_unit)
                    normalized_cny = numeric_val * factor if factor is not None else None

                atom_id = self.map_metric_atom(row_header)
                base_fact_id = f"{doc_id}::t{table_id}::p{page}::r{row_idx}::c{col_idx}"
                fact_id = base_fact_id
                suffix = 1
                while fact_id in seen_fact_ids:
                    suffix += 1
                    fact_id = f"{base_fact_id}#{suffix}"
                seen_fact_ids.add(fact_id)

                facts.append(
                    {
                        "fact_id": fact_id,
                        "doc_id": doc_id,
                        "statement_type": stmt_type,
                        "scope_type": scope_type,
                        "metric_raw": row_header,
                        "metric_atom_id": atom_id,
                        "period": period,
                        "raw_value": numeric_val,
                        "raw_unit": raw_unit,
                        "normalized_cny": normalized_cny,
                        "source_page": page,
                        "table_id": table_id,
                        "row_idx": row_idx,
                        "col_header": col_header,
                    }
                )

        return report_row, facts

    def _extract_markdown_cells(
        self,
        markdown: str,
        report_year: Optional[int],
    ) -> List[Tuple[int, int, str, str, str, Optional[str]]]:
        if not markdown or "|" not in markdown:
            return []
        lines = [line for line in markdown.splitlines() if "|" in line]
        if len(lines) < 2:
            return []

        sep_idx = None
        for idx, line in enumerate(lines[:5]):
            if _SEPARATOR_LINE_RE.match(line):
                sep_idx = idx
                break
        if sep_idx is None or sep_idx == 0:
            return []

        header_rows = [_split_markdown_row(line) for line in lines[:sep_idx]]
        col_headers = list(header_rows[-1])
        if len(header_rows) > 1:
            merged_headers: List[str] = []
            max_cols = max(len(r) for r in header_rows)
            for c_idx in range(max_cols):
                parts = [
                    r[c_idx]
                    for r in header_rows
                    if c_idx < len(r) and r[c_idx] and not _UNIT_HEADER_RE.search(r[c_idx])
                ]
                merged_headers.append(" ".join(parts).strip())
            col_headers = merged_headers

        active_headers = col_headers
        cells: List[Tuple[int, int, str, str, str, Optional[str]]] = []
        for data_idx, line in enumerate(lines[sep_idx + 1 :], start=1):
            row_cells = _split_markdown_row(line)
            if not row_cells:
                continue
            if _is_subheader_row(row_cells, report_year):
                active_headers = row_cells
                continue
            row_header = re.sub(r"\s+", "", row_cells[0]) if row_cells[0] else ""
            if not row_header:
                continue
            for col_idx in range(1, len(row_cells)):
                raw_val = row_cells[col_idx]
                col_header = active_headers[col_idx] if col_idx < len(active_headers) else ""
                period = _infer_period(col_header, report_year)
                cells.append((data_idx, col_idx, row_header, col_header, raw_val, period))
        return cells

    def _extract_fallback_cell_records(
        self,
        table: Dict[str, Any],
        report_year: Optional[int],
    ) -> List[Tuple[int, int, str, str, str, Optional[str]]]:
        out: List[Tuple[int, int, str, str, str, Optional[str]]] = []
        for rec in table.get("cell_records") or []:
            if not isinstance(rec, dict):
                continue
            row_headers = [str(x).strip() for x in (rec.get("matched_row_headers") or []) if str(x).strip()]
            col_headers = [str(x).strip() for x in (rec.get("matched_col_headers") or []) if str(x).strip()]
            row_header = re.sub(r"\s+", "", " ".join(row_headers))
            col_header = " ".join(col_headers)
            raw_val = str(rec.get("raw_value") or "")
            period = str(rec.get("period")) if rec.get("period") not in (None, "") else _infer_period(col_header, report_year)
            row_idx = int(rec.get("row_idx") or 0)
            col_idx = int(rec.get("col_idx") or 0)
            out.append((row_idx, col_idx, row_header, col_header, raw_val, period))
        return out

    def build_sqlite_db(
        self,
        chunked_reports_dir: Path,
        db_path: Path,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        from src.structured.sql_store import FinancialSQLStore

        reports_dir = Path(chunked_reports_dir)
        out_path = Path(db_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(f".{out_path.name}.tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        files = sorted(reports_dir.glob("*.json"))
        if limit is not None:
            files = files[: max(0, int(limit))]

        conn = sqlite3.connect(str(tmp_path))
        reports_count = 0
        facts_count = 0
        atom_mapped_facts = 0
        stmt_counts: Dict[str, int] = {}
        try:
            FinancialSQLStore.create_schema(conn)
            report_batch: List[Tuple[Any, ...]] = []
            fact_batch: List[Tuple[Any, ...]] = []
            for file_path in files:
                with open(file_path, "r", encoding="utf-8") as fh:
                    report_data = json.load(fh)
                report_row, fact_rows = self.extract_from_report_dict(report_data)
                if not report_row["doc_id"]:
                    continue
                reports_count += 1
                report_batch.append(
                    (
                        report_row["doc_id"],
                        report_row["pdf_sha1"],
                        report_row["company_name"],
                        report_row["stock_code"],
                        report_row["industry_l1"],
                        report_row["report_year"],
                        report_row["currency"],
                    )
                )
                for fact in fact_rows:
                    facts_count += 1
                    if fact["metric_atom_id"]:
                        atom_mapped_facts += 1
                    st = fact["statement_type"]
                    stmt_counts[st] = stmt_counts.get(st, 0) + 1
                    fact_batch.append(
                        (
                            fact["fact_id"],
                            fact["doc_id"],
                            fact["statement_type"],
                            fact["scope_type"],
                            fact["metric_raw"],
                            fact["metric_atom_id"],
                            fact["period"],
                            fact["raw_value"],
                            fact["raw_unit"],
                            fact["normalized_cny"],
                            fact["source_page"],
                            fact["table_id"],
                            fact["row_idx"],
                            fact["col_header"],
                        )
                    )

            conn.executemany(
                """
                INSERT OR REPLACE INTO reports (
                    doc_id, pdf_sha1, company_name, stock_code, industry_l1, report_year, currency
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                report_batch,
            )
            conn.executemany(
                """
                INSERT OR REPLACE INTO financial_facts (
                    fact_id, doc_id, statement_type, scope_type, metric_raw,
                    metric_atom_id, period, raw_value, raw_unit, normalized_cny,
                    source_page, table_id, row_idx, col_header
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                fact_batch,
            )
            FinancialSQLStore.create_indexes(conn)
            conn.commit()
        finally:
            conn.close()

        os.replace(tmp_path, out_path)
        return {
            "db_path": str(out_path),
            "reports_count": reports_count,
            "facts_count": facts_count,
            "atom_mapped_facts": atom_mapped_facts,
            "statement_type_counts": stmt_counts,
        }
