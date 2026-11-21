"""Read-only SQLite store for structured financial facts."""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_FORBIDDEN_SQL_RE = re.compile(
    r"\b(?:INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|"
    r"PRAGMA|VACUUM|REINDEX|LOAD_EXTENSION|SQLITE_MASTER|SQLITE_TEMP_MASTER)\b",
    re.IGNORECASE,
)
_ALLOWED_OPERATORS = {">", ">=", "<", "<="}
_MAX_VM_STEPS = 500_000


class StructuredQueryError(ValueError):
    """Raised when a structured query fails safely."""


class StructuredSecurityError(ValueError):
    """Raised when a query or path violates read-only SQL security constraints."""


class FinancialSQLStore:
    """Thread-safe read-only SQLite access layer for financial facts."""

    def __init__(
        self,
        db_path: Path | str,
        allowed_base_dir: Optional[Path | str] = None,
    ):
        raw_path = Path(db_path)
        if allowed_base_dir is not None:
            if ".." in raw_path.parts:
                raise StructuredSecurityError("Path traversal sequence is not allowed in db_path")
            base_resolved = Path(allowed_base_dir).expanduser().resolve(strict=False)
            resolved = raw_path.expanduser().resolve(strict=False)
            if not str(resolved).startswith(str(base_resolved) + os.sep):
                raise StructuredSecurityError("db_path is outside allowed_base_dir")
        else:
            resolved = raw_path.expanduser().resolve(strict=False)
        self.db_path = resolved

    @staticmethod
    def create_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                doc_id       TEXT PRIMARY KEY,
                pdf_sha1     TEXT,
                company_name TEXT,
                stock_code   TEXT,
                industry_l1  TEXT,
                report_year  INTEGER,
                currency     TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS financial_facts (
                fact_id        TEXT PRIMARY KEY,
                doc_id         TEXT NOT NULL,
                statement_type TEXT,
                scope_type     TEXT,
                metric_raw     TEXT,
                metric_atom_id TEXT,
                period         TEXT,
                raw_value      REAL,
                raw_unit       TEXT,
                normalized_cny REAL,
                source_page    INTEGER,
                table_id       TEXT,
                row_idx        INTEGER,
                col_header     TEXT,
                FOREIGN KEY(doc_id) REFERENCES reports(doc_id)
            )
            """
        )

    @staticmethod
    def create_indexes(conn: sqlite3.Connection) -> None:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_doc_atom_period "
            "ON financial_facts(doc_id, metric_atom_id, period)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_atom_period_val "
            "ON financial_facts(metric_atom_id, period, normalized_cny)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reports_company_year "
            "ON reports(company_name, report_year)"
        )

    def _open_readonly(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            raise StructuredQueryError("Structured SQL database does not exist")
        uri = f"{self.db_path.as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        steps = 0

        def _guard() -> int:
            nonlocal steps
            steps += 50_000
            return 1 if steps > _MAX_VM_STEPS else 0

        conn.set_progress_handler(_guard, 50_000)
        return conn

    def _run_parameterized(self, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        conn = self._open_readonly()
        try:
            cursor = conn.execute(sql, tuple(params))
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as exc:
            raise StructuredQueryError("Read-only SQL execution failed") from exc
        finally:
            conn.close()

    def execute_readonly_sql(self, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        if not isinstance(sql, str):
            raise StructuredSecurityError("SQL query must be a string")
        cleaned = sql.strip()
        if not cleaned:
            raise StructuredSecurityError("SQL query must not be empty")
        if "--" in cleaned or "/*" in cleaned or "*/" in cleaned:
            raise StructuredSecurityError("SQL comments are not allowed in read-only queries")
        if cleaned.endswith(";"):
            cleaned = cleaned[:-1].rstrip()
        if ";" in cleaned:
            raise StructuredSecurityError("Multiple SQL statements are not allowed")
        if not re.match(r"^\s*SELECT\b", cleaned, re.IGNORECASE):
            raise StructuredSecurityError("Only SELECT queries are permitted")
        if _FORBIDDEN_SQL_RE.search(cleaned):
            raise StructuredSecurityError("Forbidden SQL keyword detected")
        return self._run_parameterized(cleaned, params)

    def get_report_summary(self) -> Dict[str, Any]:
        rep = self._run_parameterized("SELECT COUNT(*) AS cnt FROM reports")[0]["cnt"]
        facts = self._run_parameterized("SELECT COUNT(*) AS cnt FROM financial_facts")[0]["cnt"]
        mapped = self._run_parameterized(
            "SELECT COUNT(*) AS cnt FROM financial_facts WHERE metric_atom_id IS NOT NULL"
        )[0]["cnt"]
        return {"reports_count": int(rep), "facts_count": int(facts), "atom_mapped_facts": int(mapped)}

    def query_company_metric(
        self,
        company_name: str,
        metric_atom_id: Optional[str] = None,
        period: Optional[str] = None,
        prefer_consolidated: bool = True,
        exclude_pct: bool = True,
        canonical_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        clauses = ["r.company_name = ?"]
        params: List[Any] = [company_name]
        if metric_atom_id:
            clauses.append("f.metric_atom_id = ?")
            params.append(metric_atom_id)
        if period:
            clauses.append("f.period = ?")
            params.append(str(period))
        if exclude_pct:
            clauses.append("f.raw_unit != '%'")
            clauses.append("f.normalized_cny IS NOT NULL")

        exact_name = canonical_name or ""
        scope_order = "CASE f.scope_type WHEN 'consolidated' THEN 0 WHEN 'unknown' THEN 1 ELSE 2 END"
        if not prefer_consolidated:
            scope_order = "CASE f.scope_type WHEN 'parent' THEN 0 WHEN 'consolidated' THEN 1 ELSE 2 END"

        sql = f"""
            SELECT
                r.doc_id, r.pdf_sha1, r.company_name, r.stock_code, r.industry_l1, r.report_year, r.currency,
                f.fact_id, f.statement_type, f.scope_type, f.metric_raw, f.metric_atom_id,
                f.period, f.raw_value, f.raw_unit, f.normalized_cny, f.source_page,
                f.table_id, f.row_idx, f.col_header
            FROM financial_facts f
            JOIN reports r ON f.doc_id = r.doc_id
            WHERE {' AND '.join(clauses)}
            ORDER BY
                {scope_order},
                CASE WHEN f.metric_raw = ? THEN 0 ELSE 1 END,
                CASE f.statement_type
                    WHEN 'key_indicators' THEN 0
                    WHEN 'income_statement' THEN 1
                    WHEN 'balance_sheet' THEN 2
                    WHEN 'cash_flow' THEN 3
                    ELSE 4
                END,
                f.source_page ASC,
                f.row_idx ASC
        """
        params.append(exact_name)
        return self._run_parameterized(sql, params)

    def _canonical_facts_for_atom_period(
        self,
        metric_atom_id: str,
        period: str,
        industry_l1: Optional[str] = None,
        canonical_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        clauses = [
            "f.metric_atom_id = ?",
            "f.period = ?",
            "f.raw_unit != '%'",
            "f.normalized_cny IS NOT NULL",
        ]
        params: List[Any] = [metric_atom_id, str(period)]
        if industry_l1:
            clauses.append("r.industry_l1 = ?")
            params.append(industry_l1)

        sql = f"""
            SELECT
                r.doc_id, r.pdf_sha1, r.company_name, r.stock_code, r.industry_l1, r.report_year, r.currency,
                f.fact_id, f.statement_type, f.scope_type, f.metric_raw, f.metric_atom_id,
                f.period, f.raw_value, f.raw_unit, f.normalized_cny, f.source_page,
                f.table_id, f.row_idx, f.col_header
            FROM financial_facts f
            JOIN reports r ON f.doc_id = r.doc_id
            WHERE {' AND '.join(clauses)}
            ORDER BY
                r.doc_id ASC,
                CASE f.scope_type WHEN 'consolidated' THEN 0 WHEN 'unknown' THEN 1 ELSE 2 END,
                CASE WHEN f.metric_raw = ? THEN 0 ELSE 1 END,
                CASE f.statement_type
                    WHEN 'key_indicators' THEN 0
                    WHEN 'income_statement' THEN 1
                    WHEN 'balance_sheet' THEN 2
                    WHEN 'cash_flow' THEN 3
                    ELSE 4
                END,
                f.source_page ASC,
                f.row_idx ASC
        """
        params.append(canonical_name or "")
        rows = self._run_parameterized(sql, params)

        deduped_by_company: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            key = str(row.get("company_name") or row["doc_id"])
            if key not in deduped_by_company:
                deduped_by_company[key] = row
        return list(deduped_by_company.values())

    def query_company_ranking(
        self,
        metric_atom_id: str,
        period: str,
        top_k: int = 5,
        ascending: bool = False,
        industry_l1: Optional[str] = None,
        canonical_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        rows = self._canonical_facts_for_atom_period(
            metric_atom_id=metric_atom_id,
            period=period,
            industry_l1=industry_l1,
            canonical_name=canonical_name,
        )
        rows.sort(key=lambda r: float(r["normalized_cny"]), reverse=not ascending)
        return rows[: max(1, int(top_k))]

    def query_threshold_companies(
        self,
        metric_atom_id: str,
        period: str,
        threshold_cny: float,
        operator: str = ">",
        industry_l1: Optional[str] = None,
        canonical_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if operator not in _ALLOWED_OPERATORS:
            raise StructuredSecurityError(f"Unsupported comparison operator: {operator!r}")
        rows = self._canonical_facts_for_atom_period(
            metric_atom_id=metric_atom_id,
            period=period,
            industry_l1=industry_l1,
            canonical_name=canonical_name,
        )
        thresh = float(threshold_cny)
        comparators = {
            ">": lambda v: v > thresh,
            ">=": lambda v: v >= thresh,
            "<": lambda v: v < thresh,
            "<=": lambda v: v <= thresh,
        }
        fn = comparators[operator]
        matched = [row for row in rows if fn(float(row["normalized_cny"]))]
        matched.sort(key=lambda r: float(r["normalized_cny"]), reverse=True)
        return matched

    def query_compare_companies(
        self,
        companies: Sequence[str],
        metric_atom_id: str,
        period: str,
        canonical_name: Optional[str] = None,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        result: Dict[str, Optional[Dict[str, Any]]] = {}
        for company in companies:
            rows = self.query_company_metric(
                company_name=company,
                metric_atom_id=metric_atom_id,
                period=period,
                prefer_consolidated=True,
                exclude_pct=True,
                canonical_name=canonical_name,
            )
            result[company] = rows[0] if rows else None
        return result
