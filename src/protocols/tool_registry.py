"""Strict, read-only financial tools shared by the MCP bindings."""

from __future__ import annotations

import ast
import asyncio
import json
import math
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, copy_context
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.harness.trajectory_store import capture_active, capture_step, harness_enabled, record_trajectory
from src.storage.base import StorageError, validate_safe_path

MAX_ARGUMENT_BYTES = 64 * 1024
MAX_RESULT_BYTES = 256 * 1024
DOC_ID_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"
DocId = Annotated[str, Field(pattern=DOC_ID_PATTERN)]
Name = Annotated[str, Field(min_length=1, max_length=160, pattern=r"\S")]
Query = Annotated[str, Field(min_length=1, max_length=1000, pattern=r"\S")]
Year = Annotated[int, Field(ge=1000, le=9999)]
Page = Annotated[int, Field(ge=1, le=100000)]
Amount = Annotated[float, Field(ge=-1e24, le=1e24)]
DocIds = Annotated[list[DocId], Field(min_length=1, max_length=12)]
PotUnit = Literal["", "%", "％", "元", "千元", "万元", "百万元", "千万元", "亿元",
                  "cny", "rmb", "人民币元", "人民币千元", "人民币万元",
                  "人民币百万元", "人民币千万元", "人民币亿元", "万元人民币"]
_MONEY_FACTORS = {"元": 1, "千元": 1000, "万元": 10000, "百万元": 1000000,
                  "千万元": 10000000, "亿元": 100000000}
_RESOURCE_URIS = frozenset({"finarag://schemas/config", "finarag://schemas/report"})
_WORKER_DEADLINE = ContextVar("mcp_worker_deadline", default=math.inf)


def canonical_json(value: Any) -> str:
    # ASCII escaping also permits JSON strings containing lone escaped surrogates.
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _finite_number(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite JSON number")
    return number


def decode_json(payload: bytes | str, *, limit: int = MAX_ARGUMENT_BYTES) -> Any:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    if len(payload) > limit:
        raise ValueError("Payload too large")
    return json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object,
                      parse_constant=_finite_number, parse_float=_finite_number)


class StrictObject(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


class FinancialFactsInput(StrictObject):
    operation: Literal["metric", "compare", "rank", "threshold"]
    metric_atom_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
    period: Annotated[str, Field(pattern=r"^[0-9]{4}$")]
    company_name: Name | None = None
    companies: Annotated[list[Name], Field(min_length=1, max_length=12)] | None = None
    industry_l1: Name | None = None
    scope: Literal["consolidated", "parent"] = "consolidated"
    limit: int = Field(default=20, ge=1, le=50)
    ascending: bool = False
    threshold_cny: Amount | None = None
    operator: Literal[">", ">=", "<", "<="] | None = None

    @model_validator(mode="after")
    def validate_operation(self):
        if (self.company_name is not None) != (self.operation == "metric"):
            raise ValueError("Expected company_name for metric")
        if (self.companies is not None) != (self.operation == "compare"):
            raise ValueError("Expected companies for compare")
        if self.companies is not None and len(set(self.companies)) != len(self.companies):
            raise ValueError("Repeated company")
        if self.scope == "parent" and self.operation != "metric":
            raise ValueError("Parent scope requires metric")
        if "ascending" in self.model_fields_set and self.operation != "rank":
            raise ValueError("Ascending requires rank")
        threshold_fields = self.threshold_cny is not None, self.operator is not None
        if threshold_fields != ((True, True) if self.operation == "threshold" else (False, False)):
            raise ValueError("Threshold requires a value and operator")
        return self


class PotInputVariable(StrictObject):
    name: Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,31}$")]
    value: Amount
    unit: PotUnit
    doc_id: DocId | None = None
    page: Page | None = None

    @model_validator(mode="after")
    def validate_source(self):
        if (self.doc_id is None) != (self.page is None):
            raise ValueError("Source requires a document and page")
        return self


def _pot_dimension(node: ast.AST, dimensions: dict[str, int]) -> int:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        if abs(node.value) <= 1e24:
            return 0
    if isinstance(node, ast.Name) and node.id in dimensions:
        return dimensions[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _pot_dimension(node.operand, dimensions)
    if isinstance(node, ast.BinOp):
        left = _pot_dimension(node.left, dimensions)
        right = _pot_dimension(node.right, dimensions)
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mod)) and left == right:
            return left
        if isinstance(node.op, ast.Mult):
            return left + right
        if isinstance(node.op, (ast.Div, ast.FloorDiv)):
            return left - right
        if isinstance(node.op, ast.Pow) and right == 0:
            if not left:
                return 0
            if (isinstance(node.right, ast.Constant) and type(node.right.value) is int
                    and abs(node.right.value) <= 100):
                return left * node.right.value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        args = [_pot_dimension(arg, dimensions) for arg in node.args]
        if node.func.id == "abs" and len(args) == 1:
            return args[0]
        if node.func.id == "round" and (len(args) == 1 or len(args) == 2 and args[1] == 0):
            return args[0]
        if node.func.id in {"min", "max"} and args and len(set(args)) == 1:
            return args[0]
    raise ValueError("Unsupported expression or incompatible units")


class PotInput(StrictObject):
    expression: str = Field(min_length=1, max_length=512)
    variables: list[PotInputVariable] = Field(min_length=1, max_length=16)
    result_unit: PotUnit = ""
    ndigits: int = Field(default=4, ge=0, le=8)

    @model_validator(mode="after")
    def validate_expression(self):
        from src.structured.pot_sandbox import normalize_monetary_unit

        dimensions = {item.name: int(normalize_monetary_unit(item.unit) in _MONEY_FACTORS)
                      for item in self.variables}
        if len(dimensions) != len(self.variables):
            raise ValueError("Repeated variable")
        try:
            tree = ast.parse(self.expression.strip(), mode="eval")
        except (SyntaxError, RecursionError) as exc:
            raise ValueError("Invalid expression") from exc
        if sum(1 for _ in ast.walk(tree)) > 64:
            raise ValueError("Expression too large")
        expected = int(normalize_monetary_unit(self.result_unit) in _MONEY_FACTORS)
        if _pot_dimension(tree.body, dimensions) != expected:
            raise ValueError("Incompatible result unit")
        return self


class TocInput(StrictObject):
    doc_id: DocId
    question: Query
    max_sections: int = Field(default=3, ge=1, le=5)
    max_pages: int = Field(default=4, ge=1, le=10)


class KgInput(StrictObject):
    question: Query
    company_name: Name
    doc_ids: DocIds
    max_hops: int = Field(default=2, ge=1, le=2)
    max_paths: int = Field(default=8, ge=1, le=8)
    max_pages: int = Field(default=4, ge=1, le=8)


class HybridInput(StrictObject):
    query: Query
    doc_ids: DocIds
    company_name: Name | None = None
    report_year: Year | None = None
    top_k: int = Field(default=10, ge=1, le=20)


class EvidenceInput(StrictObject):
    doc_id: DocId
    page: Page


class ClaimsInput(StrictObject):
    answer_text: str = Field(min_length=1, max_length=8000, pattern=r"\S")
    evidence: list[EvidenceInput] = Field(min_length=1, max_length=20)
    company_name: Name | None = None
    report_year: Year | None = None


_TOOLS = {
    "query_financial_facts_sql": (FinancialFactsInput, "Query published financial facts"),
    "pot_calculate": (PotInput, "Evaluate bounded financial arithmetic"),
    "navigate_toc_sections": (TocInput, "Find relevant table-of-contents sections and pages"),
    "query_financial_kg": (KgInput, "Find report-scoped financial relationships"),
    "hybrid_search_reports": (HybridInput, "Search specified financial reports"),
    "verify_claims": (ClaimsInput, "Check claims against specified report pages"),
}


@dataclass(frozen=True)
class ToolPrincipal:
    subject: str
    roles: frozenset[str]


def valid_principal(principal: ToolPrincipal | None) -> bool:
    return (isinstance(principal, ToolPrincipal)
            and isinstance(principal.subject, str) and 0 < len(principal.subject) <= 128
            and principal.subject == principal.subject.strip()
            and principal.subject.isprintable()
            and isinstance(principal.roles, (set, frozenset, tuple))
            and len(principal.roles) <= 16
            and all(isinstance(role, str) and 0 < len(role) <= 64 for role in principal.roles))


def default_policy(principal: ToolPrincipal, action: str, arguments: dict | None) -> bool:
    if not valid_principal(principal) or "read" not in principal.roles:
        return False
    if action in {"resources/list", "resources/read"}:
        return arguments is not None and arguments.get("uri") in _RESOURCE_URIS
    return action in _TOOLS


@dataclass(frozen=True)
class ToolResult:
    data: Any = None
    error_code: str | None = None


class _ToolError(Exception):
    def __init__(self, code: str):
        self.code = code


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise _ToolError("timeout")


class MCPToolRegistry:
    """Handlers receive validated argument dictionaries and monotonic deadlines."""

    def __init__(self, *, handlers, policy, max_workers=4, timeout_s=5.0):
        if not 1 <= max_workers <= 32 or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Invalid executor configuration")
        self._handlers = dict(handlers)
        if set(self._handlers) - _TOOLS.keys():
            raise ValueError("Unknown tool handler")
        self._policy = policy
        self._timeout_s = timeout_s
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="finarag-mcp")
        self._slots = threading.BoundedSemaphore(max_workers)
        self._lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._closers: tuple[Callable[[], None], ...] = ()
        self._resolve_scope: Callable | None = None

    def allows(self, principal, action, arguments) -> bool:
        if not valid_principal(principal):
            return False
        try:
            return self._policy(principal, action, arguments) is True
        except Exception:
            return False

    def list_tools(self, principal):
        return [
            {"name": name, "description": description, "inputSchema": model.model_json_schema()}
            for name, (model, description) in _TOOLS.items()
            if name in self._handlers and self.allows(principal, name, None)
        ]

    async def call(self, name, arguments, *, principal, timeout_s=None) -> ToolResult:
        if not capture_active() and not harness_enabled():
            return await self._call(name, arguments, principal=principal, timeout_s=timeout_s)
        safe_name = name if isinstance(name, str) and name in _TOOLS else "unknown"
        with record_trajectory(entry_point="tool_registry", request={"tool_name": safe_name}) as observer:
            if not capture_active():
                return await self._call(name, arguments, principal=principal, timeout_s=timeout_s)
            started = time.monotonic()
            observed = {"tool_name": safe_name}
            try:
                result = await self._call(name, arguments, principal=principal, timeout_s=timeout_s, observed=observed)
            except asyncio.CancelledError:
                capture_step(
                    "tool", operation=safe_name, step_key="tool", inputs=observed, outputs={},
                    status="cancelled", costs={"elapsed_ms": (time.monotonic() - started) * 1000},
                )
                raise
            status = "ok" if result.error_code is None else "timeout" if result.error_code == "timeout" else "failed"
            capture_step(
                "tool", operation=safe_name, step_key="tool", inputs=observed,
                outputs={"data": result.data, "error_code": result.error_code},
                status=status, error_code=result.error_code,
                costs={"elapsed_ms": (time.monotonic() - started) * 1000},
            )
            if observer is not None:
                observer.finish(status=status)
            return result

    async def _call(self, name, arguments, *, principal, timeout_s=None, observed=None) -> ToolResult:
        started = time.monotonic()
        if not valid_principal(principal):
            return ToolResult(error_code="forbidden")
        if not isinstance(name, str) or name not in self._handlers:
            return ToolResult(error_code="not_found")
        model, _ = _TOOLS[name]
        try:
            if isinstance(arguments, (str, bytes)):
                arguments = decode_json(arguments)
            if not isinstance(arguments, dict) or len(canonical_json(arguments)) > MAX_ARGUMENT_BYTES:
                raise ValueError("Invalid arguments")
            normalized = model.model_validate(arguments, strict=True)
            if timeout_s is not None and (type(timeout_s) not in (int, float)
                                          or not math.isfinite(timeout_s) or timeout_s < 0):
                raise ValueError("Invalid timeout")
        except (TypeError, ValueError, OverflowError, RecursionError):
            return ToolResult(error_code="invalid_arguments")
        payload = normalized.model_dump()
        if not self.allows(principal, name, normalized.model_dump()):
            return ToolResult(error_code="forbidden")
        if observed is not None:
            observed["arguments"] = payload
        budget = self._timeout_s if timeout_s is None else min(timeout_s, self._timeout_s)
        deadline = started + budget
        with self._lock:
            if self._closed:
                return ToolResult(error_code="unavailable")
            if time.monotonic() >= deadline:
                return ToolResult(error_code="timeout")
            if not self._slots.acquire(blocking=False):
                return ToolResult(error_code="busy")
            try:
                if observed is None:
                    future = self._executor.submit(self._invoke, name, payload, principal, deadline)
                else:
                    future = self._executor.submit(copy_context().run, self._invoke, name, payload, principal, deadline)
            except RuntimeError:
                self._slots.release()
                return ToolResult(error_code="unavailable")
            future.add_done_callback(lambda _: self._slots.release())
        waiting = asyncio.wrap_future(future)
        try:
            result = await asyncio.wait_for(waiting, timeout=max(0, deadline - time.monotonic()))
            return result if time.monotonic() < deadline else ToolResult(error_code="timeout")
        except TimeoutError:
            return ToolResult(error_code="timeout")
        finally:
            if waiting.cancelled():
                future.cancel()

    def _invoke(self, name, arguments, principal, deadline):
        token = _WORKER_DEADLINE.set(deadline)
        try:
            _check_deadline(deadline)
            if self._resolve_scope is not None:
                resolved = self._resolve_scope(arguments, deadline)
                if resolved != arguments and not self.allows(principal, name, resolved):
                    return ToolResult(error_code="forbidden")
                arguments = resolved
            _check_deadline(deadline)
            data = self._handlers[name](arguments, deadline)
            _check_deadline(deadline)
            if len(canonical_json(data)) > MAX_RESULT_BYTES:
                return ToolResult(error_code="internal_error")
            return ToolResult(data=data)
        except _ToolError as exc:
            return ToolResult(error_code=exc.code)
        except (sqlite3.Error, StorageError):
            code = "timeout" if time.monotonic() >= deadline else "unavailable"
            return ToolResult(error_code=code)
        except Exception:
            return ToolResult(error_code="internal_error")
        finally:
            _WORKER_DEADLINE.reset(token)

    def close(self):
        with self._close_lock:
            with self._lock:
                if self._closed:
                    return
                self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)
            for close in self._closers:
                close()


def _excerpt(text: Any, limit: int = 2400) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    # Treat contiguous pipe rows as a block, including tables without outer pipes.
    prefix = text[:limit - 48]
    boundary = prefix.rfind("\n") + 1
    last_lines = prefix[:boundary].splitlines(keepends=True)
    next_line_end = text.find("\n", boundary)
    if next_line_end == -1:
        next_line_end = len(text)
    if text.find("|", boundary, next_line_end) >= 0:
        while last_lines and "|" in last_lines[-1]:
            last_lines.pop()
        return "".join(last_lines) + "[Markdown table omitted]\n[Excerpt truncated]"
    return (prefix[:boundary] if boundary else prefix) + "\n[Excerpt truncated]"


def _bounded_page(page):
    return type(page) is int and 1 <= page <= 100000


def _corpus_paths(corpus_root):
    raw = Path(corpus_root).expanduser().absolute()
    root = validate_safe_path(raw, raw.parent)
    if not root.is_dir() or raw.is_symlink() or root.stat().st_mode & 0o022:
        raise ValueError("Invalid protocol corpus configuration")
    files = {}
    for key, name in {
        "facts": "finarag_facts.sqlite",
        "docs": "finarag_docs.sqlite",
        "toc": "finarag_toc_trees.sqlite",
        "kg": "finarag_kg.sqlite",
    }.items():
        files[key] = _corpus_file(root, name)
    unified = validate_safe_path(root / "unified_store", root)
    if unified.is_dir():
        if unified.stat().st_mode & 0o022:
            raise ValueError("Invalid protocol corpus configuration")
        files["hybrid"] = _corpus_file(unified, "finarag_hybrid_meta.sqlite")
        files["faiss"] = _corpus_file(unified, "finarag_unified.faiss")
    if not any(files.values()):
        raise ValueError("Protocol corpus has no supported databases")
    return root, files


def _corpus_file(root, name):
    target = validate_safe_path(root / name, root)
    if target.suffix == ".sqlite":
        for suffix in ("-wal", "-shm", "-journal"):
            validate_safe_path(Path(str(target) + suffix), root)
    if target.is_file():
        if target.stat().st_mode & 0o022:
            raise ValueError("Invalid protocol corpus configuration")
        return target
    return None


def _guard_readonly_connection(conn):
    # These adapters reuse a connection per worker. The progress handler reads
    # the deadline of the current call, including when that connection is reused.
    conn.set_progress_handler(lambda: time.monotonic() >= _WORKER_DEADLINE.get(), 5000)
    remaining_ms = int(max(0, min(100, (_WORKER_DEADLINE.get() - time.monotonic()) * 1000)))
    conn.execute(f"PRAGMA busy_timeout = {remaining_ms}")
    return conn


def build_financial_tool_registry(corpus_root, *, policy, dense_encoder=None, sparse_encoder=None):
    """Bind the allow-listed tool adapters to a deployment-owned local corpus."""
    from src.claim_verifier import ClaimVerifier
    from src.knowledge_graph.kg_builder import FinancialKGStore
    from src.knowledge_graph.lightrag_retriever import LightRAGRetriever
    from src.metric_atoms import MetricAtomStore
    from src.multimodal_nav.toc_tree_builder import TocTreeStore
    from src.multimodal_nav.tree_navigator import TocTreeNavigator
    from src.retrieval_filters import RetrievalFilters
    from src.storage.qdrant_store import QdrantHybridStore, UnifiedHybridRetriever
    from src.storage.sqlite_doc_store import SqliteDocumentRepository
    from src.structured.pot_sandbox import (
        PotCalculationError, PotSandbox, PotSecurityError, PotVariable, normalize_monetary_unit,
    )
    from src.structured.sql_store import FinancialSQLStore, StructuredQueryError

    class BoundedCorpusReader:
        def _get_conn(self):
            return _guard_readonly_connection(super()._get_conn())

    class DocumentReader(BoundedCorpusReader, SqliteDocumentRepository):
        pass

    class TocReader(BoundedCorpusReader, TocTreeStore):
        pass

    class KgReader(BoundedCorpusReader, FinancialKGStore):
        pass

    class HybridReader(QdrantHybridStore):
        def _get_embedded_conn(self):
            return _guard_readonly_connection(super()._get_embedded_conn())

    root, paths = _corpus_paths(corpus_root)
    docs = DocumentReader(paths["docs"], allowed_base_dir=root) if paths["docs"] else None
    toc = TocReader(paths["toc"], allowed_base_dir=root) if paths["toc"] else None
    kg = KgReader(paths["kg"], allowed_base_dir=root) if paths["kg"] else None
    hybrid = (HybridReader(mode="embedded", store_dir=root / "unified_store", allowed_base_dir=root)
              if paths.get("hybrid") else None)
    if hybrid:
        hybrid.meta_db_path = paths["hybrid"]
        if paths.get("faiss"):
            hybrid.faiss_path = paths["faiss"]
    atoms = MetricAtomStore.load() if paths["facts"] else None

    def report(doc_id, deadline, *, company=None, year=None):
        _check_deadline(deadline)
        if docs is None:
            raise _ToolError("unavailable")
        canonical = docs.resolve_doc_id(doc_id)
        meta = docs.get_metainfo(canonical)
        if (not meta or re.fullmatch(DOC_ID_PATTERN, canonical) is None
                or company is not None and meta.get("company_name") != company
                or year is not None and str(meta.get("report_year") or meta.get("fiscal_year")) != str(year)):
            raise _ToolError("not_found")
        return canonical, meta

    def page(doc_id, page_number, deadline):
        _check_deadline(deadline)
        return docs.get_page(doc_id, page_number) if _bounded_page(page_number) else None

    def resolve_scope(args, deadline):
        def canonical(doc_id):
            return report(doc_id, deadline)[0]

        def resolve_reference(item):
            doc_id = item.get("doc_id")
            return {**item, "doc_id": canonical(doc_id)} if doc_id is not None else item

        if "doc_id" in args:
            return {**args, "doc_id": canonical(args["doc_id"])}
        if "doc_ids" in args:
            return {**args, "doc_ids": [canonical(doc_id) for doc_id in args["doc_ids"]]}
        for key in ("evidence", "variables"):
            if key in args:
                return {**args, key: [resolve_reference(item) for item in args[key]]}
        return args

    class ScopedFacts(FinancialSQLStore):
        def __init__(self, scope, industry, unit, deadline):
            super().__init__(paths["facts"], allowed_base_dir=root)
            self.scope = scope
            self.industry = industry
            self.unit = unit
            self.deadline = deadline

        def _open_readonly(self):
            conn = _guard_readonly_connection(super()._open_readonly())
            steps = 0

            def guard():
                nonlocal steps
                steps += 5000
                return steps > 500000 or time.monotonic() >= self.deadline

            conn.set_progress_handler(guard, 5000)
            return conn

        def _run_parameterized(self, sql, params=()):
            # Scope and validate rows before upstream compare/rank/threshold deduplication.
            scoped = """FROM (SELECT * FROM financial_facts WHERE scope_type = ?) f
                         JOIN (SELECT * FROM reports WHERE (? IS NULL OR industry_l1 = ?)) r"""
            original = "FROM financial_facts f\n            JOIN reports r"
            if original not in sql:
                raise _ToolError("internal_error")
            sql = sql.replace(original, scoped)
            _check_deadline(self.deadline)
            rows = super()._run_parameterized(sql, (self.scope, self.industry, self.industry, *params))
            valid = []
            for row in rows:
                raw_unit = normalize_monetary_unit(row.get("raw_unit"))
                raw_value = row.get("raw_value")
                if (not _bounded_page(row.get("source_page")) or type(raw_value) not in (int, float)
                        or not math.isfinite(raw_value) or row["scope_type"] != self.scope):
                    continue
                if self.unit in _MONEY_FACTORS:
                    factor = _MONEY_FACTORS.get(raw_unit)
                    value = row.get("normalized_cny")
                    if (factor is None or type(value) not in (int, float)
                            or str(row.get("currency") or "").upper() not in {"CNY", "RMB", "人民币"}
                            or not math.isclose(value, raw_value * factor, rel_tol=1e-9, abs_tol=1e-9)):
                        continue
                elif raw_unit != self.unit:
                    continue
                else:
                    row["normalized_cny"] = None
                valid.append(row)
            return valid

        def query_company_metric(self, *args, **kwargs):
            kwargs["exclude_pct"] = self.unit in _MONEY_FACTORS
            return super().query_company_metric(*args, **kwargs)

    def query_facts(args, deadline):
        if atoms is None:
            raise _ToolError("unavailable")
        atom = atoms.get(args["metric_atom_id"])
        if atom is None:
            raise _ToolError("not_found")
        operation, period = args["operation"], args["period"]
        unit = normalize_monetary_unit(atom.unit)
        if operation in {"rank", "threshold"} and unit not in _MONEY_FACTORS:
            raise _ToolError("invalid_arguments")
        store = ScopedFacts(args["scope"], args["industry_l1"], unit, deadline)
        metric = {"metric_atom_id": atom.atom_id, "period": period, "canonical_name": atom.standard_name}
        try:
            if operation == "metric":
                rows = store.query_company_metric(
                    company_name=args["company_name"], prefer_consolidated=args["scope"] == "consolidated", **metric)
            elif operation == "compare":
                rows = [row for row in store.query_compare_companies(args["companies"], **metric).values() if row]
            elif operation == "rank":
                rows = store.query_company_ranking(top_k=args["limit"], ascending=args["ascending"], **metric)
            else:
                rows = store.query_threshold_companies(threshold_cny=args["threshold_cny"], operator=args["operator"], **metric)
        except StructuredQueryError:
            _check_deadline(deadline)
            raise _ToolError("unavailable") from None
        fields = ("fact_id", "doc_id", "company_name", "currency", "source_page", "period",
                  "raw_value", "raw_unit", "normalized_cny", "scope_type", "table_id",
                  "row_idx", "col_header", "statement_type", "metric_atom_id", "metric_raw")
        return {"rows": [{key: _excerpt(row[key], 512) if isinstance(row.get(key), str) else row.get(key)
                          for key in fields} for row in rows[:args["limit"]]]}

    def calculate(args, deadline):
        variables = []
        for item in args["variables"]:
            if item["doc_id"] is not None:
                if page(item["doc_id"], item["page"], deadline) is None:
                    raise _ToolError("not_found")
            variables.append(PotVariable(**item))
        _check_deadline(deadline)
        try:
            result = PotSandbox().evaluate(variables, args["expression"], args["result_unit"], args["ndigits"])
        except (PotCalculationError, PotSecurityError, ArithmeticError):
            raise _ToolError("invalid_arguments") from None
        return {**asdict(result), "inputs": [asdict(item) for item in variables]}

    def navigate(args, deadline):
        if toc is None:
            raise _ToolError("unavailable")
        doc_id, meta = report(args["doc_id"], deadline)
        tree = toc.get_tree(doc_id)
        if tree is None:
            raise _ToolError("not_found")
        navigator = TocTreeNavigator()
        options = {"max_sections": args["max_sections"], "llm_fn": None}
        sections = navigator.select_sections(tree, args["question"], **options)
        hits = navigator.retrieve_section_pages(
            tree, args["question"], lambda p: page(doc_id, p, deadline),
            doc_metainfo=meta, max_pages=args["max_pages"], **options,
        )
        return {
            "doc_id": doc_id,
            "sections": [
                {"title": _excerpt(item["title"], 512), "section_path": _excerpt(item["section_path"], 512),
                 "start_page": item["start_page"], "end_page": item["end_page"]}
                for item in sections
            ],
            "pages": [
                {"doc_id": doc_id, "page": hit["page"], "section_path": _excerpt(hit["metadata"]["section_path"], 512),
                 "excerpt": _excerpt(hit["text"])}
                for hit in hits
            ],
        }

    @lru_cache(maxsize=1)
    def kg_retriever():
        return LightRAGRetriever(kg)

    def query_kg(args, deadline):
        if kg is None:
            raise _ToolError("unavailable")
        scoped = {report(doc, deadline)[0] for doc in args["doc_ids"]}
        fetched = {}

        def scoped_page(doc_id, number):
            if doc_id not in scoped or not _bounded_page(number):
                return None
            key = (doc_id, number)
            if key not in fetched:
                fetched[key] = page(doc_id, number, deadline)
            return fetched[key]

        def has_provenance(edge):
            doc_id, number = edge["evidence_doc_id"], edge["evidence_page"]
            if doc_id not in scoped or not _bounded_page(number):
                return False
            # The P3 graph converts NULL/0 pages to 1 when loading edges.
            # Validate the source value before treating that page as evidence.
            row = kg._get_conn().execute(
                "SELECT evidence_page FROM kg_edges WHERE edge_id = ? AND evidence_doc_id = ?",
                (edge["edge_id"], doc_id),
            ).fetchone()
            return row is not None and row["evidence_page"] == number and bool(scoped_page(doc_id, number))

        retriever = kg_retriever()
        _check_deadline(deadline)
        found = retriever.retrieve(
            args["question"], company_name=args["company_name"], page_fetcher=scoped_page,
            max_hops=args["max_hops"], max_paths=args["max_paths"], max_pages=args["max_pages"],
        )
        paths_out = []
        evidence = {}
        for path in found["paths"]:
            edges = path.get("edges") or []
            if not edges or not all(has_provenance(edge) for edge in edges):
                continue
            public_edges = []
            for edge in edges:
                doc_id, number = edge["evidence_doc_id"], edge["evidence_page"]
                evidence[(doc_id, number)] = fetched[(doc_id, number)]
                public_edges.append({
                    "head": _excerpt(edge["head"], 256), "tail": _excerpt(edge["tail"], 256),
                    "relation_type": _excerpt(edge["relation_type"], 128),
                    "evidence_doc_id": doc_id, "evidence_page": number,
                })
            summary = " | ".join(f"{e['head']} -[{e['relation_type']}]-> {e['tail']} (p{e['evidence_page']})"
                                 for e in public_edges)
            paths_out.append({"hops": len(public_edges), "score": path["score"], "summary": summary,
                              "edges": public_edges})
        return {"paths": paths_out, "context_summary": "\n".join(p["summary"] for p in paths_out),
                "evidence_pages": [{"doc_id": doc, "page": number, "excerpt": _excerpt(obj["text"])}
                                   for (doc, number), obj in list(evidence.items())[:args["max_pages"]]]}

    def search(args, deadline):
        if hybrid is None:
            raise _ToolError("unavailable")
        scoped = list(dict.fromkeys(report(doc, deadline, company=args["company_name"],
                                          year=args["report_year"])[0] for doc in args["doc_ids"]))

        def encode_dense(query):
            import numpy as np

            _check_deadline(deadline)
            index = hybrid._get_faiss_index()
            vector = np.asarray(dense_encoder(query), dtype=np.float32)
            if vector.shape not in {(index.d,), (1, index.d)} or not np.isfinite(vector).all() or not vector.any():
                return None
            return vector

        def encode_sparse(query):
            _check_deadline(deadline)
            weights = sparse_encoder(query)
            if (not isinstance(weights, dict) or not 0 < len(weights) <= 1024
                    or any(not isinstance(key, str) or not 0 < len(key) <= 64
                           or type(value) not in (int, float) or not -1e9 <= value <= 1e9
                           for key, value in weights.items())):
                return None
            return weights

        _check_deadline(deadline)
        retriever = UnifiedHybridRetriever(
            hybrid, docs, dense_encoder=encode_dense if dense_encoder is not None and paths.get("faiss") else None,
            sparse_encoder=encode_sparse if sparse_encoder is not None else None, lazy_load_encoders=False,
        )
        hits = retriever.retrieve_candidates_by_company_name(
            args["company_name"], args["query"], top_n=args["top_k"],
            candidate_doc_ids=scoped, filters=RetrievalFilters(year=args["report_year"]), backend_scope="all",
        )
        result = []
        for hit in hits:
            meta = hit["metadata"]
            doc_id = meta.get("doc_id") or meta.get("sha1_name")
            if doc_id in scoped and _bounded_page(hit["page"]):
                result.append({"doc_id": doc_id, "page": hit["page"], "chunk_id": meta.get("chunk_id"),
                               "score": hit["combined_score"], "retrieval_sources": hit["retrieval_sources"],
                               "excerpt": _excerpt(hit["text"], 1500)})
        mode = "hybrid" if any({"dense", "bge_sparse"} & set(hit["retrieval_sources"])
                               for hit in result) else "lexical_only"
        return {"mode": mode, "results": result}

    def verify(args, deadline):
        results = []
        for ref in args["evidence"]:
            doc_id, _ = report(ref["doc_id"], deadline, company=args["company_name"], year=args["report_year"])
            evidence = page(doc_id, ref["page"], deadline)
            if evidence is None:
                raise _ToolError("not_found")
            results.append({"pdf_sha1": doc_id, "page": ref["page"], "text": _excerpt(evidence["text"], 4000),
                            "metadata": {"doc_id": doc_id}})
        _check_deadline(deadline)
        audit = ClaimVerifier(nli_scorer_fn=None).verify(
            args["answer_text"], retrieval_results=results, company_name=args["company_name"],
            report_year=args["report_year"],
        ).to_dict()
        # The audit counts apply to the complete answer even when the detail is capped.
        audit.pop("latency_ms", None)
        audit["claims"] = audit["claims"][:20]
        for claim in audit["claims"]:
            claim["claim"] = _excerpt(claim["claim"], 512)
            claim["contradiction_reason"] = _excerpt(claim["contradiction_reason"], 512)
            claim["numeric_checks"] = claim["numeric_checks"][:8]
            for check in claim["numeric_checks"]:
                check["raw"] = _excerpt(check["raw"], 128)
                check["reason"] = _excerpt(check["reason"], 256)
        return audit

    registry = MCPToolRegistry(handlers={
        "query_financial_facts_sql": query_facts, "pot_calculate": calculate,
        "navigate_toc_sections": navigate, "query_financial_kg": query_kg,
        "hybrid_search_reports": search, "verify_claims": verify,
    }, policy=policy)
    registry._closers = tuple(store.close for store in (docs, toc, kg, hybrid) if store is not None)
    registry._resolve_scope = resolve_scope
    return registry
