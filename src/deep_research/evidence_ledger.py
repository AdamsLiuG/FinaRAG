"""Scoped evidence admission, normalization and adjudication."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, localcontext
from typing import TYPE_CHECKING

from src.retrieval_filters import build_result_metadata
from src.structured.pot_sandbox import normalize_monetary_unit

if TYPE_CHECKING:
    from src.deep_research.planner import ResearchTask
    from src.protocols.tool_registry import ToolResult

_MONEY = {"元": 1, "千元": 1000, "万元": 10000, "百万元": 1000000,
          "千万元": 10000000, "亿元": 100000000}
_RANK = {"sql": 0, "table": 1, "kg": 2, "narrative": 3, "derived": 4}
_CURRENCIES = {"CNY": "CNY", "RMB": "CNY", "人民币": "CNY", "USD": "USD", "HKD": "HKD"}
_CNY_UNIT = re.compile(r"(?:金额)?单位[:：](?:人民币)?(千万元|百万元|亿元|万元|千元|元)")
_PERIOD = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_RELATION_CUES = {
    "MAJOR_SHAREHOLDER": ("前十名股东", "持股", "股东", "shareholder"),
    "ACTUAL_CONTROLLER": ("实际控制人", "controller"),
    "HOLDS_SUBSIDIARY": ("子公司", "subsidiar"),
    "EQUITY_INVESTMENT": ("参股", "股权投资", "equity investment"),
    "RELATED_PARTY_TX": ("关联交易", "关联方", "related part"),
}


def _decimal(value):
    text = str(value).replace(",", "").strip()
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1].strip()
    number = Decimal(text)
    if not number.is_finite() or abs(number) > Decimal("1e24"):
        raise ValueError("Invalid amount")
    return number


def measurement_basis(text):
    text = re.sub(r"\s+", " ", text).lower()
    labels = (
        ("unadjusted", r"未调整|调整前|unadjusted"),
        ("adjusted", r"扣非|扣除非经常性|经调整|调整后|(?<!un)adjusted"),
        ("original", r"重述前|originally reported|\boriginal\b"),
        ("restated", r"重述后|经重述|(?<!未)重述(?!前)|restated"),
        ("per_share", r"每股|per.share|/股"),
    )
    matched = [label for label, pattern in labels if re.search(pattern, text)]
    return "+".join(matched) or "reported"


def metric_basis(metric, basis):
    if metric == "MET-001":
        if basis == "reported":
            return "adjusted"
        if "adjusted" not in basis.split("+") and "unadjusted" not in basis.split("+"):
            return f"adjusted+{basis}"
    return basis


def _amount(value, unit, currency):
    amount = _decimal(value)
    unit = normalize_monetary_unit(unit)
    if unit in _MONEY:
        code = _CURRENCIES.get(str(currency or "").upper())
        if not code:
            raise ValueError("Unknown currency")
        if code != "CNY" and str(unit) != "元":
            raise ValueError("Unknown currency scale")
        with localcontext() as context:
            context.prec = 60
            return amount * _MONEY[unit], f"{code}:money"
    if unit in {"%", "％"}:
        return amount, "percent"
    if unit in {"元/股", "元／股"} and _CURRENCIES.get(str(currency or "").upper()) == "CNY":
        return amount, "CNY:per_share"
    raise ValueError("Unknown unit")


def _compact(text):
    return re.sub(r"\s+", "", str(text)).casefold()


def _signature(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()[:24]


@dataclass(frozen=True, order=True)
class EvidenceKey:
    entity_id: str
    metric_atom: str
    period: str
    scope: str
    dimension: str
    basis: str
    segment: str = ""


@dataclass(frozen=True)
class EvidenceEntry:
    evidence_id: str
    key: EvidenceKey
    task_id: str
    tool_name: str
    source_class: str
    doc_id: str
    page: int | None
    report_year: int | None = None
    company_name: str = ""
    label: str = ""
    excerpt: str = ""
    raw_value: str | None = None
    normalized_value: Decimal | None = None
    unit: str = ""
    table_id: str | None = None
    row_idx: int | None = None
    column: str | None = None
    source_id: str | None = None
    precision: int | None = None
    expression: str | None = None
    inputs: tuple[str, ...] = ()

    def to_dict(self):
        value = asdict(self)
        value["inputs"] = list(self.inputs)
        if self.normalized_value is not None:
            value["normalized_value"] = format(self.normalized_value, "f")
        return value


@dataclass(frozen=True)
class EvidenceResolution:
    key: EvidenceKey
    winner_id: str | None
    competing_ids: tuple[str, ...]
    disagreements: tuple[str, ...]
    reason: str


def _equivalent(left, right):
    if left.normalized_value == right.normalized_value:
        return True
    if left.precision is None and right.precision is None:
        return False
    precision = min(p for p in (left.precision, right.precision) if p is not None)
    with localcontext() as context:
        context.prec = 60
        return round(left.normalized_value, precision) == round(right.normalized_value, precision)


class EvidenceLedger:
    def __init__(self, *, document_repository=None):
        self._entries = {}
        self._notes = {}
        self._repository = document_repository

    @property
    def entries(self):
        return tuple(self._entries.values())

    def entry(self, evidence_id):
        return self._entries[evidence_id]

    def metadata(self, entry):
        meta = self._repository.get_metainfo(entry.doc_id) if self._repository else {}
        return build_result_metadata(
            {**(meta or {}), "sha1_name": entry.doc_id, "company_name": entry.company_name,
             "report_year": entry.report_year},
            {"chunk_id": entry.source_id, "table_id": entry.table_id, "evidence_type": entry.source_class},
        )

    def _record(self, task, key, source, **fields):
        candidate = EvidenceEntry("", key, task.task_id, task.tool_name, source, **fields)
        evidence_id = "ev-" + _signature(candidate.to_dict())
        self._entries.setdefault(evidence_id, EvidenceEntry(
            evidence_id, key, task.task_id, task.tool_name, source, **fields,
        ))
        return evidence_id

    def _note(self, task, reason, *, row=None, **details):
        row = row or {}
        note = {"task_id": task.task_id, "reason": reason, **details}
        for key in ("doc_id", "source_page", "fact_id", "col_header", "scope_type", "period", "raw_value"):
            if row.get(key) is not None:
                note[key] = str(row[key])[:160]
        self._notes.setdefault(_signature(note), note)

    def _location(self, task, row, page):
        doc = row.get("doc_id")
        if not doc or type(page) is not int or not 0 < page <= 100000:
            self._note(task, "missing_provenance", row=row)
            return None
        if doc not in task.doc_ids or row.get("company_name", task.company_name) != task.company_name:
            self._note(task, "out_of_scope", row=row)
            return None
        if self._repository and not self._repository.get_page(doc, page):
            self._note(task, "missing_provenance", row=row)
            return None
        return doc

    def _numeric(self, task, row, default_source):
        source = row.get("source_class", default_source)
        page = row.get("source_page")
        doc = self._location(task, row, page)
        if not doc:
            return None
        column = row.get("col_header")
        row_index = row.get("row_idx")
        label = str(row.get("metric_raw") or "")[:160]
        table_id = row.get("table_id")
        source_id = row.get("fact_id") if source == "sql" else row.get("edge_id") if source == "kg" else row.get("chunk_id")
        if source in {"sql", "table"}:
            provenance = (table_id and type(row_index) is int and row_index >= 0 and column and label
                          and (source_id if source == "sql" else type(row.get("col_idx")) is int))
        else:
            provenance = source_id if source in {"kg", "narrative"} else None
        if not provenance:
            self._note(task, "missing_provenance", row=row)
            return None
        period = str(row.get("period") or "")
        metric = row.get("metric_atom_id")
        scope = row.get("scope_type")
        if not re.fullmatch(r"\d{4}", period) or not metric or scope not in {"parent", "consolidated"}:
            self._note(task, "incompatible_caliber", row=row)
            return None
        header = str(column or "")
        header_years = set(_PERIOD.findall(header))
        observed_scope = ("parent" if re.search(r"(?:年(?:度)?\s*公司\s*$|母公司|parent)", header, re.I) else
                          "consolidated" if re.search(r"合并|consolidated", header, re.I) else scope)
        if observed_scope != scope:
            self._note(task, "scope_label_mismatch", row=row, observed_scope=observed_scope)
            scope = observed_scope
        if (header_years and header_years != {period}
                or re.search(r"增减|增加额|变动率|同比|change|growth", header, re.I)):
            self._note(task, "incompatible_caliber", row=row)
            return None
        unit = row.get("raw_unit")
        raw = row.get("raw_value")
        try:
            number, dimension = _amount(raw, unit, row.get("currency"))
        except (InvalidOperation, TypeError, ValueError):
            self._note(task, "invalid_amount_or_unit", row=row)
            return None
        basis = metric_basis(metric, str(row.get("measurement_basis") or measurement_basis(f"{label} {header} {unit}")))
        key = EvidenceKey(task.entity_id, metric, period, scope, dimension, basis, str(row.get("segment") or ""))
        if key not in task.required_keys:
            self._note(task, "incompatible_caliber", row=row, excluded_key=asdict(key))
        year = task.report_years.get(doc) or row.get("report_year")
        precision = row.get("rounding_precision")
        return self._record(
            task, key, source, doc_id=doc, page=page, report_year=year, company_name=task.company_name,
            label=label, excerpt=str(row.get("excerpt") or f"{label} | {header} | {raw} {unit}")[:400],
            raw_value=str(raw), normalized_value=number, unit=unit, table_id=str(table_id) if table_id else None,
            row_idx=row_index, column=column, source_id=str(source_id) if source_id else None,
            precision=precision if type(precision) is int and -12 <= precision <= 12 else None,
        )

    def _table(self, task, hit):
        if not self._repository or not task.metric_label:
            return
        page = hit.get("page")
        doc = self._location(task, hit, page)
        if not doc:
            return
        meta = self._repository.get_metainfo(doc) or {}
        page_text = _compact((self._repository.get_page(doc, page) or {}).get("text", ""))
        for table in self._repository.get_tables(doc, page):
            markdown = str(table.get("markdown") or "").strip()
            if not markdown:
                continue
            # A parser hint can be inherited from a percentage column or omit a scale prefix.
            # Read the published unit before this specific table when its header does not declare it.
            header_text = _compact(markdown.splitlines()[0])
            direct_unit = _CNY_UNIT.search(header_text)
            header_start = page_text.find(header_text)
            before_table = page_text[:header_start][-160:] if header_start >= 0 else ""
            nearby_units = _CNY_UNIT.findall(before_table)
            unit = direct_unit[1] if direct_unit else nearby_units[-1] if nearby_units else None
            for cell in table.get("cell_records") or ():
                label = " ".join(cell.get("matched_row_headers") or ())
                header = " ".join(cell.get("matched_col_headers") or ())
                if _compact(label) != _compact(task.metric_label):
                    continue
                years = _PERIOD.findall(header)
                if len(years) != 1 or (cell.get("period") and str(cell["period"]) != years[0]):
                    continue
                period = years[0]
                # A generic summary column gets its scope from the published table title.
                title = str(table.get("caption") or table.get("title") or "")
                scope_context = _compact(f"{header} {title}")
                if "母公司" in scope_context or re.search(r"年度?公司$", _compact(header)):
                    scope = "parent"
                elif "合并" in scope_context or "主要会计数据" in scope_context:
                    scope = "consolidated"
                elif "主要会计数据" in _compact(markdown[:400]) or "主要会计数据" in before_table:
                    scope = "consolidated"
                else:
                    continue
                for key in task.required_keys:
                    if key.period == period:
                        self._numeric(task, {
                            "doc_id": doc, "source_page": page, "period": period, "scope_type": scope,
                            "table_id": table.get("table_id"), "col_header": header, "metric_raw": label,
                            "row_idx": cell.get("row_idx"), "col_idx": cell.get("col_idx"),
                            "metric_atom_id": key.metric_atom, "currency": meta.get("currency"),
                            "raw_value": cell.get("raw_value"), "raw_unit": unit,
                        }, "table")

    def _observe(self, task, row):
        text = str(row.get("excerpt") or "")
        if row.get("doc_id") not in task.doc_ids or not text:
            return
        self._table(task, row)
        cues = tuple(_compact(cue) for cue in task.cues)
        if not cues:
            return
        page = row.get("page")
        doc = self._location(task, row, page)
        if not doc:
            return
        for line in text.splitlines():
            compact = _compact(line)
            if not any(cue in compact and len(compact) >= len(cue) + 4 for cue in cues):
                continue
            if re.match(r"^\s*(?:#|第.{0,6}节)", line):
                continue
            self._record(
                task, task.required_keys[0], "narrative", doc_id=doc, page=page,
                report_year=task.report_years.get(doc), company_name=task.company_name,
                excerpt=" ".join(line.split())[:350], source_id=str(row["chunk_id"]) if row.get("chunk_id") else None,
            )
            break

    def _relations(self, task, data):
        pages = {(p.get("doc_id"), p.get("page")): p.get("excerpt") for p in data.get("evidence_pages", ())}
        seen = set()
        for path in data.get("paths", ()):
            for edge in path.get("edges", ()):
                doc, page = edge.get("evidence_doc_id"), edge.get("evidence_page")
                relation = edge.get("relation_type")
                head, tail = str(edge.get("head") or ""), str(edge.get("tail") or "")
                identity = (doc, page, head, relation, tail)
                if identity in seen or doc not in task.doc_ids:
                    continue
                seen.add(identity)
                if relation not in task.relations:
                    continue
                text = pages.get((doc, page))
                if self._repository and type(page) is int and page > 0:
                    text = (self._repository.get_page(doc, page) or {}).get("text")
                cues = _RELATION_CUES.get(relation, ())
                lines = (text or "").splitlines()
                match = None
                for idx, line in enumerate(lines):
                    compact = _compact(line)
                    issuer = _compact(task.company_name)
                    if not head or not tail or not all(
                        party == issuer or party in compact for party in (_compact(head), _compact(tail))
                    ):
                        continue
                    # Shareholder/transaction tables may put the relation in the preceding heading.
                    context = _compact(" ".join(lines[max(0, idx - 8):idx + 1]))
                    if any(_compact(cue) in context for cue in cues):
                        match = " ".join(line.split())[:350]
                        break
                if not match or not self._location(task, {"doc_id": doc}, page):
                    self._note(task, "relationship_uncorroborated", row={"doc_id": doc, "source_page": page},
                               relation=relation, head=head[:160], tail=tail[:160])
                    continue
                self._record(
                    task, task.required_keys[0], "kg", doc_id=doc, page=page, report_year=task.report_years.get(doc),
                    company_name=task.company_name, label=f"{head} → {relation} → {tail}", excerpt=match,
                    source_id=str(edge.get("edge_id") or _signature(identity)),
                )

    def ingest(self, task: ResearchTask, tool_result: ToolResult, *, inputs=()):
        """Admit supported evidence in the caller's deterministic publication order."""
        if tool_result.error_code or not isinstance(tool_result.data, dict):
            return ()
        before = set(self._entries)
        data = tool_result.data
        if task.kind == "pot_calc":
            if (inputs and data.get("expression") == task.arguments["expression"]
                    and data.get("unit") == task.arguments["result_unit"]):
                try:
                    value = _decimal(data.get("value"))
                except (InvalidOperation, ValueError):
                    return ()
                self._record(
                    task, task.required_keys[0], "derived", doc_id="", page=None, company_name=task.company_name,
                    label=task.metric_label, raw_value=str(data["value"]), normalized_value=value, unit=data["unit"],
                    expression=data["expression"], inputs=tuple(inputs),
                )
        elif task.kind == "kg_multihop":
            self._relations(task, data)
        elif task.kind == "sql_fact":
            for row in data.get("rows", ()):
                self._numeric(task, row, "sql")
        else:
            for hit in data.get("pages", data.get("results", ())):
                self._observe(task, hit)
        # Typed evidence supplied by a trusted adapter still passes the same admission rules.
        for row in data.get("evidence", ()):
            self._numeric(task, row, "narrative")
        return tuple(eid for eid in self._entries if eid not in before)

    def adjudicate(self):
        groups = defaultdict(list)
        derived = []
        for entry in self.entries:
            if entry.source_class == "derived":
                derived.append(entry)
            else:
                groups[entry.key].append(entry)
        resolutions = {key: self._resolve(key, entries) for key, entries in groups.items()}
        for entry in derived:
            ids = {resolution.winner_id for resolution in resolutions.values() if resolution.winner_id}
            if all(eid in ids for eid in entry.inputs):
                groups[entry.key].append(entry)
                resolutions[entry.key] = self._resolve(entry.key, groups[entry.key])
            else:
                resolutions[entry.key] = EvidenceResolution(
                    entry.key, None, (entry.evidence_id,), (), "upstream_unresolved",
                )
        return dict(sorted(resolutions.items()))

    @staticmethod
    def _resolve(key, entries):
        ordered = sorted(entries, key=lambda e: (_RANK[e.source_class], e.doc_id, e.page or 0, e.evidence_id))
        first = ordered[0]
        tied = [entry for entry in ordered if entry.source_class == first.source_class]
        competing = tuple(entry.evidence_id for entry in ordered)
        if any(not _equivalent(left, right) for idx, left in enumerate(tied) for right in tied[idx + 1:]):
            return EvidenceResolution(key, None, competing, competing, "same_priority_conflict")
        dissent = tuple(entry.evidence_id for entry in ordered[len(tied):] if not _equivalent(first, entry))
        reason = "higher_priority_source" if dissent else "agreement" if len(ordered) > 1 else "single_source"
        return EvidenceResolution(key, first.evidence_id, competing, dissent, reason)

    def coverage(self, required_keys):
        resolutions = self.adjudicate()
        return all(key in resolutions and resolutions[key].winner_id for key in required_keys)

    def to_dict(self):
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "resolutions": [
                {**asdict(result), "competing_ids": list(result.competing_ids),
                 "disagreements": list(result.disagreements)}
                for result in self.adjudicate().values()
            ],
            "notes": list(self._notes.values()),
        }
