"""Plan and execute scoped, typed research waves through the P4 tools."""

from __future__ import annotations

import asyncio
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import FunctionType, MappingProxyType, SimpleNamespace
from typing import Literal

import networkx as nx
from langgraph.runtime import Runtime

from src.deep_research.evidence_ledger import EvidenceKey, EvidenceLedger, measurement_basis, metric_basis
from src.graph.state import FinaRAGRunContext
from src.harness.trajectory_store import capture_active, capture_step, current_observer, current_prefix, step_scope
from src.protocols.tool_registry import (
    DOC_ID_PATTERN,
    FinancialFactsInput,
    HybridInput,
    KgInput,
    MCPToolRegistry,
    PotInput,
    TocInput,
    ToolPrincipal,
    ToolResult,
    _corpus_paths,
    build_financial_tool_registry,
    default_policy,
)
from src.query_plan import QueryPlan
from src.storage.base import StorageError

TaskKind = Literal["sql_fact", "kg_multihop", "toc_nav", "hybrid_rag", "pot_calc"]
TaskStatus = Literal["ok", "empty", "failed", "timed_out", "blocked", "skipped_sufficient", "skipped_budget"]
_TOOLS = {
    "sql_fact": ("query_financial_facts_sql", FinancialFactsInput),
    "kg_multihop": ("query_financial_kg", KgInput),
    "toc_nav": ("navigate_toc_sections", TocInput),
    "hybrid_rag": ("hybrid_search_reports", HybridInput),
    "pot_calc": ("pot_calculate", PotInput),
}
_TOPICS = {
    "related_party": (("关联方", "关联交易", "related party", "related-party"), ("RELATED_PARTY_TX",)),
    "shareholder": (("股东", "持股", "控股股东", "shareholder"), ("MAJOR_SHAREHOLDER", "ACTUAL_CONTROLLER")),
    "subsidiary": (("子公司", "控股路径", "股权投资", "subsidiar"), ("HOLDS_SUBSIDIARY", "EQUITY_INVESTMENT")),
    "risk": (("风险", "risk"), ()),
}
_CASH = ("现金流", "cash flow", "cash-flow")
_CHANGE = ("变化", "变动", "增减", "同比", "change", "delta")
_COMPARE = ("比较", "对比", "差异", "分别", "compare", "versus", "cross-report")
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_PROJECTED_ANALYSIS = re.compile(r"百分比|变动率|同比增幅|percentage change|growth rate", re.I)
_PRINCIPAL = ToolPrincipal("finarag-deep-research", frozenset({"read"}))


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _valid_docs(docs):
    return bool(docs) and len(docs) <= 12 and all(isinstance(doc, str) and re.fullmatch(DOC_ID_PATTERN, doc) for doc in docs)


@dataclass(frozen=True)
class ResearchTask:
    task_id: str
    kind: TaskKind
    tool_name: str
    arguments: Mapping
    entity_id: str
    company_name: str
    doc_ids: tuple[str, ...]
    required_keys: tuple[EvidenceKey, ...]
    depends_on: tuple[str, ...] = ()
    priority: int = 0
    fallback_for: tuple[str, ...] = ()
    retry_for: str = ""
    bindings: tuple[tuple[str, EvidenceKey], ...] = ()
    report_years: Mapping[str, int] = field(default_factory=dict)
    metric_label: str = ""
    cues: tuple[str, ...] = ()
    relations: tuple[str, ...] = ()

    def __post_init__(self):
        if (self.kind not in _TOOLS or self.tool_name != _TOOLS[self.kind][0]
                or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", self.task_id)
                or not self.entity_id or not self.company_name.strip() or not _valid_docs(self.doc_ids)
                or not self.required_keys or any(key.entity_id != self.entity_id for key in self.required_keys)
                or type(self.priority) is not int):
            raise ValueError("Invalid task or scope")
        for name in ("doc_ids", "required_keys", "depends_on", "fallback_for", "cues", "relations"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "bindings", tuple(tuple(binding) for binding in self.bindings))
        arguments = _plain(self.arguments)
        if not isinstance(arguments, dict):
            raise ValueError("Invalid tool arguments")
        if self.kind == "sql_fact":
            if arguments.get("operation") != "metric" or arguments.get("company_name") != self.company_name:
                raise ValueError("Invalid fact scope")
        if (("doc_id" in arguments and arguments["doc_id"] not in self.doc_ids)
                or ("doc_ids" in arguments and (not isinstance(arguments["doc_ids"], list)
                    or not set(arguments["doc_ids"]).issubset(self.doc_ids)))
                or arguments.get("company_name", self.company_name) != self.company_name):
            raise ValueError("Invalid tool scope")
        if self.kind == "pot_calc":
            if (not self.bindings or not self.depends_on or len(self.required_keys) != 1
                    or self.required_keys[0].dimension != "CNY:money" or arguments.get("result_unit") != "元"
                    or "variables" in arguments or self.fallback_for):
                raise ValueError("Invalid calculation bindings")
            sample = {**arguments, "variables": [{"name": name, "value": 0, "unit": "元"} for name, _ in self.bindings]}
        else:
            if self.bindings or self.retry_for:
                raise ValueError("Unexpected dependency bindings")
            sample = arguments
        _TOOLS[self.kind][1].model_validate(sample, strict=True)
        years = dict(self.report_years)
        if not set(years).issubset(self.doc_ids) or any(type(year) is not int or not 1000 <= year <= 9999
                                                       for year in years.values()):
            raise ValueError("Invalid report year")
        object.__setattr__(self, "report_years", MappingProxyType(years))
        object.__setattr__(self, "arguments", _freeze(arguments))

    def to_dict(self):
        return {
            "task_id": self.task_id, "kind": self.kind, "tool_name": self.tool_name,
            "arguments": _plain(self.arguments), "entity_id": self.entity_id, "company_name": self.company_name,
            "doc_ids": list(self.doc_ids), "required_keys": [asdict(key) for key in self.required_keys],
            "depends_on": list(self.depends_on), "priority": self.priority, "fallback_for": list(self.fallback_for),
            "retry_for": self.retry_for,
            "bindings": [{"variable": name, "key": asdict(key)} for name, key in self.bindings],
            "report_years": dict(self.report_years),
        }


@dataclass(frozen=True)
class ResearchDAG:
    tasks: tuple[ResearchTask, ...]
    waves: tuple[tuple[ResearchTask, ...], ...] = field(init=False)

    def __post_init__(self):
        tasks = tuple(self.tasks)
        by_id = {task.task_id: task for task in tasks}
        if not tasks or len(tasks) > 24 or len(by_id) != len(tasks):
            raise ValueError("Invalid research task count or duplicate IDs")
        graph = nx.DiGraph()
        graph.add_nodes_from(by_id)
        for task in tasks:
            parent_keys = set()
            if not set(task.fallback_for).issubset(task.depends_on):
                raise ValueError("Invalid fallback dependency")
            if task.retry_for:
                previous = by_id.get(task.retry_for)
                if (task.retry_for not in task.depends_on or previous is None or previous.kind != task.kind
                        or previous.bindings != task.bindings or previous.required_keys != task.required_keys
                        or previous.arguments != task.arguments):
                    raise ValueError("Invalid calculation retry")
            for parent_id in task.depends_on:
                parent = by_id.get(parent_id)
                if parent is None or parent is task:
                    raise ValueError("Invalid research dependency")
                if parent.entity_id != task.entity_id or parent.company_name != task.company_name:
                    raise ValueError("Dependency crosses entity scope")
                if not set(parent.doc_ids).issubset(task.doc_ids):
                    raise ValueError("Dependency crosses report scope")
                if parent.fallback_for and not task.fallback_for and not task.bindings:
                    raise ValueError("Mandatory task depends on speculative evidence")
                parent_keys.update(parent.required_keys)
                graph.add_edge(parent_id, task.task_id)
            for _, key in task.bindings:
                result_key = task.required_keys[0]
                if (key not in parent_keys or key.dimension != "CNY:money"
                        or (key.scope, key.basis, key.segment) != (result_key.scope, result_key.basis, result_key.segment)):
                    raise ValueError("Unresolvable calculation input")
        try:
            waves = tuple(tuple(sorted((by_id[task_id] for task_id in wave), key=lambda t: (t.priority, t.task_id)))
                          for wave in nx.topological_generations(graph))
        except nx.NetworkXUnfeasible as exc:
            raise ValueError("Cyclic research dependencies") from exc
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "waves", waves)

    @property
    def required_keys(self):
        return tuple(sorted({key for task in self.tasks if not task.fallback_for for key in task.required_keys}))

    def to_dict(self):
        return {"tasks": [task.to_dict() for task in self.tasks],
                "waves": [[task.task_id for task in wave] for wave in self.waves]}


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    wave: int
    status: TaskStatus
    code: str = ""
    ms: float = 0.0
    evidence_ids: tuple[str, ...] = ()
    retrieval_mode: str | None = None

    def to_dict(self):
        return {**asdict(self), "evidence_ids": list(self.evidence_ids)}


@dataclass(frozen=True)
class ResearchRun:
    required_keys: tuple[EvidenceKey, ...]
    outcomes: tuple[TaskOutcome, ...]
    wave_count: int
    stop_reason: str
    deadline: float | None = None

    def to_dict(self):
        return {
            "wave_count": self.wave_count, "stop_reason": self.stop_reason,
            "completed_task_ids": [o.task_id for o in self.outcomes if o.status in {"ok", "empty", "failed", "timed_out"}],
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


def _companies(query_plan):
    filters = query_plan.filters
    return tuple(dict.fromkeys(query_plan.mentioned_companies or query_plan.scope.get("companies")
                               or [filters.company_name or query_plan.scope.get("company_name") or ""]))


def _requested(question, query_plan):
    text = question.casefold()
    playbook = query_plan.playbook or {}
    signals = [*query_plan.evidence_targets, *(playbook.get("metric_names") or [])]
    topic_labels = " ".join(item for item in signals if isinstance(item, str)).casefold()
    topics = [name for name, (cues, _) in _TOPICS.items()
              if any(cue in text for cue in cues)
              or (name in query_plan.topic_flags and any(cue in topic_labels for cue in cues))]
    atom = query_plan.metric_atom or {}
    metric = atom.get("atom_id") or atom.get("metric_atom_id")
    label = atom.get("standard_name") or query_plan.metric or ""
    if not metric and any(cue in text for cue in _CASH):
        metric, label = "MET-003", "经营活动产生的现金流量净额"
    dimension = "percent" if atom.get("unit") in {"%", "％"} else "CNY:money"
    if not label and metric:
        label = metric
    if metric is None and any(cue in topic_labels for cue in _CASH) and playbook.get("required_metrics") == ["MET-003"]:
        metric, label = "MET-003", "经营活动产生的现金流量净额"
    change = bool(metric and dimension == "CNY:money" and any(cue in text for cue in _CHANGE)
                  and not _PROJECTED_ANALYSIS.search(text))
    return metric, label, dimension, change, topics


def should_route_deep_research(query_plan, *, schema, question, route_decision):
    if schema not in {"text", "long_text"} or not isinstance(query_plan, QueryPlan):
        return False
    metric, _, _, change, topics = _requested(question, query_plan)
    cross_analysis = (len(route_decision.get("companies") or _companies(query_plan)) >= 2
                      and any(cue in question.casefold() for cue in _COMPARE))
    return bool((bool(metric) + bool(change) + len(topics) >= 2) or (cross_analysis and (metric or topics)))


def _report_year(doc):
    match = re.search(r"(?:^|_)((?:19|20)\d{2})(?:_|$)", doc)
    return int(match[1]) if match else None


def _scoped_docs(query_plan, doc_ids):
    companies = _companies(query_plan)
    if isinstance(doc_ids, Mapping):
        scoped = dict(doc_ids)
    elif len(companies) == 1 and isinstance(doc_ids, Sequence) and not isinstance(doc_ids, str):
        scoped = {companies[0]: doc_ids}
    else:
        raise ValueError("Unresolved company/report scope")
    if (set(scoped) != set(companies) or not all(isinstance(company, str) and company.strip() for company in companies)
            or any(not isinstance(docs, Sequence) or isinstance(docs, str) or not _valid_docs(docs) for docs in scoped.values())):
        raise ValueError("Unresolved company/report scope")
    known = set(query_plan.filters.candidate_doc_ids or ())
    owners = {}
    for company, docs in scoped.items():
        for doc in docs:
            if (known and doc not in known) or owners.setdefault(doc, company) != company:
                raise ValueError("Inconsistent company/report scope")
    return {company: tuple(sorted(set(docs))) for company, docs in scoped.items()}


def _periods(question, query_plan, docs, change):
    years = sorted({int(year) for year in _YEAR.findall(question)})
    if not years:
        for value in (query_plan.scope.get("value_period"), query_plan.filters.period,
                      query_plan.filters.year, query_plan.scope.get("year")):
            if value is not None and re.fullmatch(r"[0-9]{4}", str(value)):
                years = [int(value)]
                break
    if not years:
        report_years = [_report_year(doc) for doc in docs if _report_year(doc)]
        if report_years:
            years = [max(report_years)]
    if change and len(years) == 1:
        years.insert(0, years[0] - 1)
    if any(year < 1000 or year > 9999 for year in years):
        raise ValueError("Invalid fiscal period")
    return [str(year) for year in years]


def _task_group(company, docs, question, query_plan, requested):
    metric, label, dimension, change, topics = requested
    years = {doc: year for doc in docs if (year := _report_year(doc)) is not None}
    security_code = query_plan.filters.security_code if len(_companies(query_plan)) == 1 else None
    report_codes = {doc.split("_", 1)[0] for doc in docs if re.match(r"^\d{5,6}_", doc)}
    if len(report_codes) > 1 or (security_code and report_codes and security_code not in report_codes):
        raise ValueError("Inconsistent entity scope")
    entity = security_code or next(iter(report_codes), company)
    prefix = f"entity-{_companies(query_plan).index(company) + 1}"
    task_group = []

    def add(kind, suffix, arguments, keys, **options):
        task = ResearchTask(
            task_id=f"{prefix}:{suffix}", kind=kind, tool_name=_TOOLS[kind][0], arguments=arguments,
            entity_id=entity, company_name=company, doc_ids=docs, required_keys=tuple(keys), report_years=years,
            **options,
        )
        task_group.append(task)
        return task.task_id

    calculation = None
    if metric:
        periods = _periods(question, query_plan, docs, change)
        if not periods:
            raise ValueError("Unresolved fiscal period")
        scope = query_plan.scope.get("scope_type") or ("parent" if re.search(r"母公司(?!股东|所有者|普通股)|parent.company", question, re.I)
                                                        else "consolidated")
        basis = metric_basis(metric, query_plan.scope.get("measurement_basis") or measurement_basis(question))
        if "per_share" in basis and dimension == "CNY:money":
            dimension, change = "CNY:per_share", False
        if query_plan.filters.currency in {"USD", "HKD"} and dimension == "CNY:money":
            dimension, change = f"{query_plan.filters.currency}:money", False
        segment = str(query_plan.scope.get("segment") or "")
        keys = [EvidenceKey(entity, metric, period, scope, dimension, basis, segment) for period in periods]
        sources = {}
        for key in keys:
            sources[key] = add("sql_fact", f"sql-{key.period}", {
                "operation": "metric", "metric_atom_id": metric, "period": key.period,
                "company_name": company, "scope": scope, "limit": 50,
            }, [key], priority=0, metric_label=label)

        if change and len(keys) >= 2:
            prior, current = keys[0], keys[-1]
            delta = EvidenceKey(entity, f"{metric}.delta", f"{prior.period}–{current.period}", scope, dimension, basis, segment)
            calculation = add(
                "pot_calc", "delta", {"expression": "current - prior", "result_unit": "元"}, [delta],
                depends_on=(sources[prior], sources[current]), bindings=(("current", current), ("prior", prior)),
                metric_label=f"{label}变动额",
            )
        for key in keys:
            parent = sources[key]
            add("hybrid_rag", f"fallback-{key.period}", {
                "query": f"{company} {label} {key.period} {question}"[:1000],
                "doc_ids": list(docs), "company_name": company, "top_k": 5,
            }, [key], priority=20, depends_on=(parent, calculation) if calculation else (parent,),
               fallback_for=(parent,), metric_label=label)
        if calculation:
            # A missing SQL input can be recovered by corroboration before a second calculation attempt.
            fallbacks = tuple(task.task_id for task in task_group if task.fallback_for)
            add("pot_calc", "delta-after-fallback", {"expression": "current - prior", "result_unit": "元"}, [delta],
                depends_on=(*fallbacks, calculation), bindings=(("current", current), ("prior", prior)),
                metric_label=f"{label}变动额", retry_for=calculation)

    for aspect in topics:
        cues, relations = _TOPICS[aspect]
        key = EvidenceKey(entity, aspect, "", "", "observation", "reported")
        request = f"{question} {' '.join(cues)}"[:1000]
        parents = []
        if relations:
            parents.append(add("kg_multihop", f"kg-{aspect}", {
                "question": request, "company_name": company, "doc_ids": list(docs),
            }, [key], relations=relations))
        for index, doc in enumerate(docs):
            parents.append(add("toc_nav", f"toc-{aspect}-{index + 1}", {
                "doc_id": doc, "question": request,
            }, [key], priority=5, cues=cues if aspect != "shareholder" else ()))
        dependencies = (*parents, calculation) if calculation else tuple(parents)
        add("hybrid_rag", f"fallback-{aspect}", {
            "query": request, "doc_ids": list(docs), "company_name": company, "top_k": 6,
        }, [key], priority=20, depends_on=dependencies, fallback_for=tuple(parents),
           cues=cues if aspect != "shareholder" else ())
    return task_group


class ResearchPlanner:
    def __init__(self, *, max_tasks=24, max_parallel=4, task_timeout_s=5.0):
        if (type(max_tasks) is not int or not 1 <= max_tasks <= 24
                or type(max_parallel) is not int or not 1 <= max_parallel <= 32
                or not math.isfinite(task_timeout_s) or task_timeout_s <= 0):
            raise ValueError("Invalid research limits")
        self.max_tasks = max_tasks
        self.max_parallel = max_parallel
        self.task_timeout_s = task_timeout_s

    def plan(self, question, query_plan, *, doc_ids) -> ResearchDAG:
        if not isinstance(query_plan, QueryPlan) or not isinstance(question, str) or not question.strip():
            raise ValueError("Invalid research query")
        scoped = _scoped_docs(query_plan, doc_ids)
        requested = _requested(question, query_plan)
        tasks = [task for company, docs in scoped.items()
                 for task in _task_group(company, docs, question, query_plan, requested)]
        if len(tasks) > self.max_tasks:
            raise ValueError("Research task limit exceeded")
        return ResearchDAG(tuple(tasks))

    async def _call(self, task, *, registry, principal, ledger, deadline, resolutions):
        started = time.monotonic()
        timeout = min(self.task_timeout_s, deadline - started)
        if timeout <= 0:
            return "skipped_budget", "request_deadline", 0.0, None, ()
        arguments = _plain(task.arguments)
        inputs = []
        if task.bindings:
            variables = []
            for name, key in task.bindings:
                resolution = resolutions.get(key)
                if not resolution or not resolution.winner_id:
                    return "blocked", "unresolved_input", 0.0, None, ()
                entry = ledger.entry(resolution.winner_id)
                if (entry.doc_id not in task.doc_ids or entry.page is None or entry.normalized_value is None
                        or entry.company_name != task.company_name):
                    return "blocked", "incompatible_input", 0.0, None, ()
                number = float(entry.normalized_value)
                if entry.normalized_value != type(entry.normalized_value)(str(number)):
                    return "blocked", "input_precision_loss", 0.0, None, ()
                variables.append({"name": name, "value": number, "unit": "元", "doc_id": entry.doc_id, "page": entry.page})
                inputs.append(entry.evidence_id)
            arguments["variables"] = variables
        try:
            _TOOLS[task.kind][1].model_validate(arguments, strict=True)
            async with asyncio.timeout(timeout):
                result = await registry.call(task.tool_name, arguments, principal=principal, timeout_s=timeout)
        except TimeoutError:
            result = ToolResult(error_code="timeout")
        except (TypeError, ValueError):
            result = ToolResult(error_code="invalid_arguments")
        elapsed = time.monotonic() - started
        ms = round(min(timeout, elapsed) * 1000, 2)
        if elapsed >= timeout or result.error_code == "timeout":
            return "timed_out", "timeout", ms, None, ()
        if result.error_code:
            return "failed", result.error_code, ms, None, ()
        return "ok", "", ms, result, tuple(inputs)

    async def execute(self, dag, *, registry, principal, ledger, deadline) -> ResearchRun:
        if not isinstance(dag, ResearchDAG) or len(dag.tasks) > self.max_tasks or not math.isfinite(deadline):
            raise ValueError("Invalid research run")
        outcomes = []
        completed = {}
        observing = capture_active()
        prefix = current_prefix() if observing else ""
        task_paths = {}
        if observing:
            task_paths = {
                task.task_id: f"wave-{index}/task-{ordinal}"
                for index, wave in enumerate(dag.waves, 1)
                for ordinal, task in enumerate(wave, 1)
            }

        def absolute(path):
            return f"{prefix}/{path}" if prefix else path

        async def execute_task(task, resolutions, wave_key):
            if observing:
                parents = (wave_key, *(absolute(task_paths[parent] + "/result") for parent in task.depends_on))
                with step_scope(task_paths[task.task_id], parents=parents):
                    return await self._call(task, registry=registry, principal=principal, ledger=ledger,
                                            deadline=deadline, resolutions=resolutions)
            return await self._call(task, registry=registry, principal=principal, ledger=ledger,
                                    deadline=deadline, resolutions=resolutions)
        for index, wave in enumerate(dag.waves, 1):
            wave_key = None
            if observing:
                wave_key = capture_step(
                    "plan", operation="research_wave", step_key=f"wave-{index}",
                    inputs={}, outputs={"wave": index, "task_ids": [task.task_id for task in wave]},
                )
            resolutions = ledger.adjudicate()
            sufficient = self._sufficient(dag, resolutions, completed)
            pending = []
            wave_outcomes = []
            for task in wave:
                if time.monotonic() >= deadline:
                    wave_outcomes.append(TaskOutcome(task.task_id, index, "skipped_budget", "request_deadline"))
                elif (task.fallback_for or task.retry_for) and sufficient:
                    wave_outcomes.append(TaskOutcome(task.task_id, index, "skipped_sufficient", "required_evidence_resolved"))
                elif task.retry_for and completed.get(task.retry_for) not in {"blocked", "empty"}:
                    wave_outcomes.append(TaskOutcome(task.task_id, index, "blocked", "calculation_already_attempted"))
                else:
                    pending.append(task)
            results = []
            for start in range(0, len(pending), self.max_parallel):
                batch = pending[start:start + self.max_parallel]
                results.extend(zip(batch, await asyncio.gather(*(
                    execute_task(task, resolutions, wave_key) for task in batch
                )), strict=True))
            for task, (status, code, ms, result, inputs) in sorted(results, key=lambda item: item[0].task_id):
                evidence = ledger.ingest(task, result, inputs=inputs) if result else ()
                mode = None
                if result and task.kind == "hybrid_rag" and isinstance(result.data, dict):
                    reported_mode = result.data.get("mode")
                    mode = reported_mode if reported_mode in ("hybrid", "lexical_only") else None
                wave_outcomes.append(TaskOutcome(task.task_id, index, "empty" if status == "ok" and not evidence else status,
                                                 code, ms, evidence, mode))
            wave_outcomes.sort(key=lambda outcome: outcome.task_id)
            if observing:
                for outcome in wave_outcomes:
                    task_path = task_paths[outcome.task_id]
                    tool_key = absolute(task_path + "/tool")
                    observer = current_observer()
                    capture_step(
                        "tool", operation="research_task", step_key=task_path + "/result",
                        inputs={"task_id": outcome.task_id, "wave": index},
                        outputs={"evidence_ids": list(outcome.evidence_ids), "retrieval_mode": outcome.retrieval_mode},
                        parents=(tool_key,) if observer.has_step(tool_key) else (wave_key,),
                        status=outcome.status, error_code=outcome.code or None, costs={"elapsed_ms": outcome.ms},
                    )
            outcomes.extend(wave_outcomes)
            completed.update((outcome.task_id, outcome.status) for outcome in wave_outcomes)
        if time.monotonic() >= deadline:
            reason = "budget_exhausted"
        elif self._sufficient(dag, ledger.adjudicate(), completed):
            reason = "sufficient"
        else:
            reason = "incomplete"
        return ResearchRun(dag.required_keys, tuple(outcomes), len(dag.waves), reason, deadline)

    @staticmethod
    def _sufficient(dag, resolutions, completed):
        return (all(key in resolutions and resolutions[key].winner_id for key in dag.required_keys)
                and all(task.task_id in completed for task in dag.tasks if not task.fallback_for and not task.retry_for))


def _graph_scope(processor, query_plan, route_info):
    companies = _companies(query_plan)
    candidates = route_info.get("candidate_doc_ids") or query_plan.filters.candidate_doc_ids or []
    selected = route_info.get("selected_report") or {}
    selected_id = selected.get("sha1") or selected.get("doc_id")
    catalog = getattr(processor, "report_catalog", None)
    if len(companies) == 1 and not candidates and not selected_id:
        selector = getattr(processor, "_selected_doc_id", None)
        selected_id = selector(companies[0], route_info, query_plan) if selector else None
    if len(companies) == 1:
        return {companies[0]: [selected_id] if selected_id else candidates}
    if not catalog:
        return {}
    if not candidates:
        candidates = [report.sha1 for report in catalog.get_reports()
                      if catalog.report_matches_query_filters(report, query_plan)]
    scoped = {company: [] for company in companies}
    for doc in candidates:
        report = catalog.get_report_by_doc_id(doc)
        if report and report.company_name in scoped:
            scoped[report.company_name].append(doc)
    return scoped


def _run_sync(make_coroutine):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make_coroutine())
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="finarag-research") as worker:
        return worker.submit(copy_context().run, lambda: asyncio.run(make_coroutine())).result()


def _replay_synthesis(synthesizer, *, verify_claims):
    # P6's synthesizer reads a process flag. Use its real implementation with
    # a local flag lookup so offline runs cannot affect concurrent live requests.
    method = synthesizer.synthesize
    implementation = method.__func__
    flag = "1" if verify_claims else "0"

    def getenv(name, default=None):
        return flag if name == "FINARAG_CLAIM_VERIFY_ENABLED" else default

    globals_for_replay = {**implementation.__globals__, "os": SimpleNamespace(getenv=getenv)}
    run = FunctionType(
        implementation.__code__, globals_for_replay, implementation.__name__,
        implementation.__defaults__, implementation.__closure__,
    )
    return run.__get__(synthesizer, type(synthesizer))


def make_deep_research_node(processor, *, registry_provider=None):
    from src.deep_research.synthesizer import ResearchSynthesizer

    provider = registry_provider or getattr(processor, "deep_research_registry_provider", None)
    planner = ResearchPlanner()
    synthesizer = ResearchSynthesizer(claim_verifier=getattr(processor, "deep_research_claim_verifier", None))
    replay_config = getattr(processor, "offline_replay_config", None)
    synthesize = (
        synthesizer.synthesize if replay_config is None else
        _replay_synthesis(synthesizer, verify_claims=replay_config.get("claim_verify", False))
    )

    def research_node(state, runtime: Runtime[FinaRAGRunContext] = None):
        query_plan = state["query_plan"]
        question = query_plan.original_query
        route_info = {**(state.get("route_info") or {}), "route_target": "deep_research"}
        context = runtime.context if runtime is not None else None
        budget = context.budget_context if context else state.get("budget_context")
        available_s = budget.remaining_latency_ms / 1000 if budget else 30.0
        if budget and not budget.remaining_tokens:
            available_s = 0.0
        deadline = time.monotonic() + available_s
        dag = None
        try:
            dag = planner.plan(question, query_plan, doc_ids=_graph_scope(processor, query_plan, route_info))
        except (TypeError, ValueError, StorageError):
            pass

        async def run():
            from src.storage.sqlite_doc_store import SqliteDocumentRepository

            registry = None
            repository = None
            ledger = EvidenceLedger()
            try:
                if provider:
                    registry = provider()
                else:
                    configured_root = os.getenv("FINARAG_PROTOCOL_CORPUS_ROOT")
                    if configured_root is None and getattr(processor, "documents_dir", None):
                        configured_root = Path(processor.documents_dir).parent
                    if not configured_root:
                        raise ValueError("Missing corpus configuration")
                    root, paths = _corpus_paths(Path(configured_root))
                    if paths["docs"] is None:
                        raise ValueError("Missing document repository")
                    repository = SqliteDocumentRepository(paths["docs"], allowed_base_dir=root)
                    ledger = EvidenceLedger(document_repository=repository)
                    for task in dag.tasks:
                        for doc in task.doc_ids:
                            meta = repository.get_metainfo(doc) or {}
                            if (repository.resolve_doc_id(doc) != doc or meta.get("company_name") != task.company_name):
                                raise ValueError("Unresolved corpus scope")
                    registry = build_financial_tool_registry(root, policy=default_policy)
                if not isinstance(registry, MCPToolRegistry):
                    raise ValueError("Invalid tool registry")
                execution = await planner.execute(dag, registry=registry, principal=_PRINCIPAL,
                                                  ledger=ledger, deadline=deadline)
                return response(execution, ledger)
            except (OSError, ValueError, StorageError):
                return response(unavailable("research_unavailable"), ledger)
            finally:
                if isinstance(registry, MCPToolRegistry):
                    registry.close()
                if repository is not None:
                    repository.close()

        def unavailable(reason):
            return ResearchRun(dag.required_keys if dag else (), (), 0, reason, deadline)

        def response(execution, ledger):
            observing = capture_active()
            started = time.perf_counter() if observing else None
            answer = synthesize(question, execution, ledger, query_plan=query_plan, route_info=route_info)
            status = execution.to_dict()
            if observing:
                prefix = current_prefix()
                task_keys = (
                    f"wave-{index}/task-{ordinal}/result"
                    for index, wave in enumerate(dag.waves if dag and execution.outcomes else (), 1)
                    for ordinal, _ in enumerate(wave, 1)
                )
                parents = tuple(f"{prefix}/{key}" if prefix else key for key in task_keys)
                capture_step(
                    "verification", operation="research_synthesis", step_key="verification",
                    inputs={"stop_reason": execution.stop_reason},
                    outputs={"answer": answer}, parents=parents,
                    costs={"elapsed_ms": (time.perf_counter() - started) * 1000},
                )
            return {
                "deep_research_plan": dag.to_dict() if dag else {},
                "evidence_ledger": ledger.to_dict(),
                "deep_research_status": status,
                "final_answer_dict": answer,
                "stage_events": [{"stage": "deep_research", **status,
                                  "flags": answer["validation_flags"], "confidence": answer["confidence"],
                                  "claim_contradicted": (answer.get("claim_audit") or {}).get("contradicted_count", 0)}],
            }

        if dag is None:
            return response(unavailable("invalid_research_scope"), EvidenceLedger())
        if time.monotonic() >= deadline:
            return response(unavailable("budget_exhausted"), EvidenceLedger())
        return _run_sync(run)

    return research_node
