"""Run reviewed fixture requests through the current graph with offline adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
from collections import defaultdict, deque
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.harness.trajectory_store import TrajectoryRecord, TrajectoryStore, collect_trajectory, current_prefix

_OPERATIONS = {
    "retrieve": "retrieval", "remote_rank": "rerank", "generate": "generation",
    "rephrase": "generation", "tool_provider": "tool",
}
_OPERATION_BOUNDARIES = {
    "retrieve": "retrieval", "remote_rank": "rerank", "generate": "generation",
    "rephrase": "rephrase", "tool_provider": "tool",
}
_CATEGORIES = frozenset({"plan", "tool", "retrieval", "rerank", "generation", "verification"})
_SET_FIELDS = frozenset({"flags", "validation_flags", "validated_flags", "citations", "references"})
_VOLATILE_FIELDS = frozenset({
    "elapsed_ms", "latency_ms", "duration_ms", "ms", "deadline", "trace_id", "turn_id",
    "session_id", "thread_id", "request_id", "checkpoint_id", "interrupt_id", "created_at",
    "graph_stage_events", "response_data",
})
_TOKEN_FIELDS = frozenset({"prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"})
_KEY = re.compile(r"[A-Za-z0-9_.:/-]{1,256}\Z")
_SUPPORTED_SCHEMAS = frozenset({"text", "long_text", "number", "boolean", "name", "names", "comparative"})
_FIXTURE_CORPUS_VERSION = "synthetic-financial-1"


def _canonical(value):
    return json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _fingerprint(value):
    return hashlib.sha256(_canonical(value).encode("ascii")).hexdigest()


def _semantic(value, field=None):
    if isinstance(value, dict):
        return {key: _semantic(item, key) for key, item in value.items()
                if key not in _VOLATILE_FIELDS | _TOKEN_FIELDS}
    if isinstance(value, (list, tuple)):
        result = [_semantic(item) for item in value]
        return sorted(result, key=_canonical) if field in _SET_FIELDS else result
    return value


def _cost_projection(cost):
    result = {key: cost[key] for key in _TOKEN_FIELDS | {
        "elapsed_ms", "degraded_level", "max_tokens", "spent_tokens", "remaining_tokens", "max_latency_ms",
    }
              if type(cost.get(key)) in {int, float}}
    for field in ("token_usage", "budget"):
        if isinstance(cost.get(field), dict):
            result[field] = _cost_projection(cost[field])
    return result


@dataclass(frozen=True)
class ReplayDiff:
    step_key: str
    category: str
    aspect: str
    reason: str
    expected: Any = None
    actual: Any = None
    root_step_key: str | None = None


@dataclass(frozen=True)
class ReplayResult:
    trace_id: str
    diffs: tuple[ReplayDiff, ...] = ()
    coverage_errors: tuple[str, ...] = ()
    recorded_cost: dict | None = None

    @property
    def passed(self):
        return not (self.diffs or self.coverage_errors)

    @property
    def roots(self):
        return tuple(diff for diff in self.diffs if diff.root_step_key is None)


@dataclass(frozen=True)
class ReplaySummary:
    results: tuple[ReplayResult, ...]

    @property
    def passed(self):
        return bool(self.results) and all(result.passed for result in self.results)

    @property
    def exit_code(self):
        if not self.results or any(result.coverage_errors for result in self.results):
            return 2
        return 0 if self.passed else 1


class _FixtureError(ValueError):
    """A fixture is incomplete or an adapter could not find the exact input."""


def _require(condition):
    if not condition:
        raise _FixtureError("Invalid replay fixture")


@dataclass(frozen=True)
class _Call:
    step_key: str
    operation: str
    inputs: dict
    output: Any
    status: str
    usage: dict

    @classmethod
    def load(cls, item):
        _require(isinstance(item, dict) and set(item) <= {"step_key", "operation", "inputs", "output", "status", "usage"})
        step_key = item.get("step_key")
        operation = item.get("operation")
        inputs = item.get("inputs")
        status = item.get("status", "ok")
        usage = item.get("usage", {})
        _require(isinstance(step_key, str) and _KEY.fullmatch(step_key))
        _require(isinstance(operation, str) and operation in _OPERATIONS and isinstance(inputs, dict))
        _require("output" in item and status in ("ok", "error") and isinstance(usage, dict))
        _require(all(key in _TOKEN_FIELDS and type(value) is int and value >= 0
                     for key, value in usage.items()))
        output = item["output"]
        if operation == "retrieve":
            _require(status == "ok" and isinstance(output, list) and all(isinstance(row, dict) for row in output))
        elif operation == "generate":
            _require(status == "ok" and isinstance(output, dict) and "final_answer" in output)
            _require(set(output) <= {"final_answer", "relevant_pages", "reasoning_summary", "step_by_step_analysis", "pot_calculation"})
        elif operation == "rephrase":
            _require(status == "ok" and isinstance(output, dict) and isinstance(output.get("questions"), list))
            _require(all(isinstance(item, dict) and set(item) == {"company_name", "question"}
                         and all(isinstance(value, str) for value in item.values())
                         for item in output["questions"]))
        elif operation == "remote_rank":
            _require(status == "ok" and isinstance(output, dict) and isinstance(output.get("results"), list))
        else:
            _require(isinstance(inputs.get("tool_name"), str) and isinstance(inputs.get("arguments"), dict))
            if status == "error":
                _require(isinstance(output, str) and output in {
                    "unavailable", "timeout", "internal_error", "invalid_arguments", "forbidden", "not_found", "busy",
                })
        return cls(step_key, operation, inputs, output, status, usage)


class _Exchange:
    def __init__(self, calls):
        self._calls = defaultdict(deque)
        self._issues = []
        self._lock = threading.Lock()
        for call in calls:
            key = (call.step_key, call.operation, _canonical(call.inputs))
            self._calls[key].append(call)

    def reply(self, operation, inputs):
        prefix = current_prefix()
        step_key = f"{prefix}/{operation}" if prefix else operation
        key = (step_key, operation, _canonical(inputs))
        with self._lock:
            if calls := self._calls.get(key):
                return deepcopy(calls.popleft())
            candidates = [pending[0] for reference, pending in self._calls.items()
                          if pending and reference[:2] == key[:2]]
            expected = candidates[0].inputs if len(candidates) == 1 else None
            self._issues.append(ReplayDiff(
                step_key, _OPERATIONS[operation], "inputs",
                "external_input_mismatch" if candidates else "missing_external_fixture",
                expected, deepcopy(inputs),
            ))
        raise _FixtureError("External input unavailable")

    def differences(self):
        with self._lock:
            unused = [
                ReplayDiff(call.step_key, _OPERATIONS[call.operation], "coverage", "unused_external_fixture", call.inputs)
                for pending in self._calls.values() for call in pending
            ]
            return sorted([*self._issues, *unused],
                          key=lambda diff: (diff.step_key, diff.reason, _canonical(diff.expected), _canonical(diff.actual)))


class _ReplayModel:
    def __init__(self, exchange):
        self.exchange = exchange
        self._response = threading.local()

    @property
    def response_data(self):
        return getattr(self._response, "data", {})

    def _api(self, operation):
        from src.api_requests import APIProcessor

        exchange = self.exchange
        response = self._response

        class OfflineTransport:
            default_model = "fixture-answer-model"

            def send_message(self, *, system_content, human_content, response_format, **inputs):
                call = exchange.reply(operation, {
                    **inputs,
                    "system_sha256": _fingerprint(system_content),
                    "human_sha256": _fingerprint(human_content),
                    "response_schema_sha256": _fingerprint(response_format.model_json_schema()),
                })
                self.response_data = response.data = {"usage": call.usage}
                return call.output

        api = object.__new__(APIProcessor)
        api.provider = "qwen"
        api.processor = OfflineTransport()
        return api

    def get_answer_from_rag_context(self, **inputs):
        return self._api("generate").get_answer_from_rag_context(**inputs)

    def get_rephrased_questions(self, **inputs):
        return self._api("rephrase").get_rephrased_questions(**inputs)


class _ReplayRetriever:
    _parallel_backends_enabled = False
    reranking_strategy = "single"
    final_reranking_backend = "vllm_api"
    colbert_top_n = None

    def __init__(self, exchange):
        from src.reranking import VLLMApiReranker

        class OfflineReranker(VLLMApiReranker):
            def __init__(self):
                self.model_name = "fixture-reranker"

            def _post(self, payload):
                call = exchange.reply("remote_rank", payload)
                return SimpleNamespace(ok=True, json=lambda: call.output)

        self.exchange = exchange
        self.reranker = OfflineReranker()
        self.last_rerank_debug = {}

    def _retrieve(self, method, inputs):
        inputs = dict(inputs)
        if inputs.get("filters") is not None:
            inputs["filters"] = inputs["filters"].to_dict()
        return self.exchange.reply("retrieve", {"method": method, **inputs}).output

    def retrieve_by_company_name(self, **inputs):
        return self._retrieve("search", inputs)

    def retrieve_candidates_by_company_name(self, **inputs):
        return self._retrieve("candidates", inputs)

    def retrieve_all(self, company_name, **inputs):
        return self._retrieve("full_context", {"company_name": company_name, **inputs})

    def rerank_candidates(self, **inputs):
        from src.retrieval import HybridRetriever

        return HybridRetriever.rerank_candidates(self, **inputs)


def _offline_processor(request, exchange):
    from src.questions_processing import QuestionsProcessor
    from src.query_rewrite import QuestionRewriter
    from src.report_catalog import CompanyReport, ReportCatalog
    from src.retrieval import HybridRetriever

    class OfflineHybridRetriever(_ReplayRetriever):
        retrieve_by_company_name = HybridRetriever.retrieve_by_company_name

    class OfflineProcessor(QuestionsProcessor):
        def __init__(self):
            self.offline_replay_config = request["config"]
            self.structured_router = None
            self.table_grounder = None
            self.chart_grounder = None
            self.question_rewriter = QuestionRewriter()
            self.report_catalog = ReportCatalog(None)
            self.report_catalog._reports = [CompanyReport(**row) for row in request["reports"]]
            self.doc_router_enabled = True
            self.candidate_doc_top_k = 5
            self.use_vector_dbs = True
            self.use_bm25_db = self.use_sparse_lexical_db = self.use_tag_db = False
            self.api_processor = _ReplayModel(exchange)
            self.api_provider = "offline-fixture"
            self.answering_model = "fixture-answer-model"
            self.answer_temperature = 0.0
            self.llm_reranking_sample_size = 8
            self.final_reranking_batch_size = 2
            self.reranking_strategy = "single"
            self.final_reranking_backend = "vllm_api"
            self.retrieval_debug_top_n = 8
            self.top_n_retrieval = 4
            self.max_doc_chars = 2500
            self.max_context_chars = 8000
            self.parent_retrieval_mode = "chunk"
            self.reasoning_debug_enabled = False
            self.hyde_enabled = False
            self.hyde_top_score_threshold = 0.3
            self.hyde_margin_threshold = 0.1
            self.hyde_generation_model = None
            self._fixture_documents = request.get("documents", {})
            self._fixture_retriever = threading.local()

        def _build_retriever(self):
            if not hasattr(self._fixture_retriever, "instance"):
                mode = self.offline_replay_config["retrieval_mode"]
                adapter = OfflineHybridRetriever if mode == "hybrid_rerank" else _ReplayRetriever
                self._fixture_retriever.instance = (adapter(exchange), mode)
            return self._fixture_retriever.instance

        def _load_document_payload(self, doc_id):
            return self._fixture_documents.get(doc_id)

        def _record_adaptive_trace(self, state, entry=None):
            return None

        def _resolved_graph_engine(self):
            return "langgraph"

    return OfflineProcessor()


def _offline_registry(exchange):
    from src.protocols.tool_registry import MCPToolRegistry, _TOOLS, _ToolError, default_policy
    from src.structured.pot_sandbox import PotCalculationError, PotSandbox, PotSecurityError, PotVariable

    def handler(name):
        def reply(arguments, deadline):
            call = exchange.reply("tool_provider", {"tool_name": name, "arguments": arguments})
            if call.status != "ok":
                raise _ToolError(call.output)
            return call.output
        return reply

    def calculate(arguments, deadline):
        variables = [PotVariable(**item) for item in arguments["variables"]]
        try:
            result = PotSandbox().evaluate(
                variables, arguments["expression"], arguments["result_unit"], arguments["ndigits"],
            )
        except (PotCalculationError, PotSecurityError, ArithmeticError):
            raise _ToolError("invalid_arguments") from None
        return {**asdict(result), "inputs": [asdict(variable) for variable in variables]}

    handlers = {name: handler(name) for name in _TOOLS}
    handlers["pot_calculate"] = calculate
    return MCPToolRegistry(handlers=handlers, policy=default_policy)


def _budget(config, clock):
    from src.service.budget_controller import BudgetContext

    class ReplayBudget(BudgetContext):
        @property
        def elapsed_ms(self):
            return max(0.0, (clock() - self.start_ts) * 1000.0)

    return ReplayBudget(
        max_tokens=config.get("max_tokens", 8000),
        max_latency_ms=config.get("max_latency_ms", 30000),
        spent_tokens=config.get("spent_tokens", 0),
        start_ts=clock(),
    )


def _fixture(record):
    _require(record.replayable and bool(record.steps))
    _require(record.versions.get("replay_policy_version") == "1")
    _require(record.versions.get("corpus_version") == _FIXTURE_CORPUS_VERSION)
    _require(all(isinstance(record.versions.get(name), str) and record.versions[name]
                 for name in ("atom_schema_version", "prompt_version")))
    _require(record.entry_point in {"graph", "qa", "process_question", "run_finarag_graph"})
    request = deepcopy(record.request)
    _require(set(request) <= {"question", "schema", "question_meta", "config", "reports", "documents",
                              "external_calls", "expected_tokens"})
    _require(isinstance(request.get("question"), str) and bool(request["question"].strip()))
    _require(isinstance(request.get("schema"), str) and request["schema"] in _SUPPORTED_SCHEMAS)
    _require(isinstance(request.get("question_meta", {}), dict))
    _require(isinstance(request.get("reports"), list) and bool(request["reports"]))
    _require(all(isinstance(report, dict) for report in request["reports"]))
    _require(isinstance(request.get("documents", {}), dict))
    config = request.get("config")
    _require(isinstance(config, dict) and set(config) <= {"deep_research", "claim_verify", "adaptive", "retrieval_mode", "budget"})
    _require(all(type(config.get(name)) is bool for name in ("deep_research", "claim_verify", "adaptive")))
    _require(config.get("retrieval_mode") in ("vector", "hybrid", "hybrid_rerank", "full_context"))
    budget = config.get("budget")
    _require(isinstance(budget, dict) and set(budget) == {"max_tokens", "max_latency_ms", "spent_tokens"})
    _require(all(type(value) is int and 0 <= value <= 1_000_000 for value in budget.values()))
    tokens = request.get("expected_tokens", {})
    _require(isinstance(tokens, dict) and all(key in _TOKEN_FIELDS and type(value) is int and value >= 0
                                              for key, value in tokens.items()))
    calls = request.get("external_calls")
    _require(isinstance(calls, list))
    _require(any(step.stage in _CATEGORIES for step in record.steps))
    return request, [_Call.load(item) for item in calls]


def _ordered_steps(expected, actual):
    # Expected and observed edge sets both matter when a branch is added or removed.
    steps = {step.step_key: step for step in (*expected, *actual)}
    parents = {key: set(step.parents) for key, step in steps.items()}
    for step in expected:
        parents[step.step_key].update(step.parents)
    # A node observation summarizes its local operations. Establish their order
    # even when observation hooks attach the node and its operations to the same parent.
    for key, step in steps.items():
        if step.stage == "node":
            prefix = key.rpartition("/")[0] + "/"
            parents[key].update(candidate for candidate in steps
                                if candidate.startswith(prefix) and candidate != key
                                and key not in parents[candidate])
    ordered = []
    remaining = set(steps)
    while remaining:
        ready = sorted(key for key in remaining if not (parents[key] & remaining))
        if not ready:
            ready = sorted(remaining)
        ordered.extend(ready)
        remaining.difference_update(ready)
    return ordered, parents


def _stage_category(step):
    if step.stage in _CATEGORIES:
        return step.stage
    for boundary, category in (("deep_research", "plan"), ("plan", "plan"), ("retriev", "retrieval"),
                               ("reduc", "generation"), ("ground_generate", "generation"), ("reflect", "retrieval")):
        if boundary in step.step_key:
            return category
    return "verification"


def _step_changes(expected, actual):
    if expected is None:
        yield "coverage", "unexpected_step", None, actual.operation
    elif actual is None:
        yield "coverage", "missing_step", expected.operation, None
    else:
        for aspect in ("stage", "operation", "parents", "status", "error_code", "source_ids", "inputs", "outputs"):
            before = getattr(expected, aspect)
            after = getattr(actual, aspect)
            if aspect in {"parents", "source_ids"}:
                before, after = sorted(before), sorted(after)
            left, right = _semantic(before, aspect), _semantic(after, aspect)
            if left != right:
                yield aspect, "changed", left, right


def _compare(expected, actual, external, usage):
    before = {step.step_key: step for step in expected.steps}
    after = {step.step_key: step for step in actual.steps}
    ordered, parents = _ordered_steps(expected.steps, actual.steps)
    roots = {}
    propagated = {}
    diffs = []
    for key in ordered:
        upstream = {propagated[parent] for parent in sorted(parents[key]) if parent in propagated}
        cause = next((candidate for candidate in ordered if candidate in upstream), None)
        changes = tuple(_step_changes(before.get(key), after.get(key)))
        if cause or changes:
            propagated[key] = cause or key
        if changes and not cause:
            roots[key] = changes
        for index, (aspect, reason, left, right) in enumerate(changes):
            diffs.append(ReplayDiff(
                key, _stage_category(after.get(key) or before[key]), aspect, reason, left, right,
                cause if index == 0 else cause or key,
            ))

    coverage = []
    for issue in external:
        scope, _, operation = issue.step_key.rpartition("/")
        boundary = _OPERATION_BOUNDARIES[operation]
        key = f"{scope}/{boundary}" if scope else boundary
        upstream = set()
        pending = list(parents.get(key, ()))
        # A changed route can remove the entire node. Fall back to its captured
        # node dependencies when no adapter-specific observation was emitted.
        if key not in parents and scope:
            pending.extend(parent for parent in parents.get(f"{scope}/node", ())
                           if not parent.startswith(scope + "/"))
        while pending:
            parent = pending.pop()
            if parent not in upstream and parent != key:
                upstream.add(parent)
                pending.extend(parents.get(parent, ()))
        cause = next((candidate for candidate in ordered if candidate in upstream and
                      any(aspect not in {"coverage", "status", "error_code"}
                          for aspect, _, _, _ in roots.get(candidate, ()))), None)
        if cause is None:
            coverage.append(issue.reason)
        diffs.append(ReplayDiff(issue.step_key, issue.category, issue.aspect, issue.reason,
                                issue.expected, issue.actual, cause))

    for aspect in ("query_plan", "stage_events", "operation_status", "final_answer", "validation_flags", "confidence"):
        left, right = _semantic(getattr(expected, aspect), aspect), _semantic(getattr(actual, aspect), aspect)
        if left != right:
            category = "plan" if aspect == "query_plan" else "verification"
            cause = next(iter(roots), None)
            diffs.append(ReplayDiff("result", category, aspect, "changed", left, right, cause))

    counters = expected.request.get("expected_tokens", {})
    observed = {key: usage.get(key, 0) for key in counters}
    if counters != observed:
        generation = [step.step_key for step in (*expected.steps, *actual.steps) if step.stage == "generation"]
        cause = next((propagated[key] for key in generation if key in propagated), None)
        diffs.append(ReplayDiff("result/tokens", "generation", "cost", "changed", counters, observed, cause))

    left, right = (_semantic(record.cost.get("budget")) for record in (expected, actual))
    if left is not None and left != right:
        diffs.append(ReplayDiff("result", "verification", "budget", "changed", left, right, next(iter(roots), None)))
    return tuple(diffs), tuple(sorted(set(coverage)))


class ReplayRunner:
    """Replay a trajectory with trusted factory overrides for local tests.

    The processor factory takes the validated request and the fixture exchange.
    The registry factory takes the exchange and returns a fresh registry per graph run.
    """

    def __init__(self, *, processor_factory=None, registry_factory=None, clock=None):
        self.processor_factory = processor_factory or _offline_processor
        self.registry_factory = registry_factory or _offline_registry
        self.clock = clock or (lambda: 0.0)

    def replay(self, record: TrajectoryRecord) -> ReplayResult:
        from src.graph.builder import run_finarag_graph
        from src.service.semantic_cache import resolve_cache_versions
        from src.service.tracing import token_usage_of

        trace_id = record.trace_id
        recorded_cost = _cost_projection(record.cost)
        try:
            request, calls = _fixture(record)
        except (TypeError, ValueError, KeyError):
            return ReplayResult(trace_id, coverage_errors=("invalid_or_incomplete_fixture",), recorded_cost=recorded_cost)
        exchange = _Exchange(calls)
        answer = {}
        failed = False
        with collect_trajectory(record.entry_point, request) as observer:
            try:
                processor = self.processor_factory(request, exchange)
                processor.deep_research_registry_provider = lambda: self.registry_factory(exchange)
                answer = run_finarag_graph(
                    processor, request["question"], request["schema"], request.get("question_meta"),
                    budget_context=_budget(request["config"]["budget"], self.clock),
                )
            except Exception:
                failed = True
            observer.finish(answer, status="error" if failed else None)
        external = exchange.differences()
        usage = token_usage_of(answer)
        usage["input_tokens"] = usage["prompt_tokens"]
        usage["output_tokens"] = usage["completion_tokens"]
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        diffs, coverage = _compare(record, observer.record, external, usage)
        active_versions = asdict(resolve_cache_versions(corpus_version=_FIXTURE_CORPUS_VERSION))
        for key in ("atom_schema_version", "prompt_version"):
            if record.versions[key] != active_versions[key]:
                diffs += (ReplayDiff("result", "plan", key, "version_changed", record.versions[key], active_versions[key]),)
        if failed and not external:
            if any(diff.aspect not in {"coverage", "stage_events", "operation_status", "final_answer",
                                       "validation_flags", "confidence"} for diff in diffs):
                diffs += (ReplayDiff("result", "verification", "execution", "execution_failed",
                                     root_step_key=next((diff.step_key for diff in diffs if diff.root_step_key is None), None)),)
            else:
                coverage += ("execution_failed",)
        return ReplayResult(trace_id, diffs, coverage, recorded_cost)

    def gate(self, records) -> ReplaySummary:
        return ReplaySummary(tuple(self.replay(record) for record in records))


def _safe_cli_value(value):
    # Display hashes for free text and mappings, including unreviewed JSON passed to the CLI.
    if value is None or type(value) in {int, float, bool}:
        return value
    return {"sha256": _fingerprint(value)}


_SAFE_STEP_SEGMENTS = frozenset({
    "approval_gate", "candidate_runs", "company", "company_worker", "comparative_plan", "comparative_reduce",
    "deep_research", "dispatch", "finalize_single", "generate", "generation", "ground_generate", "hyde",
    "node", "plan", "plan_route", "pot", "prepare_approval", "prepare_durable_turn", "prepare_single",
    "reflect_supplement", "remote_rank", "rephrase", "rerank", "result", "retrieval", "retrieve",
    "retrieve_rerank", "tokens", "tool", "tool_provider", "verification", "worker",
})


def _safe_cli_key(key):
    if key is None or all(segment in _SAFE_STEP_SEGMENTS or
                          re.fullmatch(r"(?:branch|query|reflect|task|wave)-[0-9]{1,4}", segment)
                          for segment in key.split("/")):
        return key
    return "sha256:" + hashlib.sha256(key.encode("ascii")).hexdigest()


def _safe_cli_summary(summary):
    results = []
    for index, result in enumerate(summary.results, 1):
        diffs = []
        for diff in result.diffs:
            item = asdict(diff)
            item["step_key"] = _safe_cli_key(diff.step_key)
            item["root_step_key"] = _safe_cli_key(diff.root_step_key)
            item["expected"] = _safe_cli_value(diff.expected)
            item["actual"] = _safe_cli_value(diff.actual)
            diffs.append(item)
        results.append({"case": index, "passed": result.passed, "diffs": diffs,
                        "coverage_errors": result.coverage_errors, "recorded_cost": result.recorded_cost})
    return {"passed": summary.passed, "exit_code": summary.exit_code, "results": results}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Replay FinaRAG fixture trajectories offline")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--allowed-root", required=True, type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        root = args.allowed_root.absolute()
        store = TrajectoryStore(root, allowed_base_dir=root.parent)
        with open(os.devnull, "w", encoding="utf-8") as quiet:
            with redirect_stdout(quiet), redirect_stderr(quiet):
                summary = ReplayRunner().gate(store.iter_records(args.input, strict_trailing=True))
    except Exception:
        if args.json:
            print(_canonical({"passed": False, "exit_code": 2, "error": "invalid_or_incomplete_fixture"}))
        else:
            print("Invalid or incomplete replay fixture", file=sys.stderr)
        return 2

    payload = _safe_cli_summary(summary)
    if args.json:
        print(_canonical(payload))
    else:
        print(f"{sum(result.passed for result in summary.results)}/{len(summary.results)} replay cases matched")
        for case in payload["results"]:
            for item in case["diffs"]:
                print(f"{case['case']} {item['category']} {item['step_key']} {item['aspect']} {item['reason']}")
            for reason in case["coverage_errors"]:
                print(f"{case['case']} {reason}")
    return summary.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
