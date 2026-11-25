"""P7 trajectory, replay and facade contracts."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from threading import Lock
from types import SimpleNamespace

import pytest


FIXTURE = Path("tests/fixtures/agent_harness/qa_replay.jsonl")
PRIVATE = "private@example.test"


class _RuleProcessor:
    def _resolved_graph_engine(self):
        return "langgraph"

    def route_question(self, question, schema):
        return {}

    def _answer_metric_definition_question(self, **kwargs):
        return {
            "final_answer": PRIVATE,
            "confidence": "low",
            "validation_flags": ["claim_contradicted"],
            "response_data": {"usage": {"prompt_tokens": 7, "completion_tokens": 2}},
        }


def _fixture_records():
    from src.harness.trajectory_store import TrajectoryStore

    store = TrajectoryStore(FIXTURE.parent, allowed_base_dir=FIXTURE.parent.parent)
    return {record.trace_id: record for record in store.iter_records(FIXTURE)}


def test_trajectory_capture_redaction_and_isolation(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from src.graph.builder import run_finarag_graph
    from src.harness.trajectory_store import (
        TrajectoryError, TrajectoryStore, capture_step, observation_stats, record_trajectory, step_scope,
    )
    from src.protocols.tool_registry import MCPToolRegistry, ToolPrincipal, default_policy
    from src.service.fastapi_app import create_fastapi_app
    from src.service.semantic_cache import SemanticCache
    from src.service.streaming import async_stream_qa_events

    root = tmp_path / "trajectories"
    monkeypatch.setenv("FINARAG_TRAJECTORY_ROOT", str(root))
    monkeypatch.setenv("FINARAG_HARNESS_ENABLED", "0")
    store = TrajectoryStore(root, allowed_base_dir=tmp_path)
    assert run_finarag_graph(_RuleProcessor(), PRIVATE, "text")["final_answer"] == PRIVATE
    assert not root.exists()

    monkeypatch.setenv("FINARAG_HARNESS_ENABLED", "yes")
    monkeypatch.setenv("FINARAG_CHECKPOINT_ENABLED", "1")
    monkeypatch.setenv("FINARAG_HITL_ENABLED", "1")
    monkeypatch.setenv("FINARAG_CHECKPOINT_ROOT", str(tmp_path / "sessions"))
    monkeypatch.setenv("SERVICE_API_KEY", "secret-service-key")
    principal = ToolPrincipal("private-owner", frozenset({"read", "reviewer"}))
    cache = SemanticCache()
    cache.set("cached", "text", {"final_answer": PRIVATE, "confidence": "high"})
    processor = _RuleProcessor()

    def graph():
        return run_finarag_graph(processor, PRIVATE, "text")

    def tool():
        def handle(arguments, deadline):
            with step_scope("wave-1/task-2"):
                capture_step(
                    "tool", operation="lookup", step_key="backend",
                    inputs={"authorization": "secret-tool-key"},
                    outputs={"doc_id": "private-doc", "text": PRIVATE},
                )
            raise RuntimeError("private-provider-error")

        name = "query_financial_facts_sql"
        registry = MCPToolRegistry(handlers={name: handle}, policy=default_policy)
        try:
            return asyncio.run(registry.call(name, {
                "operation": "metric", "metric_atom_id": "MET-003", "period": "2024", "company_name": "private-company",
            }, principal=principal))
        finally:
            registry.close()

    def stream():
        async def consume():
            async def disconnected():
                return True

            return [event async for event in async_stream_qa_events(processor, PRIVATE, "text",
                                                                    is_disconnected=disconnected)]
        return asyncio.run(consume())

    with ThreadPoolExecutor(max_workers=3) as workers:
        graph_future = workers.submit(graph)
        tool_future = workers.submit(tool)
        stream_future = workers.submit(stream)
        assert graph_future.result()["final_answer"] == PRIVATE
        assert tool_future.result().error_code == "internal_error"
        assert stream_future.result() == []

    app = create_fastapi_app(processor, cache=cache, protocol_principal_resolver=lambda request: principal)
    headers = {"X-API-Key": "secret-service-key", "Idempotency-Key": "private-turn"}
    with TestClient(app) as client:
        hit = client.post("/qa", json={"question": "cached", "schema": "text"}, headers=headers)
        assert hit.json()["cache_hit"] is True
        pending = client.post("/qa", json={"question": PRIVATE, "schema": "text", "thread_id": "private-thread"},
                              headers=headers)
        assert pending.status_code == 202, pending.text
        prior = pending.json()
        response = client.post(
            "/sessions/private-thread/resume",
            json={key: prior[key] for key in ("checkpoint_id", "interrupt_id")} | {"decision": "reject"},
            headers={**headers, "Idempotency-Key": "private-resume"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["final_answer"] == "N/A"

    path = next(root.glob("*.jsonl"))
    assert root.stat().st_mode & 0o777 == 0o700
    assert path.stat().st_mode & 0o777 == 0o600
    raw = path.read_text()
    for private in (PRIVATE, "secret-tool-key", "secret-service-key", "private-doc", "private-owner",
                    "private-company", "private-thread", "private-provider-error", prior["checkpoint_id"]):
        assert private not in raw
    records = list(store.iter_records(path))
    assert len(records) == 6
    assert len({record.trace_id for record in records}) == len(records)
    assert all(not record.replayable for record in records)
    graph_record = next(r for r in records if r.entry_point == "graph")
    assert graph_record.cost["token_usage"] == {"prompt_tokens": 7, "completion_tokens": 2}
    assert {step.stage for step in graph_record.steps} == {"plan", "node"}
    assert next(step for step in graph_record.steps if step.stage == "plan").outputs["rule_route"] is True
    tool_record = next(r for r in records if r.entry_point == "tool_registry")
    assert {step.step_key for step in tool_record.steps} == {"wave-1/task-2/backend", "tool"}
    assert next(step for step in tool_record.steps if step.step_key == "tool").error_code == "internal_error"
    assert next(r for r in records if r.entry_point == "qa_stream").operation_status == "disconnected"
    cached = next(r for r in records if r.operation_status == "cache_hit")
    assert [step.stage for step in cached.steps] == ["cache"]
    review = next(r for r in records if r.operation_status == "pending_approval")
    continuation = next(r for r in records if r.entry_point == "resume")
    assert review.run_key and continuation.continuation_of == review.run_key

    with path.open("ab") as handle:
        handle.write(b'{"schema_version":')
    before = observation_stats()["capture_failures"]
    assert list(store.iter_records(path)) == records
    assert observation_stats()["capture_failures"] > before
    store.append(records[0])
    assert len(list(store.iter_records(path))) == 7
    with pytest.raises(TrajectoryError):
        list(store.iter_records(root / "../unknown"))
    invalid = root / "invalid.jsonl"
    invalid.write_text(json.dumps({**records[0].to_dict(), "schema_version": 999}) + "\n" + raw)
    with pytest.raises(TrajectoryError, match="version"):
        list(store.iter_records(invalid))
    (root / "escape.jsonl").symlink_to(tmp_path / "outside.jsonl")
    with pytest.raises(TrajectoryError):
        list(store.iter_records(root / "escape.jsonl"))

    def failed_write(*args):
        raise OSError("private-storage-error")

    before = observation_stats()["capture_failures"]
    monkeypatch.setattr(TrajectoryStore, "append", failed_write)
    with record_trajectory(store=store, entry_point="fault", request={}) as observer:
        observer.finish({"final_answer": PRIVATE})
    assert observation_stats()["capture_failures"] == before + 1


def test_offline_replay_and_step_attribution(monkeypatch):
    import httpx
    import requests

    from src.api_requests import APIProcessor
    from src.claim_verifier import ClaimVerifier
    from src.harness.replay_runner import ReplayRunner, _ReplayModel, _ReplayRetriever
    from src.questions_processing import QuestionsProcessor
    from src.storage.sqlite_doc_store import SqliteDocumentRepository

    reached = []

    def forbidden(*args, **kwargs):
        reached.append(True)
        raise AssertionError("Offline replay reached a production boundary")

    for owner, name in ((httpx.Client, "send"), (requests.Session, "send"),
                        (socket.socket, "connect"), (subprocess, "Popen"),
                        (APIProcessor, "__init__"), (SqliteDocumentRepository, "__init__")):
        monkeypatch.setattr(owner, name, forbidden)

    records = _fixture_records()
    runner = ReplayRunner()
    summary = runner.gate(records.values())
    assert summary.passed
    assert summary == runner.gate(records.values())
    with monkeypatch.context() as ambient:
        ambient.setenv("FINARAG_CLAIM_VERIFY_ENABLED", "1")
        assert runner.replay(records["research-ledger"]) == summary.results[2]

    def changed(owner, name, change):
        original = getattr(owner, name)
        return lambda self, *args, **kwargs: change(original(self, *args, **kwargs))

    def change_plan(result):
        result["query_plan"].search_queries = ["fixture intentionally has no reply for this query"]
        return result

    def change_retrieval(results):
        return [{**result, "page": result["page"] + 10} for result in results]

    def change_generation(answer):
        return {**answer, "final_answer": "The synthetic fixture was changed."}

    def check(category, record, *, linked=True):
        result = runner.replay(record)
        assert not result.passed
        roots = {diff.step_key for diff in result.roots if diff.category == category}
        assert roots, result.roots
        if linked:
            assert any(diff.root_step_key in roots and diff.step_key not in roots for diff in result.diffs), result
        return result

    rag = records["rag-vector"]
    for category, record, owner, method, change in (
        ("plan", rag, QuestionsProcessor, "route_question", change_plan),
        ("retrieval", rag, _ReplayRetriever, "retrieve_by_company_name", change_retrieval),
        ("rerank", records["rag-reranked"], _ReplayRetriever, "rerank_candidates", lambda ranked: list(reversed(ranked))),
        ("generation", rag, _ReplayModel, "get_answer_from_rag_context", change_generation),
    ):
        with monkeypatch.context() as mutation:
            mutation.setattr(owner, method, changed(owner, method, change))
            diff = check(category, record)
            if category == "plan":
                assert not diff.coverage_errors
                assert any(item.reason == "external_input_mismatch" and item.root_step_key == "plan_route/plan"
                           for item in diff.diffs)

    research = records["research-ledger"]
    request = deepcopy(research.request)
    tool = next(call for call in request["external_calls"] if call["operation"] == "tool_provider")
    tool.update(status="error", output="unavailable")
    check("tool", replace(research, request=request))

    original_verify = ClaimVerifier.verify

    def stricter_verify(self, *args, **kwargs):
        audit = original_verify(self, *args, **kwargs)
        data = audit.to_dict()
        return SimpleNamespace(to_dict=lambda: data, contradicted_count=audit.contradicted_count + 1)

    with monkeypatch.context() as mutation:
        mutation.setattr(ClaimVerifier, "verify", stricter_verify)
        difference = check("verification", rag, linked=False)
        assert not any(item.category == "generation" for item in difference.roots)
    assert reached == []


def test_replay_cli_ci_gate(tmp_path):
    def cli(path, root):
        command = [
            sys.executable, "-m", "src.harness.replay_runner", "--input", str(path),
            "--allowed-root", str(root), "--json",
        ]
        return subprocess.run(command, capture_output=True, text=True, env=os.environ.copy(), check=False)

    match = cli(FIXTURE, FIXTURE.parent)
    assert match.returncode == 0, match.stdout
    assert json.loads(match.stdout)["passed"] is True

    record = next(iter(_fixture_records().values())).to_dict()
    changed = tmp_path / "changed.jsonl"
    record["final_answer"]["final_answer"] = PRIVATE
    changed.write_text(json.dumps(record) + "\n")
    regressed = cli(changed, tmp_path)
    assert regressed.returncode == 1, regressed.stdout
    assert PRIVATE not in regressed.stdout + regressed.stderr

    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(PRIVATE)
    rejected = cli(invalid, tmp_path)
    assert rejected.returncode == 2
    assert PRIVATE not in rejected.stdout + rejected.stderr

    record["request"]["external_calls"] = []
    invalid.write_text(json.dumps(record) + "\n")
    incomplete = cli(invalid, tmp_path)
    assert incomplete.returncode == 2, incomplete.stdout


def test_facade_signatures_overrides_and_formatting(tmp_path):
    from src import grounding_facade, result_formatter
    from src.questions_processing import QuestionsProcessor, VectorRetriever, _dividend_mention_terms, _legal_rep_event
    from src.retrieval import VectorRetriever as OriginalRetriever

    processor = object.__new__(QuestionsProcessor)
    assert str(inspect.signature(processor.process_question)).startswith("(question: str, schema: str,")
    assert isinstance(inspect.getattr_static(QuestionsProcessor, "_extract_references_from_results"), staticmethod)
    assert isinstance(inspect.getattr_static(QuestionsProcessor, "_document_pages"), staticmethod)
    assert grounding_facade.document_pages({}) == processor._document_pages({})
    assert VectorRetriever is OriginalRetriever and _dividend_mention_terms() and _legal_rep_event()
    assert result_formatter.format_retrieval_results(processor, []) == ""

    processor.documents_dir = tmp_path
    processor._document_cache = {}
    processor._document_cache_lock = Lock()
    (tmp_path / "sample.json").write_text('{"content":{"pages":[{"page":4,"text":"Sample"}]}}')
    document = processor._load_document_payload("sample")
    assert processor._load_document_payload("sample") is document
    assert processor._document_pages(document) == [(4, "Sample")]

    processor.max_doc_chars = 0
    processor.max_context_chars = 0
    processor.table_grounder = None
    processor.chart_grounder = None
    table = processor._table_grounding_retrieval_result({
        "source_doc_id": "A", "page": 4, "table_snippet": "| Revenue | 12 |",
    })
    chart = processor._chart_grounding_retrieval_result({
        "source_doc_id": "B", "page": 8, "chart_context": "Cash flow 9",
    })
    assert processor._format_retrieval_results([table, chart]) == (
        'Text retrieved from page 4 | doc_id: A | chunk_type: serialized_table | node_type: table: \n'
        '"""\n| Revenue | 12 |\n"""\n\n---\n\n'
        'Text retrieved from page 8 | doc_id: B | chunk_type: chart_to_table | node_type: chart: \n'
        '"""\nCash flow 9\n"""'
    )
    references = [{"pdf_sha1": "A", "page": 4}, {"pdf_sha1": "B", "page": 8}]
    assert QuestionsProcessor._extract_references_from_results([4, 8], [table, chart]) == references
    assert processor._extract_references_from_results([4, 8], [table, chart]) == references

    class Override(QuestionsProcessor):
        def _table_grounding_retrieval_result(self, *args, **kwargs):
            return chart

    overridden = Override.__new__(Override)
    assert overridden._ensure_table_grounding_retrieval_result([], {"source_doc_id": "A", "page": 4}) == [chart]
    overridden._table_grounding_retrieval_result = lambda *args, **kwargs: table
    assert overridden._ensure_table_grounding_retrieval_result([], {"source_doc_id": "A", "page": 4}) == [table]
