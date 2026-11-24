"""Offline P5 durability and approval regression tests."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
from typing_extensions import TypedDict


class _State(TypedDict, total=False):
    value: int


def _minimal_graph(saver):
    graph = StateGraph(_State)

    def review(state):
        return {"value": state["value"] + interrupt("review")}

    graph.add_node("review", review)
    graph.add_edge(START, "review")
    graph.add_edge("review", END)
    return graph.compile(checkpointer=saver)


def _config(thread="thread", **fields):
    return {"configurable": {"thread_id": thread, **fields}}


def test_sqlite_saver_sync_async_restart_and_writes(tmp_path):
    from langgraph.checkpoint.base import WRITES_IDX_MAP
    from src.memory import SqliteCheckpointSaver
    from src.query_plan import QueryPlan
    from src.retrieval_filters import RetrievalFilters

    def reopen():
        return SqliteCheckpointSaver(tmp_path / "checkpoints.sqlite", allowed_base_dir=tmp_path)

    saver = reopen()
    graph = _minimal_graph(saver)
    paused = graph.invoke({"value": 2}, _config(), durability="sync")
    checkpoint = saver.inspect_checkpoint("thread")
    assert paused["__interrupt__"][0] == graph.get_state(_config()).interrupts[0]

    async def exercise():
        restarted = reopen()
        resumed = await _minimal_graph(restarted).ainvoke(
            Command(resume=3), _config(), durability="sync",
        )
        assert resumed["value"] == 5
        history = [row async for row in restarted.alist(_config())]
        assert history[0].parent_config == checkpoint.config
        plan = QueryPlan("question", "question", ["question"], RetrievalFilters(company_name="Alpha", year=2023), "explicit_company", "number")
        config = _config("typed", checkpoint_ns="branch", checkpoint_id="one")
        await restarted.aput_writes(config, [("plan", plan), ("send", Send("review", {}))], "typed")
        restarted.put(config, dict(history[0].checkpoint, id="one"), {"test": True}, {})
        assert restarted.get_tuple(config).pending_writes == [
            ("typed", "plan", plan), ("typed", "send", Send("review", {})),
        ]
        for value in ("first", "second"):
            restarted.put_writes(config, [("regular", value), *[
                (channel, value) for channel in WRITES_IDX_MAP
            ]], "duplicate")
        writes = {(task, channel): value for task, channel, value in restarted.get_tuple(config).pending_writes}
        assert writes["duplicate", "regular"] == "first"
        assert all(writes["duplicate", channel] == "second" for channel in WRITES_IDX_MAP)
        assert list(restarted.list(_config("typed"), filter={"test": True}, limit=1))
        assert not list(restarted.list(_config("typed"), before=config))
        assert not list(restarted.list(None, limit=0))
        assert restarted.inspect_checkpoint("typed") is None
        await restarted.adelete_thread("typed")
        restarted.delete_thread("thread")
        assert not list(restarted.list(None))

    asyncio.run(exercise())


def test_checkpoint_history_and_independent_fork(tmp_path):
    from src.memory import SqliteCheckpointSaver

    path = tmp_path / "checkpoints.sqlite"
    saver = SqliteCheckpointSaver(path, allowed_base_dir=tmp_path)
    graph = _minimal_graph(saver)
    graph.invoke({"value": 2}, _config(), durability="sync")
    with pytest.raises(ValueError, match="pending"):
        saver.fork_checkpoint(saver.get_tuple(_config()).config, "forbidden")
    graph.invoke(Command(resume=3), _config(), durability="sync")
    source = graph.get_state(_config())
    fork = saver.fork_checkpoint(source.config, "fork")
    restarted = _minimal_graph(SqliteCheckpointSaver(path, allowed_base_dir=tmp_path))
    assert restarted.get_state(fork).values == source.values
    assert len(list(restarted.get_state_history(_config("fork")))) == len(list(graph.get_state_history(_config())))
    restarted.invoke({"value": 9}, _config("fork"), durability="sync")
    assert restarted.invoke(Command(resume=1), _config("fork"), durability="sync")["value"] == 10
    assert graph.get_state(_config()).config == source.config


def test_compactor_preserves_markdown_tables_and_numeric_citations():
    from src.memory import ContextCompactor

    table = "Name | Value\r\n--- | ---:\r\nA \\| B | 12 [1, 3]\r\n"
    fence = "```text\n| leave | intact |\n```\n"
    turns = [{"question": "Old narrative " * 100, "answer": f"Old [1]\n\n{table}\n{fence}[第12页]"}]
    compactor = ContextCompactor(token_threshold=1, recent_turns=0, estimate_tokens=len)
    compacted = compactor.compact(turns)
    assert compacted.protected_blocks == ("[1]", table, fence, "[第12页]")
    assert compacted.over_budget
    assert compacted == compactor.compact(turns)
    assert "Old narrative" not in compacted.text


class _RouteProcessor:
    """Real graph execution with a local deterministic direct-answer adapter."""

    def __init__(self):
        self.calls = []

    def _resolved_graph_engine(self):
        return "langgraph"

    def route_question(self, question, schema):
        self.calls.append(question)
        return {}

    def _answer_metric_definition_question(self, **kwargs):
        if "safe" in kwargs["question"]:
            return {"final_answer": "verified", "confidence": "high"}
        if "empty" in kwargs["question"]:
            return {"final_answer": "N/A", "confidence": "low", "validation_flags": ["no_context"]}
        return {"final_answer": "private draft", "confidence": "low", "validation_flags": [
            "metric_caliber_conflict_refused" if "caliber" in kwargs["question"] else "claim_contradicted",
        ]}


@pytest.mark.parametrize("question,decision,expected", [
    ("caliber", "approve", "private draft"),
    ("empty", "approve", "N/A"),
    ("contradiction", "reject", "N/A"),
])
def test_approval_resume_without_regeneration(tmp_path, question, decision, expected):
    from src.graph.builder import build_finarag_graph, resume_finarag_graph, run_finarag_graph
    from src.graph.state import FinaRAGRunContext
    from src.memory import SqliteCheckpointSaver

    processor = _RouteProcessor()
    path = tmp_path / "checkpoints.sqlite"
    graph = build_finarag_graph(processor, SqliteCheckpointSaver(path, allowed_base_dir=tmp_path))
    runtime = FinaRAGRunContext(turn_id="turn", approval_enabled=True)
    pending = run_finarag_graph(processor, question, "number", graph=graph, config=_config(), context=runtime)
    assert pending["approval_status"] == "pending"
    assert "private draft" not in repr(pending)
    graph = build_finarag_graph(processor, SqliteCheckpointSaver(path, allowed_base_dir=tmp_path))
    state = graph.get_state(_config())
    answer = resume_finarag_graph(
        processor, graph=graph, config=state.config, context=runtime,
        interrupt_id=state.interrupts[0].id, decision=decision,
    )
    assert answer["final_answer"] == expected
    assert processor.calls == [question]


def test_sessions_auth_history_and_resume_race(tmp_path, monkeypatch):
    from src.protocols.tool_registry import ToolPrincipal
    from src.service.fastapi_app import create_fastapi_app

    monkeypatch.setenv("FINARAG_CHECKPOINT_ENABLED", "yes")
    monkeypatch.setenv("FINARAG_HITL_ENABLED", "true")
    monkeypatch.setenv("FINARAG_CHECKPOINT_ROOT", str(tmp_path))
    monkeypatch.setenv("SERVICE_API_KEY", "local-test")
    processor = _RouteProcessor()
    app = create_fastapi_app(
        processor,
        protocol_principal_resolver=lambda request: ToolPrincipal(
            request.headers.get("test-subject", "owner"),
            frozenset({"read", "reviewer"} if request.headers.get("test-review") else {"read"}),
        ),
    )
    auth = {"X-API-Key": "local-test"}
    with TestClient(app, headers=auth) as client:
        data = {"question": "contradiction", "schema": "number", "thread_id": "alpha"}
        response = client.post("/qa", json=data, headers={"Idempotency-Key": "first"})
        assert response.status_code == 202
        pending = response.json()
        assert "private draft" not in response.text
        assert client.get("/sessions/alpha", headers={"test-subject": "other"}).status_code == 404
        assert client.post("/qa", json=data, headers={"Idempotency-Key": "first"}).status_code == 202
        assert client.post("/qa", json=dict(data, question="another"), headers={"Idempotency-Key": "first"}).status_code == 409
        decision = {name: pending[name] for name in ("checkpoint_id", "interrupt_id")}
        decision["decision"] = "approve"
        headers = {"Idempotency-Key": "review"}
        assert client.post("/sessions/alpha/resume", json=decision, headers=headers).status_code == 403
        headers["test-review"] = "1"
        assert client.post("/sessions/alpha/resume", json=decision, headers=headers).json()["final_answer"] == "private draft"
        assert client.post("/sessions/alpha/resume", json=decision, headers=headers).status_code == 200
        assert client.post("/sessions/alpha/resume", json=decision, headers=dict(headers, **{"Idempotency-Key": "stale"})).status_code == 409
        first = client.get("/sessions/alpha/history?limit=1").json()["history"]
        assert len(first) == 1
        following = client.get("/sessions/alpha/history", params={"before": first[0]["checkpoint_id"]}).json()["history"]
        assert following and first[0]["checkpoint_id"] != following[0]["checkpoint_id"]
        assert "private draft" not in repr(first + following)
        assert processor.calls == ["contradiction"]


def test_session_routes_disabled(monkeypatch):
    from src.service.fastapi_app import create_fastapi_app

    monkeypatch.setenv("FINARAG_CHECKPOINT_ENABLED", "0")
    monkeypatch.setenv("FINARAG_HITL_ENABLED", "1")
    with TestClient(create_fastapi_app(_RouteProcessor())) as client:
        assert client.get("/sessions/alpha").status_code == 404


@pytest.fixture
def session_app(tmp_path, monkeypatch):
    from src.protocols.tool_registry import ToolPrincipal
    from src.service.fastapi_app import create_fastapi_app

    monkeypatch.setenv("FINARAG_CHECKPOINT_ENABLED", "1")
    monkeypatch.setenv("FINARAG_HITL_ENABLED", "1")
    monkeypatch.setenv("FINARAG_CHECKPOINT_ROOT", str(tmp_path / "sessions"))
    monkeypatch.setenv("SERVICE_API_KEY", "local-test")
    return lambda processor, **kwargs: create_fastapi_app(
        processor,
        protocol_principal_resolver=lambda request: ToolPrincipal("owner", frozenset({"read", "reviewer"})),
        **kwargs,
    )


def _headers(key):
    return {"X-API-Key": "local-test", "Idempotency-Key": key}


def _review_request(client):
    pending = client.post(
        "/qa", json={"question": "caliber", "schema": "number", "thread_id": "audit"},
        headers=_headers("turn"),
    )
    assert pending.status_code == 202
    return {name: pending.json()[name] for name in ("checkpoint_id", "interrupt_id")} | {"decision": "approve"}


@pytest.mark.parametrize("phase", ["before_input", "input_checkpoint", "after_input", "before_projection"])
def test_turn_recovery_across_app_restart(session_app, monkeypatch, phase):
    from src.storage.base import StorageError

    processor = _RouteProcessor()
    app = session_app(processor)
    data = {"question": "safe", "thread_id": "audit"}
    with TestClient(app) as client:
        memory, _, _ = app.state.session_supplier.get()
        if phase == "before_input":
            def fail_input(*args, **kwargs):
                raise StorageError("private DB error")
            monkeypatch.setattr(memory.saver, "put", fail_input)
        elif phase == "input_checkpoint":
            put = memory.saver.put

            def fail_first_loop(config, checkpoint, metadata, new_versions):
                if metadata["source"] != "input":
                    raise StorageError("interrupted after input checkpoint")
                return put(config, checkpoint, metadata, new_versions)
            monkeypatch.setattr(memory.saver, "put", fail_first_loop)
        elif phase == "after_input":
            def fail_route(question, schema):
                raise RuntimeError("private generation error")
            monkeypatch.setattr(processor, "route_question", fail_route)
        else:
            def fail_projection(*args, **kwargs):
                raise StorageError("private DB error")
            monkeypatch.setattr(memory, "_finish", fail_projection)
        response = client.post("/qa", json=data, headers=_headers("turn"))
        assert response.status_code in (500, 503)
        assert "private" not in response.text
        assert client.post("/qa", json=data, headers=_headers("different")).status_code == 409

    if phase == "after_input":
        monkeypatch.delattr(processor, "route_question")
    with TestClient(session_app(processor)) as client:
        retried = client.post("/qa", json=data, headers=_headers("turn"))
        assert retried.status_code == 200, retried.text
        assert retried.json()["final_answer"] == "verified"
        assert client.get("/sessions/audit", headers=_headers("")).json()["memory"]["turn_count"] == 1
        assert processor.calls == ["safe"]


@pytest.mark.parametrize("phase", ["before_resume_write", "after_resume_write", "before_projection"])
def test_resume_recovery_across_app_restart(session_app, monkeypatch, phase):
    from langgraph.checkpoint.serde.types import RESUME
    from src.storage.base import StorageError

    processor = _RouteProcessor()
    app = session_app(processor)
    with TestClient(app) as client:
        body = _review_request(client)
        memory, _, _ = app.state.session_supplier.get()
        if phase == "before_resume_write":
            put_writes = memory.saver.put_writes

            def fail_before_resume(config, writes, task_id, task_path=""):
                if any(channel == RESUME for channel, _ in writes):
                    raise StorageError("simulated process interruption")
                return put_writes(config, writes, task_id, task_path)
            monkeypatch.setattr(memory.saver, "put_writes", fail_before_resume)
        elif phase == "after_resume_write":
            put = memory.saver.put

            def fail_after_resume(config, checkpoint, metadata, new_versions):
                if checkpoint["channel_values"].get("approval_status") == "approved":
                    raise StorageError("simulated process interruption")
                return put(config, checkpoint, metadata, new_versions)
            monkeypatch.setattr(memory.saver, "put", fail_after_resume)
        else:
            def fail_projection(*args, **kwargs):
                raise StorageError("simulated process interruption")
            monkeypatch.setattr(memory, "_finish", fail_projection)

        failure = client.post("/sessions/audit/resume", json=body, headers=_headers("resume"))
        assert failure.status_code == 503
        if phase == "after_resume_write":
            thread = memory.open_thread("audit", subject="owner")["internal_id"]
            saved = memory.saver.inspect_checkpoint(thread)
            assert any(channel == RESUME for _, channel, _ in saved.pending_writes)
            assert saved.checkpoint["channel_values"]["approval_status"] == "pending"

    with TestClient(session_app(processor)) as client:
        recovered = client.post("/sessions/audit/resume", json=body, headers=_headers("resume"))
        assert recovered.status_code == 200, recovered.text
        assert recovered.json()["approval_status"] == "approved"
        assert processor.calls == ["caliber"]
        assert client.get("/sessions/audit", headers=_headers("")).json()["memory"]["turn_count"] == 1


def test_cross_worker_resume_lock(session_app, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    processor = _RouteProcessor()
    first_app = session_app(processor)
    with TestClient(first_app) as first, TestClient(session_app(processor)) as second:
        body = _review_request(first)
        memory, _, _ = first_app.state.session_supplier.get()
        finish = memory._finish
        entered, release = Event(), Event()

        def pause_projection(*args, **kwargs):
            entered.set()
            assert release.wait(10)
            return finish(*args, **kwargs)

        monkeypatch.setattr(memory, "_finish", pause_projection)
        with ThreadPoolExecutor() as pool:
            first_write = pool.submit(first.post, "/sessions/audit/resume", json=body, headers=_headers("resume"))
            try:
                assert entered.wait(10)
                racing = second.post("/sessions/audit/resume", json=body, headers=_headers("racer"))
                assert racing.status_code == 409
                assert racing.json()["error"] == "session_busy"
            finally:
                release.set()
            assert first_write.result(10).status_code == 200


def test_session_sse_dispatch_and_cache_bypass(session_app):
    import json

    from src.service.semantic_cache import SemanticCache

    processor = _RouteProcessor()
    cache = SemanticCache()
    cache.set("caliber", "number", {"final_answer": "cached", "confidence": "high"})
    with TestClient(session_app(processor, cache=cache)) as client:
        response = client.post("/qa/stream", json={
            "question": "caliber", "schema": "number", "thread_id": "audit",
        }, headers=_headers("turn"))
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["x-content-type-options"] == "nosniff"
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        assert events[0]["stage"] == "route", events
        assert events[-1]["stage"] == "approval_pending"
        assert not any(event["stage"] == "done" for event in events)
        assert "private draft" not in response.text
        assert processor.calls == ["caliber"]
        assert cache.get("caliber", "number")["final_answer"] == "cached"


def test_enabled_sessions_keep_fake_and_legacy_threadless_paths(session_app):
    class Fake:
        def process_question(self, question, schema):
            return {"final_answer": "fake", "confidence": "high"}

        def stream_question_events(self, question, schema, **kwargs):
            yield {"stage": "result", "answer_dict": {"final_answer": "fake"}}

    class Legacy(Fake):
        def _resolved_graph_engine(self):
            return "legacy"

    for processor in (Fake(), Legacy()):
        with TestClient(session_app(processor)) as client:
            assert client.post("/qa", json={"question": "test"}, headers=_headers("")).json()["final_answer"] == "fake"
            thread = client.post("/qa", json={"question": "test", "thread_id": "unsupported"}, headers=_headers("key"))
            assert thread.status_code == 503
            assert thread.json()["error"] == "session_engine_unavailable"
            stream = client.post("/qa/stream", json={"question": "test"}, headers=_headers(""))
            assert '"stage": "done"' in stream.text


def test_durable_real_processor_reflection_working_context_and_reset(session_app, monkeypatch):
    from test_langgraph_pipeline import LangGraphPipelineTests, _FakeRetriever, _make_result
    from src.query_plan import QueryPlan

    fixture = LangGraphPipelineTests()
    fixture.setUp()
    try:
        processor = fixture._build_processor(use_vector_dbs=True, llm_reranking=False)
        monkeypatch.setenv("ADAPTIVE_RETRIEVAL_ENABLED", "1")
        monkeypatch.setenv("ADAPTIVE_MAX_ROUNDS", "1")
        initial = [_make_result(2, "2023年公司实现归母净利润367亿元", report_year=2023)]
        supplement = [_make_result(19, "2024年公司实现营业收入170748百万元")]
        retriever = _FakeRetriever(initial, supplement)
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            page = 19 if "170748" in kwargs["rag_context"] else 2
            return {
                "final_answer": "1707.48亿元" if page == 19 else "367亿元",
                "relevant_pages": [page], "reasoning_summary": "", "step_by_step_analysis": "",
            }

        monkeypatch.setattr(processor, "_build_retriever", lambda: (retriever, "vector"))
        monkeypatch.setattr(processor, "_record_adaptive_trace", lambda *args, **kwargs: None)
        monkeypatch.setattr(processor.api_processor, "get_answer_from_rag_context", generate)
        app = session_app(processor)
        with TestClient(app) as client:
            response = client.post("/qa", json={
                "question": "浦发银行2024年的主要经营业绩如何？", "schema": "text", "thread_id": "real",
            }, headers=_headers("one"))
            assert response.status_code == 200, response.text
            assert response.json()["final_answer"] == "1707.48亿元"
            assert len(calls) == 2 and retriever.supplement_calls
            memory, graph, _ = app.state.session_supplier.get()
            internal_id = memory.open_thread("real", subject="owner")["internal_id"]
            first = graph.get_state(_config(internal_id)).values
            assert isinstance(first["query_plan"], QueryPlan)
            assert "retriever_ref" not in first and "budget_context" not in first
            assert len(first["adaptive_traces"]) == 1
            retriever._initial = supplement
            response = client.post("/qa", json={
                "question": "那么主要风险管理举措有哪些？", "schema": "text", "thread_id": "real",
            }, headers=_headers("two"))
            assert response.status_code == 200, response.text
            second = graph.get_state(_config(internal_id)).values
            assert second["query_plan"].filters.company_name == "浦发银行"
            assert second["query_plan"].filters.year == 2024
            assert second["adaptive_traces"] == []
            assert len([event for event in second["stage_events"] if event["stage"] == "retrieve_rerank"]) == 1
            assert "Untrusted session history" in calls[-1]["rag_context"]
            assert calls[-1]["question"] == "那么主要风险管理举措有哪些？"
    finally:
        fixture.tearDown()


def test_sqlite_reader_snapshot_and_strict_types(tmp_path):
    from dataclasses import dataclass

    from langgraph.checkpoint.base import empty_checkpoint
    from src.memory import SqliteCheckpointSaver
    from src.storage.base import StorageError

    path = tmp_path / "checkpoints.sqlite"
    saver = SqliteCheckpointSaver(path, allowed_base_dir=tmp_path)
    independent = SqliteCheckpointSaver(path, allowed_base_dir=tmp_path)
    for i in range(70):
        saver.put(_config(), dict(empty_checkpoint(), id=f"{i:03d}"), {}, {})

    async def exercise():
        history = saver.alist(_config())
        head = await anext(history)
        assert head.checkpoint["id"] == "069"
        await asyncio.gather(
            independent.aput(_config(), dict(empty_checkpoint(), id="000-async"), {}, {}),
            asyncio.to_thread(independent.put, _config(), dict(empty_checkpoint(), id="000-sync"), {}, {}),
        )
        assert len([item async for item in history]) == 69
        assert len(list(saver.list(_config()))) == 72
    asyncio.run(exercise())

    @dataclass
    class Unsupported:
        data: int

    snapshot = dict(empty_checkpoint(), id="unsupported", channel_values={"result": Unsupported(5)})
    with pytest.raises(StorageError, match="Unsupported checkpoint value"):
        saver.put(_config(), snapshot, {}, {})
    with saver._connection() as conn:
        conn.execute("UPDATE checkpoints SET checkpoint_blob = ? WHERE checkpoint_id = ?", (b"\xc1", "069"))
    with pytest.raises(StorageError, match="Corrupt checkpoint data"):
        saver.inspect_checkpoint("thread", "069")


def test_compaction_preserves_conflicts_and_source_identity():
    from src.memory import ContextCompactor

    facts = [
        {"value": value, "unit": "万元", "metric": "营业收入", "entity": "Alpha", "period": 2024,
         "source": {"document": source, "page": 19}, "citation_id": "[1]", "verified": True}
        for value, source in [("10.00", "doc-a"), ("12.50", "doc-a"), ("10.00", "doc-b")]
    ]
    turns = [
        {"turn_id": "first", "answer": "Old narrative " * 500 + "[1]", "facts": facts[:1], "sources": ["doc-a"]},
        {"turn_id": "latest", "answer": "Updated [1]", "facts": facts, "sources": ["doc-b"]},
    ]
    context = ContextCompactor(token_threshold=2000, recent_turns=1, estimate_tokens=len).compact(turns)
    assert not context.over_budget
    assert context.facts == tuple(facts)
    assert context.protected_blocks == ("[1]", "[1]")
    assert [item["sources"] for item in context.protected_provenance] == [["doc-a"], ["doc-b"]]
    assert "Old narrative" not in context.text


def test_sse_conflicting_turn_is_409(session_app):
    with TestClient(session_app(_RouteProcessor())) as client:
        _review_request(client)
        data = {"question": "new turn", "thread_id": "audit"}
        response = client.post("/qa/stream", json=data, headers=_headers("another"))
        assert response.status_code == 409
        assert response.json()["error"] == "session_pending"


def test_bounded_episode_keeps_table_line_boundaries():
    from src.memory.context_compactor import ContextCompactor, bounded_narrative

    table = "| Metric | Value |\r\n|---|---|\r\n| Revenue | 12 [1] |\r\n"
    bounded = bounded_narrative("Long narrative before the table\r\n" + table, 4)
    assert ContextCompactor().compact([{"answer": bounded}]).protected_blocks == (table,)


def test_checkpoint_enabled_without_hitl(session_app, monkeypatch):
    monkeypatch.setenv("FINARAG_HITL_ENABLED", "0")
    with TestClient(session_app(_RouteProcessor())) as client:
        response = client.post("/qa", json={
            "question": "caliber", "schema": "number", "thread_id": "audit",
        }, headers=_headers("turn"))
        assert response.status_code == 200
        assert response.json()["approval_status"] == "not_required"
        assert response.json()["final_answer"] == "private draft"
