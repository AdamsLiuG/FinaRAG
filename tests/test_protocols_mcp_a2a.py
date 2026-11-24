import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.protocols.a2a_server import A2ATaskStore
from src.protocols.mcp_server import MCPServer
from src.protocols.tool_registry import (
    MCPToolRegistry,
    ToolPrincipal,
    build_financial_tool_registry,
    default_policy,
)
from src.service.fastapi_app import create_fastapi_app
from src.structured.sql_store import FinancialSQLStore


PRINCIPAL = ToolPrincipal("reader", frozenset({"read"}))
POT = {"expression": "a * 2", "variables": [{"name": "a", "value": 3, "unit": "元"}], "result_unit": "元"}
RESOURCES = Path(__file__).resolve().parents[1] / "src/protocols/resources"


def rpc(method, params=None, request_id=1):
    message = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def notice(method):
    return {"jsonrpc": "2.0", "method": method}


@pytest.fixture
def corpus(tmp_path):
    facts = tmp_path / "finarag_facts.sqlite"
    with sqlite3.connect(facts) as conn:
        FinancialSQLStore.create_schema(conn)
        FinancialSQLStore.create_indexes(conn)
        conn.execute("INSERT INTO reports VALUES (?, ?, ?, ?, ?, ?, ?)",
                     ("report_1", "report_1", "示例公司", "000001", "制造业", 2024, "CNY"))
        conn.executemany("INSERT INTO financial_facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
            ("c", "report_1", "income_statement", "consolidated", "营业收入", "MET-004",
             "2024", 90, "元", 90, 2, "table_1", 1, "2024"),
            ("p", "report_1", "income_statement", "parent", "营业收入", "MET-004",
             "2024", 9, "元", 9, 2, "table_1", 2, "2024"),
        ])
    return tmp_path


def test_registry_strict_schema_policy_and_timeout(corpus):
    before = (corpus / "finarag_facts.sqlite").read_bytes()
    release, entered, exited = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def blocked(arguments, deadline):
        calls.append(arguments)
        assert deadline > time.monotonic()
        entered.set()
        try:
            assert release.wait(3)
            return {"ok": True}
        finally:
            exited.set()

    registry = MCPToolRegistry(handlers={"pot_calculate": blocked}, policy=default_policy, max_workers=1)
    real = build_financial_tool_registry(corpus, policy=default_policy)

    async def exercise():
        assert {tool["name"] for tool in real.list_tools(PRINCIPAL)} == {
            "query_financial_facts_sql", "pot_calculate", "navigate_toc_sections",
            "query_financial_kg", "hybrid_search_reports", "verify_claims",
        }
        schema = next(tool["inputSchema"] for tool in real.list_tools(PRINCIPAL)
                      if tool["name"] == "pot_calculate")
        assert schema["additionalProperties"] is False
        assert schema["$defs"]["PotInputVariable"]["additionalProperties"] is False
        for bad in (
            {**POT, "ndigits": True}, {**POT, "ndigits": "3"}, {**POT, "extra": 3},
            {**POT, "variables": [{"name": "a", "value": float("inf"), "unit": "元"}]},
            {**POT, "variables": [{"name": "a", "value": 1, "unit": "元", "doc_id": "../escape", "page": 1}]},
            {**POT, "expression": "a.real"}, {**POT, "expression": "unknown(a)"},
            {**POT, "variables": POT["variables"] * 2},
        ):
            assert (await registry.call("pot_calculate", bad, principal=PRINCIPAL)).error_code == "invalid_arguments"
        for raw in ('{"expression":"x","expression":"y"}', '{"x":NaN}', '{"x":1e400}'):
            assert (await registry.call("pot_calculate", raw, principal=PRINCIPAL)).error_code == "invalid_arguments"
        for principal in (None, ToolPrincipal("", frozenset({"read"})), ToolPrincipal("guest", frozenset())):
            assert (await registry.call("pot_calculate", POT, principal=principal)).error_code == "forbidden"
        assert calls == []
        pending = asyncio.create_task(registry.call("pot_calculate", POT, principal=PRINCIPAL, timeout_s=.15))
        assert await asyncio.to_thread(entered.wait, 2)
        assert (await pending).error_code == "timeout"
        assert (await registry.call("pot_calculate", POT, principal=PRINCIPAL)).error_code == "busy"
        release.set()
        assert await asyncio.to_thread(exited.wait, 2)
        for _ in range(100):
            result = await registry.call("pot_calculate", POT, principal=PRINCIPAL)
            if result.error_code != "busy":
                break
            await asyncio.sleep(.005)
        assert result.data == {"ok": True}
        for scope, fact_id, value in (("consolidated", "c", 90), ("parent", "p", 9)):
            result = await real.call("query_financial_facts_sql", {
                "operation": "metric", "period": "2024", "metric_atom_id": "MET-004",
                "company_name": "示例公司", "scope": scope,
            }, principal=PRINCIPAL)
            row, = result.data["rows"]
            assert (row["fact_id"], row["doc_id"], row["source_page"], row["table_id"],
                    row["normalized_cny"], row["scope_type"]) == (fact_id, "report_1", 2, "table_1", value, scope)

    try:
        asyncio.run(exercise())
    finally:
        release.set()
        registry.close()
        real.close()
    assert (corpus / "finarag_facts.sqlite").read_bytes() == before


def test_mcp_stdio_and_http_contract(corpus, monkeypatch):
    frames = [
        rpc("tools/list"),
        rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}),
        notice("notifications/initialized"), rpc("ping"),
        rpc("tools/list"), rpc("tools/call", {"name": "pot_calculate", "arguments": POT}),
        rpc("resources/list"), rpc("resources/read", {"uri": "finarag://schemas/config"}),
        rpc("resources/read", {"uri": "finarag://schemas/../config"}), rpc("unknown"),
        [notice("notifications/ignored"), rpc("ping", request_id="batch")],
    ]
    for index, frame in enumerate(frames, 1):
        if isinstance(frame, dict) and "id" in frame:
            frame["id"] = index
    env = {**os.environ, "FINARAG_MCP_ENABLED": "0", "FINARAG_A2A_ENABLED": "0"}
    wire = "\n".join(map(json.dumps, [*frames, rpc("ping", request_id="batch")]))
    wire += '\n{\n{"jsonrpc":"2.0","method":"ping","id":1,"id":2}\n'
    process = subprocess.run([sys.executable, "-m", "src.protocols.mcp_server", "--corpus-root", str(corpus)],
                             input=wire, text=True, capture_output=True, env=env, timeout=25, check=True)
    lines = [json.loads(line) for line in process.stdout.splitlines()]
    assert "error" in lines[0]
    assert lines[1]["result"]["protocolVersion"] == "2025-03-26"
    assert lines[2]["result"] == {}
    assert len(lines[3]["result"]["tools"]) == 6
    assert json.loads(lines[4]["result"]["content"][0]["text"])["value"] == 6
    assert len(lines[5]["result"]["resources"]) == 2
    assert json.loads(lines[6]["result"]["contents"][0]["text"])["type"] == "object"
    assert lines[7]["error"]["code"] == -32602
    assert lines[8]["error"]["code"] == -32601
    assert lines[9] == [{"jsonrpc": "2.0", "id": "batch", "result": {}}]
    assert [line["error"]["code"] for line in lines[10:]] == [-32600, -32700, -32700]

    monkeypatch.setenv("FINARAG_MCP_ENABLED", "true")
    monkeypatch.setenv("FINARAG_A2A_ENABLED", "0")
    monkeypatch.setenv("FINARAG_MCP_ALLOWED_ORIGINS", "https://client.example")
    monkeypatch.setenv("SERVICE_API_KEY", "test-key")
    registry = build_financial_tool_registry(corpus, policy=default_policy)
    server = MCPServer(registry, resources_root=RESOURCES)
    headers = {"Authorization": "Bearer test-key", "Origin": "https://client.example",
               "Accept": "application/json, text/event-stream"}
    with TestClient(create_fastapi_app(mcp_server_provider=lambda: server)) as client:
        assert client.get("/mcp").status_code == 405
        assert client.post("/mcp", json=rpc("ping")).status_code == 401
        monkeypatch.delenv("SERVICE_API_KEY")
        assert client.post("/mcp", json=rpc("ping"), headers=headers).status_code == 401
        monkeypatch.setenv("SERVICE_API_KEY", "test-key")
        assert client.post("/mcp", json=rpc("ping"),
                           headers={**headers, "Origin": "https://other.example"}).status_code == 403
        assert client.post("/mcp", json=rpc("ping"),
                           headers={**headers, "Accept": "text/event-stream"}).status_code == 406
        assert client.post("/mcp", json=rpc("ping"),
                           headers={**headers, "MCP-Protocol-Version": "wrong"}).status_code == 400
        assert client.post("/mcp", content=b"x" * (64 * 1024 + 4097),
                           headers={**headers, "Content-Type": "application/json"}).status_code == 413
        for frame in frames[1:10]:
            response = client.post("/mcp", json=frame, headers=headers)
            assert response.headers["x-content-type-options"] == "nosniff"
            assert response.headers["x-trace-id"]
            if "id" not in frame:
                assert (response.status_code, response.content) == (202, b"")
            else:
                assert response.json()["id"] == frame["id"]
        rejected = client.post("/mcp", headers=headers, json=rpc("tools/call", {
            "name": "pot_calculate", "arguments": {**POT, "ndigits": True},
        }))
        assert rejected.json()["result"]["isError"] is True
        batch = [rpc("ping", request_id=9), notice("notifications/ignored"), rpc("ping", request_id=8)]
        assert [row["id"] for row in client.post("/mcp", json=batch, headers=headers).json()] == [9, 8]
        assert client.post("/mcp", json=[batch[0], batch[0]], headers=headers).json()["error"]["code"] == -32600
    assert (asyncio.run(registry.call("pot_calculate", POT, principal=PRINCIPAL))).error_code == "unavailable"


def test_a2a_lifecycle_cancel_and_ownership(monkeypatch):
    monkeypatch.setenv("FINARAG_A2A_ENABLED", "yes")
    monkeypatch.setenv("FINARAG_MCP_ENABLED", "0")
    monkeypatch.setenv("FINARAG_A2A_PUBLIC_URL", "https://finarag.example")
    monkeypatch.setenv("SERVICE_API_KEY", "test-key")
    release, entered = threading.Event(), threading.Event()

    def run(question, schema):
        if question == "blocked":
            entered.set()
            assert release.wait(3)
        if question == "failure":
            raise RuntimeError("private provider detail")
        return {"final_answer": "answer", "confidence": .9, "references": [{"doc_id": "report_1", "page": 2}]}

    def resolve(request):
        return ToolPrincipal("secondary" if request.headers.get("x-fixture-owner") == "2" else "primary", frozenset({"read"}))

    def poll(client, url, headers, state):
        for _ in range(100):
            response = client.get(url, headers=headers)
            if response.json().get("status") == state:
                return response.json()
            time.sleep(.01)
        pytest.fail(f"task did not reach {state}")

    store = A2ATaskStore(run_question=run, max_active=1, max_retained=5, ttl_s=.4)
    auth = {"Authorization": "Bearer test-key"}
    other = {**auth, "x-fixture-owner": "2"}
    payload = {"question": "blocked", "schema": "string"}
    with TestClient(create_fastapi_app(a2a_store_provider=lambda: store,
                                       protocol_principal_resolver=resolve)) as client:
        card = client.get("/.well-known/agent-card.json", headers={"Host": "untrusted.example"}).json()
        assert card == client.get("/.well-known/agent.json").json()
        assert (card["protocolVersion"], card["url"], card["preferredTransport"]) == ("0.3.0", "https://finarag.example/a2a", "JSONRPC")
        assert client.post("/a2a/tasks", json=payload).status_code == 401
        assert client.post("/a2a/tasks", headers=auth, json={**payload, "roles": ["admin"]}).status_code == 422
        blocked = client.post("/a2a/tasks", json=payload, headers=auth)
        assert (blocked.status_code, blocked.json()["status"]) == (202, "submitted")
        url = blocked.headers["Location"]
        assert entered.wait(2)
        poll(client, url, auth, "working")
        assert client.get(url, headers=other).json() == client.get("/a2a/tasks/missing", headers=other).json()
        assert client.get(url, headers=other).status_code == 404
        assert client.post(url + "/cancel", headers=other).status_code == 404
        for _ in range(2):
            assert client.post(url + "/cancel", headers=auth).json()["status"] == "canceled"
        assert client.post("/a2a/tasks", json=payload, headers=auth).status_code == 429
        release.set()
        send = rpc("message/send", {"message": {"kind": "message", "role": "user", "messageId": "msg",
                                                 "parts": [{"kind": "text", "text": "execute"}]}})
        for _ in range(100):
            response = client.post("/a2a", json=send, headers=auth).json()
            if "result" in response:
                break
            time.sleep(.01)
        task_id = response["result"]["id"]
        complete_url = f"/a2a/tasks/{task_id}"
        done = poll(client, complete_url, auth, "completed")
        assert done["artifacts"][0]["parts"][0]["text"] == "answer"
        assert client.get(url, headers=auth).json()["status"] == "canceled"
        assert client.post(complete_url + "/cancel", headers=auth).status_code == 409
        get = rpc("tasks/get", {"id": task_id})
        assert client.post("/a2a", json=get, headers=auth).json()["result"]["status"]["state"] == "completed"
        assert client.post("/a2a", json=get, headers=other).json()["error"]["code"] == -32001
        cancel = rpc("tasks/cancel", {"id": blocked.json()["id"]})
        assert client.post("/a2a", json=cancel, headers=auth).json()["result"]["status"]["state"] == "canceled"
        failed = client.post("/a2a/tasks", json={"question": "failure", "schema": "string"}, headers=auth)
        assert "private" not in str(poll(client, failed.headers["Location"], auth, "failed"))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and client.get(url, headers=auth).status_code != 404:
            time.sleep(.02)
        assert client.get(url, headers=auth).status_code == 404


def test_protocol_routes_disabled_by_default(monkeypatch):
    for name in ("FINARAG_MCP_ENABLED", "FINARAG_A2A_ENABLED", "SERVICE_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    def unused():
        pytest.fail("disabled provider was used")

    class FakeProcessor:
        def process_question(self, question, schema):
            return {"final_answer": "legacy", "references": []}

    with TestClient(create_fastapi_app(FakeProcessor(), mcp_server_provider=unused,
                                       a2a_store_provider=unused)) as client:
        for path in ("/mcp", "/a2a", "/a2a/tasks", "/.well-known/agent.json", "/.well-known/agent-card.json"):
            assert client.get(path).status_code == 404
            assert client.post(path).status_code == 404
        assert client.get("/healthz").status_code == 200
        assert client.get("/metrics/cache").status_code == 200
        assert client.post("/qa", json={"question": "legacy", "use_cache": False}).json()["final_answer"] == "legacy"


def test_report_adapters_filter_provenance_and_fuse_local_ranks(corpus):
    import numpy as np

    from src.knowledge_graph.kg_builder import EntityNode, FinancialKGStore, RelationEdge
    from src.multimodal_nav.toc_tree_builder import ReportTocTree, TocNode, TocTreeStore
    from src.storage.base import StorageSecurityError
    from src.storage.qdrant_store import QdrantHybridStore
    from src.storage.sqlite_doc_store import SqliteDocumentRepository

    chunked = corpus / "chunked_reports"
    chunked.mkdir()
    text = "示例公司2024年营业收入为90元。"
    table = "| Revenue | Amount |\n|---|---|\n" + "| Sales | 90 |\n" * 300
    chunks = [{"page": 2, "chunk_id": 0, "text": f"{text}\n{table}"},
              {"page": 3, "chunk_id": 1, "text": "Other disclosures"}]
    for stem, doc_id, company in (("report_alias", "report_1", "示例公司"),
                                  ("report_2", "report_2", "外国公司")):
        (chunked / f"{stem}.json").write_text(json.dumps({
            "metainfo": {"doc_id": doc_id, "sha1_name": doc_id, "company_name": company,
                         "report_year": 2024, "currency": "CNY"},
            "content": {"pages": [{"page": chunk["page"], "text": chunk["text"]} for chunk in chunks],
                        "chunks": chunks},
        }), encoding="utf-8")
    SqliteDocumentRepository.build_from_chunked_dir(chunked, corpus / "finarag_docs.sqlite", allowed_base_dir=corpus)
    TocTreeStore._write_trees_sqlite([
        ReportTocTree("report_1", "示例公司", 2024, 3,
                      [TocNode("s1", 1, "营业收入", 2, 2)]),
    ], corpus / "finarag_toc_trees.sqlite", allowed_base_dir=corpus)
    names = ("示例公司", "证据公司", "外国公司", "缺页公司")
    FinancialKGStore.write_kg_sqlite(
        [EntityNode(f"n{i}", name) for i, name in enumerate(names)],
        [RelationEdge("e1", names[0], "HAS_SHAREHOLDER", names[1], evidence_doc_id="report_1", evidence_page=2),
         RelationEdge("e2", names[0], "HAS_SHAREHOLDER", names[2], evidence_doc_id="report_2", evidence_page=2),
         RelationEdge("e3", names[0], "HAS_SHAREHOLDER", names[3], evidence_doc_id="report_1", evidence_page=9)],
        corpus / "finarag_kg.sqlite", allowed_base_dir=corpus,
    )
    store = QdrantHybridStore(store_dir=corpus / "unified_store", allowed_base_dir=corpus, dense_dim=2)
    try:
        store.upsert_points([
            {"point_id": i, "doc_id": doc, "chunk_idx": index,
             "page": chunks[index]["page"], "text": chunks[index]["text"],
             "dense_vector": vector, "bge_sparse": {"financial": sparse}}
            for i, (doc, index, vector, sparse) in enumerate([
                ("report_1", 0, [1, 0], 2.0), ("report_1", 1, [0, 1], 0.1),
                ("report_2", 0, [1, 0], 3.0),
            ])
        ])
    finally:
        store.close()

    db_files = [path for path in corpus.rglob("*") if path.suffix in {".sqlite", ".faiss"}]
    before = {path: path.read_bytes() for path in db_files}
    registry = build_financial_tool_registry(
        corpus, policy=default_policy, dense_encoder=lambda _: np.array([1, 0]),
        sparse_encoder=lambda _: {"financial": 1.0},
    )
    lexical = build_financial_tool_registry(corpus, policy=default_policy, dense_encoder=lambda _: np.array([1]))
    alias_only = build_financial_tool_registry(
        corpus, policy=lambda _, name, args: name == "navigate_toc_sections" and args["doc_id"] == "report_alias",
    )

    async def exercise():
        async def call(name, args, tool_registry=registry):
            result = await tool_registry.call(name, args, principal=PRINCIPAL)
            assert result.error_code is None, (name, result.error_code)
            return result.data

        navigation_args = {"doc_id": "report_alias", "question": "营业收入"}
        denied = await alias_only.call("navigate_toc_sections", navigation_args, principal=PRINCIPAL)
        assert denied.error_code == "forbidden"
        toc = await call("navigate_toc_sections", navigation_args)
        assert toc["doc_id"] == "report_1"
        page, = toc["pages"]
        assert page["section_path"] == "营业收入"
        assert "[Markdown table omitted]" in page["excerpt"]
        assert "| Sales |" not in page["excerpt"]
        kg = await call("query_financial_kg", {"question": "示例公司的股东", "company_name": "示例公司",
                                               "doc_ids": ["report_1"]})
        path, = kg["paths"]
        assert path["edges"][0]["evidence_doc_id"] == "report_1"
        assert [entry["page"] for entry in kg["evidence_pages"]] == [2]
        assert "外国公司" not in kg["context_summary"] and "缺页公司" not in kg["context_summary"]
        search_args = {"query": "Revenue", "doc_ids": ["report_1"], "top_k": 2}
        fused = await call("hybrid_search_reports", search_args)
        assert fused["mode"] == "hybrid"
        hit = fused["results"][0]
        assert hit["doc_id"] == "report_1" and hit["page"] == 2
        assert set(hit["retrieval_sources"]) == {"bm25", "dense", "bge_sparse"}
        fts = await call("hybrid_search_reports", search_args, lexical)
        assert fts["mode"] == "lexical_only"
        assert fts["results"][0]["retrieval_sources"] == ["bm25"]
        assert hit["score"] > fts["results"][0]["score"]
        claim = await call("verify_claims", {"answer_text": text, "evidence": [{"doc_id": "report_1", "page": 2}]})
        assert claim["claims"][0]["evidence_ref"]["doc_id"] == "report_1"

    try:
        asyncio.run(exercise())
    finally:
        registry.close()
        lexical.close()
        alias_only.close()
    assert {path: path.read_bytes() for path in db_files} == before
    index = corpus / "unified_store/finarag_unified.faiss"
    index.unlink()
    index.symlink_to(corpus / "finarag_facts.sqlite")
    with pytest.raises(StorageSecurityError):
        build_financial_tool_registry(corpus, policy=default_policy)
