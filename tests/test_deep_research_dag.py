import asyncio
import json
import subprocess
import sys
import threading
import time
from dataclasses import replace
from decimal import Decimal

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from src.graph.builder import build_finarag_graph, run_finarag_graph
from src.graph.state import FinaRAGRunContext
from src.protocols.tool_registry import MCPToolRegistry, ToolPrincipal, ToolResult, default_policy
from src.query_plan import QueryPlan
from src.retrieval_filters import RetrievalFilters

DOC = "600000_2024_20250329"
COMPANY = "浦发银行"
QUESTION = "解释浦发银行2024年经营活动现金流量净额的变化及其关联方披露"
PRINCIPAL = ToolPrincipal("test-researcher", frozenset({"read"}))


def _plan(question=QUESTION):
    return QueryPlan(
        question, question, [question],
        RetrievalFilters(company_name=COMPANY, year=2024, candidate_doc_ids=[DOC]),
        "explicit_company", "text", mentioned_companies=[COMPANY],
    )


def _row(period="2024", value=-333654, **overrides):
    return {
        "doc_id": DOC, "company_name": COMPANY, "scope_type": "consolidated",
        "metric_atom_id": "MET-003", "period": period, "report_year": 2024,
        "currency": "CNY", "source_page": 19, "raw_unit": "百万元",
        "raw_value": value, "table_id": "11", "row_idx": 6,
        "col_header": f"{period}年", "metric_raw": "经营活动产生的现金流量净额",
        "fact_id": f"fixture-{period}", **overrides,
    }


def _registry(*, overlap=None, conflict=False, missing=False, calls=None):
    def traced(name, handler):
        def run(args, deadline):
            if calls is not None:
                calls.append(name)
            return handler(args, deadline)
        return run

    def facts(args, deadline):
        if overlap:
            overlap.wait(timeout=2)
        period = args["period"]
        if period == "2023":
            return {"rows": [] if missing else [_row(period, 388397)]}
        rows = [_row()]
        if conflict:
            rows.append(_row(value=-333653, fact_id="conflicting"))
        return {"rows": rows}

    def calculate(args, deadline):
        values = {v["name"]: v["value"] for v in args["variables"]}
        assert values == {"current": -333654000000, "prior": 388397000000}
        return {"value": values["current"] - values["prior"], "unit": "元",
                "expression": args["expression"]}

    def toc(args, deadline):
        return {"pages": [{"doc_id": DOC, "page": 40, "excerpt": "本公司披露关联方交易。"}]}

    def kg(args, deadline):
        return {"paths": [], "evidence_pages": []}

    def search(args, deadline):
        return {"mode": "lexical_only", "results": []}

    return MCPToolRegistry(handlers={
        name: traced(name, handler) for name, handler in {
            "query_financial_facts_sql": facts, "pot_calculate": calculate,
            "navigate_toc_sections": toc, "query_financial_kg": kg,
            "hybrid_search_reports": search,
        }.items()
    }, policy=default_policy)


def _execute(planner, dag, registry, ledger, principal=PRINCIPAL, budget=5):
    try:
        return asyncio.run(planner.execute(
            dag, registry=registry, principal=principal, ledger=ledger,
            deadline=time.monotonic() + budget,
        ))
    finally:
        registry.close()


def test_typed_dag_dependencies_and_parallel_waves():
    from src.deep_research import EvidenceLedger, ResearchDAG, ResearchPlanner

    planner = ResearchPlanner()
    dag = planner.plan(QUESTION, _plan(), doc_ids={COMPANY: [DOC]})
    assert {task.kind for task in dag.tasks} == {
        "sql_fact", "kg_multihop", "toc_nav", "hybrid_rag", "pot_calc",
    }
    calls = []
    ledger = EvidenceLedger()
    run = _execute(planner, dag, _registry(overlap=threading.Barrier(2), calls=calls), ledger)
    outcomes = {outcome.task_id: outcome for outcome in run.outcomes}
    calc = next(task for task in dag.tasks if task.kind == "pot_calc")
    assert outcomes[calc.task_id].status == "ok"
    assert all(outcomes[parent].wave < outcomes[calc.task_id].wave for parent in calc.depends_on)
    assert calls.index("pot_calculate") > max(i for i, name in enumerate(calls) if name == "query_financial_facts_sql")
    assert [o.task_id for o in run.outcomes if o.wave == 1] == sorted(o.task_id for o in run.outcomes if o.wave == 1)
    assert json.dumps(ledger.to_dict(), allow_nan=False)
    assert any(e.normalized_value == Decimal("-722051000000") and len(e.inputs) == 2 for e in ledger.entries)

    first = dag.tasks[0]
    with pytest.raises(ValueError):
        ResearchDAG((first, first))
    with pytest.raises(ValueError):
        ResearchDAG((replace(first, depends_on=(calc.task_id,)), *dag.tasks[1:]))
    with pytest.raises(ValueError):
        replace(first, arguments={**first.arguments, "untrusted_field": True})
    with pytest.raises(ValueError):
        planner.plan(QUESTION, _plan(), doc_ids=[])
    assert all(o.status == "skipped_budget" for o in _execute(
        planner, dag, _registry(), EvidenceLedger(), budget=-1,
    ).outcomes)
    denied = _execute(planner, dag, _registry(), EvidenceLedger(), ToolPrincipal("guest", frozenset()))
    assert any(o.code == "forbidden" for o in denied.outcomes)
    missing = _execute(planner, dag, _registry(missing=True), EvidenceLedger())
    assert next(o for o in missing.outcomes if o.task_id == calc.task_id).status == "blocked"


def test_evidence_caliber_precedence_and_conflicts():
    from src.deep_research import EvidenceLedger, ResearchPlanner

    task = next(t for t in ResearchPlanner().plan(QUESTION, _plan(), doc_ids=[DOC]).tasks
                if t.kind == "sql_fact" and t.arguments["period"] == "2024")
    key, = task.required_keys
    ledger = EvidenceLedger()
    ledger.ingest(task, ToolResult({"rows": [_row(value=10)]}))
    ledger.ingest(task, ToolResult({"evidence": [
        _row(source_class="table", raw_unit="亿元", value="0.1", col_idx=1),
        _row(source_class="kg", value=12, edge_id="edge-1"),
        _row(source_class="narrative", value=13, chunk_id="chunk-1"),
        _row(source_class="table", value=99, col_idx=1, scope_type="parent"),
        _row(source_class="table", value=99, col_idx=1, measurement_basis="adjusted"),
        _row(source_class="table", value=99, col_idx=1, measurement_basis="restated"),
        _row(source_class="table", value=99, col_idx=1, raw_unit="元/股"),
        _row(source_class="table", value=99, col_idx=1, metric_atom_id="MET-001"),
        _row(source_class="table", value=99, col_idx=1, company_name="浦发"),
        _row(source_class="table", value=99, col_idx=1, source_page=None),
    ]}))
    resolution = ledger.adjudicate()[key]
    assert ledger.entry(resolution.winner_id).source_class == "sql"
    assert ledger.entry(resolution.winner_id).normalized_value == Decimal("10000000")
    assert resolution.reason == "higher_priority_source"
    assert len(resolution.disagreements) == 2
    assert len(ledger.adjudicate()) == 6
    assert {"incompatible_caliber", "out_of_scope", "missing_provenance"} <= {
        item["reason"] for item in ledger.to_dict()["notes"]
    }
    ledger.ingest(task, ToolResult({"rows": [_row(value=11, fact_id="conflicting")]}))
    assert ledger.adjudicate()[key].reason == "same_priority_conflict"
    assert not ledger.coverage([key])


def test_sufficiency_preserves_required_calculations():
    from src.deep_research import EvidenceLedger, ResearchPlanner

    planner = ResearchPlanner()
    dag = planner.plan(QUESTION, _plan(), doc_ids=[DOC])
    calls = []
    run = _execute(planner, dag, _registry(calls=calls), EvidenceLedger())
    assert "pot_calculate" in calls
    assert "hybrid_search_reports" not in calls
    assert run.stop_reason == "sufficient"
    calls.clear()
    run = _execute(planner, dag, _registry(conflict=True, calls=calls), EvidenceLedger())
    assert "pot_calculate" not in calls
    assert "hybrid_search_reports" in calls
    assert run.stop_reason == "incomplete"


def test_synthesis_and_opt_in_graph_route(monkeypatch):
    from src.claim_verifier import ClaimAuditItem, ClaimAuditReport
    from src.deep_research import EvidenceLedger, ResearchPlanner, ResearchSynthesizer
    from src.graph.legacy_runner import run_legacy_question
    from src.graph.nodes.plan_route import make_plan_route_node

    class Processor:
        use_vector_dbs = True
        report_catalog = None

        def route_question(self, question, schema):
            return {"query_plan": _plan(question), "company_name": COMPANY, "companies": [COMPANY],
                    "route_info": {"selected_report": {"sha1": DOC}}}

        def _answer_metric_definition_question(self, **kwargs):
            return None

        def _answer_structured_revenue_threshold(self, **kwargs):
            return None

        _answer_metadata_count_question = _answer_metric_definition_question
        _answer_metadata_names_question = _answer_metric_definition_question
        _answer_metadata_membership_question = _answer_metric_definition_question
        _answer_legal_representative_question = _answer_metric_definition_question
        _answer_cash_dividend_mention_question = _answer_metric_definition_question

        def _selected_doc_id(self, *args):
            return DOC

    class Verifier:
        def verify(self, text, *, citations, retrieval_results, **scope):
            assert citations and all(r["pdf_sha1"] == DOC for r in retrieval_results)
            claim = ClaimAuditItem("-333654000000", "contradicted")
            return ClaimAuditReport([claim], 1, 0, 0, 1, 1.0, 1.0, 0.0)

    processor = Processor()
    processor.deep_research_registry_provider = _registry
    monkeypatch.setenv("FINARAG_DEEP_RESEARCH_ENABLED", "1")
    monkeypatch.setenv("FINARAG_CLAIM_VERIFY_ENABLED", "0")
    answer = run_finarag_graph(processor, QUESTION, "long_text")
    assert "-722051000000" in answer["final_answer"]
    assert {citation["doc_id"] for citation in answer["citations"]} == {DOC}
    assert sum(event.get("route_target") == "deep_research" for event in answer["graph_stage_events"]) == 1
    state = {"question": QUESTION, "schema": "long_text"}
    assert make_plan_route_node(processor)(state)["route_target"] == "single_rag"
    monkeypatch.setenv("FINARAG_DEEP_RESEARCH_ENABLED", "0")
    assert "deep_research" not in build_finarag_graph(processor).get_graph().nodes
    monkeypatch.setenv("FINARAG_DEEP_RESEARCH_ENABLED", "1")
    assert make_plan_route_node(processor, deep_research_enabled=True)({
        **state, "schema": "number",
    })["route_target"] == "single_rag"
    with monkeypatch.context() as context:
        context.setattr(processor, "_answer_metric_definition_question", lambda **kw: {"final_answer": "rule"})
        assert run_finarag_graph(processor, QUESTION, "long_text")["final_answer"] == "rule"
    with monkeypatch.context() as context:
        context.setattr(processor, "get_answer_for_company", lambda **kw: {"final_answer": "override"}, raising=False)
        assert run_finarag_graph(processor, QUESTION, "long_text")["final_answer"] == "override"
    with monkeypatch.context() as context:
        context.setattr("src.graph.legacy_runner._run_single_rag_loop", lambda proc, state: state["route_target"])
        assert run_legacy_question(processor, question=QUESTION, schema="long_text") == "single_rag"

    planner = ResearchPlanner()
    dag = planner.plan(QUESTION, _plan(), doc_ids=[DOC])
    ledger = EvidenceLedger()
    run = _execute(planner, dag, _registry(), ledger)
    assert ResearchSynthesizer().synthesize(
        QUESTION, run, EvidenceLedger(), query_plan=_plan(), route_info={},
    )["final_answer"] == "N/A"
    monkeypatch.setenv("FINARAG_CLAIM_VERIFY_ENABLED", "1")
    processor.deep_research_claim_verifier = Verifier()
    audited = ResearchSynthesizer(claim_verifier=Verifier()).synthesize(
        QUESTION, run, ledger, query_plan=_plan(), route_info={},
    )
    assert audited["claim_audit"]["contradicted_count"] == 1
    assert "-333654000000" not in audited["final_answer"]
    assert "-722051000000" not in audited["final_answer"]
    graph = build_finarag_graph(processor, InMemorySaver())
    config = {"configurable": {"thread_id": "research-fixture"}}
    pending = run_finarag_graph(
        processor, QUESTION, "long_text", config=config, graph=graph,
        context=FinaRAGRunContext(turn_id="first", approval_enabled=True),
    )
    assert pending["approval_status"] == "pending"
    assert pending["approval_reasons"] == ["claim_contradicted"]
    for channel in ("deep_research_plan", "evidence_ledger", "deep_research_status"):
        assert json.dumps(graph.get_state(config).values[channel], allow_nan=False)


def test_lazy_import_and_sync_bridge(monkeypatch):
    subprocess.run(
        [sys.executable, "-c",
         "import sys; import src.deep_research as p; assert 'src.deep_research.planner' not in sys.modules; "
         "assert p.ResearchPlanner is p.__dict__['ResearchPlanner']"],
        check=True,
    )
    from src.deep_research.planner import make_deep_research_node

    async def invoke():
        node = make_deep_research_node(object(), registry_provider=_registry)
        return node({"question": QUESTION, "query_plan": _plan(),
                     "route_info": {"selected_report": {"sha1": DOC}}, "company_name": COMPANY})

    monkeypatch.setenv("FINARAG_CLAIM_VERIFY_ENABLED", "0")
    assert "-722051000000" in asyncio.run(invoke())["final_answer_dict"]["final_answer"]
