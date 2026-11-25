"""Verify P6 against the read-only, offline 183-report corpus."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sqlite3
import sys
from collections import Counter
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.graph.builder import build_finarag_graph
from src.protocols.tool_registry import _corpus_paths
from src.query_plan import QueryPlan
from src.retrieval_filters import RetrievalFilters
from src.storage.base import StorageError
from src.storage.sqlite_doc_store import SqliteDocumentRepository

BANK = "600000_2024_20250329"
TRADE = "600007_2024_20250401"
BANK_NAME = "浦发银行"
TRADE_NAME = "中国国贸"
SHAREHOLDER = "中国国际贸易中心有限公司"
_STORES = (
    ("docs", "documents", "doc_id", None),
    ("facts", "reports", "doc_id", None),
    ("toc", "toc_trees", "doc_id", None),
    ("facts", "financial_facts", "doc_id", 722653),
    ("kg", "kg_edges", "evidence_doc_id", 15453),
    ("hybrid", "chunks_meta", "doc_id", 297287),
)
_CASES = {
    "cash_flow_delta": (
        f"解释{BANK_NAME}2024年经营活动产生的现金流量净额变化及其关联方披露",
        {BANK_NAME: BANK},
    ),
    "cross_report_cash_flow": (
        f"对比{BANK_NAME}和{TRADE_NAME}的2023年和2024年合并经营活动产生的现金流量净额",
        {BANK_NAME: BANK, TRADE_NAME: TRADE},
    ),
    "report_relationship_disclosure": (
        f"说明{TRADE_NAME}2024年的控股股东{SHAREHOLDER}持股披露及经营活动产生的现金流量净额",
        {TRADE_NAME: TRADE},
    ),
    "missing_period": (
        f"解释{TRADE_NAME}2099年经营活动产生的现金流量净额变化",
        {TRADE_NAME: TRADE},
    ),
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _readonly(path):
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return closing(connection)


def _counts(paths, expected_reports):
    expected_ids = None
    counts = {}
    for key, table, doc_column, expected_rows in _STORES:
        with _readonly(paths[key]) as conn:
            counts[table], = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            ids = {doc for (doc,) in conn.execute(f"SELECT DISTINCT {doc_column} FROM {table}")}
        _require(len(ids) == expected_reports and None not in ids, f"Unexpected document count in {table}")
        if expected_ids is None:
            expected_ids = ids
        _require(ids == expected_ids, f"Document identities differ in {table}")
        _require(counts[table] == (expected_rows if expected_rows is not None else expected_reports),
                 f"Unexpected row count in {table}")
    with paths["faiss"].open("rb") as index:
        _require(bool(index.read(4)), "Empty FAISS artifact")
    return counts


def _validate_sources(paths, repo):
    with _readonly(paths["facts"]) as conn:
        rows = conn.execute(
            """SELECT fact_id, period, source_page, col_header, scope_type, normalized_cny
               FROM financial_facts WHERE doc_id = ? AND metric_atom_id = ? AND source_page = ?
                 AND period IN (?, ?) AND normalized_cny IS NOT NULL ORDER BY fact_id""",
            (TRADE, "MET-003", 93, "2023", "2024"),
        ).fetchall()
    page = (repo.get_page(TRADE, 93) or {}).get("text", "").replace(" ", "")
    oracles = {
        ("2023", "2023年度合并"): (1918660921, "consolidated"),
        ("2024", "2024年度合并"): (1796295378, "consolidated"),
        ("2023", "2023年度公司"): (1906603225, "parent"),
        ("2024", "2024年度公司"): (1753270426, "parent"),
    }
    _require(len(rows) == len(oracles), "Unexpected 中国国贸 page 93 source rows")
    conflicts = []
    for fact_id, period, number, column, scope, amount in rows:
        compact_column = column.replace(" ", "")
        reported_amount, reported_scope = oracles[period, compact_column]
        _require(scope == "consolidated" and amount == reported_amount
                 and compact_column in page and f"{reported_amount:,}" in page, "Page 93 source oracle differs")
        if scope != reported_scope:
            conflicts.append({"fact_id": fact_id, "page": number, "value_period": period, "column": column,
                              "value_cny": reported_amount, "sqlite_scope": scope, "report_scope": reported_scope})
    with _readonly(paths["kg"]) as conn:
        edge = conn.execute(
            """SELECT edge_id FROM kg_edges WHERE head_entity = ? AND relation_type = ?
               AND tail_entity = ? AND evidence_doc_id = ? AND evidence_page = ?""",
            (SHAREHOLDER, "MAJOR_SHAREHOLDER", TRADE_NAME, TRADE, 70),
        ).fetchone()
    page = "".join((repo.get_page(TRADE, 70) or {}).get("text", "").split())
    _require(edge and "前十名股东持股情况" in page and SHAREHOLDER + "|其他|812,360,241|80.65%" in page,
             "The KG edge lacks the expected page 70 disclosure")
    return conflicts, edge[0]


class _ScopedProcessor:
    """Provide prepared report scope for fixed cases while exercising full graph routing."""

    use_vector_dbs = False

    def __init__(self, docs, repository):
        self.docs = docs
        self.report_catalog = self
        self.reports = {}
        for company, doc in docs.items():
            meta = repository.get_metainfo(doc) or {}
            _require(meta.get("company_name") == company and meta.get("report_year") == 2024,
                     "Case report identity differs")
            self.reports[doc] = SimpleNamespace(sha1=doc, company_name=company)

    def get_report_by_doc_id(self, doc):
        return self.reports.get(doc)

    def route_question(self, question, schema):
        companies = list(self.docs)
        comparative = len(companies) > 1
        company = None if comparative else companies[0]
        mode = "comparative_explicit" if comparative else "explicit_company"
        ids = list(self.docs.values())
        plan = QueryPlan(
            question, question, [question],
            RetrievalFilters(company_name=company, year=2024, candidate_doc_ids=ids),
            mode, "text", mentioned_companies=companies,
        )
        return {
            "query_plan": plan, "company_name": company, "companies": companies, "is_comparative": comparative,
            "route_info": {"route_mode": mode, "candidate_doc_ids": ids,
                           "selected_report": {"sha1": self.docs[company]} if company else None},
        }

    def _selected_doc_id(self, company, route_info, query_plan):
        return self.docs.get(company)

    def _get_rule_answer(self, *args, **kwargs):
        return None

    _answer_structured_revenue_comparison = _get_rule_answer
    _answer_structured_revenue_threshold = _get_rule_answer
    _answer_metric_definition_question = _get_rule_answer
    _answer_metadata_count_question = _get_rule_answer
    _answer_metadata_names_question = _get_rule_answer
    _answer_metadata_membership_question = _get_rule_answer
    _answer_legal_representative_question = _get_rule_answer
    _answer_cash_dividend_mention_question = _get_rule_answer


def _resolved(ledger, scope="consolidated"):
    entries = {entry["evidence_id"]: entry for entry in ledger["entries"]}
    return {
        (result["key"]["entity_id"], result["key"]["metric_atom"], result["key"]["period"]): entries[result["winner_id"]]
        for result in ledger["resolutions"]
        if result["winner_id"] and result["key"]["scope"] == scope and result["key"]["basis"] == "reported"
    }


def _run_case(name, repo, conflicts, edge_id):
    question, docs = _CASES[name]
    graph = build_finarag_graph(_ScopedProcessor(docs, repo))
    state = graph.invoke({"question": question, "schema": "long_text"})
    ledger = state["evidence_ledger"]
    answer = state["final_answer_dict"]
    status = state["deep_research_status"]
    _require(state["route_target"] == "deep_research"
             and sum(event.get("route_target") == "deep_research" for event in state["stage_events"]) == 1,
             f"{name} missed the enabled route")
    citations = []
    for citation in answer["citations"]:
        doc, page = citation["doc_id"], citation["page"]
        _require(doc in docs.values() and repo.get_page(doc, page) and citation["report_year"] == 2024,
                 f"{name} emitted an unsupported citation")
        citations.append({key: citation[key] for key in ("doc_id", "page", "report_year", "value_period", "evidence_id")})
    tasks = {task["task_id"]: task for task in state["deep_research_plan"]["tasks"]}
    outcomes = status["outcomes"]
    _require(all(outcome["status"] not in {"failed", "skipped_budget", "timed_out"} for outcome in outcomes),
             f"{name} encountered missing or timed-out tool execution")
    for outcome in outcomes:
        task = tasks[outcome["task_id"]]
        _require(set(task["doc_ids"]).issubset(docs.values())
                 and (task["kind"] != "hybrid_rag" or outcome["retrieval_mode"] == "lexical_only"
                      or outcome["status"] == "skipped_sufficient"), f"{name} requested an unexpected retrieval backend")
    resolved = _resolved(ledger)
    if name == "cash_flow_delta":
        expected = {("600000", "MET-003", "2023"): (388397000000, 19),
                    ("600000", "MET-003", "2024"): (-333654000000, 19),
                    ("600000", "MET-003.delta", "2023–2024"): (-722051000000, None)}
    elif name == "cross_report_cash_flow":
        expected = {("600000", "MET-003", "2023"): (388397000000, 19),
                    ("600000", "MET-003", "2024"): (-333654000000, 19),
                    ("600007", "MET-003", "2023"): (1918660921, 11),
                    ("600007", "MET-003", "2024"): (1796295378, 11)}
    else:
        expected = {}
    observed = []
    cited_ids = {citation["evidence_id"] for citation in citations}
    for key, (amount, page) in expected.items():
        entry = resolved.get(key)
        _require(entry and Decimal(entry["normalized_value"]) == amount and entry["page"] == page,
                 f"{name} value or page oracle differs for {key}")
        _require(str(amount) in answer["final_answer"], f"{name} missing memo value")
        _require(set(entry["inputs"] or [entry["evidence_id"]]).issubset(cited_ids),
                 f"{name} numeric finding is missing its source citations")
        observed.append({"entity": key[0], "metric": key[1], "value_period": key[2],
                         "expected_cny": amount, "observed_cny": int(Decimal(entry["normalized_value"])),
                         "page": page})
    extra = {}
    if name == "cross_report_cash_flow":
        mismatches = {note["fact_id"] for note in ledger["notes"] if note["reason"] == "scope_label_mismatch"}
        _require(mismatches == {row["fact_id"] for row in conflicts}, "Page 93 mismatch missing from ledger")
        extra["scope_discrepancies"] = conflicts
        extra["unresolved_required_conflicts"] = [
            result["key"] for result in ledger["resolutions"] if result["reason"] == "same_priority_conflict"
        ]
    if name == "report_relationship_disclosure":
        admitted = [entry for entry in ledger["entries"] if entry["source_class"] == "kg" and
                    entry["label"] == f"{SHAREHOLDER} → MAJOR_SHAREHOLDER → {TRADE_NAME}"]
        _require(admitted and admitted[0]["page"] == 70 and admitted[0]["evidence_id"] in cited_ids
                 and SHAREHOLDER in answer["final_answer"], "Corroborated shareholder edge missing from memo")
        extra["relationship"] = {"edge_id": edge_id, "head": SHAREHOLDER, "type": "MAJOR_SHAREHOLDER",
                                 "tail": TRADE_NAME, "page": 70, "shares": 812360241, "percentage": "80.65%"}
    if name == "missing_period":
        _require(answer["final_answer"] == "N/A" and not citations and not resolved
                 and any("2099" in gap for gap in answer["evidence_gaps"]), "Missing period was substituted")
        extra["refused"] = True
        extra["requested_period"] = "2099"

    return {
        "route": state["route_target"], "waves": status["wave_count"], "stop_reason": status["stop_reason"],
        "tool_statuses": dict(sorted(Counter(
            f"{tasks[outcome['task_id']]['kind']}:{outcome['status']}" for outcome in outcomes
        ).items())),
        "retrieval_modes": sorted({outcome["retrieval_mode"] for outcome in outcomes if outcome["retrieval_mode"]}),
        "values": observed, "citations": citations, **extra,
    }, {
        "plan": state["deep_research_plan"], "ledger": ledger, "answer": answer,
        "outcomes": [{key: value for key, value in outcome.items() if key != "ms"} for outcome in outcomes],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--expected-reports", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        _require(os.getenv("FINARAG_DEEP_RESEARCH_ENABLED", "0").strip().lower() in {"1", "true", "yes"},
                 "Enable FINARAG_DEEP_RESEARCH_ENABLED for the spot check")
        root, paths = _corpus_paths(args.corpus_root)
        _require(all(paths.get(key) is not None for key in ("docs", "facts", "toc", "kg", "hybrid", "faiss")),
                 "A required corpus artifact is missing")
        counts = _counts(paths, args.expected_reports)
        network_calls = []

        def reject_network(*_args, **_kwargs):
            network_calls.append(True)
            raise OSError("Network unavailable in the offline spot check")

        with (closing(SqliteDocumentRepository(paths["docs"], allowed_base_dir=root)) as repo,
              patch.dict(os.environ, {"FINARAG_PROTOCOL_CORPUS_ROOT": str(root), "FINARAG_CLAIM_VERIFY_ENABLED": "0"}),
              patch.object(socket.socket, "connect", reject_network),
              patch.object(socket.socket, "connect_ex", reject_network),
              patch.object(socket, "getaddrinfo", reject_network)):
            conflicts, edge_id = _validate_sources(paths, repo)
            first = {name: _run_case(name, repo, conflicts, edge_id) for name in _CASES}
            second = {name: _run_case(name, repo, conflicts, edge_id) for name in _CASES}
        _require(first == second, "Stable case projections differed between consecutive runs")
        _require(not network_calls, "The offline spot check attempted a network request")
        print(json.dumps({"reports": counts["documents"], "cross_store_ids_match": True, "counts": counts,
                          "enabled_route": "deep_research", "consecutive_runs_match": True,
                          "cases": {name: summary for name, (summary, _) in first.items()}},
                         ensure_ascii=False, allow_nan=False, sort_keys=True))
        return 0
    except (AssertionError, KeyError, OSError, ValueError, StorageError, sqlite3.Error) as exc:
        print(f"P6 spot check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
