"""Tests for P3 Knowledge Flywheel, Targeted Financial KG (LightRAG+PPR), and Claim-Level Verifier."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.analyze_adaptive_trace import (
    alias_expansion_suggestions,
    load_trace,
    per_atom_stats,
    render_report,
)
from src.adaptive_retrieval import TRIGGER_CLAIM_CONTRADICTED, detect_post_validate_gap
from src.claim_verifier import ClaimVerifier, extract_atomic_claims, extract_numeric_mentions
from src.graph.nodes.ground_generate import make_ground_generate_node
from src.graph.nodes.reflect_supplement import decide_reflection_edge
from src.graph.nodes.retrieve_rerank import make_retrieve_rerank_node
from src.knowledge_graph.kg_builder import (
    FinancialKGBuilder,
    FinancialKGStore,
    canonicalize_entity_name,
)
from src.knowledge_graph.lightrag_retriever import (
    LightRAGRetriever,
    fuse_kg_and_hybrid_results,
    should_trigger_kg_retrieval,
)
from src.knowledge_loop import cluster_candidate_atoms, normalize_trace_entry
from src.metric_atoms import get_default_store
from src.query_plan import QueryPlan
from src.query_rewrite import _TOPIC_PATTERNS
from src.retrieval_filters import RetrievalFilters
from src.service.telemetry import TelemetryManager
from src.storage.base import StorageSecurityError
from src.structured.sql_store import FinancialSQLStore
from src.topic_flag_contract import coverage_report


class P3KnowledgeAndClaimsTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_p3_a_topic_contract_and_telemetry_flywheel(self):
        # 1) Confirm 17/17 topic_flag contract coverage remains intact
        cov = coverage_report(_TOPIC_PATTERNS)
        self.assertEqual(cov["total_flags"], 17)
        self.assertEqual(cov["covered_flags"], 17)
        self.assertEqual(cov["coverage_rate"], 1.0)

        atoms_yaml = Path("config/metric_atoms.yaml")
        before_sha = hashlib.sha256(atoms_yaml.read_bytes()).hexdigest()

        # 2) TelemetryManager emits low-confidence + claim_contradicted traces to adaptive_trace.jsonl
        tm = TelemetryManager(trace_dir=self.root)
        for idx in range(3):
            tm.record_request(
                trace_id=f"tr-new-{idx}",
                question="示例公司2024年拨备前利润是多少？",
                schema="number",
                answer_dict={
                    "final_answer": "N/A",
                    "confidence": "low",
                    "validation_flags": ["metric_evidence_mismatch"],
                    "route_info": {"route_mode": "explicit_company"},
                    "query_plan": {
                        "metric": "拨备前利润",
                        "normalized_query": "拨备前利润",
                        "search_queries": ["拨备前利润"],
                    },
                },
                spans=[],
                latency_ms=12.0,
            )
            tm.record_request(
                trace_id=f"tr-con-{idx}",
                question="示例公司2024年营业收入增长情况？",
                schema="text",
                answer_dict={
                    "final_answer": "同比增长15%",
                    "confidence": "low",
                    "validation_flags": ["claim_contradicted"],
                    "claim_audit": {"contradicted_count": 1},
                    "route_info": {"route_mode": "explicit_company"},
                    "query_plan": {
                        "metric": "营业收入",
                        "metric_atom": {"atom_id": "MET-004", "standard_name": "营业收入"},
                        "search_queries": ["营业收入 2024"],
                    },
                },
                spans=[],
                latency_ms=15.0,
            )
        # Generic blocked alias term must be rejected by validate_atom_drafts
        for idx in range(3):
            tm.record_request(
                trace_id=f"tr-bad-{idx}",
                question="收入是多少？",
                schema="number",
                answer_dict={
                    "final_answer": "N/A",
                    "confidence": "low",
                    "validation_flags": ["metric_evidence_mismatch"],
                    "query_plan": {"metric": "收入", "search_queries": ["收入"]},
                },
                spans=[],
                latency_ms=8.0,
            )

        trace_path = self.root / "adaptive_trace.jsonl"
        entries, skipped = load_trace(trace_path)
        self.assertEqual(len(entries), 9)
        self.assertEqual(skipped, 0)

        norm0 = normalize_trace_entry(entries[0])
        self.assertEqual(norm0["source"], "online_telemetry")
        self.assertEqual(norm0["trigger"], "metric_evidence_mismatch")

        store = get_default_store()
        rows = per_atom_stats(entries)
        self.assertTrue(any(r["atom_id"] == "MET-004" for r in rows))

        clustered = cluster_candidate_atoms(entries, store, min_samples=3)
        rules = {(s["rule"], s["term"]) for s in clustered["suggestions"]}
        self.assertIn(("R4_candidate_new_atom", "拨备前利润"), rules)
        self.assertIn(("R5_claim_contradiction_hotspot", "MET-004"), rules)
        # Rejected generic term "收入" must NOT enter validated_drafts
        self.assertTrue(clustered["draft_errors"].get("收入"))
        self.assertNotIn("收入", [d["standard_name"] for d in clustered["validated_drafts"]])
        self.assertIn("standard_name: 拨备前利润", clustered["draft_yaml"])

        report_md = render_report(
            rows,
            alias_expansion_suggestions(entries, store),
            skipped,
            candidate_suggestions=clustered["suggestions"],
        )
        self.assertIn("R4_candidate_new_atom", report_md)
        self.assertIn("R5_claim_contradiction_hotspot", report_md)

        # CLI end-to-end check
        out_md = self.root / "report.md"
        out_yaml = self.root / "drafts.yaml"
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_adaptive_trace.py",
                "--trace",
                str(trace_path),
                "--output",
                str(out_md),
                "--draft-output",
                str(out_yaml),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("条建议", proc.stdout)
        self.assertTrue(out_yaml.exists())
        self.assertIn("拨备前利润", out_yaml.read_text(encoding="utf-8"))
        # Builder never self-approves / never mutates metric_atoms.yaml
        self.assertEqual(hashlib.sha256(atoms_yaml.read_bytes()).hexdigest(), before_sha)

    def test_p3_b_financial_kg_isolation_security_and_lightrag_ppr(self):
        # Entity canonicalization & anonymized counterparty isolation
        canon_ocr, etype = canonicalize_entity_name("上海人寿保险股份有限 公司")
        self.assertEqual(canon_ocr, "上海人寿保险股份有限公司")
        self.assertEqual(etype, "organization")

        anon1, atype1 = canonicalize_entity_name("客户一", company_scope="示例银行")
        anon2, atype2 = canonicalize_entity_name("客户一", company_scope="华东能源")
        self.assertEqual(atype1, "anonymized_counterparty")
        self.assertEqual(atype2, "anonymized_counterparty")
        self.assertNotEqual(anon1, anon2)

        pages_bank = [
            {
                "page": 38,
                "text": (
                    "# 四、控股股东及实际控制人情况\n"
                    "公司控股股东为国资投资集团有限公司。\n"
                    "## 主要子公司\n"
                    "上海国际信托有限公司现注册资本50亿元，公司持有其97.33%的股份。\n"
                ),
            },
            {
                "page": 62,
                "text": (
                    "# 五、重大关联交易\n"
                    "| 交易对方 | 交易类型 | 金额 | 占净资产比例 |\n"
                    "|---|---|---|---|\n"
                    "| 上海人寿保险股份有限 公司 | 综合授信 | 人民币25亿元 | 1.2% |\n"
                    "| 合计 | - | 人民币25亿元 | 1.2% |\n"
                    "# 前五名客户\n"
                    "| 客户名称 | 销售额 | 占比 |\n"
                    "|---|---|---|\n"
                    "| 客户一 | 10亿元 | 5.0% |\n"
                ),
            },
        ]
        pages_energy = [
            {
                "page": 45,
                "text": (
                    "# 六、控股股东及实际控制人情况\n"
                    "公司实际控制人为国资投资集团有限公司。\n"
                    "# 前五名供应商\n"
                    "| 供应商名称 | 采购额 | 占比 |\n"
                    "|---|---|---|\n"
                    "| 客户一 | 8亿元 | 6.0% |\n"
                    "| 华能发电设备有限公司 | 12亿元 | 9.0% |\n"
                ),
            }
        ]
        pages_by_doc = {
            "doc-bank": {p["page"]: {**p, "company_name": "示例银行"} for p in pages_bank},
            "doc-energy": {p["page"]: {**p, "company_name": "华东能源"} for p in pages_energy},
        }

        builder = FinancialKGBuilder()
        alias_map = {"国资投资集团有限公司": "国资投资集团"}
        e1, r1 = builder.extract_from_report(
            doc_id="doc-bank", company_name="示例银行", report_year=2024, pages=pages_bank, alias_to_canonical=alias_map
        )
        e2, r2 = builder.extract_from_report(
            doc_id="doc-energy", company_name="华东能源", report_year=2024, pages=pages_energy, alias_to_canonical=alias_map
        )

        kg_db = self.root / "finarag_kg.sqlite"
        stats = FinancialKGStore.write_kg_sqlite(e1 + e2, r1 + r2, kg_db, reports_count=2, allowed_base_dir=self.root)
        self.assertGreaterEqual(stats["edges_count"], 5)
        self.assertGreaterEqual(stats["cross_report_bridge_entities"], 1)

        # Security guards: traversal and sibling prefix bypass rejected
        with self.assertRaises(StorageSecurityError):
            FinancialKGStore(self.root / "../bad_kg.sqlite", allowed_base_dir=self.root)
        sibling = self.root.parent / f"{self.root.name}-evil" / "kg.sqlite"
        with self.assertRaises(StorageSecurityError):
            FinancialKGStore(sibling, allowed_base_dir=self.root)

        store = FinancialKGStore(kg_db, allowed_base_dir=self.root)
        self.addCleanup(store.close)
        qdrant_pts = store.export_qdrant_entity_points()
        self.assertTrue(any(p["payload"]["canonical_name"] == "国资投资集团" for p in qdrant_pts))
        names = {ent.canonical_name for ent in store.list_entities()}
        self.assertNotIn("合计", names)
        self.assertNotIn("客户一", names)
        self.assertIn("示例银行::客户一", names)
        self.assertIn("华东能源::客户一", names)
        self.assertIn("上海人寿保险股份有限公司", names)

        # Verify 2-hop cross-report path via shared shareholder (and NEVER via anonymized 客户一)
        retriever = LightRAGRetriever(store)
        q_cross = "示例银行与华东能源是否存在共同控股股东或实际控制人关系？"
        self.assertTrue(should_trigger_kg_retrieval(None, q_cross))
        ret = retriever.retrieve(
            q_cross,
            company_name="示例银行",
            page_fetcher=lambda did, pno: (pages_by_doc.get(did) or {}).get(int(pno)),
        )
        self.assertIn("华东能源", ret["seed_entities"])
        self.assertTrue(any(p["hops"] == 2 and "国资投资集团" in p["summary"] for p in ret["paths"]))
        self.assertTrue(ret["results"])
        self.assertEqual(ret["results"][0]["chunk_type"], "kg_path_summary")

        # End-to-end retrieve_rerank_node integration: OFF by default, ON with FINARAG_KG_ENABLED=1
        docs_dir = self.root / "documents"
        docs_dir.mkdir(parents=True, exist_ok=True)
        hybrid_hit = {
            "page": 10,
            "text": "财务指标摘要",
            "pdf_sha1": "doc-bank",
            "metadata": {"sha1_name": "doc-bank", "node_type": "child"},
            "retrieval_sources": ["vector"],
            "result_scope": "child",
        }
        fake_proc = SimpleNamespace(
            documents_dir=docs_dir,
            retrieval_debug_top_n=6,
            _build_retriever=lambda: (
                SimpleNamespace(retrieve_by_company_name=lambda **kw: [dict(hybrid_hit)]),
                "vector",
            ),
            _run_retrieval=lambda *a, **kw: [dict(hybrid_hit)],
            _merge_multi_query_results=lambda runs, top_n: [dict(hybrid_hit)],
            _build_reranking_debug_payload=lambda ret, cap: {},
            _build_hyde_debug_payload=lambda retrieval_results, candidate_pool_size_before_rerank: {},
            _load_document_payload=lambda did: {
                "metainfo": {"sha1_name": did, "company_name": "示例银行", "report_year": 2024},
                "content": {"pages": pages_bank if did == "doc-bank" else pages_energy},
            },
        )
        rr_node = make_retrieve_rerank_node(fake_proc)
        plan_rel = QueryPlan(
            original_query="示例银行的主要控股子公司与关联交易对手方有哪些？",
            normalized_query="示例银行 控股子公司 关联交易",
            search_queries=["示例银行 控股子公司 关联交易"],
            filters=RetrievalFilters(company_name="示例银行", year=2024),
            route_mode="explicit_company",
            expected_answer_type="text",
        )
        state_in = {
            "question": plan_rel.original_query,
            "schema": "text",
            "company_name": "示例银行",
            "query_plan": plan_rel,
            "selected_doc_id": "doc-bank",
            "candidate_doc_ids": ["doc-bank"],
        }
        out_off = rr_node(state_in)
        self.assertEqual(out_off["stage_events"][0]["kg_hits"], 0)

        with patch.dict(os.environ, {"FINARAG_KG_ENABLED": "1", "FINARAG_KG_DB": str(kg_db)}):
            out_on = rr_node(state_in)
        self.assertGreater(out_on["stage_events"][0]["kg_hits"], 0)
        self.assertTrue(any("kg_lightrag" in (r.get("retrieval_sources") or []) for r in out_on["retrieval_results"]))
        fused_direct = fuse_kg_and_hybrid_results([hybrid_hit], ret["results"], top_n=5)
        self.assertTrue(any(r.get("chunk_type") == "kg_path_summary" for r in fused_direct))

    def test_p3_c_claim_verifier_numeric_sql_polarity_and_reflection_hook(self):
        # 1) Sentence splitting must preserve decimal numbers like 3.64% and 1,707.48
        prose = "2024年浦发银行营业收入1,707.48亿元。核心资本充足率增长至13.2%。净利润同比增长5.4%。"
        claims = extract_atomic_claims(prose)
        self.assertEqual(len(claims), 3)
        mentions = extract_numeric_mentions(claims[0])
        self.assertEqual([m["kind"] for m in mentions], ["year", "money_cny"])

        # 2) Build temp FinancialSQLStore with consolidated 2024 revenue fact
        facts_db = self.root / "finarag_facts.sqlite"
        conn = sqlite3.connect(str(facts_db))
        FinancialSQLStore.create_schema(conn)
        FinancialSQLStore.create_indexes(conn)
        conn.execute(
            "INSERT INTO reports VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("doc-spdb", "doc-spdb", "浦发银行", "600000", "银行", 2024, "CNY"),
        )
        conn.execute(
            "INSERT INTO financial_facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("f1", "doc-spdb", "income_statement", "consolidated", "营业收入", "MET-004", "2024", 170748.0, "百万元", 170748e6, 14, "t1", 1, "2024年"),
        )
        conn.commit()
        conn.close()

        sql_store = FinancialSQLStore(facts_db, allowed_base_dir=self.root)
        verifier = ClaimVerifier(sql_store=sql_store)

        ev_results = [
            {
                "page": 14,
                "pdf_sha1": "doc-spdb",
                "text": "主要财务指标\n单位：人民币百万元\n2024年营业收入170,748，归母净利润同比下降2.1%。",
                "metadata": {"sha1_name": "doc-spdb", "company_name": "浦发银行", "report_year": 2024},
            }
        ]

        # Supported cross-unit conversion: 1,707.48亿元 == 170,748百万元 (and page citations do not false-trigger)
        ok_audit = verifier.verify(
            "2024年浦发银行营业收入达到1,707.48亿元。数据来源见年报第14页主要财务指标。",
            retrieval_results=ev_results,
            company_name="浦发银行",
            report_year=2024,
        )
        self.assertEqual(ok_audit.claims[0].status, "supported")
        self.assertEqual(ok_audit.contradicted_count, 0)
        self.assertLess(ok_audit.latency_ms, 400.0)

        # Contradicted: wrong number vs SQL fact + flipped trend polarity
        bad_audit = verifier.verify(
            "2024年浦发银行营业收入达到2,500亿元。归母净利润同比增长2.1%。",
            retrieval_results=ev_results,
            company_name="浦发银行",
            report_year=2024,
        )
        self.assertEqual(bad_audit.contradicted_count, 2)
        self.assertEqual(bad_audit.claims[0].status, "contradicted")
        self.assertEqual(bad_audit.claims[1].status, "contradicted")

        # 3) Custom NLI scorer hook on qualitative claim
        nli_verifier = ClaimVerifier(nli_scorer_fn=lambda c, p: 0.91)
        qual_audit = nli_verifier.verify(
            "公司聚焦五大赛道全面深化数字化金融转型升级服务实体经济。",
            retrieval_results=[{"page": 23, "pdf_sha1": "doc-spdb", "text": "五大赛道数智化转型工作情况。"}],
        )
        self.assertEqual(qual_audit.supported_count, 1)

        # 4) End-to-end ground_generate_node + adaptive_retrieval reflection hook
        fake_proc = SimpleNamespace(
            table_grounder=None,
            chart_grounder=None,
            sql_store=sql_store,
            top_n_retrieval=5,
            answering_model="dummy-model",
            answer_temperature=0.0,
            reasoning_debug_enabled=False,
            api_processor=SimpleNamespace(
                get_answer_from_rag_context=lambda **kw: {
                    "final_answer": "2024年浦发银行营业收入为2,500亿元。",
                    "relevant_pages": [14],
                    "confidence": "high",
                    "reasoning_summary": "test",
                },
                response_data={"usage": {"prompt_tokens": 10, "completion_tokens": 10}},
            ),
            _format_retrieval_results=lambda res: "context",
            _build_metric_caliber_hint=lambda qp: "",
            _validate_page_references=lambda pages, res: pages,
            _extract_references=lambda pages, comp, pdf_sha1=None: [{"pdf_sha1": "doc-spdb", "page_index": 13}],
            _table_support_results=lambda t: [],
            _serialize_retrieval_result=lambda r: r,
            _aggregate_retrieval_results_by_report=lambda res: [],
        )
        gg_node = make_ground_generate_node(fake_proc)
        qplan = QueryPlan(
            original_query="浦发银行2024年营业收入与业绩表现如何？",
            normalized_query="浦发银行 营业收入",
            search_queries=["浦发银行 营业收入"],
            filters=RetrievalFilters(company_name="浦发银行", year=2024),
            route_mode="explicit_company",
            expected_answer_type="text",
            metric="营业收入",
        )
        state_in = {
            "question": qplan.original_query,
            "schema": "text",
            "company_name": "浦发银行",
            "query_plan": qplan,
            "route_info": {"selected_report": {"sha1": "doc-spdb"}, "route_mode": "explicit_company"},
            "candidate_doc_ids": ["doc-spdb"],
            "allowed_grounding_doc_ids": ["doc-spdb"],
            "reranking_debug": {
                "reranking_strategy": "single",
                "initial_candidate_pool_size": None,
                "colbert_candidate_pool_size": None,
                "colbert_top_n": None,
                "final_reranking_backend": "none",
            },
            "hyde_debug": {"enabled": False, "triggered": False},
            "retrieval_results": ev_results,
        }

        with patch.dict(os.environ, {"FINARAG_CLAIM_VERIFY_ENABLED": "1"}):
            gg_out = gg_node(state_in)

        self.assertIn("claim_audit", gg_out["final_answer_dict"])
        self.assertIn("claim_contradicted", gg_out["validated_flags"])
        self.assertEqual(gg_out["final_answer_dict"]["confidence"], "low")

        # adaptive_retrieval + decide_reflection_edge must route claim_contradicted to reflect
        gap = detect_post_validate_gap(gg_out["validated_flags"], qplan, ev_results)
        self.assertIsNotNone(gap)
        self.assertEqual(gap.trigger, TRIGGER_CLAIM_CONTRADICTED)
        with patch.dict(os.environ, {"ADAPTIVE_RETRIEVAL_ENABLED": "1"}):
            edge = decide_reflection_edge(
                {
                    "query_plan": qplan,
                    "retrieval_results": ev_results,
                    "validated_flags": gg_out["validated_flags"],
                    "adaptive_rounds_used": 0,
                    "adaptive_seen_queries": [],
                    "retrieval_mode": "hybrid",
                }
            )
        self.assertEqual(edge, "reflect")


if __name__ == "__main__":
    unittest.main()
