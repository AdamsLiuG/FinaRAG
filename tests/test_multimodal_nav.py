"""Tests for P2 PageIndex TOC trees, tree navigation, Contextual Retrieval, Selective Visual ColBERT, and VLM Reader."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import numpy as np
from PIL import Image

from src.graph.nodes.ground_generate import make_ground_generate_node
from src.graph.nodes.retrieve_rerank import make_retrieve_rerank_node
from src.multimodal_nav.contextual_enricher import ContextualEnricher
from src.multimodal_nav.toc_tree_builder import TocTreeBuilder, TocTreeStore
from src.multimodal_nav.tree_navigator import (
    TocTreeNavigator,
    fuse_tree_and_hybrid_results,
    should_trigger_tree_navigation,
)
from src.multimodal_nav.visual_index import (
    VisualColBERTIndex,
    render_pdf_pages,
    select_visual_pages,
)
from src.multimodal_nav.vlm_page_reader import VLMPageReader
from src.query_plan import QueryPlan
from src.retrieval_filters import RetrievalFilters
from src.storage.base import StorageSecurityError
from src.text_splitter import TextSplitter


class MultimodalNavTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_toc_tree_builder_and_store_handles_ocr_dash_and_toc_alignment(self):
        # Report 1: SPDB-like body headings with OCR dash "第-节"
        pages_body = [
            {
                "page": 3,
                "text": "# 目录 Contents\n| 第一节 重要提示 | 5 |\n| 第二节 公司简介 | 7 |\n| 第三节 管理层讨论与分析 | 15 |\n",
            },
            {"page": 5, "text": "# 第-节 重要提示\n本公司董事会保证本报告内容真实准确完整。\n"},
            {"page": 7, "text": "# 第二节 公司简介和主要财务指标\n公司中文名称：浦发银行。\n"},
            {"page": 15, "text": "# 第三节 管理层讨论与分析\n## 3.1 公司总体经营情况\n报告期内，本行聚焦五大赛道推进数智化转型。\n"},
            {"page": 28, "text": "## 3.2 核心竞争力与可能面对的风险因素\n主要风险包括信用风险、市场风险和流动性风险。\n"},
            {"page": 45, "text": "# 第四节 公司治理\n报告期内股东大会依法合规召开。\n"},
        ]
        builder = TocTreeBuilder()
        tree = builder.build_from_pages(
            doc_id="600000_2024",
            company_name="浦发银行",
            report_year=2024,
            pages=pages_body,
        )
        self.assertGreaterEqual(tree.node_count(), 5)
        self.assertIn("第三节 管理层讨论与分析", tree.find_section_path(30))
        self.assertIn("核心竞争力", tree.find_section_path(30))
        compact_json = tree.to_compact_prompt_json()
        self.assertIn("浦发银行", compact_json)
        self.assertGreater(tree.token_estimate, 10)

        # Report 2: TOC-table-fallback with printed-vs-PDF page offset (+2)
        pages_toc_only = [
            {"page": 1, "text": "2024年年度报告封面"},
            {
                "page": 2,
                "text": (
                    "# 目录\n"
                    "| 第-章 | 公司基本情况简介 | 1 |\n"
                    "| 第二章 | 会计数据和财务指标摘要 | 4 |\n"
                    "| 第三章 | 管理层讨论与分析 | 7 |\n"
                ),
            },
            {"page": 3, "text": "公司基本情况简介\n股票代码：600016\n"},
            {"page": 6, "text": "会计数据和财务指标摘要\n总资产平稳增长\n"},
            {"page": 9, "text": "管理层讨论与分析\n坚持做强做优主业。\n"},
        ]
        tree_toc = builder.build_from_pages(
            doc_id="600016_2024",
            company_name="民生银行",
            report_year=2024,
            pages=pages_toc_only,
        )
        self.assertEqual(tree_toc.source_mode, "toc_table_aligned")
        roots_titles = [r.title for r in tree_toc.roots]
        self.assertEqual(roots_titles[0], "第一章 公司基本情况简介")
        # Offset +2 aligns printed page 7 -> PDF page 9
        self.assertEqual(tree_toc.roots[2].start_page, 9)

        # Running headers + Kangxi radicals + repeated L1 headings are deduplicated
        pages_noisy = [
            {"page": 1, "text": "# 中国民⽣银⾏股份有限公司\n# 董事长致辞\n尊敬的各位股东："},
            {"page": 2, "text": "# 中国民⽣银⾏股份有限公司\n# 董事长致辞\n深化改革。"},
            {"page": 3, "text": "# 中国民⽣银⾏股份有限公司\n# 董事长致辞\n稳健经营。"},
            {"page": 4, "text": "# 中国民⽣银⾏股份有限公司\n# 第-章 公司简介\n## 1.1 公司沿革\n内容"},
            {"page": 5, "text": "# 中国民⽣银⾏股份有限公司\n# 第⼆章 主要财务指标\n## 2.1 营收利润\n内容"},
            {"page": 6, "text": "# 第三章 管理层讨论与分析\n### 3.1.1 资产质量\n内容"},
            {"page": 7, "text": "# 第四章 公司治理\n## 4.1 董事会运作\n内容"},
        ]
        tree_clean = builder.build_from_pages(
            doc_id="noisy_bank",
            company_name="民生银行",
            report_year=2024,
            pages=pages_noisy,
        )
        titles_all = [n.title for n, _ in tree_clean.iter_nodes()]
        self.assertEqual(titles_all.count("董事长致辞"), 1)
        self.assertFalse(any("股份" in t for t in titles_all))
        self.assertIn("第一章 公司简介", titles_all)
        self.assertIn("第二章 主要财务指标", titles_all)

        # SQLite WAL store + read-only LRU lookup + security checks
        db_path = self.root / "finarag_toc_trees.sqlite"
        stats = TocTreeStore._write_trees_sqlite([tree, tree_toc], db_path, allowed_base_dir=self.root)
        self.assertEqual(stats["reports_count"], 2)
        store = TocTreeStore(db_path, allowed_base_dir=self.root)
        self.addCleanup(store.close)
        loaded = store.get_tree("600000_2024")
        self.assertIsNotNone(loaded)
        loaded2 = store.get_tree_by_company("浦发银行", 2024)
        self.assertEqual(loaded2.doc_id, "600000_2024")

        with self.assertRaises(StorageSecurityError):
            TocTreeStore(self.root / "../bad.sqlite", allowed_base_dir=self.root)
        sibling = self.root.parent / f"{self.root.name}-evil" / "toc.sqlite"
        with self.assertRaises(StorageSecurityError):
            TocTreeStore(sibling, allowed_base_dir=self.root)

    def test_tree_navigator_llm_fallback_rrf_and_retrieve_rerank_wiring(self):
        pages = [
            {"page": 5, "text": "# 第一节 重要提示\n声明文本。\n"},
            {"page": 15, "text": "# 第三节 管理层讨论与分析\n收入与盈利结构概览。\n"},
            {"page": 28, "text": "## 3.2 核心竞争力与风险因素\n信用风险防控、不良率管理与宏观利率波动挑战。\n"},
            {"page": 29, "text": "银行通过数字化风控降低潜在信用风险暴露，强化资本充足率缓冲。\n"},
            {"page": 45, "text": "# 第四节 公司治理\n股东大会与董事会运作。\n"},
        ]
        pages_map = {p["page"]: p for p in pages}
        tree = TocTreeBuilder().build_from_pages(
            doc_id="doc-spdb",
            company_name="浦发银行",
            report_year=2024,
            pages=pages,
        )
        nav = TocTreeNavigator()

        plan_synth = QueryPlan(
            original_query="浦发银行2024年面对哪些核心风险与应对措施？",
            normalized_query="浦发银行 风险 应对措施",
            search_queries=["浦发银行 风险因素"],
            filters=RetrievalFilters(company_name="浦发银行", year=2024),
            route_mode="explicit_company",
            expected_answer_type="text",
            task_type="generation_risk_synthesis",
        )
        self.assertTrue(should_trigger_tree_navigation(plan_synth, plan_synth.original_query))

        plan_simple = QueryPlan(
            original_query="浦发银行法定代表人是谁？",
            normalized_query="法定代表人",
            search_queries=["法定代表人"],
            filters=RetrievalFilters(company_name="浦发银行"),
            route_mode="explicit_company",
            expected_answer_type="entity",
            task_type="single_metric_lookup",
        )
        self.assertFalse(should_trigger_tree_navigation(plan_simple, plan_simple.original_query))

        # Deterministic selection picks the risk section
        secs = nav.select_sections(tree, plan_synth.original_query, max_sections=2)
        self.assertTrue(any("风险" in s["title"] for s in secs))

        # Invalid LLM output falls back fail-closed to deterministic selector
        secs_fallback = nav.select_sections(
            tree,
            plan_synth.original_query,
            max_sections=2,
            llm_fn=lambda sys_p, usr_p: "{bad json nonexistent",
        )
        self.assertEqual([s["node_id"] for s in secs_fallback], [s["node_id"] for s in secs])

        # Valid LLM output selects explicit node_id
        target_nid = secs[0]["node_id"]
        secs_llm = nav.select_sections(
            tree,
            plan_synth.original_query,
            max_sections=2,
            llm_fn=lambda sys_p, usr_p: json.dumps({"selected_nodes": [{"node_id": target_nid, "reason": "direct risk chapter"}]}),
        )
        self.assertEqual(secs_llm[0]["node_id"], target_nid)

        tree_hits = nav.retrieve_section_pages(
            tree,
            plan_synth.original_query,
            lambda pno: pages_map.get(pno),
            doc_metainfo={"sha1_name": "doc-spdb", "company_name": "浦发银行", "report_year": 2024},
        )
        self.assertTrue(tree_hits)
        self.assertEqual(tree_hits[0]["retrieval_sources"], ["toc_tree_nav"])
        self.assertIn(tree_hits[0]["page"], {28, 29})

        hybrid_hits = [
            {
                "page": 29,
                "text": "短切片：风险暴露",
                "pdf_sha1": "doc-spdb",
                "metadata": {"sha1_name": "doc-spdb", "node_type": "child"},
                "retrieval_sources": ["vector"],
                "result_scope": "child",
            },
            {
                "page": 5,
                "text": "重要提示",
                "pdf_sha1": "doc-spdb",
                "metadata": {"sha1_name": "doc-spdb", "node_type": "child"},
                "retrieval_sources": ["bm25"],
                "result_scope": "child",
            },
        ]
        fused = fuse_tree_and_hybrid_results(hybrid_hits, tree_hits, top_n=4)
        top_p29 = [h for h in fused if h["page"] == 29][0]
        self.assertEqual(top_p29["retrieval_sources"], ["vector", "toc_tree_nav"])
        self.assertEqual(top_p29["result_scope"], "page")

        # End-to-end retrieve_rerank_node integration with FINARAG_TREE_NAV_ENABLED=1
        docs_dir = self.root / "documents"
        docs_dir.mkdir(parents=True, exist_ok=True)
        toc_db = self.root / "finarag_toc_trees.sqlite"
        TocTreeStore._write_trees_sqlite([tree], toc_db)

        fake_proc = SimpleNamespace(
            documents_dir=docs_dir,
            retrieval_debug_top_n=6,
            _build_retriever=lambda: (
                SimpleNamespace(retrieve_by_company_name=lambda **kw: [dict(hybrid_hits[1])]),
                "vector",
            ),
            _run_retrieval=lambda *a, **kw: [dict(hybrid_hits[1])],
            _merge_multi_query_results=lambda runs, top_n: [dict(hybrid_hits[1])],
            _build_reranking_debug_payload=lambda ret, cap: {},
            _build_hyde_debug_payload=lambda retrieval_results, candidate_pool_size_before_rerank: {},
            _load_document_payload=lambda did: {"metainfo": {"sha1_name": did, "company_name": "浦发银行", "report_year": 2024}, "content": {"pages": pages}},
        )
        node = make_retrieve_rerank_node(fake_proc)
        with patch.dict(os.environ, {"FINARAG_TREE_NAV_ENABLED": "1", "FINARAG_TOC_TREE_DB": str(toc_db)}):
            out = node(
                {
                    "question": plan_synth.original_query,
                    "schema": "text",
                    "company_name": "浦发银行",
                    "query_plan": plan_synth,
                    "selected_doc_id": "doc-spdb",
                    "candidate_doc_ids": ["doc-spdb"],
                }
            )
        self.assertGreater(out["stage_events"][0]["tree_nav_hits"], 0)
        self.assertTrue(any("toc_tree_nav" in r.get("retrieval_sources", []) for r in out["retrieval_results"]))

    def test_contextual_enricher_idempotency_and_text_splitter_toggle(self):
        enricher = ContextualEnricher()
        meta = {
            "company_name": "浦发银行",
            "report_year": 2024,
            "doc_source_type": "annual_report",
            "language": "zh",
        }
        chunk = {
            "page": 109,
            "section_path": "第四节 财务报表附注 > 4.2.10 利润分配",
            "section_name": "财务报表附注",
            "text": "合并口径下期末未分配利润人民币百万元。",
            "embedding_text": "正文：\n合并口径下期末未分配利润。",
            "search_text": "浦发银行\n合并口径下期末未分配利润。",
        }
        llm_calls = []
        enriched1 = enricher.enrich_chunk(
            chunk,
            meta,
            page_text="单位：人民币百万元\n",
            llm_summary_fn=lambda txt, m: (llm_calls.append(txt), "解释合并未分配利润变动情况")[1],
        )
        self.assertIn("[上下文定位] 浦发银行 2024年年度报告", enriched1["embedding_text"])
        self.assertIn("合并口径", enriched1["embedding_text"])
        self.assertIn("单位：人民币百万元", enriched1["embedding_text"])
        self.assertIn("第109页", enriched1["embedding_text"])
        self.assertIn("解释合并未分配利润变动情况", enriched1["search_text"])
        self.assertEqual(len(llm_calls), 1)

        # Idempotent re-enrichment does not duplicate prefix header
        enriched2 = enricher.enrich_chunk(enriched1, meta, page_text="单位：人民币百万元\n")
        self.assertEqual(enriched2["embedding_text"].count("[上下文定位]"), 1)
        self.assertEqual(enriched2["search_text"].count("[上下文定位]"), 1)

        # TextSplitter toggle off (default) vs on
        splitter = TextSplitter()
        off_payload = splitter._build_chunk_payload(
            "正文样本内容", {"page": 5, "section_title": "公司概况", "text": "正文样本内容"}, meta, chunk_id=0, node_type="child"
        )
        self.assertNotIn("[上下文定位]", off_payload["embedding_text"])

        with patch.dict(os.environ, {"FINARAG_CONTEXTUAL_ENRICH_ENABLED": "1"}):
            on_payload = splitter._build_chunk_payload(
                "正文样本内容", {"page": 5, "section_title": "公司概况", "text": "正文样本内容"}, meta, chunk_id=0, node_type="child"
            )
            self.assertTrue(on_payload["embedding_text"].startswith("[上下文定位]"))
            self.assertTrue(on_payload["search_text"].startswith("[上下文定位]"))

    def test_selective_visual_scoring_real_pdf_rendering_and_colbert_maxsim(self):
        # 10 pages total -> max_ratio=0.25 caps kept visual pages to <= 3
        pages = [{"page": i, "text": f"普通文字段落内容 {i} " * 40} for i in range(1, 11)]
        pages[2]["text"] = "参见下图：2024年季度营收走势图\n" + "内容 " * 50
        pages[6]["text"] = "短表页"
        report_dict = {
            "metainfo": {"doc_id": "dummy_doc", "sha1_name": "dummy_doc", "company_name": "示例股份"},
            "content": {
                "pages": pages,
                "structured_tables": [
                    {"table_id": "1", "page": 7, "nrows": 18, "ncols": 6, "continuation_of": "0", "markdown": "| a | b |\n" * 18},
                    {"table_id": "2", "page": 8, "nrows": 4, "ncols": 2, "markdown": "| x | y |"},
                ],
                "charts": [{"chart_id": "c1", "page": 3}],
            },
        }
        selected = select_visual_pages(report_dict, max_ratio=0.25, min_score=0.35)
        self.assertLessEqual(len(selected), 3)
        selected_pages = {s.page for s in selected}
        self.assertIn(3, selected_pages)
        self.assertIn(7, selected_pages)
        self.assertNotIn(8, selected_pages)  # Simple single small table should NOT trigger visual track

        # Real 150 DPI PDF rendering with pypdfium2 on src/dummy_report.pdf
        dummy_pdf = Path("src/dummy_report.pdf")
        out_dir = self.root / "renders"
        rendered = render_pdf_pages(dummy_pdf, [1, 2], out_dir, dpi=150, allowed_output_dir=self.root)
        self.assertEqual(set(rendered.keys()), {1, 2})
        with Image.open(rendered[1]) as img:
            self.assertGreater(img.width, 500)
            self.assertGreater(img.height, 500)

        vis_db = self.root / "visual_colbert.sqlite"
        vis_idx = VisualColBERTIndex(vis_db, allowed_base_dir=self.root)
        self.addCleanup(vis_idx.close)
        n_indexed = vis_idx.index_rendered_pages(
            [
                {"doc_id": "dummy_doc", "page": 1, "company_name": "示例股份", "image_path": rendered[1], "complexity_score": 0.75},
                {"doc_id": "dummy_doc", "page": 2, "company_name": "示例股份", "image_path": rendered[2], "complexity_score": 0.55},
            ],
            allowed_image_dir=self.root,
        )
        self.assertEqual(n_indexed, 2)

        q_vecs = vis_idx._image_encoder(rendered[2])
        hits = vis_idx.search(q_vecs, company_name="示例股份", top_k=2)
        self.assertEqual(hits[0]["page"], 2)
        self.assertEqual(hits[0]["retrieval_sources"], ["visual_colbert"])

        qdrant_pts = vis_idx.export_qdrant_visual_points()
        self.assertEqual(len(qdrant_pts), 2)
        self.assertIn("colbert_vectors", qdrant_pts[0])

        with self.assertRaises(StorageSecurityError):
            render_pdf_pages(dummy_pdf, [1], self.root / "../bad_renders", allowed_output_dir=self.root)

    def test_vlm_page_reader_extraction_security_and_ground_generate_hook(self):
        img_path = self.root / "chart_p19.png"
        Image.new("RGB", (64, 64), color=(240, 240, 240)).save(img_path, format="PNG")

        seen_bodies = []

        def mock_vlm(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8"))
            seen_bodies.append(body)
            user_content = body["messages"][1]["content"]
            prompt_txt = user_content[0]["text"]
            if "待核验候选答案" in prompt_txt:
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "verified": True,
                                            "answer_value": 170748,
                                            "unit": "百万元",
                                            "confidence": 0.96,
                                            "evidence_quote": "2024年营业收入 170,748 百万元",
                                            "reason": "截图核验一致",
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    },
                )
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "title": "营业收入趋势图",
                                        "unit": "百万元",
                                        "headers": ["指标", "2024年", "2023年"],
                                        "rows": [["营业收入", "170,748", "173,434"]],
                                        "confidence": 0.94,
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ]
                },
            )

        reader = VLMPageReader(
            base_url="http://127.0.0.1:9001/v1",
            api_key="test-vlm-key",
            http_transport=httpx.MockTransport(mock_vlm),
        )
        self.addCleanup(reader.close)

        ext = reader.extract_structured_table_from_image(
            img_path, doc_id="doc-1", page=19, allowed_base_dir=self.root
        )
        self.assertEqual(ext.unit, "百万元")
        self.assertIn("170,748", ext.table_markdown)
        self.assertEqual(len(ext.records), 2)
        self.assertEqual(seen_bodies[0]["messages"][1]["content"][1]["type"], "image_url")
        self.assertTrue(seen_bodies[0]["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))

        ver = reader.verify_grounding_with_vlm(
            img_path, question="2024年营业收入是多少？", candidate_answer=170748, page=19, allowed_base_dir=self.root
        )
        self.assertTrue(ver.verified)
        self.assertEqual(ver.answer_value, 170748)

        # Security guards: URL scheme, path traversal, invalid suffix, oversized file, fail-closed HTTP 500
        with self.assertRaises(StorageSecurityError):
            VLMPageReader(base_url="ftp://unsafe.local")
        with self.assertRaises(StorageSecurityError):
            reader._encode_image_data_url(self.root / "../escape.png", allowed_base_dir=self.root)
        txt_file = self.root / "not_img.txt"
        txt_file.write_text("bad", encoding="utf-8")
        with self.assertRaises(StorageSecurityError):
            reader._encode_image_data_url(txt_file, allowed_base_dir=self.root)
        with self.assertRaises(StorageSecurityError):
            reader._encode_image_data_url(img_path, allowed_base_dir=self.root, max_bytes=10)

        err_reader = VLMPageReader(
            base_url="http://127.0.0.1:9001/v1",
            http_transport=httpx.MockTransport(lambda req: httpx.Response(500, json={"error": "boom"})),
        )
        self.addCleanup(err_reader.close)
        fail_ver = err_reader.verify_grounding_with_vlm(img_path, question="营收?", candidate_answer=100, page=19)
        self.assertFalse(fail_ver.verified)
        self.assertTrue(fail_ver.reason.startswith("fail_closed:"))

        # Online ground_generate_node VLM hook when chart confidence is low
        pdf_dir = self.root / "pdf_reports"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy("src/dummy_report.pdf", pdf_dir / "doc-1.pdf")

        mock_proc = SimpleNamespace(
            table_grounder=None,
            chart_grounder=SimpleNamespace(
                ground_number_query=lambda **kw: {
                    "page": 1,
                    "answer_value": 170748,
                    "unit": "百万元",
                    "confidence": 0.45,
                    "text": "低置信图表",
                    "source_doc_id": "doc-1",
                }
            ),
            vlm_page_reader=reader,
            pdf_reports_dir=pdf_dir,
            top_n_retrieval=5,
            answering_model="dummy-model",
            answer_temperature=0.0,
            reasoning_debug_enabled=False,
            _grounded_number_value=lambda g: g.get("answer_value"),
            _ensure_chart_grounding_retrieval_result=lambda res, g: res,
            _build_chart_grounded_number_answer=lambda g: {
                "final_answer": g["answer_value"],
                "relevant_pages": [1],
                "chart_grounding_result": g,
                "reasoning_summary": "图表接地",
                "step_by_step_analysis": "图表接地",
            },
            _validate_page_references=lambda pages, res: pages,
            _extract_references=lambda pages, comp, pdf_sha1=None: [{"pdf_sha1": "doc-1", "page_index": 0}],
            _table_support_results=lambda t: [],
            _serialize_retrieval_result=lambda r: r,
            _aggregate_retrieval_results_by_report=lambda res: [],
        )
        gg_node = make_ground_generate_node(mock_proc)
        qplan = QueryPlan(
            original_query="2024年营业收入是多少？",
            normalized_query="营业收入",
            search_queries=["营业收入"],
            filters=RetrievalFilters(company_name="浦发银行", year=2024),
            route_mode="explicit_company",
            expected_answer_type="numeric",
        )
        with patch.dict(os.environ, {"FINARAG_VLM_VERIFY_ENABLED": "1"}):
            gg_out = gg_node(
                {
                    "question": qplan.original_query,
                    "schema": "number",
                    "company_name": "浦发银行",
                    "query_plan": qplan,
                    "route_info": {"selected_report": {"sha1": "doc-1"}, "route_mode": "explicit_company"},
                    "candidate_doc_ids": ["doc-1"],
                    "allowed_grounding_doc_ids": ["doc-1"],
                    "reranking_debug": {
                        "reranking_strategy": "single",
                        "initial_candidate_pool_size": None,
                        "colbert_candidate_pool_size": None,
                        "colbert_top_n": None,
                        "final_reranking_backend": "none",
                    },
                    "hyde_debug": {"enabled": False, "triggered": False},
                    "retrieval_results": [
                        {
                            "page": 1,
                            "text": "营收数据",
                            "pdf_sha1": "doc-1",
                            "metadata": {"sha1_name": "doc-1", "company_name": "浦发银行", "report_year": 2024},
                            "retrieval_sources": ["vector"],
                        }
                    ],
                }
            )
        self.assertIn("vlm_verification", gg_out["final_answer_dict"])
        self.assertTrue(gg_out["final_answer_dict"]["vlm_verification"]["verified"])


if __name__ == "__main__":
    unittest.main()
