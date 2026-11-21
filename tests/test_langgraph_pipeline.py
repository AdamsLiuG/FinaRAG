"""Integration tests for LangGraph state machine, plugins, reflection loop, Map-Reduce, and legacy parity."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.graph.builder import stream_finarag_events
from src.questions_processing import QuestionsProcessor
from src.structured.fin_table_extractor import FinancialTableExtractor


class _FakeRetriever:
    def __init__(self, initial_results, supplement_results):
        self._initial = list(initial_results)
        self._supplement = list(supplement_results)
        self.supplement_calls = []

    def retrieve_by_company_name(self, **kwargs):
        return list(self._initial)

    def retrieve_candidates_by_company_name(self, **kwargs):
        self.supplement_calls.append(kwargs)
        return list(self._supplement)

    def rerank_candidates(self, query, candidate_results, documents_batch_size=2, top_n=10, task_type=None):
        return list(candidate_results)[:top_n]


def _make_result(page: int, text: str, score: float = 0.8, sha1: str = "doc-spdb", report_year: int = 2024):
    return {
        "page": page,
        "text": text,
        "pdf_sha1": sha1,
        "combined_score": score,
        "distance": score,
        "ranking_score": score,
        "metadata": {
            "sha1_name": sha1,
            "company_name": "浦发银行",
            "report_year": report_year,
            "currency": "CNY",
            "chunk_id": page,
            "chunk_type": "content",
            "node_type": "child",
        },
        "retrieval_sources": ["vector"],
    }


class LangGraphPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        self.docs_dir = self.root / "documents"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.subset_path = self.root / "document_manifest.csv"
        self.subset_path.write_text(
            "doc_id,company_name,company_aliases,security_code,doc_source_type,report_title,language,currency,report_year\n"
            "doc-spdb,浦发银行,浦发银行|600000,600000,annual_report,浦发银行2024年年报,zh,CNY,2024\n"
            "doc-jn,京能置业,京能置业|600791,600791,annual_report,京能置业2024年年报,zh,CNY,2024\n",
            encoding="utf-8",
        )
        spdb_doc = {
            "metainfo": {
                "doc_id": "doc-spdb",
                "sha1_name": "doc-spdb",
                "company_name": "浦发银行",
                "stock_code": "600000",
                "report_year": 2024,
                "currency": "CNY",
            },
            "content": {
                "pages": [
                    {"page": 4, "text": "公司简介\n法定代表人：张为忠\n"},
                    {"page": 19, "text": "| 单位：人民币百万元 | 2024 年 | 2023 年 |\n|---|---|---|\n| 营业收入 | 170,748 | 173,434 |\n"},
                ],
                "structured_tables": [
                    {
                        "table_id": "11",
                        "page": 19,
                        "markdown": "| 单位：人民币百万元 | 2024 年 | 2023 年 |\n|---|---|---|\n| 营业收入 | 170,748 | 173,434 |\n",
                    }
                ],
            },
        }
        jn_doc = {
            "metainfo": {
                "doc_id": "doc-jn",
                "sha1_name": "doc-jn",
                "company_name": "京能置业",
                "stock_code": "600791",
                "report_year": 2024,
                "currency": "CNY",
            },
            "content": {
                "pages": [
                    {"page": 5, "text": "单位：元 币种：人民币\n| 主要会计数据 | 2024 年 | 2023 年 |\n|---|---|---|\n| 营业收入 | 5,458,594,505.42 | 4,535,104,206.81 |\n"},
                ],
                "structured_tables": [
                    {
                        "table_id": "9",
                        "page": 5,
                        "markdown": "| 主要会计数据 | 2024 年 | 2023 年 |\n|---|---|---|\n| 营业收入 | 5,458,594,505.42 | 4,535,104,206.81 |\n",
                    }
                ],
            },
        }
        (self.docs_dir / "doc-spdb.json").write_text(json.dumps(spdb_doc, ensure_ascii=False), encoding="utf-8")
        (self.docs_dir / "doc-jn.json").write_text(json.dumps(jn_doc, ensure_ascii=False), encoding="utf-8")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _build_processor(self, **kwargs):
        return QuestionsProcessor(
            documents_dir=self.docs_dir,
            subset_path=self.subset_path,
            parallel_requests=2,
            **kwargs,
        )

    def test_rule_plugin_delegation_in_langgraph_mode(self):
        processor = self._build_processor()
        self.assertEqual(processor._resolved_graph_engine(), "langgraph")
        ans = processor.process_question("浦发银行2024年年报中的法定代表人是谁？", "name")
        self.assertEqual(ans["final_answer"], "张为忠")
        self.assertEqual(ans["route_info"]["route_mode"], "legal_representative_rule")

    def test_post_validate_reflection_loop_handles_report_year_mismatch(self):
        processor = self._build_processor(use_vector_dbs=True, llm_reranking=False)
        # First retrieval returns a 2023-only snippet (triggers report_year_mismatch);
        # Supplement returns 2024 parent page snippet which clears the mismatch flag.
        initial = [_make_result(page=2, text="2023 年公司实现归母净利润 367 亿元", report_year=2023)]
        supplement = [_make_result(page=19, text="2024 年公司实现营业收入 170748 百万元", report_year=2024)]
        fake_retriever = _FakeRetriever(initial, supplement)
        processor._build_retriever = lambda: (fake_retriever, "vector")

        call_count = {"n": 0}

        def fake_llm(question, rag_context, schema, model, temperature=0.0, metric_caliber_hint=None):
            call_count["n"] += 1
            if "170748" in rag_context:
                return {
                    "final_answer": "1707.48亿元",
                    "relevant_pages": [19],
                    "reasoning_summary": "依据2024年年报第19页。",
                    "step_by_step_analysis": "从第19页读取2024年数据。",
                }
            return {
                "final_answer": "367亿元",
                "relevant_pages": [2],
                "reasoning_summary": "第2页。",
                "step_by_step_analysis": "第2页。",
            }

        processor.api_processor.get_answer_from_rag_context = fake_llm

        with patch.dict(os.environ, {"ADAPTIVE_RETRIEVAL_ENABLED": "1", "ADAPTIVE_MAX_ROUNDS": "2"}):
            # Also patch _record_adaptive_trace so test doesn't write to repo logs/
            recorded = []
            processor._record_adaptive_trace = lambda state, entry=None: recorded.append((state, entry))
            ans = processor.process_question("浦发银行2024年的主要经营业绩如何？", "text")

        self.assertEqual(call_count["n"], 2)
        self.assertGreaterEqual(len(fake_retriever.supplement_calls), 1)
        self.assertTrue(
            all(call["parent_retrieval_mode"] == "page" for call in fake_retriever.supplement_calls)
        )
        self.assertEqual(ans["final_answer"], "1707.48亿元")
        self.assertNotIn("report_year_mismatch", ans.get("validation_flags") or [])
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0][0]["traces"][0]["rounds"], 1)

    def test_comparative_map_reduce_and_incremental_streaming(self):
        processor = self._build_processor(use_vector_dbs=True, llm_reranking=False)
        initial = [_make_result(page=19, text="2024年业务发展稳健。")]
        fake_retriever = _FakeRetriever(initial, [])
        processor._build_retriever = lambda: (fake_retriever, "vector")
        processor.api_processor.get_rephrased_questions = lambda original_question, companies, model=None: {
            c: f"{c} 2024年战略亮点是什么？" for c in companies
        }

        def fake_llm(question, rag_context, schema, model, temperature=0.0, metric_caliber_hint=None):
            if schema == "comparative" or isinstance(rag_context, dict):
                return {
                    "final_answer": "两家公司均聚焦主业。",
                    "relevant_pages": [],
                    "reasoning_summary": "汇总两家公司。",
                    "step_by_step_analysis": "Map-Reduce 聚合。",
                }
            return {
                "final_answer": "稳健发展",
                "relevant_pages": [19],
                "reasoning_summary": "第19页。",
                "step_by_step_analysis": "第19页。",
            }

        processor.api_processor.get_answer_from_rag_context = fake_llm

        comp_res = processor.process_question("对比浦发银行和京能置业2024年经营策略差异", "text")
        self.assertEqual(comp_res["final_answer"], "两家公司均聚焦主业。")
        self.assertEqual(comp_res["route_info"]["candidate_companies"], ["浦发银行", "京能置业"])

        # Verify incremental stage streaming emits ordered node events
        events = list(stream_finarag_events(processor, "浦发银行2024年经营策略如何？", "text"))
        stages = [ev["stage"] for ev in events]
        self.assertEqual(stages[0], "prepare_single")
        self.assertIn("retrieve_rerank", stages)
        self.assertIn("ground_generate", stages)
        self.assertEqual(stages[-1], "result")

    def test_structured_sql_fast_track_and_legacy_parity(self):
        db_path = self.root / "finarag_facts.sqlite"
        FinancialTableExtractor().build_sqlite_db(self.docs_dir, db_path)

        sql_processor = self._build_processor(structured_db_path=db_path)
        # Ensure LLM is NOT called for SQL fast-track
        sql_processor.api_processor.get_answer_from_rag_context = lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("SQL fast-track should bypass LLM")
        )
        yoy = sql_processor.process_question("浦发银行2024年营业收入同比增速是多少？", "number")
        self.assertAlmostEqual(yoy["final_answer"], -1.55, places=2)
        self.assertEqual(yoy["route_info"]["route_mode"], "structured_sql_pot_yoy")

        # Verify fail-closed fallback to RAG when SQL cannot answer
        fake_retriever = _FakeRetriever([_make_result(page=19, text="2024年发展良好")], [])
        sql_processor._build_retriever = lambda: (fake_retriever, "vector")
        sql_processor.api_processor.get_answer_from_rag_context = lambda **kwargs: {
            "final_answer": "发展良好",
            "relevant_pages": [19],
            "reasoning_summary": "RAG 兜底",
            "step_by_step_analysis": "RAG 兜底",
        }
        fallback = sql_processor.process_question("浦发银行2024年风险管理举措有哪些？", "text")
        self.assertEqual(fallback["final_answer"], "发展良好")

        # Legacy engine parity check
        lg_ans = self._build_processor(graph_engine="langgraph").process_question("浦发银行2024年年报中的法定代表人是谁？", "name")
        legacy_ans = self._build_processor(graph_engine="legacy").process_question("浦发银行2024年年报中的法定代表人是谁？", "name")
        self.assertEqual(lg_ans["final_answer"], legacy_ans["final_answer"])
        self.assertEqual(lg_ans["relevant_pages"], legacy_ans["relevant_pages"])
        self.assertEqual(lg_ans["confidence"], legacy_ans["confidence"])


if __name__ == "__main__":
    unittest.main()
