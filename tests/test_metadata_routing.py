import json
import tempfile
import unittest
from pathlib import Path

from src.query_plan import QueryPlan
from src.questions_processing import QuestionsProcessor
from src.retrieval_filters import RetrievalFilters


class MetadataRoutingTests(unittest.TestCase):
    def test_process_question_routes_by_document_catalog_when_company_name_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            subset_path = Path(tmp_dir) / "document_manifest.csv"
            subset_path.write_text(
                "doc_id,company_name,company_aliases,security_code,doc_source_type,broker_name,report_title,language,currency\n"
                "doc-alpha,宁德时代,宁王|CATL|300750,300750,research_report,中信证券,中信证券-宁德时代深度报告,zh,CNY\n"
                "doc-beta,贵州茅台,茅台|600519,600519,annual_report,,贵州茅台2024年年报,zh,CNY\n",
                encoding="utf-8",
            )

            processor = QuestionsProcessor(subset_path=subset_path, parallel_requests=1, doc_router_enabled=True)
            seen = {}

            def fake_get_answer_for_company(company_name: str, question: str, schema: str, query_plan=None, route_info=None):
                seen["company_name"] = company_name
                return {
                    "final_answer": True,
                    "references": [],
                    "citations": [],
                    "confidence": "medium",
                    "confidence_reason": "test",
                    "validation_flags": [],
                    "route_info": route_info,
                    "query_plan": query_plan.to_dict() if query_plan else {},
                    "relevant_pages": [2],
                }

            processor.get_answer_for_company = fake_get_answer_for_company

            result = processor.process_question(
                "中信证券这篇研报提到的公司营业收入是多少？",
                "number",
            )

            self.assertEqual(seen["company_name"], "宁德时代")
            self.assertEqual(result["route_info"]["route_mode"], "document_catalog")
            self.assertEqual(result["route_info"]["selected_company"], "宁德时代")
            self.assertIn("doc-alpha", result["route_info"]["candidate_doc_ids"])

    def test_process_question_routes_metadata_names_queries_to_multi_document_catalog(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            subset_path = root / "document_manifest.csv"
            subset_path.write_text(
                "doc_id,company_name,company_aliases,security_code,doc_source_type,report_title,language,currency,report_year\n"
                "doc-alpha,星云科技,星云科技|688001,688001,annual_report,星云科技2024年年报,zh,CNY,2024\n"
                "doc-beta,光谱软件,光谱软件|688002,688002,annual_report,光谱软件2024年年报,zh,CNY,2024\n"
                "doc-gamma,海川制造,海川制造|600001,600001,annual_report,海川制造2024年年报,zh,CNY,2024\n",
                encoding="utf-8",
            )

            metadata_store = root / "metadata_store"
            metadata_store.mkdir(parents=True, exist_ok=True)
            snapshot_path = metadata_store / "company_label_snapshot.jsonl"
            snapshot_path.write_text(
                "\n".join(
                    json.dumps(payload, ensure_ascii=False)
                    for payload in [
                        {
                            "report_id": "doc-alpha",
                            "board": "科创板",
                            "industry_l1": "信息技术",
                            "strategy_tags": ["国产替代", "人工智能"],
                        },
                        {
                            "report_id": "doc-beta",
                            "board": "科创板",
                            "industry_l1": "信息技术",
                            "strategy_tags": ["国产替代"],
                        },
                        {
                            "report_id": "doc-gamma",
                            "board": "主板",
                            "industry_l1": "工业",
                            "strategy_tags": ["智能制造"],
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            processor = QuestionsProcessor(subset_path=subset_path, parallel_requests=1, doc_router_enabled=True)
            seen = {}

            def fake_build_query_plan(question: str, schema: str, company_name=None, mentioned_companies=None, route_mode=None):
                return QueryPlan(
                    original_query=question,
                    normalized_query=question,
                    search_queries=[question],
                    filters=RetrievalFilters(
                        doc_source_type="annual_report",
                        board="科创板",
                        industry_l1="信息技术",
                        strategy_tags=["国产替代"],
                    ),
                    route_mode=route_mode or "document_catalog",
                    expected_answer_type="entity_list",
                    mentioned_companies=list(mentioned_companies or []),
                )

            def fake_get_answer_for_company(company_name: str, question: str, schema: str, query_plan=None, route_info=None):
                seen["company_name"] = company_name
                seen["query_plan"] = query_plan
                seen["route_info"] = route_info
                return {
                    "final_answer": ["星云科技", "光谱软件"],
                    "references": [],
                    "citations": [],
                    "confidence": "medium",
                    "confidence_reason": "test",
                    "validation_flags": [],
                    "route_info": route_info,
                    "query_plan": query_plan.to_dict() if query_plan else {},
                    "relevant_pages": [3, 5],
                }

            processor._build_query_plan = fake_build_query_plan
            processor.get_answer_for_company = fake_get_answer_for_company

            result = processor.process_question(
                "在信息技术行业的科创板公司中，哪些公司在2024年年报中提到“国产替代”？",
                "names",
            )

            self.assertEqual(result["route_info"]["route_mode"], "document_catalog_multi")
            self.assertIsNone(seen["query_plan"].filters.company_name)
            self.assertEqual(
                set(seen["query_plan"].filters.candidate_doc_ids or []),
                {"doc-alpha", "doc-beta"},
            )
            self.assertEqual(
                set(seen["route_info"]["candidate_doc_ids"]),
                {"doc-alpha", "doc-beta"},
            )
            self.assertEqual(
                set(seen["route_info"]["candidate_companies"]),
                {"星云科技", "光谱软件"},
            )
            self.assertIn(seen["company_name"], {"星云科技", "光谱软件"})

    def test_get_answer_for_company_uses_multi_document_references_without_company_filter(self):
        processor = QuestionsProcessor(parallel_requests=1)

        query_plan = QueryPlan(
            original_query="哪些公司提到国产替代？",
            normalized_query="哪些公司提到国产替代？",
            search_queries=["哪些公司提到国产替代？"],
            filters=RetrievalFilters(
                board="科创板",
                strategy_tags=["国产替代"],
                candidate_doc_ids=["doc-alpha", "doc-beta"],
            ),
            route_mode="document_catalog_multi",
            expected_answer_type="entity_list",
        )
        route_info = {
            "route_mode": "document_catalog_multi",
            "candidate_doc_ids": ["doc-alpha", "doc-beta"],
            "candidate_companies": ["星云科技", "光谱软件"],
            "selected_report": None,
        }

        recorded = {}

        def fake_run_retrieval(retriever, mode: str, company_name: str, query: str, filters, candidate_doc_ids=None):
            recorded["filters"] = filters
            recorded["candidate_doc_ids"] = list(candidate_doc_ids or [])
            return [
                {
                    "page": 3,
                    "text": "星云科技在年报中提到国产替代。",
                    "metadata": {
                        "sha1_name": "doc-alpha",
                        "company_name": "星云科技",
                        "stock_code": "688001",
                        "report_year": 2024,
                        "chunk_type": "content",
                        "node_type": "child",
                    },
                    "combined_score": 0.92,
                    "matched_tags": ["国产替代"],
                    "retrieval_sources": ["tag"],
                },
                {
                    "page": 5,
                    "text": "光谱软件在年报中提到国产替代。",
                    "metadata": {
                        "sha1_name": "doc-beta",
                        "company_name": "光谱软件",
                        "stock_code": "688002",
                        "report_year": 2024,
                        "chunk_type": "content",
                        "node_type": "child",
                    },
                    "combined_score": 0.87,
                    "matched_tags": ["国产替代"],
                    "retrieval_sources": ["tag"],
                },
            ]

        processor._build_retriever = lambda: (object(), "hybrid")
        processor._run_retrieval = fake_run_retrieval
        processor.api_processor.response_data = {}
        processor.api_processor.get_answer_from_rag_context = lambda **_: {
            "final_answer": ["星云科技", "光谱软件"],
            "relevant_pages": [3, 5],
            "reasoning_summary": "stubbed",
            "step_by_step_analysis": "",
        }

        answer = processor.get_answer_for_company(
            company_name="星云科技",
            question="哪些公司提到国产替代？",
            schema="names",
            query_plan=query_plan,
            route_info=route_info,
        )

        self.assertIsNone(recorded["filters"].company_name)
        self.assertEqual(recorded["candidate_doc_ids"], ["doc-alpha", "doc-beta"])
        self.assertEqual(
            answer["references"],
            [
                {"pdf_sha1": "doc-alpha", "page_index": 3},
                {"pdf_sha1": "doc-beta", "page_index": 5},
            ],
        )
        self.assertEqual(answer["route_info"]["route_mode"], "document_catalog_multi")
        self.assertEqual(len(answer["retrieval_report_groups"]), 2)


if __name__ == "__main__":
    unittest.main()
