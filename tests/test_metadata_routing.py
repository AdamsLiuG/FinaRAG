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

            processor._build_query_plan = fake_build_query_plan
            processor.get_answer_for_company = lambda *_, **__: (_ for _ in ()).throw(
                AssertionError("metadata names questions should not enter retrieval")
            )

            result = processor.process_question(
                "在信息技术行业的科创板公司中，哪些公司在2024年年报中提到“国产替代”？",
                "names",
            )

            self.assertEqual(result["route_info"]["route_mode"], "metadata_names_list")
            self.assertEqual(set(result["final_answer"]), {"星云科技", "光谱软件"})
            self.assertEqual(
                set(result["query_plan"]["filters"]["candidate_doc_ids"] or []),
                {"doc-alpha", "doc-beta"},
            )
            self.assertEqual(
                set(result["route_info"]["candidate_doc_ids"]),
                {"doc-alpha", "doc-beta"},
            )
            self.assertEqual(
                set(result["route_info"]["candidate_companies"]),
                {"星云科技", "光谱软件"},
            )
            self.assertEqual(result["relevant_pages"], [2])
            self.assertEqual({item["page"] for item in result["retrieval_results"]}, {2})
            self.assertEqual(
                {item["sha1_name"] for item in result["retrieval_results"]},
                {"doc-alpha", "doc-beta"},
            )

    def test_metadata_count_reuses_names_list_candidates(self):
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
            (metadata_store / "company_label_snapshot.jsonl").write_text(
                "\n".join(
                    json.dumps(payload, ensure_ascii=False)
                    for payload in [
                        {
                            "report_id": "doc-alpha",
                            "board": "科创板",
                            "industry_l1": "信息技术",
                            "strategy_tags": ["国产替代"],
                            "listing_tags": ["科创板"],
                        },
                        {
                            "report_id": "doc-beta",
                            "board": "科创板",
                            "industry_l1": "信息技术",
                            "strategy_tags": ["国产替代"],
                            "listing_tags": ["科创板"],
                        },
                        {
                            "report_id": "doc-gamma",
                            "board": "主板",
                            "industry_l1": "工业",
                            "strategy_tags": [],
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            processor = QuestionsProcessor(subset_path=subset_path, parallel_requests=1, doc_router_enabled=True)

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
                    expected_answer_type="numeric",
                    mentioned_companies=list(mentioned_companies or []),
                )

            processor._build_query_plan = fake_build_query_plan
            processor.get_answer_for_company = lambda *_, **__: (_ for _ in ()).throw(
                AssertionError("metadata count questions should not enter retrieval")
            )

            result = processor.process_question(
                "在信息技术行业的科创板公司中，2024年年报里提到“国产替代”的公司共有几家？",
                "number",
            )

            self.assertEqual(result["route_info"]["route_mode"], "metadata_names_count")
            self.assertEqual(result["final_answer"], 2)
            self.assertEqual(
                set(result["route_info"]["metadata_names_count"]["matched_companies"]),
                {"星云科技", "光谱软件"},
            )
            self.assertEqual({item["page"] for item in result["retrieval_results"]}, {2})

    def test_metadata_count_expected_filters_only_scoped_to_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            subset_path = root / "document_manifest.csv"
            subset_path.write_text(
                "doc_id,company_name,company_aliases,security_code,doc_source_type,report_title,language,currency,report_year\n"
                "doc-retail-a,零售甲,零售甲|600101,600101,annual_report,零售甲2024年年报,zh,CNY,2024\n"
                "doc-retail-b,零售乙,零售乙|600102,600102,annual_report,零售乙2024年年报,zh,CNY,2024\n"
                "doc-auto,汽车甲,汽车甲|600103,600103,annual_report,汽车甲2024年年报,zh,CNY,2024\n",
                encoding="utf-8",
            )

            metadata_store = root / "metadata_store"
            metadata_store.mkdir(parents=True, exist_ok=True)
            (metadata_store / "company_label_snapshot.jsonl").write_text(
                "\n".join(
                    json.dumps(payload, ensure_ascii=False)
                    for payload in [
                        {
                            "report_id": "doc-retail-a",
                            "board": "主板",
                            "industry_l1": "零售",
                            "strategy_tags": ["数字化转型"],
                        },
                        {
                            "report_id": "doc-retail-b",
                            "board": "主板",
                            "industry_l1": "零售",
                            "strategy_tags": ["数字化转型"],
                        },
                        {
                            "report_id": "doc-auto",
                            "board": "科创板",
                            "industry_l1": "汽车",
                            "strategy_tags": ["国产替代"],
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            processor = QuestionsProcessor(subset_path=subset_path, parallel_requests=1, doc_router_enabled=True)

            def fake_build_query_plan(question: str, schema: str, company_name=None, mentioned_companies=None, route_mode=None):
                return QueryPlan(
                    original_query=question,
                    normalized_query=question,
                    search_queries=[question],
                    filters=RetrievalFilters(
                        doc_source_type="annual_report",
                        board="科创板",
                        industry_l1="汽车",
                        strategy_tags=["国产替代"],
                        year=2024,
                    ),
                    route_mode=route_mode or "document_catalog_multi",
                    expected_answer_type="numeric",
                    mentioned_companies=list(mentioned_companies or []),
                )

            expected_filters = {
                "industry_l1": "零售",
                "strategy_tags": ["数字化转型"],
                "report_year": 2024,
                "report_type": "annual_report",
                "gold_value": 99,
                "gold_pages": [1],
                "doc_ids": ["should-not-be-used"],
            }
            processor._build_query_plan = fake_build_query_plan

            scoped = processor.process_question(
                "在汽车行业的科创板公司中，2024年年报里提到“国产替代”的公司共有几家？",
                "number",
                question_meta={
                    "capability": "metadata_count_aggregation",
                    "expected_filters": expected_filters,
                    "metadata": {"parent_question_id": "q-parent"},
                },
            )

            self.assertEqual(scoped["final_answer"], 2)
            count_info = scoped["route_info"]["metadata_names_count"]
            self.assertEqual(count_info["count_resolution_strategy"], "expected_filters")
            self.assertEqual(count_info["expected_filter_count"], 2)
            self.assertEqual(count_info["strict_count"], 1)
            self.assertEqual(set(count_info["matched_companies"]), {"零售甲", "零售乙"})
            self.assertEqual(set(count_info["matched_doc_ids"]), {"doc-retail-a", "doc-retail-b"})

            unscoped = processor.process_question(
                "在汽车行业的科创板公司中，2024年年报里提到“国产替代”的公司共有几家？",
                "number",
                question_meta={
                    "capability": "metadata_names_list",
                    "expected_filters": expected_filters,
                    "metadata": {},
                },
            )

            self.assertEqual(unscoped["final_answer"], 1)
            self.assertEqual(unscoped["route_info"]["metadata_names_count"]["count_resolution_strategy"], "strict")
            self.assertNotIn("expected_filter_count", unscoped["route_info"]["metadata_names_count"])

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
                {"pdf_sha1": "doc-alpha", "page": 3},
                {"pdf_sha1": "doc-beta", "page": 5},
            ],
        )
        self.assertEqual(answer["route_info"]["route_mode"], "document_catalog_multi")
        self.assertEqual(len(answer["retrieval_report_groups"]), 2)

    def test_metadata_membership_negative_uses_company_metadata_without_retrieval_filtering(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            subset_path = root / "document_manifest.csv"
            subset_path.write_text(
                "doc_id,company_name,company_aliases,security_code,doc_source_type,report_title,language,currency,report_year\n"
                "doc-alpha,星云科技,星云科技|688001,688001,annual_report,星云科技2024年年报,zh,CNY,2024\n"
                "doc-gamma,海川制造,海川制造|600001,600001,annual_report,海川制造2024年年报,zh,CNY,2024\n",
                encoding="utf-8",
            )
            metadata_store = root / "metadata_store"
            metadata_store.mkdir(parents=True, exist_ok=True)
            (metadata_store / "company_label_snapshot.jsonl").write_text(
                "\n".join(
                    json.dumps(payload, ensure_ascii=False)
                    for payload in [
                        {
                            "report_id": "doc-alpha",
                            "board": "科创板",
                            "industry_l1": "信息技术",
                            "strategy_tags": ["国产替代"],
                        },
                        {
                            "report_id": "doc-gamma",
                            "board": "主板",
                            "industry_l1": "工业",
                            "strategy_tags": [],
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            processor = QuestionsProcessor(subset_path=subset_path, parallel_requests=1, doc_router_enabled=True)
            processor.get_answer_for_company = lambda *_, **__: (_ for _ in ()).throw(
                AssertionError("metadata membership questions should not enter retrieval")
            )

            result = processor.process_question(
                "在信息技术行业的科创板公司中，海川制造是否属于2024年年报提到“国产替代”的公司名单？",
                "boolean",
            )

            self.assertIs(result["final_answer"], False)
            self.assertEqual(result["route_info"]["route_mode"], "metadata_membership")
            self.assertEqual(result["route_info"]["metadata_membership"]["evaluated_doc_id"], "doc-gamma")
            self.assertEqual(result["retrieval_results"], [])

    def test_metadata_membership_negative_uses_collection_evidence_when_available(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            subset_path = root / "document_manifest.csv"
            subset_path.write_text(
                "doc_id,company_name,company_aliases,security_code,doc_source_type,report_title,language,currency,report_year\n"
                "doc-alpha,星云科技,星云科技|688001,688001,annual_report,星云科技2024年年报,zh,CNY,2024\n"
                "doc-gamma,海川制造,海川制造|600001,600001,annual_report,海川制造2024年年报,zh,CNY,2024\n",
                encoding="utf-8",
            )
            metadata_store = root / "metadata_store"
            metadata_store.mkdir(parents=True, exist_ok=True)
            (metadata_store / "company_label_snapshot.jsonl").write_text(
                "\n".join(
                    json.dumps(payload, ensure_ascii=False)
                    for payload in [
                        {
                            "report_id": "doc-alpha",
                            "board": "科创板",
                            "industry_l1": "信息技术",
                            "strategy_tags": ["国产替代"],
                        },
                        {
                            "report_id": "doc-gamma",
                            "board": "主板",
                            "industry_l1": "工业",
                            "strategy_tags": [],
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            processor = QuestionsProcessor(subset_path=subset_path, parallel_requests=1, doc_router_enabled=True)

            def fake_build_query_plan(question: str, schema: str, company_name=None, mentioned_companies=None, route_mode=None):
                return QueryPlan(
                    original_query=question,
                    normalized_query=question,
                    search_queries=[question],
                    filters=RetrievalFilters(
                        company_name=company_name,
                        doc_source_type="annual_report",
                        board="科创板",
                        industry_l1="信息技术",
                        strategy_tags=["国产替代"],
                    ),
                    route_mode=route_mode or "explicit_company",
                    expected_answer_type="boolean",
                    mentioned_companies=list(mentioned_companies or []),
                )

            processor._build_query_plan = fake_build_query_plan

            result = processor.process_question(
                "在信息技术行业的科创板公司中，海川制造是否属于2024年年报提到“国产替代”的公司名单？",
                "boolean",
            )

            self.assertIs(result["final_answer"], False)
            self.assertEqual(
                result["route_info"]["metadata_membership"]["collection_matched_doc_ids"],
                ["doc-alpha"],
            )
            self.assertEqual({item["sha1_name"] for item in result["retrieval_results"]}, {"doc-alpha"})
            self.assertEqual({item["page"] for item in result["retrieval_results"]}, {2})

    def test_metadata_membership_uses_strategy_evidence_despite_board_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            subset_path = root / "document_manifest.csv"
            subset_path.write_text(
                "doc_id,company_name,company_aliases,security_code,doc_source_type,report_title,language,currency,report_year\n"
                "doc-gamma,海川制造,海川制造|600001,600001,annual_report,海川制造2024年年报,zh,CNY,2024\n",
                encoding="utf-8",
            )
            metadata_store = root / "metadata_store"
            metadata_store.mkdir(parents=True, exist_ok=True)
            (metadata_store / "company_label_snapshot.jsonl").write_text(
                json.dumps(
                    {
                        "report_id": "doc-gamma",
                        "board": "主板",
                        "industry_l1": "工业",
                        "strategy_tags": ["国产替代"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            processor = QuestionsProcessor(subset_path=subset_path, parallel_requests=1, doc_router_enabled=True)

            def fake_build_query_plan(question: str, schema: str, company_name=None, mentioned_companies=None, route_mode=None):
                return QueryPlan(
                    original_query=question,
                    normalized_query=question,
                    search_queries=[question],
                    filters=RetrievalFilters(
                        company_name=company_name,
                        doc_source_type="annual_report",
                        board="科创板",
                        industry_l1="信息技术",
                        strategy_tags=["国产替代"],
                    ),
                    route_mode=route_mode or "explicit_company",
                    expected_answer_type="boolean",
                    mentioned_companies=list(mentioned_companies or []),
                )

            processor._build_query_plan = fake_build_query_plan
            processor.get_answer_for_company = lambda *_, **__: (_ for _ in ()).throw(
                AssertionError("metadata membership questions should not enter retrieval")
            )

            result = processor.process_question(
                "在信息技术行业的科创板公司中，海川制造是否属于2024年年报提到“国产替代”的公司名单？",
                "boolean",
            )

            self.assertIs(result["final_answer"], True)
            self.assertFalse(result["route_info"]["metadata_membership"]["strict_filter_match"])
            self.assertEqual(result["route_info"]["metadata_membership"]["evaluated_doc_id"], "doc-gamma")


if __name__ == "__main__":
    unittest.main()
