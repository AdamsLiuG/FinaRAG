import json
import tempfile
import unittest
from pathlib import Path

from src.questions_processing import QuestionsProcessor


class ComparativeQuestionTests(unittest.TestCase):
    def test_process_comparative_question_uses_api_processor_and_preserves_schema(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            subset_path = Path(tmp_dir) / "subset.csv"
            subset_path.write_text(
                "sha1,company_name,cur\n"
                "sha1-alpha,Alpha Corp,USD\n"
                "sha1-beta,Beta Corp,USD\n",
                encoding="utf-8",
            )

            processor = QuestionsProcessor(subset_path=subset_path, parallel_requests=2)
            processor.api_processor.response_data = {"model": "test-model"}

            processor.api_processor.get_rephrased_questions = lambda original_question, companies, model=None: {
                "Alpha Corp": "What was Alpha Corp revenue in 2023?",
                "Beta Corp": "What was Beta Corp revenue in 2023?",
            }

            seen_schemas = []

            def fake_get_answer_for_company(company_name: str, question: str, schema: str):
                seen_schemas.append(schema)
                return {
                    "final_answer": company_name,
                    "references": [{"pdf_sha1": f"sha1-{company_name}", "page": 8}],
                    "citations": [{"source": f"sha1-{company_name}", "page": 8, "chunk_id": 1, "chunk_type": "content"}],
                    "confidence": "high",
                }

            processor.get_answer_for_company = fake_get_answer_for_company

            def fake_compare(question, rag_context, schema, model, temperature=0.0):
                self.assertEqual(schema, "comparative")
                self.assertIn("Alpha Corp", rag_context)
                self.assertIn("Beta Corp", rag_context)
                return {
                    "final_answer": "Alpha Corp",
                    "reasoning_summary": "Alpha is larger.",
                    "step_by_step_analysis": "Compared Alpha and Beta.",
                    "relevant_pages": [],
                }

            processor.api_processor.get_answer_from_rag_context = fake_compare

            result = processor.process_comparative_question(
                'Which company had higher revenue, "Alpha Corp" or "Beta Corp"?',
                ["Alpha Corp", "Beta Corp"],
                "name",
            )

            self.assertEqual(seen_schemas, ["name", "name"])
            self.assertEqual(result["final_answer"], "Alpha Corp")
            self.assertEqual(len(result["references"]), 2)
            self.assertEqual(result["confidence"], "high")

    def test_process_comparative_text_question_uses_text_generation_schema(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            subset_path = Path(tmp_dir) / "subset.csv"
            subset_path.write_text(
                "sha1,company_name,cur\n"
                "sha1-alpha,Alpha Corp,USD\n"
                "sha1-beta,Beta Corp,USD\n",
                encoding="utf-8",
            )

            processor = QuestionsProcessor(subset_path=subset_path, parallel_requests=2)
            processor.api_processor.response_data = {"model": "test-model"}
            processor.api_processor.get_rephrased_questions = lambda original_question, companies, model=None: {
                "Alpha Corp": "Summarize Alpha Corp AI strategy.",
                "Beta Corp": "Summarize Beta Corp AI strategy.",
            }

            def fake_get_answer_for_company(company_name: str, question: str, schema: str):
                self.assertEqual(schema, "text")
                return {
                    "final_answer": f"{company_name} emphasizes product upgrades.",
                    "references": [{"pdf_sha1": f"sha1-{company_name}", "page": 8}],
                    "citations": [{"source": f"sha1-{company_name}", "page": 8, "chunk_id": 1, "chunk_type": "content"}],
                    "confidence": "high",
                }

            processor.get_answer_for_company = fake_get_answer_for_company

            def fake_compare(question, rag_context, schema, model, temperature=0.0):
                self.assertEqual(schema, "text")
                return {
                    "final_answer": "两家公司都强调产品升级，但 Alpha 更强调平台化，Beta 更强调客户场景。",
                    "reasoning_summary": "两家公司战略表述有共同点也有差异。",
                    "step_by_step_analysis": "分别读取两家公司答案后综合比较。",
                    "relevant_pages": [],
                }

            processor.api_processor.get_answer_from_rag_context = fake_compare

            result = processor.process_comparative_question(
                "Alpha Corp 和 Beta Corp 的人工智能战略表述有何共同点和差异？",
                ["Alpha Corp", "Beta Corp"],
                "text",
            )

            self.assertIn("两家公司", result["final_answer"])
            self.assertIn("Alpha", result["final_answer"])
            self.assertIn("Beta", result["final_answer"])
            self.assertEqual(result["confidence"], "high")

    def test_structured_revenue_comparison_returns_company_name(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            documents_dir = root / "documents"
            documents_dir.mkdir()
            subset_path = root / "document_manifest.csv"
            subset_path.write_text(
                "doc_id,company_name,company_aliases,security_code,doc_source_type,report_title,language,currency,report_year\n"
                "doc-alpha,甲公司,甲公司|600001,600001,annual_report,甲公司2024年年报,zh,CNY,2024\n"
                "doc-beta,乙公司,乙公司|600002,600002,annual_report,乙公司2024年年报,zh,CNY,2024\n",
                encoding="utf-8",
            )

            for doc_id, company_name, value in (
                ("doc-alpha", "甲公司", "1000"),
                ("doc-beta", "乙公司", "2000"),
            ):
                payload = {
                    "metainfo": {
                        "sha1_name": doc_id,
                        "company_name": company_name,
                        "security_code": "600001" if company_name == "甲公司" else "600002",
                        "currency": "CNY",
                        "doc_source_type": "annual_report",
                        "report_year": 2024,
                        "language": "zh",
                    },
                    "content": {
                        "pages": [{"page": 10, "text": "主要会计数据和财务指标"}],
                        "chunks": [],
                        "structured_tables": [
                            {
                                "table_id": f"tbl-{doc_id}",
                                "page": 10,
                                "markdown": "营业收入 2024年报 人民币万元",
                                "cell_records": [
                                    {
                                        "table_id": f"tbl-{doc_id}",
                                        "page": 10,
                                        "row_idx": 1,
                                        "col_idx": 1,
                                        "raw_value": value,
                                        "matched_row_headers": ["营业收入"],
                                        "matched_col_headers": ["2024年报"],
                                        "unit_hint": "人民币万元",
                                        "period": "2024年报",
                                    }
                                ],
                            }
                        ],
                    },
                }
                (documents_dir / f"{doc_id}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            processor = QuestionsProcessor(
                documents_dir=documents_dir,
                subset_path=subset_path,
                parallel_requests=1,
                doc_router_enabled=True,
                numeric_grounding_enabled=True,
            )

            result = processor.process_question("甲公司和乙公司谁的营业收入更高？", "name")

            self.assertEqual(result["final_answer"], "乙公司")
            self.assertEqual(result["route_info"]["route_mode"], "numeric_comparison_rule")
            self.assertEqual(len(result["references"]), 2)
            self.assertTrue(all(citation["chunk_type"] in {"serialized_table", "table_grounding"} for citation in result["citations"]))

    def test_structured_revenue_threshold_returns_boolean(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            documents_dir = root / "documents"
            documents_dir.mkdir()
            subset_path = root / "document_manifest.csv"
            subset_path.write_text(
                "doc_id,company_name,company_aliases,security_code,doc_source_type,report_title,language,currency,report_year\n"
                "doc-alpha,甲公司,甲公司|600001,600001,annual_report,甲公司2024年年报,zh,CNY,2024\n",
                encoding="utf-8",
            )
            payload = {
                "metainfo": {
                    "sha1_name": "doc-alpha",
                    "company_name": "甲公司",
                    "security_code": "600001",
                    "currency": "CNY",
                    "doc_source_type": "annual_report",
                    "report_year": 2024,
                    "language": "zh",
                },
                "content": {
                    "pages": [{"page": 10, "text": "主要会计数据和财务指标"}],
                    "chunks": [],
                    "structured_tables": [
                        {
                            "table_id": "tbl-alpha",
                            "page": 10,
                            "markdown": "营业收入 2024年报 人民币万元",
                            "cell_records": [
                                {
                                    "table_id": "tbl-alpha",
                                    "page": 10,
                                    "row_idx": 1,
                                    "col_idx": 1,
                                    "raw_value": "15000",
                                    "matched_row_headers": ["营业收入"],
                                    "matched_col_headers": ["2024年报"],
                                    "unit_hint": "人民币万元",
                                    "period": "2024年报",
                                }
                            ],
                        }
                    ],
                },
            }
            (documents_dir / "doc-alpha.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            processor = QuestionsProcessor(
                documents_dir=documents_dir,
                subset_path=subset_path,
                parallel_requests=1,
                doc_router_enabled=True,
                numeric_grounding_enabled=True,
            )

            result = processor.process_question("甲公司2024年年报营业收入是否超过1亿元？", "boolean")

            self.assertIs(result["final_answer"], True)
            self.assertEqual(result["route_info"]["route_mode"], "numeric_threshold_rule")
            self.assertEqual(result["references"], [{"pdf_sha1": "doc-alpha", "page": 10}])

    def test_structured_revenue_rules_share_cached_grounding(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            documents_dir = root / "documents"
            documents_dir.mkdir()
            subset_path = root / "document_manifest.csv"
            subset_path.write_text(
                "doc_id,company_name,company_aliases,security_code,doc_source_type,report_title,language,currency,report_year\n"
                "doc-alpha,甲公司,甲公司|600001,600001,annual_report,甲公司2024年年报,zh,CNY,2024\n",
                encoding="utf-8",
            )
            payload = {
                "metainfo": {
                    "sha1_name": "doc-alpha",
                    "company_name": "甲公司",
                    "security_code": "600001",
                    "currency": "CNY",
                    "doc_source_type": "annual_report",
                    "report_year": 2024,
                    "language": "zh",
                },
                "content": {
                    "pages": [{"page": 10, "text": "主要会计数据和财务指标"}],
                    "chunks": [],
                    "structured_tables": [
                        {
                            "table_id": "tbl-alpha",
                            "page": 10,
                            "markdown": "主要会计数据和财务指标 营业收入 2024年报 人民币万元",
                            "cell_records": [
                                {
                                    "table_id": "tbl-alpha",
                                    "page": 10,
                                    "row_idx": 1,
                                    "col_idx": 1,
                                    "raw_value": "15000",
                                    "matched_row_headers": ["营业收入"],
                                    "matched_col_headers": ["2024年报"],
                                    "unit_hint": "人民币万元",
                                    "period": "2024年报",
                                }
                            ],
                        }
                    ],
                },
            }
            (documents_dir / "doc-alpha.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            processor = QuestionsProcessor(
                documents_dir=documents_dir,
                subset_path=subset_path,
                parallel_requests=1,
                doc_router_enabled=True,
                numeric_grounding_enabled=True,
            )
            calls = {"count": 0}
            original_ground_number_query = processor.table_grounder.ground_number_query

            def counted_ground_number_query(*args, **kwargs):
                calls["count"] += 1
                return original_ground_number_query(*args, **kwargs)

            processor.table_grounder.ground_number_query = counted_ground_number_query

            first = processor.process_question("甲公司2024年年报营业收入是否超过1亿元？", "boolean")
            second = processor.process_question("甲公司2024年年报营业收入是否超过1亿元？", "boolean")

            self.assertIs(first["final_answer"], True)
            self.assertIs(second["final_answer"], True)
            self.assertEqual(calls["count"], 1)

    def test_structured_revenue_comparison_prefers_total_over_bank_segments(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            documents_dir = root / "documents"
            documents_dir.mkdir()
            subset_path = root / "document_manifest.csv"
            subset_path.write_text(
                "doc_id,company_name,company_aliases,security_code,doc_source_type,report_title,language,currency,report_year\n"
                "doc-alpha,甲银行,甲银行|600001,600001,annual_report,甲银行2024年年报,zh,CNY,2024\n"
                "doc-beta,乙银行,乙银行|600002,600002,annual_report,乙银行2024年年报,zh,CNY,2024\n",
                encoding="utf-8",
            )

            for doc_id, company_name, total_value, segment_value in (
                ("doc-alpha", "甲银行", "3000", "9999"),
                ("doc-beta", "乙银行", "4000", "1000"),
            ):
                payload = {
                    "metainfo": {
                        "sha1_name": doc_id,
                        "company_name": company_name,
                        "security_code": "600001" if doc_id == "doc-alpha" else "600002",
                        "currency": "CNY",
                        "doc_source_type": "annual_report",
                        "report_year": 2024,
                        "language": "zh",
                    },
                    "content": {
                        "pages": [{"page": 10, "text": "主要会计数据和财务指标"}],
                        "chunks": [],
                        "structured_tables": [
                            {
                                "table_id": f"tbl-{doc_id}",
                                "page": 10,
                                "markdown": "营业收入 本集团合计 公司业务 个人业务 人民币百万元",
                                "cell_records": [
                                    {
                                        "table_id": f"tbl-{doc_id}",
                                        "page": 10,
                                        "row_idx": 1,
                                        "col_idx": 1,
                                        "raw_value": total_value,
                                        "matched_row_headers": ["营业收入"],
                                        "matched_col_headers": ["本集团合计"],
                                        "unit_hint": "人民币百万元",
                                        "period": "2024年报",
                                    },
                                    {
                                        "table_id": f"tbl-{doc_id}",
                                        "page": 10,
                                        "row_idx": 1,
                                        "col_idx": 2,
                                        "raw_value": segment_value,
                                        "matched_row_headers": ["营业收入"],
                                        "matched_col_headers": ["公司业务"],
                                        "unit_hint": "人民币百万元",
                                        "period": "2024年报",
                                    },
                                ],
                            }
                        ],
                    },
                }
                (documents_dir / f"{doc_id}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            processor = QuestionsProcessor(
                documents_dir=documents_dir,
                subset_path=subset_path,
                parallel_requests=1,
                doc_router_enabled=True,
                numeric_grounding_enabled=True,
            )

            result = processor.process_question("甲银行和乙银行谁的营业收入更高？", "name")

            self.assertEqual(result["final_answer"], "乙银行")
            values = result["route_info"]["numeric_comparison_rule"]["company_values"]
            self.assertEqual([item["value_yuan"] for item in values], [3000000000.0, 4000000000.0])


if __name__ == "__main__":
    unittest.main()
