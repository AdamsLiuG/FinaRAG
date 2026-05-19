import json
import tempfile
import unittest
from pathlib import Path

from src.query_plan import QueryPlan
from src.questions_processing import QuestionsProcessor
from src.retrieval_filters import RetrievalFilters
from src.table_grounding import TableGrounder
from src.text_normalization import parse_numeric_value


class TableGroundingTests(unittest.TestCase):
    def test_ground_number_query_returns_structured_cell_match(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            documents_dir = Path(tmp_dir)
            payload = {
                "metainfo": {
                    "sha1_name": "doc-alpha",
                    "company_name": "宁德时代",
                    "language": "zh",
                },
                "content": {
                    "structured_tables": [
                        {
                            "table_id": "tbl-1",
                            "page": 12,
                            "markdown": "主要财务数据 营业收入 2024年报 4000 人民币百万元",
                            "cell_records": [
                                {
                                    "table_id": "tbl-1",
                                    "page": 12,
                                    "row_idx": 1,
                                    "col_idx": 2,
                                    "raw_value": "4000",
                                    "normalized_value": 4000,
                                    "matched_row_headers": ["营业收入"],
                                    "matched_col_headers": ["2024年报"],
                                    "unit_hint": "人民币百万元",
                                    "period": "2024年报",
                                    "footnote_refs": [],
                                }
                            ],
                        }
                    ]
                },
            }
            (documents_dir / "doc-alpha.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            grounder = TableGrounder(documents_dir)
            result = grounder.ground_number_query(
                question="宁德时代2024年报营业收入是多少人民币？",
                retrieval_results=[
                    {
                        "page": 12,
                        "metadata": {"sha1_name": "doc-alpha"},
                    }
                ],
                filters=RetrievalFilters(
                    company_name="宁德时代",
                    currency="CNY",
                    year=2024,
                    period="2024年报",
                    question_kind="number",
                ),
                candidate_doc_ids=["doc-alpha"],
            )

            self.assertIsNotNone(result)
            self.assertEqual(result["table_id"], "tbl-1")
            self.assertEqual(result["page"], 12)
            self.assertEqual(result["normalized_value"], 4000000000.0)
            self.assertEqual(result["answer_value"], 4000000000.0)
            self.assertIsNone(result["target_unit"])
            self.assertEqual(result["matched_row_headers"], ["营业收入"])
            self.assertEqual(result["matched_col_headers"], ["2024年报"])

    def test_ground_number_query_converts_to_requested_unit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            documents_dir = Path(tmp_dir)
            payload = {
                "metainfo": {
                    "sha1_name": "doc-alpha",
                    "company_name": "宁德时代",
                    "language": "zh",
                },
                "content": {
                    "structured_tables": [
                        {
                            "table_id": "tbl-1",
                            "page": 12,
                            "markdown": "主要财务数据 营业收入 2024年报 4000 人民币百万元",
                            "cell_records": [
                                {
                                    "table_id": "tbl-1",
                                    "page": 12,
                                    "row_idx": 1,
                                    "col_idx": 2,
                                    "raw_value": "4000",
                                    "matched_row_headers": ["营业收入"],
                                    "matched_col_headers": ["2024年报"],
                                    "unit_hint": "人民币百万元",
                                    "period": "2024年报",
                                }
                            ],
                        }
                    ]
                },
            }
            (documents_dir / "doc-alpha.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            result = TableGrounder(documents_dir).ground_number_query(
                question="宁德时代2024年报营业收入是多少亿元？",
                retrieval_results=[],
                filters=RetrievalFilters(currency="CNY", year=2024, period="2024年报", question_kind="number"),
                candidate_doc_ids=["doc-alpha"],
            )

            self.assertIsNotNone(result)
            self.assertEqual(result["normalized_value"], 4000000000.0)
            self.assertEqual(result["target_unit"], "亿元")
            self.assertEqual(result["answer_value"], 40.0)

    def test_parse_numeric_value_prefers_long_unit_match(self):
        self.assertEqual(parse_numeric_value("136,290", unit_hint="人民币百万元"), 136290000000.0)

    def test_ground_number_query_inherits_table_wide_baiwan_unit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            documents_dir = Path(tmp_dir)
            payload = {
                "metainfo": {
                    "sha1_name": "doc-bank",
                    "company_name": "民生银行",
                    "language": "zh",
                },
                "content": {
                    "structured_tables": [
                        {
                            "table_id": "tbl-bank",
                            "page": 16,
                            "markdown": "一、主要会计数据和财务指标 经营业绩 （人民币百万元） 营业收入 136,290",
                            "cell_records": [
                                {
                                    "table_id": "tbl-bank",
                                    "page": 16,
                                    "row_idx": 3,
                                    "col_idx": 1,
                                    "raw_value": "136,290",
                                    "matched_row_headers": ["营业收入"],
                                    "matched_col_headers": ["2024 年"],
                                    "unit_hint": "万元",
                                    "period": "2024年报",
                                }
                            ],
                        }
                    ]
                },
            }
            (documents_dir / "doc-bank.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            result = TableGrounder(documents_dir).ground_number_query(
                question="民生银行2024年年报中的营业收入是多少元？",
                retrieval_results=[],
                filters=RetrievalFilters(company_name="民生银行", year=2024, question_kind="number"),
                candidate_doc_ids=["doc-bank"],
            )

            self.assertIsNotNone(result)
            self.assertEqual(result["unit"], "人民币百万元")
            self.assertEqual(result["normalized_value"], 136290000000.0)

    def test_ground_number_query_inherits_bank_unit_from_page_context(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            documents_dir = Path(tmp_dir)
            payload = {
                "metainfo": {
                    "sha1_name": "doc-bank",
                    "company_name": "民生银行",
                    "language": "zh",
                },
                "content": {
                    "pages": [
                        {
                            "page": 16,
                            "text": "一、主要会计数据和财务指标。除特别注明外，金额单位为⼈民币百万元。",
                        }
                    ],
                    "chunks": [
                        {
                            "page": 16,
                            "section_title": "公司简介和主要财务指标",
                            "text": "经营业绩表金额单位为人民币百万元。",
                        }
                    ],
                    "structured_tables": [
                        {
                            "table_id": "tbl-bank",
                            "page": 16,
                            "markdown": "经营业绩 营业收入 136,290",
                            "cell_records": [
                                {
                                    "table_id": "tbl-bank",
                                    "page": 16,
                                    "row_idx": 3,
                                    "col_idx": 1,
                                    "raw_value": "136,290",
                                    "matched_row_headers": ["营业收入"],
                                    "matched_col_headers": ["2024 年"],
                                    "unit_hint": "万元",
                                    "period": "2024年报",
                                }
                            ],
                        }
                    ],
                },
            }
            (documents_dir / "doc-bank.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            result = TableGrounder(documents_dir).ground_number_query(
                question="民生银行2024年年报中的营业收入是多少元？",
                retrieval_results=[],
                filters=RetrievalFilters(company_name="民生银行", year=2024, question_kind="number"),
                candidate_doc_ids=["doc-bank"],
            )

            self.assertIsNotNone(result)
            self.assertEqual(result["unit"], "人民币百万元")
            self.assertEqual(result["normalized_value"], 136290000000.0)

    def test_ground_number_query_returns_same_value_supporting_matches(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            documents_dir = Path(tmp_dir)
            payload = {
                "metainfo": {
                    "sha1_name": "doc-alpha",
                    "company_name": "宁德时代",
                    "language": "zh",
                },
                "content": {
                    "pages": [
                        {"page": 12, "text": "主要会计数据和财务指标。"},
                        {"page": 88, "text": "合并利润表。"},
                    ],
                    "structured_tables": [
                        {
                            "table_id": "tbl-main",
                            "page": 12,
                            "markdown": "主要会计数据和财务指标 营业收入 2024年 4000 人民币百万元",
                            "cell_records": [
                                {
                                    "table_id": "tbl-main",
                                    "page": 12,
                                    "row_idx": 1,
                                    "col_idx": 1,
                                    "raw_value": "4000",
                                    "matched_row_headers": ["营业收入"],
                                    "matched_col_headers": ["2024年"],
                                    "unit_hint": "人民币百万元",
                                    "period": "2024年报",
                                }
                            ],
                        },
                        {
                            "table_id": "tbl-profit",
                            "page": 88,
                            "markdown": "合并利润表 营业总收入 2024年 4000 人民币百万元",
                            "cell_records": [
                                {
                                    "table_id": "tbl-profit",
                                    "page": 88,
                                    "row_idx": 1,
                                    "col_idx": 1,
                                    "raw_value": "4000",
                                    "matched_row_headers": ["营业总收入"],
                                    "matched_col_headers": ["2024年"],
                                    "unit_hint": "人民币百万元",
                                    "period": "2024年报",
                                }
                            ],
                        },
                    ],
                },
            }
            (documents_dir / "doc-alpha.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            result = TableGrounder(documents_dir).ground_number_query(
                question="宁德时代2024年年报中的营业收入是多少元？",
                retrieval_results=[],
                filters=RetrievalFilters(company_name="宁德时代", year=2024, question_kind="number"),
                candidate_doc_ids=["doc-alpha"],
            )

            self.assertIsNotNone(result)
            self.assertEqual(result["normalized_value"], 4000000000.0)
            self.assertTrue(result.get("supporting_matches"))
            self.assertEqual({match["page"] for match in result["supporting_matches"]}, {88})

    def test_ground_number_query_materializes_confirmed_logical_table_for_missing_unit_and_headers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            documents_dir = Path(tmp_dir)
            payload = {
                "metainfo": {
                    "sha1_name": "doc-alpha",
                    "company_name": "宁德时代",
                    "language": "zh",
                },
                "content": {
                    "pages": [{"page": 10, "text": "主要会计数据和财务指标。"}, {"page": 11, "text": "续表。"}],
                    "chunks": [],
                    "logical_tables": [
                        {
                            "logical_table_id": "lt-doc-alpha-10-11",
                            "head_table_id": "tbl-head",
                            "member_table_ids": ["tbl-head", "tbl-tail"],
                            "member_pages": [10, 11],
                            "page_span": [10, 11],
                            "merge_confidence": 0.86,
                            "merge_state": "confirmed",
                            "materializable": True,
                        }
                    ],
                    "structured_tables": [
                        {
                            "table_id": "tbl-head",
                            "page": 10,
                            "markdown": "主要会计数据 单位：人民币百万元",
                            "unit_hint": "人民币百万元",
                            "period": "2024年报",
                            "section_title": "主要会计数据和财务指标",
                            "section_name": "主要会计数据和财务指标",
                            "report_section": "主要会计数据和财务指标",
                            "logical_table_id": "lt-doc-alpha-10-11",
                            "continuation_of": None,
                            "logical_role": "head",
                            "page_span": [10, 11],
                            "merge_confidence": 0.86,
                            "merge_state": "confirmed",
                            "col_headers_by_col": {"1": ["2024年报"]},
                            "row_headers_by_row": {"1": ["营业成本"]},
                            "cell_records": [
                                {
                                    "table_id": "tbl-head",
                                    "page": 10,
                                    "row_idx": 1,
                                    "col_idx": 1,
                                    "raw_value": "3000",
                                    "matched_row_headers": ["营业成本"],
                                    "matched_col_headers": ["2024年报"],
                                    "unit_hint": "人民币百万元",
                                    "period": "2024年报",
                                    "footnote_refs": [],
                                }
                            ],
                        },
                        {
                            "table_id": "tbl-tail",
                            "page": 11,
                            "markdown": "主要会计数据（续表）",
                            "unit_hint": None,
                            "period": None,
                            "section_title": "主要会计数据和财务指标",
                            "section_name": "主要会计数据和财务指标",
                            "report_section": "主要会计数据和财务指标",
                            "logical_table_id": "lt-doc-alpha-10-11",
                            "continuation_of": "tbl-head",
                            "logical_role": "tail",
                            "page_span": [10, 11],
                            "merge_confidence": 0.86,
                            "merge_state": "confirmed",
                            "col_headers_by_col": {},
                            "row_headers_by_row": {"1": ["营业收入"]},
                            "cell_records": [
                                {
                                    "table_id": "tbl-tail",
                                    "page": 11,
                                    "row_idx": 1,
                                    "col_idx": 1,
                                    "raw_value": "4000",
                                    "matched_row_headers": ["营业收入"],
                                    "matched_col_headers": [],
                                    "unit_hint": None,
                                    "period": None,
                                    "footnote_refs": [],
                                }
                            ],
                        },
                    ],
                },
            }
            (documents_dir / "doc-alpha.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            result = TableGrounder(documents_dir).ground_number_query(
                question="宁德时代2024年报营业收入是多少元？",
                retrieval_results=[{"page": 11, "metadata": {"sha1_name": "doc-alpha"}}],
                filters=RetrievalFilters(company_name="宁德时代", year=2024, period="2024年报", question_kind="number"),
                candidate_doc_ids=["doc-alpha"],
            )

            self.assertIsNotNone(result)
            self.assertEqual(result["table_id"], "tbl-tail")
            self.assertEqual(result["page"], 11)
            self.assertEqual(result["logical_table_id"], "lt-doc-alpha-10-11")
            self.assertTrue(result["logical_table_materialized"])
            self.assertEqual(result["matched_col_headers"], ["2024年报"])
            self.assertEqual(result["unit"], "人民币百万元")
            self.assertEqual(result["normalized_value"], 4000000000.0)

    def test_ground_number_query_rejects_revenue_growth_percent_column(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            documents_dir = Path(tmp_dir)
            payload = {
                "metainfo": {
                    "sha1_name": "doc-auto",
                    "company_name": "江淮汽车",
                    "language": "zh",
                },
                "content": {
                    "structured_tables": [
                        {
                            "table_id": "tbl-auto",
                            "page": 7,
                            "markdown": "主要会计数据 单位：元 币种：人民币 营业收入 2024年 本期比上年同期增减(%)",
                            "cell_records": [
                                {
                                    "table_id": "tbl-auto",
                                    "page": 7,
                                    "row_idx": 1,
                                    "col_idx": 3,
                                    "raw_value": "-6.28",
                                    "matched_row_headers": ["营业收入"],
                                    "matched_col_headers": ["本期比上年同期 增减(%)"],
                                    "unit_hint": "%",
                                    "period": "2024年报",
                                },
                                {
                                    "table_id": "tbl-auto",
                                    "page": 7,
                                    "row_idx": 1,
                                    "col_idx": 1,
                                    "raw_value": "42,115,891,831.00",
                                    "matched_row_headers": ["营业收入"],
                                    "matched_col_headers": ["2024年"],
                                    "unit_hint": "元",
                                    "period": "2024年报",
                                },
                            ],
                        }
                    ]
                },
            }
            (documents_dir / "doc-auto.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            result = TableGrounder(documents_dir).ground_number_query(
                question="江淮汽车2024年年报中的营业收入是多少万元？",
                retrieval_results=[],
                filters=RetrievalFilters(company_name="江淮汽车", year=2024, question_kind="number"),
                candidate_doc_ids=["doc-auto"],
            )

            self.assertIsNotNone(result)
            self.assertEqual(result["col_idx"], 1)
            self.assertEqual(result["answer_value"], 4211589.1831)

    def test_ground_number_query_respects_allowed_doc_ids(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            documents_dir = Path(tmp_dir)
            for doc_id, value in (("doc-wrong", "9999"), ("doc-right", "1234")):
                payload = {
                    "metainfo": {
                        "sha1_name": doc_id,
                        "company_name": "同名公司",
                        "language": "zh",
                    },
                    "content": {
                        "structured_tables": [
                            {
                                "table_id": f"tbl-{doc_id}",
                                "page": 8,
                                "markdown": "营业收入 2024年报 人民币万元",
                                "cell_records": [
                                    {
                                        "table_id": f"tbl-{doc_id}",
                                        "page": 8,
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
                        ]
                    },
                }
                (documents_dir / f"{doc_id}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            result = TableGrounder(documents_dir).ground_number_query(
                question="同名公司2024年报营业收入是多少？",
                retrieval_results=[],
                filters=RetrievalFilters(year=2024, period="2024年报", question_kind="number"),
                candidate_doc_ids=["doc-wrong", "doc-right"],
                allowed_doc_ids=["doc-right"],
            )

            self.assertIsNotNone(result)
            self.assertEqual(result["source_doc_id"], "doc-right")
            self.assertEqual(result["normalized_value"], 12340000.0)

    def test_ground_number_query_prefers_exact_revenue_and_rejects_noise(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            documents_dir = Path(tmp_dir)
            payload = {
                "metainfo": {
                    "sha1_name": "doc-alpha",
                    "company_name": "宁德时代",
                    "language": "zh",
                },
                "content": {
                    "structured_tables": [
                        {
                            "table_id": "tbl-1",
                            "page": 12,
                            "markdown": "主要财务数据 营业收入 扣除与主营业务无关后的营业收入 销售费用占营业收入比例",
                            "cell_records": [
                                {
                                    "table_id": "tbl-1",
                                    "page": 12,
                                    "row_idx": 1,
                                    "col_idx": 1,
                                    "raw_value": "公司于2024年持续推进主营业务，营业收入同比变动较大，详见管理层讨论分析。",
                                    "matched_row_headers": ["营业收入"],
                                    "matched_col_headers": ["2024年报"],
                                    "unit_hint": "人民币万元",
                                    "period": "2024年报",
                                },
                                {
                                    "table_id": "tbl-1",
                                    "page": 12,
                                    "row_idx": 2,
                                    "col_idx": 1,
                                    "raw_value": "88.8",
                                    "matched_row_headers": ["销售费用占营业收入比例"],
                                    "matched_col_headers": ["2024年报"],
                                    "unit_hint": "%",
                                    "period": "2024年报",
                                },
                                {
                                    "table_id": "tbl-1",
                                    "page": 12,
                                    "row_idx": 3,
                                    "col_idx": 1,
                                    "raw_value": "5000",
                                    "matched_row_headers": ["扣除与主营业务无关的业务收入后的营业收入"],
                                    "matched_col_headers": ["2024年报"],
                                    "unit_hint": "人民币万元",
                                    "period": "2024年报",
                                },
                                {
                                    "table_id": "tbl-1",
                                    "page": 12,
                                    "row_idx": 4,
                                    "col_idx": 1,
                                    "raw_value": "4000",
                                    "matched_row_headers": ["营业收入"],
                                    "matched_col_headers": ["2024年报"],
                                    "unit_hint": "人民币万元",
                                    "period": "2024年报",
                                },
                            ],
                        }
                    ]
                },
            }
            (documents_dir / "doc-alpha.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            result = TableGrounder(documents_dir).ground_number_query(
                question="宁德时代2024年报营业收入是多少？",
                retrieval_results=[],
                filters=RetrievalFilters(company_name="宁德时代", year=2024, period="2024年报", question_kind="number"),
                candidate_doc_ids=["doc-alpha"],
            )

            self.assertIsNotNone(result)
            self.assertEqual(result["row_idx"], 4)
            self.assertEqual(result["normalized_value"], 40000000.0)

    def test_questions_processor_answers_number_from_table_grounding_without_llm(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            documents_dir = Path(tmp_dir) / "documents"
            documents_dir.mkdir()
            payload = {
                "metainfo": {
                    "sha1_name": "doc-alpha",
                    "company_name": "宁德时代",
                    "security_code": "300750",
                    "currency": "CNY",
                    "doc_source_type": "annual_report",
                    "report_year": 2024,
                    "language": "zh",
                },
                "content": {
                    "structured_tables": [
                        {
                            "table_id": "tbl-1",
                            "page": 12,
                            "markdown": "主要财务数据 营业收入 2024年报 4000 人民币百万元",
                            "cell_records": [
                                {
                                    "table_id": "tbl-1",
                                    "page": 12,
                                    "row_idx": 1,
                                    "col_idx": 2,
                                    "raw_value": "4000",
                                    "matched_row_headers": ["营业收入"],
                                    "matched_col_headers": ["2024年报"],
                                    "unit_hint": "人民币百万元",
                                    "period": "2024年报",
                                }
                            ],
                        },
                        {
                            "table_id": "tbl-2",
                            "page": 88,
                            "markdown": "合并利润表 营业总收入 2024年报 4000 人民币百万元",
                            "cell_records": [
                                {
                                    "table_id": "tbl-2",
                                    "page": 88,
                                    "row_idx": 1,
                                    "col_idx": 2,
                                    "raw_value": "4000",
                                    "matched_row_headers": ["营业总收入"],
                                    "matched_col_headers": ["2024年报"],
                                    "unit_hint": "人民币百万元",
                                    "period": "2024年报",
                                }
                            ],
                        }
                    ],
                    "pages": [{"page": 12, "text": "主要财务数据表"}, {"page": 88, "text": "合并利润表"}],
                    "chunks": [],
                },
            }
            (documents_dir / "doc-alpha.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            processor = QuestionsProcessor(
                documents_dir=documents_dir,
                parallel_requests=1,
                numeric_grounding_enabled=True,
            )
            processor._build_retriever = lambda: (object(), "hybrid")
            processor._run_retrieval = lambda *_, **__: []
            processor.api_processor.get_answer_from_rag_context = lambda **__: (_ for _ in ()).throw(
                AssertionError("table-grounded number questions should not call the LLM")
            )
            query_plan = QueryPlan(
                original_query="宁德时代2024年报营业收入是多少亿元？",
                normalized_query="宁德时代2024年报营业收入是多少亿元？",
                search_queries=["宁德时代2024年报营业收入是多少亿元？"],
                filters=RetrievalFilters(
                    company_name="宁德时代",
                    currency="CNY",
                    year=2024,
                    period="2024年报",
                    doc_source_type="annual_report",
                    candidate_doc_ids=["doc-alpha"],
                    question_kind="number",
                ),
                route_mode="explicit_company",
                expected_answer_type="numeric",
                mentioned_companies=["宁德时代"],
            )

            answer = processor.get_answer_for_company(
                company_name="宁德时代",
                question="宁德时代2024年报营业收入是多少亿元？",
                schema="number",
                query_plan=query_plan,
                route_info={
                    "route_mode": "explicit_company",
                    "selected_company": "宁德时代",
                    "candidate_doc_ids": ["doc-alpha"],
                    "selected_report": {"sha1": "doc-alpha"},
                },
            )

            self.assertEqual(answer["final_answer"], 40)
            self.assertEqual(answer["table_grounding_result"]["target_unit"], "亿元")
            self.assertEqual(
                answer["references"],
                [{"pdf_sha1": "doc-alpha", "page": 12}, {"pdf_sha1": "doc-alpha", "page": 88}],
            )
            self.assertEqual(answer["relevant_pages"], [12, 88])
            self.assertIn("table_grounding", {citation["chunk_type"] for citation in answer["citations"]})
            self.assertIn("table_support", {citation["chunk_type"] for citation in answer["citations"]})

    def test_legal_representative_rule_accepts_company_leader_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            documents_dir = Path(tmp_dir) / "documents"
            documents_dir.mkdir()
            payload = {
                "metainfo": {
                    "sha1_name": "doc-alpha",
                    "company_name": "国网信通",
                    "security_code": "600131",
                    "currency": "CNY",
                    "doc_source_type": "annual_report",
                    "report_year": 2024,
                    "language": "zh",
                },
                "content": {
                    "pages": [
                        {"page": 2, "text": "张三曾任法定代表人，后辞去法定代表人职务。"},
                        {"page": 97, "text": "公司负责人：王奔\n主管会计工作负责人：向杰"},
                    ],
                    "chunks": [],
                    "structured_tables": [],
                },
            }
            (documents_dir / "doc-alpha.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            processor = QuestionsProcessor(documents_dir=documents_dir, parallel_requests=1)
            query_plan = QueryPlan(
                original_query="国网信通2024年年报中的法定代表人是谁？",
                normalized_query="国网信通2024年年报中的法定代表人是谁？",
                search_queries=["国网信通2024年年报中的法定代表人是谁？"],
                filters=RetrievalFilters(company_name="国网信通", year=2024, doc_source_type="annual_report"),
                route_mode="explicit_company",
                expected_answer_type="entity",
                mentioned_companies=["国网信通"],
            )

            answer = processor.get_answer_for_company(
                company_name="国网信通",
                question="国网信通2024年年报中的法定代表人是谁？",
                schema="name",
                query_plan=query_plan,
                route_info={
                    "route_mode": "explicit_company",
                    "selected_company": "国网信通",
                    "candidate_doc_ids": ["doc-alpha"],
                    "selected_report": {"sha1": "doc-alpha"},
                },
            )

            self.assertEqual(answer["final_answer"], "王奔")
            self.assertEqual(answer["references"], [{"pdf_sha1": "doc-alpha", "page": 97}])
            self.assertEqual(answer["route_info"]["route_mode"], "legal_representative_rule")

    def test_cash_dividend_positive_rule_uses_metadata_matched_report(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            documents_dir = Path(tmp_dir) / "documents"
            documents_dir.mkdir()
            payload = {
                "metainfo": {
                    "sha1_name": "doc-alpha",
                    "company_name": "香江控股",
                    "security_code": "600162",
                    "currency": "CNY",
                    "doc_source_type": "annual_report",
                    "report_year": 2024,
                    "language": "zh",
                },
                "content": {
                    "pages": [
                        {"page": 2, "text": "董事会决议通过的本报告期利润分配预案：不派发现金红利，不送红股。"},
                    ],
                    "chunks": [],
                    "structured_tables": [],
                },
            }
            (documents_dir / "doc-alpha.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            processor = QuestionsProcessor(documents_dir=documents_dir, parallel_requests=1)
            query_plan = QueryPlan(
                original_query="香江控股2024年年度利润分配预案中是否提到现金分红？",
                normalized_query="香江控股2024年年度利润分配预案中是否提到现金分红？",
                search_queries=["香江控股2024年年度利润分配预案中是否提到现金分红？"],
                filters=RetrievalFilters(company_name="香江控股", year=2024, doc_source_type="annual_report"),
                route_mode="explicit_company",
                expected_answer_type="boolean",
                mentioned_companies=["香江控股"],
            )

            answer = processor.get_answer_for_company(
                company_name="香江控股",
                question="香江控股2024年年度利润分配预案中是否提到现金分红？",
                schema="boolean",
                query_plan=query_plan,
                route_info={
                    "route_mode": "explicit_company",
                    "selected_company": "香江控股",
                    "candidate_doc_ids": ["doc-alpha"],
                    "selected_report": {"sha1": "doc-alpha"},
                },
            )

            self.assertIs(answer["final_answer"], True)
            self.assertEqual(answer["references"], [{"pdf_sha1": "doc-alpha", "page": 2}])
            self.assertEqual(answer["route_info"]["route_mode"], "cash_dividend_rule")

    def test_cash_dividend_rule_falls_through_without_positive_evidence(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            documents_dir = Path(tmp_dir) / "documents"
            documents_dir.mkdir()
            payload = {
                "metainfo": {
                    "sha1_name": "doc-alpha",
                    "company_name": "香江控股",
                    "security_code": "600162",
                    "currency": "CNY",
                    "doc_source_type": "annual_report",
                    "report_year": 2024,
                    "language": "zh",
                },
                "content": {
                    "pages": [{"page": 2, "text": "董事会审议年度利润分配预案。"}],
                    "chunks": [],
                    "structured_tables": [],
                },
            }
            (documents_dir / "doc-alpha.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            processor = QuestionsProcessor(documents_dir=documents_dir, parallel_requests=1)
            processor._build_retriever = lambda: (object(), "hybrid")
            processor._run_retrieval = lambda *_, **__: [
                {
                    "page": 2,
                    "text": "董事会审议年度利润分配预案。",
                    "metadata": {
                        "sha1_name": "doc-alpha",
                        "company_name": "香江控股",
                        "currency": "CNY",
                        "report_year": 2024,
                        "doc_source_type": "annual_report",
                        "chunk_type": "content",
                        "node_type": "child",
                    },
                    "combined_score": 0.9,
                    "matched_tags": [],
                    "retrieval_sources": ["test"],
                }
            ]
            processor.api_processor.get_answer_from_rag_context = lambda **__: {
                "final_answer": "N/A",
                "relevant_pages": [],
                "reasoning_summary": "stubbed",
                "step_by_step_analysis": "",
            }
            query_plan = QueryPlan(
                original_query="香江控股2024年年度利润分配预案中是否提到现金分红？",
                normalized_query="香江控股2024年年度利润分配预案中是否提到现金分红？",
                search_queries=["香江控股2024年年度利润分配预案中是否提到现金分红？"],
                filters=RetrievalFilters(company_name="香江控股", year=2024, doc_source_type="annual_report"),
                route_mode="explicit_company",
                expected_answer_type="boolean",
                mentioned_companies=["香江控股"],
            )

            answer = processor.get_answer_for_company(
                company_name="香江控股",
                question="香江控股2024年年度利润分配预案中是否提到现金分红？",
                schema="boolean",
                query_plan=query_plan,
                route_info={
                    "route_mode": "explicit_company",
                    "selected_company": "香江控股",
                    "candidate_doc_ids": ["doc-alpha"],
                    "selected_report": {"sha1": "doc-alpha"},
                },
            )

            self.assertEqual(answer["final_answer"], "N/A")
            self.assertNotEqual(answer["route_info"].get("route_mode"), "cash_dividend_rule")


if __name__ == "__main__":
    unittest.main()
