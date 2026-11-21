"""Unit, regression, and security tests for FinancialTableExtractor, SQLStore, and StructuredQueryRouter."""

import tempfile
import unittest
from pathlib import Path

from src.structured.fin_table_extractor import FinancialTableExtractor
from src.structured.query_router import StructuredQueryRouter
from src.structured.sql_store import FinancialSQLStore, StructuredQueryError, StructuredSecurityError


def _sample_reports():
    spdb = {
        "metainfo": {
            "doc_id": "600000_2024_20250329",
            "sha1_name": "600000_2024_20250329",
            "company_name": "浦发银行",
            "stock_code": "600000",
            "industry_l1": "银行",
            "report_year": 2024,
            "currency": "CNY",
        },
        "content": {
            "pages": [
                {"page": 19, "text": "## 主要会计数据和财务指标\n| 单位：人民币百万元 | 2024 年 | 2023 年 | 本期比上年同期增减（%） |\n"}
            ],
            "structured_tables": [
                {
                    "table_id": "11",
                    "page": 19,
                    "markdown": (
                        "| 单位：人民币百万元 | 2024 年 | 2023 年 | 本期比上年同期增减（%） |\n"
                        "|---|---|---|---|\n"
                        "| 主要会计数据 | | | |\n"
                        "| 营业收入 | 170,748 | 173,434 | -1.55 |\n"
                        "| 归属于母公司股东的净利润 | 45,257 | 36,702 | 23.31 |\n"
                        "| 基本每股收益 | 1.36 | 1.07 | 27.10 |\n"
                    ),
                    # Simulate real noisy cell_records from tables_serialization
                    "cell_records": [
                        {
                            "table_id": 11,
                            "page": 19,
                            "row_idx": 2,
                            "col_idx": 1,
                            "raw_value": "170,748",
                            "normalized_value": 1707480000.0,
                            "matched_row_headers": ["营业收入"],
                            "matched_col_headers": ["2024 年"],
                            "unit_hint": "万元",
                        }
                    ],
                }
            ],
        },
    }

    jingneng = {
        "metainfo": {
            "doc_id": "600791_2024_20250430",
            "sha1_name": "600791_2024_20250430",
            "company_name": "京能置业",
            "stock_code": "600791",
            "industry_l1": "房地产",
            "report_year": 2024,
            "currency": "CNY",
        },
        "content": {
            "pages": [
                {
                    "page": 5,
                    "text": (
                        "## 七、 近三年主要会计数据和财务指标\n"
                        "单位：元 币种：人民币\n"
                        "| 主要会计数据 | 2024 年 | 2023 年 | 本期比上年同期增减 (%) |\n"
                    ),
                }
            ],
            "structured_tables": [
                {
                    "table_id": "9",
                    "page": 5,
                    "markdown": (
                        "| 主要会计数据 | 2024 年 | 2023 年 | 本期比上年同期增减 (%) |\n"
                        "|---|---|---|---|\n"
                        "| 营业收入 | 5,458,594,505.42 | 4,535,104,206.81 | 20.36 |\n"
                        "| 归属于上市公司股东的净利润 | -109,126,892.90 | -407,033,087.82 | 不适用 |\n"
                    ),
                }
            ],
        },
    }
    return [spdb, jingneng]


class StructuredSQLTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        self.chunked_dir = self.root / "chunked_reports"
        self.chunked_dir.mkdir(parents=True, exist_ok=True)
        import json

        for report in _sample_reports():
            doc_id = report["metainfo"]["doc_id"]
            (self.chunked_dir / f"{doc_id}.json").write_text(
                json.dumps(report, ensure_ascii=False),
                encoding="utf-8",
            )
        self.db_path = self.root / "finarag_facts.sqlite"
        self.extractor = FinancialTableExtractor()
        self.stats = self.extractor.build_sqlite_db(self.chunked_dir, self.db_path)
        self.store = FinancialSQLStore(self.db_path, allowed_base_dir=self.root)
        self.router = StructuredQueryRouter(self.store)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_extractor_fixes_unit_and_pct_bugs(self):
        self.assertEqual(self.stats["reports_count"], 2)

        spdb_rev = self.store.query_company_metric("浦发银行", metric_atom_id="MET-004", period="2024")
        self.assertTrue(spdb_rev)
        # Markdown header "单位：人民币百万元" MUST override noisy cell_records unit_hint="万元"
        self.assertEqual(spdb_rev[0]["raw_unit"], "百万元")
        self.assertAlmostEqual(spdb_rev[0]["normalized_cny"], 170748 * 1e6, places=2)
        self.assertEqual(spdb_rev[0]["source_page"], 19)

        # Verify percentage & per-share columns are stored as "%" with normalized_cny=None
        all_spdb = self.store.query_company_metric(
            "浦发银行", metric_atom_id="MET-004", period="2024", exclude_pct=False
        )
        pct_rows = [r for r in all_spdb if r["raw_unit"] == "%"]
        self.assertEqual(len(pct_rows), 1)
        self.assertAlmostEqual(pct_rows[0]["raw_value"], -1.55, places=2)
        self.assertIsNone(pct_rows[0]["normalized_cny"])

        # Verify page context "单位：元 币种：人民币" detection for 京能置业
        jn_rev = self.store.query_company_metric("京能置业", metric_atom_id="MET-004", period="2024")
        self.assertTrue(jn_rev)
        self.assertEqual(jn_rev[0]["raw_unit"], "元")
        self.assertAlmostEqual(jn_rev[0]["normalized_cny"], 5458594505.42, places=2)

    def test_sql_store_security_readonly_and_path_traversal(self):
        with self.assertRaises(StructuredSecurityError):
            FinancialSQLStore(self.root / "../outside.sqlite", allowed_base_dir=self.root)

        sibling_trap = self.root.parent / f"{self.root.name}-malicious" / "bad.sqlite"
        with self.assertRaises(StructuredSecurityError):
            FinancialSQLStore(sibling_trap, allowed_base_dir=self.root)

        bad_queries = [
            "DROP TABLE reports",
            "DELETE FROM financial_facts",
            "INSERT INTO reports(doc_id) VALUES('x')",
            "UPDATE reports SET company_name='hack'",
            "PRAGMA table_info(reports)",
            "ATTACH DATABASE '/tmp/x.db' AS x",
            "SELECT * FROM sqlite_master",
            "SELECT * FROM reports; DROP TABLE reports",
            "SELECT * FROM reports -- comment",
        ]
        for sql in bad_queries:
            with self.subTest(sql=sql):
                with self.assertRaises(StructuredSecurityError):
                    self.store.execute_readonly_sql(sql)

        with self.assertRaises(StructuredSecurityError):
            self.store.query_threshold_companies("MET-004", "2024", 1e8, operator="= 1 OR 1=1 --")

        # Raw SQLite errors must be sanitized without leaking column/table internals
        with self.assertRaises(StructuredQueryError) as ctx:
            self.store.execute_readonly_sql("SELECT nonexistent_secret_col FROM reports")
        self.assertNotIn("nonexistent_secret_col", str(ctx.exception))

        ok = self.store.execute_readonly_sql("SELECT company_name FROM reports WHERE stock_code = ?", ("600000",))
        self.assertEqual(ok, [{"company_name": "浦发银行"}])

    def test_structured_router_yoy_compare_ranking_threshold_and_fail_closed(self):
        # 1. Single-company YoY growth
        yoy = self.router.try_answer(
            question="浦发银行2024年营业收入同比增速是多少？",
            schema="number",
            companies=["浦发银行"],
        )
        self.assertIsNotNone(yoy)
        self.assertAlmostEqual(yoy["final_answer"], -1.55, places=2)
        self.assertEqual(yoy["relevant_pages"], [19])

        # 2. Cross-company comparison
        cmp_ans = self.router.try_answer(
            question="浦发银行和京能置业2024年谁的营业收入更高？",
            schema="name",
            companies=["浦发银行", "京能置业"],
        )
        self.assertIsNotNone(cmp_ans)
        self.assertEqual(cmp_ans["final_answer"], "浦发银行")

        # 3. Cross-company ranking
        rank_ans = self.router.try_answer(
            question="2024年营业收入排名前2的公司是哪些？",
            schema="names",
            companies=[],
        )
        self.assertIsNotNone(rank_ans)
        self.assertEqual(rank_ans["final_answer"], ["浦发银行", "京能置业"])

        # 4. Single-company & cross-company threshold
        single_th = self.router.try_answer(
            question="浦发银行2024年营业收入是否超过1000亿元？",
            schema="boolean",
            companies=["浦发银行"],
        )
        self.assertIsNotNone(single_th)
        self.assertTrue(single_th["final_answer"])

        cross_th = self.router.try_answer(
            question="2024年营业收入超过1000亿元的有哪些公司？",
            schema="names",
            companies=[],
        )
        self.assertIsNotNone(cross_th)
        self.assertEqual(cross_th["final_answer"], ["浦发银行"])

        # 5. Fail-closed on unsupported/ambiguous queries
        self.assertIsNone(
            self.router.try_answer(
                question="浦发银行2024年的核心竞争力战略是什么？",
                schema="text",
                companies=["浦发银行"],
            )
        )

    def test_csrc_atom_aliases_and_ratio_deduction_exclusions(self):
        self.assertEqual(
            self.extractor.map_metric_atom("归属于上市公司股东的扣除非经常性损益的净利润"),
            "MET-001",
        )
        self.assertEqual(
            self.extractor.map_metric_atom("归属于上市公司股东的净利润"),
            "MET-002",
        )
        self.assertEqual(
            self.extractor.map_metric_atom("经营活动产生的现金流量净额"),
            "MET-003",
        )
        self.assertIsNone(
            self.extractor.map_metric_atom("利息净收入比营业收入")
        )
        self.assertIsNone(
            self.extractor.map_metric_atom("扣除与主营业务无关的业务收入和不具备商业实质的收入后的营业收入")
        )

        sample_report = {
            "metainfo": {
                "doc_id": "test_csrc_1",
                "sha1_name": "test_csrc_1",
                "company_name": "测试股份",
                "stock_code": "600999",
                "report_year": 2024,
                "currency": "CNY",
            },
            "content": {
                "pages": [{"page": 8, "text": "单位：百万元"}],
                "structured_tables": [
                    {
                        "table_id": "2",
                        "page": 8,
                        "markdown": (
                            "| 单位：人民币百万元 | 2024 年 |\n"
                            "|---|---|\n"
                            "| 营业收入 | 100,000 |\n"
                            "| 扣除与主营业务无关的业务收入和不具备商业实质的收入后的营业收入 | 98,500 |\n"
                            "| 利息净收入比营业收入 | 67.18 |\n"
                            "| 归属于上市公司股东的净利润 | 12,345 |\n"
                            "| 归属于上市公司股东的扣除非经常性损益的净利润 | 11,111 |\n"
                            "| 经营活动产生的现金流量净额 | 23,456 |\n"
                        ),
                    }
                ],
            },
        }
        _, facts = self.extractor.extract_from_report_dict(sample_report)
        by_name = {f["metric_raw"]: f for f in facts}
        self.assertEqual(by_name["归属于上市公司股东的扣除非经常性损益的净利润"]["metric_atom_id"], "MET-001")
        self.assertEqual(by_name["归属于上市公司股东的净利润"]["metric_atom_id"], "MET-002")
        self.assertEqual(by_name["经营活动产生的现金流量净额"]["metric_atom_id"], "MET-003")
        self.assertEqual(by_name["利息净收入比营业收入"]["raw_unit"], "%")
        self.assertIsNone(by_name["利息净收入比营业收入"]["normalized_cny"])
        self.assertIsNone(by_name["利息净收入比营业收入"]["metric_atom_id"])
        self.assertIsNone(
            by_name["扣除与主营业务无关的业务收入和不具备商业实质的收入后的营业收入"]["metric_atom_id"]
        )


if __name__ == "__main__":
    unittest.main()
