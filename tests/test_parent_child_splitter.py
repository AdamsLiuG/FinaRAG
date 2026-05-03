import json
import tempfile
import unittest
from pathlib import Path

from src.text_splitter import TextSplitter


class ParentChildSplitterTests(unittest.TestCase):
    def test_short_parent_block_emits_parent_and_single_child(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            reports_dir = Path(tmp_dir) / "reports"
            output_dir = Path(tmp_dir) / "chunked"
            reports_dir.mkdir()

            report = {
                "metainfo": {
                    "company_name": "Alpha Corp",
                    "currency": "USD",
                    "report_type": "annual",
                },
                "content": {
                    "pages": [
                        {
                            "page": 1,
                            "text": "# Overview\nRevenue increased in 2023.",
                        }
                    ]
                },
            }
            (reports_dir / "alpha.json").write_text(json.dumps(report), encoding="utf-8")

            splitter = TextSplitter(child_chunk_size=200, child_chunk_overlap=20)
            splitter.split_all_reports(reports_dir, output_dir)

            output = json.loads((output_dir / "alpha.json").read_text(encoding="utf-8"))
            parent_chunks = output["content"]["parent_chunks"]
            child_chunks = output["content"]["chunks"]

            self.assertEqual(len(parent_chunks), 1)
            self.assertEqual(len(child_chunks), 1)

            parent_chunk = parent_chunks[0]
            child_chunk = child_chunks[0]
            self.assertEqual(parent_chunk["node_type"], "parent")
            self.assertEqual(child_chunk["node_type"], "child")
            self.assertEqual(child_chunk["parent_chunk_id"], parent_chunk["chunk_id"])
            self.assertEqual(parent_chunk["child_chunk_ids"], [child_chunk["chunk_id"]])
            self.assertEqual(child_chunk["text"], parent_chunk["text"])

    def test_serialized_table_emits_parent_and_child_nodes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            reports_dir = Path(tmp_dir) / "reports"
            serialized_tables_dir = Path(tmp_dir) / "tables"
            output_dir = Path(tmp_dir) / "chunked"
            reports_dir.mkdir()
            serialized_tables_dir.mkdir()

            report = {
                "metainfo": {
                    "company_name": "Alpha Corp",
                    "currency": "USD",
                    "report_type": "annual",
                },
                "content": {
                    "pages": [
                        {
                            "page": 1,
                            "text": "# Financial Tables\nPlease see the table below.",
                        }
                    ]
                },
            }
            serialized_tables = {
                "tables": [
                    {
                        "page": 1,
                        "table_id": "tbl-1",
                        "serialized": {
                            "information_blocks": [
                                {"information_block": "Revenue | 100 USD"},
                                {"information_block": "Operating margin | 10%"},
                            ]
                        },
                    }
                ]
            }

            (reports_dir / "alpha.json").write_text(json.dumps(report), encoding="utf-8")
            (serialized_tables_dir / "alpha.json").write_text(json.dumps(serialized_tables), encoding="utf-8")

            splitter = TextSplitter(child_chunk_size=400, child_chunk_overlap=20)
            splitter.split_all_reports(reports_dir, output_dir, serialized_tables_dir)

            output = json.loads((output_dir / "alpha.json").read_text(encoding="utf-8"))
            parent_chunks = output["content"]["parent_chunks"]
            child_chunks = output["content"]["chunks"]

            table_parents = [chunk for chunk in parent_chunks if chunk["chunk_type"] == "serialized_table"]
            table_children = [chunk for chunk in child_chunks if chunk["chunk_type"] == "serialized_table"]

            self.assertEqual(len(table_parents), 1)
            self.assertEqual(len(table_children), 1)
            self.assertEqual(table_parents[0]["node_type"], "parent")
            self.assertEqual(table_children[0]["node_type"], "child")
            self.assertEqual(table_children[0]["parent_chunk_id"], table_parents[0]["chunk_id"])
            self.assertEqual(table_parents[0]["child_chunk_ids"], [table_children[0]["chunk_id"]])

    def test_chart_evidence_emits_chart_to_table_chunks_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            reports_dir = Path(tmp_dir) / "reports"
            output_dir = Path(tmp_dir) / "chunked"
            metadata_store_dir = Path(tmp_dir) / "metadata_store"
            reports_dir.mkdir()

            report = {
                "metainfo": {
                    "sha1_name": "600000_2024",
                    "company_name": "浦发银行",
                    "currency": "CNY",
                    "report_year": 2024,
                    "language": "zh",
                },
                "content": {
                    "pages": [
                        {
                            "page": 35,
                            "text": "# 经营情况\n[Chart Evidence]\n图表ID：600000_2024_p35_pic2",
                        }
                    ],
                    "charts": [
                        {
                            "chart_id": "600000_2024_p35_pic2",
                            "picture_id": 2,
                            "page": 35,
                            "bbox": [120, 180, 520, 430],
                            "table_markdown": "| 年份 | 营业收入 |\n| --- | --- |\n| 2024 | 135.8 |",
                            "context_text": "营业收入趋势图，单位：亿元。",
                            "status": "ok",
                            "records": [
                                {
                                    "chart_id": "600000_2024_p35_pic2",
                                    "page": 35,
                                    "picture_id": 2,
                                    "series_name": "营业收入",
                                    "x_label": "2024",
                                    "raw_value": "135.8",
                                    "normalized_value": 13580000000.0,
                                    "unit": "亿元",
                                    "confidence": 0.82,
                                }
                            ],
                        }
                    ],
                },
            }
            (reports_dir / "600000_2024.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

            splitter = TextSplitter(child_chunk_size=400, child_chunk_overlap=20)
            splitter.split_all_reports(reports_dir, output_dir, metadata_store_dir=metadata_store_dir)

            output = json.loads((output_dir / "600000_2024.json").read_text(encoding="utf-8"))
            chart_parents = [chunk for chunk in output["content"]["parent_chunks"] if chunk["chunk_type"] == "chart_to_table"]
            chart_children = [chunk for chunk in output["content"]["chunks"] if chunk["chunk_type"] == "chart_to_table"]

            self.assertEqual(len(chart_parents), 1)
            self.assertEqual(len(chart_children), 1)
            self.assertEqual(chart_parents[0]["evidence_type"], "chart")
            self.assertEqual(chart_parents[0]["chart_id"], "600000_2024_p35_pic2")
            self.assertEqual(chart_parents[0]["picture_id"], 2)
            self.assertEqual(chart_parents[0]["unit_hint"], "亿元")
            self.assertTrue(chart_parents[0]["has_chart_context"])
            self.assertEqual(output["content"]["chart_records"][0]["series_name"], "营业收入")

            metadata_rows = [
                json.loads(line)
                for line in (metadata_store_dir / "chunk_metadata.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            chart_rows = [row for row in metadata_rows if row["chunk_type"] == "chart_to_table"]
            self.assertGreaterEqual(len(chart_rows), 2)
            self.assertEqual(chart_rows[0]["chart_id"], "600000_2024_p35_pic2")
            self.assertEqual(chart_rows[0]["picture_id"], 2)
            self.assertEqual(chart_rows[0]["evidence_type"], "chart")

    def test_splitter_inherits_pdfcrawl_page_metadata_and_writes_chunk_metadata_store(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            reports_dir = Path(tmp_dir) / "reports"
            output_dir = Path(tmp_dir) / "chunked"
            metadata_store_dir = Path(tmp_dir) / "metadata_store"
            reports_dir.mkdir()
            metadata_store_dir.mkdir()

            report = {
                "metainfo": {
                    "company_name": "中芯国际",
                    "security_code": "688981",
                    "doc_source_type": "annual_report",
                    "report_year": 2024,
                    "exchange": "上海证券交易所",
                    "board": "科创板",
                    "market_type": "A股",
                    "industry_l1": "半导体",
                    "industry_l2": "晶圆代工",
                },
                "content": {
                    "pages": [
                        {
                            "page": 1,
                            "text": "# 第一节 重要提示\n公司坚持先进工艺研发。",
                        }
                    ]
                },
            }
            (reports_dir / "688981_2024_20250328.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
            (metadata_store_dir / "report_page.jsonl").write_text(
                json.dumps(
                    {
                        "report_id": "688981_2024_20250328",
                        "page": 1,
                        "page_start": 1,
                        "page_end": 1,
                        "chunk_id": "688981_2024_20250328_page_0001",
                        "stock_code": "688981",
                        "company_name": "中芯国际",
                        "exchange": "上海证券交易所",
                        "board": "科创板",
                        "market_type": "A股",
                        "industry_l1": "半导体",
                        "industry_l2": "晶圆代工",
                        "business_tags": ["晶圆制造"],
                        "strategy_tags": ["国产替代"],
                        "factor_tags": [],
                        "chain_position_major": "中游制造",
                        "chain_position_minor": ["晶圆代工"],
                        "listing_tags": ["A股", "科创板"],
                        "ownership_tags": [],
                        "status_tags": ["龙头"],
                        "style_tags": ["硬科技"],
                        "section_name": "第一节 重要提示",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (metadata_store_dir / "company_label_snapshot.jsonl").write_text(
                json.dumps(
                    {
                        "report_id": "688981_2024_20250328",
                        "stock_code": "688981",
                        "industry_l1": "半导体",
                        "industry_l2": "晶圆代工",
                        "business_tags": ["晶圆制造"],
                        "strategy_tags": ["国产替代"],
                        "factor_tags": [],
                        "chain_position_major": "中游制造",
                        "chain_position_minor": ["晶圆代工"],
                        "listing_tags": ["A股", "科创板"],
                        "ownership_tags": [],
                        "status_tags": ["龙头"],
                        "style_tags": ["硬科技"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            splitter = TextSplitter(child_chunk_size=200, child_chunk_overlap=20)
            splitter.split_all_reports(reports_dir, output_dir, metadata_store_dir=metadata_store_dir)

            output = json.loads((output_dir / "688981_2024_20250328.json").read_text(encoding="utf-8"))
            child_chunk = output["content"]["chunks"][0]
            self.assertEqual(child_chunk["section_name"], "第一节 重要提示")
            self.assertEqual(child_chunk["industry_l1"], "半导体")
            self.assertEqual(child_chunk["strategy_tags"], ["国产替代"])
            self.assertIn("章节：第一节 重要提示", child_chunk["embedding_text"])
            self.assertNotIn("战略主题：国产替代", child_chunk["embedding_text"])
            self.assertNotIn("国产替代", child_chunk["search_text"])

            chunk_metadata_rows = [
                json.loads(line)
                for line in (metadata_store_dir / "chunk_metadata.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreaterEqual(len(chunk_metadata_rows), 2)
            self.assertEqual(chunk_metadata_rows[0]["doc_id"], "688981_2024_20250328")
            evidence_rows = [
                json.loads(line)
                for line in (metadata_store_dir / "company_label_evidence.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(evidence_rows), 1)
            self.assertEqual(evidence_rows[0]["label_field"], "strategy_tags")
            self.assertEqual(evidence_rows[0]["label"], "国产替代")
            self.assertFalse(evidence_rows[0]["has_literal_evidence"])
            self.assertIsNone(evidence_rows[0]["evidence_page"])

    def test_company_label_evidence_index_records_literal_evidence(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            reports_dir = Path(tmp_dir) / "reports"
            output_dir = Path(tmp_dir) / "chunked"
            metadata_store_dir = Path(tmp_dir) / "metadata_store"
            reports_dir.mkdir()
            metadata_store_dir.mkdir()

            report = {
                "metainfo": {
                    "sha1_name": "688981_2024_20250328",
                    "company_name": "中芯国际",
                    "security_code": "688981",
                    "doc_source_type": "annual_report",
                    "report_year": 2024,
                },
                "content": {
                    "pages": [
                        {"page": 1, "text": "# 第一节 重要提示\n公司持续推进自主可控技术。"}
                    ]
                },
            }
            (reports_dir / "688981_2024_20250328.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
            (metadata_store_dir / "company_label_snapshot.jsonl").write_text(
                json.dumps(
                    {
                        "report_id": "688981_2024_20250328",
                        "strategy_tags": ["国产替代"],
                        "label_source": "company_labels.jsonl",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            splitter = TextSplitter(child_chunk_size=200, child_chunk_overlap=20)
            splitter.split_all_reports(reports_dir, output_dir, metadata_store_dir=metadata_store_dir)

            evidence_rows = [
                json.loads(line)
                for line in (metadata_store_dir / "company_label_evidence.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(evidence_rows[0]["label"], "国产替代")
            self.assertTrue(evidence_rows[0]["has_literal_evidence"])
            self.assertEqual(evidence_rows[0]["evidence_page"], 1)
            self.assertEqual(evidence_rows[0]["match_term"], "自主可控")
            self.assertIn("自主可控技术", evidence_rows[0]["evidence_snippet"])


if __name__ == "__main__":
    unittest.main()
