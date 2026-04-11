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
            self.assertIn("国产替代", child_chunk["search_text"])

            chunk_metadata_rows = [
                json.loads(line)
                for line in (metadata_store_dir / "chunk_metadata.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreaterEqual(len(chunk_metadata_rows), 2)
            self.assertEqual(chunk_metadata_rows[0]["doc_id"], "688981_2024_20250328")


if __name__ == "__main__":
    unittest.main()
