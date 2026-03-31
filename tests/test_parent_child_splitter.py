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


if __name__ == "__main__":
    unittest.main()
