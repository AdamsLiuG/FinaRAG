import json
import tempfile
import unittest
from pathlib import Path

from src.retrieval_filters import RetrievalFilters
from src.table_grounding import TableGrounder


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
            self.assertEqual(result["matched_row_headers"], ["营业收入"])
            self.assertEqual(result["matched_col_headers"], ["2024年报"])


if __name__ == "__main__":
    unittest.main()
