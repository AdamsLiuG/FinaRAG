import json
import tempfile
import unittest
from pathlib import Path

from src.chart_grounding import ChartGrounder
from src.retrieval_filters import RetrievalFilters


class ChartGroundingTests(unittest.TestCase):
    def test_ground_number_query_returns_chart_record_match(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            documents_dir = Path(tmp_dir)
            payload = {
                "metainfo": {
                    "sha1_name": "doc-alpha",
                    "company_name": "浦发银行",
                    "currency": "CNY",
                    "report_year": 2024,
                    "language": "zh",
                },
                "content": {
                    "pages": [
                        {"page": 35, "text": "[Chart Evidence]\n图表ID：doc-alpha_p35_pic2\n营业收入 2024 135.8亿元"}
                    ],
                    "chart_records": [
                        {
                            "chart_id": "doc-alpha_p35_pic2",
                            "page": 35,
                            "picture_id": 2,
                            "series_name": "营业收入",
                            "x_label": "2024",
                            "raw_value": "135.8",
                            "normalized_value": 13580000000.0,
                            "unit": "亿元",
                            "context_text": "营业收入趋势图，单位：亿元。",
                            "confidence": 0.82,
                        }
                    ],
                },
            }
            (documents_dir / "doc-alpha.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            result = ChartGrounder(documents_dir).ground_number_query(
                question="浦发银行2024年营业收入是多少亿元？",
                retrieval_results=[{"page": 35, "metadata": {"sha1_name": "doc-alpha"}}],
                filters=RetrievalFilters(company_name="浦发银行", year=2024, question_kind="number"),
                candidate_doc_ids=["doc-alpha"],
            )

            self.assertIsNotNone(result)
            self.assertEqual(result["chart_id"], "doc-alpha_p35_pic2")
            self.assertEqual(result["picture_id"], 2)
            self.assertEqual(result["page"], 35)
            self.assertEqual(result["series_name"], "营业收入")
            self.assertEqual(result["x_label"], "2024")
            self.assertEqual(result["answer_value"], 135.8)
            self.assertEqual(result["target_unit"], "亿元")
            self.assertGreater(result["match_score"], 2.2)

    def test_unit_mismatch_lowers_match_score(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            documents_dir = Path(tmp_dir)
            payload = {
                "metainfo": {"sha1_name": "doc-alpha", "company_name": "浦发银行", "currency": "CNY"},
                "content": {
                    "chart_records": [
                        {
                            "chart_id": "chart-1",
                            "page": 35,
                            "picture_id": 2,
                            "series_name": "营业收入",
                            "x_label": "2024",
                            "raw_value": "135.8",
                            "normalized_value": 13580000000.0,
                            "unit": "亿元",
                            "context_text": "营业收入趋势图，单位：亿元。",
                            "confidence": 0.82,
                        }
                    ]
                },
            }
            (documents_dir / "doc-alpha.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            grounder = ChartGrounder(documents_dir)

            yuan_result = grounder.ground_number_query(
                question="浦发银行2024年营业收入是多少元？",
                retrieval_results=[],
                filters=RetrievalFilters(company_name="浦发银行", year=2024, question_kind="number"),
                candidate_doc_ids=["doc-alpha"],
            )
            percent_result = grounder.ground_number_query(
                question="浦发银行2024年营业收入占比是多少？",
                retrieval_results=[],
                filters=RetrievalFilters(company_name="浦发银行", year=2024, question_kind="number"),
                candidate_doc_ids=["doc-alpha"],
            )

            self.assertIsNotNone(yuan_result)
            self.assertIsNotNone(percent_result)
            self.assertLess(percent_result["match_score"], yuan_result["match_score"])


if __name__ == "__main__":
    unittest.main()
