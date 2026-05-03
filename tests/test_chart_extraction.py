import json
import tempfile
import unittest
from pathlib import Path

from src.chart_extraction import ChartResultWriter, ChartTableParser, generate_chart_id


class ChartExtractionTests(unittest.TestCase):
    def test_chart_table_parser_converts_pipe_table_to_records(self):
        parser = ChartTableParser()

        parsed = parser.parse(
            raw_output="年份 | 营业收入\n2023 | 120.5\n2024 | 135.8",
            chart_id="600000_2024_p35_pic2",
            page=35,
            picture_id=2,
            context_text="营业收入趋势图，单位：亿元。",
        )

        self.assertIn("| 年份 | 营业收入 |", parsed["table_markdown"])
        self.assertEqual(len(parsed["records"]), 2)
        record = parsed["records"][1]
        self.assertEqual(record["chart_id"], "600000_2024_p35_pic2")
        self.assertEqual(record["series_name"], "营业收入")
        self.assertEqual(record["x_label"], "2024")
        self.assertEqual(record["raw_value"], "135.8")
        self.assertEqual(record["normalized_value"], 13580000000.0)
        self.assertEqual(record["unit"], "亿元")
        self.assertGreaterEqual(record["confidence"], 0.7)

    def test_generate_chart_id_is_stable(self):
        self.assertEqual(
            generate_chart_id("600000_2024", 35, 2),
            "600000_2024_p35_pic2",
        )

    def test_writer_persists_failed_chart_without_blocking(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "demo.json"
            report_path.write_text(
                json.dumps(
                    {
                        "metainfo": {"sha1_name": "demo"},
                        "content": [],
                        "pictures": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            writer = ChartResultWriter()
            writer.write_results(
                report_path,
                [
                    {
                        "chart_id": "demo_p1_pic0",
                        "picture_id": 0,
                        "page": 1,
                        "bbox": [1, 2, 3, 4],
                        "status": "error",
                        "error": "crop failed",
                    }
                ],
            )

            updated = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["charts"][0]["status"], "error")
            self.assertEqual(updated["charts"][0]["error"], "crop failed")


if __name__ == "__main__":
    unittest.main()
