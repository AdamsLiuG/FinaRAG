import unittest

from src.parsed_reports_merging import PageTextPreparation


class ParsedReportsMergingTests(unittest.TestCase):
    def test_prepare_page_text_treats_code_block_as_plain_text(self):
        merger = PageTextPreparation()
        merger.report_data = {
            "metainfo": {"sha1_name": "demo"},
            "content": [
                {
                    "page": 1,
                    "content": [
                        {"type": "page_header", "text": "示例公司 2024 年年度报告"},
                        {"type": "code", "text": "十、是否存在违反规定决策程序对外提供担保的情况 否"},
                        {"type": "text", "text": "□适用 √不适用"},
                    ],
                }
            ],
            "tables": [],
        }

        page_text = merger.prepare_page_text(1)

        self.assertIn("十、是否存在违反规定决策程序对外提供担保的情况 否", page_text)
        self.assertIn("□适用 √不适用", page_text)

    def test_prepare_page_text_includes_chart_evidence_for_extracted_picture(self):
        merger = PageTextPreparation()
        merger.report_data = {
            "metainfo": {"sha1_name": "600000_2024"},
            "content": [
                {
                    "page": 35,
                    "content": [
                        {"type": "text", "text": "营业收入趋势如下。"},
                        {"type": "picture", "picture_id": 2},
                    ],
                }
            ],
            "tables": [],
            "charts": [
                {
                    "chart_id": "600000_2024_p35_pic2",
                    "picture_id": 2,
                    "page": 35,
                    "table_markdown": "| 年份 | 营业收入 |\n| --- | --- |\n| 2024 | 135.8 |",
                    "context_text": "单位：亿元",
                    "status": "ok",
                }
            ],
        }

        page_text = merger.prepare_page_text(35)

        self.assertIn("[Chart Evidence]", page_text)
        self.assertIn("图表ID：600000_2024_p35_pic2", page_text)
        self.assertIn("| 2024 | 135.8 |", page_text)
        self.assertIn("周边说明：", page_text)


if __name__ == "__main__":
    unittest.main()
