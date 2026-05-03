import unittest

from src.retrieval_filters import RetrievalFilters, apply_retrieval_filters


class RetrievalFilterTests(unittest.TestCase):
    def test_required_topic_flags_filter_and_boolean_bonus(self):
        results = [
            {
                "distance": 0.3,
                "metadata": {
                    "chunk_type": "content",
                    "topic_flags": ["mentions_recent_mergers_and_acquisitions"],
                },
            },
            {
                "distance": 0.9,
                "metadata": {
                    "chunk_type": "content",
                    "topic_flags": [],
                },
            },
        ]

        filtered = apply_retrieval_filters(
            results,
            RetrievalFilters(
                required_topic_flags=["mentions_recent_mergers_and_acquisitions"],
                question_kind="boolean",
            ),
        )

        self.assertEqual(len(filtered), 1)
        self.assertGreater(filtered[0]["ranking_score"], filtered[0]["distance"])

    def test_metadata_filters_support_section_and_tag_fields(self):
        results = [
            {
                "distance": 0.6,
                "metadata": {
                    "section_name": "管理层讨论与分析",
                    "industry_l1": "半导体",
                    "strategy_tags": ["国产替代", "人工智能"],
                    "listing_tags": ["A股", "科创板"],
                },
            },
            {
                "distance": 0.9,
                "metadata": {
                    "section_name": "财务报告",
                    "industry_l1": "消费",
                    "strategy_tags": ["出海"],
                    "listing_tags": ["A股"],
                },
            },
        ]

        filtered = apply_retrieval_filters(
            results,
            RetrievalFilters(
                section_name="管理层讨论与分析",
                industry_l1="半导体",
                strategy_tags=["国产替代"],
                listing_tags=["科创板"],
            ),
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["metadata"]["section_name"], "管理层讨论与分析")

    def test_section_filter_falls_back_to_section_title_and_report_section(self):
        results = [
            {
                "distance": 0.4,
                "metadata": {
                    "section_name": "第一节 释义",
                    "section_title": "第二节 公司简介和主要财务指标",
                    "report_section": "第二节 公司简介和主要财务指标",
                },
            },
            {
                "distance": 0.7,
                "metadata": {
                    "section_name": "财务报告",
                    "section_title": "第十节 财务报告",
                    "report_section": "第十节 财务报告",
                },
            },
        ]

        filtered = apply_retrieval_filters(
            results,
            RetrievalFilters(section_name="公司简介和主要财务指标"),
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["metadata"]["section_title"], "第二节 公司简介和主要财务指标")

    def test_section_filter_matches_hierarchical_section_path(self):
        results = [
            {
                "distance": 0.51,
                "metadata": {
                    "section_name": "1. 产品结构",
                    "section_leaf": "1. 产品结构",
                    "section_l1": "第三节 管理层讨论与分析",
                    "section_l2": "一、经营情况讨论与分析",
                    "section_l3": "（一）主营业务分析",
                    "section_path": "第三节 管理层讨论与分析 > 一、经营情况讨论与分析 > （一）主营业务分析 > 1. 产品结构",
                },
            },
            {
                "distance": 0.82,
                "metadata": {
                    "section_name": "财务报告",
                    "section_path": "第十节 财务报告",
                },
            },
        ]

        filtered = apply_retrieval_filters(
            results,
            RetrievalFilters(section_name="经营情况讨论与分析"),
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["metadata"]["section_name"], "1. 产品结构")


if __name__ == "__main__":
    unittest.main()
