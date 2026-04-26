import unittest

from training.reranker_distill.scripts.build_auto_evidence import (
    build_auto_evidence_record,
    select_positive_evidence_records,
)
from training.reranker_distill.scripts.build_pointwise_labels import summarize_answer_evidence


class RerankerAutoEvidenceTests(unittest.TestCase):
    def test_select_revenue_evidence_prefers_current_year_revenue_page(self):
        records = [
            {
                "query_id": "seed-auto-revenue-600000_2024_20250329",
                "candidate_id": "cand-2023",
                "doc_id": "600000_2024_20250329",
                "page": 258,
                "teacher_rank": 1,
                "teacher_score": 0.9,
                "section_name": "一、营业收入 173,434",
                "text": "2023 年度 营业收入 173,434",
            },
            {
                "query_id": "seed-auto-revenue-600000_2024_20250329",
                "candidate_id": "cand-2024",
                "doc_id": "600000_2024_20250329",
                "page": 151,
                "teacher_rank": 2,
                "teacher_score": 0.8,
                "section_name": "一、营业收入 170,748",
                "text": "2024 年度 营业收入 170,748",
            },
        ]

        selected = select_positive_evidence_records(
            "seed-auto-revenue-600000_2024_20250329",
            records,
            max_positive_pages=2,
        )

        self.assertEqual([record["page"] for record in selected], [151])

    def test_generated_evidence_record_is_compatible_with_pointwise_labeler(self):
        score_record = {
            "query_id": "seed-auto-legal_representative-600016_2024_20250329",
            "candidate_id": "cand-0005",
            "question_text": "民生银行2024年年报中的法定代表人是谁？",
            "schema": "name",
            "doc_id": "600016_2024_20250329",
            "company_name": "民生银行",
            "page": 12,
            "teacher_rank": 1,
            "teacher_score": 0.448,
            "text": "公司法定代表人：高迎欣",
        }
        debug_record = {
            "query_id": "seed-auto-legal_representative-600016_2024_20250329",
            "question_text": "民生银行2024年年报中的法定代表人是谁？",
            "schema": "name",
            "route_info": {
                "candidate_doc_ids": ["600016_2024_20250329"],
                "selected_report": {"sha1": "600016_2024_20250329"},
            },
        }

        evidence_record = build_auto_evidence_record(debug_record, [score_record])
        summary = summarize_answer_evidence(evidence_record)

        self.assertEqual(summary.query_id, "seed-auto-legal_representative-600016_2024_20250329")
        self.assertEqual(summary.positive_page_keys, {("600016_2024_20250329", 12)})
        self.assertEqual(summary.citation_page_keys, {("600016_2024_20250329", 12)})


if __name__ == "__main__":
    unittest.main()
