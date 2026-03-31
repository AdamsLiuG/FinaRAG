import tempfile
import unittest
from pathlib import Path

from src.questions_processing import QuestionsProcessor


class MetadataRoutingTests(unittest.TestCase):
    def test_process_question_routes_by_document_catalog_when_company_name_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            subset_path = Path(tmp_dir) / "document_manifest.csv"
            subset_path.write_text(
                "doc_id,company_name,company_aliases,security_code,doc_source_type,broker_name,report_title,language,currency\n"
                "doc-alpha,宁德时代,宁王|CATL|300750,300750,research_report,中信证券,中信证券-宁德时代深度报告,zh,CNY\n"
                "doc-beta,贵州茅台,茅台|600519,600519,annual_report,,贵州茅台2024年年报,zh,CNY\n",
                encoding="utf-8",
            )

            processor = QuestionsProcessor(subset_path=subset_path, parallel_requests=1, doc_router_enabled=True)
            seen = {}

            def fake_get_answer_for_company(company_name: str, question: str, schema: str, query_plan=None, route_info=None):
                seen["company_name"] = company_name
                return {
                    "final_answer": True,
                    "references": [],
                    "citations": [],
                    "confidence": "medium",
                    "confidence_reason": "test",
                    "validation_flags": [],
                    "route_info": route_info,
                    "query_plan": query_plan.to_dict() if query_plan else {},
                    "relevant_pages": [2],
                }

            processor.get_answer_for_company = fake_get_answer_for_company

            result = processor.process_question(
                "中信证券这篇研报提到的公司营业收入是多少？",
                "number",
            )

            self.assertEqual(seen["company_name"], "宁德时代")
            self.assertEqual(result["route_info"]["route_mode"], "document_catalog")
            self.assertEqual(result["route_info"]["selected_company"], "宁德时代")
            self.assertIn("doc-alpha", result["route_info"]["candidate_doc_ids"])


if __name__ == "__main__":
    unittest.main()
