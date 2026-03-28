import tempfile
import unittest
from pathlib import Path

from src.questions_processing import QuestionsProcessor


class MetadataRoutingTests(unittest.TestCase):
    def test_process_question_routes_by_topic_flags_when_company_name_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            subset_path = Path(tmp_dir) / "subset.csv"
            subset_path.write_text(
                "sha1,company_name,cur,mentions_recent_mergers_and_acquisitions,has_leadership_changes\n"
                "sha1-alpha,Alpha Corp,USD,True,False\n"
                "sha1-beta,Beta Corp,USD,False,True\n",
                encoding="utf-8",
            )

            processor = QuestionsProcessor(subset_path=subset_path, parallel_requests=1)
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
                "Did the company mention any mergers or acquisitions in the annual report?",
                "boolean",
            )

            self.assertEqual(seen["company_name"], "Alpha Corp")
            self.assertEqual(result["route_info"]["route_mode"], "metadata_inference")
            self.assertEqual(result["route_info"]["selected_company"], "Alpha Corp")


if __name__ == "__main__":
    unittest.main()
