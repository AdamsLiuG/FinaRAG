import tempfile
import unittest
from pathlib import Path

from src.questions_processing import QuestionsProcessor


class ComparativeQuestionTests(unittest.TestCase):
    def test_process_comparative_question_uses_api_processor_and_preserves_schema(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            subset_path = Path(tmp_dir) / "subset.csv"
            subset_path.write_text(
                "sha1,company_name,cur\n"
                "sha1-alpha,Alpha Corp,USD\n"
                "sha1-beta,Beta Corp,USD\n",
                encoding="utf-8",
            )

            processor = QuestionsProcessor(subset_path=subset_path, parallel_requests=2)
            processor.api_processor.response_data = {"model": "test-model"}

            processor.api_processor.get_rephrased_questions = lambda original_question, companies, model=None: {
                "Alpha Corp": "What was Alpha Corp revenue in 2023?",
                "Beta Corp": "What was Beta Corp revenue in 2023?",
            }

            seen_schemas = []

            def fake_get_answer_for_company(company_name: str, question: str, schema: str):
                seen_schemas.append(schema)
                return {
                    "final_answer": company_name,
                    "references": [{"pdf_sha1": f"sha1-{company_name}", "page_index": 8}],
                    "citations": [{"source": f"sha1-{company_name}", "page": 8, "chunk_id": 1, "chunk_type": "content"}],
                    "confidence": "high",
                }

            processor.get_answer_for_company = fake_get_answer_for_company

            def fake_compare(question, rag_context, schema, model):
                self.assertEqual(schema, "comparative")
                self.assertIn("Alpha Corp", rag_context)
                self.assertIn("Beta Corp", rag_context)
                return {
                    "final_answer": "Alpha Corp",
                    "reasoning_summary": "Alpha is larger.",
                    "step_by_step_analysis": "Compared Alpha and Beta.",
                    "relevant_pages": [],
                }

            processor.api_processor.get_answer_from_rag_context = fake_compare

            result = processor.process_comparative_question(
                'Which company had higher revenue, "Alpha Corp" or "Beta Corp"?',
                ["Alpha Corp", "Beta Corp"],
                "name",
            )

            self.assertEqual(seen_schemas, ["name", "name"])
            self.assertEqual(result["final_answer"], "Alpha Corp")
            self.assertEqual(len(result["references"]), 2)
            self.assertEqual(result["confidence"], "high")


if __name__ == "__main__":
    unittest.main()
