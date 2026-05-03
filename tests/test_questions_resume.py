from __future__ import annotations

import json
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.questions_processing import QuestionsProcessor


class QuestionsResumeTests(unittest.TestCase):
    def _build_processor(self) -> QuestionsProcessor:
        processor = QuestionsProcessor.__new__(QuestionsProcessor)
        processor.parallel_requests = 1
        processor.reasoning_debug_enabled = True
        processor._lock = threading.Lock()
        processor.answer_details = []
        processor.processed_calls = []

        def _process_single_question(question_data):
            question_index = question_data["_question_index"]
            processor.processed_calls.append(question_data["id"])
            processor.answer_details[question_index] = {
                "self": f"#/answer_details/{question_index}",
                "step_by_step_analysis": f"processed-{question_index}",
            }
            return {
                "question_id": question_data["id"],
                "question_text": question_data["text"],
                "kind": question_data["kind"],
                "value": f"value-{question_index}",
                "references": [],
                "citations": [],
                "confidence": "high",
                "confidence_reason": "ok",
                "validation_flags": [],
                "route_info": {},
                "answer_details": {"$ref": f"#/answer_details/{question_index}"},
            }

        processor._process_single_question = _process_single_question
        return processor

    def test_process_questions_list_resume_skips_completed_and_retries_errors(self):
        questions = [
            {"id": "q1", "text": "问题1", "kind": "name"},
            {"id": "q2", "text": "问题2", "kind": "name"},
            {"id": "q3", "text": "问题3", "kind": "name"},
        ]

        partial_answers = {
            "answers": [
                {
                    "question_text": "问题1",
                    "kind": "name",
                    "value": "已有答案",
                    "references": [],
                    "citations": [],
                    "confidence": "high",
                    "confidence_reason": "done",
                    "validation_flags": [],
                    "route_info": {},
                },
                {
                    "question_text": "问题2",
                    "kind": "name",
                    "value": "N/A",
                    "references": [],
                    "citations": [],
                    "confidence": "low",
                    "confidence_reason": "处理失败：ValueError",
                    "validation_flags": ["processing_error"],
                    "route_info": {},
                },
            ],
            "details": "partial",
        }
        partial_debug = {
            "questions": [
                {
                    "question_id": "q1",
                    "question_text": "问题1",
                    "kind": "name",
                    "value": "已有答案",
                    "references": [],
                    "citations": [],
                    "confidence": "high",
                    "confidence_reason": "done",
                    "validation_flags": [],
                    "route_info": {},
                    "answer_details": {"$ref": "#/answer_details/0"},
                },
                {
                    "question_id": "q2",
                    "question_text": "问题2",
                    "kind": "name",
                    "value": None,
                    "references": [],
                    "citations": [],
                    "confidence": "low",
                    "confidence_reason": "处理失败：ValueError",
                    "validation_flags": ["processing_error"],
                    "route_info": {},
                    "error": "处理失败：ValueError",
                    "answer_details": {"$ref": "#/answer_details/1"},
                },
            ],
            "answer_details": [
                {"self": "#/answer_details/0", "step_by_step_analysis": "existing"},
                {"self": "#/answer_details/1", "error_traceback": "boom"},
                None,
            ],
            "statistics": {
                "total_questions": 2,
                "error_count": 1,
                "na_count": 0,
                "success_count": 1,
            },
        }

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "answers_resume.json"
            debug_path = output_path.with_name(output_path.stem + "_debug" + output_path.suffix)
            output_path.write_text(json.dumps(partial_answers, ensure_ascii=False, indent=2), encoding="utf-8")
            debug_path.write_text(json.dumps(partial_debug, ensure_ascii=False, indent=2), encoding="utf-8")

            processor = self._build_processor()
            result = QuestionsProcessor.process_questions_list(
                processor,
                questions,
                output_path=str(output_path),
                pipeline_details="resume test",
                resume_from=output_path,
            )

            self.assertEqual(processor.processed_calls, ["q2", "q3"])
            self.assertEqual(result["statistics"]["total_questions"], 3)
            self.assertEqual(result["statistics"]["error_count"], 0)
            self.assertEqual(result["statistics"]["success_count"], 3)

            saved_answers = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [item["value"] for item in saved_answers["answers"]],
                ["已有答案", "value-1", "value-2"],
            )

            saved_debug = json.loads(debug_path.read_text(encoding="utf-8"))
            self.assertEqual(len(saved_debug["questions"]), 3)
            self.assertNotIn("error", saved_debug["questions"][1])


if __name__ == "__main__":
    unittest.main()
