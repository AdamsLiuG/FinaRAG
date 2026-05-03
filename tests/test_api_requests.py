import unittest

from src.api_requests import (
    APIProcessor,
    BaseCompatibleProcessor,
    _construct_structured_payload_from_text,
    _unwrap_singleton_json_container,
)
from src.prompts import AnswerWithRAGContextNamePrompt, AnswerWithRAGContextNamesPrompt


class ApiRequestsTests(unittest.TestCase):
    def test_text_schema_uses_text_generation_prompt(self):
        processor = APIProcessor(provider="qwen")

        system_prompt, response_format, user_prompt = processor._build_rag_context_prompts("text")

        self.assertIn("综合", system_prompt)
        self.assertIn("{context}", user_prompt)
        self.assertIn("{question}", user_prompt)
        parsed = response_format.model_validate(
            {
                "step_by_step_analysis": "1. 先定位证据。\n2. 再按问题归纳。",
                "reasoning_summary": "基于年报证据归纳。",
                "relevant_pages": [12, 38],
                "final_answer": "公司主要从业务拓展和成本控制两方面说明变化。",
            }
        )
        self.assertEqual(parsed.final_answer, "公司主要从业务拓展和成本控制两方面说明变化。")

    def test_unwrap_singleton_json_container_returns_single_dict(self):
        payload = [{"final_answer": "N/A", "relevant_pages": []}]

        unwrapped = _unwrap_singleton_json_container(payload)

        self.assertEqual(unwrapped, {"final_answer": "N/A", "relevant_pages": []})

    def test_unwrap_singleton_json_container_keeps_non_singleton_values(self):
        payload = [
            {"final_answer": "A", "relevant_pages": [1]},
            {"final_answer": "B", "relevant_pages": [2]},
        ]

        unwrapped = _unwrap_singleton_json_container(payload)

        self.assertIs(unwrapped, payload)

    def test_parse_structured_response_accepts_singleton_list_after_reparse(self):
        processor = BaseCompatibleProcessor.__new__(BaseCompatibleProcessor)
        processor._reparse_response = lambda response, response_format, model, original_system_prompt=None: (
            '[{"step_by_step_analysis":"证据不足。","reasoning_summary":"返回N/A。","relevant_pages":[],"final_answer":"N/A"}]'
        )

        parsed = processor._parse_structured_response(
            "not valid json",
            AnswerWithRAGContextNamePrompt.AnswerSchema,
            "dummy-model",
        )

        self.assertEqual(parsed["final_answer"], "N/A")
        self.assertEqual(parsed["relevant_pages"], [])

    def test_parse_structured_response_accepts_mixed_list_with_last_valid_dict(self):
        processor = BaseCompatibleProcessor.__new__(BaseCompatibleProcessor)
        processor._reparse_response = lambda response, response_format, model, original_system_prompt=None: (
            '[{}, "noise", {"step_by_step_analysis":"证据不足。","reasoning_summary":"返回N/A。","relevant_pages":[],"final_answer":"N/A"}]'
        )

        parsed = processor._parse_structured_response(
            "not valid json",
            AnswerWithRAGContextNamePrompt.AnswerSchema,
            "dummy-model",
            original_system_prompt="ORIGINAL_SYSTEM_PROMPT",
        )

        self.assertEqual(parsed["final_answer"], "N/A")
        self.assertEqual(parsed["relevant_pages"], [])

    def test_construct_structured_payload_from_text_recovers_na_name_schema(self):
        free_text = """
The user wants me to format the LLM response into a valid JSON object that matches the target schema.
The LLM response concludes that the answer should be "N/A" because:
1. The provided context (pages 38, 89, 91) does not contain the term "法定代表人"
2. Page 38 mentions 伍锐 as chairman, but chairman is not necessarily the legal representative
3. Page 91 mentions "公司负责人：伍锐", but this is not the same as "法定代表人"
4. According to the rules, the model cannot infer across concepts and must return N/A if there is no direct evidence
"""

        parsed = _construct_structured_payload_from_text(
            free_text,
            AnswerWithRAGContextNamePrompt.AnswerSchema,
        )

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["final_answer"], "N/A")
        self.assertEqual(parsed["relevant_pages"], [])
        self.assertIn("N/A", parsed["reasoning_summary"])

    def test_parse_structured_response_heuristic_fallback_recovers_na_name_schema(self):
        processor = BaseCompatibleProcessor.__new__(BaseCompatibleProcessor)
        processor._reparse_response = lambda response, response_format, model, original_system_prompt=None: """
The user wants me to format the LLM response into a valid JSON object that matches the target schema.
The LLM response concludes that the answer should be "N/A" because:
1. The provided context (pages 38, 89, 91) does not contain the term "法定代表人"
2. Page 38 mentions 伍锐 as chairman
3. Page 91 mentions "公司负责人：伍锐"
4. According to the rules, the model must return N/A if there is no direct evidence
"""

        parsed = processor._parse_structured_response(
            "not valid json",
            AnswerWithRAGContextNamePrompt.AnswerSchema,
            "dummy-model",
        )

        self.assertEqual(parsed["final_answer"], "N/A")
        self.assertEqual(parsed["relevant_pages"], [])

    def test_construct_structured_payload_from_text_recovers_names_list_from_free_text(self):
        free_text = """
The user wants me to format the raw LLM response into a valid JSON object.
The key information is:
1. 北京博睿宏远数据科技股份有限公司 mentioned "国产化替代"
2. 武汉达梦数据库股份有限公司 mentioned "国产化替代"

The raw response also checked Page 34, Page 148 and Page 216.
Actually, I should format it as:
{
  "answer": ["北京博睿宏远数据科技股份有限公司", "武汉达梦数据库股份有限公司"]
}
"""

        parsed = _construct_structured_payload_from_text(
            free_text,
            AnswerWithRAGContextNamesPrompt.AnswerSchema,
        )

        self.assertIsNotNone(parsed)
        self.assertEqual(
            parsed["final_answer"],
            ["北京博睿宏远数据科技股份有限公司", "武汉达梦数据库股份有限公司"],
        )
        self.assertEqual(parsed["relevant_pages"], [34, 148, 216])

    def test_parse_structured_response_heuristic_fallback_prefers_terminal_list_answer(self):
        processor = BaseCompatibleProcessor.__new__(BaseCompatibleProcessor)
        processor._reparse_response = lambda response, response_format, model, original_system_prompt=None: """
The model is uncertain and also discusses an alternative N/A option:
{"answer": "N/A"}

But the final structure should be:
{
  "answer": ["北京博睿宏远数据科技股份有限公司", "武汉达梦数据库股份有限公司"]
}
"""

        parsed = processor._parse_structured_response(
            "not valid json",
            AnswerWithRAGContextNamesPrompt.AnswerSchema,
            "dummy-model",
        )

        self.assertEqual(
            parsed["final_answer"],
            ["北京博睿宏远数据科技股份有限公司", "武汉达梦数据库股份有限公司"],
        )

    def test_reparse_response_includes_original_system_prompt_and_schema(self):
        processor = BaseCompatibleProcessor.__new__(BaseCompatibleProcessor)
        captured = {}

        def fake_send_message(**kwargs):
            captured.update(kwargs)
            return '{"step_by_step_analysis":"证据不足。","reasoning_summary":"返回N/A。","relevant_pages":[],"final_answer":"N/A"}'

        processor.send_message = fake_send_message

        processor._reparse_response(
            "raw text",
            AnswerWithRAGContextNamePrompt.AnswerSchema,
            "dummy-model",
            original_system_prompt="ORIGINAL_SYSTEM_PROMPT",
        )

        self.assertIn("ORIGINAL_SYSTEM_PROMPT", captured["human_content"])
        self.assertIn('"final_answer"', captured["human_content"])
        self.assertIn('"relevant_pages"', captured["human_content"])

    def test_send_message_reparse_uses_unaugmented_original_system_prompt(self):
        processor = BaseCompatibleProcessor.__new__(BaseCompatibleProcessor)
        processor.default_model = "dummy-model"
        processor.max_tokens = None
        processor.wire_api = "chat_completions"
        processor.api_key = None
        processor.use_stream = False
        processor.provider = "qwen"
        processor.enable_thinking = False
        processor.response_data = {}
        captured = {}

        class FakeResponse:
            ok = True
            url = "http://unit.test/chat/completions"
            text = "{}"

            def json(self):
                return {
                    "model": "dummy-model",
                    "usage": {},
                    "choices": [{"message": {"content": "{}"}}],
                }

            def close(self):
                pass

        processor._get_request_url = lambda: "http://unit.test/chat/completions"
        processor._is_dashscope_compatible = lambda: False
        processor._post_request = lambda request_kwargs: FakeResponse()
        processor._close_response = lambda response: response.close()

        original_build_payload = processor._build_payload

        def capture_build_payload(**kwargs):
            captured["request_system_content"] = kwargs["system_content"]
            return original_build_payload(**kwargs)

        def capture_parse(response_text, response_format, model, original_system_prompt=None):
            captured["original_system_prompt"] = original_system_prompt
            return {
                "step_by_step_analysis": "证据不足。",
                "reasoning_summary": "返回N/A。",
                "relevant_pages": [],
                "final_answer": "N/A",
            }

        processor._build_payload = capture_build_payload
        processor._parse_structured_response = capture_parse

        processor.send_message(
            model="dummy-model",
            temperature=0,
            system_content="ORIGINAL_SYSTEM_PROMPT",
            human_content="question",
            is_structured=True,
            response_format=AnswerWithRAGContextNamePrompt.AnswerSchema,
        )

        self.assertIn("strictly follow this schema", captured["request_system_content"])
        self.assertEqual(captured["original_system_prompt"], "ORIGINAL_SYSTEM_PROMPT")

    def test_relevant_pages_prompt_allows_empty_pages_for_na(self):
        description = AnswerWithRAGContextNamePrompt.AnswerSchema.model_fields["relevant_pages"].description

        self.assertIn("当 `final_answer` 不是 `N/A` 时，至少填写 1 个页码。", description)
        self.assertIn("当证据不足并返回 `N/A` 时，填写空列表 `[]`。", description)


if __name__ == "__main__":
    unittest.main()
