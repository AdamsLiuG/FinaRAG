import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.api_requests import APIProcessor, BaseCompatibleProcessor
from training.common import (
    build_rag_prompt_bundle,
    build_split_group_key,
    deterministic_split_for_key,
    load_yaml_mapping,
    normalize_training_query_record,
    prune_answer_to_schema,
    resolve_dataset_root,
)
from training.generator_sft.scripts.build_seed_queries import _normalize_base_question
from training.generator_sft.scripts.convert_to_chat_sft import build_chat_record
from training.generator_sft.scripts.filter_sft_samples import _build_sample_record
from training.generator_sft.scripts.mine_teacher_answers import _reset_output_file
from training.reranker_distill.scripts.build_pointwise_labels import (
    EvidenceSummary,
    LabelConfig,
    assign_hard_label,
)
from training.reranker_distill.scripts.collect_candidate_pool import build_candidate_payload
from training.reranker_distill.scripts.export_for_trainer import build_export_record


class TrainingScaffoldTests(unittest.TestCase):
    def test_reset_output_file_truncates_existing_content(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "teacher_answers_raw.jsonl"
            output_path.write_text('{"query_id":"seed-1"}\n', encoding="utf-8")

            _reset_output_file(output_path)

            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.read_text(encoding="utf-8"), "")

    def test_sub2api_provider_uses_responses_api_shape(self):
        sub2api_model = "test-sub2api-model"
        with patch.dict(
            os.environ,
            {
                "SUB2API_BASE_URL": "https://sub2api.daw111.asia/v1",
                "SUB2API_MODEL": sub2api_model,
                "SUB2API_WIRE_API": "responses",
                "SUB2API_REASONING_EFFORT": "high",
                "SUB2API_DISABLE_RESPONSE_STORAGE": "true",
            },
            clear=False,
        ):
            processor = BaseCompatibleProcessor(provider="sub2api")
            payload = processor._build_payload(
                model=os.environ["SUB2API_MODEL"],
                temperature=0.0,
                max_tokens=1024,
                seed=None,
                system_content="system",
                human_content="user",
            )

        self.assertEqual(processor.wire_api, "responses")
        self.assertEqual(processor._get_request_url(), "https://sub2api.daw111.asia/v1/responses")
        self.assertEqual(payload["model"], sub2api_model)
        self.assertEqual(payload["input"][0]["role"], "system")
        self.assertEqual(payload["input"][1]["role"], "user")
        self.assertEqual(payload["reasoning"]["effort"], "high")
        self.assertFalse(payload["store"])

    def test_sub2api_chat_completions_can_enable_stream_for_working_proxy_path(self):
        sub2api_model = "test-sub2api-model"
        with patch.dict(
            os.environ,
            {
                "SUB2API_BASE_URL": "https://sub2api.daw111.asia/v1",
                "SUB2API_MODEL": sub2api_model,
                "SUB2API_WIRE_API": "chat_completions",
                "SUB2API_STREAM": "true",
            },
            clear=False,
        ):
            processor = BaseCompatibleProcessor(provider="sub2api")
            payload = processor._build_payload(
                model=os.environ["SUB2API_MODEL"],
                temperature=0.0,
                max_tokens=128,
                seed=None,
                system_content="system",
                human_content="user",
            )

        self.assertEqual(processor._get_request_url(), "https://sub2api.daw111.asia/v1/chat/completions")
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][1]["role"], "user")

    def test_qwen_vllm_provider_uses_chat_completions_url(self):
        with patch.dict(
            os.environ,
            {
                "QWEN_VLLM_BASE_URL": "http://127.0.0.1:8002/v1",
                "QWEN_VLLM_MODEL": "Qwen3.5-35B-A3B-AWQ-4bit",
                "QWEN_VLLM_WIRE_API": "chat_completions",
            },
            clear=False,
        ):
            processor = BaseCompatibleProcessor(provider="qwen_vllm")
            api_processor = APIProcessor(provider="qwen_vllm")

        self.assertEqual(processor._get_request_url(), "http://127.0.0.1:8002/v1/chat/completions")
        self.assertIsInstance(api_processor.processor, BaseCompatibleProcessor)

    def test_extract_responses_content_reads_message_output_text(self):
        content = BaseCompatibleProcessor._extract_responses_content(
            {
                "model": "test-sub2api-model",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "{\"final_answer\":\"王奔\"}",
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 8,
                },
            }
        )

        self.assertEqual(content, "{\"final_answer\":\"王奔\"}")

    def test_streaming_reader_aggregates_multiline_sse_event(self):
        class FakeStreamingResponse:
            def iter_lines(self, decode_unicode=True):
                return iter(
                    [
                        'data: {',
                        '  "model": "test-sub2api-model",',
                        '  "choices": [',
                        '    {"delta": {"content": "OK"}}',
                        '  ],',
                        '  "usage": {"prompt_tokens": 3, "completion_tokens": 1}',
                        '}',
                        "",
                        "data: [DONE]",
                        "",
                    ]
                )

        processor = BaseCompatibleProcessor(provider="sub2api")
        content = processor._read_streaming_content(FakeStreamingResponse(), "test-sub2api-model")

        self.assertEqual(content, "OK")
        self.assertEqual(processor.response_data["model"], "test-sub2api-model")
        self.assertEqual(processor.response_data["input_tokens"], 3)
        self.assertEqual(processor.response_data["output_tokens"], 1)

    def test_streaming_reader_supports_list_content_chunks(self):
        class FakeStreamingResponse:
            def iter_lines(self, decode_unicode=True):
                return iter(
                    [
                        'data: {"choices":[{"delta":{"content":[{"text":"O"},{"text":"K"}]}}]}',
                        "",
                        "data: [DONE]",
                        "",
                    ]
                )

        processor = BaseCompatibleProcessor(provider="sub2api")
        content = processor._read_streaming_content(FakeStreamingResponse(), "test-sub2api-model")

        self.assertEqual(content, "OK")

    def test_parse_streaming_chunk_repairs_invalid_control_characters(self):
        chunk = BaseCompatibleProcessor._parse_streaming_chunk(
            """{"choices":[{"delta":{"content":"abc
def"}}]}"""
        )

        self.assertEqual(chunk["choices"][0]["delta"]["content"], "abc\ndef")

    def test_iter_sse_event_data_decodes_utf8_bytes(self):
        class FakeStreamingResponse:
            def iter_lines(self, decode_unicode=False):
                return iter(
                    [
                        'data: {"choices":[{"delta":{"content":"中文"}}]}'.encode("utf-8"),
                        b"",
                        b"data: [DONE]",
                        b"",
                    ]
                )

        events = list(BaseCompatibleProcessor._iter_sse_event_data(FakeStreamingResponse()))

        self.assertEqual(events, ['{"choices":[{"delta":{"content":"中文"}}]}', "[DONE]"])

    def test_load_yaml_mapping_expands_env_placeholders(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(
                "teacher_answer_model: ${SUB2API_MODEL}\nparallel_requests: ${TRAIN_PARALLEL_REQUESTS:-1}\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"SUB2API_MODEL": "test-sub2api-model"}, clear=False):
                config = load_yaml_mapping(config_path)

        self.assertEqual(config["teacher_answer_model"], "test-sub2api-model")
        self.assertEqual(config["parallel_requests"], 1)

    def test_resolve_dataset_root_defaults_to_repo_root(self):
        repo_root = Path("/tmp/finarag")
        self.assertEqual(resolve_dataset_root(repo_root, None), repo_root)
        self.assertEqual(
            resolve_dataset_root(repo_root, "data/top10_industries_2024_20each"),
            (repo_root / "data/top10_industries_2024_20each").resolve(),
        )

    def test_normalize_training_query_record_reads_questions_json_shape(self):
        record = {
            "id": "zh-top10-01-02",
            "text": "华胜天成2024年年报中的营业收入是多少元？",
            "kind": "number",
            "doc_ids": ["600410_2024_20250426"],
            "company_name": "华胜天成",
        }

        normalized = normalize_training_query_record(record)

        self.assertEqual(normalized["query_id"], "zh-top10-01-02")
        self.assertEqual(normalized["question_text"], "华胜天成2024年年报中的营业收入是多少元？")
        self.assertEqual(normalized["schema"], "number")
        self.assertEqual(normalized["doc_ids"], ["600410_2024_20250426"])

    def test_build_seed_question_normalizes_existing_question_set(self):
        record = {
            "id": "zh-top10-01-04",
            "text": "在浪潮软件2024年年报《公司简介和主要财务指标》章节中，法定代表人是谁？",
            "kind": "name",
            "doc_ids": ["600756_2024_20250329"],
            "company_name": "浪潮软件",
            "report_year": 2024,
            "section_name": "公司简介和主要财务指标",
            "expected_filters": {"company_name": "浪潮软件"},
        }

        seed_record = _normalize_base_question(record, index=1)

        self.assertEqual(seed_record["query_id"], "seed-zh-top10-01-04")
        self.assertEqual(seed_record["task_type"], "section_filter")
        self.assertEqual(seed_record["expected_filters"]["report_year"], 2024)
        self.assertEqual(seed_record["expected_filters"]["section_name"], "公司简介和主要财务指标")

    def test_normalize_training_query_record_reads_teacher_answer_shape(self):
        record = {
            "query_id": "seed-0001",
            "question_text": "华胜天成2024年年报中的营业收入是多少元？",
            "schema": "number",
            "doc_ids": ["600410_2024_20250426"],
            "answer": {
                "route_info": {
                    "selected_company": "华胜天成",
                }
            },
        }

        normalized = normalize_training_query_record(record)

        self.assertEqual(normalized["query_id"], "seed-0001")
        self.assertEqual(normalized["schema"], "number")
        self.assertEqual(normalized["company_name"], "华胜天成")
        self.assertEqual(normalized["doc_ids"], ["600410_2024_20250426"])

    def test_build_candidate_payload_keeps_pre_rerank_fields(self):
        result = {
            "page": 9,
            "text": "公司简介和主要财务指标显示，报告期内公司实现营业收入4,270,629,476.42元。",
            "distance": 0.8421,
            "retrieval_sources": ["vector", "bm25"],
            "matched_queries": ["营业收入", "营收"],
            "query_hit_count": 2,
            "result_scope": "parent",
            "metadata": {
                "sha1_name": "600410_2024_20250426",
                "chunk_id": 321,
                "section_name": "公司简介和主要财务指标",
            },
        }

        candidate = build_candidate_payload(result, rank_index=1)

        self.assertEqual(candidate["candidate_id"], "cand-0001")
        self.assertEqual(candidate["doc_id"], "600410_2024_20250426")
        self.assertEqual(candidate["page"], 9)
        self.assertEqual(candidate["base_score"], 0.8421)
        self.assertEqual(candidate["retrieval_sources"], ["vector", "bm25"])
        self.assertEqual(candidate["query_hit_count"], 2)

    def test_prompt_bundle_and_pruned_answer_follow_schema(self):
        system_prompt, user_prompt, response_format = build_rag_prompt_bundle("number", provider="qwen")
        pruned = prune_answer_to_schema(
            {
                "step_by_step_analysis": "分析",
                "reasoning_summary": "总结",
                "relevant_pages": [9],
                "final_answer": 4270629476.42,
                "confidence": "high",
            },
            schema="number",
            provider="qwen",
        )

        self.assertIn("JSON", system_prompt)
        self.assertIn("{context}", user_prompt)
        self.assertEqual(response_format.__name__, "AnswerSchema")
        self.assertEqual(
            list(pruned.keys()),
            ["step_by_step_analysis", "reasoning_summary", "relevant_pages", "final_answer"],
        )
        self.assertNotIn("confidence", pruned)

    def test_deterministic_split_is_stable_and_group_key_reads_meta(self):
        record = {
            "messages": [],
            "meta": {
                "sample_id": "sft-000001",
                "doc_ids": ["600410_2024_20250426"],
                "company_name": "华胜天成",
            },
        }

        split_a = deterministic_split_for_key(
            build_split_group_key(record, ["doc_ids", "company_name"]),
            dev_ratio=0.1,
            test_ratio=0.1,
            salt="finarag_generator_v1",
        )
        split_b = deterministic_split_for_key(
            build_split_group_key(record, ["doc_ids", "company_name"]),
            dev_ratio=0.1,
            test_ratio=0.1,
            salt="finarag_generator_v1",
        )

        self.assertEqual(split_a, split_b)
        self.assertIn(split_a, {"train", "dev", "test"})

    def test_build_chat_record_preserves_messages_and_meta(self):
        record = {
            "sample_id": "sft-000001",
            "query_id": "seed-000001",
            "schema": "number",
            "company_name": "华胜天成",
            "doc_ids": ["600410_2024_20250426"],
            "source": "teacher_filtered",
            "accepted_checks": ["retrieval_present", "table_grounded"],
            "retrieval_pages": [9],
            "should_refuse": False,
            "system_prompt": "You are a RAG answering system. Return valid JSON only.",
            "user_prompt": "Question goes here",
            "assistant_response": {
                "step_by_step_analysis": "分析",
                "reasoning_summary": "总结",
                "relevant_pages": [9],
                "final_answer": 4270629476.42,
            },
        }

        chat_record = build_chat_record(record)

        self.assertEqual([message["role"] for message in chat_record["messages"]], ["system", "user", "assistant"])
        self.assertEqual(chat_record["meta"]["sample_id"], "sft-000001")
        self.assertEqual(chat_record["meta"]["doc_ids"], ["600410_2024_20250426"])
        self.assertIn('"final_answer":4270629476.42', chat_record["messages"][2]["content"])

    def test_build_export_record_matches_pointwise_example_shape(self):
        record = {
            "pair_id": "pair-000001",
            "query_id": "seed-000001",
            "candidate_id": "cand-0001",
            "query": "华胜天成2024年年报中的营业收入是多少元？",
            "passage": "公司简介和主要财务指标显示，报告期内公司实现营业收入4,270,629,476.42元。",
            "schema": "number",
            "teacher_score": 0.9382,
            "teacher_rank": 1,
            "hard_label": 2,
            "label_source": ["teacher_reranker", "answer_relevant_page"],
            "doc_id": "600410_2024_20250426",
            "page": 9,
            "chunk_id": 321,
            "base_score": 0.8421,
            "retrieval_sources": ["vector", "bm25"],
            "section_name": "公司简介和主要财务指标",
            "is_hard_negative": False,
        }

        export_record = build_export_record(record)

        self.assertEqual(export_record["query"], record["query"])
        self.assertEqual(export_record["passage"], record["passage"])
        self.assertEqual(export_record["teacher_score"], 0.9382)
        self.assertEqual(export_record["meta"]["pair_id"], "pair-000001")
        self.assertEqual(export_record["meta"]["teacher_rank"], 1)

    def test_assign_hard_label_prefers_direct_page_hits(self):
        evidence = EvidenceSummary(
            query_id="seed-0001",
            question_text="营业收入是多少？",
            schema="number",
            query_doc_ids={"600410_2024_20250426"},
            positive_page_keys={("600410_2024_20250426", 9)},
            citation_page_keys={("600410_2024_20250426", 9)},
            table_grounding_keys={("600410_2024_20250426", 9)},
            positive_doc_ids={"600410_2024_20250426"},
        )
        score_record = {
            "doc_id": "600410_2024_20250426",
            "page": 9,
            "teacher_score": 0.91,
            "teacher_rank": 1,
            "schema": "number",
        }

        hard_label, label_source, is_hard_negative = assign_hard_label(score_record, evidence, LabelConfig())

        self.assertEqual(hard_label, 2)
        self.assertIn("answer_relevant_page", label_source)
        self.assertIn("citation_hit", label_source)
        self.assertIn("table_grounding_hit", label_source)
        self.assertFalse(is_hard_negative)

    def test_assign_hard_label_uses_neighbor_page_as_medium_positive(self):
        evidence = EvidenceSummary(
            query_id="seed-0001",
            question_text="营业收入是多少？",
            schema="number",
            query_doc_ids={"600410_2024_20250426"},
            positive_page_keys={("600410_2024_20250426", 9)},
            citation_page_keys=set(),
            table_grounding_keys=set(),
            positive_doc_ids={"600410_2024_20250426"},
        )
        score_record = {
            "doc_id": "600410_2024_20250426",
            "page": 10,
            "teacher_score": 0.22,
            "teacher_rank": 6,
            "schema": "number",
        }

        hard_label, label_source, is_hard_negative = assign_hard_label(
            score_record,
            evidence,
            LabelConfig(neighbor_page_window=1),
        )

        self.assertEqual(hard_label, 1)
        self.assertIn("answer_neighbor_page", label_source)
        self.assertFalse(is_hard_negative)

    def test_assign_hard_label_marks_same_report_low_score_as_hard_negative(self):
        evidence = EvidenceSummary(
            query_id="seed-0001",
            question_text="营业收入是多少？",
            schema="number",
            query_doc_ids={"600410_2024_20250426"},
            positive_page_keys={("600410_2024_20250426", 9)},
            citation_page_keys=set(),
            table_grounding_keys=set(),
            positive_doc_ids={"600410_2024_20250426"},
        )
        score_record = {
            "doc_id": "600410_2024_20250426",
            "page": 3,
            "teacher_score": 0.11,
            "teacher_rank": 17,
            "schema": "number",
        }

        hard_label, label_source, is_hard_negative = assign_hard_label(score_record, evidence, LabelConfig())

        self.assertEqual(hard_label, 0)
        self.assertIn("hard_negative_same_report", label_source)
        self.assertTrue(is_hard_negative)

    def test_filter_rejects_weak_legal_representative_grounding(self):
        record = {
            "query_id": "seed-legal-1",
            "question_text": "国网信通2024年年报中的法定代表人是谁？",
            "schema": "name",
            "doc_ids": ["600131_2024_20250425"],
            "company_name": "国网信通",
            "rag_context": "Text retrieved from page 97\n公司负责人：王奔\n主管会计工作负责人：向杰",
            "retrieval_pages": [97],
            "retrieval_results": [{"page": 97, "text": "公司负责人：王奔"}],
            "teacher_answer_provider": "sub2api",
            "teacher_answer_model": "gpt-5.4",
            "answer": {
                "step_by_step_analysis": "分析",
                "reasoning_summary": "总结",
                "relevant_pages": [97],
                "final_answer": "王奔",
            },
            "validation_result": {"validation_flags": []},
        }

        with self.assertRaisesRegex(ValueError, "weak_legal_representative_grounding"):
            _build_sample_record(record, 1)

    def test_filter_accepts_explicit_legal_representative_grounding(self):
        record = {
            "query_id": "seed-legal-2",
            "question_text": "民生银行2024年年报中的法定代表人是谁？",
            "schema": "name",
            "doc_ids": ["600016_2024_20250329"],
            "company_name": "民生银行",
            "rag_context": "Text retrieved from page 12\n公司法定代表人：高迎欣",
            "retrieval_pages": [12, 164],
            "retrieval_results": [{"page": 12, "text": "公司法定代表人：高迎欣"}],
            "teacher_answer_provider": "sub2api",
            "teacher_answer_model": "gpt-5.4",
            "answer": {
                "step_by_step_analysis": "分析",
                "reasoning_summary": "总结",
                "relevant_pages": [12],
                "final_answer": "高迎欣",
            },
            "validation_result": {"validation_flags": []},
        }

        sample_record, validation_flags = _build_sample_record(record, 1)

        self.assertEqual(sample_record["assistant_response"]["final_answer"], "高迎欣")
        self.assertEqual(validation_flags, [])


if __name__ == "__main__":
    unittest.main()
