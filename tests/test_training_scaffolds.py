import json
import os
import sys
import tempfile
import time
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
    reset_output_file,
    resolve_dataset_root,
)
from training.generator_sft.scripts.build_seed_queries import _normalize_base_question, main as build_seed_main
from training.generator_sft.scripts.convert_to_chat_sft import build_chat_record
from training.generator_sft.scripts.filter_sft_samples import _build_sample_record
from training.generator_sft.scripts.backfill_cross_doc_retrieval_from_citations import (
    backfill_record_from_citations,
)
from training.generator_sft.scripts.mine_teacher_answers import _reset_output_file, _run_teacher_answer
from training.generator_sft.scripts.recheck_hard_context_samples import (
    _text_answers_semantically_consistent,
)
from training.generator_sft.scripts.build_wrong_context_refusal import (
    _compose_source_candidates,
    _prepare_filtered_record_for_teacher_validation,
    _run_record_tasks,
)
from training.generator_sft.validators import _BOOLEAN_TARGET_REGISTRY, _NAME_FIELD_REGISTRY
from training.reranker_distill.scripts.build_distill_dataset import build_stage_commands
from training.reranker_distill.scripts.build_pointwise_labels import (
    EvidenceSummary,
    LabelConfig,
    assign_hard_label,
)
from training.reranker_distill.scripts.collect_candidate_pool import build_candidate_payload
from training.reranker_distill.scripts.score_with_teacher_reranker import _extract_scores
from training.reranker_distill.scripts.build_pointwise_labels import main as build_pointwise_labels_main
from training.reranker_distill.scripts.export_for_trainer import build_export_record
from training.reranker_distill.scripts.export_to_qwen3_reranker_sft import (
    BinaryLabelPolicy,
    build_qwen3_reranker_prompt,
    build_qwen3_reranker_sft_record,
)
from training.reranker_distill.scripts.train_qwen3_reranker_sft import (
    build_supervised_example,
    build_trainer_kwargs,
    build_training_arguments_kwargs,
)


class TrainingScaffoldTests(unittest.TestCase):
    class _FakeTokenizer:
        def encode(self, text, add_special_tokens=False, truncation=False, max_length=None):
            token_ids = [ord(char) for char in text]
            if truncation and max_length is not None:
                token_ids = token_ids[:max_length]
            return token_ids

    def test_reset_output_file_truncates_existing_content(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "teacher_answers_raw.jsonl"
            output_path.write_text('{"query_id":"seed-1"}\n', encoding="utf-8")

            _reset_output_file(output_path)

            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.read_text(encoding="utf-8"), "")

    def test_common_reset_output_file_truncates_existing_content(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "pointwise_labels_raw.jsonl"
            output_path.write_text('{"pair_id":"pair-1"}\n', encoding="utf-8")

            reset_output_file(output_path)

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

    def test_normalize_training_query_record_promotes_legacy_comparative_question(self):
        record = {
            "id": "zh-top10-01-06",
            "text": "在2024年年报中，电科数字和泛微网络谁的营业收入更高？",
            "kind": "name",
            "doc_ids": ["600850_2024_20250415", "603039_2024_20250328"],
            "mentioned_companies": ["电科数字", "泛微网络"],
            "capability": "cross_doc_compare",
        }

        normalized = normalize_training_query_record(record)

        self.assertEqual(normalized["schema"], "comparative")
        self.assertEqual(normalized["task_type"], "cross_doc_compare")

    def test_normalize_training_query_record_preserves_explicit_text_cross_doc_schema(self):
        record = {
            "query_id": "seed-cross-doc-text-001",
            "question_text": "对比甲公司和乙公司2024年年报中的业务发展表述，概括共同点和差异。",
            "schema": "text",
            "task_type": "cross_doc_compare",
            "doc_ids": ["600001_2024_20250330", "600002_2024_20250330"],
            "mentioned_companies": ["甲公司", "乙公司"],
        }

        normalized = normalize_training_query_record(record)

        self.assertEqual(normalized["schema"], "text")
        self.assertEqual(normalized["task_type"], "cross_doc_compare")

    def test_run_teacher_answer_routes_comparative_queries_to_comparative_processor(self):
        class FakeProcessor:
            def __init__(self):
                self.calls = []

            def process_comparative_question(self, question_text, companies, schema):
                self.calls.append(("comparative", question_text, tuple(companies), schema))
                return {"mode": "comparative"}

            def get_answer_for_company(self, *args, **kwargs):
                raise AssertionError("comparative queries should not use get_answer_for_company")

        processor = FakeProcessor()
        result = _run_teacher_answer(
            processor,
            {
                "normalized": {
                    "schema": "comparative",
                    "question_text": "在2024年年报中，电科数字和泛微网络谁的营业收入更高？",
                },
                "doc_ids": ["600850_2024_20250415", "603039_2024_20250328"],
                "mentioned_companies": ["电科数字", "泛微网络"],
                "company_name": None,
                "query_plan": None,
                "route_info": None,
            },
        )

        self.assertEqual(result["mode"], "comparative")
        self.assertEqual(len(processor.calls), 1)

    def test_run_teacher_answer_routes_text_cross_doc_queries_to_comparative_processor(self):
        class FakeProcessor:
            def __init__(self):
                self.calls = []

            def process_comparative_question(self, question_text, companies, schema):
                self.calls.append(("comparative", question_text, tuple(companies), schema))
                return {"mode": "comparative_text"}

            def get_answer_for_company(self, *args, **kwargs):
                raise AssertionError("cross-doc text queries should not use get_answer_for_company")

        processor = FakeProcessor()
        result = _run_teacher_answer(
            processor,
            {
                "normalized": {
                    "schema": "text",
                    "task_type": "cross_doc_compare",
                    "question_text": "对比甲公司和乙公司2024年年报中的业务发展表述，概括共同点和差异。",
                },
                "doc_ids": ["600001_2024_20250330", "600002_2024_20250330"],
                "mentioned_companies": ["甲公司", "乙公司"],
                "company_name": None,
                "query_plan": None,
                "route_info": None,
            },
        )

        self.assertEqual(result["mode"], "comparative_text")
        self.assertEqual(processor.calls[0][3], "text")

    def test_query_template_catalog_targets_align_with_validators(self):
        catalog = load_yaml_mapping(
            Path(__file__).resolve().parents[1] / "training/generator_sft/configs/query_templates.v2.yaml"
        )
        templates = catalog["templates"]
        name_targets = {
            str(template["target_key"])
            for template in templates
            if str(template.get("schema")) == "name"
        }
        boolean_targets = {
            str(template["target_key"])
            for template in templates
            if str(template.get("schema")) == "boolean"
        }
        number_targets = {
            str(template["target_key"])
            for template in templates
            if str(template.get("schema")) == "number"
        }

        self.assertTrue(name_targets.issubset({str(item["field_key"]) for item in _NAME_FIELD_REGISTRY}))
        self.assertTrue(boolean_targets.issubset({str(item["target_key"]) for item in _BOOLEAN_TARGET_REGISTRY}))
        self.assertEqual(
            number_targets,
            {
                "revenue",
                "attributable_net_profit",
                "deducted_attributable_net_profit",
                "operating_cashflow_net",
                "basic_eps",
                "asset_liability_ratio",
            },
        )

    def test_build_seed_v2_generates_all_schema_families_and_drops_legacy_single_doc_number(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            metadata_dir = tmp_root / "metadata_store"
            metadata_dir.mkdir(parents=True, exist_ok=True)

            questions_path = tmp_root / "questions.json"
            questions_payload = [
                {
                    "id": "legacy-sec-001",
                    "text": "在甲公司2024年年报《公司简介和主要财务指标》章节中，法定代表人是谁？",
                    "kind": "name",
                    "doc_ids": ["600001_2024_20250330"],
                    "company_name": "甲公司",
                    "report_year": 2024,
                    "section_name": "公司简介和主要财务指标",
                    "annotation_status": "auto_filled",
                },
                {
                    "id": "legacy-num-001",
                    "text": "甲公司2024年年报中的营业收入是多少元？",
                    "kind": "number",
                    "doc_ids": ["600001_2024_20250330"],
                    "company_name": "甲公司",
                    "report_year": 2024,
                    "annotation_status": "auto_filled",
                },
                {
                    "id": "legacy-cmp-001",
                    "text": "在2024年年报中，甲公司和乙公司谁的营业收入更高？",
                    "kind": "name",
                    "doc_ids": ["600001_2024_20250330", "600002_2024_20250330"],
                    "mentioned_companies": ["甲公司", "乙公司"],
                    "report_year": 2024,
                    "capability": "cross_doc_compare",
                    "annotation_status": "auto_filled",
                },
            ]
            questions_path.write_text(json.dumps(questions_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            annual_report_path = metadata_dir / "annual_report.jsonl"
            annual_reports = [
                {
                    "doc_id": "600001_2024_20250330",
                    "report_id": "600001_2024_20250330",
                    "stock_code": "600001",
                    "company_name": "甲公司",
                    "report_year": 2024,
                    "industry_l1": "信息技术",
                    "board": "沪主板",
                },
                {
                    "doc_id": "600002_2024_20250330",
                    "report_id": "600002_2024_20250330",
                    "stock_code": "600002",
                    "company_name": "乙公司",
                    "report_year": 2024,
                    "industry_l1": "信息技术",
                    "board": "沪主板",
                },
                {
                    "doc_id": "600003_2024_20250330",
                    "report_id": "600003_2024_20250330",
                    "stock_code": "600003",
                    "company_name": "丙公司",
                    "report_year": 2024,
                    "industry_l1": "信息技术",
                    "board": "沪主板",
                },
                {
                    "doc_id": "600004_2024_20250330",
                    "report_id": "600004_2024_20250330",
                    "stock_code": "600004",
                    "company_name": "丁公司",
                    "report_year": 2024,
                    "industry_l1": "信息技术",
                    "board": "沪主板",
                },
            ]
            with annual_report_path.open("w", encoding="utf-8") as handle:
                for record in annual_reports:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            chunk_metadata_path = metadata_dir / "chunk_metadata.jsonl"
            chunk_rows = []
            for index, report in enumerate(annual_reports[:3], start=1):
                chunk_rows.append(
                    {
                        "doc_id": report["doc_id"],
                        "report_year": 2024,
                        "industry_l1": "信息技术",
                        "board": "沪主板",
                        "strategy_tags": ["国产替代", "数字化转型"] if index == 1 else ["国产替代"],
                    }
                )
            with chunk_metadata_path.open("w", encoding="utf-8") as handle:
                for row in chunk_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            output_path = tmp_root / "seed_queries.jsonl"
            stats_output_path = tmp_root / "seed_build_stats.json"
            config_path = tmp_root / "build_seed.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"dataset_root_path: {tmp_root}",
                        f"questions_path: {questions_path}",
                        f"annual_report_path: {annual_report_path}",
                        f"chunk_metadata_path: {chunk_metadata_path}",
                        f"output_path: {output_path}",
                        f"stats_output_path: {stats_output_path}",
                        "template_catalog_path: training/generator_sft/configs/query_templates.v2.yaml",
                        "template_version: v2",
                        "legacy_questions_mode: supplement_multidoc_only",
                        "include_template_bootstrap: true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("sys.argv", ["build_seed_queries.py", "--config-path", str(config_path)]):
                build_seed_main()

            records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            stats = json.loads(stats_output_path.read_text(encoding="utf-8"))

        schemas = {record["schema"] for record in records}
        self.assertTrue({"name", "number", "boolean", "names", "comparative"}.issubset(schemas))
        self.assertFalse(any(record["query_id"] == "seed-legacy-num-001" for record in records))
        self.assertTrue(any(record["query_id"] == "seed-legacy-cmp-001" and record["schema"] == "comparative" for record in records))
        self.assertTrue(all(3 <= len(record["doc_ids"]) <= 20 for record in records if record["schema"] == "names"))
        self.assertIn("template_family_counts", stats)
        self.assertIn("target_key_counts", stats)
        self.assertIn("split_pool_counts", stats)
        self.assertIn("surface_variant_counts", stats)
        self.assertEqual(stats["legacy_question_dropped_count"], 1)

    def test_build_seed_main_supports_text_and_long_text_templates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            metadata_dir = tmp_root / "metadata_store"
            metadata_dir.mkdir(parents=True, exist_ok=True)

            questions_path = tmp_root / "questions.json"
            questions_path.write_text("[]\n", encoding="utf-8")

            annual_report_path = metadata_dir / "annual_report.jsonl"
            annual_reports = [
                {
                    "doc_id": "600001_2024_20250330",
                    "report_id": "600001_2024_20250330",
                    "stock_code": "600001",
                    "company_name": "甲公司",
                    "report_year": 2024,
                    "industry_l1": "信息技术",
                    "board": "沪主板",
                },
                {
                    "doc_id": "600002_2024_20250330",
                    "report_id": "600002_2024_20250330",
                    "stock_code": "600002",
                    "company_name": "乙公司",
                    "report_year": 2024,
                    "industry_l1": "信息技术",
                    "board": "沪主板",
                },
            ]
            with annual_report_path.open("w", encoding="utf-8") as handle:
                for record in annual_reports:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            chunk_metadata_path = metadata_dir / "chunk_metadata.jsonl"
            chunk_metadata_path.write_text("", encoding="utf-8")

            template_catalog_path = tmp_root / "query_templates.gen120_text_test.yaml"
            template_catalog_path.write_text(
                "\n".join(
                    [
                        "version: v2",
                        "templates:",
                        "  - template_id: long_text_summary_test",
                        "    template_family: gen120_summary",
                        "    schema: long_text",
                        "    task_type: generation_summary",
                        "    generation_mode: per_doc",
                        "    target_key: summary_synthesis",
                        "    surface_forms:",
                        "      - \"请结合{company_name}{report_year}年年报，概括年度经营表现变化的主要原因，列出2-3点。\"",
                        "    section_hints: []",
                        "    split_pool: core_single_doc",
                        "    answer_policy: evidence_grounded_synthesis",
                        "    validator_target: generic_query_v1",
                        "    max_per_doc: 1",
                        "    difficulty: medium",
                        "  - template_id: text_cross_doc_compare_test",
                        "    template_family: gen120_cross_doc",
                        "    schema: text",
                        "    task_type: cross_doc_compare",
                        "    generation_mode: per_pair",
                        "    target_key: cross_doc_compare_synthesis",
                        "    surface_forms:",
                        "      - \"对比{company_a}和{company_b}{year}年年报中的业务发展表述，概括共同点和差异。\"",
                        "    section_hints: []",
                        "    split_pool: aux_multidoc",
                        "    answer_policy: evidence_grounded_synthesis",
                        "    validator_target: generic_query_v1",
                        "    max_per_doc: 1",
                        "    difficulty: hard",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            output_path = tmp_root / "seed_queries.jsonl"
            stats_output_path = tmp_root / "seed_build_stats.json"
            config_path = tmp_root / "build_seed.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"dataset_root_path: {tmp_root}",
                        f"questions_path: {questions_path}",
                        f"annual_report_path: {annual_report_path}",
                        f"chunk_metadata_path: {chunk_metadata_path}",
                        f"output_path: {output_path}",
                        f"stats_output_path: {stats_output_path}",
                        f"template_catalog_path: {template_catalog_path}",
                        "template_version: v2",
                        "legacy_questions_mode: none",
                        "include_template_bootstrap: true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("sys.argv", ["build_seed_queries.py", "--config-path", str(config_path)]):
                build_seed_main()

            records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        long_text_records = [record for record in records if record["schema"] == "long_text"]
        cross_doc_text_records = [
            record
            for record in records
            if record["schema"] == "text" and record["task_type"] == "cross_doc_compare"
        ]
        self.assertEqual(len(long_text_records), 2)
        self.assertEqual(len(cross_doc_text_records), 1)
        self.assertEqual(cross_doc_text_records[0]["mentioned_companies"], ["甲公司", "乙公司"])
        self.assertEqual(len(cross_doc_text_records[0]["doc_ids"]), 2)

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

    def test_text_prompt_bundle_and_pruned_answer_follow_schema(self):
        system_prompt, user_prompt, response_format = build_rag_prompt_bundle("text", provider="qwen")
        pruned = prune_answer_to_schema(
            {
                "step_by_step_analysis": "先看经营情况，再看业务布局。",
                "reasoning_summary": "经营表现改善主要来自产品结构优化和订单恢复。",
                "relevant_pages": [9, 12],
                "final_answer": "公司通过订单恢复和产品结构优化改善了经营表现。",
                "confidence": "high",
            },
            schema="text",
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

    def test_extract_scores_preserves_teacher_scale_without_query_level_minmax(self):
        scores = _extract_scores(
            {
                "results": [
                    {"index": 1, "relevance_score": 1.2},
                    {"index": 0, "relevance_score": 2.5},
                ]
            },
            total_documents=2,
        )

        self.assertEqual(scores, [2.5, 1.2])

    def test_build_stage_commands_wires_train_dev_test_exports(self):
        settings = {
            "python_bin": "/usr/bin/python3",
            "data_config_path": Path("/tmp/data.yaml"),
            "split_config_path": Path("/tmp/split.yaml"),
            "export_config_path": Path("/tmp/export.yaml"),
            "resume": True,
            "start_stage": "collect",
            "end_stage": "export",
            "export_train_input_path": Path("/tmp/pointwise_train_raw.jsonl"),
            "export_train_output_path": Path("/tmp/pointwise_train.jsonl"),
            "export_train_stats_output_path": Path("/tmp/export_train_stats.json"),
            "export_dev_input_path": Path("/tmp/pointwise_dev_raw.jsonl"),
            "export_dev_output_path": Path("/tmp/pointwise_dev.jsonl"),
            "export_dev_stats_output_path": Path("/tmp/export_dev_stats.json"),
            "export_test_input_path": Path("/tmp/pointwise_test_raw.jsonl"),
            "export_test_output_path": Path("/tmp/pointwise_test.jsonl"),
            "export_test_stats_output_path": Path("/tmp/export_test_stats.json"),
        }

        commands = build_stage_commands(settings)

        self.assertEqual(
            [command["name"] for command in commands],
            ["collect", "score", "label", "split", "export_train", "export_dev", "export_test"],
        )
        self.assertIn("--resume", commands[0]["argv"])
        self.assertIn("--resume", commands[1]["argv"])
        self.assertEqual(commands[5]["argv"][-6:], [
            "--input-path",
            "/tmp/pointwise_dev_raw.jsonl",
            "--output-path",
            "/tmp/pointwise_dev.jsonl",
            "--stats-output-path",
            "/tmp/export_dev_stats.json",
        ])

    def test_reranker_split_config_prefers_doc_and_company_holdout_before_query_id(self):
        config = load_yaml_mapping(
            Path(__file__).resolve().parents[1] / "training/reranker_distill/configs/split.example.yaml"
        )

        self.assertEqual(config["group_fields"], ["doc_id", "company_name", "query_id"])

    def test_build_qwen3_reranker_prompt_uses_native_yes_no_template(self):
        prompt = build_qwen3_reranker_prompt(
            query="华胜天成2024年年报中的营业收入是多少元？",
            passage="报告期内公司实现营业收入4,270,629,476.42元。",
        )

        self.assertIn("<|im_start|>system", prompt)
        self.assertIn("<Instruct>:", prompt)
        self.assertIn("<Query>: 华胜天成2024年年报中的营业收入是多少元？", prompt)
        self.assertIn("<Document>: 报告期内公司实现营业收入4,270,629,476.42元。", prompt)
        self.assertTrue(prompt.endswith("<think>\n\n</think>\n\n"))

    def test_build_qwen3_reranker_sft_record_maps_strict_positive_to_yes(self):
        record = {
            "query": "华胜天成2024年年报中的营业收入是多少元？",
            "passage": "报告期内公司实现营业收入4,270,629,476.42元。",
            "teacher_score": 0.9382,
            "hard_label": 2,
            "meta": {
                "pair_id": "pair-000001",
                "query_id": "seed-000001",
            },
        }

        sft_record = build_qwen3_reranker_sft_record(
            record,
            policy=BinaryLabelPolicy(),
        )

        self.assertIsNotNone(sft_record)
        self.assertEqual(sft_record["target"], "yes")
        self.assertEqual(sft_record["label"], 1)
        self.assertEqual(sft_record["meta"]["target_source"], "hard_label")
        self.assertEqual(sft_record["meta"]["pair_id"], "pair-000001")

    def test_build_qwen3_reranker_sft_record_skips_ambiguous_medium_label(self):
        record = {
            "query": "华胜天成2024年年报中的营业收入是多少元？",
            "passage": "公司在多个业务条线继续增长。",
            "teacher_score": 0.62,
            "hard_label": 1,
        }

        sft_record = build_qwen3_reranker_sft_record(
            record,
            policy=BinaryLabelPolicy(
                label_source="hard_label",
                positive_hard_labels=(2,),
                negative_hard_labels=(0,),
            ),
        )

        self.assertIsNone(sft_record)

    def test_build_supervised_example_masks_only_target_tokens(self):
        tokenizer = self._FakeTokenizer()
        record = {
            "query": "营业收入是多少？",
            "passage": "报告期内公司实现营业收入4,270,629,476.42元。",
            "target": "yes",
        }

        example = build_supervised_example(
            record,
            tokenizer,
            cutoff_len=4096,
        )

        target_ids = tokenizer.encode("yes", add_special_tokens=False)
        self.assertEqual(example["labels"][-len(target_ids):], target_ids)
        self.assertTrue(all(value == -100 for value in example["labels"][:-len(target_ids)]))
        self.assertEqual(len(example["input_ids"]), len(example["attention_mask"]))
        self.assertEqual(len(example["input_ids"]), len(example["labels"]))

    def test_training_arguments_kwargs_drop_unsupported_transformers_parameters(self):
        settings = {
            "output_dir": Path("/tmp/qwen3-reranker-test"),
            "learning_rate": 2.0e-4,
            "weight_decay": 0.0,
            "num_train_epochs": 1.0,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "warmup_ratio": 0.05,
            "lr_scheduler_type": "cosine",
            "logging_steps": 10,
            "save_steps": 200,
            "eval_steps": 200,
            "save_total_limit": 3,
            "bf16": True,
            "fp16": False,
            "gradient_checkpointing": True,
            "max_grad_norm": 1.0,
            "optim": "adamw_torch",
        }
        supported_parameters = {
            "output_dir",
            "do_train",
            "do_eval",
            "evaluation_strategy",
            "learning_rate",
            "per_device_train_batch_size",
            "report_to",
        }

        kwargs = build_training_arguments_kwargs(
            settings,
            has_eval_records=True,
            world_size=1,
            supported_parameters=supported_parameters,
        )

        self.assertNotIn("overwrite_output_dir", kwargs)
        self.assertNotIn("eval_strategy", kwargs)
        self.assertEqual(kwargs["evaluation_strategy"], "steps")
        self.assertEqual(kwargs["output_dir"], "/tmp/qwen3-reranker-test")

    def test_trainer_kwargs_uses_processing_class_when_tokenizer_is_unsupported(self):
        kwargs = build_trainer_kwargs(
            model="model",
            training_args="args",
            train_dataset="train",
            eval_dataset="eval",
            tokenizer="tokenizer",
            data_collator="collator",
            preprocess_logits_for_metrics="preprocess",
            compute_metrics="metrics",
            supported_parameters={
                "model",
                "args",
                "train_dataset",
                "eval_dataset",
                "processing_class",
                "data_collator",
                "preprocess_logits_for_metrics",
                "compute_metrics",
            },
        )

        self.assertNotIn("tokenizer", kwargs)
        self.assertEqual(kwargs["processing_class"], "tokenizer")
        self.assertEqual(kwargs["eval_dataset"], "eval")

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

    def test_build_pointwise_labels_filters_queries_failing_quality_gates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            teacher_scores_path = tmp_root / "teacher_scores.jsonl"
            teacher_answers_path = tmp_root / "teacher_answers.jsonl"
            output_path = tmp_root / "pointwise_labels_raw.jsonl"
            rejected_pairs_path = tmp_root / "rejected_pairs.jsonl"
            stats_output_path = tmp_root / "pointwise_stats.json"
            config_path = tmp_root / "data_build.yaml"

            teacher_scores_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "query_id": "q-good",
                                "candidate_id": "cand-0001",
                                "question_text": "营业收入是多少？",
                                "schema": "number",
                                "teacher_reranker_model": "teacher",
                                "teacher_score": 0.95,
                                "teacher_rank": 1,
                                "doc_id": "doc-good",
                                "company_name": "甲公司",
                                "page": 9,
                                "chunk_id": 1,
                                "text": "报告期内公司实现营业收入。",
                                "base_score": 0.91,
                                "retrieval_sources": ["vector"],
                                "section_name": "公司简介和主要财务指标",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "query_id": "q-good",
                                "candidate_id": "cand-0002",
                                "question_text": "营业收入是多少？",
                                "schema": "number",
                                "teacher_reranker_model": "teacher",
                                "teacher_score": 0.12,
                                "teacher_rank": 2,
                                "doc_id": "doc-good",
                                "company_name": "甲公司",
                                "page": 2,
                                "chunk_id": 2,
                                "text": "无关段落1",
                                "base_score": 0.33,
                                "retrieval_sources": ["vector"],
                                "section_name": "其他",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "query_id": "q-good",
                                "candidate_id": "cand-0003",
                                "question_text": "营业收入是多少？",
                                "schema": "number",
                                "teacher_reranker_model": "teacher",
                                "teacher_score": 0.11,
                                "teacher_rank": 3,
                                "doc_id": "doc-good",
                                "company_name": "甲公司",
                                "page": 3,
                                "chunk_id": 3,
                                "text": "无关段落2",
                                "base_score": 0.21,
                                "retrieval_sources": ["vector"],
                                "section_name": "其他",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "query_id": "q-good",
                                "candidate_id": "cand-0004",
                                "question_text": "营业收入是多少？",
                                "schema": "number",
                                "teacher_reranker_model": "teacher",
                                "teacher_score": 0.1,
                                "teacher_rank": 4,
                                "doc_id": "doc-good",
                                "company_name": "甲公司",
                                "page": 4,
                                "chunk_id": 4,
                                "text": "无关段落3",
                                "base_score": 0.2,
                                "retrieval_sources": ["vector"],
                                "section_name": "其他",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "query_id": "q-bad",
                                "candidate_id": "cand-1001",
                                "question_text": "净利润是多少？",
                                "schema": "number",
                                "teacher_reranker_model": "teacher",
                                "teacher_score": 0.62,
                                "teacher_rank": 1,
                                "doc_id": "doc-bad",
                                "company_name": "乙公司",
                                "page": 11,
                                "chunk_id": 11,
                                "text": "可能相关但不是直接答案",
                                "base_score": 0.44,
                                "retrieval_sources": ["vector"],
                                "section_name": "其他",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "query_id": "q-bad",
                                "candidate_id": "cand-1002",
                                "question_text": "净利润是多少？",
                                "schema": "number",
                                "teacher_reranker_model": "teacher",
                                "teacher_score": 0.12,
                                "teacher_rank": 2,
                                "doc_id": "doc-bad",
                                "company_name": "乙公司",
                                "page": 12,
                                "chunk_id": 12,
                                "text": "无关段落A",
                                "base_score": 0.22,
                                "retrieval_sources": ["vector"],
                                "section_name": "其他",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "query_id": "q-bad",
                                "candidate_id": "cand-1003",
                                "question_text": "净利润是多少？",
                                "schema": "number",
                                "teacher_reranker_model": "teacher",
                                "teacher_score": 0.1,
                                "teacher_rank": 3,
                                "doc_id": "doc-bad",
                                "company_name": "乙公司",
                                "page": 13,
                                "chunk_id": 13,
                                "text": "无关段落B",
                                "base_score": 0.2,
                                "retrieval_sources": ["vector"],
                                "section_name": "其他",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            teacher_answers_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "query_id": "q-good",
                                "question_text": "营业收入是多少？",
                                "schema": "number",
                                "doc_ids": ["doc-good"],
                                "answer": {
                                    "relevant_pages": [9],
                                    "references": [{"pdf_sha1": "doc-good", "page_index": 9}],
                                },
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "query_id": "q-bad",
                                "question_text": "净利润是多少？",
                                "schema": "number",
                                "doc_ids": ["doc-bad"],
                                "answer": {
                                    "relevant_pages": [],
                                    "references": [],
                                },
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config_path.write_text(
                "\n".join(
                    [
                        f"teacher_scores_path: {teacher_scores_path}",
                        f"teacher_answers_path: {teacher_answers_path}",
                        f"pointwise_output_path: {output_path}",
                        f"rejected_pairs_path: {rejected_pairs_path}",
                        f"pointwise_stats_output_path: {stats_output_path}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                ["build_pointwise_labels.py", "--config-path", str(config_path)],
            ):
                build_pointwise_labels_main()

            output_records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            rejected_records = [
                json.loads(line)
                for line in rejected_pairs_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual({record["query_id"] for record in output_records}, {"q-good"})
            self.assertTrue(all(record["query_id"] == "q-bad" for record in rejected_records))
            self.assertTrue(
                all(
                    "quality_gate_missing_label2_or_min_negatives" in record["reason"]
                    for record in rejected_records
                )
            )

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

    def test_backfill_cross_doc_record_from_citations_restores_retrieval_payload(self):
        raw_record = {
            "query_id": "seed-cross-doc-1",
            "question_text": "请比较甲公司和乙公司2024年的业务推进差异。",
            "schema": "text",
            "task_type": "cross_doc_compare",
            "doc_ids": ["doc-a", "doc-b"],
            "mentioned_companies": ["甲公司", "乙公司"],
            "rag_context": "",
            "retrieval_pages": [10, 14],
            "retrieval_results": [],
            "answer": {
                "final_answer": "两家公司都推进新业务，但方向不同。",
                "relevant_pages": [10, 14],
                "retrieval_pages": [10, 14],
                "retrieval_results": [],
                "citations": [
                    {
                        "page": 10,
                        "chunk_id": 51,
                        "chunk_type": "content",
                        "node_type": "parent",
                        "parent_chunk_id": None,
                        "matched_child_chunk_ids": [57, 58],
                        "matched_tags": ["技术创新"],
                        "section_title": "董事会报告",
                        "section_name": "董事会报告",
                        "report_section": "董事会报告",
                        "source": "doc-a",
                        "company_name": "甲公司",
                        "security_code": "600001",
                        "stock_code": "600001",
                        "currency": "CNY",
                        "report_year": 2024,
                        "doc_source_type": "annual_report",
                        "retrieval_sources": ["bm25", "tag"],
                        "evidence_snippet": "甲公司围绕技术创新推进新业务布局。",
                        "score": 0.81,
                    },
                    {
                        "page": 14,
                        "chunk_id": 88,
                        "chunk_type": "content",
                        "node_type": "parent",
                        "parent_chunk_id": None,
                        "matched_child_chunk_ids": [91],
                        "matched_tags": ["海外拓展"],
                        "section_title": "管理层讨论",
                        "section_name": "管理层讨论",
                        "report_section": "管理层讨论",
                        "source": "doc-b",
                        "company_name": "乙公司",
                        "security_code": "600002",
                        "stock_code": "600002",
                        "currency": "CNY",
                        "report_year": 2024,
                        "doc_source_type": "annual_report",
                        "retrieval_sources": ["vector"],
                        "evidence_snippet": "乙公司围绕海外市场拓展推进业务。",
                        "score": 0.79,
                    },
                ],
            },
        }

        updated = backfill_record_from_citations(raw_record)

        self.assertTrue(updated["rag_context"])
        self.assertEqual(len(updated["retrieval_results"]), 2)
        self.assertEqual(updated["answer"]["retrieval_pages"], [10, 14])
        self.assertEqual(len(updated["answer"]["retrieval_results"]), 2)
        self.assertEqual(updated["retrieval_results"][0]["sha1_name"], "doc-a")
        self.assertIn("company: 甲公司", updated["rag_context"])
        self.assertIn("company: 乙公司", updated["rag_context"])

    def test_recheck_text_semantic_match_accepts_close_paraphrase(self):
        parent_answer = (
            "公司围绕科技金融、供应链金融和数字化转型推进年度战略布局，"
            "同时强化风险管理与客户服务能力。"
        )
        rechecked_answer = (
            "2024年公司主要通过科技金融、供应链金融以及数智化建设推进战略落地，"
            "并同步加强风控和客户服务能力。"
        )

        consistent, metrics = _text_answers_semantically_consistent(parent_answer, rechecked_answer)

        self.assertTrue(consistent)
        self.assertGreater(metrics["sequence_ratio"], 0.4)

    def test_recheck_text_semantic_match_rejects_unrelated_answer(self):
        parent_answer = "公司通过科技金融、供应链金融和数字化转型推动年度战略。"
        rechecked_answer = "公司主要面临信用风险、市场风险和流动性风险。"

        consistent, metrics = _text_answers_semantically_consistent(parent_answer, rechecked_answer)

        self.assertFalse(consistent)
        self.assertLess(metrics["sequence_ratio"], 0.5)

    def test_wrong_context_compose_source_candidates_skips_lazy_loader_when_not_needed(self):
        groups = {
            "same_company_wrong_section": [
                {"doc_id": "doc-a", "page": 1, "text": "alpha", "candidate_id": "cand-1"}
            ],
            "same_metric_wrong_page": [],
            "support_replaced_high_similarity_non_support": [],
            "same_year_wrong_company": [],
        }
        loader_calls = []

        def source_loader(source_name):
            loader_calls.append(source_name)
            return [{"doc_id": "doc-b", "page": 2, "text": "beta", "candidate_id": "cand-2"}]

        selected, source_mix = _compose_source_candidates(
            "same_company_wrong_section",
            groups,
            min_context_results=1,
            max_context_results=2,
            source_loader=source_loader,
        )

        self.assertEqual(loader_calls, [])
        self.assertEqual(len(selected), 1)
        self.assertEqual(source_mix, ["same_company_wrong_section"])

    def test_wrong_context_compose_source_candidates_lazily_loads_same_year_when_needed(self):
        groups = {
            "same_company_wrong_section": [],
            "same_metric_wrong_page": [],
            "support_replaced_high_similarity_non_support": [],
            "same_year_wrong_company": [],
        }
        loader_calls = []

        def source_loader(source_name):
            loader_calls.append(source_name)
            return [{"doc_id": "doc-b", "page": 2, "text": "beta", "candidate_id": "cand-2"}]

        selected, source_mix = _compose_source_candidates(
            "same_year_wrong_company",
            groups,
            min_context_results=1,
            max_context_results=2,
            source_loader=source_loader,
        )

        self.assertEqual(loader_calls, ["same_year_wrong_company"])
        self.assertEqual(len(selected), 1)
        self.assertEqual(source_mix, ["same_year_wrong_company"])

    def test_wrong_context_parallel_worker_pool_preserves_input_order(self):
        def worker(value):
            time.sleep(max(0.0, 0.03 - (value * 0.005)))
            return value * 2

        results = _run_record_tasks([1, 2, 3, 4], worker, max_workers=2)

        self.assertEqual(results, [2, 4, 6, 8])

    def test_wrong_context_prepare_does_not_build_processor_without_same_year_need(self):
        record = {
            "sample_id": "sample-1",
            "query_id": "q1",
            "schema": "text",
            "question_text": "概括公司战略。",
            "company_name": "甲公司",
            "doc_ids": ["doc-a"],
            "assistant_response": {"final_answer": "公司推进数字化转型。"},
        }
        cache_by_query_id = {
            "q1": {
                "query_id": "q1",
                "retrieval_results": [
                    {
                        "page": 1,
                        "chunk_id": "support",
                        "section_name": "战略",
                        "text": "公司推进数字化转型。",
                    }
                ],
            }
        }
        candidate_by_query_id = {
            "q1": {
                "query_id": "q1",
                "candidates": [
                    {
                        "doc_id": "doc-a",
                        "page": 2,
                        "chunk_id": "wrong",
                        "section_name": "风险",
                        "text": "公司面临市场竞争风险。",
                    }
                ],
            }
        }
        settings = {
            "source_priority": ["same_company_wrong_section"],
            "min_context_results": 1,
            "max_context_results": 1,
            "max_support_results": 1,
            "candidate_pool_size": 16,
            "per_query_retrieval_top_k": 12,
            "parallel_requests": 4,
            "retrieval_config_path": Path("unused.yaml"),
            "dataset_root_path": Path("."),
            "cross_doc_candidate_doc_cap": 8,
            "max_doc_chars": 1200,
            "max_context_chars": 7000,
            "teacher_answer_provider": "sub2api",
        }
        processor_calls = []

        def processor_factory():
            processor_calls.append("called")
            raise AssertionError("processor should be lazy and unused")

        result = _prepare_filtered_record_for_teacher_validation(
            record,
            settings=settings,
            cache_by_query_id=cache_by_query_id,
            candidate_by_query_id=candidate_by_query_id,
            processor_factory=processor_factory,
        )

        self.assertEqual(result["status"], "prepared")
        self.assertEqual(len(result["prepared_attempts"]), 1)
        self.assertEqual(processor_calls, [])

    def test_gen120_text_postprocess_script_exists_and_covers_pipeline_steps(self):
        script_path = Path("/media/main/lgd/llm/FinaRAG/training/generator_sft/run_gen120_text_postprocess.sh")

        self.assertTrue(script_path.exists(), f"missing script: {script_path}")

        content = script_path.read_text(encoding="utf-8")
        self.assertIn("collect_candidate_pool.py", content)
        self.assertIn("build_hard_context_samples.py", content)
        self.assertIn("recheck_hard_context_samples.py", content)
        self.assertIn("build_truncated_refusal.py", content)
        self.assertIn("build_wrong_context_refusal.py", content)


if __name__ == "__main__":
    unittest.main()
