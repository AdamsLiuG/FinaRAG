from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.quantization.common import (
    build_awq_quant_config,
    discover_default_calibration_path,
    load_calibration_texts,
)
from training.quantization.merge_lora import (
    build_merge_command,
    write_merge_export_config,
)
from training.quantization.pipeline import run_pipeline_stage
from training.quantization.quantize_awq import (
    build_llmcompressor_ignore_patterns,
    build_llmcompressor_sequential_targets,
    build_arg_parser as build_quantize_arg_parser,
    patch_llmcompressor_disable_lm_head_for_offloaded_weights,
    prepare_llmcompressor_model_for_oneshot,
    resolve_quantization_backend,
)


class QuantizationPipelineTests(unittest.TestCase):
    _REPO_ROOT = Path(__file__).resolve().parents[1]

    class _FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            del tokenize, add_generation_prompt
            return "\n".join(f"{message['role']}: {message['content']}" for message in messages)

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [ord(char) for char in text]

        def decode(self, token_ids, skip_special_tokens=False):
            del skip_special_tokens
            return "".join(chr(token_id) for token_id in token_ids)

    def test_build_awq_quant_config_defaults(self):
        config = build_awq_quant_config()
        self.assertEqual(config["w_bit"], 4)
        self.assertEqual(config["q_group_size"], 128)
        self.assertTrue(config["zero_point"])
        self.assertEqual(config["version"], "GEMM")

    def test_load_calibration_texts_from_chat_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "samples.jsonl"
            path.write_text(
                '{"messages":[{"role":"user","content":"Q"},{"role":"assistant","content":"A"}]}\n',
                encoding="utf-8",
            )

            texts = load_calibration_texts(
                path,
                dataset_format="chat_jsonl",
                max_samples=8,
                tokenizer=self._FakeTokenizer(),
                max_length=64,
            )

        self.assertEqual(len(texts), 1)
        self.assertIn("user: Q", texts[0])
        self.assertIn("assistant: A", texts[0])

    def test_load_calibration_texts_from_llamafactory_json_instruction_format(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "samples.json"
            path.write_text(
                '[{"instruction":"Question","input":"Context","output":"Answer"}]',
                encoding="utf-8",
            )

            texts = load_calibration_texts(
                path,
                dataset_format="llamafactory_json",
                max_samples=8,
                tokenizer=self._FakeTokenizer(),
                max_length=64,
            )

        self.assertEqual(len(texts), 1)
        self.assertIn("Question", texts[0])
        self.assertIn("Context", texts[0])
        self.assertIn("Answer", texts[0])

    def test_load_calibration_texts_from_reranker_sft_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "reranker_sft.jsonl"
            path.write_text(
                '{"prompt":"<|im_start|>user\\n<Query>: 测试\\n<Document>: 文档<|im_end|>\\n<|im_start|>assistant\\n","target":"yes"}\n',
                encoding="utf-8",
            )

            texts = load_calibration_texts(
                path,
                dataset_format="reranker_sft_jsonl",
                max_samples=8,
                tokenizer=self._FakeTokenizer(),
                max_length=256,
            )

        self.assertEqual(len(texts), 1)
        self.assertIn("<Query>: 测试", texts[0])
        self.assertTrue(texts[0].endswith("yes"))

    def test_load_calibration_texts_from_reranker_pointwise_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "pointwise.jsonl"
            path.write_text(
                '{"query":"测试问题","passage":"测试段落","teacher_score":0.9,"hard_label":2}\n',
                encoding="utf-8",
            )

            texts = load_calibration_texts(
                path,
                dataset_format="reranker_pointwise_jsonl",
                max_samples=8,
                tokenizer=self._FakeTokenizer(),
                max_length=1024,
            )

        self.assertEqual(len(texts), 1)
        self.assertIn("<Query>: 测试问题", texts[0])
        self.assertIn("<Document>: 测试段落", texts[0])

    def test_discover_default_calibration_path_uses_task_type_specific_candidates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            generator_path = Path(tmp_dir) / "generator.jsonl"
            reranker_path = Path(tmp_dir) / "reranker.jsonl"
            reranker_path.write_text("{}", encoding="utf-8")

            with patch(
                "training.quantization.common._DEFAULT_CALIBRATION_CANDIDATES_BY_TASK",
                {
                    "generator": (generator_path,),
                    "reranker": (reranker_path,),
                },
            ):
                discovered = discover_default_calibration_path(task_type="reranker")

        self.assertEqual(discovered, reranker_path)

    def test_write_merge_export_config_contains_base_and_adapter(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "merge.yaml"
            write_merge_export_config(
                config_path=config_path,
                base_model_path="/models/Qwen3.5-9B",
                adapter_path="/adapters/qwen3.5-9b-lora",
                export_dir="/tmp/merged-model",
                template="default",
            )
            text = config_path.read_text(encoding="utf-8")

        self.assertIn("model_name_or_path: /models/Qwen3.5-9B", text)
        self.assertIn("adapter_name_or_path: /adapters/qwen3.5-9b-lora", text)
        self.assertIn("export_dir: /tmp/merged-model", text)

    def test_build_merge_command_uses_export_subcommand(self):
        command = build_merge_command("llamafactory-cli", Path("/tmp/merge.yaml"))
        self.assertEqual(command, ["llamafactory-cli", "export", "/tmp/merge.yaml"])

    def test_quantize_awq_parser_defaults_to_awq_4bit(self):
        parser = build_quantize_arg_parser()
        args = parser.parse_args(["--model-path", "/tmp/merged", "--output-dir", "/tmp/awq"])
        self.assertEqual(args.backend, "auto")
        self.assertEqual(args.task_type, None)
        self.assertEqual(args.w_bit, None)
        self.assertEqual(args.q_group_size, None)

    def test_resolve_quantization_backend_rejects_qwen3_5_without_llmcompressor(self):
        with patch("training.quantization.quantize_awq.AutoConfig.from_pretrained") as mock_config_loader, patch(
            "training.quantization.quantize_awq.find_spec", return_value=None
        ):
            mock_config_loader.return_value.model_type = "qwen3_5"
            with self.assertRaisesRegex(ValueError, "llmcompressor"):
                resolve_quantization_backend(
                    model_path=Path("/tmp/merged"),
                    requested_backend="auto",
                    trust_remote_code=True,
                )

    def test_resolve_quantization_backend_prefers_llmcompressor_for_qwen3_5(self):
        with patch("training.quantization.quantize_awq.AutoConfig.from_pretrained") as mock_config_loader, patch(
            "training.quantization.quantize_awq.find_spec"
        ) as mock_find_spec:
            mock_config_loader.return_value.model_type = "qwen3_5"
            mock_find_spec.side_effect = lambda name: object() if name == "llmcompressor" else None
            backend, model_type = resolve_quantization_backend(
                model_path=Path("/tmp/merged"),
                requested_backend="auto",
                trust_remote_code=True,
            )

        self.assertEqual((backend, model_type), ("llmcompressor", "qwen3_5"))

    def test_resolve_quantization_backend_prefers_llmcompressor_for_reranker_auto(self):
        with patch("training.quantization.quantize_awq.AutoConfig.from_pretrained") as mock_config_loader, patch(
            "training.quantization.quantize_awq.find_spec"
        ) as mock_find_spec:
            mock_config_loader.return_value.model_type = "qwen3"
            mock_find_spec.side_effect = lambda name: object() if name == "llmcompressor" else None
            backend, model_type = resolve_quantization_backend(
                model_path=Path("/tmp/merged"),
                requested_backend="auto",
                trust_remote_code=True,
                task_type="reranker",
            )

        self.assertEqual((backend, model_type), ("llmcompressor", "qwen3"))

    def test_build_llmcompressor_ignore_patterns_skips_qwen3_5_visual_tower(self):
        ignore_patterns = build_llmcompressor_ignore_patterns(
            model_type="qwen3_5",
            task_type="generator",
        )

        self.assertIn("lm_head", ignore_patterns)
        self.assertIn(r"re:^model\.visual(\..*)?$", ignore_patterns)

    def test_build_llmcompressor_sequential_targets_prefers_decoder_layers_for_qwen3_5_generator(self):
        sequential_targets = build_llmcompressor_sequential_targets(
            model_type="qwen3_5",
            task_type="generator",
        )

        self.assertEqual(sequential_targets, ["Qwen3_5DecoderLayer"])

    def test_prepare_llmcompressor_model_for_oneshot_moves_qwen3_5_generator_lm_head_to_cpu_onload(self):
        class _FakeOffloadCache:
            pass

        class _FakeModel:
            def __init__(self):
                self.lm_head = type("_FakeLmHead", (), {"_parameters": _FakeOffloadCache()})()

        fake_model = _FakeModel()
        with patch("training.quantization.quantize_awq.set_onload_device") as mock_set_onload_device, patch(
            "training.quantization.quantize_awq.OffloadCache", _FakeOffloadCache
        ):
            prepare_llmcompressor_model_for_oneshot(
                fake_model,
                model_type="qwen3_5",
                task_type="generator",
            )

        mock_set_onload_device.assert_called_once_with(fake_model.lm_head, "cpu")

    def test_prepare_llmcompressor_model_for_oneshot_moves_accelerate_wrapped_lm_head_to_cpu(self):
        class _FakeHook:
            def __init__(self):
                self.execution_device = "cuda:1"

        class _FakeLinear:
            def __init__(self):
                self._parameters = {}
                self._hf_hook = _FakeHook()
                self.to_calls = []

            def to(self, device):
                self.to_calls.append(device)
                return self

        class _FakeModel:
            def __init__(self):
                self.lm_head = _FakeLinear()

        fake_model = _FakeModel()
        prepare_llmcompressor_model_for_oneshot(
            fake_model,
            model_type="qwen3_5",
            task_type="generator",
        )

        self.assertEqual(fake_model.lm_head.to_calls, ["cpu"])
        self.assertEqual(fake_model.lm_head._hf_hook.execution_device, "cpu")

    def test_patch_llmcompressor_disable_lm_head_avoids_onloading_offloaded_weight(self):
        try:
            import torch
            from llmcompressor.utils import helpers as llm_helpers
        except ImportError:
            self.skipTest("llmcompressor and torch are required for this regression test.")

        class _FakeOffloadCache(dict):
            onloading_disabled = False

            def __getitem__(self, key):
                if not self.onloading_disabled:
                    raise RuntimeError("would onload lm_head.weight to GPU")
                return super().__getitem__(key)

        class _FakeLmHead(torch.nn.Linear):
            pass

        class _FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = _FakeLmHead(2, 3, bias=False)
                self.lm_head._parameters = _FakeOffloadCache(
                    {"weight": torch.nn.Parameter(torch.ones(3, 2))}
                )

            def get_input_embeddings(self):
                return None

            def get_output_embeddings(self):
                return self.lm_head

        @contextlib.contextmanager
        def fake_disable_onloading():
            old_value = _FakeOffloadCache.onloading_disabled
            _FakeOffloadCache.onloading_disabled = True
            try:
                yield
            finally:
                _FakeOffloadCache.onloading_disabled = old_value

        fake_model = _FakeModel()
        original_disable_lm_head = llm_helpers.disable_lm_head
        with patch("training.quantization.quantize_awq.OffloadCache", _FakeOffloadCache), patch(
            "training.quantization.quantize_awq.disable_onloading", fake_disable_onloading
        ):
            with patch_llmcompressor_disable_lm_head_for_offloaded_weights():
                self.assertIsNot(llm_helpers.disable_lm_head, original_disable_lm_head)
                with llm_helpers.disable_lm_head(fake_model):
                    output = fake_model.lm_head(torch.ones(1, 2))

        self.assertEqual(output.device.type, "meta")
        self.assertIs(llm_helpers.disable_lm_head, original_disable_lm_head)

    def test_pipeline_runs_merge_then_quantize(self):
        calls = []

        def fake_merge(**kwargs):
            calls.append(("merge", str(kwargs["output_dir"])))
            return {
                "output_dir": str(kwargs["output_dir"]),
                "manifest_path": "/tmp/merge_manifest.json",
                "artifacts": {"weight_files": ["model.safetensors"]},
            }

        def fake_quantize(**kwargs):
            calls.append(("quantize", str(kwargs["output_dir"])))
            self.assertEqual(kwargs["task_type"], "reranker")
            return {
                "output_dir": str(kwargs["output_dir"]),
                "manifest_path": "/tmp/awq_manifest.json",
                "artifacts": {"weight_files": ["model.safetensors"]},
                "quant_config": {"w_bit": 4},
                "calibration_sample_count": 8,
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "pipeline_manifest.json"
            with patch("training.quantization.pipeline.run_merge_stage", side_effect=fake_merge), patch(
                "training.quantization.pipeline.run_quantize_stage", side_effect=fake_quantize
            ):
                result = run_pipeline_stage(
                    base_model_path="/models/Qwen3.5-9B",
                    adapter_path="/adapters/qwen3.5-9b-lora",
                    merged_output_dir=Path(tmp_dir) / "merged",
                    quantized_output_dir=Path(tmp_dir) / "awq",
                    task_type="reranker",
                    manifest_path=manifest_path,
                )
                self.assertEqual(calls[0][0], "merge")
                self.assertEqual(calls[1][0], "quantize")
                self.assertTrue(result["quantized_output_dir"].endswith("/awq"))
                self.assertTrue(manifest_path.exists())

    def test_pipeline_rejects_invalid_task_type_before_merge(self):
        calls = []

        def fake_merge(**kwargs):
            del kwargs
            calls.append("merge")
            return {}

        def fake_quantize(**kwargs):
            del kwargs
            calls.append("quantize")
            return {}

        with patch("training.quantization.pipeline.run_merge_stage", side_effect=fake_merge), patch(
            "training.quantization.pipeline.run_quantize_stage", side_effect=fake_quantize
        ):
            with self.assertRaises(ValueError):
                run_pipeline_stage(
                    task_type="invalid",
                    base_model_path="/models/Qwen3.5-9B",
                    adapter_path="/adapters/qwen3.5-9b-lora",
                    merged_output_dir="/tmp/merged",
                    quantized_output_dir="/tmp/awq",
                )

        self.assertEqual(calls, [])

    def test_shell_wrappers_exist_for_generator_and_reranker_workflows(self):
        scripts_dir = self._REPO_ROOT / "training/quantization/scripts"
        expected_scripts = {
            "generator_merge_qwen3_5_9b.sh": "-m training.quantization.merge_lora",
            "generator_awq_qwen3_5_9b.sh": "-m training.quantization.quantize_awq",
            "reranker_merge_qwen3_0_6b.sh": "-m training.quantization.merge_lora",
            "reranker_awq_qwen3_0_6b.sh": "-m training.quantization.quantize_awq",
        }

        for script_name, expected_entrypoint in expected_scripts.items():
            script_path = scripts_dir / script_name
            self.assertTrue(script_path.exists(), msg=f"Missing wrapper script: {script_path}")
            content = script_path.read_text(encoding="utf-8")
            self.assertTrue(content.startswith("#!/usr/bin/env sh\n"))
            self.assertIn(expected_entrypoint, content)

        reranker_quantize_script = (scripts_dir / "reranker_awq_qwen3_0_6b.sh").read_text(encoding="utf-8")
        self.assertIn("--task-type reranker", reranker_quantize_script)
        self.assertIn('BACKEND="${BACKEND:-llmcompressor}"', reranker_quantize_script)


if __name__ == "__main__":
    unittest.main()
