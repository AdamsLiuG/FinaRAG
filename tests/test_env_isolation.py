import importlib
import os
import unittest
from unittest.mock import patch

from eval.ragas_adapter import DEFAULT_RAGAS_LLM_MODEL, RagasRuntimeConfig
from src.api_requests import BaseCompatibleProcessor


class EnvIsolationTests(unittest.TestCase):
    def test_answer_processor_ignores_ragas_env(self):
        with patch.dict(
            os.environ,
            {
                "RAGAS_LLM_BASE_URL": "https://judge.example/v1",
                "RAGAS_LLM_API_KEY": "judge-key",
                "RAGAS_LLM_MODEL": "judge-model",
                "RAGAS_LLM_WIRE_API": "responses",
                "QWEN_BASE_URL": "https://answer.example/v1",
                "QWEN_API_KEY": "answer-key",
                "QWEN_MODEL": "answer-model",
                "QWEN_WIRE_API": "chat_completions",
            },
            clear=True,
        ):
            processor = BaseCompatibleProcessor(provider="qwen")

        self.assertEqual(processor.base_url, "https://answer.example/v1")
        self.assertEqual(processor.api_key, "answer-key")
        self.assertEqual(processor.default_model, "answer-model")
        self.assertEqual(processor.wire_api, "chat_completions")

    def test_ragas_runtime_config_ignores_answer_side_qwen_env(self):
        with patch.dict(
            os.environ,
            {
                "RAGAS_LLM_MODEL": "judge-model",
                "RAGAS_LLM_BASE_URL": "",
                "RAGAS_LLM_API_KEY": "",
                "OPENAI_API_KEY": "",
                "QWEN_BASE_URL": "https://answer.example/v1",
                "QWEN_API_KEY": "answer-key",
                "QWEN_MODEL": "answer-model",
            },
            clear=True,
        ):
            config = RagasRuntimeConfig.from_env()

        self.assertEqual(config.llm_provider, "openai")
        self.assertEqual(config.llm_model, "judge-model")
        self.assertIsNone(config.llm_base_url)
        self.assertIsNone(config.llm_api_key)

    def test_pipeline_named_configs_and_runtime_overrides_ignore_ragas_model(self):
        import src.pipeline as pipeline_module

        try:
            with patch.dict(
                os.environ,
                {
                    "RAGAS_LLM_PROVIDER": "openai",
                    "RAGAS_LLM_MODEL": "judge-model",
                    "QWEN_MODEL": "online-answer-model",
                },
                clear=True,
            ):
                pipeline_module = importlib.reload(pipeline_module)
                config = pipeline_module.RunConfig(
                    api_provider="qwen",
                    answering_model="config-answer-model",
                )
                overridden = pipeline_module.apply_runtime_overrides(config)

                self.assertEqual(overridden.api_provider, "qwen")
                self.assertEqual(overridden.answering_model, "online-answer-model")
                self.assertEqual(
                    pipeline_module.configs["qwen_base"].answering_model,
                    "online-answer-model",
                )
        finally:
            importlib.reload(pipeline_module)


if __name__ == "__main__":
    unittest.main()
