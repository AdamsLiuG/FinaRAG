import unittest
from unittest.mock import patch

from src.reranking import (
    FastTokenizerFlagRerankerModel,
    _load_flag_reranker_model,
    _load_flag_reranker_model_cached,
)


class _DummySequenceClassificationModel:
    def __init__(self):
        self.half_called = False
        self.to_calls = []
        self.eval_called = False

    def half(self):
        self.half_called = True
        return self

    def to(self, device):
        self.to_calls.append(device)
        return self

    def eval(self):
        self.eval_called = True
        return self


class RerankingModelLoadingTests(unittest.TestCase):
    def setUp(self):
        _load_flag_reranker_model_cached.cache_clear()

    def tearDown(self):
        _load_flag_reranker_model_cached.cache_clear()

    def test_fast_flag_reranker_disables_low_cpu_mem_usage_before_moving_to_device(self):
        dummy_model = _DummySequenceClassificationModel()

        with patch("src.reranking.AutoTokenizer.from_pretrained", return_value=object()) as mock_tokenizer:
            with patch(
                "src.reranking.AutoModelForSequenceClassification.from_pretrained",
                return_value=dummy_model,
            ) as mock_model:
                model = FastTokenizerFlagRerankerModel(
                    model_name="dummy-reranker",
                    device="cuda:0",
                    use_fp16=True,
                    trust_remote_code=True,
                )

        self.assertIs(model.model, dummy_model)
        self.assertEqual(mock_tokenizer.call_count, 1)
        self.assertTrue(mock_model.call_args.kwargs["low_cpu_mem_usage"] is False)
        self.assertTrue(mock_model.call_args.kwargs["trust_remote_code"])
        self.assertTrue(dummy_model.half_called)
        self.assertEqual(dummy_model.to_calls, ["cuda:0"])
        self.assertTrue(dummy_model.eval_called)

    def test_flag_reranker_loader_caches_per_device(self):
        first_model = object()
        second_model = object()

        with patch(
            "src.reranking.FastTokenizerFlagRerankerModel",
            side_effect=[first_model, second_model],
        ) as mock_ctor:
            cached_first = _load_flag_reranker_model(
                model_name="dummy-reranker",
                device="cuda:0",
                use_fp16=True,
                trust_remote_code=False,
            )
            cached_second = _load_flag_reranker_model(
                model_name="dummy-reranker",
                device="cuda:0",
                use_fp16=True,
                trust_remote_code=False,
            )
            cached_third = _load_flag_reranker_model(
                model_name="dummy-reranker",
                device="cuda:1",
                use_fp16=True,
                trust_remote_code=False,
            )

        self.assertIs(cached_first, cached_second)
        self.assertIs(cached_third, second_model)
        self.assertEqual(mock_ctor.call_count, 2)


if __name__ == "__main__":
    unittest.main()
