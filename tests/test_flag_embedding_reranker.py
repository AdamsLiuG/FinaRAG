import unittest
from unittest.mock import patch

from src.reranking import FlagEmbeddingReranker


class _FailOnceModel:
    def __init__(self, *, should_fail: bool):
        self.should_fail = should_fail

    def compute_score(self, pairs, normalize=True):
        if self.should_fail:
            raise RuntimeError(
                "CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling cublasGemmEx(...)"
            )
        return [0.9 for _ in pairs]


class FlagEmbeddingRerankerTests(unittest.TestCase):
    def test_retries_in_fp32_after_fp16_cublas_failure(self):
        loaded_flags = []

        def fake_loader(*, model_name, device, use_fp16, trust_remote_code, batch_size=128, max_length=512):
            loaded_flags.append(use_fp16)
            return _FailOnceModel(should_fail=use_fp16)

        with patch("src.reranking._load_flag_reranker_model", side_effect=fake_loader):
            with patch("src.reranking._get_env_value") as mock_env:
                values = {
                    ("RERANKING_MODEL",): "BAAI/bge-reranker-v2-m3",
                    ("RERANKING_DEVICE",): "cuda:0",
                    ("RERANKING_USE_FP16",): "true",
                    ("RERANKING_TRUST_REMOTE_CODE",): "false",
                }

                def env_side_effect(*names, default=None):
                    return values.get(tuple(names), default)

                mock_env.side_effect = env_side_effect
                reranker = FlagEmbeddingReranker()

            results = reranker.rerank_documents(
                "法定代表人是谁？",
                [{"text": "法定代表人是张三。", "distance": 0.8}],
            )

        self.assertEqual(loaded_flags, [True, False])
        self.assertEqual(results[0]["relevance_score"], 0.9)
        self.assertTrue(reranker._fp16_runtime_disabled)


if __name__ == "__main__":
    unittest.main()
