import unittest

from src.hyde import should_trigger_hyde


def _make_result(score: float, query_hit_count: int = 1):
    return {
        "combined_score": score,
        "query_hit_count": query_hit_count,
        "matched_queries": ["orig"] * max(1, query_hit_count),
    }


class ShouldTriggerHyDETests(unittest.TestCase):
    def test_no_results_triggers_hyde(self):
        triggered, reasons = should_trigger_hyde([], top_score_threshold=0.55, margin_threshold=0.05)

        self.assertTrue(triggered)
        self.assertEqual(reasons, ["no_results"])

    def test_low_top_score_triggers_hyde(self):
        triggered, reasons = should_trigger_hyde(
            [_make_result(0.42), _make_result(0.31)],
            top_score_threshold=0.55,
            margin_threshold=0.05,
        )

        self.assertTrue(triggered)
        self.assertIn("low_top_score", reasons)

    def test_weak_margin_without_consensus_triggers_hyde(self):
        triggered, reasons = should_trigger_hyde(
            [_make_result(0.72, query_hit_count=1), _make_result(0.69, query_hit_count=1)],
            top_score_threshold=0.55,
            margin_threshold=0.05,
        )

        self.assertTrue(triggered)
        self.assertIn("weak_margin_no_consensus", reasons)

    def test_strong_top_score_and_clear_margin_do_not_trigger_hyde(self):
        triggered, reasons = should_trigger_hyde(
            [_make_result(0.81, query_hit_count=2), _make_result(0.63, query_hit_count=1)],
            top_score_threshold=0.55,
            margin_threshold=0.05,
        )

        self.assertFalse(triggered)
        self.assertEqual(reasons, [])


if __name__ == "__main__":
    unittest.main()
