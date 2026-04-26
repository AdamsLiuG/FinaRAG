import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.generator_sft.scripts.build_mix_dataset import main as build_mix_main


class BuildMixDatasetTests(unittest.TestCase):
    def _write_jsonl(self, path: Path, records):
        path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )

    def _chat_record(
        self,
        *,
        sample_id: str,
        query_id: str,
        template_family: str,
        schema: str = "text",
        should_refuse: bool = False,
        variant_type: str | None = None,
        source: str = "template_from_annual_report_v2",
    ):
        meta = {
            "sample_id": sample_id,
            "query_id": query_id,
            "template_family": template_family,
            "schema": schema,
            "should_refuse": should_refuse,
            "source": source,
            "doc_ids": [f"{sample_id}_doc"],
        }
        if variant_type is not None:
            meta["variant_type"] = variant_type
        return {
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"},
            ],
            "meta": meta,
        }

    def test_build_mix_dataset_merges_outputs_and_dedupes_globally(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_a = tmp_path / "source_a.jsonl"
            source_b = tmp_path / "source_b.jsonl"
            config_path = tmp_path / "mix.yaml"
            new_text_path = tmp_path / "new_text.jsonl"
            old_regularizer_path = tmp_path / "old_regularizer.jsonl"
            all_output_path = tmp_path / "all.jsonl"
            stats_output_path = tmp_path / "stats.json"

            self._write_jsonl(
                source_a,
                [
                    self._chat_record(sample_id="s1", query_id="q1", template_family="gen120_summary"),
                    self._chat_record(sample_id="s2", query_id="q2", template_family="gen120_summary"),
                ],
            )
            self._write_jsonl(
                source_b,
                [
                    self._chat_record(sample_id="s1", query_id="q1", template_family="gen120_summary"),
                    self._chat_record(sample_id="s3", query_id="q3", template_family="gen120_summary"),
                    self._chat_record(sample_id="s4", query_id="q4", template_family="number_metric", schema="number"),
                    self._chat_record(sample_id="s5", query_id="q5", template_family="number_metric", schema="number"),
                ],
            )

            config_path.write_text(
                "\n".join(
                    [
                        "mix_tag: unit_mix",
                        "random_seed: unit_seed",
                        "allow_shortfall: false",
                        "dedupe_fields:",
                        "  - sample_id",
                        "  - query_id",
                        "  - variant_type",
                        "bucket_outputs:",
                        f"  new_text: {new_text_path}",
                        f"  old_regularizer: {old_regularizer_path}",
                        f"all_output_path: {all_output_path}",
                        f"stats_output_path: {stats_output_path}",
                        "groups:",
                        "  - name: summary_primary",
                        "    bucket: new_text",
                        "    mix_group: new_positive",
                        f"    input_path: {source_a}",
                        "    count: 1",
                        "    match:",
                        "      template_family: gen120_summary",
                        "      should_refuse: false",
                        "  - name: summary_secondary",
                        "    bucket: new_text",
                        "    mix_group: new_positive",
                        f"    input_path: {source_b}",
                        "    count: 1",
                        "    match:",
                        "      template_family: gen120_summary",
                        "      should_refuse: false",
                        "  - name: number_regularizer",
                        "    bucket: old_regularizer",
                        "    mix_group: old_regularizer",
                        f"    input_path: {source_b}",
                        "    count: 2",
                        "    match:",
                        "      template_family: number_metric",
                        "      should_refuse: false",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("sys.argv", ["build_mix_dataset.py", "--config-path", str(config_path)]):
                build_mix_main()

            new_text_records = [
                json.loads(line)
                for line in new_text_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            old_regularizer_records = [
                json.loads(line)
                for line in old_regularizer_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            all_records = [
                json.loads(line)
                for line in all_output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            stats = json.loads(stats_output_path.read_text(encoding="utf-8"))

            self.assertEqual(len(new_text_records), 2)
            self.assertEqual(len(old_regularizer_records), 2)
            self.assertEqual(len(all_records), 4)

            new_text_query_ids = {record["meta"]["query_id"] for record in new_text_records}
            self.assertEqual(len(new_text_query_ids), 2)
            self.assertIn("q3", new_text_query_ids)
            self.assertTrue(new_text_query_ids & {"q1", "q2"})

            first_meta = new_text_records[0]["meta"]
            self.assertEqual(first_meta["mix_tag"], "unit_mix")
            self.assertEqual(first_meta["mix_bucket"], "new_text")
            self.assertEqual(first_meta["mix_group"], "new_positive")
            self.assertIn(first_meta["mix_subgroup"], {"summary_primary", "summary_secondary"})
            self.assertIsInstance(first_meta["mix_group_index"], int)

            self.assertEqual(stats["mix_tag"], "unit_mix")
            self.assertEqual(stats["bucket_counts"]["new_text"], 2)
            self.assertEqual(stats["bucket_counts"]["old_regularizer"], 2)
            self.assertEqual(stats["selection_stats"]["summary_primary"]["selected_count"], 1)
            self.assertEqual(stats["selection_stats"]["summary_secondary"]["selected_count"], 1)
            self.assertEqual(stats["selection_stats"]["number_regularizer"]["selected_count"], 2)


if __name__ == "__main__":
    unittest.main()
