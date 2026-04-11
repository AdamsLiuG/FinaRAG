import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.pipeline import Pipeline, RunConfig


class PipelineProcessReportsTests(unittest.TestCase):
    def _build_pipeline(self) -> tuple[TemporaryDirectory, Pipeline]:
        temp_dir = TemporaryDirectory()
        pipeline = Pipeline(
            Path(temp_dir.name),
            run_config=RunConfig(
                use_vector_dbs=False,
                use_bm25_db=False,
                use_sparse_lexical_db=False,
                use_tag_db=False,
            ),
        )
        return temp_dir, pipeline

    def test_process_parsed_reports_exports_markdown_by_default(self):
        temp_dir, pipeline = self._build_pipeline()
        self.addCleanup(temp_dir.cleanup)
        calls: list[str] = []

        pipeline.merge_reports = lambda: calls.append("merge")
        pipeline.export_reports_to_markdown = lambda: calls.append("markdown")
        pipeline.chunk_reports = lambda include_serialized_tables=False: calls.append("chunk")

        pipeline.process_parsed_reports()

        self.assertEqual(calls, ["merge", "markdown", "chunk"])

    def test_process_parsed_reports_can_skip_markdown_export(self):
        temp_dir, pipeline = self._build_pipeline()
        self.addCleanup(temp_dir.cleanup)
        calls: list[str] = []

        pipeline.merge_reports = lambda: calls.append("merge")
        pipeline.export_reports_to_markdown = lambda: calls.append("markdown")
        pipeline.chunk_reports = lambda include_serialized_tables=False: calls.append("chunk")

        pipeline.process_parsed_reports(export_markdown=False)

        self.assertEqual(calls, ["merge", "chunk"])


if __name__ == "__main__":
    unittest.main()
