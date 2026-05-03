import logging
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend

from src.pdf_parsing import PDFParser, _DoclingRecoverableWarningFilter


class PdfParsingFallbackTests(unittest.TestCase):
    def _make_parser(self) -> PDFParser:
        parser = object.__new__(PDFParser)
        parser.pdf_backend = DoclingParseV2DocumentBackend
        parser.output_dir = Path(".")
        parser.num_threads = None
        parser.document_language = "zh"
        parser.ocr_mode = "docling_rapidocr"
        parser.metadata_lookup = {}
        parser.debug_data_path = None
        return parser

    def test_parse_documents_retries_invalid_code_point_with_legacy_backend(self):
        parser = self._make_parser()
        parser._parse_single_document = Mock(side_effect=[RuntimeError("Invalid code point")])
        fallback_parser = Mock()
        fallback_parser._parse_single_document.return_value = 1
        parser._build_fallback_parser = Mock(return_value=fallback_parser)

        success_count, failure_count = PDFParser.parse_documents(parser, [Path("bad.pdf")])

        self.assertEqual(success_count, 1)
        self.assertEqual(failure_count, 0)
        parser._build_fallback_parser.assert_called_once_with(DoclingParseDocumentBackend)
        fallback_parser._parse_single_document.assert_called_once_with(Path("bad.pdf"))

    def test_parse_documents_reraises_non_fallback_errors(self):
        parser = self._make_parser()
        parser._parse_single_document = Mock(side_effect=RuntimeError("Some other parser error"))
        parser._build_fallback_parser = Mock()

        with self.assertRaisesRegex(RuntimeError, "Some other parser error"):
            PDFParser.parse_documents(parser, [Path("bad.pdf")])

        parser._build_fallback_parser.assert_not_called()

    def test_parse_and_export_does_not_raise_after_successful_parse(self):
        parser = self._make_parser()
        parser.parse_documents = Mock(return_value=(1, 0))

        PDFParser.parse_and_export(parser, [Path("bad.pdf")])

        parser.parse_documents.assert_called_once_with([Path("bad.pdf")])

    def test_docling_warning_filter_suppresses_recoverable_invalid_code_point_traceback(self):
        warning_filter = _DoclingRecoverableWarningFilter()

        try:
            raise RuntimeError("Invalid code point")
        except RuntimeError:
            record = logging.LogRecord(
                name="docling.pipeline.base_pipeline",
                level=logging.WARNING,
                pathname=__file__,
                lineno=1,
                msg="Encountered an error during conversion of document %s:",
                args=("bad.pdf",),
                exc_info=sys.exc_info(),
            )

        self.assertFalse(warning_filter.filter(record))


if __name__ == "__main__":
    unittest.main()
