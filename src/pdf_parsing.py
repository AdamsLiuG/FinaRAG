import os
import time
import logging
import re
import json
from tabulate import tabulate
from pathlib import Path
from typing import Iterable, List, Optional

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
# from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ConversionResult

from src.document_manifest import load_document_manifest

_log = logging.getLogger(__name__)


def _coerce_csv_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        lowered = stripped.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if stripped == "":
            return None
        return stripped
    return value


def _is_invalid_code_point_error(error: Exception | None) -> bool:
    return error is not None and "invalid code point" in str(error).lower()


class _DoclingRecoverableWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "docling.pipeline.base_pipeline":
            return True
        if "encountered an error during conversion of document" not in record.getMessage().lower():
            return True
        if not record.exc_info:
            return True
        return not _is_invalid_code_point_error(record.exc_info[1])


def _install_docling_warning_filter():
    logger = logging.getLogger("docling.pipeline.base_pipeline")
    if any(isinstance(existing_filter, _DoclingRecoverableWarningFilter) for existing_filter in logger.filters):
        return
    # We emit our own concise fallback warning for this known recoverable docling v2 failure.
    logger.addFilter(_DoclingRecoverableWarningFilter())


_install_docling_warning_filter()


def _parse_cuda_devices(value: Optional[str]) -> List[str]:
    if value is None:
        return []

    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        return []

    normalized_devices: List[str] = []
    for part in parts:
        if part.isdigit():
            normalized_devices.append(part)
        elif part.startswith("cuda:"):
            normalized_devices.append(part.split(":", 1)[1])
        else:
            normalized_devices.append(part)
    return normalized_devices


def _assign_chunks_to_devices(chunks: List[List[Path]], cuda_devices: List[str]) -> List[tuple[List[Path], str]]:
    if not cuda_devices:
        return [(chunk, "") for chunk in chunks]
    return [
        (chunk, cuda_devices[index % len(cuda_devices)])
        for index, chunk in enumerate(chunks)
    ]


def _worker_initializer(cuda_visible_devices: str):
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices


def _process_chunk(pdf_paths, pdf_backend, output_dir, num_threads, metadata_lookup, debug_data_path, document_language, ocr_mode):
    """Helper function to process a chunk of PDFs in a separate process."""
    # Create a new parser instance for this process
    parser = PDFParser(
        pdf_backend=pdf_backend,
        output_dir=output_dir,
        num_threads=num_threads,
        csv_metadata_path=None,  # Metadata lookup is passed directly
        document_language=document_language,
        ocr_mode=ocr_mode,
    )
    parser.metadata_lookup = metadata_lookup
    parser.debug_data_path = debug_data_path
    parser.parse_and_export(pdf_paths)
    return f"Processed {len(pdf_paths)} PDFs."

class PDFParser:
    def __init__(
        self,
        pdf_backend=DoclingParseV2DocumentBackend,
        output_dir: Path = Path("./parsed_pdfs"),
        num_threads: int = None,
        csv_metadata_path: Path = None,
        document_language: str = "en",
        ocr_mode: str = "docling_rapidocr",
    ):
        self.pdf_backend = pdf_backend
        self.document_language = (document_language or "en").strip().lower()
        self.ocr_mode = self._normalize_ocr_mode(ocr_mode)
        self.output_dir = output_dir
        self.doc_converter = self._create_document_converter()
        self.num_threads = num_threads
        self.metadata_lookup = {}
        self.debug_data_path = None

        if csv_metadata_path is not None:
            self.metadata_lookup = self._parse_csv_metadata(csv_metadata_path)
            
        if self.num_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(self.num_threads)

    def _build_fallback_parser(self, pdf_backend) -> "PDFParser":
        parser = PDFParser(
            pdf_backend=pdf_backend,
            output_dir=self.output_dir,
            num_threads=self.num_threads,
            csv_metadata_path=None,
            document_language=self.document_language,
            ocr_mode=self.ocr_mode,
        )
        parser.metadata_lookup = self.metadata_lookup
        parser.debug_data_path = self.debug_data_path
        return parser

    def _should_retry_with_legacy_backend(self, error: Exception) -> bool:
        return self.pdf_backend is DoclingParseV2DocumentBackend and _is_invalid_code_point_error(error)

    def _parse_single_document(self, pdf_path: Path):
        conv_results = self.convert_documents([pdf_path])
        success_count, failure_count = self.process_documents(conv_results=conv_results)
        if failure_count > 0:
            raise RuntimeError(f"Failed converting 1 out of 1 documents: {pdf_path}")
        return success_count

    def parse_documents(self, input_doc_paths: List[Path]) -> tuple[int, int]:
        total_docs = len(input_doc_paths)
        _log.info(f"Starting to process {total_docs} documents")
        start_time = time.time()
        success_count = 0

        for pdf_path in input_doc_paths:
            try:
                success_count += self._parse_single_document(pdf_path)
            except Exception as err:
                if not self._should_retry_with_legacy_backend(err):
                    raise
                _log.warning(
                    "Docling v2 failed for %s with '%s'. Retrying with legacy docling_parse v1 backend.",
                    pdf_path,
                    err,
                )
                fallback_parser = self._build_fallback_parser(DoclingParseDocumentBackend)
                success_count += fallback_parser._parse_single_document(pdf_path)

        elapsed_time = time.time() - start_time
        _log.info(
            f"{'#'*50}\nCompleted in {elapsed_time:.2f} seconds. Successfully converted {success_count}/{total_docs} documents.\n{'#'*50}"
        )
        return success_count, total_docs - success_count

    @staticmethod
    def _parse_csv_metadata(csv_path: Path) -> dict:
        """Parse a manifest-like file and create a lookup dictionary with doc_id/sha1 as key."""
        return load_document_manifest(csv_path)

    def _build_ocr_languages(self) -> List[str]:
        if self.ocr_mode == "docling_rapidocr":
            if self.document_language in {"zh", "bilingual"}:
                return ["english", "chinese"]
            return ["english"]
        if self.document_language == "zh":
            return ["ch_sim", "en"]
        if self.document_language == "bilingual":
            return ["en", "ch_sim"]
        return ["en"]

    @staticmethod
    def _normalize_ocr_mode(ocr_mode: str | None) -> str:
        normalized = (ocr_mode or "docling_rapidocr").strip().lower()
        alias_map = {
            "docling_only": "docling_rapidocr",
            "rapidocr": "docling_rapidocr",
            "docling_rapidocr": "docling_rapidocr",
            "easyocr": "docling_easyocr",
            "docling_easyocr": "docling_easyocr",
            # Backward-compatible alias for older configs in this repo.
            "docling_paddle_fallback": "docling_rapidocr",
        }
        resolved = alias_map.get(normalized)
        if resolved is None:
            _log.warning(
                "Unknown OCR mode '%s'; falling back to Docling RapidOCR.",
                normalized,
            )
            return "docling_rapidocr"
        if normalized == "docling_paddle_fallback":
            _log.warning(
                "OCR mode 'docling_paddle_fallback' is deprecated; using Docling RapidOCR instead."
            )
        return resolved

    def _create_document_converter(self) -> "DocumentConverter": # type: ignore
        """Creates and returns a DocumentConverter with default pipeline options."""
        from docling.document_converter import DocumentConverter, FormatOption
        from docling.datamodel.pipeline_options import (
            EasyOcrOptions,
            PdfPipelineOptions,
            RapidOcrOptions,
            TableFormerMode,
        )
        from docling.datamodel.base_models import InputFormat
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
        
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        if self.ocr_mode == "docling_easyocr":
            ocr_options = EasyOcrOptions(
                lang=self._build_ocr_languages(),
                force_full_page_ocr=False,
            )
        else:
            ocr_options = RapidOcrOptions(
                lang=self._build_ocr_languages(),
                force_full_page_ocr=False,
            )
        pipeline_options.ocr_options = ocr_options
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        
        format_options = {
            InputFormat.PDF: FormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=pipeline_options,
                backend=self.pdf_backend
            )
        }
        
        return DocumentConverter(format_options=format_options)

    def convert_documents(self, input_doc_paths: List[Path]) -> Iterable[ConversionResult]:
        conv_results = self.doc_converter.convert_all(source=input_doc_paths)
        return conv_results
    
    def process_documents(self, conv_results: Iterable[ConversionResult]):
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        success_count = 0
        failure_count = 0

        for conv_res in conv_results:
            if conv_res.status == ConversionStatus.SUCCESS:
                success_count += 1
                processor = JsonReportProcessor(
                    metadata_lookup=self.metadata_lookup,
                    debug_data_path=self.debug_data_path,
                    default_language=self.document_language,
                    ocr_mode=self.ocr_mode,
                )
                
                # Normalize the document data to ensure sequential pages
                data = conv_res.document.export_to_dict()
                normalized_data = self._normalize_page_sequence(data)
                
                processed_report = processor.assemble_report(conv_res, normalized_data)
                doc_filename = conv_res.input.file.stem
                if self.output_dir is not None:
                    with (self.output_dir / f"{doc_filename}.json").open("w", encoding="utf-8") as fp:
                        json.dump(processed_report, fp, indent=2, ensure_ascii=False)
            else:
                failure_count += 1
                _log.info(f"Document {conv_res.input.file} failed to convert.")

        _log.info(f"Processed {success_count + failure_count} docs, of which {failure_count} failed")
        return success_count, failure_count

    def _normalize_page_sequence(self, data: dict) -> dict:
        """Ensure that page numbers in content are sequential by filling gaps with empty pages."""
        if 'content' not in data:
            return data
        
        # Create a copy of the data to modify
        normalized_data = data.copy()
        
        # Get existing page numbers and find max page
        existing_pages = {page['page'] for page in data['content']}
        max_page = max(existing_pages)
        
        # Create template for empty page
        empty_page_template = {
            "content": [],
            "page_dimensions": {}  # or some default dimensions if needed
        }
        
        # Create new content array with all pages
        new_content = []
        for page_num in range(1, max_page + 1):
            # Find existing page or create empty one
            page_content = next(
                (page for page in data['content'] if page['page'] == page_num),
                {"page": page_num, **empty_page_template}
            )
            new_content.append(page_content)
        
        normalized_data['content'] = new_content
        return normalized_data

    def parse_and_export(self, input_doc_paths: List[Path] = None, doc_dir: Path = None):
        if input_doc_paths is None and doc_dir is not None:
            input_doc_paths = list(doc_dir.glob("*.pdf"))

        total_docs = len(input_doc_paths)
        _, failure_count = self.parse_documents(input_doc_paths)

        if failure_count > 0:
            error_message = f"Failed converting {failure_count} out of {total_docs} documents."
            failed_docs = "Paths of failed docs:\n" + '\n'.join(str(path) for path in input_doc_paths)
            _log.error(error_message)
            _log.error(failed_docs)
            raise RuntimeError(error_message)

    def parse_and_export_parallel(
        self,
        input_doc_paths: List[Path] = None,
        doc_dir: Path = None,
        optimal_workers: int = 10,
        chunk_size: int = None,
        cuda_devices: Optional[str] = None,
    ):
        """Parse PDF files in parallel using multiple processes.
        
        Args:
            input_doc_paths: List of paths to PDF files to process
            doc_dir: Directory containing PDF files (used if input_doc_paths is None)
            optimal_workers: Number of worker processes to use. If None, uses CPU count.
        """
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Get input paths if not provided
        if input_doc_paths is None and doc_dir is not None:
            input_doc_paths = list(doc_dir.glob("*.pdf"))

        total_pdfs = len(input_doc_paths)
        _log.info(f"Starting parallel processing of {total_pdfs} documents")
        
        cpu_count = multiprocessing.cpu_count()
        
        # Calculate optimal workers if not specified
        if optimal_workers is None:
            optimal_workers = min(cpu_count, total_pdfs)
        
        if chunk_size is None:
            # Calculate chunk size (ensure at least 1)
            chunk_size = max(1, total_pdfs // optimal_workers)
        
        # Split documents into chunks
        chunks = [
            input_doc_paths[i : i + chunk_size]
            for i in range(0, total_pdfs, chunk_size)
        ]
        parsed_cuda_devices = _parse_cuda_devices(cuda_devices)

        start_time = time.time()
        processed_count = 0

        if not parsed_cuda_devices:
            # Use ProcessPoolExecutor for parallel processing
            with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
                futures = [
                    executor.submit(
                        _process_chunk,
                        chunk,
                        self.pdf_backend,
                        self.output_dir,
                        self.num_threads,
                        self.metadata_lookup,
                        self.debug_data_path,
                        self.document_language,
                        self.ocr_mode,
                    )
                    for chunk in chunks
                ]

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        processed_count += int(result.split()[1])
                        _log.info(f"{'#'*50}\n{result} ({processed_count}/{total_pdfs} total)\n{'#'*50}")
                    except Exception as e:
                        _log.error(f"Error processing chunk: {str(e)}")
                        raise
        else:
            spawn_context = multiprocessing.get_context("spawn")
            active_cuda_devices = parsed_cuda_devices[: max(1, min(optimal_workers, len(parsed_cuda_devices)))]
            tasks_by_device: dict[str, List[List[Path]]] = {device: [] for device in active_cuda_devices}
            for chunk, device in _assign_chunks_to_devices(chunks, active_cuda_devices):
                tasks_by_device[device].append(chunk)

            total_requested_workers = max(1, optimal_workers)
            max_workers_by_device: dict[str, int] = {}
            remaining_workers = total_requested_workers
            for index, device in enumerate(active_cuda_devices):
                remaining_devices = len(active_cuda_devices) - index
                device_task_count = len(tasks_by_device.get(device, []))
                if device_task_count <= 0:
                    max_workers_by_device[device] = 0
                    continue
                allocated_workers = max(1, remaining_workers // max(1, remaining_devices))
                max_workers_by_device[device] = min(device_task_count, allocated_workers)
                remaining_workers = max(0, remaining_workers - max_workers_by_device[device])

            futures = []
            executors: List[ProcessPoolExecutor] = []
            try:
                for device in active_cuda_devices:
                    device_tasks = tasks_by_device.get(device, [])
                    device_workers = max_workers_by_device.get(device, 0)
                    if not device_tasks or device_workers <= 0:
                        continue
                    executor = ProcessPoolExecutor(
                        max_workers=device_workers,
                        mp_context=spawn_context,
                        initializer=_worker_initializer,
                        initargs=(device,),
                    )
                    executors.append(executor)
                    _log.info(
                        "Starting %s parse workers on visible CUDA device(s) '%s' for %s chunk(s).",
                        device_workers,
                        device,
                        len(device_tasks),
                    )
                    futures.extend(
                        executor.submit(
                            _process_chunk,
                            chunk,
                            self.pdf_backend,
                            self.output_dir,
                            self.num_threads,
                            self.metadata_lookup,
                            self.debug_data_path,
                            self.document_language,
                            self.ocr_mode,
                        )
                        for chunk in device_tasks
                    )

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        processed_count += int(result.split()[1])
                        _log.info(f"{'#'*50}\n{result} ({processed_count}/{total_pdfs} total)\n{'#'*50}")
                    except Exception as e:
                        _log.error(f"Error processing chunk: {str(e)}")
                        raise
            finally:
                for executor in executors:
                    executor.shutdown(wait=True)

        elapsed_time = time.time() - start_time
        _log.info(f"Parallel processing completed in {elapsed_time:.2f} seconds.")


class JsonReportProcessor:
    def __init__(
        self,
        metadata_lookup: dict = None,
        debug_data_path: Path = None,
        default_language: str = "en",
        ocr_mode: str = "docling_rapidocr",
    ):
        self.metadata_lookup = metadata_lookup or {}
        self.debug_data_path = debug_data_path
        self.default_language = default_language
        self.ocr_mode = ocr_mode

    def assemble_report(self, conv_result, normalized_data=None):
        """Assemble the report using either normalized data or raw conversion result."""
        data = normalized_data if normalized_data is not None else conv_result.document.export_to_dict()
        assembled_report = {}
        assembled_report['metainfo'] = self.assemble_metainfo(data)
        assembled_report['content'] = self.assemble_content(data)
        assembled_report['tables'] = self.assemble_tables(conv_result.document.tables, data, conv_result.document)
        assembled_report['pictures'] = self.assemble_pictures(data)
        self.debug_data(data)
        return assembled_report
    
    def assemble_metainfo(self, data):
        metainfo = {}
        sha1_name = data['origin']['filename'].rsplit('.', 1)[0]
        metainfo['sha1_name'] = sha1_name
        metainfo['doc_id'] = sha1_name
        metainfo['pages_amount'] = len(data.get('pages', []))
        metainfo['text_blocks_amount'] = len(data.get('texts', []))
        metainfo['tables_amount'] = len(data.get('tables', []))
        metainfo['pictures_amount'] = len(data.get('pictures', []))
        metainfo['equations_amount'] = len(data.get('equations', []))
        metainfo['footnotes_amount'] = len([t for t in data.get('texts', []) if t.get('label') == 'footnote'])
        metainfo['language'] = self.default_language
        
        # Add CSV metadata if available
        if self.metadata_lookup and sha1_name in self.metadata_lookup:
            csv_meta = self.metadata_lookup[sha1_name]
            metainfo.update(csv_meta)
            metainfo['company_name'] = csv_meta['company_name']
            metainfo['currency'] = csv_meta.get('currency')
            metainfo['company_aliases'] = csv_meta.get('company_aliases', [csv_meta['company_name']])
            metainfo['security_code'] = csv_meta.get('security_code')
            metainfo['doc_source_type'] = csv_meta.get('doc_source_type')
            metainfo['report_date'] = csv_meta.get('report_date')
            metainfo['fiscal_year'] = csv_meta.get('fiscal_year')
            metainfo['report_title'] = csv_meta.get('report_title')
            metainfo['broker_name'] = csv_meta.get('broker_name')
            metainfo['language'] = csv_meta.get('language', metainfo['language'])
        else:
            metainfo['company_aliases'] = [sha1_name]
        
        metainfo['ocr_mode'] = self.ocr_mode
        metainfo['is_low_text_density'] = metainfo['text_blocks_amount'] <= max(1, metainfo['pages_amount'] // 2)
            
        return metainfo

    def process_table(self, table_data):
        # Implement your table processing logic here
        return 'processed_table_content'

    def debug_data(self, data):
        if self.debug_data_path is None:
            return
        doc_name = data['name']
        path = self.debug_data_path / f"{doc_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)    
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def expand_groups(self, body_children, groups):
        expanded_children = []

        for item in body_children:
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)

                if ref_type == 'groups':
                    group = groups[ref_num]
                    group_id = ref_num
                    group_name = group.get('name', '')
                    group_label = group.get('label', '')

                    for child in group['children']:
                        child_copy = child.copy()
                        child_copy['group_id'] = group_id
                        child_copy['group_name'] = group_name
                        child_copy['group_label'] = group_label
                        expanded_children.append(child_copy)
                else:
                    expanded_children.append(item)
            else:
                expanded_children.append(item)

        return expanded_children
    
    def _process_text_reference(self, ref_num, data):
        """Helper method to process text references and create content items.
        
        Args:
            ref_num (int): Reference number for the text item
            data (dict): Document data dictionary
            
        Returns:
            dict: Processed content item with text information
        """
        text_item = data['texts'][ref_num]
        item_type = text_item['label']
        content_item = {
            'text': text_item.get('text', ''),
            'type': item_type,
            'text_id': ref_num
        }
        
        # Add 'orig' field only if it differs from 'text'
        orig_content = text_item.get('orig', '')
        if orig_content != text_item.get('text', ''):
            content_item['orig'] = orig_content

        # Add additional fields if they exist
        if 'enumerated' in text_item:
            content_item['enumerated'] = text_item['enumerated']
        if 'marker' in text_item:
            content_item['marker'] = text_item['marker']
            
        return content_item
    
    def assemble_content(self, data):
        pages = {}
        # Expand body children to include group references
        body_children = data['body']['children']
        groups = data.get('groups', [])
        expanded_body_children = self.expand_groups(body_children, groups)

        # Process body content
        for item in expanded_body_children:
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)

                if ref_type == 'texts':
                    text_item = data['texts'][ref_num]
                    content_item = self._process_text_reference(ref_num, data)

                    # Add group information if available
                    if 'group_id' in item:
                        content_item['group_id'] = item['group_id']
                        content_item['group_name'] = item['group_name']
                        content_item['group_label'] = item['group_label']

                    # Get page number from prov
                    if 'prov' in text_item and text_item['prov']:
                        page_num = text_item['prov'][0]['page_no']

                        # Initialize page if not exists
                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': text_item['prov'][0].get('bbox', {})
                            }

                        pages[page_num]['content'].append(content_item)

                elif ref_type == 'tables':
                    table_item = data['tables'][ref_num]
                    content_item = {
                        'type': 'table',
                        'table_id': ref_num
                    }

                    if 'prov' in table_item and table_item['prov']:
                        page_num = table_item['prov'][0]['page_no']

                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': table_item['prov'][0].get('bbox', {})
                            }

                        pages[page_num]['content'].append(content_item)
                
                elif ref_type == 'pictures':
                    picture_item = data['pictures'][ref_num]
                    content_item = {
                        'type': 'picture',
                        'picture_id': ref_num
                    }
                    
                    if 'prov' in picture_item and picture_item['prov']:
                        page_num = picture_item['prov'][0]['page_no']

                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': picture_item['prov'][0].get('bbox', {})
                            }
                        
                        pages[page_num]['content'].append(content_item)

        sorted_pages = [pages[page_num] for page_num in sorted(pages.keys())]
        return sorted_pages

    def assemble_tables(self, tables, data, doc):
        assembled_tables = []
        for i, table in enumerate(tables):
            table_json_obj = table.model_dump()
            table_md = self._table_to_md(table_json_obj)
            table_html = table.export_to_html(doc=doc)
            
            table_data = data['tables'][i]
            table_page_num = table_data['prov'][0]['page_no']
            table_bbox = table_data['prov'][0]['bbox']
            table_bbox = [
                table_bbox['l'],
                table_bbox['t'], 
                table_bbox['r'],
                table_bbox['b']
            ]
            
            # Get rows and columns from the table data structure
            nrows = table_data['data']['num_rows']
            ncols = table_data['data']['num_cols']

            ref_num = table_data['self_ref'].split('/')[-1]
            ref_num = int(ref_num)

            table_obj = {
                'table_id': ref_num,
                'page': table_page_num,
                'bbox': table_bbox,
                '#-rows': nrows,
                '#-cols': ncols,
                'markdown': table_md,
                'html': table_html,
                'json': table_json_obj
            }
            assembled_tables.append(table_obj)
        return assembled_tables

    def _table_to_md(self, table):
        # Extract text from grid cells
        table_data = []
        for row in table['data']['grid']:
            table_row = [cell['text'] for cell in row]
            table_data.append(table_row)
        
        # Check if the table has headers
        if len(table_data) > 1 and len(table_data[0]) > 0:
            try:
                md_table = tabulate(
                    table_data[1:], headers=table_data[0], tablefmt="github"
                )
            except ValueError:
                md_table = tabulate(
                    table_data[1:],
                    headers=table_data[0],
                    tablefmt="github",
                    disable_numparse=True,
                )
        else:
            md_table = tabulate(table_data, tablefmt="github")
        
        return md_table

    def assemble_pictures(self, data):
        assembled_pictures = []
        for i, picture in enumerate(data['pictures']):
            children_list = self._process_picture_block(picture, data)
            
            ref_num = picture['self_ref'].split('/')[-1]
            ref_num = int(ref_num)
            
            picture_page_num = picture['prov'][0]['page_no']
            picture_bbox = picture['prov'][0]['bbox']
            picture_bbox = [
                picture_bbox['l'],
                picture_bbox['t'], 
                picture_bbox['r'],
                picture_bbox['b']
            ]
            
            picture_obj = {
                'picture_id': ref_num,
                'page': picture_page_num,
                'bbox': picture_bbox,
                'children': children_list,
            }
            assembled_pictures.append(picture_obj)
        return assembled_pictures
    
    def _process_picture_block(self, picture, data):
        children_list = []
        
        for item in picture['children']:
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)
                
                if ref_type == 'texts':
                    content_item = self._process_text_reference(ref_num, data)
                        
                    children_list.append(content_item)

        return children_list
