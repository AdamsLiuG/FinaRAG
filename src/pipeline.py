from dataclasses import dataclass
from pathlib import Path
from pyprojroot import here
import logging
import os
import json
import pandas as pd
import yaml

from src.pdf_parsing import PDFParser
from src.parsed_reports_merging import PageTextPreparation
from src.text_splitter import TextSplitter
from src.ingestion import VectorDBIngestor
from src.ingestion import BM25Ingestor
from src.ingestion import SparseLexicalIngestor
from src.questions_processing import QuestionsProcessor
from src.tables_serialization import TableSerializer


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

@dataclass
class PipelineConfig:
    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "questions.json", pdf_reports_dir_name: str = "pdf_reports", serialized: bool = False, config_suffix: str = ""):
        self.root_path = root_path
        suffix = "_ser_tab" if serialized else ""

        self.subset_path = root_path / subset_name
        self.document_manifest_path = self._resolve_manifest_path(root_path, self.subset_path)
        self.questions_file_path = root_path / questions_file_name
        self.pdf_reports_dir = root_path / pdf_reports_dir_name
        
        self.answers_file_path = root_path / f"answers{config_suffix}.json"       
        self.debug_data_path = root_path / "debug_data"
        self.databases_path = root_path / f"databases{suffix}"
        
        self.vector_db_dir = self.databases_path / "vector_dbs"
        self.documents_dir = self.databases_path / "chunked_reports"
        self.bm25_db_path = self.databases_path / "bm25_dbs"
        self.sparse_db_dir = self.databases_path / "sparse_dbs"

        self.parsed_reports_dirname = "01_parsed_reports"
        self.parsed_reports_debug_dirname = "01_parsed_reports_debug"
        self.merged_reports_dirname = f"02_merged_reports{suffix}"
        self.reports_markdown_dirname = f"03_reports_markdown{suffix}"

        self.parsed_reports_path = self.debug_data_path / self.parsed_reports_dirname
        self.parsed_reports_debug_path = self.debug_data_path / self.parsed_reports_debug_dirname
        self.merged_reports_path = self.debug_data_path / self.merged_reports_dirname
        self.reports_markdown_path = self.debug_data_path / self.reports_markdown_dirname

    @staticmethod
    def _resolve_manifest_path(root_path: Path, subset_path: Path) -> Path:
        candidates = [
            root_path / "document_manifest.csv",
            root_path / "document_manifest.json",
            subset_path,
            root_path / "subset.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return subset_path

@dataclass
class RunConfig:
    use_serialized_tables: bool = False
    parent_document_retrieval: bool = False
    parent_retrieval_mode: str = "page"
    use_vector_dbs: bool = True
    vector_index_type: str = "flat"
    vector_search_k: int = 0
    vector_ivf_nlist: int = 32
    vector_ivf_nprobe: int = 8
    vector_hnsw_m: int = 32
    vector_hnsw_ef_construction: int = 200
    vector_hnsw_ef_search: int = 64
    retriever_cache_enabled: bool = True
    use_bm25_db: bool = False
    use_sparse_lexical_db: bool = False
    llm_reranking: bool = False
    llm_reranking_sample_size: int = 30
    top_n_retrieval: int = 10
    parallel_requests: int = 10
    pipeline_details: str = ""
    full_context: bool = False
    api_provider: str = "qwen"
    answering_model: str = "Qwen3.5-35B-A3B-AWQ-4bit"
    config_suffix: str = ""
    document_language: str = "en"
    ocr_mode: str = "docling_rapidocr"
    doc_router_enabled: bool = False
    candidate_doc_top_k: int = 5
    numeric_grounding_enabled: bool = False
    reasoning_debug_enabled: bool = True


def run_config_from_dict(data: dict) -> RunConfig:
    allowed_fields = {
        "use_serialized_tables",
        "parent_document_retrieval",
        "parent_retrieval_mode",
        "use_vector_dbs",
        "vector_index_type",
        "vector_search_k",
        "vector_ivf_nlist",
        "vector_ivf_nprobe",
        "vector_hnsw_m",
        "vector_hnsw_ef_construction",
        "vector_hnsw_ef_search",
        "retriever_cache_enabled",
        "use_bm25_db",
        "use_sparse_lexical_db",
        "llm_reranking",
        "llm_reranking_sample_size",
        "top_n_retrieval",
        "parallel_requests",
        "pipeline_details",
        "full_context",
        "api_provider",
        "answering_model",
        "config_suffix",
        "document_language",
        "ocr_mode",
        "doc_router_enabled",
        "candidate_doc_top_k",
        "numeric_grounding_enabled",
        "reasoning_debug_enabled",
    }
    payload = {key: value for key, value in data.items() if key in allowed_fields}
    return RunConfig(**payload)


def load_run_config(config_path: Path) -> RunConfig:
    with open(config_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {config_path} must define a mapping.")
    return run_config_from_dict(data)

class Pipeline:
    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "questions.json", pdf_reports_dir_name: str = "pdf_reports", run_config: RunConfig = RunConfig()):
        self.run_config = run_config
        self.paths = self._initialize_paths(root_path, subset_name, questions_file_name, pdf_reports_dir_name)
        self._convert_json_to_csv_if_needed()

    def _initialize_paths(self, root_path: Path, subset_name: str, questions_file_name: str, pdf_reports_dir_name: str) -> PipelineConfig:
        """Initialize paths configuration based on run config settings"""
        return PipelineConfig(
            root_path=root_path,
            subset_name=subset_name,
            questions_file_name=questions_file_name,
            pdf_reports_dir_name=pdf_reports_dir_name,
            serialized=self.run_config.use_serialized_tables,
            config_suffix=self.run_config.config_suffix
        )

    def _convert_json_to_csv_if_needed(self):
        """
        Checks if subset.json exists in root dir and subset.csv is absent.
        If so, converts the JSON to CSV format.
        """
        json_path = self.paths.root_path / "subset.json"
        csv_path = self.paths.root_path / "subset.csv"
        
        if json_path.exists() and not csv_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                df = pd.DataFrame(data)
                
                df.to_csv(csv_path, index=False)
                
            except Exception as e:
                print(f"Error converting JSON to CSV: {str(e)}")

# Docling automatically downloads some models from huggingface when first used
# I wanted to download them prior to running the pipeline and created this crutch
    @staticmethod
    def download_docling_models(): 
        logging.basicConfig(level=logging.DEBUG)
        parser = PDFParser(output_dir=here())
        parser.parse_and_export(input_doc_paths=[here() / "src/dummy_report.pdf"])

    def parse_pdf_reports_sequential(self, cuda_devices: str | None = None):
        logging.basicConfig(level=logging.DEBUG)
        if cuda_devices:
            first_device = str(cuda_devices).split(",", 1)[0].strip()
            if first_device:
                os.environ["CUDA_VISIBLE_DEVICES"] = first_device
        
        pdf_parser = PDFParser(
            output_dir=self.paths.parsed_reports_path,
            csv_metadata_path=self.paths.document_manifest_path,
            document_language=self.run_config.document_language,
            ocr_mode=self.run_config.ocr_mode,
        )
        pdf_parser.debug_data_path = self.paths.parsed_reports_debug_path
            
        pdf_parser.parse_and_export(doc_dir=self.paths.pdf_reports_dir)
        print(f"PDF reports parsed and saved to {self.paths.parsed_reports_path}")

    def parse_pdf_reports_parallel(self, chunk_size: int = 2, max_workers: int = 10, cuda_devices: str | None = None):
        """Parse PDF reports in parallel using multiple processes.
        
        Args:
            chunk_size: Number of PDFs to process in each worker
            num_workers: Number of parallel worker processes to use
        """
        logging.basicConfig(level=logging.DEBUG)
        
        pdf_parser = PDFParser(
            output_dir=self.paths.parsed_reports_path,
            csv_metadata_path=self.paths.document_manifest_path,
            document_language=self.run_config.document_language,
            ocr_mode=self.run_config.ocr_mode,
        )
        pdf_parser.debug_data_path = self.paths.parsed_reports_debug_path

        input_doc_paths = list(self.paths.pdf_reports_dir.glob("*.pdf"))
        
        pdf_parser.parse_and_export_parallel(
            input_doc_paths=input_doc_paths,
            optimal_workers=max_workers,
            chunk_size=chunk_size,
            cuda_devices=cuda_devices,
        )
        print(f"PDF reports parsed and saved to {self.paths.parsed_reports_path}")

    def serialize_tables(self, max_workers: int = 10):
        """Process tables in files using parallel threading"""
        serializer = TableSerializer(
            provider=self.run_config.api_provider,
            model=self.run_config.answering_model
        )
        serializer.process_directory_parallel(
            self.paths.parsed_reports_path,
            max_workers=max_workers
        )

    def merge_reports(self):
        """Merge complex JSON reports into a simpler structure with a list of pages, where all text blocks are combined into a single string."""
        ptp = PageTextPreparation(use_serialized_tables=self.run_config.use_serialized_tables)
        _ = ptp.process_reports(
            reports_dir=self.paths.parsed_reports_path,
            output_dir=self.paths.merged_reports_path
        )
        print(f"Reports saved to {self.paths.merged_reports_path}")

    def export_reports_to_markdown(self):
        """Export processed reports to markdown format for review."""
        ptp = PageTextPreparation(use_serialized_tables=self.run_config.use_serialized_tables)
        ptp.export_to_markdown(
            reports_dir=self.paths.parsed_reports_path,
            output_dir=self.paths.reports_markdown_path
        )
        print(f"Reports saved to {self.paths.reports_markdown_path}")

    def chunk_reports(self, include_serialized_tables: bool = False):
        """Split processed reports into smaller chunks for better processing."""
        text_splitter = TextSplitter()
        
        serialized_tables_dir = None
        if include_serialized_tables:
            serialized_tables_dir = self.paths.parsed_reports_path
        
        text_splitter.split_all_reports(
            self.paths.merged_reports_path,
            self.paths.documents_dir,
            serialized_tables_dir
        )
        print(f"Chunked reports saved to {self.paths.documents_dir}")

    def create_vector_dbs(self):
        """Create vector databases from chunked reports."""
        input_dir = self.paths.documents_dir
        output_dir = self.paths.vector_db_dir
        
        vdb_ingestor = VectorDBIngestor(
            index_type=self.run_config.vector_index_type,
            ivf_nlist=self.run_config.vector_ivf_nlist,
            hnsw_m=self.run_config.vector_hnsw_m,
            hnsw_ef_construction=self.run_config.vector_hnsw_ef_construction,
        )
        vdb_ingestor.process_reports(input_dir, output_dir)
        print(f"Vector databases created in {output_dir}")
    
    def create_bm25_db(self):
        """Create BM25 database from chunked reports."""
        input_dir = self.paths.documents_dir
        output_file = self.paths.bm25_db_path
        
        bm25_ingestor = BM25Ingestor()
        bm25_ingestor.process_reports(input_dir, output_file)
        print(f"BM25 database created at {output_file}")

    def create_sparse_db(self):
        """Create bge-m3 sparse lexical database from chunked reports."""
        input_dir = self.paths.documents_dir
        output_dir = self.paths.sparse_db_dir

        sparse_ingestor = SparseLexicalIngestor()
        sparse_ingestor.process_reports(input_dir, output_dir)
        print(f"Sparse lexical databases created in {output_dir}")
    
    def parse_pdf_reports(self, parallel: bool = True, chunk_size: int = 2, max_workers: int = 10, cuda_devices: str | None = None):
        if parallel:
            self.parse_pdf_reports_parallel(chunk_size=chunk_size, max_workers=max_workers, cuda_devices=cuda_devices)
        else:
            self.parse_pdf_reports_sequential(cuda_devices=cuda_devices)
    
    def process_parsed_reports(self):
        """Process already parsed PDF reports through the pipeline:
        1. Merge to simpler JSON structure
        2. Export to markdown
        3. Chunk the reports
        4. Create retrieval databases
        """
        print("Starting reports processing pipeline...")
        
        print("Step 1: Merging reports...")
        self.merge_reports()
        
        print("Step 2: Exporting reports to markdown...")
        self.export_reports_to_markdown()
        
        print("Step 3: Chunking reports...")
        self.chunk_reports(include_serialized_tables=self.run_config.use_serialized_tables)
        
        if self.run_config.use_vector_dbs:
            print("Step 4: Creating vector databases...")
            self.create_vector_dbs()

        if self.run_config.use_bm25_db:
            print("Step 5: Creating BM25 database...")
            self.create_bm25_db()

        if self.run_config.use_sparse_lexical_db:
            print("Step 6: Creating sparse lexical databases...")
            self.create_sparse_db()
        
        print("Reports processing pipeline completed successfully!")
        
    def _get_next_available_filename(self, base_path: Path) -> Path:
        """
        Returns the next available filename by adding a numbered suffix if the file exists.
        Example: If answers.json exists, returns answers_01.json, etc.
        """
        if not base_path.exists():
            return base_path
            
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        
        counter = 1
        while True:
            new_filename = f"{stem}_{counter:02d}{suffix}"
            new_path = parent / new_filename
            
            if not new_path.exists():
                return new_path
            counter += 1

    def process_questions(self):
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            bm25_db_path=self.paths.bm25_db_path,
            sparse_db_dir=self.paths.sparse_db_dir,
            documents_dir=self.paths.documents_dir,
            questions_file_path=self.paths.questions_file_path,
            subset_path=self.paths.document_manifest_path,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            parent_retrieval_mode=self.run_config.parent_retrieval_mode,
            use_vector_dbs=self.run_config.use_vector_dbs,
            use_bm25_db=self.run_config.use_bm25_db,
            use_sparse_lexical_db=self.run_config.use_sparse_lexical_db,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            vector_search_k=self.run_config.vector_search_k,
            vector_ivf_nprobe=self.run_config.vector_ivf_nprobe,
            vector_hnsw_ef_search=self.run_config.vector_hnsw_ef_search,
            retriever_cache_enabled=self.run_config.retriever_cache_enabled,
            parallel_requests=self.run_config.parallel_requests,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            full_context=self.run_config.full_context,
            document_language=self.run_config.document_language,
            doc_router_enabled=self.run_config.doc_router_enabled,
            candidate_doc_top_k=self.run_config.candidate_doc_top_k,
            numeric_grounding_enabled=self.run_config.numeric_grounding_enabled,
            reasoning_debug_enabled=self.run_config.reasoning_debug_enabled,
        )
        
        output_path = self._get_next_available_filename(self.paths.answers_file_path)
        
        _ = processor.process_all_questions(
            output_path=output_path,
            pipeline_details=self.run_config.pipeline_details
        )
        print(f"Answers saved to {output_path}")
        return output_path


preprocess_configs = {
    "ser_tab": RunConfig(
        use_serialized_tables=True,
        use_vector_dbs=True,
        use_bm25_db=True,
        use_sparse_lexical_db=True,
    ),
    "no_ser_tab": RunConfig(
        use_serialized_tables=False,
        use_vector_dbs=True,
        use_bm25_db=True,
        use_sparse_lexical_db=True,
    ),
}

# 从 .env 读取模型名称，修改 QWEN_MODEL 即可切换模型，无需改代码
from dotenv import load_dotenv as _load_dotenv
_load_dotenv()
_qwen_model = os.getenv("QWEN_MODEL", "Qwen3.5-35B-A3B-AWQ-4bit")
_qwen_parallel_requests = int(os.getenv("QWEN_PARALLEL_REQUESTS", "1"))
_qwen_parent_document_retrieval = _env_flag("QWEN_PARENT_DOCUMENT_RETRIEVAL", default=False)
_qwen_parent_retrieval_mode = os.getenv("QWEN_PARENT_RETRIEVAL_MODE", "block").strip().lower()
_qwen_top_n_retrieval = int(os.getenv("QWEN_TOP_N_RETRIEVAL", "4"))
_qwen_llm_reranking_sample_size = int(os.getenv("QWEN_LLM_RERANKING_SAMPLE_SIZE", "8"))
_qwen_document_language = os.getenv("QWEN_DOCUMENT_LANGUAGE", "en")
_qwen_ocr_mode = os.getenv("QWEN_OCR_MODE", "docling_rapidocr")
_qwen_doc_router_enabled = _env_flag("QWEN_DOC_ROUTER_ENABLED", default=False)
_qwen_candidate_doc_top_k = int(os.getenv("QWEN_CANDIDATE_DOC_TOP_K", "5"))
_qwen_numeric_grounding_enabled = _env_flag("QWEN_NUMERIC_GROUNDING_ENABLED", default=False)
_qwen_reasoning_debug_enabled = _env_flag("QWEN_REASONING_DEBUG_ENABLED", default=True)
_qwen_vector_index_type = os.getenv("QWEN_VECTOR_INDEX_TYPE", os.getenv("VECTOR_INDEX_TYPE", "flat")).strip().lower()
_qwen_vector_search_k = int(os.getenv("QWEN_VECTOR_SEARCH_K", os.getenv("VECTOR_SEARCH_K", "0")))
_qwen_vector_ivf_nlist = int(os.getenv("QWEN_VECTOR_IVF_NLIST", os.getenv("VECTOR_IVF_NLIST", "32")))
_qwen_vector_ivf_nprobe = int(os.getenv("QWEN_VECTOR_IVF_NPROBE", os.getenv("VECTOR_IVF_NPROBE", "8")))
_qwen_vector_hnsw_m = int(os.getenv("QWEN_VECTOR_HNSW_M", os.getenv("VECTOR_HNSW_M", "32")))
_qwen_vector_hnsw_ef_construction = int(
    os.getenv("QWEN_VECTOR_HNSW_EF_CONSTRUCTION", os.getenv("VECTOR_HNSW_EF_CONSTRUCTION", "200"))
)
_qwen_vector_hnsw_ef_search = int(
    os.getenv("QWEN_VECTOR_HNSW_EF_SEARCH", os.getenv("VECTOR_HNSW_EF_SEARCH", "64"))
)
_qwen_retriever_cache_enabled = _env_flag("QWEN_RETRIEVER_CACHE_ENABLED", default=True)

qwen_base_config = RunConfig(
    parent_document_retrieval=_qwen_parent_document_retrieval,
    parent_retrieval_mode=_qwen_parent_retrieval_mode,
    use_vector_dbs=True,
    vector_index_type=_qwen_vector_index_type,
    vector_search_k=_qwen_vector_search_k,
    vector_ivf_nlist=_qwen_vector_ivf_nlist,
    vector_ivf_nprobe=_qwen_vector_ivf_nprobe,
    vector_hnsw_m=_qwen_vector_hnsw_m,
    vector_hnsw_ef_construction=_qwen_vector_hnsw_ef_construction,
    vector_hnsw_ef_search=_qwen_vector_hnsw_ef_search,
    retriever_cache_enabled=_qwen_retriever_cache_enabled,
    use_bm25_db=False,
    top_n_retrieval=_qwen_top_n_retrieval,
    parallel_requests=_qwen_parallel_requests,
    pipeline_details="PDF解析 + 本地Embedding + Parent-Child检索 + CoT推理",
    api_provider="qwen",
    answering_model=_qwen_model,
    config_suffix="_qwen_base",
    document_language=_qwen_document_language,
    ocr_mode=_qwen_ocr_mode,
    doc_router_enabled=_qwen_doc_router_enabled,
    candidate_doc_top_k=_qwen_candidate_doc_top_k,
    numeric_grounding_enabled=_qwen_numeric_grounding_enabled,
    reasoning_debug_enabled=_qwen_reasoning_debug_enabled,
)

qwen_vector_rerank_config = RunConfig(
    parent_document_retrieval=_qwen_parent_document_retrieval,
    parent_retrieval_mode=_qwen_parent_retrieval_mode,
    use_vector_dbs=True,
    vector_index_type=_qwen_vector_index_type,
    vector_search_k=_qwen_vector_search_k,
    vector_ivf_nlist=_qwen_vector_ivf_nlist,
    vector_ivf_nprobe=_qwen_vector_ivf_nprobe,
    vector_hnsw_m=_qwen_vector_hnsw_m,
    vector_hnsw_ef_construction=_qwen_vector_hnsw_ef_construction,
    vector_hnsw_ef_search=_qwen_vector_hnsw_ef_search,
    retriever_cache_enabled=_qwen_retriever_cache_enabled,
    use_bm25_db=False,
    llm_reranking=True,
    llm_reranking_sample_size=_qwen_llm_reranking_sample_size,
    top_n_retrieval=_qwen_top_n_retrieval,
    parallel_requests=_qwen_parallel_requests,
    pipeline_details="PDF解析 + 本地Embedding + Parent-Child检索 + 向量召回 + LLM重排 + CoT推理",
    api_provider="qwen",
    answering_model=_qwen_model,
    config_suffix="_qwen_vector_rerank",
    document_language=_qwen_document_language,
    ocr_mode=_qwen_ocr_mode,
    doc_router_enabled=_qwen_doc_router_enabled,
    candidate_doc_top_k=_qwen_candidate_doc_top_k,
    numeric_grounding_enabled=_qwen_numeric_grounding_enabled,
    reasoning_debug_enabled=_qwen_reasoning_debug_enabled,
)

qwen_rerank_config = RunConfig(
    parent_document_retrieval=_qwen_parent_document_retrieval,
    parent_retrieval_mode=_qwen_parent_retrieval_mode,
    use_vector_dbs=True,
    vector_index_type=_qwen_vector_index_type,
    vector_search_k=_qwen_vector_search_k,
    vector_ivf_nlist=_qwen_vector_ivf_nlist,
    vector_ivf_nprobe=_qwen_vector_ivf_nprobe,
    vector_hnsw_m=_qwen_vector_hnsw_m,
    vector_hnsw_ef_construction=_qwen_vector_hnsw_ef_construction,
    vector_hnsw_ef_search=_qwen_vector_hnsw_ef_search,
    retriever_cache_enabled=_qwen_retriever_cache_enabled,
    use_bm25_db=True,
    llm_reranking=True,
    llm_reranking_sample_size=_qwen_llm_reranking_sample_size,
    top_n_retrieval=_qwen_top_n_retrieval,
    parallel_requests=_qwen_parallel_requests,
    pipeline_details="PDF解析 + 本地Embedding + BM25 + Parent-Child检索 + 混合召回 + LLM重排 + CoT推理",
    api_provider="qwen",
    answering_model=_qwen_model,
    config_suffix="_qwen_rerank",
    document_language=_qwen_document_language,
    ocr_mode=_qwen_ocr_mode,
    doc_router_enabled=_qwen_doc_router_enabled,
    candidate_doc_top_k=_qwen_candidate_doc_top_k,
    numeric_grounding_enabled=_qwen_numeric_grounding_enabled,
    reasoning_debug_enabled=_qwen_reasoning_debug_enabled,
)

qwen_sparse_rerank_config = RunConfig(
    parent_document_retrieval=_qwen_parent_document_retrieval,
    parent_retrieval_mode=_qwen_parent_retrieval_mode,
    use_vector_dbs=True,
    vector_index_type=_qwen_vector_index_type,
    vector_search_k=_qwen_vector_search_k,
    vector_ivf_nlist=_qwen_vector_ivf_nlist,
    vector_ivf_nprobe=_qwen_vector_ivf_nprobe,
    vector_hnsw_m=_qwen_vector_hnsw_m,
    vector_hnsw_ef_construction=_qwen_vector_hnsw_ef_construction,
    vector_hnsw_ef_search=_qwen_vector_hnsw_ef_search,
    retriever_cache_enabled=_qwen_retriever_cache_enabled,
    use_bm25_db=False,
    use_sparse_lexical_db=True,
    llm_reranking=True,
    llm_reranking_sample_size=_qwen_llm_reranking_sample_size,
    top_n_retrieval=_qwen_top_n_retrieval,
    parallel_requests=_qwen_parallel_requests,
    pipeline_details="PDF解析 + 本地Embedding + bge-m3 sparse lexical + Parent-Child检索 + 混合召回 + LLM重排 + CoT推理",
    api_provider="qwen",
    answering_model=_qwen_model,
    config_suffix="_qwen_sparse_rerank",
    document_language=_qwen_document_language,
    ocr_mode=_qwen_ocr_mode,
    doc_router_enabled=_qwen_doc_router_enabled,
    candidate_doc_top_k=_qwen_candidate_doc_top_k,
    numeric_grounding_enabled=_qwen_numeric_grounding_enabled,
    reasoning_debug_enabled=_qwen_reasoning_debug_enabled,
)

qwen_ser_vector_rerank_config = RunConfig(
    use_serialized_tables=True,
    parent_document_retrieval=_qwen_parent_document_retrieval,
    parent_retrieval_mode=_qwen_parent_retrieval_mode,
    use_vector_dbs=True,
    vector_index_type=_qwen_vector_index_type,
    vector_search_k=_qwen_vector_search_k,
    vector_ivf_nlist=_qwen_vector_ivf_nlist,
    vector_ivf_nprobe=_qwen_vector_ivf_nprobe,
    vector_hnsw_m=_qwen_vector_hnsw_m,
    vector_hnsw_ef_construction=_qwen_vector_hnsw_ef_construction,
    vector_hnsw_ef_search=_qwen_vector_hnsw_ef_search,
    retriever_cache_enabled=_qwen_retriever_cache_enabled,
    use_bm25_db=False,
    llm_reranking=True,
    llm_reranking_sample_size=_qwen_llm_reranking_sample_size,
    top_n_retrieval=_qwen_top_n_retrieval,
    parallel_requests=_qwen_parallel_requests,
    pipeline_details="PDF解析 + 表格序列化 + 本地Embedding + Parent-Child检索 + 向量召回 + LLM重排 + CoT推理",
    api_provider="qwen",
    answering_model=_qwen_model,
    config_suffix="_qwen_ser_vector_rerank",
    document_language=_qwen_document_language,
    ocr_mode=_qwen_ocr_mode,
    doc_router_enabled=_qwen_doc_router_enabled,
    candidate_doc_top_k=_qwen_candidate_doc_top_k,
    numeric_grounding_enabled=_qwen_numeric_grounding_enabled,
    reasoning_debug_enabled=_qwen_reasoning_debug_enabled,
)

qwen_ser_rerank_config = RunConfig(
    use_serialized_tables=True,
    parent_document_retrieval=_qwen_parent_document_retrieval,
    parent_retrieval_mode=_qwen_parent_retrieval_mode,
    use_vector_dbs=True,
    vector_index_type=_qwen_vector_index_type,
    vector_search_k=_qwen_vector_search_k,
    vector_ivf_nlist=_qwen_vector_ivf_nlist,
    vector_ivf_nprobe=_qwen_vector_ivf_nprobe,
    vector_hnsw_m=_qwen_vector_hnsw_m,
    vector_hnsw_ef_construction=_qwen_vector_hnsw_ef_construction,
    vector_hnsw_ef_search=_qwen_vector_hnsw_ef_search,
    retriever_cache_enabled=_qwen_retriever_cache_enabled,
    use_bm25_db=True,
    llm_reranking=True,
    llm_reranking_sample_size=_qwen_llm_reranking_sample_size,
    top_n_retrieval=_qwen_top_n_retrieval,
    parallel_requests=_qwen_parallel_requests,
    pipeline_details="PDF解析 + 表格序列化 + 本地Embedding + BM25 + Parent-Child检索 + 混合召回 + LLM重排 + CoT推理",
    api_provider="qwen",
    answering_model=_qwen_model,
    config_suffix="_qwen_ser_rerank",
    document_language=_qwen_document_language,
    ocr_mode=_qwen_ocr_mode,
    doc_router_enabled=_qwen_doc_router_enabled,
    candidate_doc_top_k=_qwen_candidate_doc_top_k,
    numeric_grounding_enabled=_qwen_numeric_grounding_enabled,
    reasoning_debug_enabled=_qwen_reasoning_debug_enabled,
)

qwen_ser_sparse_rerank_config = RunConfig(
    use_serialized_tables=True,
    parent_document_retrieval=_qwen_parent_document_retrieval,
    parent_retrieval_mode=_qwen_parent_retrieval_mode,
    use_vector_dbs=True,
    vector_index_type=_qwen_vector_index_type,
    vector_search_k=_qwen_vector_search_k,
    vector_ivf_nlist=_qwen_vector_ivf_nlist,
    vector_ivf_nprobe=_qwen_vector_ivf_nprobe,
    vector_hnsw_m=_qwen_vector_hnsw_m,
    vector_hnsw_ef_construction=_qwen_vector_hnsw_ef_construction,
    vector_hnsw_ef_search=_qwen_vector_hnsw_ef_search,
    retriever_cache_enabled=_qwen_retriever_cache_enabled,
    use_bm25_db=False,
    use_sparse_lexical_db=True,
    llm_reranking=True,
    llm_reranking_sample_size=_qwen_llm_reranking_sample_size,
    top_n_retrieval=_qwen_top_n_retrieval,
    parallel_requests=_qwen_parallel_requests,
    pipeline_details="PDF解析 + 表格序列化 + 本地Embedding + bge-m3 sparse lexical + Parent-Child检索 + 混合召回 + LLM重排 + CoT推理",
    api_provider="qwen",
    answering_model=_qwen_model,
    config_suffix="_qwen_ser_sparse_rerank",
    document_language=_qwen_document_language,
    ocr_mode=_qwen_ocr_mode,
    doc_router_enabled=_qwen_doc_router_enabled,
    candidate_doc_top_k=_qwen_candidate_doc_top_k,
    numeric_grounding_enabled=_qwen_numeric_grounding_enabled,
    reasoning_debug_enabled=_qwen_reasoning_debug_enabled,
)

configs = {
    "qwen_base": qwen_base_config,
    "qwen_vector_rerank": qwen_vector_rerank_config,
    "qwen_rerank": qwen_rerank_config,
    "qwen_sparse_rerank": qwen_sparse_rerank_config,
    "qwen_ser_vector_rerank": qwen_ser_vector_rerank_config,
    "qwen_ser_rerank": qwen_ser_rerank_config,
    "qwen_ser_sparse_rerank": qwen_ser_sparse_rerank_config,
}


# 可以直接运行此文件来执行 pipeline 的某个阶段
# python src/pipeline.py
# 取消注释想要执行的方法即可
if __name__ == "__main__":
    root_path = here() / "data" / "test_set"
    pipeline = Pipeline(root_path, run_config=qwen_base_config)
    
    # 1. 解析 PDF -> JSON（含版面分析、表格识别）
    # pipeline.parse_pdf_reports_sequential() 
    
    # 2. 表格序列化（仅在使用 ser_tab 配置时需要）
    # pipeline.serialize_tables(max_workers=5) 
    
    # 3. 合并解析结果为简化的页面级 JSON
    # pipeline.merge_reports() 

    # 4. 导出为 Markdown 格式（用于人工审查）
    # pipeline.export_reports_to_markdown() 

    # 5. 文档切块
    # pipeline.chunk_reports() 
    
    # 6. 构建向量数据库
    # pipeline.create_vector_dbs() 
    
    # 7. 处理问答
    # pipeline.process_questions() 
