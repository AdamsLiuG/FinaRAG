from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.document_manifest import load_document_manifest
from src.pipeline import load_run_config
from src.questions_processing import QuestionsProcessor


st.set_page_config(
    page_title="FinaRAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

CONFIG_DIR = PROJECT_ROOT / "config"
AVAILABLE_CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))
PREFERRED_CONFIGS = [
    CONFIG_DIR / "qwen_zh_finance.yaml",
    CONFIG_DIR / "qwen_ser_rerank.yaml",
]
DEFAULT_CONFIG = next((path for path in PREFERRED_CONFIGS if path.exists()), AVAILABLE_CONFIGS[0] if AVAILABLE_CONFIGS else None)

PREFERRED_DATASETS = [
    PROJECT_ROOT / "data" / "chinese_annual_reports_2024",
    PROJECT_ROOT / "data" / "test_set",
]
DEFAULT_DATASET_DIR = next((path for path in PREFERRED_DATASETS if path.exists()), PROJECT_ROOT / "data" / "test_set")
UPLOAD_WORKSPACE_DIR = PROJECT_ROOT / "data" / "upload_workspace"

QUESTION_KIND_OPTIONS = ["boolean", "number", "name"]
WORKSPACE_OPTIONS = ["研究工作台", "检索依据", "系统监控"]
CORPUS_VIEW_OPTIONS = ["混合语料", "中文年报", "中文研报"]
MAX_SAMPLE_QUESTIONS = 12
MAX_HISTORY_ITEMS = 6
DEFAULT_QUESTION = "贵州茅台2024年年报中的法定代表人是谁？"
QUESTION_KIND_LABELS = {
    "boolean": "是非判断",
    "number": "数值指标",
    "name": "名称实体",
}


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #07111f;
            --bg-soft: #0b1628;
            --panel: rgba(10, 22, 40, 0.88);
            --panel-strong: rgba(11, 24, 42, 0.96);
            --panel-elevated: linear-gradient(180deg, rgba(13, 27, 47, 0.98), rgba(8, 18, 32, 0.94));
            --border: rgba(120, 154, 194, 0.20);
            --border-strong: rgba(120, 154, 194, 0.34);
            --text: #f4f8ff;
            --muted: #8fa5c5;
            --accent: #20c9c9;
            --accent-strong: #78f0ff;
            --gold: #d3aa61;
            --success: #7ce0b4;
            --danger: #ff8b92;
            --shadow: 0 24px 54px rgba(0, 0, 0, 0.34);
        }

        html, body, [class*="css"]  {
            font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
        }

        .stApp {
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(32, 201, 201, 0.09), transparent 30%),
                radial-gradient(circle at top right, rgba(211, 170, 97, 0.08), transparent 32%),
                linear-gradient(180deg, #06101c 0%, #07111f 100%);
        }

        [data-testid="stAppViewContainer"] > .main {
            background: transparent;
        }

        .block-container {
            max-width: 1580px;
            padding-top: 1.2rem;
            padding-bottom: 3rem;
        }

        header[data-testid="stHeader"],
        #MainMenu,
        footer {
            visibility: hidden;
        }

        section[data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(8, 16, 29, 0.98) 0%, rgba(7, 16, 28, 0.94) 100%);
            border-right: 1px solid var(--border);
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
        }

        h1, h2, h3, h4 {
            color: var(--text);
            letter-spacing: -0.02em;
            font-family: "IBM Plex Sans Condensed", "IBM Plex Sans", "Avenir Next", sans-serif;
        }

        p, li, span, div, label {
            color: var(--text);
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--panel-elevated);
            border: 1px solid var(--border);
            border-radius: 24px;
            box-shadow: var(--shadow);
            padding: 0.25rem 0.35rem;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(16, 31, 54, 0.94), rgba(9, 19, 33, 0.92));
            border: 1px solid var(--border);
            border-radius: 18px;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
            padding: 0.9rem 0.95rem;
        }

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: var(--muted);
        }

        div[data-testid="stMetricValue"] {
            color: var(--text);
            font-weight: 700;
        }

        .stTextArea textarea,
        .stTextInput input,
        div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"] > div {
            background: rgba(9, 20, 35, 0.96) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 18px !important;
        }

        .stTextArea textarea {
            min-height: 220px;
            font-size: 1rem;
            line-height: 1.7;
        }

        .stButton button,
        .stDownloadButton button,
        div[data-testid="stFormSubmitButton"] button {
            border-radius: 999px;
            border: 1px solid rgba(35, 208, 222, 0.22);
            background: linear-gradient(135deg, #0b8ca0 0%, #0fb7bc 100%);
            color: #ffffff;
            font-weight: 700;
            box-shadow: 0 12px 28px rgba(9, 153, 173, 0.28);
            transition: transform 120ms ease, box-shadow 120ms ease;
        }

        .stButton button:hover,
        .stDownloadButton button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {
            transform: translateY(-1px);
            box-shadow: 0 16px 30px rgba(9, 153, 173, 0.34);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.55rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            background: rgba(11, 22, 39, 0.86);
            border: 1px solid var(--border);
            padding: 0.45rem 0.95rem;
            color: var(--muted);
        }

        .stTabs [aria-selected="true"] {
            background: rgba(32, 201, 201, 0.12);
            border-color: rgba(32, 201, 201, 0.34);
            color: var(--text);
        }

        .stAlert {
            border-radius: 18px;
            border: 1px solid var(--border);
            background: rgba(10, 20, 35, 0.94);
        }

        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 2.1fr) minmax(320px, 0.9fr);
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .brand-hero,
        .status-panel,
        .sidebar-brand {
            background: linear-gradient(180deg, rgba(12, 25, 44, 0.98) 0%, rgba(8, 18, 32, 0.96) 100%);
            border: 1px solid var(--border);
            border-radius: 28px;
            box-shadow: var(--shadow);
        }

        .brand-hero {
            position: relative;
            overflow: hidden;
            padding: 1.8rem 1.9rem;
            background:
                radial-gradient(circle at top right, rgba(32, 201, 201, 0.16), transparent 32%),
                radial-gradient(circle at bottom left, rgba(211, 170, 97, 0.14), transparent 34%),
                linear-gradient(180deg, rgba(12, 25, 44, 0.98) 0%, rgba(7, 16, 28, 0.98) 100%);
        }

        .brand-hero::after {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg, transparent 0%, rgba(120, 154, 194, 0.06) 50%, transparent 100%);
            pointer-events: none;
        }

        .status-panel {
            padding: 1.2rem 1.25rem;
        }

        .sidebar-brand {
            padding: 1rem 1rem 0.95rem 1rem;
            margin-bottom: 0.8rem;
        }

        .overline {
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            color: var(--muted);
            margin-bottom: 0.7rem;
        }

        .hero-title {
            margin: 0;
            font-size: 3rem;
            line-height: 1.02;
            color: var(--text);
        }

        .hero-subtitle {
            margin: 0.85rem 0 0 0;
            max-width: 54rem;
            color: #c6d5e8;
            font-size: 1.02rem;
            line-height: 1.7;
        }

        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.38rem 0.78rem;
            border-radius: 999px;
            border: 1px solid rgba(32, 201, 201, 0.22);
            background: rgba(32, 201, 201, 0.10);
            color: #d8fcff;
            font-size: 0.82rem;
            font-weight: 600;
        }

        .badge.gold {
            border-color: rgba(211, 170, 97, 0.28);
            background: rgba(211, 170, 97, 0.12);
            color: #f0d59c;
        }

        .sidebar-heading,
        .panel-heading {
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: var(--muted);
            margin: 0.3rem 0 0.8rem 0;
        }

        .sidebar-divider {
            margin: 0.8rem 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(120, 154, 194, 0.18), transparent);
        }

        .workspace-lead {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.65;
        }

        .answer-shell,
        .callout-shell,
        .source-shell,
        .monitor-shell {
            background: linear-gradient(180deg, rgba(13, 27, 47, 0.92), rgba(9, 19, 33, 0.94));
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 1rem 1.05rem;
        }

        .answer-shell {
            padding: 1.15rem 1.15rem 1.2rem 1.15rem;
            background:
                radial-gradient(circle at top right, rgba(32, 201, 201, 0.14), transparent 34%),
                linear-gradient(180deg, rgba(14, 31, 53, 0.98), rgba(9, 19, 33, 0.98));
        }

        .answer-label {
            font-size: 0.76rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.6rem;
        }

        .answer-value {
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: var(--text);
            line-height: 1.2;
        }

        .answer-caption {
            margin-top: 0.9rem;
            color: #c9d7ea;
            line-height: 1.7;
        }

        .bullet-list {
            margin: 0;
            padding-left: 1.15rem;
            color: var(--text);
            line-height: 1.8;
        }

        .bullet-list li {
            margin-bottom: 0.4rem;
        }

        .source-title {
            font-weight: 700;
            color: var(--text);
            letter-spacing: -0.01em;
        }

        .source-meta {
            margin-top: 0.2rem;
            color: var(--muted);
            font-size: 0.86rem;
        }

        .source-snippet {
            margin-top: 0.8rem;
            color: #d5e2f4;
            font-size: 0.95rem;
            line-height: 1.7;
            word-break: break-word;
        }

        .tiny-note {
            color: var(--muted);
            font-size: 0.87rem;
            line-height: 1.6;
        }

        .workflow-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 0.6rem;
        }

        .workflow-step {
            background: linear-gradient(180deg, rgba(15, 28, 49, 0.92), rgba(9, 18, 33, 0.94));
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1rem 1rem 0.95rem 1rem;
        }

        .workflow-step .step-title {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: var(--muted);
            margin-bottom: 0.55rem;
        }

        .workflow-step .step-body {
            font-size: 0.96rem;
            color: var(--text);
            line-height: 1.65;
        }

        .history-item {
            padding: 0.65rem 0.7rem;
            border-radius: 16px;
            background: rgba(9, 20, 35, 0.78);
            border: 1px solid rgba(120, 154, 194, 0.16);
            margin-bottom: 0.55rem;
        }

        .history-item .history-question {
            font-size: 0.92rem;
            font-weight: 600;
            color: var(--text);
            line-height: 1.5;
        }

        .history-item .history-meta {
            color: var(--muted);
            font-size: 0.78rem;
            margin-top: 0.3rem;
        }

        .placeholder-card {
            background: linear-gradient(180deg, rgba(10, 21, 37, 0.72), rgba(8, 18, 32, 0.86));
            border: 1px dashed rgba(120, 154, 194, 0.28);
            border-radius: 22px;
            padding: 1.1rem 1.1rem;
        }

        .placeholder-card h4 {
            margin: 0 0 0.65rem 0;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
            border: 1px solid rgba(124, 224, 180, 0.22);
            background: rgba(124, 224, 180, 0.09);
            color: #c9ffe6;
            font-size: 0.78rem;
            font-weight: 600;
        }

        .status-pill.warn {
            border-color: rgba(211, 170, 97, 0.22);
            background: rgba(211, 170, 97, 0.10);
            color: #f7deb1;
        }

        .status-pill.danger {
            border-color: rgba(255, 139, 146, 0.28);
            background: rgba(255, 139, 146, 0.11);
            color: #ffc9cd;
        }

        @media (max-width: 1080px) {
            .hero-grid,
            .workflow-row {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _truncate(text: str, limit: int = 96) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _render_chip_row(values: List[Any], tone: str = "cyan") -> str:
    filtered = [str(value).strip() for value in values if value not in (None, "", [], {})]
    if not filtered:
        return ""

    chip_class = "badge gold" if tone == "gold" else "badge"
    chips = "".join(f"<span class='{chip_class}'>{escape(value)}</span>" for value in filtered)
    return f"<div class='badge-row'>{chips}</div>"


def _dataset_label(path: Path) -> str:
    labels = {
        "chinese_annual_reports_2024": "中文年报数据集 2024",
        "test_set": "内置测试集",
        "upload_workspace": "上传工作区",
    }
    return labels.get(path.name, path.name.replace("_", " ").title())


def _question_kind_label(kind: str) -> str:
    return QUESTION_KIND_LABELS.get(kind, kind)


def _discover_dataset_options() -> List[Tuple[str, Path]]:
    data_root = PROJECT_ROOT / "data"
    options: List[Tuple[str, Path]] = []
    if not data_root.exists():
        return options

    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        if any(
            (child / name).exists()
            for name in ("document_manifest.csv", "document_manifest.json", "subset.csv", "subset.json", "pdf_reports")
        ):
            options.append((_dataset_label(child), child))

    if not options and DEFAULT_DATASET_DIR.exists():
        options.append((_dataset_label(DEFAULT_DATASET_DIR), DEFAULT_DATASET_DIR))

    return options


def _provider_status(run_config) -> Dict[str, str]:
    provider_prefix = run_config.api_provider.upper()
    base_url = os.getenv(f"{provider_prefix}_BASE_URL") or os.getenv("LLM_BASE_URL") or ""
    model_name = os.getenv(f"{provider_prefix}_MODEL") or os.getenv("LLM_MODEL") or run_config.answering_model
    host = urlparse(base_url).netloc or base_url or "Not configured"
    return {
        "configured": "yes" if base_url else "no",
        "host": host,
        "model": model_name,
    }


def _resolve_data_paths(dataset_dir: Path, run_config) -> Dict[str, Path]:
    manifest_candidates = [
        dataset_dir / "document_manifest.csv",
        dataset_dir / "document_manifest.json",
        dataset_dir / "subset.csv",
        dataset_dir / "subset.json",
    ]
    manifest_path = next((path for path in manifest_candidates if path.exists()), manifest_candidates[0])
    database_root = dataset_dir / ("databases_ser_tab" if run_config.use_serialized_tables else "databases")
    return {
        "pdf_reports": dataset_dir / "pdf_reports",
        "manifest": manifest_path,
        "subset": manifest_path,
        "questions": dataset_dir / "questions.json",
        "documents": database_root / "chunked_reports",
        "vector": database_root / "vector_dbs",
        "bm25": database_root / "bm25_dbs",
        "sparse": database_root / "sparse_dbs",
    }


def _required_asset_messages(paths: Dict[str, Path], run_config) -> List[str]:
    messages = []
    if not paths["manifest"].exists():
        messages.append(f"缺少文档清单: {paths['manifest']}")
    if not paths["documents"].exists():
        messages.append(f"缺少切块结果目录: {paths['documents']}")
    if run_config.use_vector_dbs and not paths["vector"].exists():
        messages.append(f"缺少向量索引: {paths['vector']}")
    if run_config.use_bm25_db and not paths["bm25"].exists():
        messages.append(f"缺少 BM25 索引: {paths['bm25']}")
    if run_config.use_sparse_lexical_db and not paths["sparse"].exists():
        messages.append(f"缺少 sparse 索引: {paths['sparse']}")
    return messages


def _count_manifest_rows(manifest_path: Path) -> int:
    if not manifest_path.exists():
        return 0
    return len(load_document_manifest(manifest_path))


def _count_matching(directory: Path, pattern: str) -> int:
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.glob(pattern))


@st.cache_data(show_spinner=False)
def _load_question_samples(dataset_dir: str) -> List[Dict[str, str]]:
    questions_path = Path(dataset_dir) / "questions.json"
    if not questions_path.exists():
        return []

    try:
        payload = json.loads(questions_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    samples: List[Dict[str, str]] = []
    for item in payload[:MAX_SAMPLE_QUESTIONS]:
        question_text = str(item.get("text") or "").strip()
        question_kind = str(item.get("kind") or "boolean").strip().lower()
        if not question_text:
            continue
        if question_kind not in QUESTION_KIND_OPTIONS:
            question_kind = "boolean"
        samples.append({"text": question_text, "kind": question_kind})
    return samples


@st.cache_data(show_spinner=False)
def _load_company_snapshot(manifest_path: str, limit: int = 8) -> List[Dict[str, str]]:
    manifest = load_document_manifest(Path(manifest_path))
    if not manifest:
        return []

    companies: List[Dict[str, str]] = []
    for idx, row in enumerate(manifest.values()):
        if idx >= limit:
            break
        companies.append(
            {
                "company_name": str(row.get("company_name") or "").strip(),
                "currency": str(row.get("currency") or "").strip(),
                "major_industry": str(row.get("major_industry") or "").strip(),
                "doc_source_type": str(row.get("doc_source_type") or "").strip(),
                "broker_name": str(row.get("broker_name") or "").strip(),
            }
        )
    return companies


def _build_runtime_config(config_path: Path, model_name: str, top_k: int):
    runtime_config = deepcopy(load_run_config(config_path))
    runtime_config.answering_model = (model_name or runtime_config.answering_model).strip()
    runtime_config.top_n_retrieval = int(top_k)
    return runtime_config


def _processor_signature(dataset_dir: Path, run_config) -> Tuple[Any, ...]:
    paths = _resolve_data_paths(dataset_dir, run_config)
    return (
        str(dataset_dir.resolve()),
        str(paths["manifest"].resolve()),
        str(paths["documents"].resolve()),
        str(paths["vector"].resolve()),
        str(paths["bm25"].resolve()),
        str(paths["sparse"].resolve()),
        bool(run_config.parent_document_retrieval),
        str(run_config.parent_retrieval_mode),
        bool(run_config.use_vector_dbs),
        bool(run_config.use_bm25_db),
        bool(run_config.use_sparse_lexical_db),
        bool(run_config.llm_reranking),
        int(run_config.llm_reranking_sample_size),
        int(run_config.top_n_retrieval),
        int(run_config.parallel_requests),
        str(run_config.api_provider),
        str(run_config.answering_model),
        bool(run_config.full_context),
        str(run_config.document_language),
        bool(run_config.doc_router_enabled),
        int(run_config.candidate_doc_top_k),
        bool(run_config.numeric_grounding_enabled),
        bool(run_config.reasoning_debug_enabled),
        bool(getattr(run_config, "use_serialized_tables", False)),
    )


def _build_processor(dataset_dir: Path, run_config):
    paths = _resolve_data_paths(dataset_dir, run_config)
    return QuestionsProcessor(
        vector_db_dir=paths["vector"],
        bm25_db_path=paths["bm25"],
        sparse_db_dir=paths["sparse"],
        documents_dir=paths["documents"],
        subset_path=paths["subset"],
        parent_document_retrieval=run_config.parent_document_retrieval,
        parent_retrieval_mode=run_config.parent_retrieval_mode,
        use_vector_dbs=run_config.use_vector_dbs,
        use_bm25_db=run_config.use_bm25_db,
        use_sparse_lexical_db=run_config.use_sparse_lexical_db,
        llm_reranking=run_config.llm_reranking,
        llm_reranking_sample_size=run_config.llm_reranking_sample_size,
        top_n_retrieval=run_config.top_n_retrieval,
        parallel_requests=run_config.parallel_requests,
        api_provider=run_config.api_provider,
        answering_model=run_config.answering_model,
        answer_temperature=0.0,
        full_context=run_config.full_context,
        document_language=run_config.document_language,
        doc_router_enabled=run_config.doc_router_enabled,
        candidate_doc_top_k=run_config.candidate_doc_top_k,
        numeric_grounding_enabled=run_config.numeric_grounding_enabled,
        reasoning_debug_enabled=run_config.reasoning_debug_enabled,
    )


def _get_or_create_processor(dataset_dir: Path, run_config, answer_temperature: float) -> Tuple[QuestionsProcessor, bool, float]:
    signature = _processor_signature(dataset_dir, run_config)
    cache = st.session_state.processor_cache
    if signature in cache:
        processor = cache.pop(signature)
        cache[signature] = processor
        processor.answer_temperature = float(answer_temperature)
        cached_bundle = getattr(processor, "_demo_cached_retriever_bundle", None)
        if cached_bundle is not None:
            processor._retriever_cache.bundle = cached_bundle
        else:
            cached_bundle = processor._build_retriever()
            processor._demo_cached_retriever_bundle = cached_bundle
        return processor, True, 0.0

    started_at = time.perf_counter()
    processor = _build_processor(dataset_dir, run_config)
    processor.answer_temperature = float(answer_temperature)
    cached_bundle = processor._build_retriever()
    processor._demo_cached_retriever_bundle = cached_bundle
    processor._retriever_cache.bundle = cached_bundle
    cache[signature] = processor

    while len(cache) > 2:
        oldest_signature = next(iter(cache))
        if oldest_signature == signature:
            break
        cache.pop(oldest_signature, None)

    return processor, False, time.perf_counter() - started_at


def _humanize_route_mode(route_mode: str | None) -> str:
    mapping = {
        "explicit_company": "显式公司路由",
        "document_catalog": "文档目录路由",
        "comparative_explicit": "对比问答路由",
        "only_report_available": "单文档回退",
    }
    return mapping.get(route_mode or "", route_mode or "未知")


def _humanize_validation_flag(flag: str) -> str:
    mapping = {
        "missing_citations": "缺少 citation 覆盖",
        "missing_relevant_pages": "缺少相关页码",
        "currency_mismatch": "币种与问题不一致",
        "report_year_mismatch": "报告年份不一致",
        "doc_source_type_mismatch": "文档类型不一致",
        "period_filter_weak_match": "期间匹配较弱",
        "topic_filter_weak_match": "主题匹配较弱",
        "numeric_grounding_missing_value": "数字 grounding 缺少值",
        "numeric_grounding_period_mismatch": "grounding 期间不一致",
        "numeric_grounding_currency_mismatch": "grounding 币种不一致",
        "numeric_answer_without_table_grounding": "数字答案缺少表格 grounding",
        "no_retrieval_results": "未返回检索证据",
        "processing_error": "处理过程报错",
    }
    return mapping.get(flag, flag.replace("_", " ").title())


def _humanize_confidence(confidence: str | None) -> str:
    mapping = {
        "high": "高",
        "medium": "中",
        "low": "低",
        "unknown": "未知",
        "": "未知",
    }
    return mapping.get(str(confidence or "").lower(), str(confidence or "未知"))


def _split_summary_into_bullets(text: str, limit: int = 4) -> List[str]:
    if not text:
        return []
    normalized = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[。！？.!?])\s+", normalized)
    bullets = [part.strip().strip(" .") for part in parts if part.strip()]
    return bullets[:limit]


def _build_key_conclusions(result: Dict[str, Any]) -> List[str]:
    conclusions: List[str] = []
    final_answer = result.get("final_answer")
    if final_answer not in (None, "", "N/A"):
        conclusions.append(f"核心结论：{final_answer}")

    reasoning_summary = result.get("reasoning_summary") or ""
    conclusions.extend(_split_summary_into_bullets(reasoning_summary, limit=3))

    citations = result.get("citations") or []
    if citations:
        first = citations[0]
        page = first.get("page")
        source = first.get("company_name") or first.get("source") or "来源文档"
        conclusions.append(f"主要证据来自 {source} 的第 {page} 页。")

    deduped: List[str] = []
    seen = set()
    for item in conclusions:
        cleaned = item.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped[:4]


def _build_risk_notes(result: Dict[str, Any]) -> List[str]:
    notes: List[str] = []
    confidence = str(result.get("confidence") or "").lower()
    if confidence == "medium":
        notes.append("当前回答为中等置信度，建议在正式使用前复核引用页。")
    elif confidence == "low":
        notes.append("当前回答置信度较低，更适合作为线索而非直接决策依据。")

    validation_flags = result.get("validation_flags") or []
    notes.extend(_humanize_validation_flag(flag) for flag in validation_flags)

    if not notes:
        notes.append("当前未发现显著校验风险，检索、citation 覆盖和答案检查整体正常。")
    return notes[:4]


def _infer_question_kind(question_text: str) -> str | None:
    normalized = " ".join((question_text or "").split()).strip()
    if not normalized:
        return None

    lowered = normalized.lower()
    boolean_keywords = (
        "是否",
        "有没有",
        "有无",
        "是不是",
        "是否提到",
        "did ",
        "does ",
        "do ",
        "is ",
        "are ",
        "was ",
        "were ",
        "has ",
        "have ",
        "had ",
        "mention",
    )
    if any(keyword in lowered for keyword in boolean_keywords):
        return "boolean"

    name_keywords = (
        "是谁",
        "哪位",
        "姓名",
        "名字",
        "名称",
        "法定代表人",
        "董事长",
        "总经理",
        "ceo",
        "cfo",
        "president",
        "legal representative",
        "who is",
        "who was",
        "name of",
    )
    if any(keyword in lowered for keyword in name_keywords):
        return "name"

    number_keywords = (
        "多少",
        "几",
        "金额",
        "收入",
        "利润",
        "资产",
        "负债",
        "营收",
        "市值",
        "元",
        "万元",
        "亿元",
        "百分比",
        "比率",
        "比例",
        "毛利率",
        "净利率",
        "revenue",
        "income",
        "asset",
        "assets",
        "amount",
        "value",
        "margin",
        "ratio",
        "percentage",
        "how much",
        "what was",
    )
    if any(keyword in lowered for keyword in number_keywords):
        return "number"

    return None


def _run_query_with_kind_retry(processor, question: str, selected_kind: str):
    try:
        return processor.process_question(question, selected_kind), selected_kind, None
    except Exception as err:
        inferred_kind = _infer_question_kind(question)
        should_retry = (
            inferred_kind in QUESTION_KIND_OPTIONS
            and inferred_kind != selected_kind
            and "Structured response parsing failed" in str(err)
        )
        if not should_retry:
            raise

        retry_result = processor.process_question(question, inferred_kind)
        retry_note = (
            f"系统检测到这个问题更像 `{_question_kind_label(inferred_kind)}`，已在首次 schema 失败后自动切换并重试。"
        )
        return retry_result, inferred_kind, retry_note


def _apply_corpus_view_hint(question_text: str, corpus_view: str) -> str:
    normalized = question_text.strip()
    if not normalized or corpus_view == "混合语料":
        return normalized
    if corpus_view == "中文年报" and all(keyword not in normalized for keyword in ("年报", "年度报告", "annual report")):
        return f"{normalized}（请基于中文年报回答）"
    if corpus_view == "中文研报" and all(keyword not in normalized for keyword in ("研报", "券商", "深度报告")):
        return f"{normalized}（请基于中文券商研报回答）"
    return normalized


def _ensure_upload_workspace() -> Path:
    pdf_dir = UPLOAD_WORKSPACE_DIR / "pdf_reports"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    return pdf_dir


def _save_uploaded_files(uploaded_files) -> Tuple[int, Path | None]:
    if not uploaded_files:
        return 0, None

    target_dir = _ensure_upload_workspace()
    saved_count = 0
    for uploaded_file in uploaded_files:
        target_file = target_dir / uploaded_file.name
        target_file.write_bytes(uploaded_file.getbuffer())
        saved_count += 1
    return saved_count, target_dir


def _initialize_state(question_samples: List[Dict[str, str]]) -> None:
    if "question_text" not in st.session_state:
        st.session_state.question_text = question_samples[0]["text"] if question_samples else DEFAULT_QUESTION
    if "question_kind" not in st.session_state:
        default_kind = question_samples[0]["kind"] if question_samples else "name"
        st.session_state.question_kind = default_kind if default_kind in QUESTION_KIND_OPTIONS else "name"
    if "corpus_view" not in st.session_state:
        st.session_state.corpus_view = CORPUS_VIEW_OPTIONS[0]
    if "workspace_mode" not in st.session_state:
        st.session_state.workspace_mode = WORKSPACE_OPTIONS[0]
    if "demo_result" not in st.session_state:
        st.session_state.demo_result = None
    if "demo_error" not in st.session_state:
        st.session_state.demo_error = None
    if "last_run_meta" not in st.session_state:
        st.session_state.last_run_meta = None
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "upload_notice" not in st.session_state:
        st.session_state.upload_notice = None
    if "processor_cache" not in st.session_state:
        st.session_state.processor_cache = {}


def _load_selected_sample(question_samples: List[Dict[str, str]]) -> None:
    sample_index = int(st.session_state.get("sample_index", 0))
    if sample_index < 0 or sample_index >= len(question_samples):
        return
    sample = question_samples[sample_index]
    st.session_state.question_text = sample["text"]
    st.session_state.question_kind = sample["kind"]


def _render_sidebar_brand() -> None:
    st.markdown(
        """
        <div class="sidebar-brand">
          <div class="overline">金融智能研究平台</div>
          <h3 style="margin:0;">FinaRAG</h3>
          <p class="workspace-lead" style="margin:0.45rem 0 0 0;">
            面向投研与管理决策场景的智能检索、问答与证据追踪工作台。
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar_history() -> None:
    history = st.session_state.query_history[:MAX_HISTORY_ITEMS]
    if not history:
        st.caption("当前会话还没有最近研究记录。")
        return

    for item in history:
        st.markdown(
            f"""
            <div class="history-item">
              <div class="history-question">{escape(_truncate(item['question'], 84))}</div>
              <div class="history-meta">
                {escape(item['timestamp'])} | {escape(item['dataset'])} | {escape(item['confidence'])}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_header(runtime_config, dataset_dir: Path, asset_ready: bool, provider: Dict[str, str], report_count: int, indexed_reports: int) -> None:
    asset_badge_class = "status-pill" if asset_ready else "status-pill warn"
    endpoint_badge_class = "status-pill" if provider["configured"] == "yes" else "status-pill danger"
    dataset_label = _dataset_label(dataset_dir)

    st.markdown(
        f"""
        <div class="hero-grid">
          <div class="brand-hero">
            <div class="overline">金融 RAG 研究终端</div>
            <h1 class="hero-title">FinaRAG</h1>
            <p class="hero-subtitle">
              面向金融场景的智能检索与分析助手。聚焦问答、检索依据、来源追踪与风险提示，
              更像正式的金融分析产品，而不是普通的聊天 Demo。
            </p>
            {_render_chip_row(
                [
                    "研究工作台",
                    f"模型：{runtime_config.answering_model}",
                    f"数据集：{dataset_label}",
                    f"Top-K：{runtime_config.top_n_retrieval}",
                ]
            )}
          </div>
          <div class="status-panel">
            <div class="panel-heading">平台状态</div>
            <div style="display:flex; flex-direction:column; gap:0.8rem;">
              <div>
                <div class="tiny-note">接口服务</div>
                <div style="margin-top:0.3rem;">
                  <span class="{endpoint_badge_class}">{'已就绪' if provider['configured'] == 'yes' else '待配置'}</span>
                </div>
                <div class="tiny-note" style="margin-top:0.45rem;">{escape(provider['host'])}</div>
              </div>
              <div>
                <div class="tiny-note">资产就绪度</div>
                <div style="margin-top:0.3rem;">
                  <span class="{asset_badge_class}">{'已建索引' if asset_ready else '缺少资产'}</span>
                </div>
              </div>
              <div class="tiny-note">报告数：{report_count} | 已索引文档：{indexed_reports}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_overview_metrics(report_count: int, company_count: int, indexed_reports: int, sample_count: int, provider_ready: bool) -> None:
    metric_cols = st.columns(5)
    metric_cols[0].metric("文档数", report_count)
    metric_cols[1].metric("公司数", company_count)
    metric_cols[2].metric("索引块数", indexed_reports)
    metric_cols[3].metric("样例问题", sample_count)
    metric_cols[4].metric("接口状态", "已就绪" if provider_ready else "待配置")


def _render_company_snapshot(manifest_path: Path) -> None:
    companies_preview = _load_company_snapshot(str(manifest_path))
    if not companies_preview:
        st.caption("当前数据集暂无清单预览信息。")
        return

    for company in companies_preview:
        label = company["company_name"] or "未知公司"
        details = " | ".join(
            part
            for part in [company["currency"], company["major_industry"], company["doc_source_type"], company["broker_name"]]
            if part
        )
        st.markdown(f"**{label}**")
        if details:
            st.caption(details)


def _render_query_panel(question_samples: List[Dict[str, str]], runtime_config, dataset_dir: Path, answer_temperature: float) -> bool:
    with st.container(border=True):
        st.markdown("### 研究工作台")
        st.markdown(
            "<p class='workspace-lead'>发起金融问题查询，系统会执行检索、生成、来源回填与校验，并在下方展示关键结论与证据卡片。</p>",
            unsafe_allow_html=True,
        )

        if question_samples:
            st.markdown("<div class='panel-heading'>快速样例</div>", unsafe_allow_html=True)
            sample_cols = st.columns(min(3, len(question_samples[:3])))
            for idx, sample in enumerate(question_samples[:3]):
                with sample_cols[idx]:
                    if st.button(
                        f"{_question_kind_label(sample['kind'])} | {_truncate(sample['text'], 44)}",
                        key=f"quick_sample_{idx}",
                        use_container_width=True,
                    ):
                        st.session_state.question_text = sample["text"]
                        st.session_state.question_kind = sample["kind"]

        with st.form("fina_research_form"):
            left_col, right_col = st.columns([1.75, 1.0], gap="large")
            with left_col:
                st.markdown("<div class='panel-heading'>问题输入</div>", unsafe_allow_html=True)
                st.text_area(
                    "输入金融问题",
                    key="question_text",
                    label_visibility="collapsed",
                    placeholder="例如：贵州茅台2024年年报中的法定代表人是谁？或者：中芯国际2024年的营业收入是多少？",
                )
            with right_col:
                st.markdown("<div class='panel-heading'>问题控制</div>", unsafe_allow_html=True)
                st.radio(
                    "问题类型",
                    QUESTION_KIND_OPTIONS,
                    key="question_kind",
                    horizontal=True,
                    format_func=_question_kind_label,
                )
                st.caption("`boolean` 适用于是非判断，`number` 适用于数字指标，`name` 适用于公司、人名或事项名称。")
                st.markdown(
                    _render_chip_row(
                        [
                            _dataset_label(dataset_dir),
                            runtime_config.answering_model,
                            f"Top-K {runtime_config.top_n_retrieval}",
                            f"温度 {answer_temperature:.2f}",
                            f"模式 {st.session_state.corpus_view}",
                        ]
                    ),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    """
                    <div class="callout-shell" style="margin-top:0.9rem;">
                      <div class="panel-heading">RAG 流程</div>
                      <div class="tiny-note">
                        提问 -> 检索 -> 生成 -> 校验<br>
                        如果问题类型选错，系统会在 schema 解析失败时按问题文本自动纠偏并重试一次。
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            submitted = st.form_submit_button("开始金融分析", use_container_width=True)
    return submitted


def _render_answer_workspace(result: Dict[str, Any], run_meta: Dict[str, Any]) -> None:
    route_info = result.get("route_info") or {}
    relevant_pages = result.get("relevant_pages") or []
    citations = result.get("citations") or []
    confidence = _humanize_confidence(result.get("confidence"))
    key_conclusions = _build_key_conclusions(result)
    risk_notes = _build_risk_notes(result)

    metric_cols = st.columns(4)
    metric_cols[0].metric("置信度", confidence)
    metric_cols[1].metric("路由对象", route_info.get("selected_company") or "自动推断")
    metric_cols[2].metric("相关页码", len(relevant_pages))
    metric_cols[3].metric("来源数", len(citations))

    left_col, right_col = st.columns([1.55, 1.0], gap="large")
    with left_col:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="answer-shell">
                  <div class="answer-label">最终答案</div>
                  <div class="answer-value">{escape(str(result.get('final_answer') or '暂无'))}</div>
                  <div class="answer-caption">{escape(result.get('reasoning_summary') or '暂无推理摘要。')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            st.markdown("#### 关键结论")
            if key_conclusions:
                st.markdown(
                    "<ul class='bullet-list'>" + "".join(f"<li>{escape(item)}</li>" for item in key_conclusions) + "</ul>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("暂未生成关键结论。")

        with st.container(border=True):
            st.markdown("#### 检索证据预览")
            _render_sources(result, limit=4)

    with right_col:
        with st.container(border=True):
            st.markdown("#### 分析画像")
            st.markdown(
                _render_chip_row(
                    [
                        f"路由方式：{_humanize_route_mode(route_info.get('route_mode'))}",
                        f"工作区：{run_meta.get('workspace_mode', '研究工作台')}",
                        f"问题类型：{_question_kind_label(run_meta.get('question_kind', 'name'))}",
                        f"耗时：{run_meta.get('latency_seconds', 0):.2f}s",
                    ],
                    tone="gold",
                ),
                unsafe_allow_html=True,
            )
            if result.get("search_queries"):
                st.markdown("**扩展检索式**")
                st.markdown(_render_chip_row(result["search_queries"]), unsafe_allow_html=True)
            else:
                st.caption("未记录检索改写结果。")

        with st.container(border=True):
            st.markdown("#### 风险提示")
            st.markdown(
                "<ul class='bullet-list'>" + "".join(f"<li>{escape(item)}</li>" for item in risk_notes) + "</ul>",
                unsafe_allow_html=True,
            )
            if result.get("validation_flags"):
                st.caption(
                    "校验标记："
                    + "，".join(_humanize_validation_flag(flag) for flag in result["validation_flags"])
                )

        with st.container(border=True):
            st.markdown("#### 引用足迹")
            refs = result.get("references") or []
            if refs:
                for ref in refs[:6]:
                    st.markdown(
                        f"- `{ref.get('pdf_sha1')}` | 第 `{ref.get('page_index')}` 页"
                    )
            else:
                st.caption("未返回引用记录。")

            st.download_button(
                "下载结果 JSON",
                data=json.dumps(result, ensure_ascii=False, indent=2),
                file_name="fina_rag_result.json",
                mime="application/json",
                use_container_width=True,
            )


def _render_source_card(item: Dict[str, Any], index: int) -> None:
    title = item.get("company_name") or item.get("source") or item.get("pdf_sha1") or "来源"
    page = item.get("page") or item.get("page_index") or "未知"
    score = item.get("score")
    meta_parts = [f"第 {page} 页"]
    if score not in (None, ""):
        try:
            meta_parts.append(f"分数 {float(score):.3f}")
        except (TypeError, ValueError):
            meta_parts.append(f"分数 {score}")

    tags = [
        item.get("doc_source_type"),
        item.get("chunk_type"),
        item.get("currency"),
        item.get("report_year"),
        item.get("security_code"),
        *(item.get("retrieval_sources") or []),
    ]
    snippet = item.get("evidence_snippet") or item.get("text_preview") or "暂无片段摘要。"

    with st.container(border=True):
        st.markdown(
            f"""
            <div class="source-shell">
              <div class="answer-label">来源 {index:02d}</div>
              <div class="source-title">{escape(str(title))}</div>
              <div class="source-meta">{escape(" | ".join(meta_parts))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        chips = _render_chip_row(tags)
        if chips:
            st.markdown(chips, unsafe_allow_html=True)
        st.markdown(f"<div class='source-snippet'>{escape(str(snippet))}</div>", unsafe_allow_html=True)


def _render_sources(result: Dict[str, Any], limit: int | None = None) -> None:
    sources = result.get("citations") or result.get("retrieval_results") or []
    if limit is not None:
        sources = sources[:limit]

    if not sources:
        st.caption("当前结果暂无来源卡片。")
        return

    cols = st.columns(2, gap="large")
    for idx, item in enumerate(sources, start=1):
        with cols[(idx - 1) % 2]:
            _render_source_card(item, idx)


def _render_evidence_center(result: Dict[str, Any]) -> None:
    st.markdown("### 检索依据中心")
    st.markdown(
        "<p class='workspace-lead'>在这里查看检索依据、来源卡片、路由策略与 Query Plan，确保回答可追溯、可验证。</p>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("#### 来源卡片")
        _render_sources(result)

    route_info = result.get("route_info") or {}
    query_plan = result.get("query_plan") or {}
    retrieval_results = result.get("retrieval_results") or []

    left_col, right_col = st.columns([1.0, 1.0], gap="large")
    with left_col:
        with st.container(border=True):
            st.markdown("#### 路由策略")
            st.markdown(
                _render_chip_row(
                    [
                        _humanize_route_mode(route_info.get("route_mode")),
                        route_info.get("selected_company") or "自动推断",
                    ],
                    tone="gold",
                ),
                unsafe_allow_html=True,
            )
            candidate_scores = route_info.get("candidate_scores") or []
            if candidate_scores:
                st.dataframe(candidate_scores, use_container_width=True, hide_index=True)
            else:
                st.caption("暂无候选路由表。")

    with right_col:
        with st.container(border=True):
            st.markdown("#### 查询计划 / Query Plan")
            search_queries = result.get("search_queries") or []
            if search_queries:
                st.markdown(_render_chip_row(search_queries), unsafe_allow_html=True)
            st.json(query_plan)

    if retrieval_results:
        with st.container(border=True):
            st.markdown("#### 检索概览")
            summary_rows = []
            for index, item in enumerate(retrieval_results, start=1):
                summary_rows.append(
                    {
                        "排名": index,
                        "页码": item.get("page"),
                        "公司": item.get("company_name"),
                        "范围": item.get("result_scope"),
                        "切块类型": item.get("chunk_type"),
                        "分数": item.get("score"),
                        "来源": ", ".join(item.get("retrieval_sources") or []),
                    }
                )
            st.dataframe(summary_rows, use_container_width=True, hide_index=True)


def _render_monitor_panel(run_meta: Dict[str, Any], result: Dict[str, Any] | None, dataset_dir: Path, config_path: Path, paths: Dict[str, Path], provider: Dict[str, str], asset_messages: List[str]) -> None:
    st.markdown("### 系统监控")
    st.markdown(
        "<p class='workspace-lead'>查看当前会话配置、数据资产、上传工作区与原始调试数据，方便排查接口与索引问题。</p>",
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.0, 1.0], gap="large")
    with left_col:
        with st.container(border=True):
            st.markdown("#### 运行环境")
            st.markdown(
                _render_chip_row(
                    [
                        f"数据集 {_dataset_label(dataset_dir)}",
                        f"配置 {config_path.name}",
                        f"接口 {provider['host']}",
                    ],
                    tone="gold",
                ),
                unsafe_allow_html=True,
            )
            st.code(
                "\n".join(
                    [
                        f"数据集目录: {dataset_dir}",
                        f"配置文件: {config_path}",
                        f"文档清单: {paths['manifest']}",
                        f"文档目录: {paths['documents']}",
                        f"向量索引: {paths['vector']}",
                        f"BM25 索引: {paths['bm25']}",
                        f"Sparse 索引: {paths['sparse']}",
                    ]
                ),
                language="text",
            )

        with st.container(border=True):
            st.markdown("#### 资产诊断")
            if asset_messages:
                for message in asset_messages:
                    st.markdown(f"- {message}")
            else:
                st.success("当前配置所需的检索资产均已就绪。")

        with st.container(border=True):
            st.markdown("#### 上传工作区")
            st.code(str(_ensure_upload_workspace()), language="text")
            st.caption("上传的 PDF 会暂存于此，仍需解析并建立索引后才能查询。")

    with right_col:
        with st.container(border=True):
            st.markdown("#### 会话摘要")
            if run_meta:
                st.json(run_meta)
            else:
                st.caption("暂无运行元数据。")

        with st.container(border=True):
            st.markdown("#### 最近研究记录")
            _render_sidebar_history()

        if result:
            with st.container(border=True):
                st.markdown("#### 原始结果 JSON")
                st.json(result)


def _render_empty_state(dataset_dir: Path, runtime_config, report_count: int, indexed_reports: int, sample_count: int) -> None:
    st.markdown("### 工作区概览")
    st.markdown(
        "<p class='workspace-lead'>准备好之后，直接在上方输入金融问题。这里会展示检索、生成、校验与来源追踪的完整工作流。</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="workflow-row">
          <div class="workflow-step">
            <div class="step-title">01 提问</div>
            <div class="step-body">定义问题类型、语料模式与数据源，发起面向金融场景的精准提问。</div>
          </div>
          <div class="workflow-step">
            <div class="step-title">02 检索</div>
            <div class="step-body">组合向量、BM25、稀疏检索与文档路由，返回最相关的金融证据块。</div>
          </div>
          <div class="workflow-step">
            <div class="step-title">03 生成</div>
            <div class="step-body">结合检索证据与模型推理给出最终答案、关键结论与置信度。</div>
          </div>
          <div class="workflow-step">
            <div class="step-title">04 校验</div>
            <div class="step-body">补充引用来源、相关页码、表格 grounding 与校验标记，提升可验证性。</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    preview_cols = st.columns([1.2, 1.0], gap="large")
    with preview_cols[0]:
        with st.container(border=True):
            st.markdown("#### 当前工作区快照")
            st.markdown(
                _render_chip_row(
                    [
                        f"数据集 {_dataset_label(dataset_dir)}",
                        f"模型 {runtime_config.answering_model}",
                        f"报告 {report_count}",
                        f"已索引 {indexed_reports}",
                        f"样例 {sample_count}",
                    ]
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="placeholder-card">
                  <h4>专业级 RAG 工作台</h4>
                  <p class="tiny-note">
                    当前重构后的界面更强调金融工作流的清晰度：
                    问题输入、证据追踪、路由可见性，以及带风险意识的答案呈现。
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with preview_cols[1]:
        with st.container(border=True):
            st.markdown("#### 推荐用法")
            st.markdown(
                "<ul class='bullet-list'>"
                "<li>企业年报问答与核心指标核对</li>"
                "<li>券商研报要点提取与出处追溯</li>"
                "<li>多公司对比问答与候选文档路由</li>"
                "<li>数字类问题的表格 grounding 与验证</li>"
                "</ul>",
                unsafe_allow_html=True,
            )


_inject_styles()

if not AVAILABLE_CONFIGS or DEFAULT_CONFIG is None:
    st.error("在 `config/` 目录下未找到 YAML 配置文件。")
    st.stop()

dataset_options = _discover_dataset_options()
dataset_lookup = {label: path for label, path in dataset_options}
default_dataset_label = next(
    (label for label, path in dataset_options if path == DEFAULT_DATASET_DIR),
    dataset_options[0][0] if dataset_options else str(DEFAULT_DATASET_DIR),
)

config_names = [path.name for path in AVAILABLE_CONFIGS]
default_config_name = DEFAULT_CONFIG.name
default_config_index = config_names.index(default_config_name)

with st.sidebar:
    _render_sidebar_brand()
    st.markdown("<div class='sidebar-heading'>工作区</div>", unsafe_allow_html=True)
    st.radio("工作区", WORKSPACE_OPTIONS, key="workspace_mode", label_visibility="collapsed")

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-heading'>模型与检索</div>", unsafe_allow_html=True)
    selected_config_name = st.selectbox("配置方案", config_names, index=default_config_index)

config_path = CONFIG_DIR / selected_config_name
base_run_config = load_run_config(config_path)

with st.sidebar:
    model_name = st.text_input("回答模型", value=base_run_config.answering_model)
    top_k = st.slider("Top-K 检索", min_value=2, max_value=12, value=int(base_run_config.top_n_retrieval), step=1)
    answer_temperature = st.slider("生成温度", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-heading'>数据源</div>", unsafe_allow_html=True)
    selected_dataset_label = st.selectbox(
        "数据集",
        options=list(dataset_lookup.keys()) if dataset_lookup else [str(DEFAULT_DATASET_DIR)],
        index=(list(dataset_lookup.keys()).index(default_dataset_label) if dataset_lookup else 0),
    )
    selected_dataset_dir = dataset_lookup.get(selected_dataset_label, DEFAULT_DATASET_DIR)
    dataset_dir = Path(
        st.text_input("数据集路径", value=str(selected_dataset_dir))
    ).expanduser()
    st.selectbox("语料聚焦", CORPUS_VIEW_OPTIONS, key="corpus_view")

runtime_config = _build_runtime_config(config_path, model_name, top_k)
provider = _provider_status(runtime_config)
paths = _resolve_data_paths(dataset_dir, runtime_config)
asset_messages = _required_asset_messages(paths, runtime_config)
question_samples = _load_question_samples(str(dataset_dir))
_initialize_state(question_samples)

with st.sidebar:
    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-heading'>文档接入</div>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "上传 PDF",
        type=["pdf"],
        accept_multiple_files=True,
        help="上传文件会暂存到仓库内，方便后续解析和建立索引。",
    )
    if st.button("保存上传的 PDF", use_container_width=True):
        saved_count, saved_path = _save_uploaded_files(uploaded_files)
        if saved_count:
            st.session_state.upload_notice = f"已将 {saved_count} 个 PDF 保存到 {saved_path}。"
            st.rerun()
        else:
            st.session_state.upload_notice = "未选择 PDF 文件。"
            st.rerun()
    if st.session_state.upload_notice:
        st.success(st.session_state.upload_notice)

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-heading'>样例问题库</div>", unsafe_allow_html=True)
    if question_samples:
        st.selectbox(
            "样例问题",
            options=list(range(len(question_samples))),
            key="sample_index",
            format_func=lambda idx: f"{_question_kind_label(question_samples[idx]['kind'])} | {_truncate(question_samples[idx]['text'], 58)}",
        )
        if st.button("载入样例问题", use_container_width=True):
            _load_selected_sample(question_samples)
    else:
        st.caption("当前数据集未找到样例问题文件。")

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-heading'>系统状态</div>", unsafe_allow_html=True)
    if asset_messages:
        st.warning("当前数据集尚未为当前检索栈完成索引。")
    else:
        st.success("检索资产已就绪。")
    st.caption(f"接口：{provider['host']}")
    st.caption(f"模型：{provider['model']}")

    with st.expander("公司覆盖范围", expanded=False):
        _render_company_snapshot(paths["manifest"])

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-heading'>最近研究</div>", unsafe_allow_html=True)
    _render_sidebar_history()

report_count = _count_matching(paths["pdf_reports"], "*.pdf")
company_count = _count_manifest_rows(paths["manifest"])
indexed_reports = _count_matching(paths["documents"], "*.json")
sample_count = len(question_samples)

asset_ready = not asset_messages
_render_header(runtime_config, dataset_dir, asset_ready, provider, report_count, indexed_reports)
_render_overview_metrics(report_count, company_count, indexed_reports, sample_count, provider["configured"] == "yes")

submitted = _render_query_panel(question_samples, runtime_config, dataset_dir, answer_temperature)

if submitted:
    current_question = st.session_state.question_text.strip()
    if not current_question:
        st.session_state.demo_result = None
        st.session_state.demo_error = {"message": "问题不能为空。", "traceback": ""}
    elif asset_messages:
        st.session_state.demo_result = None
        st.session_state.demo_error = {
            "message": "当前数据集尚未针对当前配置完成索引。",
            "traceback": "\n".join(asset_messages),
        }
    else:
        start_time = time.perf_counter()
        with st.status("正在执行金融研究流程...", expanded=True) as status:
            try:
                status.write("1. 正在校验数据集、模型与检索栈")
                processor, processor_cache_hit, processor_init_seconds = _get_or_create_processor(
                    dataset_dir,
                    runtime_config,
                    answer_temperature,
                )
                if processor_cache_hit:
                    status.write("2. 命中常驻缓存，复用已加载的检索与路由组件")
                else:
                    status.write(
                        f"2. 首次加载当前配置的检索栈，用时 {processor_init_seconds:.2f}s"
                    )
                effective_question = _apply_corpus_view_hint(current_question, st.session_state.corpus_view)

                status.write("3. 正在检索证据并生成答案")
                result, effective_kind, retry_note = _run_query_with_kind_retry(
                    processor,
                    effective_question,
                    st.session_state.question_kind,
                )

                elapsed = time.perf_counter() - start_time
                status.write("4. 正在补充引用、置信度与校验元数据")
                status.update(label="金融分析已完成", state="complete")

                st.session_state.demo_result = result
                st.session_state.demo_error = None
                st.session_state.last_run_meta = {
                    "dataset_dir": str(dataset_dir),
                    "config_path": str(config_path),
                    "workspace_mode": st.session_state.workspace_mode,
                    "question_kind": effective_kind,
                    "selected_question_kind": st.session_state.question_kind,
                    "question_text": current_question,
                    "effective_question": effective_question,
                    "corpus_view": st.session_state.corpus_view,
                    "latency_seconds": elapsed,
                    "retry_note": retry_note,
                    "top_k": runtime_config.top_n_retrieval,
                    "model": runtime_config.answering_model,
                    "temperature": answer_temperature,
                    "processor_cache_hit": processor_cache_hit,
                    "processor_init_seconds": processor_init_seconds,
                    "processor_cache_size": len(st.session_state.processor_cache),
                }
                st.session_state.query_history = [
                    {
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "question": current_question,
                        "dataset": _dataset_label(dataset_dir),
                        "confidence": _humanize_confidence(result.get("confidence")),
                    }
                ] + st.session_state.query_history[: MAX_HISTORY_ITEMS - 1]
            except Exception as err:
                status.update(label="金融分析失败", state="error")
                st.session_state.demo_result = None
                st.session_state.demo_error = {
                    "message": str(err),
                    "traceback": traceback.format_exc(),
                }

run_meta = st.session_state.last_run_meta
result = st.session_state.demo_result
error = st.session_state.demo_error

if run_meta:
    cache_note = "缓存命中" if run_meta.get("processor_cache_hit") else "冷启动"
    meta_line = (
        f"最近一次运行 | 配置 `{Path(run_meta['config_path']).name}` | "
        f"类型 `{_question_kind_label(run_meta['question_kind'])}` | "
        f"语料 `{run_meta['corpus_view']}` | "
        f"模式 `{cache_note}` | "
        f"耗时 `{run_meta['latency_seconds']:.2f}s`"
    )
    st.caption(meta_line)
    if run_meta.get("processor_cache_hit"):
        st.caption(
            f"本次复用了常驻 processor 缓存，当前会话内缓存配置数：{run_meta.get('processor_cache_size', 1)}。"
        )
    else:
        st.caption(
            f"本次为当前配置首次冷启动，processor 初始化耗时约 {run_meta.get('processor_init_seconds', 0.0):.2f}s。"
        )
    if run_meta.get("retry_note"):
        st.info(run_meta["retry_note"])
    if run_meta.get("effective_question") and run_meta["effective_question"] != run_meta["question_text"]:
        st.caption(f"实际送入系统的问题：{run_meta['effective_question']}")

if error:
    with st.container(border=True):
        st.error(error["message"])
        st.caption("工作区已安全降级处理，你可以展开下面的异常栈继续排查后端问题。")
        if error.get("traceback"):
            with st.expander("异常栈", expanded=False):
                st.code(error["traceback"], language="text")

if result:
    workspace_mode = st.session_state.workspace_mode
    if workspace_mode == "研究工作台":
        _render_answer_workspace(result, run_meta or {})
    elif workspace_mode == "检索依据":
        _render_evidence_center(result)
    else:
        _render_monitor_panel(run_meta or {}, result, dataset_dir, config_path, paths, provider, asset_messages)
else:
    if st.session_state.workspace_mode == "系统监控":
        _render_monitor_panel(run_meta or {}, None, dataset_dir, config_path, paths, provider, asset_messages)
    else:
        _render_empty_state(dataset_dir, runtime_config, report_count, indexed_reports, sample_count)
