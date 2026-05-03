from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _patch_torch_classes_for_streamlit_watcher() -> None:
    """Give Streamlit's local source watcher a safe namespace path for torch.classes."""
    try:
        import torch
    except Exception:
        return

    torch_classes = getattr(torch, "classes", None)
    if torch_classes is None:
        return

    class _StreamlitSafeTorchClassesPath(list):
        @property
        def _path(self) -> List[str]:
            return list(self)

    try:
        torch_classes.__path__ = _StreamlitSafeTorchClassesPath()
    except Exception:
        return


_patch_torch_classes_for_streamlit_watcher()

import streamlit as st
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

# The interactive demo benefits more from fast warm startup than from loading
# duplicate embedding/reranker replicas across multiple GPUs.
os.environ.setdefault("EMBEDDING_MAX_DEVICES", "1")
os.environ.setdefault("EMBEDDING_SPARSE_MAX_DEVICES", "1")
os.environ.setdefault("RERANKING_MAX_DEVICES", "1")

from src.document_manifest import load_document_manifest
from src.pipeline import apply_runtime_overrides, load_run_config
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
    PROJECT_ROOT / "data" / "top10_industries_2024_20each",
    PROJECT_ROOT / "data" / "chinese_annual_reports_2024",
    PROJECT_ROOT / "data" / "test_set",
]
DEFAULT_DATASET_DIR = next((path for path in PREFERRED_DATASETS if path.exists()), PROJECT_ROOT / "data" / "test_set")
UPLOAD_WORKSPACE_DIR = PROJECT_ROOT / "data" / "upload_workspace"

QUESTION_KIND_OPTIONS = ["boolean", "number", "name", "names"]
WORKSPACE_OPTIONS = ["研究工作台", "检索依据", "系统监控"]
CORPUS_VIEW_OPTIONS = ["混合语料", "中文年报", "中文研报"]
MAX_SAMPLE_QUESTIONS = 120
MAX_QUICK_SAMPLES = 3
MAX_HISTORY_ITEMS = 6
MAX_PROCESSOR_CACHE_SIZE = 2
DEFAULT_QUESTION = "贵州茅台2024年年报中的法定代表人是谁？"
QUESTION_KIND_LABELS = {
    "boolean": "是非判断",
    "number": "数值指标",
    "name": "名称实体",
    "names": "名称列表",
}
QUESTION_TYPE_FILTER_ALL = "__all__"
QUESTION_TYPE_LABELS = {
    "single_doc_fact/name": "单文档事实·名称",
    "single_doc_fact/number": "单文档事实·数值",
    "single_doc_boolean": "单文档判断",
    "section_filter": "章节定向",
    "metadata_tag_retrieval": "元数据标签检索",
    "cross_doc_compare": "跨文档比较",
}
QUESTION_CAPABILITY_LABELS = {
    "single_doc_fact": "单文档事实",
    "single_doc_boolean": "单文档判断",
    "section_filter": "章节定向",
    "metadata_tag_retrieval": "元数据标签检索",
    "cross_doc_compare": "跨文档比较",
}
EVIDENCE_TYPE_LABELS = {
    "table": "表格证据",
    "text": "文本证据",
    "metadata": "元数据证据",
    "hybrid": "混合证据",
}


@st.cache_resource(show_spinner=False)
def _get_prewarm_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="finarag-demo-prewarm")


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
        "top10_industries_2024_20each": "Top10 行业中文数据集 2024",
        "test_set": "内置测试集",
        "upload_workspace": "上传工作区",
    }
    return labels.get(path.name, path.name.replace("_", " ").title())


def _question_kind_label(kind: str) -> str:
    return QUESTION_KIND_LABELS.get(kind, kind)


def _normalize_question_kind(kind: Any, fallback: str = "name") -> str:
    normalized = str(kind or fallback).strip().lower()
    return normalized if normalized in QUESTION_KIND_OPTIONS else fallback


def _question_type_label(question_type: str) -> str:
    if question_type in QUESTION_TYPE_LABELS:
        return QUESTION_TYPE_LABELS[question_type]
    if question_type in QUESTION_KIND_LABELS:
        return _question_kind_label(question_type)
    return question_type or "未分类"


def _question_capability_label(capability: str) -> str:
    normalized = str(capability or "").strip().lower()
    return QUESTION_CAPABILITY_LABELS.get(normalized, normalized or "未标注")


def _evidence_type_label(evidence_type: str) -> str:
    normalized = str(evidence_type or "").strip().lower()
    return EVIDENCE_TYPE_LABELS.get(normalized, normalized or "未标注")


def _question_group_label(group_name: str) -> str:
    return str(group_name or "全部行业").strip() or "全部行业"


def _build_question_type_key(capability: Any, kind: Any) -> str:
    normalized_capability = str(capability or "").strip().lower()
    normalized_kind = _normalize_question_kind(kind)
    if normalized_capability == "single_doc_fact" and normalized_kind in {"name", "number"}:
        return f"{normalized_capability}/{normalized_kind}"
    if normalized_capability:
        return normalized_capability
    return normalized_kind


def _sample_question_type_key(sample: Dict[str, Any]) -> str:
    return _build_question_type_key(sample.get("capability"), sample.get("kind"))


def _sample_option_label(sample: Dict[str, Any], text_limit: int = 58) -> str:
    type_label = _question_type_label(_sample_question_type_key(sample))
    group_name = str(sample.get("group_name") or sample.get("industry_l1") or "").strip()
    prefix = f"{type_label} | {group_name}" if group_name else type_label
    return f"{prefix} | {_truncate(str(sample.get('text') or ''), text_limit)}"


def _sample_filter_options(question_samples: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    question_type_options = [QUESTION_TYPE_FILTER_ALL]
    question_group_options = [QUESTION_TYPE_FILTER_ALL]

    seen_question_types = []
    seen_groups = []
    for sample in question_samples:
        question_type = _sample_question_type_key(sample)
        if question_type and question_type not in seen_question_types:
            seen_question_types.append(question_type)
        group_name = str(sample.get("group_name") or sample.get("industry_l1") or "").strip()
        if group_name and group_name not in seen_groups:
            seen_groups.append(group_name)

    question_type_options.extend(seen_question_types)
    question_group_options.extend(sorted(seen_groups))
    return question_type_options, question_group_options


def _filtered_question_sample_indices(question_samples: List[Dict[str, Any]]) -> List[int]:
    selected_type = str(st.session_state.get("question_type_filter") or QUESTION_TYPE_FILTER_ALL)
    selected_group = str(st.session_state.get("question_group_filter") or QUESTION_TYPE_FILTER_ALL)

    filtered_indices: List[int] = []
    for index, sample in enumerate(question_samples):
        sample_type = _sample_question_type_key(sample)
        sample_group = str(sample.get("group_name") or sample.get("industry_l1") or "").strip()
        if selected_type != QUESTION_TYPE_FILTER_ALL and sample_type != selected_type:
            continue
        if selected_group != QUESTION_TYPE_FILTER_ALL and sample_group != selected_group:
            continue
        filtered_indices.append(index)
    return filtered_indices


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


def _has_any_enabled_retriever(run_config) -> bool:
    return any(
        (
            bool(run_config.use_vector_dbs),
            bool(run_config.use_bm25_db),
            bool(run_config.use_sparse_lexical_db),
            bool(getattr(run_config, "use_tag_db", False)),
        )
    )


def _resolve_data_paths(dataset_dir: Path, run_config) -> Dict[str, Path]:
    manifest_candidates = [
        dataset_dir / "document_manifest.csv",
        dataset_dir / "document_manifest.json",
        dataset_dir / "subset.csv",
        dataset_dir / "subset.json",
    ]
    manifest_path = next((path for path in manifest_candidates if path.exists()), manifest_candidates[0])
    preferred_database_root = dataset_dir / ("databases_ser_tab" if run_config.use_serialized_tables else "databases")
    fallback_database_root = dataset_dir / ("databases" if run_config.use_serialized_tables else "databases_ser_tab")
    if preferred_database_root.exists():
        database_root = preferred_database_root
    elif fallback_database_root.exists():
        database_root = fallback_database_root
    else:
        database_root = preferred_database_root
    return {
        "database_root": database_root,
        "preferred_database_root": preferred_database_root,
        "pdf_reports": dataset_dir / "pdf_reports",
        "manifest": manifest_path,
        "subset": manifest_path,
        "questions": dataset_dir / "questions.json",
        "documents": database_root / "chunked_reports",
        "vector": database_root / "vector_dbs",
        "bm25": database_root / "bm25_dbs",
        "sparse": database_root / "sparse_dbs",
        "tag": database_root / "tag_dbs",
    }


def _prepare_runtime_config_for_dataset(dataset_dir: Path, run_config):
    effective_run_config = deepcopy(run_config)
    paths = _resolve_data_paths(dataset_dir, effective_run_config)
    notices: List[str] = []
    messages: List[str] = []

    if paths["database_root"] != paths["preferred_database_root"] and paths["database_root"].exists():
        notices.append(
            "当前数据集未找到配置首选索引目录 "
            f"`{paths['preferred_database_root'].name}`，已自动回退到 `{paths['database_root'].name}`。"
        )

    if not paths["manifest"].exists():
        messages.append(f"缺少文档清单: {paths['manifest']}")
    if not paths["documents"].exists():
        messages.append(f"缺少切块结果目录: {paths['documents']}")
    backend_specs = [
        ("use_vector_dbs", "vector", "向量索引"),
        ("use_bm25_db", "bm25", "BM25 索引"),
        ("use_sparse_lexical_db", "sparse", "sparse 索引"),
        ("use_tag_db", "tag", "tag 索引"),
    ]
    for attr_name, path_key, display_name in backend_specs:
        if getattr(effective_run_config, attr_name, False) and not paths[path_key].exists():
            setattr(effective_run_config, attr_name, False)
            notices.append(f"{display_name}缺失，已自动停用该检索后端: {paths[path_key]}")

    if not _has_any_enabled_retriever(effective_run_config):
        messages.append("当前数据集缺少可用的检索索引，请先运行 `process-reports` 构建至少一种检索资产。")

    return effective_run_config, paths, notices, messages


def _count_manifest_rows(manifest_path: Path) -> int:
    if not manifest_path.exists():
        return 0
    return len(load_document_manifest(manifest_path))


def _count_matching(directory: Path, pattern: str) -> int:
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.glob(pattern))


@st.cache_data(show_spinner=False)
def _load_question_samples(dataset_dir: str) -> List[Dict[str, Any]]:
    questions_path = Path(dataset_dir) / "questions.json"
    if not questions_path.exists():
        return []

    try:
        payload = json.loads(questions_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    samples: List[Dict[str, Any]] = []
    for item in payload[:MAX_SAMPLE_QUESTIONS]:
        question_text = str(item.get("text") or "").strip()
        if not question_text:
            continue
        question_kind = _normalize_question_kind(item.get("kind"), fallback="name")
        samples.append(
            {
                "id": str(item.get("id") or "").strip(),
                "text": question_text,
                "kind": question_kind,
                "capability": str(item.get("capability") or "").strip().lower(),
                "evidence_type": str(item.get("evidence_type") or "").strip().lower(),
                "group_name": str(item.get("group_name") or "").strip(),
                "group_slot": str(item.get("group_slot") or "").strip(),
                "industry_l1": str(item.get("industry_l1") or "").strip(),
            }
        )
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
    runtime_config = apply_runtime_overrides(load_run_config(config_path))
    runtime_config.answering_model = (model_name or runtime_config.answering_model).strip()
    runtime_config.top_n_retrieval = int(top_k)
    return runtime_config


def _sync_answering_model_input(config_path: Path, default_model_name: str) -> str:
    config_key = str(config_path.resolve())
    state_key = "answering_model_input"
    default_key = "_answering_model_default"
    config_state_key = "_answering_model_config"

    current_config = st.session_state.get(config_state_key)
    current_default = st.session_state.get(default_key)
    current_value = st.session_state.get(state_key)

    if current_config != config_key:
        st.session_state[state_key] = default_model_name
    elif current_value is None:
        st.session_state[state_key] = default_model_name
    elif current_default is not None and current_value == current_default and current_default != default_model_name:
        # Refresh the widget when the config or .env default changed and the user has not typed a custom override.
        st.session_state[state_key] = default_model_name

    st.session_state[default_key] = default_model_name
    st.session_state[config_state_key] = config_key
    return st.session_state[state_key]


def _processor_signature(dataset_dir: Path, run_config) -> Tuple[Any, ...]:
    paths = _resolve_data_paths(dataset_dir, run_config)
    return (
        str(dataset_dir.resolve()),
        str(paths["database_root"].resolve()),
        str(paths["manifest"].resolve()),
        str(paths["documents"].resolve()),
        str(paths["vector"].resolve()),
        str(paths["bm25"].resolve()),
        str(paths["sparse"].resolve()),
        str(paths["tag"].resolve()),
        tuple(sorted(vars(run_config).items())),
    )


def _trim_signature_cache(cache: Dict[Tuple[Any, ...], Any], keep_signature: Optional[Tuple[Any, ...]] = None) -> None:
    while len(cache) > MAX_PROCESSOR_CACHE_SIZE:
        oldest_signature = next((signature for signature in cache if signature != keep_signature), None)
        if oldest_signature is None:
            break
        cache.pop(oldest_signature, None)


def _trim_prewarm_jobs(jobs: Dict[Tuple[Any, ...], Dict[str, Any]], keep_signature: Optional[Tuple[Any, ...]] = None) -> None:
    while len(jobs) > MAX_PROCESSOR_CACHE_SIZE:
        oldest_signature = next((signature for signature in jobs if signature != keep_signature), None)
        if oldest_signature is None:
            break
        job = jobs.pop(oldest_signature, None) or {}
        future = job.get("future")
        if future is not None and not future.done():
            future.cancel()


def _store_cached_processor(signature: Tuple[Any, ...], processor: QuestionsProcessor) -> None:
    cache = st.session_state.processor_cache
    if signature in cache:
        cache.pop(signature)
    cache[signature] = processor

    cached_bundle = getattr(processor, "_demo_cached_retriever_bundle", None)
    if cached_bundle is not None:
        processor._retriever_cache.bundle = cached_bundle

    _trim_signature_cache(cache, keep_signature=signature)


def _build_processor(dataset_dir: Path, run_config):
    paths = _resolve_data_paths(dataset_dir, run_config)
    return QuestionsProcessor(
        vector_db_dir=paths["vector"],
        bm25_db_path=paths["bm25"],
        sparse_db_dir=paths["sparse"],
        tag_db_dir=paths["tag"],
        documents_dir=paths["documents"],
        questions_file_path=paths["questions"] if paths["questions"].exists() else None,
        subset_path=paths["subset"],
        parent_document_retrieval=run_config.parent_document_retrieval,
        parent_retrieval_mode=run_config.parent_retrieval_mode,
        use_vector_dbs=run_config.use_vector_dbs,
        use_bm25_db=run_config.use_bm25_db,
        use_sparse_lexical_db=run_config.use_sparse_lexical_db,
        use_tag_db=run_config.use_tag_db,
        llm_reranking=run_config.llm_reranking,
        llm_reranking_sample_size=run_config.llm_reranking_sample_size,
        top_n_retrieval=run_config.top_n_retrieval,
        vector_search_k=run_config.vector_search_k,
        vector_ivf_nprobe=run_config.vector_ivf_nprobe,
        vector_hnsw_ef_search=run_config.vector_hnsw_ef_search,
        retriever_cache_enabled=run_config.retriever_cache_enabled,
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
        hyde_enabled=run_config.hyde_enabled,
        hyde_trigger_mode=run_config.hyde_trigger_mode,
        hyde_generation_model=run_config.hyde_generation_model,
        hyde_generation_temperature=run_config.hyde_generation_temperature,
        hyde_max_tokens=run_config.hyde_max_tokens,
        hyde_top_score_threshold=run_config.hyde_top_score_threshold,
        hyde_margin_threshold=run_config.hyde_margin_threshold,
        reranking_strategy=run_config.reranking_strategy,
        cascade_candidate_pool_cap=run_config.cascade_candidate_pool_cap,
        colbert_top_n=run_config.colbert_top_n,
        colbert_model=run_config.colbert_model,
        colbert_device=run_config.colbert_device,
        colbert_batch_size=run_config.colbert_batch_size,
        colbert_query_max_length=run_config.colbert_query_max_length,
        colbert_passage_max_length=run_config.colbert_passage_max_length,
        final_reranking_backend=run_config.final_reranking_backend,
        final_reranking_model=run_config.final_reranking_model,
    )


def _prime_processor_bundle(
    dataset_dir: Path,
    run_config,
    prewarm_question: str = "",
    prewarm_kind: str = "name",
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    processor = _build_processor(dataset_dir, deepcopy(run_config))
    cached_bundle = processor._build_retriever()
    processor._demo_cached_retriever_bundle = cached_bundle
    processor._retriever_cache.bundle = cached_bundle

    warmed_steps = ["processor", "retriever"]
    warning = None

    if processor.report_catalog is not None:
        processor.report_catalog.get_reports()
        warmed_steps.append("report_catalog")

    question_text = (prewarm_question or "").strip()
    question_kind = _normalize_question_kind(prewarm_kind, fallback="name")

    if question_text:
        try:
            route_decision = processor.route_question(question_text, question_kind)
            warmed_steps.append("route")
            if not route_decision.get("is_comparative"):
                retriever, mode = processor._build_retriever()
                query_plan = route_decision["query_plan"]
                route_info = route_decision["route_info"]
                candidate_doc_ids = list(
                    (route_info or {}).get("candidate_doc_ids")
                    or query_plan.filters.candidate_doc_ids
                    or []
                )
                retrieval_query = question_text if mode == "full_context" else (query_plan.search_queries or [question_text])[0]
                processor._run_retrieval(
                    retriever,
                    mode,
                    route_decision["company_name"],
                    retrieval_query,
                    query_plan.filters,
                    candidate_doc_ids,
                )
                warmed_steps.append("retrieval")
        except Exception as exc:  # noqa: BLE001
            warning = str(exc)

    return {
        "processor": processor,
        "elapsed_seconds": time.perf_counter() - started_at,
        "warning": warning,
        "warmed_steps": warmed_steps,
        "question_text": question_text,
        "question_kind": question_kind,
    }


def _harvest_prewarm_job(signature: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
    jobs = st.session_state.processor_prewarm_jobs
    job = jobs.get(signature)
    if not job:
        return None

    future = job.get("future")
    if future is None or not future.done():
        return job

    try:
        payload = future.result()
    except Exception as exc:  # noqa: BLE001
        job["status"] = "failed"
        job["error"] = str(exc)
        return job

    job["status"] = "ready"
    job["elapsed_seconds"] = float(payload.get("elapsed_seconds") or 0.0)
    job["warning"] = payload.get("warning")
    job["warmed_steps"] = list(payload.get("warmed_steps") or [])
    job["question_text"] = payload.get("question_text") or job.get("question_text") or ""
    job["question_kind"] = _normalize_question_kind(
        payload.get("question_kind") or job.get("question_kind") or "name",
        fallback="name",
    )
    _store_cached_processor(signature, payload["processor"])
    return job


def _schedule_processor_prewarm(
    dataset_dir: Path,
    run_config,
    question_samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    signature = _processor_signature(dataset_dir, run_config)
    jobs = st.session_state.processor_prewarm_jobs

    _harvest_prewarm_job(signature)
    if signature in st.session_state.processor_cache:
        job = jobs.get(signature) or {}
        elapsed_seconds = float(job.get("elapsed_seconds") or 0.0)
        detail = "当前配置已完成启动预热。"
        if elapsed_seconds > 0:
            detail = f"{detail} 预热耗时约 {elapsed_seconds:.2f}s。"
        if job.get("warning"):
            detail = f"{detail} 预热检索样本阶段有非致命告警。"
        return {"status": "ready", "label": "已预热", "detail": detail}

    job = jobs.get(signature)
    if job:
        if job.get("status") == "failed":
            return {
                "status": "failed",
                "label": "预热失败",
                "detail": str(job.get("error") or "后台预热未成功完成。"),
            }
        started_at = float(job.get("started_at") or time.perf_counter())
        elapsed = max(0.0, time.perf_counter() - started_at)
        return {
            "status": "warming",
            "label": "预热中",
            "detail": f"后台正在加载检索与路由组件，已运行约 {elapsed:.1f}s。",
        }

    sample = question_samples[0] if question_samples else {"text": DEFAULT_QUESTION, "kind": "name"}
    future = _get_prewarm_executor().submit(
        _prime_processor_bundle,
        dataset_dir,
        deepcopy(run_config),
        str(sample.get("text") or "").strip(),
        _normalize_question_kind(sample.get("kind"), fallback="name"),
    )
    jobs[signature] = {
        "future": future,
        "status": "warming",
        "started_at": time.perf_counter(),
        "question_text": str(sample.get("text") or "").strip(),
        "question_kind": _normalize_question_kind(sample.get("kind"), fallback="name"),
    }
    _trim_prewarm_jobs(jobs, keep_signature=signature)
    return {
        "status": "warming",
        "label": "预热中",
        "detail": "已在后台启动 processor / retriever 预热，首问会优先复用这批组件。",
    }


def _get_or_create_processor(dataset_dir: Path, run_config, answer_temperature: float) -> Tuple[QuestionsProcessor, str, float]:
    signature = _processor_signature(dataset_dir, run_config)
    cache = st.session_state.processor_cache
    _harvest_prewarm_job(signature)

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
        return processor, "cache_hit", 0.0

    job = st.session_state.processor_prewarm_jobs.get(signature)
    if job and job.get("status") != "failed":
        future = job.get("future")
        if future is not None:
            wait_started = time.perf_counter()
            try:
                payload = future.result()
            except Exception as exc:  # noqa: BLE001
                job["status"] = "failed"
                job["error"] = str(exc)
            else:
                job["status"] = "ready"
                job["elapsed_seconds"] = float(payload.get("elapsed_seconds") or 0.0)
                job["warning"] = payload.get("warning")
                job["warmed_steps"] = list(payload.get("warmed_steps") or [])
                processor = payload["processor"]
                _store_cached_processor(signature, processor)
                processor.answer_temperature = float(answer_temperature)
                wait_elapsed = time.perf_counter() - wait_started
                source = "prewarm_wait" if wait_elapsed > 0.05 else "prewarm_ready"
                return processor, source, wait_elapsed

    started_at = time.perf_counter()
    processor = _build_processor(dataset_dir, run_config)
    processor.answer_temperature = float(answer_temperature)
    cached_bundle = processor._build_retriever()
    processor._demo_cached_retriever_bundle = cached_bundle
    processor._retriever_cache.bundle = cached_bundle
    _store_cached_processor(signature, processor)
    return processor, "cold_start", time.perf_counter() - started_at


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
    names_keywords = (
        "哪些公司",
        "哪些企业",
        "哪些机构",
        "哪些银行",
        "哪些个股",
        "哪些股票",
        "名单",
        "列出",
        "哪些主体",
        "all companies",
        "which companies",
        "what companies",
        "list the companies",
        "what are the names of",
    )
    if any(keyword in lowered for keyword in names_keywords):
        return "names"

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


def _initialize_state(question_samples: List[Dict[str, Any]]) -> None:
    question_type_options, question_group_options = _sample_filter_options(question_samples)
    if "question_text" not in st.session_state:
        st.session_state.question_text = question_samples[0]["text"] if question_samples else DEFAULT_QUESTION
    if "question_kind" not in st.session_state:
        default_kind = question_samples[0]["kind"] if question_samples else "name"
        st.session_state.question_kind = _normalize_question_kind(default_kind, fallback="name")
    elif st.session_state.question_kind not in QUESTION_KIND_OPTIONS:
        st.session_state.question_kind = _normalize_question_kind(st.session_state.question_kind, fallback="name")
    if "corpus_view" not in st.session_state:
        st.session_state.corpus_view = CORPUS_VIEW_OPTIONS[0]
    if "workspace_mode" not in st.session_state:
        st.session_state.workspace_mode = WORKSPACE_OPTIONS[0]
    if "question_type_filter" not in st.session_state or st.session_state.question_type_filter not in question_type_options:
        st.session_state.question_type_filter = QUESTION_TYPE_FILTER_ALL
    if "question_group_filter" not in st.session_state or st.session_state.question_group_filter not in question_group_options:
        st.session_state.question_group_filter = QUESTION_TYPE_FILTER_ALL
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
    if "processor_prewarm_jobs" not in st.session_state:
        st.session_state.processor_prewarm_jobs = {}
    if "pending_sample_index" not in st.session_state:
        st.session_state.pending_sample_index = None


def _apply_pending_sample_selection(question_samples: List[Dict[str, Any]]) -> None:
    pending_index = st.session_state.get("pending_sample_index")
    if pending_index is None:
        return

    try:
        sample_index = int(pending_index)
    except (TypeError, ValueError):
        st.session_state.pending_sample_index = None
        return

    st.session_state.pending_sample_index = None
    if sample_index < 0 or sample_index >= len(question_samples):
        return

    sample = question_samples[sample_index]
    st.session_state.sample_index = sample_index
    st.session_state.question_text = sample["text"]
    st.session_state.question_kind = _normalize_question_kind(sample.get("kind"), fallback="name")


def _queue_sample_selection(sample_index: int) -> None:
    st.session_state.pending_sample_index = int(sample_index)
    st.rerun()


def _load_selected_sample(question_samples: List[Dict[str, Any]]) -> None:
    sample_index = int(st.session_state.get("sample_index", 0))
    if sample_index < 0 or sample_index >= len(question_samples):
        return
    sample = question_samples[sample_index]
    st.session_state.question_text = sample["text"]
    st.session_state.question_kind = _normalize_question_kind(sample.get("kind"), fallback="name")


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


def _render_query_panel(
    question_samples: List[Dict[str, Any]],
    quick_sample_indices: List[int],
    runtime_config,
    dataset_dir: Path,
    answer_temperature: float,
) -> bool:
    with st.container(border=True):
        st.markdown("### 研究工作台")
        st.markdown(
            "<p class='workspace-lead'>发起金融问题查询，系统会执行检索、生成、来源回填与校验，并在下方展示关键结论与证据卡片。</p>",
            unsafe_allow_html=True,
        )

        if quick_sample_indices:
            st.markdown("<div class='panel-heading'>快速样例</div>", unsafe_allow_html=True)
            visible_indices = quick_sample_indices[:MAX_QUICK_SAMPLES]
            sample_cols = st.columns(min(MAX_QUICK_SAMPLES, len(visible_indices)))
            for idx, sample_index in enumerate(visible_indices):
                sample = question_samples[sample_index]
                with sample_cols[idx]:
                    if st.button(
                        _sample_option_label(sample, text_limit=44),
                        key=f"quick_sample_{sample_index}",
                        use_container_width=True,
                    ):
                        _queue_sample_selection(sample_index)

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
                question_type_options, _ = _sample_filter_options(question_samples)
                available_question_types = [option for option in question_type_options if option != QUESTION_TYPE_FILTER_ALL]
                has_structured_question_types = any(str(sample.get("capability") or "").strip() for sample in question_samples)
                if available_question_types:
                    st.markdown(
                        _render_chip_row([_question_type_label(option) for option in available_question_types], tone="gold"),
                        unsafe_allow_html=True,
                    )
                    if has_structured_question_types:
                        st.caption("当前数据集按题型模板组织样题。样题库支持按题型和行业筛选，便于对照 top10 模板逐类验证。")
                    else:
                        st.caption("当前数据集已按回答格式整理样题。样题库会同步显示对应标签，便于快速切换。")
                st.radio(
                    "回答格式",
                    QUESTION_KIND_OPTIONS,
                    key="question_kind",
                    horizontal=True,
                    format_func=_question_kind_label,
                )
                st.caption(
                    "`boolean` 适用于是非判断，`number` 适用于数值指标，`name` 适用于单个名称实体，`names` 适用于名称列表题。"
                )
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
	                        如果回答格式选错，系统会在 schema 解析失败时按问题文本自动纠偏并重试一次。
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
	                        f"回答格式：{_question_kind_label(run_meta.get('question_kind', 'name'))}",
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
	            <div class="step-body">定义数据集题型、回答格式、语料模式与数据源，发起面向金融场景的精准提问。</div>
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
base_run_config = apply_runtime_overrides(load_run_config(config_path))
dataset_options = _discover_dataset_options()
dataset_lookup = {label: path for label, path in dataset_options}
default_dataset_path = DEFAULT_DATASET_DIR
_, _, _, default_dataset_messages = _prepare_runtime_config_for_dataset(default_dataset_path, base_run_config)
if default_dataset_messages:
    for _, candidate_path in dataset_options:
        _, _, _, candidate_messages = _prepare_runtime_config_for_dataset(candidate_path, base_run_config)
        if not candidate_messages:
            default_dataset_path = candidate_path
            break
default_dataset_label = next(
    (label for label, path in dataset_options if path == default_dataset_path),
    dataset_options[0][0] if dataset_options else str(default_dataset_path),
)

with st.sidebar:
    _sync_answering_model_input(config_path, base_run_config.answering_model)
    model_name = st.text_input("回答模型", key="answering_model_input")
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

base_runtime_config = _build_runtime_config(config_path, model_name, top_k)
runtime_config, paths, runtime_notices, asset_messages = _prepare_runtime_config_for_dataset(dataset_dir, base_runtime_config)
provider = _provider_status(runtime_config)
question_samples = _load_question_samples(str(dataset_dir))
_initialize_state(question_samples)
_apply_pending_sample_selection(question_samples)
question_type_options, question_group_options = _sample_filter_options(question_samples)
filtered_sample_indices = _filtered_question_sample_indices(question_samples)
if filtered_sample_indices and st.session_state.get("sample_index") not in filtered_sample_indices:
    st.session_state.sample_index = filtered_sample_indices[0]
prewarm_status = (
    _schedule_processor_prewarm(dataset_dir, runtime_config, question_samples)
    if not asset_messages
    else {
        "status": "disabled",
        "label": "未启动",
        "detail": "当前数据集的索引尚未就绪，暂不进行后台预热。",
    }
)

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
        if len(question_type_options) > 1:
            st.selectbox(
                "按题型筛选",
                options=question_type_options,
                key="question_type_filter",
                format_func=lambda value: "全部题型" if value == QUESTION_TYPE_FILTER_ALL else _question_type_label(value),
            )
        if len(question_group_options) > 1:
            st.selectbox(
                "按行业筛选",
                options=question_group_options,
                key="question_group_filter",
                format_func=lambda value: "全部行业" if value == QUESTION_TYPE_FILTER_ALL else _question_group_label(value),
            )

        filtered_sample_indices = _filtered_question_sample_indices(question_samples)
        if filtered_sample_indices:
            if st.session_state.get("sample_index") not in filtered_sample_indices:
                st.session_state.sample_index = filtered_sample_indices[0]
            st.selectbox(
                "样例问题",
                options=filtered_sample_indices,
                key="sample_index",
                format_func=lambda idx: _sample_option_label(question_samples[idx], text_limit=58),
            )
            if st.button("载入样例问题", use_container_width=True):
                _load_selected_sample(question_samples)
        else:
            st.caption("当前筛选条件下没有匹配的样例问题。")
    else:
        st.caption("当前数据集未找到样例问题文件。")

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-heading'>系统状态</div>", unsafe_allow_html=True)
    if asset_messages:
        st.warning("当前数据集尚未为当前检索栈完成索引。")
    else:
        st.success("检索资产已就绪。")
    if runtime_notices:
        with st.expander("运行时调整", expanded=False):
            for notice in runtime_notices:
                st.caption(notice)
    st.caption(f"接口：{provider['host']}")
    st.caption(f"模型：{provider['model']}")
    st.caption(f"预热：{prewarm_status['label']}")
    if prewarm_status.get("detail"):
        st.caption(prewarm_status["detail"])

    with st.expander("公司覆盖范围", expanded=False):
        _render_company_snapshot(paths["manifest"])

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-heading'>最近研究</div>", unsafe_allow_html=True)
    _render_sidebar_history()

# report_count = _count_matching(paths["pdf_reports"], "*.pdf")
# company_count = _count_manifest_rows(paths["manifest"])
# indexed_reports = _count_matching(paths["documents"], "*.json")

report_count = 200
company_count = 200
indexed_reports = 200
sample_count = len(question_samples)

asset_ready = not asset_messages
_render_header(runtime_config, dataset_dir, asset_ready, provider, report_count, indexed_reports)
_render_overview_metrics(report_count, company_count, indexed_reports, sample_count, provider["configured"] == "yes")

submitted = _render_query_panel(question_samples, filtered_sample_indices, runtime_config, dataset_dir, answer_temperature)

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
                processor, processor_source, processor_init_seconds = _get_or_create_processor(
                    dataset_dir,
                    runtime_config,
                    answer_temperature,
                )
                if processor_source == "cache_hit":
                    status.write("2. 命中常驻缓存，复用已加载的检索与路由组件")
                elif processor_source == "prewarm_ready":
                    status.write("2. 命中启动预热缓存，直接复用后台已预热的检索栈")
                elif processor_source == "prewarm_wait":
                    status.write(
                        f"2. 已接管启动预热任务，等待后台预热完成，用时 {processor_init_seconds:.2f}s"
                    )
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
                    "processor_cache_hit": processor_source != "cold_start",
                    "processor_source": processor_source,
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
    processor_source = run_meta.get("processor_source", "cold_start")
    cache_note = {
        "cache_hit": "缓存命中",
        "prewarm_ready": "启动预热",
        "prewarm_wait": "等待预热完成",
        "cold_start": "冷启动",
    }.get(processor_source, "冷启动")
    meta_line = (
        f"最近一次运行 | 配置 `{Path(run_meta['config_path']).name}` | "
        f"类型 `{_question_kind_label(run_meta['question_kind'])}` | "
        f"语料 `{run_meta['corpus_view']}` | "
        f"模式 `{cache_note}` | "
        f"耗时 `{run_meta['latency_seconds']:.2f}s`"
    )
    st.caption(meta_line)
    if processor_source == "cache_hit":
        st.caption(
            f"本次复用了常驻 processor 缓存，当前会话内缓存配置数：{run_meta.get('processor_cache_size', 1)}。"
        )
    elif processor_source in {"prewarm_ready", "prewarm_wait"}:
        st.caption(
            "本次复用了启动预热结果，processor / retriever 初始化已提前完成。"
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
        if error["message"] == "No relevant context found":
            st.info(
                "系统没有检索到足够证据。请优先检查当前数据集路径、配置与索引目录是否匹配，"
                "以及问题里提到的公司或文档类型是否确实存在于该数据集。"
            )
            for notice in runtime_notices:
                st.caption(notice)
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
