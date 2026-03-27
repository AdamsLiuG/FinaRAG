from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.pipeline import load_run_config
from src.questions_processing import QuestionsProcessor


st.set_page_config(page_title="FinaRAG Demo", layout="wide")
st.title("FinaRAG Demo")
st.caption("A financial-report RAG demo with query rewrite, citations, and confidence labels.")

project_root = Path(__file__).resolve().parents[1]
default_dataset = project_root / "data" / "test_set"
default_config = project_root / "config" / "qwen_base.yaml"

dataset_dir = Path(st.text_input("Dataset Directory", str(default_dataset)))
config_path = Path(st.text_input("Config Path", str(default_config)))
question = st.text_area(
    "Question",
    value="Did Mercia Asset Management PLC mention any mergers or acquisitions in the annual report?",
    height=120,
)

if st.button("Run Question"):
    run_config = load_run_config(config_path)
    processor = QuestionsProcessor(
        vector_db_dir=dataset_dir / ("databases_ser_tab" if run_config.use_serialized_tables else "databases") / "vector_dbs",
        bm25_db_path=dataset_dir / ("databases_ser_tab" if run_config.use_serialized_tables else "databases") / "bm25_dbs",
        sparse_db_dir=dataset_dir / ("databases_ser_tab" if run_config.use_serialized_tables else "databases") / "sparse_dbs",
        documents_dir=dataset_dir / ("databases_ser_tab" if run_config.use_serialized_tables else "databases") / "chunked_reports",
        subset_path=dataset_dir / "subset.csv",
        parent_document_retrieval=run_config.parent_document_retrieval,
        use_vector_dbs=run_config.use_vector_dbs,
        use_bm25_db=run_config.use_bm25_db,
        use_sparse_lexical_db=run_config.use_sparse_lexical_db,
        llm_reranking=run_config.llm_reranking,
        llm_reranking_sample_size=run_config.llm_reranking_sample_size,
        top_n_retrieval=run_config.top_n_retrieval,
        parallel_requests=run_config.parallel_requests,
        api_provider=run_config.api_provider,
        answering_model=run_config.answering_model,
        full_context=run_config.full_context,
    )
    result = processor.process_question(question, "boolean" if question.lower().startswith("did ") else "number")

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Answer")
        st.json(
            {
                "final_answer": result.get("final_answer"),
                "reasoning_summary": result.get("reasoning_summary"),
                "relevant_pages": result.get("relevant_pages"),
                "confidence": result.get("confidence"),
                "references": result.get("references"),
                "citations": result.get("citations"),
            }
        )
    with right:
        st.subheader("Debug")
        st.code(json.dumps(processor.response_data, ensure_ascii=False, indent=2), language="json")
