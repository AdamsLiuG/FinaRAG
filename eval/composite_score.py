from __future__ import annotations

from typing import Any, Dict, Iterable, List

from eval.dataset_schema import FinanceGoldAnswer


ANSWER_VALUE_WEIGHTS = {
    "type_aware_value_score": 0.65,
    "entity_score": 0.20,
    "keyword_score": 0.15,
}

EVIDENCE_RETRIEVAL_WEIGHTS = {
    "doc_hit": 0.15,
    "page_hit": 0.25,
    "page_recall": 0.20,
    "table_hit": 0.15,
    "ragas_context_recall": 0.15,
    "ragas_context_precision": 0.10,
}

CITATION_GROUNDING_WEIGHTS = {
    "citation_page_hit": 0.35,
    "citation_precision": 0.25,
    "citation_coverage": 0.20,
    "ragas_faithfulness": 0.20,
}

FINAL_COMPONENT_WEIGHTS = {
    "answer_score": 0.45,
    "retrieval_score": 0.35,
    "citation_score": 0.20,
}

RETRIEVAL_WEIGHTS = {
    "doc_hit": 0.2,
    "page_hit": 0.35,
    "page_recall": 0.25,
    "table_hit": 0.2,
}

CITATION_WEIGHTS = {
    "citation_page_hit": 0.4,
    "citation_precision": 0.3,
    "citation_coverage": 0.3,
}

RAGAS_INTERNAL_WEIGHTS = {
    "answer_correctness": 0.30,
    "faithfulness": 0.20,
    "answer_relevancy": 0.10,
    "context_recall": 0.25,
    "context_precision": 0.15,
}


def get_finance_scoring_profile() -> Dict[str, Any]:
    return {
        "profile_name": "finance_atomic_answer_rag_three_layer_v3",
        "answer_value_weights": ANSWER_VALUE_WEIGHTS,
        "evidence_retrieval_weights": EVIDENCE_RETRIEVAL_WEIGHTS,
        "citation_grounding_weights": CITATION_GROUNDING_WEIGHTS,
        "ragas_internal_weights": RAGAS_INTERNAL_WEIGHTS,
        "final_component_weights": FINAL_COMPONENT_WEIGHTS,
        "retrieval_weights": RETRIEVAL_WEIGHTS,
        "citation_weights": CITATION_WEIGHTS,
        "layers": {
            "layer_1_type_aware_answer_value": [
                "type_aware_value_score",
                "entity_score",
                "keyword_score",
            ],
            "layer_2_evidence_retrieval_quality": [
                "doc_hit",
                "page_hit",
                "page_recall",
                "table_hit",
                "ragas_context_recall",
                "ragas_context_precision",
            ],
            "layer_3_citation_grounding": [
                "citation_page_hit",
                "citation_precision",
                "citation_coverage",
                "ragas_faithfulness",
            ],
            "ragas_auxiliary_breakdown": [
                "answer_correctness",
                "answer_relevancy",
                "faithfulness",
                "context_recall",
                "context_precision",
            ],
        },
    }


def _weighted_average(scores: Dict[str, float | None], weights: Dict[str, float]) -> float | None:
    weighted_sum = 0.0
    active_weight = 0.0
    for key, score in scores.items():
        if score is None:
            continue
        weight = weights.get(key, 0.0)
        if weight <= 0:
            continue
        weighted_sum += score * weight
        active_weight += weight
    if active_weight == 0:
        return None
    return round(weighted_sum / active_weight, 4)


def _extract_gold_pages(gold_answer: FinanceGoldAnswer) -> List[int]:
    if gold_answer.gold_pages:
        return sorted(set(gold_answer.gold_pages))
    return sorted({reference.page_index + 1 for reference in gold_answer.references})


def _extract_gold_doc_ids(gold_answer: FinanceGoldAnswer) -> List[str]:
    if gold_answer.doc_ids:
        return sorted(set(gold_answer.doc_ids))
    return sorted({reference.pdf_sha1 for reference in gold_answer.references if reference.pdf_sha1})


def _extract_pred_pages(pred_answer: Dict) -> List[int]:
    pages = {
        citation.get("page")
        for citation in pred_answer.get("citations", []) or []
        if isinstance(citation.get("page"), int)
    }
    pages.update(
        reference.get("page_index", -1) + 1
        for reference in pred_answer.get("references", []) or []
        if isinstance(reference.get("page_index"), int)
    )
    return sorted(page for page in pages if page > 0)


def _extract_pred_doc_ids(pred_answer: Dict) -> List[str]:
    doc_ids = {
        citation.get("source")
        for citation in pred_answer.get("citations", []) or []
        if citation.get("source")
    }
    doc_ids.update(
        reference.get("pdf_sha1")
        for reference in pred_answer.get("references", []) or []
        if reference.get("pdf_sha1")
    )
    return sorted(doc_id for doc_id in doc_ids if doc_id)


def _extract_retrieval_pages(debug_detail: Dict | None, pred_answer: Dict) -> List[int]:
    if debug_detail:
        retrieval_results = debug_detail.get("retrieval_results") or []
        pages = [
            result.get("page")
            for result in retrieval_results
            if isinstance(result.get("page"), int)
        ]
        if pages:
            return sorted({page for page in pages if page > 0})
        retrieval_pages = [
            page
            for page in (debug_detail.get("retrieval_pages") or [])
            if isinstance(page, int)
        ]
        if retrieval_pages:
            return sorted({page for page in retrieval_pages if page > 0})
    return _extract_pred_pages(pred_answer)


def _extract_retrieval_doc_ids(debug_detail: Dict | None, pred_answer: Dict) -> List[str]:
    doc_ids = set(_extract_pred_doc_ids(pred_answer))
    for result in (debug_detail or {}).get("retrieval_results", []) or []:
        metadata = result.get("metadata") or {}
        if metadata.get("sha1_name"):
            doc_ids.add(metadata["sha1_name"])
    return sorted(doc_id for doc_id in doc_ids if doc_id)


def _has_table_evidence(pred_answer: Dict, debug_detail: Dict | None = None) -> bool:
    table_chunk_types = {"table", "serialized_table", "table_grounding"}
    if any((citation.get("chunk_type") in table_chunk_types) for citation in pred_answer.get("citations", []) or []):
        return True
    for result in (debug_detail or {}).get("retrieval_results", []) or []:
        metadata = result.get("metadata") or {}
        if metadata.get("chunk_type") in table_chunk_types:
            return True
    return False


def compute_retrieval_support(
    pred_answer: Dict,
    gold_answer: FinanceGoldAnswer,
    *,
    debug_detail: Dict | None = None,
) -> Dict[str, Any]:
    if gold_answer.should_refuse:
        return {"retrieval_score": None, "skipped": True, "skip_reason": "refusal_case"}

    gold_pages = set(_extract_gold_pages(gold_answer))
    gold_doc_ids = set(_extract_gold_doc_ids(gold_answer))
    observed_pages = set(_extract_retrieval_pages(debug_detail, pred_answer))
    observed_doc_ids = set(_extract_retrieval_doc_ids(debug_detail, pred_answer))

    doc_hit = 1.0 if gold_doc_ids and gold_doc_ids & observed_doc_ids else 0.0 if gold_doc_ids else None
    page_hit = 1.0 if gold_pages and gold_pages & observed_pages else 0.0 if gold_pages else None
    page_recall = round(len(gold_pages & observed_pages) / len(gold_pages), 4) if gold_pages else None

    table_hit = None
    if gold_answer.evidence_type == "table":
        table_hit = 1.0 if _has_table_evidence(pred_answer, debug_detail=debug_detail) else 0.0

    retrieval_score = _weighted_average(
        {
            "doc_hit": doc_hit,
            "page_hit": page_hit,
            "page_recall": page_recall,
            "table_hit": table_hit,
        },
        RETRIEVAL_WEIGHTS,
    )
    return {
        "retrieval_score": retrieval_score,
        "doc_hit": doc_hit,
        "page_hit": page_hit,
        "page_recall": page_recall,
        "table_hit": table_hit,
        "observed_pages": sorted(observed_pages),
        "observed_doc_ids": sorted(observed_doc_ids),
        "skipped": False,
    }


def compute_citation_support(pred_answer: Dict, gold_answer: FinanceGoldAnswer) -> Dict[str, Any]:
    if gold_answer.should_refuse:
        return {"citation_score": None, "skipped": True, "skip_reason": "refusal_case"}

    gold_pages = set(_extract_gold_pages(gold_answer))
    cited_pages = {
        citation.get("page")
        for citation in pred_answer.get("citations", []) or []
        if isinstance(citation.get("page"), int)
    }

    citation_page_hit = 1.0 if gold_pages and gold_pages & cited_pages else 0.0 if gold_pages else None
    citation_precision = round(len(gold_pages & cited_pages) / len(cited_pages), 4) if cited_pages else 0.0
    citation_coverage = round(len(gold_pages & cited_pages) / len(gold_pages), 4) if gold_pages else None

    citation_score = _weighted_average(
        {
            "citation_page_hit": citation_page_hit,
            "citation_precision": citation_precision,
            "citation_coverage": citation_coverage,
        },
        CITATION_WEIGHTS,
    )
    return {
        "citation_score": citation_score,
        "citation_page_hit": citation_page_hit,
        "citation_precision": citation_precision,
        "citation_coverage": citation_coverage,
        "cited_pages": sorted(cited_pages),
        "skipped": False,
    }


def compute_finance_case_score(
    *,
    semantic_result: Dict[str, Any],
    entity_result: Dict[str, Any],
    keyword_result: Dict[str, Any] | None,
    ragas_result: Dict[str, Any],
    retrieval_result: Dict[str, Any],
    citation_result: Dict[str, Any],
) -> Dict[str, Any]:
    type_aware_value_score = semantic_result.get("semantic_score")
    answer_score = _weighted_average(
        {
            "type_aware_value_score": type_aware_value_score,
            "entity_score": entity_result.get("entity_score"),
            "keyword_score": (keyword_result or {}).get("keyword_score"),
        },
        ANSWER_VALUE_WEIGHTS,
    )
    retrieval_score = _weighted_average(
        {
            "doc_hit": retrieval_result.get("doc_hit"),
            "page_hit": retrieval_result.get("page_hit"),
            "page_recall": retrieval_result.get("page_recall"),
            "table_hit": retrieval_result.get("table_hit"),
            "ragas_context_recall": ragas_result.get("context_recall"),
            "ragas_context_precision": ragas_result.get("context_precision"),
        },
        EVIDENCE_RETRIEVAL_WEIGHTS,
    )
    citation_score = _weighted_average(
        {
            "citation_page_hit": citation_result.get("citation_page_hit"),
            "citation_precision": citation_result.get("citation_precision"),
            "citation_coverage": citation_result.get("citation_coverage"),
            "ragas_faithfulness": ragas_result.get("faithfulness"),
        },
        CITATION_GROUNDING_WEIGHTS,
    )

    final_quality_score = _weighted_average(
        {
            "answer_score": answer_score,
            "retrieval_score": retrieval_score,
            "citation_score": citation_score,
        },
        FINAL_COMPONENT_WEIGHTS,
    )
    return {
        "answer_score": answer_score,
        "type_aware_value_score": type_aware_value_score,
        "semantic_score": semantic_result.get("semantic_score"),
        "entity_score": entity_result.get("entity_score"),
        "keyword_score": (keyword_result or {}).get("keyword_score"),
        "retrieval_score": retrieval_score,
        "retrieval_rule_score": retrieval_result.get("retrieval_score"),
        "citation_score": citation_score,
        "citation_rule_score": citation_result.get("citation_score"),
        "ragas_context_recall": ragas_result.get("context_recall"),
        "ragas_context_precision": ragas_result.get("context_precision"),
        "ragas_faithfulness": ragas_result.get("faithfulness"),
        "ragas_score": ragas_result.get("ragas_score"),
        "final_quality_score": final_quality_score,
    }
