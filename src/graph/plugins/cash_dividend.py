"""Cash dividend mention rule extraction and question plugin."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from src.analysis_playbooks import get_default_playbook_store
from src.query_plan import QueryPlan


def dividend_mention_terms() -> Tuple[str, ...]:
    """Fetch cash_dividend_mentions from PLB-001 playbook."""
    playbook = get_default_playbook_store().get("PLB-001")
    if playbook is None:
        return ()
    return tuple(playbook.cash_dividend_mentions)


def is_cash_dividend_mention_question(question: str, schema: str) -> bool:
    if (schema or "").lower() != "boolean":
        return False
    question = question or ""
    if not any(term in question for term in dividend_mention_terms()):
        return False
    return any(term in question for term in ("是否提到", "是否提及", "年度利润分配预案", "利润分配预案", "重要事项"))


def find_cash_dividend_mention(processor: Any, document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    best_match: Optional[Dict[str, Any]] = None
    best_score = 0.0
    terms = dividend_mention_terms()
    for page, page_text in processor._document_pages(document):
        matched_term = next((term for term in terms if term in page_text), None)
        if not matched_term:
            continue
        score = 50.0
        if page <= 5:
            score += 20.0
        if any(term in page_text for term in ("利润分配预案", "利润分配", "重要提示", "重要事项")):
            score += 10.0
        lines = [line.strip() for line in page_text.splitlines() if matched_term in line]
        snippet = (lines[0] if lines else page_text)[:500]
        if score > best_score:
            best_score = score
            best_match = {
                "page": page,
                "term": matched_term,
                "snippet": snippet,
                "score": round(score, 4),
            }
    return best_match


def answer_cash_dividend_mention_question(
    processor: Any,
    *,
    question: str,
    schema: str,
    company_name: str,
    query_plan: QueryPlan,
    route_info: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not processor._is_cash_dividend_mention_question(question, schema):
        return None
    doc_id = processor._selected_doc_id(company_name, route_info, query_plan)
    document = processor._load_document_payload(doc_id)
    if not document:
        return None
    match = processor._find_cash_dividend_mention(document)
    if not match:
        return None

    evidence = processor._document_page_result(
        document=document,
        page=match["page"],
        text=match["snippet"],
        chunk_type="cash_dividend_rule",
        source="cash_dividend_rule",
        score=1.0,
    )
    reasoning = f"现金分红 mention 规则在第 {match['page']} 页命中“{match['term']}”，因此文本已提到现金分红。"
    return processor._build_rule_answer(
        final_answer=True,
        question=question,
        schema=schema,
        query_plan=query_plan,
        route_info=route_info,
        retrieval_results=[evidence],
        relevant_pages=[match["page"]],
        rule_name="cash_dividend_rule",
        reasoning=reasoning,
        extra_debug=match,
    )


class CashDividendPlugin:
    name = "cash_dividend_rule"

    def execute(self, processor: Any, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return processor._answer_cash_dividend_mention_question(
            question=ctx["question"],
            schema=ctx["schema"],
            company_name=ctx["company_name"],
            query_plan=ctx["query_plan"],
            route_info=ctx.get("route_info"),
        )
