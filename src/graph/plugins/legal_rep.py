"""Legal representative rule extraction and question plugin."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from src.query_plan import QueryPlan
from src.text_normalization import normalize_text


def legal_rep_event():
    """Fetch EVT-001 legal representative event atom."""
    from src.event_atoms import get_default_event_store

    event = get_default_event_store().find_event("EVT-001")
    if event is None:
        raise RuntimeError("EVT-001（现任法定代表人识别）事件原子缺失，请检查知识层编译")
    return event


def clean_person_name(raw_value: str) -> Optional[str]:
    value = re.sub(r"^[\s|:：,，。；;]+|[\s|:：,，。；;]+$", "", str(raw_value or ""))
    value = re.split(r"[|、,，。；;\s]+", value, maxsplit=1)[0]
    value = value.strip("为系：:")
    match = re.match(r"[\u4e00-\u9fff·]{2,8}", value)
    if not match:
        return None
    name = re.sub(
        r"(先生|女士|博士|教授|董事长|副董事长|行长|董事|总经理|负责人)$",
        "",
        match.group(0),
    )
    if any(term in name for term in ("公司", "负责人", "法定", "代表人")):
        return None
    if len(name) < 2:
        return None
    return name


def legal_page_priority(page: int, text: str) -> float:
    score = 0.0
    if page <= 20:
        score += 18.0
    normalized = normalize_text(text)
    if any(term in normalized for term in ("公司简介", "公司基本情况", "公司信息", "主要财务指标")):
        score += 14.0
    if "财务报告" in normalized and "公司负责人" in normalized:
        score += 8.0
    return score


def find_legal_representative(processor: Any, document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    best_match: Optional[Dict[str, Any]] = None
    best_score = 0.0
    for page, page_text in processor._document_pages(document):
        page_score = processor._legal_page_priority(page, page_text)
        event = legal_rep_event()
        for line in page_text.splitlines():
            normalized_line = normalize_text(line).replace(" ", "").replace("⼈", "人")
            if not any(term in normalized_line for term in event.trigger_terms):
                continue
            if any(term in normalized_line for term in event.negative_terms):
                continue
            for detection in event.detection_patterns:
                match = detection.regex.search(normalized_line)
                if not match:
                    continue
                base_score, match_type = detection.score, detection.match_type
                name = processor._clean_person_name(match.group(1))
                if not name:
                    continue
                score = base_score + page_score
                if score > best_score:
                    best_score = score
                    best_match = {
                        "name": name,
                        "page": page,
                        "snippet": line.strip()[:500],
                        "match_type": match_type,
                        "score": round(score, 4),
                    }
    return best_match


def answer_legal_representative_question(
    processor: Any,
    *,
    question: str,
    schema: str,
    company_name: str,
    query_plan: QueryPlan,
    route_info: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if (schema or "").lower() != "name" or "法定代表人" not in (question or ""):
        return None
    doc_id = processor._selected_doc_id(company_name, route_info, query_plan)
    document = processor._load_document_payload(doc_id)
    if not document:
        return None
    match = processor._find_legal_representative(document)
    if not match:
        return None

    evidence = processor._document_page_result(
        document=document,
        page=match["page"],
        text=match["snippet"],
        chunk_type="legal_representative_rule",
        source="legal_representative_rule",
        score=1.0,
    )
    reasoning = (
        f"法定代表人规则在第 {match['page']} 页命中 {match['match_type']} 证据，"
        f"抽取姓名为 {match['name']}。"
    )
    return processor._build_rule_answer(
        final_answer=match["name"],
        question=question,
        schema=schema,
        query_plan=query_plan,
        route_info=route_info,
        retrieval_results=[evidence],
        relevant_pages=[match["page"]],
        rule_name="legal_representative_rule",
        reasoning=reasoning,
        extra_debug=match,
    )


class LegalRepresentativePlugin:
    name = "legal_representative_rule"

    def execute(self, processor: Any, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return processor._answer_legal_representative_question(
            question=ctx["question"],
            schema=ctx["schema"],
            company_name=ctx["company_name"],
            query_plan=ctx["query_plan"],
            route_info=ctx.get("route_info"),
        )
