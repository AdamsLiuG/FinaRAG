from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_NAME_FIELD_REGISTRY: List[Dict[str, Any]] = [
    {
        "field_key": "legal_representative",
        "question_aliases": ["法定代表人", "legal representative"],
        "evidence_aliases": ["法定代表人", "legal representative"],
    },
    {
        "field_key": "chairman",
        "question_aliases": ["董事长", "chairman of the board", "chairman"],
        "evidence_aliases": ["董事长", "chairman of the board", "chairman"],
    },
    {
        "field_key": "general_manager",
        "question_aliases": ["总经理", "总裁", "general manager", "ceo", "chief executive officer"],
        "evidence_aliases": ["总经理", "总裁", "general manager", "chief executive officer", "首席执行官"],
    },
    {
        "field_key": "board_secretary",
        "question_aliases": ["董事会秘书", "董秘", "board secretary"],
        "evidence_aliases": ["董事会秘书", "董秘", "board secretary"],
    },
    {
        "field_key": "securities_affairs_representative",
        "question_aliases": ["证券事务代表", "securities affairs representative"],
        "evidence_aliases": ["证券事务代表", "securities affairs representative"],
    },
    {
        "field_key": "company_name",
        "question_aliases": ["公司名称", "中文名称", "公司中文名称", "英文名称", "公司英文名称", "简称"],
        "evidence_aliases": ["公司名称", "中文名称", "公司中文名称", "英文名称", "公司英文名称", "简称", "中文简称"],
    },
]

_NAME_CORPORATE_SUFFIXES = [
    "股份有限公司",
    "有限责任公司",
    "股份公司",
    "有限公司",
    "集团股份有限公司",
    "集团有限公司",
    "corporation",
    "co.,ltd.",
    "co., ltd.",
    "co., ltd",
    "company limited",
    "limited",
]

_BOOLEAN_TARGET_REGISTRY: List[Dict[str, Any]] = [
    {
        "target_key": "cash_dividend_plan",
        "question_aliases": ["现金分红方案", "现金分红", "现金红利", "现金股利", "现金派息", "派现", "派息"],
        "positive_aliases": [
            "现金分红",
            "现金红利",
            "现金股利",
            "派发现金红利",
            "派发现金股利",
            "派息",
            "现金分红金额",
            "现金股利人民币",
            "每10股派息",
            "每10股派发现金",
            "累计现金分红",
        ],
        "negative_patterns": [
            "不进行现金分红",
            "不派发现金红利",
            "不派发现金股利",
            "不派发现金股息",
            "不派现",
            "不分红",
            "不进行利润分配",
            "不派发现金",
        ],
        "plan_keywords": [
            "利润分配预案",
            "利润分配方案",
            "利润分配及资本公积转增股本方案",
            "拟分配",
            "预案",
            "方案",
        ],
    },
    {
        "target_key": "share_buyback_plan",
        "question_aliases": ["股份回购", "股票回购", "回购计划", "回购方案", "回购股份"],
        "positive_aliases": [
            "股份回购",
            "股票回购",
            "回购股份",
            "回购计划",
            "回购方案",
            "拟回购",
            "实施回购",
            "回购专用证券账户",
        ],
        "negative_patterns": [
            "不进行股份回购",
            "不进行股票回购",
            "无回购计划",
            "未实施回购",
            "不存在回购计划",
            "未回购股份",
        ],
        "plan_keywords": [
            "回购方案",
            "回购计划",
            "回购报告书",
            "股份回购",
            "股票回购",
        ],
    },
    {
        "target_key": "nonstandard_audit_opinion",
        "question_aliases": ["非标准审计意见", "非标审计意见", "审计意见", "审计报告"],
        "positive_aliases": [
            "非标准审计意见",
            "非标审计意见",
            "保留意见",
            "否定意见",
            "无法表示意见",
            "带强调事项段无保留意见",
            "审计意见",
            "审计意见类型",
        ],
        "negative_patterns": [
            "标准无保留意见",
            "无保留意见审计报告",
            "审计意见类型:标准无保留意见",
            "审计意见类型：标准无保留意见",
            "出具了标准无保留意见",
        ],
        "plan_keywords": [
            "审计报告",
            "审计意见",
            "保留意见",
            "否定意见",
            "无法表示意见",
        ],
    },
    {
        "target_key": "controlling_shareholder_change",
        "question_aliases": ["控股股东变更", "控股股东发生变更", "实际控制人变更", "控制权变更"],
        "positive_aliases": [
            "控股股东变更",
            "控股股东发生变更",
            "实际控制人变更",
            "实际控制人发生变更",
            "控制权变更",
            "控制权发生变更",
        ],
        "negative_patterns": [
            "控股股东未发生变更",
            "实际控制人未发生变更",
            "控制权未发生变更",
            "控股股东无变化",
            "实际控制人无变化",
            "控制权无变化",
        ],
        "plan_keywords": [
            "股份变动及股东情况",
            "公司治理",
            "控股股东",
            "实际控制人",
            "控制权",
        ],
    },
]

_GENERIC_BOOLEAN_NEGATIONS = [
    "不",
    "无",
    "未",
    "没有",
    "不存在",
    "未见",
]


def normalize_pages(values: Any) -> List[int]:
    pages: List[int] = []
    for value in values or []:
        try:
            pages.append(int(value))
        except (TypeError, ValueError):
            continue
    return pages


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_loose_text(value: Any) -> str:
    text = normalize_text(value)
    replacements = {
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "，": ",",
        "：": ":",
        "；": ";",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    return text


def compact_snippet(text: Any, limit: int = 240) -> str:
    snippet = normalize_text(text)
    if len(snippet) <= limit:
        return snippet
    return snippet[: limit - 3].rstrip() + "..."


def boolean_answer_to_value(value: Any) -> Any:
    if value == "N/A":
        return "N/A"
    if isinstance(value, bool):
        return value
    lowered = str(value or "").strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "n/a":
        return "N/A"
    return value


def get_answer_dict(record: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(record.get("assistant_response"), dict):
        return record["assistant_response"]
    if isinstance(record.get("answer"), dict):
        return record["answer"]
    return {}


def get_teacher_signal(record: Dict[str, Any], cache_record: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if cache_record and isinstance(cache_record.get("teacher_signal"), dict):
        return cache_record["teacher_signal"]
    answer = get_answer_dict(record)
    return answer if isinstance(answer, dict) else {}


def get_retrieval_results(record: Dict[str, Any], cache_record: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    if cache_record is not None:
        cached = cache_record.get("retrieval_results")
        if isinstance(cached, list) and cached:
            return cached
    retrieval_results = record.get("retrieval_results")
    if isinstance(retrieval_results, list):
        return retrieval_results
    answer = get_answer_dict(record)
    retrieval_results = answer.get("retrieval_results")
    if isinstance(retrieval_results, list):
        return retrieval_results
    return []


def get_retrieval_pages(record: Dict[str, Any], cache_record: Optional[Dict[str, Any]] = None) -> List[int]:
    pages = record.get("retrieval_pages")
    if pages not in (None, "", []):
        return normalize_pages(pages)
    teacher_signal = get_teacher_signal(record, cache_record)
    pages = teacher_signal.get("retrieval_pages")
    if pages not in (None, "", []):
        return normalize_pages(pages)
    return normalize_pages([item.get("page") for item in get_retrieval_results(record, cache_record)])


def build_reject_log(
    *,
    stage: str,
    reason_code: str,
    schema: str,
    query_id: Any,
    sample_id: Any = None,
    message: Optional[str] = None,
    decision: str = "reject",
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "stage": stage,
        "decision": decision,
        "reason_code": reason_code,
        "reason_message": message or reason_code,
        "schema": schema,
        "query_id": query_id,
        "sample_id": sample_id,
    }
    if details:
        payload["details"] = details
    return payload


def _evidence_identity(item: Dict[str, Any]) -> Tuple[int, str]:
    return (
        int(item.get("page") or 0),
        str(item.get("chunk_id") or ""),
    )


def select_evidence_items(
    record: Dict[str, Any],
    *,
    cache_record: Optional[Dict[str, Any]] = None,
    preferred_pages: Optional[Sequence[int]] = None,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    retrieval_results = get_retrieval_results(record, cache_record)
    if not retrieval_results:
        return []

    preferred_set = {int(page) for page in preferred_pages or []}
    selected: List[Dict[str, Any]] = []
    for item in retrieval_results:
        try:
            page = int(item.get("page") or 0)
        except (TypeError, ValueError):
            page = 0
        if preferred_set and page not in preferred_set:
            continue
        selected.append(item)
    if not selected:
        selected = list(retrieval_results[: max(1, limit)])

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in selected:
        key = _evidence_identity(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def infer_name_field_profile(question_text: Any) -> Dict[str, Any]:
    normalized_question = normalize_loose_text(question_text)
    field_keys: List[str] = []
    aliases: List[str] = []
    for entry in _NAME_FIELD_REGISTRY:
        question_aliases = [normalize_loose_text(alias) for alias in entry["question_aliases"]]
        if any(alias and alias in normalized_question for alias in question_aliases):
            field_keys.append(str(entry["field_key"]))
            aliases.extend(entry["evidence_aliases"])
    deduped_aliases = []
    seen = set()
    for alias in aliases:
        normalized = normalize_loose_text(alias)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped_aliases.append(alias)
    return {
        "field_keys": field_keys,
        "evidence_aliases": deduped_aliases,
        "allow_answer_only_match": not deduped_aliases,
        "is_comparative_like": any(token in str(question_text or "") for token in ["谁的", "哪家", "哪一个更", "更高", "更低"]),
    }


def build_name_aliases(value: Any, extra_aliases: Optional[Iterable[Any]] = None) -> List[str]:
    raw = normalize_text(value)
    if not raw:
        return []

    variants = {raw}
    bracket_normalized = raw.replace("（", "(").replace("）", ")")
    variants.add(bracket_normalized)
    variants.add(re.sub(r"[()（）]", "", bracket_normalized))
    variants.add(re.sub(r"[\"'“”‘’]", "", bracket_normalized))
    variants.add(re.sub(r"\s+", "", bracket_normalized))

    stripped_brackets = re.sub(r"[（(][^）)]*[）)]", "", bracket_normalized).strip()
    if stripped_brackets:
        variants.add(stripped_brackets)

    lower_raw = bracket_normalized.lower()
    for suffix in _NAME_CORPORATE_SUFFIXES:
        suffix_norm = suffix.lower()
        if lower_raw.endswith(suffix_norm):
            trimmed = bracket_normalized[: -len(suffix)].strip(" -_,")
            if trimmed:
                variants.add(trimmed)

    for alias in extra_aliases or []:
        alias_text = normalize_text(alias)
        if alias_text:
            variants.add(alias_text)

    ordered = []
    seen = set()
    for variant in variants:
        normalized = normalize_loose_text(variant)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(variant)
    return ordered


def compare_name_values(left: Any, right: Any) -> Dict[str, Any]:
    left_raw = normalize_text(left)
    right_raw = normalize_text(right)
    if not left_raw or not right_raw:
        return {
            "matched": False,
            "match_type": "missing_value",
            "allow_partial_match": False,
        }

    if left_raw == right_raw:
        return {"matched": True, "match_type": "exact", "allow_partial_match": False}

    left_norm = normalize_loose_text(left_raw)
    right_norm = normalize_loose_text(right_raw)
    if left_norm == right_norm:
        return {"matched": True, "match_type": "normalized_exact", "allow_partial_match": False}

    left_aliases = build_name_aliases(left_raw)
    right_aliases = build_name_aliases(right_raw)
    left_alias_norm = {normalize_loose_text(item) for item in left_aliases}
    right_alias_norm = {normalize_loose_text(item) for item in right_aliases}
    if left_alias_norm & right_alias_norm:
        return {"matched": True, "match_type": "alias_exact", "allow_partial_match": False}

    shorter, longer = (left_norm, right_norm) if len(left_norm) <= len(right_norm) else (right_norm, left_norm)
    if shorter and shorter in longer:
        safe_left = build_name_aliases(left_raw)
        safe_right = build_name_aliases(right_raw)
        if any(normalize_loose_text(item) == shorter for item in [*safe_left, *safe_right]):
            return {"matched": True, "match_type": "safe_partial_alias", "allow_partial_match": True}

    return {"matched": False, "match_type": "mismatch", "allow_partial_match": False}


def classify_name_support(
    record: Dict[str, Any],
    *,
    answer_value: Any = None,
    anchor_record: Optional[Dict[str, Any]] = None,
    cache_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    answer = get_answer_dict(record)
    field_profile = infer_name_field_profile(record.get("question_text"))
    answer_value = answer_value if answer_value not in (None, "", []) else answer.get("final_answer")
    if answer_value in (None, "", [], "N/A"):
        if anchor_record is not None:
            answer_value = anchor_record.get("final_answer")
    aliases = build_name_aliases(
        answer_value,
        extra_aliases=(anchor_record or {}).get("aliases") if isinstance(anchor_record, dict) else None,
    )
    preferred_pages = normalize_pages(answer.get("relevant_pages"))
    if not preferred_pages and anchor_record is not None:
        preferred_pages = normalize_pages(anchor_record.get("anchor_pages"))
    items = select_evidence_items(
        record,
        cache_record=cache_record,
        preferred_pages=preferred_pages,
        limit=8,
    )

    explicit_hits: List[Dict[str, Any]] = []
    answer_only_hits: List[Dict[str, Any]] = []
    answer_norms = {normalize_loose_text(alias) for alias in aliases if normalize_loose_text(alias)}
    label_norms = {
        normalize_loose_text(alias)
        for alias in field_profile["evidence_aliases"]
        if normalize_loose_text(alias)
    }
    for item in items:
        text = str(item.get("text") or "")
        if not text.strip():
            continue
        page = item.get("page")
        chunk_id = item.get("chunk_id")
        lines = [line for line in re.split(r"[\n\r]+", text) if line.strip()] or [text]
        normalized_lines = [normalize_loose_text(line) for line in lines]
        for index, normalized_line in enumerate(normalized_lines):
            if not any(answer_norm and answer_norm in normalized_line for answer_norm in answer_norms):
                continue

            window_norm = "".join(normalized_lines[max(0, index - 1): min(len(normalized_lines), index + 2)])
            matched_labels = [
                alias
                for alias in field_profile["evidence_aliases"]
                if normalize_loose_text(alias) and normalize_loose_text(alias) in window_norm
            ]
            hit = {
                "page": page,
                "chunk_id": chunk_id,
                "snippet": compact_snippet(lines[index]),
                "matched_labels": matched_labels,
            }
            if matched_labels:
                explicit_hits.append(hit)
            else:
                answer_only_hits.append(hit)

    support_type = "none"
    if explicit_hits:
        support_type = "explicit_field_hit"
    elif answer_only_hits:
        support_type = "answer_text_only"

    return {
        "validator_name": "name_strict_v1",
        "question_profile": field_profile,
        "answer_value": answer_value,
        "answer_aliases": aliases,
        "preferred_pages": preferred_pages,
        "support_type": support_type,
        "explicit_hits": explicit_hits[:8],
        "answer_only_hits": answer_only_hits[:8],
        "support_pages": sorted(
            {
                int(hit["page"])
                for hit in [*explicit_hits, *answer_only_hits]
                if hit.get("page") not in (None, "")
            }
        ),
        "evidence_aliases": sorted(label_norms),
    }


def validate_name_answer(
    record: Dict[str, Any],
    *,
    anchor_record: Optional[Dict[str, Any]] = None,
    cache_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    schema = str(record.get("schema") or "")
    answer = get_answer_dict(record)
    final_answer = answer.get("final_answer")
    doc_ids = [str(item) for item in record.get("doc_ids", []) if item not in (None, "")]
    question_profile = infer_name_field_profile(record.get("question_text"))
    query_id = record.get("query_id")

    base = {
        "validator_name": "name_strict_v1",
        "schema": schema,
        "query_id": query_id,
        "anchor_available": bool(anchor_record),
        "anchor_id": (anchor_record or {}).get("anchor_id") if isinstance(anchor_record, dict) else None,
        "anchor_source_bucket": (anchor_record or {}).get("source_bucket") if isinstance(anchor_record, dict) else None,
        "answer_value": final_answer,
        "decision": "reject",
        "decision_reason": "",
        "reject_log": None,
    }

    if question_profile["is_comparative_like"] or len(doc_ids) > 1:
        base["decision_reason"] = "name_query_not_single_doc"
        base["reject_log"] = build_reject_log(
            stage="filter_name_validator",
            reason_code="name_query_not_single_doc",
            schema=schema,
            query_id=query_id,
            sample_id=record.get("sample_id"),
            details={"doc_ids": doc_ids},
        )
        return base

    if final_answer == "N/A":
        if anchor_record is not None and anchor_record.get("final_answer") not in (None, "", [], "N/A"):
            base["decision_reason"] = "name_anchor_positive_but_refused"
            base["reject_log"] = build_reject_log(
                stage="filter_name_validator",
                reason_code="name_anchor_positive_but_refused",
                schema=schema,
                query_id=query_id,
                sample_id=record.get("sample_id"),
                details={"anchor_id": anchor_record.get("anchor_id"), "anchor_answer": anchor_record.get("final_answer")},
            )
            return base
        base["decision"] = "accept"
        base["decision_reason"] = "name_refusal_without_positive_anchor"
        base["support_type"] = "refusal"
        base["accepted_checks"] = ["name_refusal_without_anchor"]
        return base

    if not isinstance(final_answer, str) or not final_answer.strip():
        base["decision_reason"] = "name_final_answer_not_string"
        base["reject_log"] = build_reject_log(
            stage="filter_name_validator",
            reason_code="name_final_answer_not_string",
            schema=schema,
            query_id=query_id,
            sample_id=record.get("sample_id"),
        )
        return base

    comparison = None
    if anchor_record is not None:
        comparison = compare_name_values(final_answer, anchor_record.get("final_answer"))
        if not comparison["matched"]:
            base["decision_reason"] = "name_anchor_value_mismatch"
            base["name_match_result"] = comparison
            base["reject_log"] = build_reject_log(
                stage="filter_name_validator",
                reason_code="name_anchor_value_mismatch",
                schema=schema,
                query_id=query_id,
                sample_id=record.get("sample_id"),
                details={
                    "anchor_id": anchor_record.get("anchor_id"),
                    "anchor_answer": anchor_record.get("final_answer"),
                    "answer_value": final_answer,
                    "match_type": comparison["match_type"],
                },
            )
            return base

    support = classify_name_support(
        record,
        answer_value=final_answer,
        anchor_record=anchor_record,
        cache_record=cache_record,
    )
    base["support_type"] = support["support_type"]
    base["support_pages"] = support["support_pages"]
    base["support_hits"] = support["explicit_hits"]
    base["answer_only_hits"] = support["answer_only_hits"]
    base["question_profile"] = support["question_profile"]
    if comparison is not None:
        base["name_match_result"] = comparison

    if support["support_type"] == "explicit_field_hit":
        base["decision"] = "accept"
        base["decision_reason"] = "name_explicit_field_hit"
        checks = ["name_validated_explicit_field_hit"]
        if comparison and comparison["matched"]:
            checks.append(f"name_anchor_match:{comparison['match_type']}")
        elif anchor_record is not None:
            checks.append("name_anchor_match:exact")
        base["accepted_checks"] = checks
        return base

    if support["support_type"] == "answer_text_only" and question_profile["allow_answer_only_match"]:
        base["decision"] = "accept"
        base["decision_reason"] = "name_answer_text_only"
        base["accepted_checks"] = ["name_validated_answer_text_only"]
        return base

    base["decision_reason"] = "name_missing_explicit_grounding"
    base["reject_log"] = build_reject_log(
        stage="filter_name_validator",
        reason_code="name_missing_explicit_grounding",
        schema=schema,
        query_id=query_id,
        sample_id=record.get("sample_id"),
        details={
            "support_type": support["support_type"],
            "anchor_id": (anchor_record or {}).get("anchor_id") if isinstance(anchor_record, dict) else None,
            "support_pages": support["support_pages"],
        },
    )
    return base


def infer_boolean_profile(question_text: Any) -> Dict[str, Any]:
    question_text = normalize_text(question_text)
    normalized_question = normalize_loose_text(question_text)
    for entry in _BOOLEAN_TARGET_REGISTRY:
        if any(normalize_loose_text(alias) in normalized_question for alias in entry["question_aliases"]):
            return {
                "target_key": entry["target_key"],
                "question_text": question_text,
                "positive_aliases": list(entry["positive_aliases"]),
                "negative_patterns": list(entry["negative_patterns"]),
                "plan_keywords": list(entry["plan_keywords"]),
                "generic_target": None,
            }

    match = re.search(r"是否(?:提到|提及|披露|存在|进行|采用|属于|有|为)?(.+?)(?:[？?。]|$)", question_text)
    generic_target = normalize_text(match.group(1)) if match else ""
    return {
        "target_key": "generic_boolean",
        "question_text": question_text,
        "positive_aliases": [generic_target] if generic_target else [],
        "negative_patterns": [],
        "plan_keywords": [],
        "generic_target": generic_target or None,
    }


def _line_has_any_alias(line_norm: str, aliases: Sequence[str]) -> bool:
    return any(normalize_loose_text(alias) in line_norm for alias in aliases if normalize_loose_text(alias))


def _classify_boolean_line(line: str, profile: Dict[str, Any]) -> Optional[str]:
    line_norm = normalize_loose_text(line)
    if not line_norm:
        return None

    if profile["target_key"] == "generic_boolean":
        generic_target = normalize_loose_text(profile.get("generic_target"))
        if not generic_target or generic_target not in line_norm:
            return None
        if any(neg in line_norm for neg in _GENERIC_BOOLEAN_NEGATIONS):
            return "explicit_negative"
        return "explicit_positive"

    if not _line_has_any_alias(line_norm, profile["positive_aliases"]):
        return None

    if any(normalize_loose_text(pattern) in line_norm for pattern in profile["negative_patterns"]):
        return "explicit_negative"
    return "explicit_positive"


def classify_boolean_context(
    record: Dict[str, Any],
    *,
    cache_record: Optional[Dict[str, Any]] = None,
    anchor_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    answer = get_answer_dict(record)
    preferred_pages = normalize_pages(answer.get("relevant_pages"))
    if not preferred_pages and anchor_record is not None:
        preferred_pages = normalize_pages(anchor_record.get("anchor_pages"))

    profile = infer_boolean_profile(record.get("question_text"))
    items = select_evidence_items(
        record,
        cache_record=cache_record,
        preferred_pages=preferred_pages,
        limit=8,
    )
    positive_hits: List[Dict[str, Any]] = []
    negative_hits: List[Dict[str, Any]] = []
    for item in items:
        text = str(item.get("text") or "")
        if not text.strip():
            continue
        page = item.get("page")
        chunk_id = item.get("chunk_id")
        for line in [segment for segment in re.split(r"[\n\r]+", text) if segment.strip()] or [text]:
            label = _classify_boolean_line(line, profile)
            if label is None:
                continue
            hit = {
                "page": page,
                "chunk_id": chunk_id,
                "snippet": compact_snippet(line),
                "classification": label,
            }
            if label == "explicit_negative":
                negative_hits.append(hit)
            elif label == "explicit_positive":
                positive_hits.append(hit)

    classification = "insufficient_evidence"
    conflict_policy = "none"
    if negative_hits and positive_hits:
        question_norm = normalize_loose_text(record.get("question_text"))
        has_plan_question = any(normalize_loose_text(keyword) in question_norm for keyword in profile.get("plan_keywords", []))
        negative_text = "".join(normalize_loose_text(hit["snippet"]) for hit in negative_hits)
        if has_plan_question or any(normalize_loose_text(keyword) in negative_text for keyword in profile.get("plan_keywords", [])):
            classification = "explicit_negative"
            conflict_policy = "prefer_negative_plan_evidence"
        else:
            classification = "conflict"
            conflict_policy = "conflicting_positive_and_negative_hits"
    elif negative_hits:
        classification = "explicit_negative"
    elif positive_hits:
        classification = "explicit_positive"

    return {
        "validator_name": "boolean_trinary_v1",
        "profile": profile,
        "preferred_pages": preferred_pages,
        "classification": classification,
        "conflict_policy": conflict_policy,
        "positive_hits": positive_hits[:10],
        "negative_hits": negative_hits[:10],
        "support_pages": sorted(
            {
                int(hit["page"])
                for hit in [*positive_hits, *negative_hits]
                if hit.get("page") not in (None, "")
            }
        ),
    }


def validate_boolean_answer(
    record: Dict[str, Any],
    *,
    anchor_record: Optional[Dict[str, Any]] = None,
    cache_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    schema = str(record.get("schema") or "")
    answer = get_answer_dict(record)
    final_answer = boolean_answer_to_value(answer.get("final_answer"))
    classification = classify_boolean_context(record, cache_record=cache_record, anchor_record=anchor_record)
    query_id = record.get("query_id")

    result = {
        "validator_name": "boolean_trinary_v1",
        "schema": schema,
        "query_id": query_id,
        "answer_value": final_answer,
        "classification": classification["classification"],
        "support_pages": classification["support_pages"],
        "positive_hits": classification["positive_hits"],
        "negative_hits": classification["negative_hits"],
        "decision": "reject",
        "decision_reason": "",
        "normalized_final_answer": final_answer,
        "should_refuse": final_answer == "N/A",
        "accepted_checks": [],
        "reject_log": None,
    }

    if final_answer not in (True, False, "N/A"):
        result["decision_reason"] = "boolean_final_answer_not_trinary"
        result["reject_log"] = build_reject_log(
            stage="filter_boolean_validator",
            reason_code="boolean_final_answer_not_trinary",
            schema=schema,
            query_id=query_id,
            sample_id=record.get("sample_id"),
        )
        return result

    cls = classification["classification"]
    if cls == "explicit_positive":
        result["accepted_checks"].append("boolean_explicit_positive")
        if final_answer is True:
            result["decision"] = "accept"
            result["decision_reason"] = "boolean_matches_explicit_positive"
            return result
        if final_answer in {"N/A", False}:
            result["decision"] = "rewrite"
            result["decision_reason"] = "boolean_answer_rewritten_to_true"
            result["normalized_final_answer"] = True
            result["should_refuse"] = False
            result["accepted_checks"].append("boolean_auto_rewrite_true")
            return result
        if final_answer is True:
            result["decision"] = "accept"
            result["decision_reason"] = "boolean_matches_explicit_positive"
            return result

    if cls == "explicit_negative":
        result["accepted_checks"].append("boolean_explicit_negative")
        if final_answer is False:
            result["decision"] = "accept"
            result["decision_reason"] = "boolean_matches_explicit_negative"
            return result
        if final_answer in {"N/A", True}:
            result["decision"] = "rewrite"
            result["decision_reason"] = "boolean_answer_rewritten_to_false"
            result["normalized_final_answer"] = False
            result["should_refuse"] = False
            result["accepted_checks"].append("boolean_auto_rewrite_false")
            return result
        if final_answer is False:
            result["decision"] = "accept"
            result["decision_reason"] = "boolean_matches_explicit_negative"
            return result

    if cls in {"insufficient_evidence", "conflict"}:
        result["accepted_checks"].append(f"boolean_{cls}")
        if final_answer == "N/A":
            result["decision"] = "accept"
            result["decision_reason"] = "boolean_refusal_with_insufficient_evidence"
            result["should_refuse"] = True
            return result
        result["decision"] = "rewrite"
        result["decision_reason"] = "boolean_downgraded_to_refusal"
        result["normalized_final_answer"] = "N/A"
        result["should_refuse"] = True
        result["accepted_checks"].append("boolean_auto_downgrade_refusal")
        return result

    result["decision_reason"] = "boolean_unhandled_classification"
    result["reject_log"] = build_reject_log(
        stage="filter_boolean_validator",
        reason_code="boolean_unhandled_classification",
        schema=schema,
        query_id=query_id,
        sample_id=record.get("sample_id"),
        details={"classification": cls},
    )
    return result
