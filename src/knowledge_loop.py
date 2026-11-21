"""FinaWiki online telemetry-to-knowledge flywheel (trace normalization & candidate atom clustering).

Advisory only: never auto-mutates config/metric_atoms.yaml. Candidate drafts are validated
against MetricAtomStore via metric_atom_mining.validate_atom_drafts and emitted for human review.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

from src.metric_atom_mining import render_draft_yaml, validate_atom_drafts
from src.metric_atoms import MetricAtomStore

_PRIORITY_FLAGS = (
    "claim_contradicted",
    "metric_caliber_conflict_refused",
    "metric_evidence_mismatch",
    "implicit_intent_evidence_mismatch",
)
_METRIC_PHRASE_RE = re.compile(
    r"([A-Za-z0-9\u4e00-\u9fff]{2,16}(?:净资产收益率|毛利率|净利率|资产负债率|资本充足率|"
    r"拨备覆盖率|不良贷款率|利息净收入|营业总收入|营业收入|归母净利润|净利润|现金流量净额|"
    r"现金流|净额|余额|周转率|周转天数|研发投入|资本开支|股利|分红))"
)
_GENERIC_STOP_TERMS = frozenset({"多少", "什么", "情况", "数据", "指标", "变化", "分析", "公司", "年度"})


def normalize_trace_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Unify LangGraph reflection traces and TelemetryManager online feedback records."""
    raw = dict(entry or {})
    qplan = raw.get("query_plan") if isinstance(raw.get("query_plan"), dict) else {}
    atom_payload = qplan.get("metric_atom") if isinstance(qplan.get("metric_atom"), dict) else {}
    flags = [str(f) for f in (raw.get("validation_flags") or []) if f]

    atom_id = raw.get("atom_id") or atom_payload.get("atom_id") or None
    metric = raw.get("metric") or qplan.get("metric") or atom_payload.get("standard_name") or None
    queries = list(raw.get("queries") or qplan.get("search_queries") or [])

    trigger = raw.get("trigger")
    if not trigger:
        trigger = next((f for f in _PRIORITY_FLAGS if f in flags), flags[0] if flags else "low_confidence")

    claim_audit = raw.get("claim_audit") if isinstance(raw.get("claim_audit"), dict) else {}
    claim_contradicted = int(
        raw.get("claim_contradicted_count")
        if raw.get("claim_contradicted_count") is not None
        else (claim_audit.get("contradicted_count") or 0)
    )

    source = raw.get("source") or ("online_telemetry" if "validation_flags" in raw else "adaptive_reflect")
    outcome = raw.get("outcome") or ("telemetry_flagged" if source == "online_telemetry" else None)

    return {
        **raw,
        "source": source,
        "atom_id": str(atom_id) if atom_id else None,
        "metric": str(metric).strip() if metric else None,
        "validation_flags": flags,
        "trigger": str(trigger),
        "queries": [str(q) for q in queries if q],
        "new_hits": int(raw.get("new_hits") or 0),
        "rounds": int(raw.get("rounds") or 0),
        "gap_closed": bool(raw.get("gap_closed", False)),
        "outcome": outcome,
        "claim_contradicted_count": claim_contradicted,
    }


def extract_unresolved_metric_term(entry: Dict[str, Any]) -> Optional[str]:
    """Extract candidate financial metric term from an adaptive/telemetry trace entry."""
    norm = normalize_trace_entry(entry)
    metric = str(norm.get("metric") or "").strip()
    if metric and metric not in _GENERIC_STOP_TERMS and len(metric) >= 2:
        return metric

    qplan = norm.get("query_plan") if isinstance(norm.get("query_plan"), dict) else {}
    for candidate_text in (
        qplan.get("normalized_query"),
        *(norm.get("queries") or []),
        norm.get("question"),
    ):
        txt = str(candidate_text or "").strip()
        if not txt:
            continue
        m = _METRIC_PHRASE_RE.search(txt)
        if m:
            term = m.group(1).strip()
            if term not in _GENERIC_STOP_TERMS and len(term) >= 2:
                return term
    return None


def cluster_candidate_atoms(
    entries: Sequence[Dict[str, Any]],
    store: MetricAtomStore,
    *,
    min_samples: int = 3,
) -> Dict[str, Any]:
    """Cluster repeated online low-confidence / mismatch / contradiction traces into candidate suggestions."""
    term_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    contradiction_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    known_terms = {term for atom in store.all_atoms() for term in atom.match_terms}

    for raw in entries:
        entry = normalize_trace_entry(raw)
        flags = set(entry.get("validation_flags") or [])
        trigger = str(entry.get("trigger") or "")
        term = extract_unresolved_metric_term(entry)

        is_mismatch = (
            entry.get("atom_id") is None
            or "metric_evidence_mismatch" in flags
            or "metric_caliber_conflict_refused" in flags
            or trigger in {"metric_evidence_mismatch", "evidence_gap"}
        )
        if is_mismatch and term and term not in known_terms:
            term_buckets[term].append(entry)

        has_contradiction = (
            "claim_contradicted" in flags
            or trigger == "claim_contradicted"
            or int(entry.get("claim_contradicted_count") or 0) > 0
        )
        if has_contradiction:
            key = str(entry.get("atom_id") or term or entry.get("question") or "unknown")[:48]
            contradiction_buckets[key].append(entry)

    drafts: List[Dict[str, Any]] = []
    draft_flags_map: Dict[str, Counter[str]] = {}
    for term, bucket in sorted(term_buckets.items()):
        if len(bucket) < max(1, int(min_samples)):
            continue
        flag_counts = Counter(f for e in bucket for f in (e.get("validation_flags") or [e.get("trigger")]))
        draft_flags_map[term] = flag_counts
        drafts.append(
            {
                "standard_name": term,
                "aliases": [term],
                "definition": f"由线上低置信/证据缺口样本聚类提取的候选金融指标：{term}",
                "caliber_notes": "待人工核对合并/母公司口径、正负号及计量单位",
                "unit": "%" if term.endswith(("率", "比例")) else "元",
                "statement_location": "待人工确认（利润表/资产负债表/现金流量表/主要财务指标）",
                "pitfalls": "; ".join(k for k, _ in flag_counts.most_common(3)) or "low_confidence",
                "evidence_targets": [term],
                "confusable_with_standard_names": [],
            }
        )

    draft_errors = validate_atom_drafts(drafts, store) if drafts else {}
    validated_drafts = [d for d in drafts if not draft_errors.get(d["standard_name"])]

    suggestions: List[Dict[str, Any]] = []
    for draft in validated_drafts:
        name = draft["standard_name"]
        bucket = term_buckets[name]
        top_flags = ", ".join(f"{k}:{v}" for k, v in draft_flags_map[name].most_common(2))
        suggestions.append(
            {
                "rule": "R4_candidate_new_atom",
                "atom_id": "NEW",
                "term": name,
                "samples": len(bucket),
                "evidence": f"未覆盖指标出现 {len(bucket)} 次低置信/错配 trace（{top_flags}），已通过规则预检",
            }
        )

    for key, bucket in sorted(contradiction_buckets.items()):
        if len(bucket) < max(1, int(min_samples)):
            continue
        suggestions.append(
            {
                "rule": "R5_claim_contradiction_hotspot",
                "atom_id": key,
                "term": key,
                "samples": len(bucket),
                "evidence": f"句级核验检测到 {len(bucket)} 次论点冲突（claim_contradicted），建议补充口径负例/别名",
            }
        )

    return {
        "suggestions": suggestions,
        "validated_drafts": validated_drafts,
        "draft_errors": draft_errors,
        "draft_yaml": render_draft_yaml(validated_drafts) if validated_drafts else "",
    }
