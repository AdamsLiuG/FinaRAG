"""Render a deterministic research memo from adjudicated report evidence."""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.deep_research.evidence_ledger import EvidenceEntry

_ENABLED = {"1", "true", "yes"}
_SCOPE = {"consolidated": "合并", "parent": "母公司"}
_ASPECTS = {"related_party": "关联方披露", "shareholder": "股东关系", "subsidiary": "子公司关系", "risk": "风险披露"}


@dataclass(frozen=True)
class _Finding:
    text: str
    sources: tuple[EvidenceEntry, ...]
    quantitative: bool
    evidence_id: str


def _number(value):
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _describe_key(key):
    scope = _SCOPE.get(key.scope, "")
    return " ".join(filter(None, (key.entity_id, key.period, _ASPECTS.get(key.metric_atom, key.metric_atom),
                                  scope, key.basis if key.basis != "reported" else "", key.segment)))


def _finding(entry, ledger):
    if entry.normalized_value is None:
        label = f"{entry.label}。报告摘录" if entry.label else f"{_ASPECTS[entry.key.metric_atom]}摘录"
        return _Finding(f"{entry.company_name} {label}「{entry.excerpt}」", (entry,), False, entry.evidence_id)
    dimension = entry.key.dimension
    unit = {"CNY:money": "元", "CNY:per_share": "元/股", "percent": "%"}.get(dimension, dimension.split(":")[0])
    sources = tuple(ledger.entry(eid) for eid in entry.inputs) if entry.inputs else (entry,)
    caliber = _SCOPE.get(entry.key.scope, entry.key.scope)
    if entry.key.basis != "reported":
        caliber = f"{caliber} {entry.key.basis}"
    if entry.key.segment:
        caliber = f"{caliber} {entry.key.segment}"
    text = f"{entry.company_name} {entry.key.period} {entry.label}（{caliber}）为 {_number(entry.normalized_value)} {unit}。"
    return _Finding(text, sources, True, entry.evidence_id)


def _citation(entry, ledger):
    metadata = ledger.metadata(entry)
    return {
        **metadata, "citation_id": entry.evidence_id, "evidence_id": entry.evidence_id,
        "doc_id": entry.doc_id, "pdf_sha1": entry.doc_id, "source": entry.doc_id, "page": entry.page,
        "value_period": entry.key.period if entry.normalized_value is not None else None,
        "row_idx": entry.row_idx, "matched_row_headers": [entry.label] if entry.row_idx is not None else [],
        "matched_col_headers": [entry.column] if entry.column is not None else [],
        "unit": entry.unit, "evidence_snippet": entry.excerpt,
        "retrieval_sources": [entry.source_class],
    }


def _annotate(text, entries):
    return f"{text} " + " ".join(f"[{entry.evidence_id}]" for entry in entries)


def _render(findings, disagreements, gaps):
    if not findings:
        return "N/A"
    metrics = [_annotate(item.text, item.sources) for item in findings if item.quantitative]
    qualitative = [_annotate(item.text, item.sources) for item in findings if not item.quantitative]
    sections = (
        ("执行摘要", [_annotate(findings[0].text, findings[0].sources)]),
        ("财务指标与计算", metrics),
        ("定性与关系发现", qualitative),
        ("来源分歧", [line for line, _ in disagreements]),
        ("证据缺口", gaps),
    )
    return "\n\n".join(f"## {heading}\n" + "\n".join(f"- {item}" for item in items)
                        for heading, items in sections if items)


class ResearchSynthesizer:
    def __init__(self, *, claim_verifier=None):
        self._claim_verifier = claim_verifier

    def _audit(self, findings, ledger, query_plan):
        verifier = self._claim_verifier
        if verifier is None:
            from src.claim_verifier import ClaimVerifier

            verifier = ClaimVerifier(sql_store=None, nli_scorer_fn=None)
        sources = {entry.evidence_id: entry for finding in findings for entry in finding.sources}
        audit = verifier.verify(
            "\n".join(finding.text for finding in findings),
            citations=[_citation(entry, ledger) for entry in sources.values()],
            retrieval_results=[{"pdf_sha1": entry.doc_id, "page": entry.page, "text": entry.excerpt,
                                "metadata": ledger.metadata(entry)} for entry in sources.values()],
            company_name=query_plan.filters.company_name,
            report_year=None,  # The scope can include multiple report years and value periods.
        ).to_dict()
        unsupported = [item["claim"].strip() for item in audit["claims"] if item["status"] != "supported"]
        suppressed = set()
        for claim in unsupported:
            matching = {item.evidence_id for item in findings if claim and (claim in item.text or item.text in claim)}
            suppressed.update(matching or (item.evidence_id for item in findings))
        return audit, suppressed

    def synthesize(self, question, run, ledger, *, query_plan, route_info):
        resolutions = ledger.adjudicate()
        findings = []
        disagreements = []
        gaps = []
        flags = []
        for key in run.required_keys:
            resolved = resolutions.get(key)
            if resolved and resolved.winner_id:
                findings.append(_finding(ledger.entry(resolved.winner_id), ledger))
            else:
                gaps.append(f"{_describe_key(key)}缺少可裁定证据。")
                alternatives = any(
                    other != key and (other.entity_id, other.metric_atom, other.period)
                    == (key.entity_id, key.metric_atom, key.period) for other in resolutions
                )
                if key.dimension != "observation" and (alternatives or resolved and resolved.reason == "same_priority_conflict"):
                    flags.append("metric_caliber_conflict_refused")
            if resolved and resolved.disagreements:
                identities = ((resolved.winner_id,) if resolved.winner_id else ()) + resolved.disagreements
                entries = tuple(ledger.entry(eid) for eid in identities)
                detail = "采用优先级更高的证据。" if resolved.winner_id else "同优先级来源存在未解决的分歧。"
                disagreements.append((_annotate(f"{_describe_key(key)}，{detail}", entries), entries))
        if run.stop_reason in {"research_unavailable", "invalid_research_scope", "budget_exhausted"}:
            gaps.append({"research_unavailable": "本地研究数据源不可用。",
                         "invalid_research_scope": "无法确认所请求的报告范围。",
                         "budget_exhausted": "研究请求已达到执行预算。"}[run.stop_reason])

        audit = None
        if findings and os.getenv("FINARAG_CLAIM_VERIFY_ENABLED", "0").strip().lower() in _ENABLED:
            try:
                audit, suppressed = self._audit(findings, ledger, query_plan)
                if audit["contradicted_count"]:
                    flags.append("claim_contradicted")
                if audit["total_claims"] >= 2 and audit["unsupported_claim_rate"] > 0.5:
                    flags.append("claim_high_unverified_rate")
                if suppressed:
                    findings = [item for item in findings if item.evidence_id not in suppressed
                                and all(entry.evidence_id not in suppressed for entry in item.sources)]
                    gaps.append("声明核查未能支持部分结论。")
            except Exception:
                flags.append("claim_verification_unavailable")
                gaps.append("声明核查不可用。")

        used = {entry.evidence_id: entry for finding in findings for entry in finding.sources}
        if findings:
            used.update((entry.evidence_id, entry) for _, entries in disagreements for entry in entries)
        references = {(entry.doc_id, entry.page): {"pdf_sha1": entry.doc_id, "page": entry.page}
                      for entry in used.values()}
        if gaps or not findings:
            flags.append("insufficient_evidence")
        return {
            "final_answer": _render(findings, disagreements, gaps),
            "references": list(references.values()),
            "citations": [_citation(entry, ledger) for entry in used.values()],
            "relevant_pages": sorted({page for _, page in references}),
            "confidence": "low" if flags else "high",
            "validation_flags": sorted(set(flags)),
            "evidence_gaps": gaps,
            "query_plan": query_plan.to_dict(),
            "route_info": route_info,
            **({"claim_audit": audit} if audit is not None else {}),
        }
