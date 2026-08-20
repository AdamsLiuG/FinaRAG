"""Agentic 自适应检索 v1：知识层驱动的证据缺口定向补检（try-then-closed 前半）。

设计原则：不是「给 RAG 加 agent 循环」，是知识层告诉系统**何时补检、往哪补、
何时停止**——触发条件来自 FinaWiki 知识资产（原子 evidence_targets / 剧本
required_metrics），补检查询由知识资产构造（排除易混口径标准名，避免强化错口径），
停止条件由预算上限控制（ADAPTIVE_MAX_ROUNDS，默认 1，防循环）。

v1 挂点：检索完成后、生成前（预检模式，一次流程内完成，无需生成段重入）。
v2 路径：校验后触发（caliber_conflict → 补检 → 重新生成 → try-then-closed
完整闭环）需生成段可重入，与 D3 trace 体系配套实施。

纯函数模块：无 IO、无实例状态，控制流全部离线可测。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.metric_atoms import MetricAtom, MetricAtomStore


TRIGGER_EVIDENCE_GAP = "evidence_gap"
TRIGGER_PLAYBOOK_INCOMPLETE = "playbook_incomplete"
TRIGGER_LOW_CONFIDENCE = "low_confidence"

ADAPTIVE_MIN_TOP_SCORE = float(os.getenv("ADAPTIVE_MIN_TOP_SCORE", "0.15"))
ADAPTIVE_MAX_SUPPLEMENT_QUERIES = 3


@dataclass(frozen=True)
class RetrievalGap:
    """一次补检的决策产物：触发原因 + 定向补检查询 + 依据（进 trace）。"""

    trigger: str
    supplement_queries: List[str] = field(default_factory=list)
    rationale: str = ""


def _result_texts(retrieval_results: List[Dict]) -> str:
    parts = []
    for result in retrieval_results or []:
        text = result.get("text") or (result.get("metadata") or {}).get("text") or ""
        parts.append(str(text))
    return "\n".join(parts)


def _targets_covered(targets: List[str], corpus_text: str) -> List[str]:
    """未被检索证据文本覆盖的 target（包含匹配，粗粒度 v1 判定）。"""
    return [t for t in targets if t and t not in corpus_text]


def _atom_supplement_queries(atom: MetricAtom, limit: int) -> List[str]:
    """原子驱动补检查询：evidence_targets 优先，aliases 兜底；**排除易混口径
    standard_name**——补检方向若含易混词，检索会强化错口径证据。"""
    queries: List[str] = []
    for target in atom.evidence_targets:
        if target not in queries:
            queries.append(target)
    for alias in atom.aliases:
        if alias not in queries and alias != atom.standard_name:
            queries.append(alias)
    return queries[:limit]


def detect_gap(
    query_plan,
    retrieval_results: List[Dict],
    top_score: Optional[float] = None,
    atom_store: Optional[MetricAtomStore] = None,
) -> Optional[RetrievalGap]:
    """预检模式缺口判定（短路，优先级即顺序）。

    1. evidence_gap：命中原子的 evidence_targets 未被检索证据覆盖 → 定向补检
    2. playbook_incomplete：剧本 required_metrics 有指标证据未覆盖 → 逐指标补检
    3. low_confidence：top combined_score 低于阈值（检索置信度不足）→ 别名扩展补检

    无缺口返回 None（零成本路径：不产生任何补检副作用）。
    """
    if atom_store is None:
        from src.metric_atoms import get_default_store
        atom_store = get_default_store()

    corpus_text = _result_texts(retrieval_results)
    metric_atom_payload = getattr(query_plan, "metric_atom", None) or {}
    atom_id = metric_atom_payload.get("atom_id")
    atom = atom_store.get(atom_id) if atom_id else None

    if atom is not None:
        uncovered = _targets_covered(atom.evidence_targets, corpus_text)
        # OR 语义：任一目标词出现即视为有本口径证据（evidence_targets 混有
        # 位置词如「合并利润表」，AND 语义会导致几乎永远触发缺口）。
        # 触发条件 = 所有目标词均未覆盖（完全无证据）。
        if uncovered and len(uncovered) == len(atom.evidence_targets):
            queries = [q for q in _atom_supplement_queries(atom, ADAPTIVE_MAX_SUPPLEMENT_QUERIES)
                       if q in uncovered or q in atom.aliases]
            # 补检方向排除易混口径标准名（防强化错口径）
            confusable = set(metric_atom_payload.get("confusable_names") or [])
            queries = [q for q in queries if q not in confusable]
            if queries:
                return RetrievalGap(
                    trigger=TRIGGER_EVIDENCE_GAP,
                    supplement_queries=queries[:ADAPTIVE_MAX_SUPPLEMENT_QUERIES],
                    rationale=f"原子 {atom.atom_id} 的证据目标全部未覆盖：{uncovered}",
                )

    playbook_payload = getattr(query_plan, "playbook", None) or {}
    playbook_id = playbook_payload.get("playbook_id") if isinstance(playbook_payload, dict) else None
    if playbook_id:
        # payload 已带 required_metrics（build_playbook_payload），不回查全局 store
        required_metrics = list(playbook_payload.get("required_metrics") or [])
        missing: List[str] = []
        for metric_atom_id in required_metrics:
            metric = atom_store.get(metric_atom_id)
            if metric is None:
                continue
            targets = [metric.standard_name] + metric.evidence_targets
            if len(_targets_covered(targets, corpus_text)) == len(targets):
                missing.append(metric.standard_name)  # 全部未覆盖 → 该指标缺证据
        if missing:
            return RetrievalGap(
                trigger=TRIGGER_PLAYBOOK_INCOMPLETE,
                supplement_queries=missing[:ADAPTIVE_MAX_SUPPLEMENT_QUERIES + 1],
                rationale=f"剧本 {playbook_id} 必需指标证据未覆盖：{missing}",
            )

    if top_score is not None and retrieval_results and top_score < ADAPTIVE_MIN_TOP_SCORE:
        if atom is not None:
            queries = _atom_supplement_queries(atom, 2)
        else:
            queries = []
        return RetrievalGap(
            trigger=TRIGGER_LOW_CONFIDENCE,
            supplement_queries=queries,
            rationale=f"检索置信度不足：top_score={top_score} < {ADAPTIVE_MIN_TOP_SCORE}",
        )

    return None


def merge_supplement_results(
    base: List[Dict], supplement: List[Dict]
) -> List[Dict]:
    """补检结果合并进候选池：按 (pdf_sha1, page) 去重，新结果标记 adaptive 来源。"""
    seen = set()
    for item in base or []:
        key = (str(item.get("pdf_sha1") or ""), item.get("page"))
        seen.add(key)
    merged = list(base or [])
    for item in supplement or []:
        key = (str(item.get("pdf_sha1") or ""), item.get("page"))
        if key in seen:
            continue
        seen.add(key)
        enriched = dict(item)
        sources = list(enriched.get("retrieval_sources") or [])
        if "adaptive" not in sources:
            sources.append("adaptive")
        enriched["retrieval_sources"] = sources
        merged.append(enriched)
    return merged


def adaptive_retrieval_enabled() -> bool:
    """总开关：默认关闭（回归零影响），ADAPTIVE_RETRIEVAL_ENABLED=1 开启。"""
    return os.getenv("ADAPTIVE_RETRIEVAL_ENABLED", "0").strip().lower() in {"1", "true", "yes"}


def adaptive_max_rounds() -> int:
    return max(1, int(os.getenv("ADAPTIVE_MAX_ROUNDS", "1")))
