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
from typing import Callable, Dict, List, Optional

from src.metric_atoms import MetricAtom, MetricAtomStore


TRIGGER_EVIDENCE_GAP = "evidence_gap"
TRIGGER_PLAYBOOK_INCOMPLETE = "playbook_incomplete"
TRIGGER_LOW_CONFIDENCE = "low_confidence"
TRIGGER_PERIOD_MISMATCH = "period_mismatch"
TRIGGER_CALIBER_MISMATCH = "caliber_mismatch"
TRIGGER_CLAIM_CONTRADICTED = "claim_contradicted"

STRATEGY_DENSE_RECALL = "dense_recall"
STRATEGY_PARENT_RECALL = "parent_recall"

ADAPTIVE_MIN_TOP_SCORE = float(os.getenv("ADAPTIVE_MIN_TOP_SCORE", "0.15"))
ADAPTIVE_MAX_SUPPLEMENT_QUERIES = 3


@dataclass(frozen=True)
class RetrievalGap:
    """一次补检的决策产物：触发原因 + 定向补检查询 + 依据 + 行动策略。"""

    trigger: str
    supplement_queries: List[str] = field(default_factory=list)
    rationale: str = ""
    strategy: str = STRATEGY_DENSE_RECALL


def _choose_strategy(round_index: int, prior_evidence_gain: bool) -> str:
    """③ 策略选择（规则版决策器；v3 换 LLM 时接口不变）。

    prior_evidence_gain=True（含初始默认——首次尝试视为有效假设）→ dense；
    False（上轮无新增）→ parent_recall（粒度切换是低置信的第二手段——
    child chunk 证据不全时回溯 page 级上下文）。

    注：不按 round_index 强制首轮 dense——切粒度重试轮不消耗 rounds_used
    （无新增不计轮），若按轮数判首轮会永远卡在 dense（S5 测试抓到）。
    round_index 参数保留给 v3 LLM 决策器。
    """
    if prior_evidence_gain:
        return STRATEGY_DENSE_RECALL
    return STRATEGY_PARENT_RECALL


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


POST_VALIDATE_TRIGGERS = {
    # 校验 flag → gap trigger（产生点：answer_validation.py / ground_generate.py）
    "metric_caliber_conflict_refused": TRIGGER_EVIDENCE_GAP,      # 口径门禁拒答
    "metric_evidence_mismatch": TRIGGER_EVIDENCE_GAP,
    "implicit_intent_evidence_incomplete": TRIGGER_PLAYBOOK_INCOMPLETE,
    "report_year_mismatch": TRIGGER_PERIOD_MISMATCH,
    "numeric_grounding_period_mismatch": TRIGGER_PERIOD_MISMATCH,
    "currency_mismatch": TRIGGER_CALIBER_MISMATCH,
    "numeric_grounding_currency_mismatch": TRIGGER_CALIBER_MISMATCH,
    "claim_contradicted": TRIGGER_CLAIM_CONTRADICTED,
}


def detect_post_validate_gap(
    validation_flags: List[str],
    query_plan,
    retrieval_results: List[Dict],
    atom_store: Optional[MetricAtomStore] = None,
    critic_fn: Optional[Callable[[List[str], object, List[Dict]], Optional[RetrievalGap]]] = None,
) -> Optional[RetrievalGap]:
    """校验后触发：flags 中首个可补救触发源 → RetrievalGap。

    构造逻辑复用预检：evidence_gap 走 _atom_supplement_queries（排除易混
    口径标准名——补的是本口径证据，caliber_conflict 场景即「定向补本口径」）；
    playbook_incomplete 走 payload.required_metrics 逐指标判定；
    period_mismatch / caliber_mismatch 走父粒度（parent_recall）重检；
    未命中规则时支持可选 v3 critic_fn 决策器扩展。
    """
    if atom_store is None:
        from src.metric_atoms import get_default_store
        atom_store = get_default_store()

    filters = getattr(query_plan, "filters", None)
    metric_atom_payload = getattr(query_plan, "metric_atom", None) or {}
    metric_name = getattr(query_plan, "metric", None) or metric_atom_payload.get("standard_name") or ""
    base_query = getattr(query_plan, "normalized_query", None) or getattr(query_plan, "original_query", None) or ""

    for flag in validation_flags or []:
        trigger = POST_VALIDATE_TRIGGERS.get(flag)
        if trigger is None:
            continue
        if trigger == TRIGGER_EVIDENCE_GAP:
            atom = atom_store.get(metric_atom_payload.get("atom_id") or "")
            if atom is None:
                continue  # 无原子命中：flags 不可由补检补救，不硬造 gap
            queries = _atom_supplement_queries(atom, ADAPTIVE_MAX_SUPPLEMENT_QUERIES)
            confusable = set(metric_atom_payload.get("confusable_names") or [])
            queries = [q for q in queries if q not in confusable]
            if not queries:
                continue
            return RetrievalGap(
                trigger=trigger,
                supplement_queries=queries,
                rationale=f"校验 flag={flag}；本口径证据定向补检",
            )
        if trigger == TRIGGER_PERIOD_MISMATCH:
            year = getattr(filters, "year", None)
            period = getattr(filters, "period", None)
            period_tokens = [str(tok) for tok in (period, f"{year}年" if year else None, str(year) if year else None) if tok]
            queries: List[str] = []
            for p_tok in period_tokens:
                candidate = " ".join(part for part in (p_tok, metric_name or base_query) if part).strip()
                if candidate and candidate not in queries:
                    queries.append(candidate)
            if queries:
                return RetrievalGap(
                    trigger=trigger,
                    supplement_queries=queries[:ADAPTIVE_MAX_SUPPLEMENT_QUERIES],
                    rationale=f"校验 flag={flag}；期间/年份不一致，切换父粒度定向补检",
                    strategy=STRATEGY_PARENT_RECALL,
                )
            continue
        if trigger == TRIGGER_CALIBER_MISMATCH:
            currency = getattr(filters, "currency", None) or "人民币"
            queries = []
            for anchor in (metric_name, base_query, "财务报表"):
                if not anchor:
                    continue
                candidate = f"{currency} {anchor}".strip()
                if candidate not in queries:
                    queries.append(candidate)
            if queries:
                return RetrievalGap(
                    trigger=trigger,
                    supplement_queries=queries[:ADAPTIVE_MAX_SUPPLEMENT_QUERIES],
                    rationale=f"校验 flag={flag}；币种/口径不一致，切换父粒度定向补检",
                    strategy=STRATEGY_PARENT_RECALL,
                )
            continue
        if trigger == TRIGGER_CLAIM_CONTRADICTED:
            queries = [q for q in (metric_name, base_query, "主要财务指标") if q]
            if queries:
                return RetrievalGap(
                    trigger=trigger,
                    supplement_queries=queries[:ADAPTIVE_MAX_SUPPLEMENT_QUERIES],
                    rationale=f"校验 flag={flag}；句级核验检测到断言冲突，切换父粒度补充原始上下文",
                    strategy=STRATEGY_PARENT_RECALL,
                )
            continue
        # TRIGGER_PLAYBOOK_INCOMPLETE
        playbook_payload = getattr(query_plan, "playbook", None) or {}
        corpus_text = _result_texts(retrieval_results)
        missing: List[str] = []
        for metric_atom_id in list(playbook_payload.get("required_metrics") or []):
            metric = atom_store.get(metric_atom_id)
            if metric is None:
                continue
            targets = [metric.standard_name] + metric.evidence_targets
            if len(_targets_covered(targets, corpus_text)) == len(targets):
                missing.append(metric.standard_name)
        if missing:
            return RetrievalGap(
                trigger=trigger,
                supplement_queries=missing[:ADAPTIVE_MAX_SUPPLEMENT_QUERIES + 1],
                rationale=f"校验 flag={flag}；剧本必需指标缺证据：{missing}",
            )

    if critic_fn is not None:
        return critic_fn(validation_flags, query_plan, retrieval_results)
    return None


def _merge_key(item: Dict) -> Tuple[str, object, str]:
    """去重键含粒度维度：child/parent 同页共存（parent 页级证据与 child
    chunk 证据不是重复信息）。scope 判定复用 retrieval.py 的 result_scope 语义。"""
    scope = (
        item.get("result_scope")
        or (item.get("metadata") or {}).get("node_type")
        or "child"
    )
    return (str(item.get("pdf_sha1") or ""), item.get("page"), str(scope))


def merge_supplement_results(
    base: List[Dict], supplement: List[Dict]
) -> List[Dict]:
    """补检结果合并进候选池：按 (pdf_sha1, page, scope) 去重，新结果标记 adaptive。"""
    seen = {_merge_key(item) for item in base or []}
    merged = list(base or [])
    for item in supplement or []:
        key = _merge_key(item)
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
