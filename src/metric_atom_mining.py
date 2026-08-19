"""指标原子扩容：LLM 生成草稿 + 规则校验 + 人工校准入口。

生产链路：多源素材（语料/会计准则）→ LLM 提炼原子草稿 → 规则质检（本模块 validate_atom_drafts）
    → 人工校准 → 合入 config/metric_atoms.yaml 正式入库。

校验器独立于 LLM 调用，可单测；LLM 调用需 API key（scripts/mine_metric_atoms.py CLI）。
质检与提炼应使用不同模型（生产者不自批）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.metric_atoms import MetricAtomStore

# 口径放宽词黑名单：aliases 中出现即拒绝（同指标层的口径内扩展原则）
_ALIAS_REJECT_TERMS = ("收入", "利润", "费用", "现金流")

MINE_SYSTEM_PROMPT = """你是金融指标知识工程专家。请从给定的语料/领域知识中提炼金融指标口径原子，
以 JSON 数组输出，每个原子包含以下字段：
- standard_name: 指标标准名（如"归母净利润"）
- aliases: 同口径别名列表（禁止口径放宽词，如"营业收入"不得挂裸"收入"）
- definition: 一句话口径定义
- caliber_notes: 易混淆口径的区分说明
- unit: 计量单位（元/百分比等）
- statement_location: 报表位置（合并利润表/合并资产负债表/合并现金流量表/主要财务指标等）
- confusable_with_standard_names: 易混淆指标的标准名列表（可为空）
- pitfalls: 常见误区一句话

只输出 JSON 数组，不要输出其他内容。"""

MINE_USER_PROMPT = """请提炼 {n_target} 个尚未包含在现有指标库中的金融指标口径原子。
现有指标库已覆盖（不要重复挖掘）：
{existing_names}

参考资料（可为空）：
{corpus_hint}"""


def validate_atom_draft(draft: Dict[str, Any], store: MetricAtomStore) -> List[str]:
    """校验单个原子草稿，返回错误列表（空列表=通过）。"""
    errors: List[str] = []
    standard_name = str(draft.get("standard_name") or "").strip()
    if not standard_name:
        errors.append("standard_name 为空")

    aliases = draft.get("aliases") or []
    if not isinstance(aliases, list) or not aliases:
        errors.append("aliases 为空或非列表")
        aliases = []
    for alias in aliases:
        alias_text = str(alias or "").strip()
        if not alias_text:
            errors.append(f"{standard_name}: 存在空 alias")
            continue
        # 口径放宽词检测：裸"收入/利润/费用/现金流"不允许作为 alias
        if alias_text in _ALIAS_REJECT_TERMS:
            errors.append(f"{standard_name}: alias '{alias_text}' 为口径放宽词，拒绝入库")

    if not str(draft.get("definition") or "").strip():
        errors.append(f"{standard_name}: definition 为空")

    if not (draft.get("evidence_targets") or []):
        # 校验口径的证据目标允许为空（入库时由人工补齐），但不给出警告则难以发现
        pass

    # 与现有原子的口径重叠检测：standard_name 或 aliases 不得与现有原子匹配词重叠
    if standard_name:
        for atom in store.all_atoms():
            overlap_terms = [t for t in atom.match_terms if t == standard_name or t in aliases]
            if overlap_terms:
                errors.append(
                    f"{standard_name}: 与现有原子 {atom.atom_id}({atom.standard_name}) "
                    f"存在口径重叠 {overlap_terms}"
                )
    return errors


def validate_atom_drafts(drafts: List[Dict[str, Any]], store: MetricAtomStore) -> Dict[str, List[str]]:
    """批量校验：返回 draft 索引 → 错误列表的映射。"""
    results: Dict[str, List[str]] = {}
    seen_names: set[str] = set()
    for idx, draft in enumerate(drafts):
        errors = validate_atom_draft(draft, store)
        name = str(draft.get("standard_name") or f"#{idx}")
        if name in seen_names:
            errors.append(f"{name}: 草稿内重复")
        seen_names.add(name)
        results[name] = errors
    return results


def render_draft_yaml(drafts: List[Dict[str, Any]], accepted: Optional[List[str]] = None) -> str:
    """渲染草稿 YAML（供人工校准后合入 metric_atoms.yaml）。"""
    accepted_set = set(accepted or [str(d.get("standard_name")) for d in drafts])
    lines = ["# 蒸馏草稿（待人工校准）：校准通过后合入 config/metric_atoms.yaml", "drafts:"]
    for idx, draft in enumerate(drafts):
        name = str(draft.get("standard_name") or "")
        if name not in accepted_set:
            continue
        lines.append(f"  - draft_id: DRAFT-{idx + 1:03d}")
        lines.append(f"    standard_name: {name}")
        aliases = "、".join(str(a) for a in (draft.get("aliases") or []))
        lines.append(f"    aliases: [{aliases}]")
        lines.append(f"    definition: {draft.get('definition') or ''}")
        lines.append(f"    caliber_notes: {draft.get('caliber_notes') or ''}")
        lines.append(f"    unit: {draft.get('unit') or ''}")
        lines.append(f"    statement_location: {draft.get('statement_location') or ''}")
        lines.append(f"    pitfalls: {draft.get('pitfalls') or ''}")
        lines.append(f"    confusable_with: {draft.get('confusable_with_standard_names') or []}")
    return "\n".join(lines) + "\n"


def mine_atoms_with_llm(api_processor, corpus_hint: str = "", n_target: int = 20) -> List[Dict[str, Any]]:
    """调用 LLM 生成原子草稿（需 API key 环境）。"""
    store = MetricAtomStore.load()
    existing_names = "、".join(atom.standard_name for atom in store.all_atoms())
    system_prompt = MINE_SYSTEM_PROMPT
    user_prompt = MINE_USER_PROMPT.format(
        n_target=n_target,
        existing_names=existing_names,
        corpus_hint=corpus_hint[:4000] or "（无，请基于金融领域通用知识提炼）",
    )
    answer = api_processor.send_message(
        system_content=system_prompt,
        human_content=user_prompt,
        is_structured=False,
    )
    text = answer if isinstance(answer, str) else str(answer)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM 返回中未找到 JSON 数组: {text[:200]}")
    import json

    return json.loads(text[start : end + 1])
