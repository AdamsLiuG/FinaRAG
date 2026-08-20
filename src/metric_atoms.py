"""金融指标口径原子库（FinaWiki 指标层）。

知识资产（config/metric_atoms.yaml）驱动，替代散落在 query_rewrite.py 的硬编码指标模式。
设计要点：
- 最长匹配：alias 间存在子串包含关系（扣非归母净利润 ⊃ 归母净利润），按词长降序扫描保证
  正确命中，不再依赖配置书写顺序；
- 口径内扩展：aliases 仅收同口径表述，扩展查询严禁放宽（如"营业收入"不得扩展为"收入"）；
- 一库多用：同一原子库驱动意图解析、检索扩展，并作为生成注入与答案校验的知识来源。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from src.text_normalization import normalize_text

DEFAULT_ATOMS_PATH = Path(__file__).resolve().parent.parent / "config" / "metric_atoms.yaml"


@dataclass(frozen=True)
class GroundingRule:
    """取证规则（FinaWiki 证据定位层）：指标专属的行列特征与报表偏好。

    匹配语义存意区分，不得混用：
    - ``exact_row_headers`` / ``secondary_row_headers``：**精确相等**（行首完全一致）；
    - ``weak_row_headers`` / ``low_priority_row_headers`` / ``bad_*``：**子串包含**。
    """

    match_terms: List[str] = field(default_factory=list)
    exact_row_headers: List[str] = field(default_factory=list)
    secondary_row_headers: List[str] = field(default_factory=list)
    weak_row_headers: List[str] = field(default_factory=list)
    low_priority_row_headers: List[str] = field(default_factory=list)
    bad_row_headers: List[str] = field(default_factory=list)
    bad_col_headers: List[str] = field(default_factory=list)
    preferred_statements: List[Tuple[List[str], float]] = field(default_factory=list)
    penalized_statements: List[Tuple[List[str], float]] = field(default_factory=list)
    segment_guard: bool = False
    # 嵌套行加成：指标作为上级科目子项出现时的行首模式 (row_prefix, requires_context, score)
    nested_row_bonus: List[Tuple[str, str, float]] = field(default_factory=list)
    # 页码先验 (max_page, score)：仅对确实出现在报告前部摘要的指标配置，缺省不加分
    early_page_bonus: List[Tuple[int, float]] = field(default_factory=list)
    # 证据交叉验证：是否收集支撑候选（同一数值的不同出处互证）
    collect_support_candidates: bool = False


def _parse_statement_scores(raw: object) -> List[Tuple[List[str], float]]:
    parsed: List[Tuple[List[str], float]] = []
    for item in raw or []:
        terms = [str(term) for term in (item.get("terms") or []) if str(term).strip()]
        if terms:
            parsed.append((terms, float(item.get("score") or 0.0)))
    return parsed


def _parse_provenance(raw: object, atom_id: str) -> Dict[str, str]:
    """解析 per-field provenance 标注；非法值在加载期暴露（fail-fast）。"""
    parsed: Dict[str, str] = {}
    for key, value in (raw or {}).items():
        value_str = str(value).strip()
        if value_str not in ("extracted", "inferred"):
            raise ValueError(
                f"{atom_id}: provenance['{key}'] 非法值 {value_str!r}（仅允许 extracted/inferred）"
            )
        parsed[str(key)] = value_str
    return parsed


def _parse_date(raw: object, field_name: str, atom_id: str) -> str:
    """解析 ISO 日期（YYYY-MM-DD）；空值返回空串（未声明）。"""
    text = str(raw or "").strip()
    if not text:
        return ""
    try:
        date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{atom_id}: {field_name} 非法日期 {text!r}（需 YYYY-MM-DD）") from exc
    return text


def _parse_grounding(raw: object) -> Optional[GroundingRule]:
    if not raw:
        return None
    return GroundingRule(
        match_terms=[str(term) for term in (raw.get("match_terms") or [])],
        exact_row_headers=[str(term) for term in (raw.get("exact_row_headers") or [])],
        secondary_row_headers=[str(term) for term in (raw.get("secondary_row_headers") or [])],
        weak_row_headers=[str(term) for term in (raw.get("weak_row_headers") or [])],
        low_priority_row_headers=[str(term) for term in (raw.get("low_priority_row_headers") or [])],
        bad_row_headers=[str(term) for term in (raw.get("bad_row_headers") or [])],
        bad_col_headers=[str(term) for term in (raw.get("bad_col_headers") or [])],
        preferred_statements=_parse_statement_scores(raw.get("preferred_statements")),
        penalized_statements=_parse_statement_scores(raw.get("penalized_statements")),
        segment_guard=bool(raw.get("segment_guard")),
        nested_row_bonus=[
            (
                str(item.get("row_prefix") or ""),
                str(item.get("requires_context") or ""),
                float(item.get("score") or 0.0),
            )
            for item in (raw.get("nested_row_bonus") or [])
            if item.get("row_prefix")
        ],
        early_page_bonus=[
            (int(item["max_page"]), float(item.get("score") or 0.0))
            for item in (raw.get("early_page_bonus") or [])
            if item.get("max_page") is not None
        ],
        collect_support_candidates=bool(raw.get("collect_support_candidates")),
    )


@dataclass(frozen=True)
class MetricAtom:
    atom_id: str
    standard_name: str
    aliases: List[str] = field(default_factory=list)
    definition: str = ""
    caliber_notes: str = ""
    unit: str = ""
    statement_location: str = ""
    confusable_with: List[str] = field(default_factory=list)
    pitfalls: str = ""
    evidence_targets: List[str] = field(default_factory=list)
    grounding: Optional[GroundingRule] = None
    version: str = "1.0"
    # 治理字段（Stage G，对齐 LLM-Wiki Claim 证据链）：
    # provenance 按字段声明知识来源——extracted（从生产代码迁移，有等价测试佐证）/ inferred（手写知识，
    # 未经语义验证）；缺省按 inferred 处理（未声明来源不得按高可信消费）。
    provenance: Dict[str, str] = field(default_factory=dict)
    # verified_at/expires_at 为 ISO 日期（YYYY-MM-DD）；空串 = 未声明，是显式缺口（validity_audit 可见），
    # 不解读为"永不过期"。
    verified_at: str = ""
    expires_at: str = ""

    _PROVENANCE_VALUES = ("extracted", "inferred")

    @property
    def match_terms(self) -> List[str]:
        # standard_name 可能未出现在 aliases 中（如研发投入）；匹配词取两者并集，保序去重
        terms: List[str] = []
        for term in [self.standard_name, *self.aliases]:
            if term and term not in terms:
                terms.append(term)
        return terms

    def provenance_of(self, field_name: str) -> str:
        """字段来源；未标注时返回 inferred（fail-closed：未知来源按低可信处理）。"""
        value = self.provenance.get(field_name)
        return value if value in self._PROVENANCE_VALUES else "inferred"

    def is_expired(self, as_of: Optional[date] = None) -> bool:
        """expires_at 已声明且不晚于 as_of 时为 True；未声明有效期不视为过期（但 audit 会暴露缺口）。"""
        if not self.expires_at:
            return False
        return date.fromisoformat(self.expires_at) <= (as_of or date.today())


class MetricAtomStore:
    """指标原子库：YAML 加载 + 完整性校验 + 最长匹配查询。"""

    def __init__(self, atoms: List[MetricAtom]):
        if not atoms:
            raise ValueError("metric atom store requires at least one atom")
        self._atoms = list(atoms)
        self._by_id: Dict[str, MetricAtom] = {}
        for atom in self._atoms:
            if atom.atom_id in self._by_id:
                raise ValueError(f"duplicate atom_id: {atom.atom_id}")
            self._by_id[atom.atom_id] = atom
        for atom in self._atoms:
            for ref in atom.confusable_with:
                if ref not in self._by_id:
                    raise ValueError(f"{atom.atom_id} references unknown atom: {ref}")
        # 最长匹配索引：子串包含关系下（扣非归母净利润 ⊃ 归母净利润），长词必须优先
        self._match_index: List[Tuple[str, MetricAtom]] = sorted(
            (
                (normalize_text(term), atom)
                for atom in self._atoms
                for term in atom.match_terms
            ),
            key=lambda pair: len(pair[0]),
            reverse=True,
        )

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "MetricAtomStore":
        path = path or DEFAULT_ATOMS_PATH
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        atoms = [
            MetricAtom(
                atom_id=str(item["atom_id"]),
                standard_name=str(item["standard_name"]),
                aliases=list(item.get("aliases") or []),
                definition=str(item.get("definition") or ""),
                caliber_notes=str(item.get("caliber_notes") or ""),
                unit=str(item.get("unit") or ""),
                statement_location=str(item.get("statement_location") or ""),
                confusable_with=list(item.get("confusable_with") or []),
                pitfalls=str(item.get("pitfalls") or ""),
                evidence_targets=list(item.get("evidence_targets") or []),
                grounding=_parse_grounding(item.get("grounding")),
                version=str(item.get("version") or "1.0"),
                provenance=_parse_provenance(item.get("provenance"), str(item["atom_id"])),
                verified_at=_parse_date(item.get("verified_at"), "verified_at", str(item["atom_id"])),
                expires_at=_parse_date(item.get("expires_at"), "expires_at", str(item["atom_id"])),
            )
            for item in (payload.get("atoms") or [])
        ]
        return cls(atoms)

    def find_metric(self, normalized_question: str) -> Optional[MetricAtom]:
        match = self.find_metric_with_term(normalized_question)
        return match[0] if match else None

    def find_metric_with_term(self, normalized_question: str) -> Optional[Tuple[MetricAtom, str]]:
        for term, atom in self._match_index:
            if term and term in normalized_question:
                return atom, term
        return None

    def grounding_for(self, question: str) -> Optional[Tuple[MetricAtom, GroundingRule]]:
        """问题命中且该原子带取证规则时返回 (原子, 规则)。"""
        atom = self.find_metric(normalize_text(question))
        if atom is None or atom.grounding is None:
            return None
        return atom, atom.grounding

    def validity_audit(self) -> List[Dict[str, object]]:
        """有效期审计：未声明 verified_at/expires_at 是显式缺口，不解读为"永不过期"。"""
        today = date.today()
        return [
            {
                "atom_id": atom.atom_id,
                "standard_name": atom.standard_name,
                "verified_at": atom.verified_at or None,
                "expires_at": atom.expires_at or None,
                "expired": atom.is_expired(as_of=today),
                "undeclared_expiry": not atom.expires_at,
            }
            for atom in self._atoms
        ]

    def get(self, atom_id: str) -> Optional[MetricAtom]:
        return self._by_id.get(atom_id)

    def all_atoms(self) -> List[MetricAtom]:
        return list(self._atoms)

    def __len__(self) -> int:
        return len(self._atoms)


def build_caliber_hint(atom: MetricAtom, store: MetricAtomStore) -> str:
    """生成阶段口径注入文本（追加至 system prompt 尾部）。

    按需注入原则：仅 QueryPlan 命中指标原子时由调用方生成，未命中时零 token 成本。
    消歧分支：confusable 原子存在时，提示模型按证据实际口径区分作答。
    """
    parts = [f"【指标口径提示】{atom.standard_name}：{atom.definition}"]
    if atom.caliber_notes:
        parts.append(atom.caliber_notes.rstrip("。") + "。")
    confusable_names = [
        conf.standard_name
        for ref in atom.confusable_with
        if (conf := store.get(ref)) is not None
    ]
    if confusable_names:
        parts.append(
            f"易混淆口径：{'、'.join(confusable_names)}。若上下文证据为上述其他口径而非 "
            f"{atom.standard_name}，请在 answer 中标注实际口径或返回 N/A，禁止混用口径作答。"
        )
    return "".join(parts)


_DEFAULT_STORE: Optional[MetricAtomStore] = None


def get_default_store() -> MetricAtomStore:
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = MetricAtomStore.load()
    return _DEFAULT_STORE


def reset_default_store(store: Optional[MetricAtomStore] = None) -> None:
    """测试隔离用：重置或替换默认原子库。"""
    global _DEFAULT_STORE
    _DEFAULT_STORE = store
