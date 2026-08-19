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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from src.text_normalization import normalize_text

DEFAULT_ATOMS_PATH = Path(__file__).resolve().parent.parent / "config" / "metric_atoms.yaml"


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
    version: str = "1.0"

    @property
    def match_terms(self) -> List[str]:
        # standard_name 可能未出现在 aliases 中（如研发投入）；匹配词取两者并集，保序去重
        terms: List[str] = []
        for term in [self.standard_name, *self.aliases]:
            if term and term not in terms:
                terms.append(term)
        return terms


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
                version=str(item.get("version") or "1.0"),
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
