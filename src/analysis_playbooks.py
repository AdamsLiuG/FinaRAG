"""分析剧本库（FinaWiki 取证规划层）。

把"某类分析问题需要看哪些指标、依据什么判断"从代码内字符串列表（原 _IMPLICIT_INTENT_PATTERNS）
提升为可引用、可校验的知识资产（config/analysis_playbooks.yaml）。

两条结构性约束：
- 剧本必须引用原子：required_metrics 只能填合法 MET- ID，加载时校验可解析，否则启动即失败
  ——分析方法不得脱离指标事实基础；
- 双粒度展开：同一份知识按环节取不同粒度——检索用各原子 standard_name（精确简短，避免超长查询），
  校验用各原子完整 evidence_targets（宽松，覆盖别名与所在报表）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from src.metric_atoms import MetricAtomStore, get_default_store
from src.text_normalization import dedupe_preserve_order, normalize_text

DEFAULT_PLAYBOOKS_PATH = Path(__file__).resolve().parent.parent / "config" / "analysis_playbooks.yaml"


@dataclass(frozen=True)
class AnalysisPlaybook:
    playbook_id: str
    intent: str
    question_forms: List[str] = field(default_factory=list)
    required_metrics: List[str] = field(default_factory=list)
    evidence_keywords: List[str] = field(default_factory=list)
    judgment: str = ""
    pitfalls: str = ""
    version: str = "1.0"


class PlaybookStore:
    """剧本库：YAML 加载 + 原子引用校验 + 最长匹配查询 + 双粒度证据展开。"""

    def __init__(self, playbooks: List[AnalysisPlaybook], atom_store: Optional[MetricAtomStore] = None):
        if not playbooks:
            raise ValueError("playbook store requires at least one playbook")
        self._atom_store = atom_store or get_default_store()
        self._playbooks = list(playbooks)
        self._by_id: Dict[str, AnalysisPlaybook] = {}
        self._by_intent: Dict[str, AnalysisPlaybook] = {}
        for playbook in self._playbooks:
            if playbook.playbook_id in self._by_id:
                raise ValueError(f"duplicate playbook_id: {playbook.playbook_id}")
            if playbook.intent in self._by_intent:
                raise ValueError(f"duplicate intent: {playbook.intent}")
            self._by_id[playbook.playbook_id] = playbook
            self._by_intent[playbook.intent] = playbook
            # 剧本→原子引用约束：非法引用在加载期暴露，而非运行期静默降级
            for atom_id in playbook.required_metrics:
                if self._atom_store.get(atom_id) is None:
                    raise ValueError(f"{playbook.playbook_id} references unknown metric atom: {atom_id}")
            if not playbook.required_metrics and not playbook.evidence_keywords:
                raise ValueError(f"{playbook.playbook_id} has neither required_metrics nor evidence_keywords")
        self._match_index: List[Tuple[str, AnalysisPlaybook]] = sorted(
            (
                (normalize_text(form), playbook)
                for playbook in self._playbooks
                for form in playbook.question_forms
            ),
            key=lambda pair: len(pair[0]),
            reverse=True,
        )

    @classmethod
    def load(
        cls,
        path: Optional[Path] = None,
        atom_store: Optional[MetricAtomStore] = None,
    ) -> "PlaybookStore":
        path = path or DEFAULT_PLAYBOOKS_PATH
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        playbooks = [
            AnalysisPlaybook(
                playbook_id=str(item["playbook_id"]),
                intent=str(item["intent"]),
                question_forms=list(item.get("question_forms") or []),
                required_metrics=list(item.get("required_metrics") or []),
                evidence_keywords=list(item.get("evidence_keywords") or []),
                judgment=str(item.get("judgment") or ""),
                pitfalls=str(item.get("pitfalls") or ""),
                version=str(item.get("version") or "1.0"),
            )
            for item in (payload.get("playbooks") or [])
        ]
        return cls(playbooks, atom_store=atom_store)

    def find_playbook(self, normalized_question: str) -> Optional[AnalysisPlaybook]:
        for form, playbook in self._match_index:
            if form and form in normalized_question:
                return playbook
        return None

    def metric_names(self, playbook: AnalysisPlaybook) -> List[str]:
        """必需指标的标准名列表（检索粒度）。"""
        names = []
        for atom_id in playbook.required_metrics:
            atom = self._atom_store.get(atom_id)
            if atom is not None:
                names.append(atom.standard_name)
        return names

    def search_targets(self, playbook: AnalysisPlaybook) -> List[str]:
        """检索粒度证据词：标准名 + 叙述性关键词（保持查询精确简短）。"""
        return dedupe_preserve_order([*self.metric_names(playbook), *playbook.evidence_keywords])

    def validation_targets(self, playbook: AnalysisPlaybook) -> List[str]:
        """校验粒度证据词：各原子完整 evidence_targets + 关键词（宽松覆盖别名与报表名）。"""
        targets: List[str] = []
        for atom_id in playbook.required_metrics:
            atom = self._atom_store.get(atom_id)
            if atom is None:
                continue
            targets.append(atom.standard_name)
            targets.extend(atom.evidence_targets or [])
        targets.extend(playbook.evidence_keywords)
        return dedupe_preserve_order(targets)

    def metric_coverage(self, playbook: AnalysisPlaybook, evidence_blob: str) -> Tuple[int, int]:
        """必需指标覆盖度：返回 (命中指标数, 必需指标总数)。"""
        total = len(playbook.required_metrics)
        if not total:
            return 0, 0
        normalized_blob = normalize_text(evidence_blob or "")
        hits = 0
        for atom_id in playbook.required_metrics:
            atom = self._atom_store.get(atom_id)
            if atom is None:
                continue
            atom_terms = [atom.standard_name, *(atom.evidence_targets or [])]
            if any(normalize_text(term) in normalized_blob for term in atom_terms if len(str(term)) >= 2):
                hits += 1
        return hits, total

    def get(self, playbook_id: str) -> Optional[AnalysisPlaybook]:
        return self._by_id.get(playbook_id)

    def get_by_intent(self, intent: str) -> Optional[AnalysisPlaybook]:
        return self._by_intent.get(intent)

    def all_playbooks(self) -> List[AnalysisPlaybook]:
        return list(self._playbooks)

    def __len__(self) -> int:
        return len(self._playbooks)


def build_playbook_payload(playbook: AnalysisPlaybook, store: PlaybookStore) -> Dict[str, object]:
    """QueryPlan 携带的剧本快照（可序列化）。"""
    return {
        "playbook_id": playbook.playbook_id,
        "intent": playbook.intent,
        "required_metrics": list(playbook.required_metrics),
        "metric_names": store.metric_names(playbook),
        "judgment": playbook.judgment,
        "pitfalls": playbook.pitfalls,
    }


def build_playbook_hint(payload: Dict[str, object]) -> Optional[str]:
    """生成阶段判断依据注入文本（按需注入，未命中剧本时零成本）。"""
    if not payload or not payload.get("playbook_id"):
        return None
    parts = ["【分析判断依据】"]
    metric_names = payload.get("metric_names") or []
    if metric_names:
        parts.append(f"须核对指标：{'、'.join(str(name) for name in metric_names)}。")
    if payload.get("judgment"):
        parts.append(f"判断标准：{payload['judgment']}。")
    if payload.get("pitfalls"):
        parts.append(f"常见误区：{payload['pitfalls']}。")
    parts.append("若上下文缺少上述指标，请在答案中显式说明缺失项，不得据不完整证据给出确定结论。")
    return "".join(parts)


_DEFAULT_PLAYBOOK_STORE: Optional[PlaybookStore] = None


def get_default_playbook_store() -> PlaybookStore:
    global _DEFAULT_PLAYBOOK_STORE
    if _DEFAULT_PLAYBOOK_STORE is None:
        _DEFAULT_PLAYBOOK_STORE = PlaybookStore.load()
    return _DEFAULT_PLAYBOOK_STORE


def reset_default_playbook_store(store: Optional[PlaybookStore] = None) -> None:
    """测试隔离用：重置或替换默认剧本库。"""
    global _DEFAULT_PLAYBOOK_STORE
    _DEFAULT_PLAYBOOK_STORE = store
