"""事件原子库（FinaWiki 事件层）：公司事实/主体类知识的识别规则资产。

与 MET-（指标口径）、PLB-（分析剧本）、flag（话题标签）正交：事件原子回答
「如何从文本识别某个公司事实」——识别模式（正则+置信度）、触发词、否定排除。
首个实例 EVT-001（现任法定代表人识别）自 questions_processing 硬编码迁移
（provenance: extracted，等价测试以 git 基线为准）。

运行时加载对齐 G-3 冻结 Generation：只读编译产物的 event_atoms 段并校验源同步；
yaml 是构建输入（compile_atoms.py 编译），源失步即拒绝加载。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from src.metric_atoms import _parse_date, _parse_provenance

DEFAULT_EVENTS_PATH = Path(__file__).resolve().parent.parent / "config" / "event_atoms.yaml"
DEFAULT_GENERATION_PATH = Path(__file__).resolve().parent.parent / "config" / "atom_snapshot.json"


@dataclass(frozen=True)
class DetectionPattern:
    """单条识别模式：编译后的正则 + 置信度基础分 + 模式标识。

    regex_source 保留原始字符串（与 git 基线逐字对照的等价性验收用）。
    """

    match_type: str
    score: float
    regex: re.Pattern
    regex_source: str


@dataclass(frozen=True)
class EventAtom:
    event_id: str
    event_name: str
    description: str = ""
    # 行级触发词：不含任一触发词的行跳过（正则只在触发行上尝试）
    trigger_terms: List[str] = field(default_factory=list)
    # 检测模式：按 score 降序尝试，首个命中决定 match_type 与基础分
    detection_patterns: List[DetectionPattern] = field(default_factory=list)
    # 否定排除：命中任一词的行不是现任表述（任职历史/变更过程）
    negative_terms: List[str] = field(default_factory=list)
    pitfalls: str = ""
    version: str = "1.0"
    provenance: Dict[str, str] = field(default_factory=dict)
    verified_at: str = ""
    expires_at: str = ""


class EventAtomStore:
    """事件原子库：加载 + 完整性校验 + 按 id 查询。"""

    def __init__(self, events: List[EventAtom]):
        if not events:
            raise ValueError("event atom store requires at least one event")
        self._events = list(events)
        self._by_id: Dict[str, EventAtom] = {}
        for event in self._events:
            if event.event_id in self._by_id:
                raise ValueError(f"duplicate event_id: {event.event_id}")
            if not event.detection_patterns:
                raise ValueError(f"{event.event_id}: detection_patterns 为空（事件原子必须有识别模式）")
            match_types = [p.match_type for p in event.detection_patterns]
            if len(match_types) != len(set(match_types)):
                raise ValueError(f"{event.event_id}: detection_patterns 存在重复 match_type")
            self._by_id[event.event_id] = event

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "EventAtomStore":
        """运行时加载：只读冻结 Generation 的 event_atoms 段（对齐 G-3）。

        - Generation 缺失或无 event_atoms 段 → 报错指路编译；
        - event_atoms.yaml 与 Generation 记录的 sha256 失步 → 拒绝加载。
        """
        generation_path = Path(path) if path else DEFAULT_GENERATION_PATH
        if not generation_path.exists():
            raise RuntimeError(
                f"冻结 Generation 不存在：{generation_path}\n"
                "请先运行 python scripts/compile_atoms.py 编译知识层"
            )
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        section = generation.get("event_atoms")
        if not section or not section.get("events"):
            raise RuntimeError(
                f"Generation 中无 event_atoms 段：{generation_path}\n"
                "请运行 python scripts/compile_atoms.py --update 纳入事件原子资产"
            )
        source_info = (generation.get("sources") or {}).get("event_atoms") or {}
        source_rel = str(source_info.get("path") or "config/event_atoms.yaml")
        source_path = generation_path.parent.parent / source_rel
        expected_sha = str(source_info.get("sha256") or "")
        if not source_path.exists():
            # 源缺失同样拒绝：静默跳过校验会让同步锁形同虚设
            raise RuntimeError(f"Generation 记录的源不存在：{source_path}")
        actual_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"源 {source_rel} 已变更但未重新编译（Generation 落后于源码）。\n"
                "请运行 python scripts/compile_atoms.py --update 确认变更并刷新 Generation。"
            )
        events = [
            _event_from_payload(item.get("event"))
            for item in section.get("events", {}).values()
            if item.get("event")
        ]
        if not events:
            raise RuntimeError(f"Generation event_atoms 段中无事件原子：{generation_path}")
        return cls(events)

    @classmethod
    def load_source(cls, path: Optional[Path] = None) -> "EventAtomStore":
        """源码直读（构建/编译工具用）；运行时路径请用 load()。"""
        path = path or DEFAULT_EVENTS_PATH
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        events = [_event_from_yaml(item) for item in (payload.get("events") or [])]
        return cls(events)

    def find_event(self, event_id: str) -> Optional[EventAtom]:
        return self._by_id.get(event_id)

    def all_events(self) -> List[EventAtom]:
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)


def _patterns_from_yaml(raw_items) -> List[DetectionPattern]:
    patterns: List[DetectionPattern] = []
    for item in raw_items or []:
        source = str(item.get("regex") or "")
        if not source:
            raise ValueError("detection_pattern 缺少 regex")
        try:
            compiled = re.compile(source)
        except re.error as exc:
            raise ValueError(f"detection_pattern 正则编译失败 {source!r}: {exc}") from exc
        patterns.append(
            DetectionPattern(
                match_type=str(item.get("match_type") or ""),
                score=float(item.get("score") or 0.0),
                regex=compiled,
                regex_source=source,
            )
        )
    return patterns


def _event_from_yaml(item: Dict) -> EventAtom:
    event_id = str(item["event_id"])
    return EventAtom(
        event_id=event_id,
        event_name=str(item["event_name"]),
        description=str(item.get("description") or ""),
        trigger_terms=[str(t) for t in (item.get("trigger_terms") or [])],
        detection_patterns=_patterns_from_yaml(item.get("detection_patterns")),
        negative_terms=[str(t) for t in (item.get("negative_terms") or [])],
        pitfalls=str(item.get("pitfalls") or ""),
        version=str(item.get("version") or "1.0"),
        provenance=_parse_provenance(item.get("provenance"), event_id),
        verified_at=_parse_date(item.get("verified_at"), "verified_at", event_id),
        expires_at=_parse_date(item.get("expires_at"), "expires_at", event_id),
    )


def _event_from_payload(data: Dict) -> EventAtom:
    """从 Generation JSON 反序列化（regex 由源串重新编译）。"""
    event_id = str(data["event_id"])
    return EventAtom(
        event_id=event_id,
        event_name=str(data["event_name"]),
        description=str(data.get("description") or ""),
        trigger_terms=[str(t) for t in (data.get("trigger_terms") or [])],
        detection_patterns=[
            DetectionPattern(
                match_type=str(p["match_type"]),
                score=float(p["score"]),
                regex=re.compile(p["regex_source"]),
                regex_source=str(p["regex_source"]),
            )
            for p in (data.get("detection_patterns") or [])
        ],
        negative_terms=[str(t) for t in (data.get("negative_terms") or [])],
        pitfalls=str(data.get("pitfalls") or ""),
        version=str(data.get("version") or "1.0"),
        provenance={str(k): str(v) for k, v in (data.get("provenance") or {}).items()},
        verified_at=str(data.get("verified_at") or ""),
        expires_at=str(data.get("expires_at") or ""),
    )


_DEFAULT_EVENT_STORE: Optional[EventAtomStore] = None


def get_default_event_store() -> EventAtomStore:
    global _DEFAULT_EVENT_STORE
    if _DEFAULT_EVENT_STORE is None:
        _DEFAULT_EVENT_STORE = EventAtomStore.load()
    return _DEFAULT_EVENT_STORE


def reset_default_event_store(store: Optional[EventAtomStore] = None) -> None:
    """测试隔离用：重置或替换默认事件原子库。"""
    global _DEFAULT_EVENT_STORE
    _DEFAULT_EVENT_STORE = store
