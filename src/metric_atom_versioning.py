"""指标原子版本化：冻结 Generation 编译 + 快照对照，使知识变更在 code review 中可见。

对齐 LLM-Wiki「编译而非检索」（决策一）与「三层 Registry 状态机」（决策三）的适配形态：
- yaml 是构建输入（源码），人编辑；
- Generation（config/atom_snapshot.json）是编译产物：完整原子序列化 + per-atom 内容哈希
  + 源 yaml sha256 + buildId，纳入版本控制；
- 运行时只加载 Generation 并校验源同步（见 MetricAtomStore.load），改 yaml 不重编译即启动失败
  ——知识变更因此必须在 code review 中显式出现。

哈希口径：对原子全部语义字段（含 grounding 与治理字段）稳定序列化取 sha256 前 12 位，
排除 version 自身（否则改 version 即改哈希，无法区分"内容变了"与"只是标了新版本"）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.event_atoms import EventAtom, EventAtomStore
from src.metric_atoms import MetricAtom, MetricAtomStore

DEFAULT_SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "config" / "atom_snapshot.json"
DEFAULT_SOURCE_PATH = Path(__file__).resolve().parent.parent / "config" / "metric_atoms.yaml"
DEFAULT_EVENT_SOURCE_PATH = Path(__file__).resolve().parent.parent / "config" / "event_atoms.yaml"

HASH_LENGTH = 12
GENERATION_SCHEMA_VERSION = "1.2"
# version 不参与哈希：它标记发布批次，不构成语义内容
_EXCLUDED_FIELDS = ("version",)


def _semantic_payload(atom: MetricAtom) -> Dict[str, Any]:
    return {key: value for key, value in asdict(atom).items() if key not in _EXCLUDED_FIELDS}


def _full_payload(atom: MetricAtom) -> Dict[str, Any]:
    """完整序列化（含 version）——Generation 存储用，保证编译产物可完整还原。"""
    return asdict(atom)


def atom_content_hash(atom: MetricAtom) -> str:
    """语义内容哈希：字段顺序无关，version 变化不影响。"""
    canonical = json.dumps(
        _semantic_payload(atom), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:HASH_LENGTH]


def hash_view(generation: Dict[str, Any]) -> Dict[str, str]:
    """Generation → {atom_id: hash} 视图（兼容旧快照格式 {id: hash}）。"""
    return {
        atom_id: (item["hash"] if isinstance(item, dict) else item)
        for atom_id, item in (generation.get("atoms") or {}).items()
    }


def build_generation(store: MetricAtomStore, source_path: Optional[Path] = None) -> Dict[str, Any]:
    """编译冻结 Generation：完整原子序列化 + 内容哈希 + 源指纹 + buildId。"""
    source_path = Path(source_path) if source_path else DEFAULT_SOURCE_PATH
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    build_id = "gen-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    # path 存相对仓库根的路径（MetricAtomStore.load 按 generation 文件位置上溯仓库根解析）
    if source_path.parent == DEFAULT_SNAPSHOT_PATH.parent:
        source_rel = f"config/{source_path.name}"
    else:
        source_rel = str(source_path)
    return {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "build_id": build_id,
        "source": {"path": source_rel, "sha256": source_sha},
        "atom_count": len(store),
        "atoms": {
            atom.atom_id: {"hash": atom_content_hash(atom), "atom": _full_payload(atom)}
            for atom in store.all_atoms()
        },
    }


def build_snapshot(store: MetricAtomStore) -> Dict[str, Any]:
    """hash 视图（diff/changelog 用）。"""
    return {"atom_count": len(store), "atoms": {a.atom_id: atom_content_hash(a) for a in store.all_atoms()}}


def _event_payload(event: EventAtom) -> Dict[str, Any]:
    """事件原子完整序列化（DetectionPattern → 可 JSON 化 dict，regex 以源串保存）。"""
    from dataclasses import asdict as _asdict

    payload = _asdict(event)
    payload["detection_patterns"] = [
        {"match_type": p.match_type, "score": p.score, "regex_source": p.regex_source}
        for p in event.detection_patterns
    ]
    return payload


def event_content_hash(event: EventAtom) -> str:
    """事件原子语义哈希（排除 version；regex 按源串参与，编译对象不参与）。"""
    payload = {k: v for k, v in _event_payload(event).items() if k != "version"}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:HASH_LENGTH]


def build_event_section(store: EventAtomStore, source_path: Optional[Path] = None) -> Dict[str, Any]:
    """编译事件资产段：完整序列化 + 内容哈希 + 源指纹。"""
    source_path = Path(source_path) if source_path else DEFAULT_EVENT_SOURCE_PATH
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return {
        "source": {"path": f"config/{source_path.name}", "sha256": source_sha},
        "event_count": len(store),
        "events": {
            e.event_id: {"hash": event_content_hash(e), "event": _event_payload(e)}
            for e in store.all_events()
        },
    }


def event_hash_view(section: Dict[str, Any]) -> Dict[str, str]:
    return {
        event_id: (item["hash"] if isinstance(item, dict) else item)
        for event_id, item in (section.get("events") or {}).items()
    }


def diff_event_section(old_section: Dict[str, Any], new_section: Dict[str, Any]) -> Dict[str, List[str]]:
    old_events = event_hash_view(old_section or {})
    new_events = event_hash_view(new_section or {})
    return {
        "added": sorted(set(new_events) - set(old_events)),
        "removed": sorted(set(old_events) - set(new_events)),
        "changed": sorted(
            e for e in set(old_events) & set(new_events) if old_events[e] != new_events[e]
        ),
    }


def load_snapshot(path: Optional[Path] = None) -> Dict[str, Any]:
    """读 Generation（或旧格式快照）；缺失返回空。"""
    path = Path(path or DEFAULT_SNAPSHOT_PATH)
    if not path.exists():
        return {"atom_count": 0, "atoms": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_generation(
    store: MetricAtomStore,
    path: Optional[Path] = None,
    source_path: Optional[Path] = None,
    event_store: Optional[EventAtomStore] = None,
    event_source_path: Optional[Path] = None,
) -> Path:
    """编译并落盘 Generation v1.2（指标 + 事件双资产，原子写入语义：整体重写）。"""
    path = Path(path or DEFAULT_SNAPSHOT_PATH)
    generation = build_generation(store, source_path)
    if event_store is None:
        event_store = EventAtomStore.load_source(event_source_path)
    event_section = build_event_section(event_store, event_source_path)
    generation["schema_version"] = GENERATION_SCHEMA_VERSION
    generation["sources"] = {
        "metric_atoms": generation.pop("source"),
        "event_atoms": event_section["source"],
    }
    generation["event_atoms"] = {
        key: value for key, value in event_section.items() if key != "source"
    }
    payload = json.dumps(generation, ensure_ascii=False, indent=2, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n", encoding="utf-8")
    return path


def diff_snapshot(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, List[str]]:
    """对照两份 Generation（或 hash 视图）：返回新增/删除/内容变更的 atom_id。"""
    old_atoms = hash_view(old)
    new_atoms = hash_view(new)
    return {
        "added": sorted(set(new_atoms) - set(old_atoms)),
        "removed": sorted(set(old_atoms) - set(new_atoms)),
        "changed": sorted(
            atom_id for atom_id in set(old_atoms) & set(new_atoms) if old_atoms[atom_id] != new_atoms[atom_id]
        ),
    }


def snapshot_is_current(store: MetricAtomStore, path: Optional[Path] = None) -> bool:
    return not any(diff_snapshot(load_snapshot(path), build_snapshot(store)).values())


def render_changelog(changes: Dict[str, List[str]], store: MetricAtomStore) -> str:
    def _names(atom_ids: List[str]) -> List[str]:
        rendered = []
        for atom_id in atom_ids:
            atom = store.get(atom_id)
            rendered.append(f"{atom_id}({atom.standard_name})" if atom else atom_id)
        return rendered

    if not any(changes.values()):
        return "知识库无变更（快照与当前库一致）。\n"
    lines = ["# 指标原子变更清单", ""]
    for key, label in (("added", "新增"), ("removed", "删除"), ("changed", "内容变更")):
        items = changes.get(key) or []
        if items:
            lines.append(f"- **{label}**（{len(items)}）：{'、'.join(_names(items))}")
    lines.append("")
    lines.append("内容变更需确认口径是否随之调整；确认后运行 scripts/compile_atoms.py 重新编译 Generation。")
    return "\n".join(lines) + "\n"
