"""指标原子版本化：内容哈希 + 快照对照，使知识变更在 code review 中可见。

知识资产改了 YAML 却无人察觉，与代码改了没人 review 是同一类风险。做法：
- 对每个原子的**语义字段**（含 grounding）稳定序列化后取哈希，排除 version 自身
  （否则改 version 即改哈希，无法区分"内容变了"与"只是标了新版本"）；
- 快照文件纳入版本控制：YAML 改动必然伴随快照 diff，改了不刷新则回归测试失败。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.metric_atoms import MetricAtom, MetricAtomStore

DEFAULT_SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "config" / "atom_snapshot.json"

HASH_LENGTH = 12
# version 不参与哈希：它标记发布批次，不构成语义内容
_EXCLUDED_FIELDS = ("version",)


def _semantic_payload(atom: MetricAtom) -> Dict[str, Any]:
    payload = {
        key: value
        for key, value in asdict(atom).items()
        if key not in _EXCLUDED_FIELDS
    }
    return payload


def atom_content_hash(atom: MetricAtom) -> str:
    """语义内容哈希：字段顺序无关，version 变化不影响。"""
    canonical = json.dumps(
        _semantic_payload(atom), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:HASH_LENGTH]


def build_snapshot(store: MetricAtomStore) -> Dict[str, Any]:
    return {
        "atom_count": len(store),
        "atoms": {atom.atom_id: atom_content_hash(atom) for atom in store.all_atoms()},
    }


def load_snapshot(path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(path or DEFAULT_SNAPSHOT_PATH)
    if not path.exists():
        return {"atom_count": 0, "atoms": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_snapshot(store: MetricAtomStore, path: Optional[Path] = None) -> Path:
    path = Path(path or DEFAULT_SNAPSHOT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(build_snapshot(store), ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(payload + "\n", encoding="utf-8")
    return path


def diff_snapshot(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, List[str]]:
    """对照两份快照：返回新增/删除/内容变更的 atom_id。"""
    old_atoms = dict(old.get("atoms") or {})
    new_atoms = dict(new.get("atoms") or {})
    return {
        "added": sorted(set(new_atoms) - set(old_atoms)),
        "removed": sorted(set(old_atoms) - set(new_atoms)),
        "changed": sorted(
            atom_id
            for atom_id in set(old_atoms) & set(new_atoms)
            if old_atoms[atom_id] != new_atoms[atom_id]
        ),
    }


def snapshot_is_current(store: MetricAtomStore, path: Optional[Path] = None) -> bool:
    changes = diff_snapshot(load_snapshot(path), build_snapshot(store))
    return not any(changes.values())


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
    lines.append("内容变更需确认口径是否随之调整；确认后运行 scripts/atom_changelog.py --update 刷新快照。")
    return "\n".join(lines) + "\n"
