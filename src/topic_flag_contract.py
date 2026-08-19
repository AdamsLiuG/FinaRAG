"""topic_flags 命名空间契约（FinaWiki 标签层）。

三处共用同一命名空间却无任何断言，导致两个已发生的缺陷：
- 命名空间污染：打标侧按 ``has_``/``mentions_`` 前缀无条件收集，内部技术字段
  （如 ``has_pdfcrawl_metadata``）被误打成 topic flag；
- 能力缺口不可见：数据集有 17 个 flag，查询侧词表只覆盖 6 个，缺口无处体现。

本模块把命名空间提升为显式契约：非法 key 在 import 期暴露，收集范围按白名单过滤，
覆盖缺口可度量。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

DEFAULT_CONTRACT_PATH = Path(__file__).resolve().parent.parent / "config" / "topic_flag_contract.yaml"


@dataclass(frozen=True)
class FlagContract:
    flags: Set[str] = field(default_factory=set)
    query_side_flags: Set[str] = field(default_factory=set)
    reserved_non_flags: Set[str] = field(default_factory=set)
    descriptions: Dict[str, str] = field(default_factory=dict)
    version: str = "1.0"

    def is_flag(self, name: str) -> bool:
        return name in self.flags and name not in self.reserved_non_flags


def load_flag_contract(path: Optional[Path] = None) -> FlagContract:
    path = path or DEFAULT_CONTRACT_PATH
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    flags: Set[str] = set()
    query_side: Set[str] = set()
    descriptions: Dict[str, str] = {}
    for item in payload.get("flags") or []:
        name = str(item["name"])
        flags.add(name)
        if item.get("query_side"):
            query_side.add(name)
        descriptions[name] = str(item.get("description") or "")
    reserved = {str(name) for name in (payload.get("reserved_non_flags") or [])}
    overlap = flags & reserved
    if overlap:
        raise ValueError(f"flag declared both as topic flag and reserved: {sorted(overlap)}")
    if not flags:
        raise ValueError("topic flag contract declares no flags")
    return FlagContract(
        flags=flags,
        query_side_flags=query_side,
        reserved_non_flags=reserved,
        descriptions=descriptions,
        version=str(payload.get("version") or "1.0"),
    )


def validate_topic_patterns(
    patterns: Dict[str, Any],
    contract: Optional[FlagContract] = None,
) -> List[str]:
    """返回违规说明列表；空列表表示词表与契约一致。"""
    contract = contract or get_default_contract()
    violations: List[str] = []
    for key in patterns:
        if key in contract.reserved_non_flags:
            violations.append(f"{key}: reserved internal field must not be used as topic flag")
        elif key not in contract.flags:
            violations.append(f"{key}: not declared in topic flag contract")
    return violations


def assert_topic_patterns_valid(
    patterns: Dict[str, Any],
    contract: Optional[FlagContract] = None,
) -> None:
    violations = validate_topic_patterns(patterns, contract=contract)
    if violations:
        raise ValueError("topic flag contract violations: " + "; ".join(violations))


def filter_report_topic_flags(
    report_meta: Dict[str, Any],
    contract: Optional[FlagContract] = None,
) -> List[str]:
    """按契约白名单收集 topic flags（替代前缀匹配，避免内部字段污染）。"""
    contract = contract or get_default_contract()
    return sorted(
        key
        for key, value in (report_meta or {}).items()
        if contract.is_flag(str(key)) and str(value).strip().lower() == "true"
    )


def coverage_report(
    patterns: Dict[str, Any],
    contract: Optional[FlagContract] = None,
) -> Dict[str, Any]:
    """查询侧词表对数据集 flag 命名空间的覆盖情况（把缺口从不可见变为可度量）。"""
    contract = contract or get_default_contract()
    covered = sorted(key for key in patterns if key in contract.flags)
    missing = sorted(contract.flags - set(covered))
    return {
        "total_flags": len(contract.flags),
        "covered_flags": len(covered),
        "coverage_rate": round(len(covered) / len(contract.flags), 4) if contract.flags else None,
        "covered": covered,
        "missing": missing,
        "declared_query_side_without_patterns": sorted(contract.query_side_flags - set(covered)),
    }


_DEFAULT_CONTRACT: Optional[FlagContract] = None


def get_default_contract() -> FlagContract:
    global _DEFAULT_CONTRACT
    if _DEFAULT_CONTRACT is None:
        _DEFAULT_CONTRACT = load_flag_contract()
    return _DEFAULT_CONTRACT


def reset_default_contract(contract: Optional[FlagContract] = None) -> None:
    """测试隔离用：重置或替换默认契约。"""
    global _DEFAULT_CONTRACT
    _DEFAULT_CONTRACT = contract
