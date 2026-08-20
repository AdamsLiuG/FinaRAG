"""知识层编译 CLI：yaml 源码 → 冻结 Generation（对齐 LLM-Wiki 决策一：编译而非检索）。

用法：
    python scripts/compile_atoms.py            # 报告变更；有未编译变更时退出码 1
    python scripts/compile_atoms.py --update   # 确认变更并编译落盘 config/atom_snapshot.json

运行时（MetricAtomStore.load）只读 Generation 并校验源同步；改 yaml 不重编译即启动失败。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metric_atom_versioning import (  # noqa: E402
    GENERATION_SCHEMA_VERSION,
    build_snapshot,
    diff_snapshot,
    load_snapshot,
    render_changelog,
    save_generation,
)
from src.metric_atoms import MetricAtomStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="编译指标知识层冻结 Generation")
    parser.add_argument("--update", action="store_true", help="确认变更并编译落盘")
    parser.add_argument("--generation", default=None, help="Generation 路径（默认 config/atom_snapshot.json）")
    parser.add_argument("--source", default=None, help="源 yaml 路径（默认 config/metric_atoms.yaml）")
    args = parser.parse_args()

    generation_path = Path(args.generation) if args.generation else None
    source_path = Path(args.source) if args.source else None
    store = MetricAtomStore.load_source(source_path)
    existing = load_snapshot(generation_path)
    changes = diff_snapshot(existing, build_snapshot(store))
    print(render_changelog(changes, store), end="")

    # 格式代差（旧快照无 schema_version）本身就是需要重编译的变更
    stale_format = existing.get("schema_version") != GENERATION_SCHEMA_VERSION
    if stale_format:
        print(f"Generation 格式代差（{existing.get('schema_version') or '旧格式'} -> {GENERATION_SCHEMA_VERSION}），需重编译。")

    if not any(changes.values()) and not stale_format:
        print("Generation 与源一致，无需编译。")
        sys.exit(0)
    if args.update:
        path = save_generation(store, path=generation_path, source_path=source_path)
        print(f"\nGeneration 已编译 -> {path}")
        sys.exit(0)
    print("\n未编译（加 --update 确认变更）")
    sys.exit(1)


if __name__ == "__main__":
    main()
