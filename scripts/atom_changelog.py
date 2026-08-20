"""原子知识变更清单 CLI：对照快照输出变更，--update 刷新快照。

用法：
    python scripts/atom_changelog.py            # 只报告变更（退出码 1 表示有未确认变更）
    python scripts/atom_changelog.py --update   # 确认变更并刷新 config/atom_snapshot.json

快照纳入版本控制：改了 metric_atoms.yaml 而不刷新快照，回归测试会失败——
知识变更因此必须在 code review 中显式出现。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metric_atom_versioning import (  # noqa: E402
    build_snapshot,
    diff_snapshot,
    load_snapshot,
    render_changelog,
    save_snapshot,
)
from src.metric_atoms import MetricAtomStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="指标原子知识变更清单")
    parser.add_argument("--update", action="store_true", help="确认变更并刷新快照")
    parser.add_argument("--snapshot", default=None, help="快照路径（默认 config/atom_snapshot.json）")
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot) if args.snapshot else None
    store = MetricAtomStore.load()
    changes = diff_snapshot(load_snapshot(snapshot_path), build_snapshot(store))
    print(render_changelog(changes, store), end="")

    if not any(changes.values()):
        sys.exit(0)
    if args.update:
        path = save_snapshot(store, snapshot_path)
        print(f"\n快照已刷新 -> {path}")
        sys.exit(0)
    print("\n未刷新快照（加 --update 确认变更）")
    sys.exit(1)


if __name__ == "__main__":
    main()
