import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from src.metric_atom_versioning import (
    HASH_LENGTH,
    atom_content_hash,
    build_snapshot,
    diff_snapshot,
    load_snapshot,
    render_changelog,
    save_generation,
    snapshot_is_current,
)
from src.metric_atoms import MetricAtomStore, get_default_store


class ContentHashTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = get_default_store()
        cls.atom = cls.store.get("MET-004")

    def test_hash_is_stable_and_short(self):
        first = atom_content_hash(self.atom)
        second = atom_content_hash(self.atom)
        self.assertEqual(first, second)
        self.assertEqual(len(first), HASH_LENGTH)

    def test_hash_ignores_version_field(self):
        # version 标记发布批次，不构成语义内容
        bumped = dataclasses.replace(self.atom, version="9.9")
        self.assertEqual(atom_content_hash(bumped), atom_content_hash(self.atom))

    def test_semantic_change_alters_hash(self):
        changed = dataclasses.replace(self.atom, definition=self.atom.definition + "（修订）")
        self.assertNotEqual(atom_content_hash(changed), atom_content_hash(self.atom))

    def test_alias_order_change_alters_hash(self):
        # aliases 是有序列表（影响最长匹配与扩展顺序），顺序变化属语义变化
        reordered = dataclasses.replace(self.atom, aliases=list(reversed(self.atom.aliases)))
        self.assertNotEqual(atom_content_hash(reordered), atom_content_hash(self.atom))

    def test_grounding_change_alters_hash(self):
        # 取证规则属知识内容，改了必须体现在哈希上
        rule = self.atom.grounding
        self.assertIsNotNone(rule)
        tweaked_rule = dataclasses.replace(rule, segment_guard=not rule.segment_guard)
        tweaked = dataclasses.replace(self.atom, grounding=tweaked_rule)
        self.assertNotEqual(atom_content_hash(tweaked), atom_content_hash(self.atom))

    def test_atoms_have_distinct_hashes(self):
        hashes = [atom_content_hash(atom) for atom in self.store.all_atoms()]
        self.assertEqual(len(hashes), len(set(hashes)))


class SnapshotDiffTests(unittest.TestCase):
    def test_diff_detects_three_change_kinds(self):
        old = {"atoms": {"MET-001": "aaa", "MET-002": "bbb", "MET-003": "ccc"}}
        new = {"atoms": {"MET-002": "bbb", "MET-003": "zzz", "MET-004": "ddd"}}
        changes = diff_snapshot(old, new)
        self.assertEqual(changes["added"], ["MET-004"])
        self.assertEqual(changes["removed"], ["MET-001"])
        self.assertEqual(changes["changed"], ["MET-003"])

    def test_diff_empty_when_identical(self):
        snapshot = build_snapshot(get_default_store())
        self.assertEqual(
            diff_snapshot(snapshot, snapshot), {"added": [], "removed": [], "changed": []}
        )

    def test_missing_snapshot_file_reports_all_added(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "absent.json"
            store = get_default_store()
            changes = diff_snapshot(load_snapshot(path), build_snapshot(store))
            self.assertEqual(len(changes["added"]), len(store))

    def test_save_then_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.json"
            store = get_default_store()
            save_generation(store, path)
            self.assertTrue(snapshot_is_current(store, path))
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["atom_count"], len(store))

    def test_render_changelog(self):
        store = get_default_store()
        text = render_changelog({"added": ["MET-004"], "removed": [], "changed": ["MET-001"]}, store)
        self.assertIn("新增", text)
        self.assertIn("营业收入", text)
        self.assertIn("内容变更", text)
        self.assertIn("扣非归母净利润", text)
        self.assertIn("无变更", render_changelog({"added": [], "removed": [], "changed": []}, store))


class FrozenGenerationTests(unittest.TestCase):
    """G-3 冻结 Generation：运行时只读编译产物，源失步即拒绝加载（对齐决策一）。"""

    def test_generation_load_equals_source_load(self):
        gen_store = MetricAtomStore.load()
        src_store = MetricAtomStore.load_source()
        self.assertEqual(len(gen_store), len(src_store))
        for atom in src_store.all_atoms():
            mirror = gen_store.get(atom.atom_id)
            self.assertIsNotNone(mirror, atom.atom_id)
            self.assertEqual(atom_content_hash(mirror), atom_content_hash(atom), atom.atom_id)

    def test_generation_carries_governance_fields(self):
        generation = load_snapshot()
        self.assertEqual(generation["schema_version"], "1.1")
        payload = generation["atoms"]["MET-004"]["atom"]
        self.assertIn("provenance", payload)
        self.assertIn("verified_at", payload)
        self.assertEqual(payload["provenance"]["grounding"], "extracted")

    def test_stale_source_refused_at_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "metric_atoms.yaml"
            source.write_text(
                "atoms:\n  - atom_id: MET-001\n    standard_name: 测试\n    aliases: [测试]\n",
                encoding="utf-8",
            )
            store = MetricAtomStore.load_source(source)
            generation_path = root / "atom_snapshot.json"
            save_generation(store, path=generation_path, source_path=source)
            loaded = MetricAtomStore.load(generation_path)
            self.assertEqual(len(loaded), 1)
            # 源变更但不重编译 → 拒绝加载（fail-closed）
            source.write_text(
                "atoms:\n  - atom_id: MET-001\n    standard_name: 测试2\n    aliases: [测试]\n",
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError) as ctx:
                MetricAtomStore.load(generation_path)
            self.assertIn("重新编译", str(ctx.exception))

    def test_missing_generation_refused_at_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as ctx:
                MetricAtomStore.load(Path(tmp) / "absent.json")
            message = str(ctx.exception)
            self.assertIn("compile_atoms", message)


class SnapshotRegressionLockTests(unittest.TestCase):
    """回归锁：改了 metric_atoms.yaml 而未刷新快照即失败。"""

    def test_committed_snapshot_matches_current_library(self):
        store = MetricAtomStore.load()
        changes = diff_snapshot(load_snapshot(), build_snapshot(store))
        self.assertEqual(
            changes,
            {"added": [], "removed": [], "changed": []},
            msg="知识库与 config/atom_snapshot.json 不一致，"
            "请运行 python scripts/compile_atoms.py --update 确认变更并重新编译",
        )


if __name__ == "__main__":
    unittest.main()
