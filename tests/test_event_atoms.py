"""EVT- 事件原子层测试：结构校验、git 黄金基线逐字等价、行为等价、Generation 同步锁。

EVT-001（现任法定代表人识别）自 questions_processing 硬编码迁移（provenance: extracted）——
等价验收沿用 Stage C 模式：基线从 git 动态加载（迁移前 HEAD），不手抄。
"""

import re
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.event_atoms import (
    DetectionPattern,
    EventAtom,
    EventAtomStore,
    get_default_event_store,
    reset_default_event_store,
)

BASELINE_COMMIT = "HEAD"  # 迁移前状态（EVT 改造未提交）


def _load_baseline_constants():
    """从 git 提取迁移前的 _LEGAL_REP_* 常量块并 exec（仅依赖 re）。"""
    source = subprocess.run(
        ["git", "show", f"{BASELINE_COMMIT}:src/questions_processing.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    lines = source.splitlines()
    blocks = {}
    for name in ("_LEGAL_REP_NEGATIVE_TERMS", "_LEGAL_REP_PATTERNS"):
        start = next(i for i, ln in enumerate(lines) if ln.startswith(f"{name} = ("))
        end = next(i for i in range(start + 1, len(lines)) if lines[i] == ")")
        blocks[name] = "\n".join(lines[start : end + 1])
    namespace = {"re": re}
    exec(blocks["_LEGAL_REP_NEGATIVE_TERMS"], namespace)
    exec(blocks["_LEGAL_REP_PATTERNS"], namespace)
    return namespace["_LEGAL_REP_NEGATIVE_TERMS"], namespace["_LEGAL_REP_PATTERNS"]


class BaselineEquivalenceTests(unittest.TestCase):
    """逐字等价：事件原子内容与迁移前硬编码完全一致（防迁移错字/语义漂移）。"""

    @classmethod
    def setUpClass(cls):
        cls.negative, cls.patterns = _load_baseline_constants()
        cls.event = get_default_event_store().find_event("EVT-001")

    def test_negative_terms_match_baseline_exactly(self):
        self.assertEqual(list(self.event.negative_terms), list(self.negative))

    def test_trigger_terms_match_baseline_inline_tuple(self):
        # 旧代码的行触发词内联在 if 里：("法定代表人", "法人代表", "公司负责人")
        self.assertEqual(list(self.event.trigger_terms), ["法定代表人", "法人代表", "公司负责人"])

    def test_detection_patterns_match_baseline_exactly(self):
        self.assertEqual(len(self.event.detection_patterns), len(self.patterns))
        for detection, (compiled, score, match_type) in zip(
            self.event.detection_patterns, self.patterns
        ):
            self.assertEqual(detection.regex_source, compiled.pattern)  # 逐字符相等
            self.assertEqual(detection.score, score)
            self.assertEqual(detection.match_type, match_type)

    def test_behavioral_equivalence_on_representative_lines(self):
        cases = [
            "法定代表人：张伟明",
            "公司的法定代表人: 李晓明 | 总经理",
            "王建国  法定代表人",
            "公司负责人 赵小红 及声明",
            "法定代表人变更公告",
            "原法定代表人 钱多多",
            "历任法定代表人名单",
        ]
        for line in cases:
            old_results = []
            normalized = line.replace(" ", "")
            if not any(t in normalized for t in ("法定代表人", "法人代表", "公司负责人")):
                old_results = None
            elif any(t in normalized for t in self.negative):
                old_results = None
            else:
                for compiled, score, match_type in self.patterns:
                    match = compiled.search(normalized)
                    if match:
                        old_results = [(match.group(1), score, match_type)]
                        break
            new_results = None
            if any(t in normalized for t in self.event.trigger_terms) and not any(
                t in normalized for t in self.event.negative_terms
            ):
                for detection in self.event.detection_patterns:
                    match = detection.regex.search(normalized)
                    if match:
                        new_results = [(match.group(1), detection.score, detection.match_type)]
                        break
            self.assertEqual(old_results, new_results, msg=f"line={line!r}")


class StoreValidationTests(unittest.TestCase):
    def _event(self, **overrides):
        base = EventAtom(
            event_id="EVT-TEST",
            event_name="测试事件",
            detection_patterns=[
                DetectionPattern(match_type="primary", score=1.0, regex=re.compile("测试(.+)"), regex_source="测试(.+)")
            ],
        )
        return replace(base, **overrides) if overrides else base

    def test_empty_patterns_rejected(self):
        with self.assertRaises(ValueError):
            EventAtomStore([self._event(detection_patterns=[])])

    def test_duplicate_match_type_rejected(self):
        dup = DetectionPattern(match_type="primary", score=2.0, regex=re.compile("x"), regex_source="x")
        with self.assertRaises(ValueError):
            EventAtomStore([self._event(detection_patterns=[self._event().detection_patterns[0], dup])])

    def test_duplicate_event_id_rejected(self):
        with self.assertRaises(ValueError):
            EventAtomStore([self._event(), self._event()])

    def test_bad_regex_rejected_at_source_load(self):
        bad_yaml = """
events:
  - event_id: EVT-BAD
    event_name: 坏正则
    detection_patterns:
      - match_type: broken
        score: 1.0
        regex: '(unclosed'
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            f.write(bad_yaml)
            path = Path(f.name)
        with self.assertRaises(ValueError):
            EventAtomStore.load_source(path)
        path.unlink()

    def test_provenance_extracted_declared(self):
        event = get_default_event_store().find_event("EVT-001")
        self.assertEqual(event.provenance.get("detection_patterns"), "extracted")
        self.assertTrue(event.verified_at)


class GenerationSyncTests(unittest.TestCase):
    """事件资产的冻结 Generation 同步锁：改 yaml 不重编译即拒绝加载。"""

    def test_generation_load_equals_source_load(self):
        gen_store = EventAtomStore.load()
        src_store = EventAtomStore.load_source()
        self.assertEqual(len(gen_store), len(src_store))
        from src.metric_atom_versioning import event_content_hash

        for event in src_store.all_events():
            mirror = gen_store.find_event(event.event_id)
            self.assertIsNotNone(mirror, event.event_id)
            self.assertEqual(event_content_hash(mirror), event_content_hash(event), event.event_id)

    def test_stale_event_source_refused_at_load(self):
        # Generation 位于 <root>/config/ 下，源按相对仓库根路径解析——
        # 复刻该目录结构，使篡改的 sha 走到同步比较分支
        import json
        import shutil

        generation = json.loads(Path("config/atom_snapshot.json").read_text(encoding="utf-8"))
        tampered = dict(generation)
        tampered["sources"] = dict(generation["sources"])
        tampered["sources"]["event_atoms"] = {
            "path": "config/event_atoms.yaml",
            "sha256": "0" * 64,
        }
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "config"
            config_dir.mkdir()
            (config_dir / "atom_snapshot.json").write_text(
                json.dumps(tampered, ensure_ascii=False), encoding="utf-8"
            )
            shutil.copy("config/event_atoms.yaml", config_dir / "event_atoms.yaml")
            with self.assertRaises(RuntimeError) as ctx:
                EventAtomStore.load(config_dir / "atom_snapshot.json")
            self.assertIn("重新编译", str(ctx.exception))

    def test_missing_event_source_refused_at_load(self):
        # 源缺失同样拒绝（fail-closed）：静默跳过校验会让同步锁形同虚设
        import json

        generation = json.loads(Path("config/atom_snapshot.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            gen_path = Path(tmp) / "atom_snapshot.json"
            gen_path.write_text(json.dumps(generation, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(RuntimeError) as ctx:
                EventAtomStore.load(gen_path)
            self.assertIn("源不存在", str(ctx.exception))


class ConsumerIntegrationTests(unittest.TestCase):
    """消费端到端：_find_legal_representative 走 EVT-001（原硬编码路径已删除）。"""

    def _processor(self):
        from src.questions_processing import QuestionsProcessor

        return QuestionsProcessor.__new__(QuestionsProcessor)

    def test_find_legal_representative_uses_event_atom(self):
        processor = self._processor()
        document = {
            "content": {
                "pages": [
                    {"page": 1, "text": "公司法定代表人：陈志远\n联系电话 021-12345678"},
                    {"page": 2, "text": "历任法定代表人：刘旧任（2018-2020）"},
                ]
            }
        }
        match = processor._find_legal_representative(document)
        self.assertIsNotNone(match)
        self.assertEqual(match["name"], "陈志远")
        self.assertEqual(match["match_type"], "explicit_legal_representative")

    def test_hardcoded_constants_removed_from_source(self):
        # 迁移完整性：常量定义不得残留（docstring 中的迁移说明除外）
        source = Path("src/questions_processing.py").read_text(encoding="utf-8")
        self.assertNotIn("_LEGAL_REP_PATTERNS = (", source)
        self.assertNotIn("_LEGAL_REP_NEGATIVE_TERMS = (", source)


if __name__ == "__main__":
    unittest.main()
