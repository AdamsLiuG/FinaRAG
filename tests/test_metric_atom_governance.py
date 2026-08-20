"""Stage G 治理机制测试：provenance 刻度、有效期门禁（对齐 LLM-Wiki Claim 证据链）。"""

from datetime import date, timedelta
import tempfile
import unittest
from pathlib import Path

import dataclasses

from src.answer_validation import validate_answer
from src.metric_atoms import MetricAtom, MetricAtomStore, reset_default_store
from src.query_rewrite import QuestionRewriter


class ProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = MetricAtomStore.load()

    def test_extracted_annotation_readable(self):
        # MET-004 的取证词表从 _revenue_row_score 逐字迁移，等价测试佐证
        self.assertEqual(self.store.get("MET-004").provenance_of("grounding"), "extracted")
        # aliases 源自 query_rewrite 旧正则词表迁移
        self.assertEqual(self.store.get("MET-001").provenance_of("aliases"), "extracted")

    def test_unannotated_field_defaults_to_inferred(self):
        # fail-closed：未声明来源不得按高可信消费
        self.assertEqual(self.store.get("MET-001").provenance_of("definition"), "inferred")
        # 无 grounding 的原子，grounding 来源也按 inferred 处理
        self.assertEqual(self.store.get("MET-002").provenance_of("grounding"), "inferred")

    def test_invalid_provenance_value_rejected_at_load(self):
        bad_yaml = """
atoms:
  - atom_id: MET-001
    standard_name: 测试
    aliases: [测试]
    provenance: {aliases: hallucinated}
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            f.write(bad_yaml)
            path = Path(f.name)
        with self.assertRaises(ValueError) as ctx:
            MetricAtomStore.load(path)
        self.assertIn("hallucinated", str(ctx.exception))
        path.unlink()

    def test_all_atoms_have_verified_at_declared(self):
        # 存量已标注：无 verified_at 即回退为缺口
        for atom in self.store.all_atoms():
            self.assertTrue(atom.verified_at, msg=f"{atom.atom_id} 未声明 verified_at")


class ValidityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = MetricAtomStore.load()

    def _atom_with_expiry(self, expires_at: str) -> MetricAtom:
        return dataclasses.replace(self.store.get("MET-001"), expires_at=expires_at)

    def test_is_expired_boundaries(self):
        today = date.today()
        self.assertTrue(self._atom_with_expiry(today.isoformat()).is_expired(as_of=today))
        self.assertTrue(self._atom_with_expiry((today - timedelta(days=1)).isoformat()).is_expired(as_of=today))
        self.assertFalse(self._atom_with_expiry((today + timedelta(days=1)).isoformat()).is_expired(as_of=today))

    def test_undeclared_expiry_never_expires(self):
        # 未声明有效期不解读为"永不过期"——audit 会把它暴露为缺口
        self.assertFalse(self._atom_with_expiry("").is_expired())

    def test_validity_audit_exposes_undeclared_gaps(self):
        audit = self.store.validity_audit()
        self.assertEqual(len(audit), len(self.store))
        undeclared = [row for row in audit if row["undeclared_expiry"]]
        self.assertEqual(len(undeclared), 12)  # 存量如实：全部未声明有效期，是显式缺口

    def test_invalid_date_rejected_at_load(self):
        bad_yaml = """
atoms:
  - atom_id: MET-001
    standard_name: 测试
    aliases: [测试]
    expires_at: "2026/08/13"
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            f.write(bad_yaml)
            path = Path(f.name)
        with self.assertRaises(ValueError) as ctx:
            MetricAtomStore.load(path)
        self.assertIn("expires_at", str(ctx.exception))
        path.unlink()


class PayloadTests(unittest.TestCase):
    def tearDown(self):
        reset_default_store()

    def test_payload_carries_expired_flag(self):
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        store = MetricAtomStore.load()
        expired_atoms = [
            dataclasses.replace(a, expires_at=yesterday) if a.atom_id == "MET-001" else a
            for a in store.all_atoms()
        ]
        reset_default_store(MetricAtomStore(expired_atoms))
        plan = QuestionRewriter().rewrite(
            "宁德时代2024年扣非归母净利润是多少？", schema="number", company_name="宁德时代"
        )
        self.assertEqual(plan.metric_atom["atom_id"], "MET-001")
        self.assertTrue(plan.metric_atom["expired"])

    def test_payload_expired_false_when_undeclared(self):
        plan = QuestionRewriter().rewrite(
            "宁德时代2024年扣非归母净利润是多少？", schema="number", company_name="宁德时代"
        )
        self.assertFalse(plan.metric_atom["expired"])


class ExpiryGateTests(unittest.TestCase):
    """有效期门禁（温和语义）：标记 + 降级，不拒答——与口径门禁的严格语义刻意区分。"""

    def _answer(self):
        # 数字答案需带表格类 citation，否则 numeric_answer_without_structured_grounding
        # 门禁会额外降一级——本测试只考察有效期门禁本身
        return {
            "final_answer": "12345",
            "confidence": "high",
            "explanation": "x",
            "citations": [{"chunk_type": "serialized_table", "page": 5}],
            "relevant_pages": [5],
        }

    def _retrieval(self, text="扣除非经常性损益后的净利润为 12,345 万元。"):
        return [{"text": text, "score": 0.9, "metadata": {}}]

    def _plan(self, expired: bool):
        plan = QuestionRewriter().rewrite(
            "宁德时代2024年扣非归母净利润是多少？", schema="number", company_name="宁德时代"
        )
        plan.metric_atom = dict(plan.metric_atom or {})
        plan.metric_atom["expired"] = expired
        return plan

    def test_expired_atom_flags_and_downgrades(self):
        validated = validate_answer(self._answer(), self._retrieval(), self._plan(expired=True))
        self.assertIn("metric_atom_expired", validated.validation_flags)
        self.assertEqual(validated.confidence, "medium")  # high 降一级，非 low 拒答

    def test_unexpired_atom_has_no_flag(self):
        validated = validate_answer(self._answer(), self._retrieval(), self._plan(expired=False))
        self.assertNotIn("metric_atom_expired", validated.validation_flags)
        self.assertEqual(validated.confidence, "high")


if __name__ == "__main__":
    unittest.main()
