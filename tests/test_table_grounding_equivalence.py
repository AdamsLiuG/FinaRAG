"""营收取证路径的黄金基线等价测试。

Stage C 把 `table_grounding` 内的营收专用硬编码规则拆成「通用守卫 + 原子专属词表」。
本阶段应有的验收线是 number 类问题端到端准确率不回退，但当前环境无 API key 与数据集，
该验收无法执行。替代方案：把改造前的实现作为黄金基线，断言改造后对同样输入**同分**。

基线实现**从 git 动态加载**（`git show <commit>:src/table_grounding.py` + 独立命名空间 exec），
不手抄复述——手抄本身就是错字来源（迁移过程中确实发生过「合并利润表」被写成「合并利润表」，
导致规则静默失效）。

局限（必须如实声明）：等价性仅证明「营收路径未改坏」，不能证明其余 11 个指标获得的通用守卫
带来了准确率提升——那属于结构性改进，未经端到端验证。
"""

import subprocess
import types
import unittest

from src import table_grounding as current

# Stage D 的提交：此时 table_grounding.py 仍是营收专用硬编码版本
BASELINE_COMMIT = "ed06dca"


def _load_baseline_module():
    try:
        source = subprocess.run(
            ["git", "show", f"{BASELINE_COMMIT}:src/table_grounding.py"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    module = types.ModuleType("table_grounding_baseline")
    module.__dict__["__name__"] = "table_grounding_baseline"
    exec(compile(source, "<baseline:table_grounding.py>", "exec"), module.__dict__)
    return module


BASELINE = _load_baseline_module()

# 输入矩阵：覆盖旧实现全部 15 条规则分支 + 边界。
# 仅使用旧门控认可的问法（含"营业收入"或"营收"）——定义域外的扩展行为单独测试。
QUESTIONS = [
    "宁德时代2024年营业收入是多少？",
    "该公司2024年营收是多少？",
    "2024年分部营业收入是多少？",
    "母公司2024年营业收入是多少？",
    "2024年第四季度营业收入是多少？",
    "营业收入同比增长%是多少？",
]

ROW_COL_CASES = [
    # (row_text, col_text) —— 精确行首 / 合并加成
    ("营业收入", "2024年"),
    ("一、营业收入", "2024年"),
    ("-、营业收入", "本集团"),
    ("营业收入", "合计"),
    ("营业收入", "本集团 2024年"),
    # 次优行首
    ("营业总收入", "2024年"),
    ("一、营业总收入", "合计"),
    ("-、营业总收入", "2024年"),
    # 弱匹配 / 兜底
    ("营业收入合计", "2024年"),
    ("其中:营业收入", "2024年"),
    ("营业收入净额", "2024年"),
    # 低优先词
    ("扣除与主营业务无关的业务收入", "2024年"),
    ("不具备商业实质的营业收入", "2024年"),
    # 坏行首（占比类衍生行）
    ("占营业收入比例", "2024年"),
    ("营业收入比例", "2024年"),
    ("销售费用占营业收入", "2024年"),
    ("研发投入占营业收入", "2024年"),
    # 坏列名（通用衍生列）
    ("营业收入", "同比"),
    ("营业收入", "变动"),
    ("营业收入", "比例"),
    ("营业收入", "占比"),
    ("营业收入", "百分比"),
    ("营业收入", "本期比上年同期"),
    ("营业收入", "期末余额"),
    ("营业收入", "期初"),
    # 坏列名（指标专属）
    ("营业收入", "分部收入"),
    ("营业收入", "关联方"),
    ("营业收入", "数量合计"),
    # 增减列：无年份否决、有年份放行
    ("营业收入", "增减"),
    ("营业收入", "增减 2024年"),
    # 季度列
    ("营业收入", "第四季度"),
    # 百分号列
    ("营业收入", "毛利率%"),
    ("营业收入", "变动百分点"),
    # 主体范围
    ("母公司营业收入", "2024年"),
    ("本行营业收入", "2024年"),
    # 口径维度
    ("分部营业收入", "2024年"),
    ("营业收入", "按地区"),
    ("营业收入", "按产品"),
    ("营业收入", "客户一"),
    # 银行业分部列（segment_guard）
    ("营业收入", "公司银行业务"),
    ("营业收入", "零售银行业务 合计"),
    ("营业收入", "金融市场业务"),
    ("营业收入", "未分配项目"),
    # 无指标词
    ("净利润", "2024年"),
    ("总资产", "合计"),
    ("", ""),
]

TABLE_CONTEXTS = [
    "合并利润表 2024年度",
    "主要会计数据和财务指标",
    "公司简介和主要财务指标",
    "利润表 母公司",
    "分部信息 按地区",
    "关联方交易明细",
    "按产品分类的营业收入",
    "资产负债表",
    "",
]

UNIT_HINTS = [None, "", "元", "万元", "%", "百分点", "人民币千元"]


@unittest.skipIf(BASELINE is None, "git baseline unavailable")
class RowScoreEquivalenceTests(unittest.TestCase):
    def test_row_score_matches_baseline(self):
        checked = 0
        for question in QUESTIONS:
            for row_text, col_text in ROW_COL_CASES:
                expected = BASELINE._revenue_row_score(question, row_text, col_text)
                actual = current._atom_row_score(question, row_text, col_text)
                self.assertEqual(
                    actual,
                    expected,
                    msg=f"q={question!r} row={row_text!r} col={col_text!r}",
                )
                checked += 1
        self.assertGreaterEqual(checked, 40)

    def test_statement_score_matches_baseline(self):
        for question in QUESTIONS:
            for table_context in TABLE_CONTEXTS:
                self.assertEqual(
                    current._atom_statement_score(question, table_context),
                    BASELINE._table_scope_score(question, table_context),
                    msg=f"q={question!r} ctx={table_context!r}",
                )

    def test_unit_rejection_matches_baseline(self):
        for question in QUESTIONS:
            for unit_hint in UNIT_HINTS:
                self.assertEqual(
                    current._reject_unit_by_atom(question, unit_hint),
                    BASELINE._reject_revenue_unit(question, unit_hint),
                    msg=f"q={question!r} unit={unit_hint!r}",
                )

    def test_gate_matches_baseline_on_legacy_domain(self):
        # 旧门控定义域内（含"营业收入"/"营收"）：新旧门控判定一致
        for question in QUESTIONS:
            self.assertEqual(
                current._has_grounding_rule(question),
                BASELINE._is_revenue_question(question),
                msg=f"q={question!r}",
            )
        for question in ["净利润是多少", "法定代表人是谁", "营业总收入是多少"]:
            self.assertEqual(
                current._has_grounding_rule(question),
                BASELINE._is_revenue_question(question),
                msg=f"q={question!r}",
            )


@unittest.skipIf(BASELINE is None, "git baseline unavailable")
class IntentionalBehaviorChangeTests(unittest.TestCase):
    """定义域外的**有意**行为变更：需显式记录，不得混入等价性结论。"""

    def test_alias_question_now_covered(self):
        # "主营业务收入"是营业收入的合法别名，旧门控只认"营业收入"/"营收"，因此完全不打分；
        # 原子驱动后按别名命中 MET-004，获得同一套取证规则。
        question = "2024年主营业务收入是多少？"
        self.assertFalse(BASELINE._is_revenue_question(question))
        self.assertEqual(BASELINE._revenue_row_score(question, "营业收入", "2024年"), 0.0)
        self.assertTrue(current._has_grounding_rule(question))
        self.assertEqual(current._atom_row_score(question, "营业收入", "2024年"), 8.0)


class UniversalGuardTests(unittest.TestCase):
    """通用守卫对**非营收**指标生效——这是拆分的核心目的（结构性改进，未经端到端验证）。"""

    def test_guard_rejects_derived_columns_for_any_metric(self):
        for question in ["2024年货币资金是多少？", "2024年研发投入是多少？", "2024年总资产是多少？"]:
            for bad_col in ["同比", "变动", "占比", "期末余额", "本期比上年同期"]:
                self.assertFalse(
                    current._universal_row_guard(question, "货币资金", bad_col),
                    msg=f"q={question!r} col={bad_col!r}",
                )

    def test_guard_rejects_parent_company_column_when_not_asked(self):
        self.assertFalse(current._universal_row_guard("2024年货币资金是多少？", "母公司货币资金", "2024年"))
        self.assertTrue(current._universal_row_guard("母公司2024年货币资金是多少？", "母公司货币资金", "2024年"))

    def test_guard_rejects_quarter_column_when_not_asked(self):
        self.assertFalse(current._universal_row_guard("2024年货币资金是多少？", "货币资金", "第三季度"))
        self.assertTrue(current._universal_row_guard("2024年第三季度货币资金是多少？", "货币资金", "第三季度"))

    def test_guard_unit_kind_respects_percent_metrics(self):
        # 比率类指标（unit 含 %）不应因列名带 % 被否决
        self.assertFalse(current._universal_row_guard("2024年毛利率是多少？", "毛利率", "%", unit_kind="currency"))
        self.assertTrue(current._universal_row_guard("2024年毛利率是多少？", "毛利率", "%", unit_kind="percent"))

    def test_atom_without_grounding_returns_zero(self):
        # 未配置 grounding 的原子：不参与取证打分（保持既有行为，不引入误加分）
        # G-4 后全库 12 原子均有 grounding，用临时去除规则的原子验证该语义
        import dataclasses as _dc

        from src.metric_atoms import MetricAtomStore, get_default_store, reset_default_store

        original = get_default_store()
        stripped = [
            _dc.replace(atom, grounding=None) if atom.atom_id == "MET-005" else atom
            for atom in original.all_atoms()
        ]
        try:
            reset_default_store(MetricAtomStore(stripped))
            self.assertEqual(current._atom_row_score("2024年货币资金是多少？", "货币资金", "2024年"), 0.0)
            self.assertEqual(current._atom_statement_score("2024年货币资金是多少？", "合并资产负债表"), 0.0)
        finally:
            reset_default_store(original)

    def test_consolidated_bonus(self):
        self.assertEqual(current._consolidated_bonus("本集团"), 2.0)
        self.assertEqual(current._consolidated_bonus("合计"), 2.0)
        self.assertEqual(current._consolidated_bonus("2024年"), 0.0)


class KnowledgeCodeConsistencyTests(unittest.TestCase):
    """知识与旧代码常量逐字一致——这道断言就是错字防线。"""

    @classmethod
    def setUpClass(cls):
        from src.metric_atoms import get_default_store

        cls.rule = get_default_store().get("MET-004").grounding

    @unittest.skipIf(BASELINE is None, "git baseline unavailable")
    def test_atom_terms_match_baseline_constants(self):
        self.assertEqual(set(self.rule.bad_row_headers), set(BASELINE._BAD_REVENUE_ROW_TERMS))
        self.assertEqual(
            set(self.rule.low_priority_row_headers), set(BASELINE._LOW_PRIORITY_REVENUE_TERMS)
        )
        self.assertEqual(set(self.rule.weak_row_headers), set(BASELINE._CONSOLIDATED_REVENUE_TERMS))
        # 坏列名拆分：原子专属 ∪ 通用守卫 == 基线全集
        self.assertEqual(
            set(self.rule.bad_col_headers) | set(current._UNIVERSAL_BAD_COL_TERMS),
            set(BASELINE._BAD_REVENUE_COL_TERMS),
        )

    def test_statement_terms_have_no_typo(self):
        # 显式断言正确字形：迁移时曾把“润”(U+6DA6) 误写为“潦”(U+6F66)，导致规则永久失配。
        # 错字以码点转义表达，避免本断言被“批量修错字”误伤（已发生过一次）。
        all_terms = [term for terms, _ in self.rule.preferred_statements for term in terms]
        self.assertIn("合并利\u6da6表", all_terms)
        self.assertNotIn("合并利\u6f66表", all_terms)
        for term in all_terms:
            self.assertNotIn("\u6f66", term, msg=f"typo in {term!r}")


if __name__ == "__main__":
    unittest.main()
