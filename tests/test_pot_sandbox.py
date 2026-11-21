"""Unit and security tests for the PoT AST safe calculator."""

import unittest

from src.structured.pot_sandbox import (
    PotCalculationError,
    PotSandbox,
    PotSecurityError,
    PotVariable,
    extract_and_execute_pot,
    normalize_monetary_unit,
)


class PotSandboxTests(unittest.TestCase):
    def setUp(self):
        self.sandbox = PotSandbox()

    def test_unit_normalization_and_yoy_growth(self):
        self.assertEqual(normalize_monetary_unit("人民币百万元"), "百万元")
        self.assertEqual(normalize_monetary_unit("万元人民币"), "万元")

        res = self.sandbox.evaluate(
            variables={
                "rev_2024": PotVariable("rev_2024", 170.748, "亿元", page=19, doc_id="spdb"),
                "rev_2023": PotVariable("rev_2023", 173434.0, "百万元", page=19, doc_id="spdb"),
            },
            expression="(rev_2024 - rev_2023) / rev_2023 * 100",
            result_unit="%",
            ndigits=2,
        )
        # 170.748 亿元 = 17,074,800,000; 173,434 百万元 = 17,343,400,000 -> -1.55%
        # Note: 173,434 百万元 = 173,434,000,000; wait: 170,748 百万元 = 1707.48 亿元!
        res2 = self.sandbox.evaluate(
            variables={
                "rev_2024": PotVariable("rev_2024", 1707.48, "亿元", page=19),
                "rev_2023": PotVariable("rev_2023", 173434.0, "百万元", page=19),
            },
            expression="(rev_2024 - rev_2023) / rev_2023 * 100",
            result_unit="%",
            ndigits=2,
        )
        self.assertAlmostEqual(res2.value, -1.55, places=2)
        self.assertEqual(res2.unit, "%")
        self.assertEqual(len(res2.citations_hint), 2)
        self.assertIsNotNone(res.value)

    def test_monetary_unit_conversion_across_wan_and_yi(self):
        res = self.sandbox.evaluate(
            variables={
                "a": PotVariable("a", 120.34, "亿元"),
                "b": PotVariable("b", 1085000.0, "万元"),
            },
            expression="a - b",
            result_unit="亿元",
            ndigits=2,
        )
        self.assertAlmostEqual(res.value, 11.84, places=2)
        self.assertEqual(res.unit, "亿元")

    def test_safe_builtin_functions_allowed(self):
        res = self.sandbox.evaluate(
            variables={"x": PotVariable("x", -15.678, "万元"), "y": PotVariable("y", 10.0, "万元")},
            expression="round(max(abs(x), y), 1)",
            result_unit="万元",
            ndigits=2,
        )
        # abs(-15.678 万元) = 156780 元 -> round(..., 1) = 156780.0 元 -> 15.68 万元
        self.assertAlmostEqual(res.value, 15.68, places=2)

    def test_division_by_zero_and_complex_pow_raise_calculation_error(self):
        with self.assertRaises(PotCalculationError):
            self.sandbox.evaluate(
                variables={"a": PotVariable("a", 10.0), "b": PotVariable("b", 0.0)},
                expression="a / b",
            )
        with self.assertRaises(PotCalculationError):
            self.sandbox.evaluate(
                variables={"a": PotVariable("a", -4.0)},
                expression="a ** 0.5",
            )

    def test_security_rejects_code_injection_and_dos(self):
        malicious_expressions = [
            "__import__('os').system('ls')",
            "getattr(x, '__class__')",
            "().__class__.__mro__[1].__subclasses__()",
            "(lambda z: z + 1)(x)",
            "[i for i in range(10)]",
            "x[0]",
            "open('/etc/passwd').read()",
            "10 ** 10 ** 10",
            "round(x, ndigits=2)",
            "undeclared_var + 1",
            "1 + " * 40 + "1",  # > 64 AST nodes
            "x + " + "1" * 520,  # > 512 chars
        ]
        vars_dict = {"x": PotVariable("x", 10.0)}
        for expr in malicious_expressions:
            with self.subTest(expr=expr[:40]):
                with self.assertRaises(PotSecurityError):
                    self.sandbox.evaluate(variables=vars_dict, expression=expr)

        with self.assertRaises(PotSecurityError):
            self.sandbox.evaluate(
                variables={"_secret": PotVariable("_secret", 1.0)},
                expression="_secret + 1",
            )

    def test_extract_and_execute_pot_helper(self):
        payload = {
            "pot_calculation": {
                "variables": {
                    "rev_2024": {"value": 54.59, "unit": "亿元", "page": 5, "quote": "54.59亿元"},
                    "rev_2023": {"value": 45.35, "unit": "亿元", "page": 5, "quote": "45.35亿元"},
                },
                "expression": "(rev_2024 - rev_2023) / rev_2023 * 100",
                "result_unit": "%",
            }
        }
        res = extract_and_execute_pot(payload)
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res.value, 20.37, places=2)
        self.assertEqual(res.citations_hint[0]["page"], 5)


if __name__ == "__main__":
    unittest.main()
