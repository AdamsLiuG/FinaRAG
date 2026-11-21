"""AST allow-list Program-of-Thought (PoT) deterministic financial calculator."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Union

from src.table_grounding import _TARGET_UNIT_FACTORS

_MAX_EXPRESSION_LENGTH = 512
_MAX_AST_NODES = 64
_MAX_POW_EXPONENT = 100.0
_MAX_POW_BASE = 1e18

_MONETARY_UNIT_FACTORS: Dict[str, float] = {
    **_TARGET_UNIT_FACTORS,
    "千元": 1e3,
}
_UNIT_ALIASES: Dict[str, str] = {
    "人民币亿元": "亿元",
    "人民币千万元": "千万元",
    "人民币百万元": "百万元",
    "人民币万元": "万元",
    "人民币千元": "千元",
    "万元人民币": "万元",
    "人民币元": "元",
    "cny": "元",
    "rmb": "元",
    "％": "%",
}
_SAFE_CALLABLES = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}


class PotSecurityError(ValueError):
    """Raised when an expression violates the PoT AST security allow-list."""


class PotCalculationError(ValueError):
    """Raised when an expression fails mathematical evaluation or unit validation."""


@dataclass(frozen=True)
class PotVariable:
    name: str
    value: float
    unit: str = ""
    page: Optional[int] = None
    doc_id: Optional[str] = None
    quote: str = ""


@dataclass(frozen=True)
class PotResult:
    value: float
    unit: str
    expression: str
    normalized_variables: Dict[str, float]
    base_unit: str
    citations_hint: List[Dict[str, Any]] = field(default_factory=list)


def normalize_monetary_unit(unit: Optional[str]) -> str:
    cleaned = "".join(str(unit or "").strip().split())
    if not cleaned:
        return ""
    lower = cleaned.lower()
    if lower in _UNIT_ALIASES:
        return _UNIT_ALIASES[lower]
    if cleaned in _UNIT_ALIASES:
        return _UNIT_ALIASES[cleaned]
    for prefix in ("人民币",):
        if cleaned.startswith(prefix) and len(cleaned) > len(prefix):
            stripped = cleaned[len(prefix):]
            return _UNIT_ALIASES.get(stripped, stripped)
    return cleaned


def _monetary_factor(unit: str) -> Optional[float]:
    normalized = normalize_monetary_unit(unit)
    return _MONETARY_UNIT_FACTORS.get(normalized)


def _count_ast_nodes(tree: ast.AST) -> int:
    return sum(1 for _ in ast.walk(tree))


def _coerce_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise PotCalculationError(f"Variable '{name}' must be numeric, got bool")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise PotCalculationError(f"Variable '{name}' is not a valid number: {value!r}") from exc
    if not math.isfinite(numeric):
        raise PotCalculationError(f"Variable '{name}' must be finite, got {numeric!r}")
    return numeric


class PotSandbox:
    """Deterministic arithmetic sandbox backed by an explicit AST allow-list."""

    def evaluate(
        self,
        variables: Union[Dict[str, PotVariable], Iterable[PotVariable]],
        expression: str,
        result_unit: Optional[str] = None,
        ndigits: Optional[int] = 4,
    ) -> PotResult:
        if not isinstance(expression, str):
            raise PotSecurityError("Expression must be a string")
        expr = expression.strip()
        if not expr:
            raise PotSecurityError("Expression must not be empty")
        if len(expr) > _MAX_EXPRESSION_LENGTH:
            raise PotSecurityError(f"Expression exceeds max length ({_MAX_EXPRESSION_LENGTH})")

        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise PotSecurityError(f"Invalid expression syntax: {exc.msg}") from exc

        node_count = _count_ast_nodes(tree)
        if node_count > _MAX_AST_NODES:
            raise PotSecurityError(f"Expression AST exceeds max nodes ({_MAX_AST_NODES})")

        var_map: Dict[str, PotVariable] = {}
        items = variables.values() if isinstance(variables, dict) else variables
        for item in items:
            if not isinstance(item, PotVariable):
                raise PotCalculationError("All variables must be PotVariable instances")
            if not item.name or item.name.startswith("_") or not item.name.isidentifier():
                raise PotSecurityError(f"Invalid variable identifier: {item.name!r}")
            var_map[item.name] = item

        normalized_env: Dict[str, float] = {}
        has_monetary_var = False
        citations_hint: List[Dict[str, Any]] = []
        for name, var in var_map.items():
            val = _coerce_finite_float(var.value, name)
            factor = _monetary_factor(var.unit)
            if factor is not None:
                val = val * factor
                has_monetary_var = True
            normalized_env[name] = val
            if var.page is not None or var.doc_id is not None or var.quote:
                citations_hint.append(
                    {
                        "name": name,
                        "doc_id": var.doc_id,
                        "page": var.page,
                        "quote": var.quote,
                        "unit": normalize_monetary_unit(var.unit),
                    }
                )

        raw_result = self._eval_node(tree, normalized_env)
        if isinstance(raw_result, complex) or not math.isfinite(raw_result):
            raise PotCalculationError("Expression produced a non-finite or complex value")

        norm_result_unit = normalize_monetary_unit(result_unit)
        target_factor = _monetary_factor(norm_result_unit) if norm_result_unit else None
        final_value = float(raw_result)
        if target_factor is not None and has_monetary_var:
            final_value = final_value / target_factor
        if ndigits is not None:
            final_value = round(final_value, ndigits)

        return PotResult(
            value=final_value,
            unit=norm_result_unit,
            expression=expr,
            normalized_variables=normalized_env,
            base_unit="元" if has_monetary_var else "",
            citations_hint=citations_hint,
        )

    def _eval_node(self, node: ast.AST, env: Dict[str, float]) -> float:
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body, env)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise PotSecurityError(f"Unsupported constant type: {type(node.value).__name__}")
            val = float(node.value)
            if not math.isfinite(val):
                raise PotCalculationError("Non-finite constant")
            return val

        if isinstance(node, ast.Name):
            if node.id.startswith("_") or node.id not in env:
                raise PotSecurityError(f"Undeclared or forbidden identifier: {node.id!r}")
            return env[node.id]

        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, env)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise PotSecurityError(f"Unsupported unary operator: {type(node.op).__name__}")

        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, env)
            right = self._eval_node(node.right, env)
            return self._eval_binop(node.op, left, right)

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_CALLABLES:
                func_desc = getattr(node.func, "id", type(node.func).__name__)
                raise PotSecurityError(f"Forbidden function call: {func_desc!r}")
            if node.keywords:
                raise PotSecurityError("Keyword arguments are not permitted in PoT expressions")
            args = [self._eval_node(arg, env) for arg in node.args]
            if not args:
                raise PotCalculationError(f"Function '{node.func.id}' requires at least one argument")
            if node.func.id == "round":
                if len(args) > 2:
                    raise PotCalculationError("round() accepts at most 2 arguments")
                if len(args) == 2:
                    digits = int(args[1])
                    if abs(digits) > 12:
                        raise PotSecurityError("round() ndigits out of safe bounds")
                    return float(round(args[0], digits))
                return float(round(args[0]))
            if node.func.id == "abs" and len(args) != 1:
                raise PotCalculationError("abs() takes exactly 1 argument")
            return float(_SAFE_CALLABLES[node.func.id](*args))

        raise PotSecurityError(f"Forbidden AST node: {type(node).__name__}")

    @staticmethod
    def _eval_binop(op: ast.operator, left: float, right: float) -> float:
        try:
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                return left * right
            if isinstance(op, ast.Div):
                if right == 0.0:
                    raise PotCalculationError("Division by zero in PoT expression")
                return left / right
            if isinstance(op, ast.FloorDiv):
                if right == 0.0:
                    raise PotCalculationError("Floor division by zero in PoT expression")
                return left // right
            if isinstance(op, ast.Mod):
                if right == 0.0:
                    raise PotCalculationError("Modulo by zero in PoT expression")
                return left % right
            if isinstance(op, ast.Pow):
                if abs(right) > _MAX_POW_EXPONENT or abs(left) > _MAX_POW_BASE:
                    raise PotSecurityError("Exponentiation bounds exceeded")
                result = left ** right
                if isinstance(result, complex):
                    raise PotCalculationError("Complex result in exponentiation")
                return float(result)
        except OverflowError as exc:
            raise PotCalculationError("Arithmetic overflow in PoT expression") from exc
        raise PotSecurityError(f"Unsupported binary operator: {type(op).__name__}")


def extract_and_execute_pot(
    payload: Dict[str, Any],
    sandbox: Optional[PotSandbox] = None,
) -> Optional[PotResult]:
    """Parse and run optional ``pot_calculation`` dict from structured LLM output."""
    if not isinstance(payload, dict):
        return None
    pot_block = payload.get("pot_calculation")
    if not isinstance(pot_block, dict):
        return None
    expression = pot_block.get("expression")
    raw_vars = pot_block.get("variables")
    if not expression or not isinstance(raw_vars, dict):
        return None

    variables: Dict[str, PotVariable] = {}
    for var_name, spec in raw_vars.items():
        if isinstance(spec, dict):
            variables[var_name] = PotVariable(
                name=str(var_name),
                value=_coerce_finite_float(spec.get("value"), str(var_name)),
                unit=str(spec.get("unit") or ""),
                page=int(spec["page"]) if spec.get("page") is not None else None,
                doc_id=str(spec["doc_id"]) if spec.get("doc_id") is not None else None,
                quote=str(spec.get("quote") or ""),
            )
        else:
            variables[var_name] = PotVariable(
                name=str(var_name),
                value=_coerce_finite_float(spec, str(var_name)),
            )

    runner = sandbox or PotSandbox()
    return runner.evaluate(
        variables=variables,
        expression=str(expression),
        result_unit=pot_block.get("result_unit"),
    )
