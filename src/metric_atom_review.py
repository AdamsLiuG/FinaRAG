"""指标原子对抗质检：LLM 多维打分 + 聚合裁决。

与 `metric_atom_mining` 分离是**结构性要求**而非组织习惯：规则质检查不出语义错误
（定义写错准则条款、caliber_notes 与 definition 自相矛盾、别名漏关键表述），需要
另一个模型独立评审——**生产者不自批**。提炼模型与质检模型相同即拒绝执行。

裁决口径（fail-closed）：任一维 0 分 → reject；总分低于阈值 → revise；
LLM 返回不可解析时判 revise 而非 accept——质检失败不得当作质检通过。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

REVIEW_DIMENSIONS = (
    "definition_accuracy",
    "caliber_completeness",
    "alias_coverage",
    "confusable_annotation",
)

_DIMENSION_LABELS = {
    "definition_accuracy": "定义准确性",
    "caliber_completeness": "口径完整性",
    "alias_coverage": "别名覆盖",
    "confusable_annotation": "易混淆标注",
}

# 满分 = 4 维 × 2 分；低于该阈值需返修
ACCEPT_THRESHOLD = 6
MAX_SCORE = len(REVIEW_DIMENSIONS) * 2

REVIEW_SYSTEM_PROMPT = """你是金融会计准则审核专家，负责审核他人提炼的金融指标口径原子。
请对每个原子按四个维度打分，每维 0/1/2 三档：
- definition_accuracy 定义准确性：口径定义是否符合会计准则，有无事实错误
- caliber_completeness 口径完整性：caliber_notes 是否说清与近似口径的边界，有无自相矛盾
- alias_coverage 别名覆盖：aliases 是否覆盖年报常见表述，且不含口径放宽词
- confusable_annotation 易混淆标注：confusable_with 是否标出真正易混的指标，有无遗漏

评分标准：2=无问题；1=有可改进处但不影响正确使用；0=存在错误或缺失，不可入库。

以 JSON 对象输出，格式：
{"scores": {"definition_accuracy": 2, "caliber_completeness": 1, "alias_coverage": 2,
"confusable_annotation": 0}, "issues": ["逐条说明扣分依据"]}

只输出 JSON 对象，不要输出其他内容。每个扣分维度必须在 issues 中给出依据。"""

REVIEW_USER_PROMPT = """请审核以下金融指标口径原子：

标准名：{standard_name}
别名：{aliases}
口径定义：{definition}
口径说明：{caliber_notes}
计量单位：{unit}
报表位置：{statement_location}
易混淆指标：{confusable_names}
常见误区：{pitfalls}"""


def build_review_prompt(atom_payload: Dict[str, Any]) -> str:
    return REVIEW_USER_PROMPT.format(
        standard_name=atom_payload.get("standard_name") or "",
        aliases="、".join(str(a) for a in (atom_payload.get("aliases") or [])) or "（无）",
        definition=atom_payload.get("definition") or "（无）",
        caliber_notes=atom_payload.get("caliber_notes") or "（无）",
        unit=atom_payload.get("unit") or "（无）",
        statement_location=atom_payload.get("statement_location") or "（无）",
        confusable_names="、".join(str(c) for c in (atom_payload.get("confusable_names") or []))
        or "（无）",
        pitfalls=atom_payload.get("pitfalls") or "（无）",
    )


def parse_review_response(text: Any) -> Dict[str, Any]:
    """解析质检响应；不可解析或字段非法时判 revise（fail-closed）。"""
    raw = text if isinstance(text, str) else str(text)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return {
            "scores": {},
            "issues": [f"质检响应不可解析：{raw[:120]}"],
            "verdict": "revise",
            "parse_failed": True,
        }
    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        return {
            "scores": {},
            "issues": [f"质检响应 JSON 解析失败：{exc}"],
            "verdict": "revise",
            "parse_failed": True,
        }

    scores: Dict[str, int] = {}
    issues = [str(item) for item in (payload.get("issues") or [])]
    raw_scores = payload.get("scores") or {}
    for dimension in REVIEW_DIMENSIONS:
        value = raw_scores.get(dimension)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            issues.append(f"{_DIMENSION_LABELS[dimension]}：缺少有效评分")
            continue
        scores[dimension] = max(0, min(2, int(value)))

    result = {"scores": scores, "issues": issues, "parse_failed": False}
    result["verdict"] = _verdict(scores)
    return result


def _verdict(scores: Dict[str, int]) -> str:
    if len(scores) < len(REVIEW_DIMENSIONS):
        # 维度缺失即视为质检未完成，不得放行
        return "revise"
    if any(value == 0 for value in scores.values()):
        return "reject"
    if sum(scores.values()) < ACCEPT_THRESHOLD:
        return "revise"
    return "accept"


def review_atom_with_llm(
    api_processor,
    atom_payload: Dict[str, Any],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """调用质检模型评审单个原子（需 API key 环境）。"""
    answer = api_processor.send_message(
        system_content=REVIEW_SYSTEM_PROMPT,
        human_content=build_review_prompt(atom_payload),
        is_structured=False,
        **({"model": model} if model else {}),
    )
    result = parse_review_response(answer)
    result["atom_id"] = atom_payload.get("atom_id")
    result["standard_name"] = atom_payload.get("standard_name")
    return result


def aggregate_review(reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总批量质检结果。"""
    counts = {"accept": 0, "revise": 0, "reject": 0}
    blocking: List[str] = []
    for review in reviews:
        verdict = str(review.get("verdict") or "revise")
        counts[verdict] = counts.get(verdict, 0) + 1
        if verdict != "accept":
            blocking.append(str(review.get("standard_name") or review.get("atom_id") or "?"))
    total = len(reviews)
    return {
        "total": total,
        "counts": counts,
        "accept_rate": round(counts["accept"] / total, 4) if total else None,
        "blocking": blocking,
        "gate_passed": total > 0 and counts["reject"] == 0 and counts["revise"] == 0,
    }


def render_review_report(reviews: List[Dict[str, Any]]) -> str:
    """渲染 Markdown 质检报告（供人工校准）。"""
    summary = aggregate_review(reviews)
    lines = [
        "# 指标原子质检报告",
        "",
        f"- 受检原子：{summary['total']}",
        f"- 通过 / 返修 / 拒绝：{summary['counts']['accept']} / "
        f"{summary['counts']['revise']} / {summary['counts']['reject']}",
        f"- 门禁结论：{'通过' if summary['gate_passed'] else '未通过'}",
        "",
        "| 原子 | 判定 | " + " | ".join(_DIMENSION_LABELS[d] for d in REVIEW_DIMENSIONS) + " | 总分 |",
        "|---|---|" + "---|" * (len(REVIEW_DIMENSIONS) + 1),
    ]
    for review in reviews:
        scores = review.get("scores") or {}
        cells = [str(scores.get(d, "-")) for d in REVIEW_DIMENSIONS]
        total_score = sum(scores.values()) if scores else "-"
        lines.append(
            f"| {review.get('standard_name') or review.get('atom_id') or '?'} "
            f"| {review.get('verdict')} | " + " | ".join(cells) + f" | {total_score}/{MAX_SCORE} |"
        )
    issue_lines = [
        f"- **{review.get('standard_name') or '?'}**：{issue}"
        for review in reviews
        for issue in (review.get("issues") or [])
    ]
    if issue_lines:
        lines.extend(["", "## 扣分依据", *issue_lines])
    return "\n".join(lines) + "\n"


class SameModelReviewError(RuntimeError):
    """提炼模型与质检模型相同——违反生产者不自批原则。"""


def assert_reviewer_differs(mining_model: str, review_model: str) -> None:
    if str(mining_model).strip() and str(mining_model).strip() == str(review_model).strip():
        raise SameModelReviewError(
            f"质检模型与提炼模型相同（{review_model}）：同一模型无法发现自己的口径误解，"
            "请指定不同的 --review-model"
        )
