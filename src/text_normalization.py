from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Optional

try:
    import jieba  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    jieba = None


_WHITESPACE_RE = re.compile(r"\s+")
_NUMBER_COMMA_RE = re.compile(r"(?<=\d)[,，](?=\d)")
_LATIN_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[./_-][a-z0-9]+)?")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_SECURITY_CODE_RE = re.compile(
    r"(?<![A-Z0-9])"
    r"(?:"
    r"(?:SH|SZ|SSE|SZSE|BJ|BSE)[\s:._-]?([03468]\d{5})"
    r"|([03468]\d{5})[\s:._-]?(?:SH|SZ|SSE|SZSE|BJ|BSE)"
    r"|([03468]\d{5})"
    r")"
    r"(?![A-Z0-9])",
    re.IGNORECASE,
)
_PERIOD_RE = re.compile(r"(20\d{2}\s*(?:q[1-4]|年报|半年报|中报|季报|一季报|三季报))", re.IGNORECASE)

_CURRENCY_ALIASES = {
    "$": "USD",
    "usd": "USD",
    "us$": "USD",
    "eur": "EUR",
    "€": "EUR",
    "gbp": "GBP",
    "£": "GBP",
    "jpy": "JPY",
    "yen": "JPY",
    "cny": "CNY",
    "rmb": "CNY",
    "人民币": "CNY",
    "aud": "AUD",
    "cad": "CAD",
    "chf": "CHF",
    "港币": "HKD",
    "hkd": "HKD",
}

_NUMERIC_UNIT_FACTORS = {
    "亿股": 1e8,
    "亿元": 1e8,
    "亿港元": 1e8,
    "亿美元": 1e8,
    "亿人民币": 1e8,
    "千万": 1e7,
    "百万": 1e6,
    "万股": 1e4,
    "万元": 1e4,
    "万港元": 1e4,
    "万美元": 1e4,
    "万元人民币": 1e4,
    "千元": 1e3,
    "百万元": 1e6,
    "千万元": 1e7,
}

_FINANCE_TERMS = {
    "营业收入",
    "营收",
    "归母净利润",
    "扣非归母净利润",
    "毛利率",
    "净利率",
    "期间费用率",
    "经营活动现金流净额",
    "资产负债率",
    "每股收益",
    "分红",
    "回购",
    "并购",
    "定增",
    "券商",
    "年报",
    "研报",
}


def contains_cjk(text: str) -> bool:
    return bool(text and _CJK_RE.search(text))


def normalize_currency_token(token: str | None) -> str | None:
    if not token:
        return None
    normalized = unicodedata.normalize("NFKC", token).strip()
    return _CURRENCY_ALIASES.get(normalized.lower(), normalized.upper())


def normalize_text(text: str) -> str:
    if not text:
        return ""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u00a0", " ")
    normalized = normalized.replace("’", "'").replace("–", "-").replace("—", "-")
    normalized = normalized.replace("（", "(").replace("）", ")")
    normalized = normalized.replace("：", ":").replace("；", ";").replace("，", ",")
    normalized = _NUMBER_COMMA_RE.sub("", normalized)
    normalized = normalized.lower()
    normalized = normalized.replace("%", " % ")
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def normalize_period_token(text: str | None) -> str | None:
    normalized = normalize_text(text or "")
    if not normalized:
        return None
    normalized = normalized.replace("年度报告", "年报").replace("季度报告", "季报")
    normalized = normalized.replace("半年度报告", "半年报")
    match = _PERIOD_RE.search(normalized)
    if match:
        return match.group(1).replace(" ", "").upper().replace("年报", "年报")
    return None


def _fallback_cjk_tokens(text: str) -> List[str]:
    tokens: List[str] = []
    for chunk in _CJK_RE.findall(text):
        if len(chunk) <= 2:
            tokens.append(chunk)
            continue
        tokens.append(chunk)
        tokens.extend(chunk[index:index + 2] for index in range(len(chunk) - 1))
        for term in _FINANCE_TERMS:
            if term in chunk:
                tokens.append(term)
    return tokens


def tokenize_for_bm25(text: str) -> List[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    tokens: List[str] = []
    tokens.extend(_LATIN_TOKEN_RE.findall(normalized))
    if contains_cjk(normalized):
        if jieba is not None:
            tokens.extend(
                token.strip()
                for token in jieba.cut(normalized, cut_all=False)
                if token.strip()
            )
        else:
            tokens.extend(_fallback_cjk_tokens(normalized))

    ordered: List[str] = []
    seen = set()
    for token in tokens:
        if not token or token.isspace() or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def parse_numeric_value(text: str | None, unit_hint: str | None = None) -> Optional[float]:
    if not text:
        return None
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _NUMBER_COMMA_RE.sub("", normalized)
    negative = bool(re.search(r"^\s*[\(\-]|[\(\-]\s*\d", normalized))
    match = re.search(r"-?\d+(?:\.\d+)?", normalized)
    if not match:
        return None

    value = float(match.group(0))
    if negative and value > 0:
        value = -value

    unit_text = f"{normalized} {unit_hint or ''}"
    if "%" in unit_text or "百分点" in unit_text:
        return value

    for unit, factor in sorted(_NUMERIC_UNIT_FACTORS.items(), key=lambda item: len(item[0]), reverse=True):
        if unit in unit_text:
            return value * factor
    return value


def extract_security_codes(text: str) -> List[str]:
    codes: List[str] = []
    for match in _SECURITY_CODE_RE.finditer(unicodedata.normalize("NFKC", text or "")):
        token = next((group for group in match.groups() if group), "").strip()
        if len(token) != 6 or token in codes:
            continue
        codes.append(token)
    return codes


def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
