from __future__ import annotations

import re
from typing import Iterable, List


_WHITESPACE_RE = re.compile(r"\s+")
_NUMBER_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[./-][a-z0-9]+)?")

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
    "aud": "AUD",
    "cad": "CAD",
    "chf": "CHF",
}


def normalize_currency_token(token: str | None) -> str | None:
    if not token:
        return None
    return _CURRENCY_ALIASES.get(token.strip().lower(), token.strip().upper())


def normalize_text(text: str) -> str:
    if not text:
        return ""

    normalized = text.replace("\u00a0", " ").replace("’", "'").replace("–", "-").replace("—", "-")
    normalized = normalized.lower()
    normalized = _NUMBER_COMMA_RE.sub("", normalized)
    normalized = normalized.replace("%", " percent ")
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def tokenize_for_bm25(text: str) -> List[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return _TOKEN_RE.findall(normalized)


def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
