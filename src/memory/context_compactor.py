"""Extractive session context that keeps evidence and provenance intact."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.service.budget_controller import estimate_text_tokens

_CITATION = re.compile(r"\[(?:\d+(?:\s*[,，]\s*\d+)*|第\s*\d+\s*页)\](?:\([^\r\n)]*\))?")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_PIPE = re.compile(r"(?<!\\)(?:\\\\)*\|")
_SEPARATOR = re.compile(r":?-+:?")
_FACT_FIELDS = ("metric", "value", "unit", "entity", "period", "source", "citation_id")


def _cells(line):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and any(match.end() == len(line) for match in _PIPE.finditer(line)):
        line = line[:-1]
    return _PIPE.split(line)


def _is_table(header, separator):
    if not _PIPE.search(header):
        return False
    cells = _cells(separator)
    return len(cells) == len(_cells(header)) and all(_SEPARATOR.fullmatch(cell.strip()) for cell in cells)


def markdown_segments(text: str):
    """Yield narrative and indivisible Markdown spans in their original order."""
    lines = text.splitlines(keepends=True)
    index = 0
    while index < len(lines):
        line = lines[index]
        end = index + 1
        if fence := _FENCE.match(line):
            marker = fence.group(1)
            closing = re.compile(r"^ {0,3}" + marker[0] + r"{%d,}\s*$" % len(marker))
            while end < len(lines):
                end += 1
                if closing.match(lines[end - 1]):
                    break
        elif end < len(lines) and _is_table(line, lines[end]):
            end += 1
            while end < len(lines) and _PIPE.search(lines[end]) and not _FENCE.match(lines[end]):
                end += 1
        else:
            previous = 0
            for match in _CITATION.finditer(line):
                yield False, line[previous:match.start()]
                yield True, match.group()
                previous = match.end()
            yield False, line[previous:]
            index = end
            continue
        yield True, "".join(lines[index:end])
        index = end


def bounded_narrative(text: str, max_chars: int = 2400) -> str:
    pieces = []
    remaining = max_chars
    for protected, part in markdown_segments(text):
        if protected:
            pieces.append(part)
        elif part:
            kept = part[:remaining]
            remaining -= len(kept)
            pieces.append(kept)
            if len(kept) < len(part):
                line_ending = re.search(r"(?:\r\n|\n|\r)$", part)
                if line_ending and not kept.endswith(("\n", "\r")):
                    pieces.append(line_ending.group())
    return "".join(pieces)


@dataclass(frozen=True)
class CompactedContext:
    text: str
    untrusted_text: str
    evidence_tables: tuple[dict[str, Any], ...]
    facts: tuple[dict[str, Any], ...]
    protected_blocks: tuple[str, ...]
    protected_provenance: tuple[dict[str, Any], ...]
    recent_turns: tuple[dict[str, Any], ...]
    estimated_tokens: int
    over_budget: bool


class ContextCompactor:
    def __init__(self, *, token_threshold=4000, recent_turns=3, estimate_tokens=None):
        if type(token_threshold) is not int or token_threshold < 0 or type(recent_turns) is not int or recent_turns < 0:
            raise ValueError("Invalid session context budget")
        self.token_threshold = token_threshold
        self.recent_turns = recent_turns
        self.estimate_tokens = estimate_tokens if estimate_tokens is not None else estimate_text_tokens

    def compact(self, turns) -> CompactedContext:
        turns = tuple(turns)
        narratives, protected_narratives = [], []
        blocks, provenance, evidence_tables = [], [], []
        facts = {}
        for turn in turns:
            fields, protected_fields = [], []
            for name in ("question", "answer"):
                field = str(turn.get(name) or "")
                fields.append(field)
                for protected, piece in markdown_segments(field):
                    if protected:
                        protected_fields.append(piece)
                        provenance.append({
                            "turn_id": turn.get("turn_id"), "field": name, "block_index": len(blocks),
                            "sources": (turn.get("sources") or []) if name == "answer" else [],
                        })
                        blocks.append(piece)
            narratives.append("\n".join(fields))
            protected_narratives.append("\n".join(protected_fields))
            for evidence in turn.get("evidence_tables") or []:
                provenance.append({
                    "turn_id": turn.get("turn_id"), "field": "evidence_table", "block_index": len(blocks),
                    "sources": [{key: value for key, value in evidence.items() if key != "text"}],
                })
                blocks.append(evidence["text"])
                evidence_tables.append(evidence)
            for fact in turn.get("facts") or []:
                if (not isinstance(fact, dict) or fact.get("verified") is not True or
                        not all(key in fact for key in _FACT_FIELDS) or
                        not isinstance(fact["value"], str) or not fact["source"] or not fact["citation_id"]):
                    continue
                key = json.dumps({field: fact[field] for field in _FACT_FIELDS}, sort_keys=True, ensure_ascii=False)
                facts.setdefault(key, fact)

        facts = tuple(facts.values())
        fact_text = json.dumps(facts, ensure_ascii=False) if facts else ""
        table_text = "\n".join(item["text"] for item in evidence_tables)
        untrusted = "\n\n".join(part for part in narratives if part.strip())
        text = "\n\n".join(part for part in (untrusted, table_text, fact_text) if part)
        count = self.estimate_tokens(text)
        cutoff = 0
        if count > self.token_threshold:
            cutoff = max(0, len(turns) - self.recent_turns)
            untrusted = "\n\n".join(part for part in (*protected_narratives[:cutoff], *narratives[cutoff:]) if part.strip())
            text = "\n\n".join(part for part in (untrusted, table_text, fact_text) if part)
            count = self.estimate_tokens(text)
        if count > self.token_threshold:
            # Keep the original protected spans even if the smallest context does not fit.
            cutoff = len(turns)
            untrusted = "\n".join(part for part in protected_narratives if part)
            text = "\n\n".join(part for part in ("\n".join(blocks), fact_text) if part)
            count = self.estimate_tokens(text)
        return CompactedContext(
            text=text, untrusted_text=untrusted, facts=facts, evidence_tables=tuple(evidence_tables),
            protected_blocks=tuple(blocks), protected_provenance=tuple(provenance), recent_turns=turns[cutoff:],
            estimated_tokens=count, over_budget=count > self.token_threshold,
        )
