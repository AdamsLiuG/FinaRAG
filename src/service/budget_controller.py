"""Runtime 4-level Token & Latency Budget Controller with dynamic degradation."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def estimate_text_tokens(text: str) -> int:
    """Rough char-based token estimate safe for offline bilingual text."""
    s = str(text or "")
    if not s:
        return 0
    return max(1, int(len(s) / 1.8))


@dataclass
class BudgetContext:
    """Thread-safe per-request budget tracker driving 4-level dynamic degradation."""

    max_tokens: int = 8000
    max_latency_ms: float = 30000.0
    spent_tokens: int = 0
    start_ts: float = field(default_factory=time.monotonic)
    fast_path: bool = False
    degraded_events: List[Dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    @property
    def remaining_tokens(self) -> int:
        with self._lock:
            if self.max_tokens <= 0:
                return 10**9
            return max(0, self.max_tokens - self.spent_tokens)

    @property
    def remaining_token_ratio(self) -> float:
        with self._lock:
            if self.max_tokens <= 0:
                return 1.0
            return max(0.0, (self.max_tokens - self.spent_tokens) / float(self.max_tokens))

    @property
    def elapsed_ms(self) -> float:
        return max(0.0, (time.monotonic() - self.start_ts) * 1000.0)

    @property
    def remaining_latency_ms(self) -> float:
        if self.max_latency_ms <= 0:
            return 1e9
        return max(0.0, self.max_latency_ms - self.elapsed_ms)

    @property
    def remaining_latency_ratio(self) -> float:
        if self.max_latency_ms <= 0:
            return 1.0
        return max(0.0, self.remaining_latency_ms / float(self.max_latency_ms))

    @property
    def degradation_level(self) -> int:
        """Compute active runtime degradation level: 0 (normal) .. 3 (circuit-break)."""
        tok_ratio = self.remaining_token_ratio
        lat_ratio = self.remaining_latency_ratio
        min_ratio = min(tok_ratio, lat_ratio)
        if min_ratio <= 0.0:
            return 3
        if min_ratio < 0.25:
            return 2
        if min_ratio < 0.50:
            return 1
        return 0

    def should_use_fast_path(self, query_plan: Any) -> bool:
        task_type = str(getattr(query_plan, "task_type", "") or "").lower()
        route_mode = str(getattr(query_plan, "route_mode", "") or "").lower()
        if route_mode in {"structured_sql", "rule_plugin"}:
            self.fast_path = True
        elif task_type in {"definition", "single_fact", "fact_lookup"} and len(getattr(query_plan, "search_queries", []) or []) <= 1:
            self.fast_path = True
        return self.fast_path

    @property
    def allow_hyde(self) -> bool:
        level = self.degradation_level
        allowed = level == 0 and not self.fast_path
        if not allowed and level >= 1:
            self._record_event("skip_hyde", level)
        return allowed

    @property
    def allow_adaptive_supplement(self) -> bool:
        level = self.degradation_level
        allowed = level == 0
        if not allowed:
            self._record_event("skip_adaptive_supplement", level)
        return allowed

    @property
    def skip_second_stage_rerank(self) -> bool:
        level = self.degradation_level
        skip = level >= 3 or self.fast_path
        if skip:
            self._record_event("skip_rerank", level)
        return skip

    def effective_top_n(self, default_top_n: int) -> int:
        level = self.degradation_level
        base = max(1, int(default_top_n))
        if level >= 3:
            self._record_event("cap_top_n_2", level)
            return min(base, 2)
        if level >= 2:
            self._record_event("cap_top_n_3", level)
            return min(base, 3)
        return base

    def compress_results(self, results: List[Dict[str, Any]], max_chars: int = 800) -> List[Dict[str, Any]]:
        level = self.degradation_level
        if level < 2:
            return results
        self._record_event("compress_context", level)
        compressed: List[Dict[str, Any]] = []
        for item in results:
            copy_item = dict(item)
            text = str(copy_item.get("text") or "")
            is_table = copy_item.get("chunk_type") == "table" or bool((copy_item.get("metadata") or {}).get("table_id"))
            if not is_table and len(text) > max_chars:
                copy_item["text"] = text[:max_chars].rstrip() + "..."
            compressed.append(copy_item)
        return compressed

    def record_usage(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> int:
        total_delta = max(0, int(prompt_tokens or 0)) + max(0, int(completion_tokens or 0))
        with self._lock:
            self.spent_tokens += total_delta
            return self.spent_tokens

    def _record_event(self, action: str, level: int) -> None:
        with self._lock:
            if any(ev.get("action") == action and ev.get("level") == level for ev in self.degraded_events):
                return
            self.degraded_events.append(
                {
                    "action": action,
                    "level": level,
                    "spent_tokens": self.spent_tokens,
                    "elapsed_ms": round(self.elapsed_ms, 2),
                }
            )

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            spent = self.spent_tokens
            events = list(self.degraded_events)
        level = self.degradation_level
        return {
            "max_tokens": self.max_tokens,
            "spent_tokens": spent,
            "remaining_tokens": max(0, self.max_tokens - spent) if self.max_tokens > 0 else None,
            "max_latency_ms": self.max_latency_ms,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "degraded_level": level,
            "fast_path": self.fast_path,
            "degraded_events": events,
            "budget_exceeded": level >= 3,
        }
