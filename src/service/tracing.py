"""D3 可观测：请求级 trace（JSONL 落盘）+ 阶段计时。

trace 同时是 F3 错误归因（eval/stage_attribution.py v2）与线上排障的数据源。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

TRACE_DIR = Path(os.getenv("SERVICE_TRACE_DIR", "logs"))
TRACE_FILE = "service_trace.jsonl"


class StageTimer:
    """阶段计时上下文管理器：with 块耗时自动记入 trace dict。"""

    def __init__(self, trace: Dict, stage: str):
        self.trace = trace
        self.stage = stage
        self._start: Optional[float] = None

    def __enter__(self) -> "StageTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed_ms = round((time.perf_counter() - self._start) * 1000, 2)
        stages = self.trace.setdefault("stages_ms", {})
        stages[self.stage] = elapsed_ms
        if exc_type is not None:
            self.trace.setdefault("stage_errors", {})[self.stage] = f"{exc_type.__name__}: {exc}"
        return False  # 异常继续传播（服务层统一兜底）


def append_service_trace(entry: Dict) -> None:
    """trace 追加 JSONL；失败不阻断主流程（可观测性不得反噬可用性）。"""
    try:
        from datetime import datetime

        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        entry = {"timestamp": datetime.now().isoformat(timespec="seconds"), **entry}
        with open(TRACE_DIR / TRACE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"Warning: service trace 写入失败（不影响主流程）：{exc}")


def token_usage_of(answer_dict: Dict) -> Dict:
    """从 answer_dict 提取 token usage（api_requests 已回收 usage 数据）。"""
    response_data = answer_dict.get("response_data") or {}
    usage = response_data.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
    }
