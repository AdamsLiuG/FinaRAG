"""Version-keyed L1 LRU + optional Redis L2 semantic cache for FinaRAG QA responses."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from src.prompts import PROMPT_TEMPLATE_VERSION
from src.text_normalization import normalize_text

_SNAPSHOT_PATH = Path(__file__).resolve().parents[2] / "config" / "atom_snapshot.json"


@dataclass(frozen=True)
class CacheVersions:
    corpus_version: str
    atom_schema_version: str
    prompt_version: str


def resolve_cache_versions(
    *,
    corpus_version: Optional[str] = None,
    snapshot_path: Optional[Path] = None,
) -> CacheVersions:
    """Resolve current corpus, FinaWiki atom snapshot, and prompt versions."""
    snap_file = snapshot_path or _SNAPSHOT_PATH
    atom_ver = "unknown"
    try:
        if snap_file.exists():
            data = json.loads(snap_file.read_text(encoding="utf-8"))
            schema_ver = str(data.get("schema_version") or "1.0")
            sources = data.get("sources") or {}
            m_sha = str((sources.get("metric_atoms") or {}).get("sha256") or "")[:12]
            e_sha = str((sources.get("event_atoms") or {}).get("sha256") or "")[:12]
            atom_ver = f"{schema_ver}:{m_sha}:{e_sha}"
    except Exception:
        atom_ver = "unknown"

    c_ver = str(corpus_version or os.getenv("FINARAG_CORPUS_VERSION", "v1")).strip() or "v1"
    return CacheVersions(
        corpus_version=c_ver,
        atom_schema_version=atom_ver,
        prompt_version=PROMPT_TEMPLATE_VERSION,
    )


class SemanticCache:
    """Thread-safe versioned response cache with L1 bounded LRU and optional L2 Redis."""

    def __init__(
        self,
        *,
        max_entries: int = 512,
        ttl_seconds: float = 3600.0,
        versions: Optional[CacheVersions] = None,
        redis_url: Optional[str] = None,
    ):
        self.max_entries = max(1, int(max_entries))
        self.ttl_seconds = max(1.0, float(ttl_seconds))
        self.versions = versions or resolve_cache_versions()
        self._lock = threading.Lock()
        self._entries: "OrderedDict[str, tuple[float, Dict[str, Any]]]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._redis = self._init_redis(redis_url if redis_url is not None else os.getenv("REDIS_URL", ""))

    @staticmethod
    def _init_redis(raw_url: str):
        url = str(raw_url or "").strip()
        if not url:
            return None
        parsed = urlparse(url)
        if parsed.scheme not in {"redis", "rediss"}:
            return None
        try:
            import redis  # type: ignore

            return redis.Redis.from_url(url, socket_timeout=1.5, socket_connect_timeout=1.5)
        except Exception:
            return None

    def make_key(
        self,
        question: str,
        schema: str,
        *,
        versions: Optional[CacheVersions] = None,
    ) -> str:
        ver = versions or self.versions
        payload = {
            "corpus_version": ver.corpus_version,
            "atom_schema_version": ver.atom_schema_version,
            "prompt_version": ver.prompt_version,
            "schema": str(schema or "string").strip().lower(),
            "query": normalize_text(question or ""),
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        return f"finarag:qa:{digest}"

    def get(
        self,
        question: str,
        schema: str,
        *,
        versions: Optional[CacheVersions] = None,
    ) -> Optional[Dict[str, Any]]:
        key = self.make_key(question, schema, versions=versions)
        now = time.monotonic()
        with self._lock:
            item = self._entries.get(key)
            if item is not None:
                expires_at, payload = item
                if expires_at <= now:
                    del self._entries[key]
                else:
                    self._entries.move_to_end(key)
                    self.hits += 1
                    return copy.deepcopy(payload)

        if self._redis is not None:
            try:
                raw = self._redis.get(key)
                if raw:
                    decoded = json.loads(raw)
                    with self._lock:
                        self._store_l1_unlocked(key, decoded, now + self.ttl_seconds)
                        self.hits += 1
                    return copy.deepcopy(decoded)
            except Exception:
                pass

        with self._lock:
            self.misses += 1
        return None

    def set(
        self,
        question: str,
        schema: str,
        payload: Dict[str, Any],
        *,
        versions: Optional[CacheVersions] = None,
    ) -> str:
        key = self.make_key(question, schema, versions=versions)
        clean_payload = copy.deepcopy(payload)
        expires_at = time.monotonic() + self.ttl_seconds
        with self._lock:
            self._store_l1_unlocked(key, clean_payload, expires_at)

        if self._redis is not None:
            try:
                self._redis.setex(key, int(self.ttl_seconds), json.dumps(clean_payload, ensure_ascii=False))
            except Exception:
                pass
        return key

    def _store_l1_unlocked(self, key: str, payload: Dict[str, Any], expires_at: float) -> None:
        if key in self._entries:
            self._entries.move_to_end(key)
            self._entries[key] = (expires_at, payload)
            return
        self._entries[key] = (expires_at, payload)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self.evictions += 1

    def update_versions(self, versions: CacheVersions) -> None:
        with self._lock:
            self.versions = versions

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "size": len(self._entries),
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "redis_enabled": self._redis is not None,
                "versions": {
                    "corpus_version": self.versions.corpus_version,
                    "atom_schema_version": self.versions.atom_schema_version,
                    "prompt_version": self.versions.prompt_version,
                },
            }
