from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from src.retrieval_filters import RetrievalFilters


def kind_to_expected_answer_type(kind: str | None) -> str:
    mapping = {
        "number": "numeric",
        "boolean": "boolean",
        "name": "entity",
        "names": "entity_list",
        "comparative": "comparative",
    }
    return mapping.get((kind or "").lower(), "text")


@dataclass
class QueryPlan:
    original_query: str
    normalized_query: str
    search_queries: List[str]
    filters: RetrievalFilters
    route_mode: str
    expected_answer_type: str
    topic_flags: List[str] = field(default_factory=list)
    mentioned_companies: List[str] = field(default_factory=list)
    route_hints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["filters"] = self.filters.to_dict()
        return payload
