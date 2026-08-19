from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

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
    task_type: str = "evidence_qa"
    route: str = "general_rag"
    operation: Optional[str] = None
    metric: Optional[str] = None
    metric_atom: Optional[Dict[str, Any]] = None
    scope: Dict[str, Any] = field(default_factory=dict)
    limit: Optional[int] = None
    sort_order: Optional[str] = None
    intent: Optional[str] = None
    evidence_targets: List[str] = field(default_factory=list)
    fallback_route: Optional[str] = None
    intent_confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["filters"] = self.filters.to_dict()
        return payload
