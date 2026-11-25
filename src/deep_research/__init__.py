"""Lazy public API for the opt-in deep research pipeline."""

from importlib import import_module

_EXPORTS = {
    "ResearchTask": "planner",
    "ResearchDAG": "planner",
    "ResearchPlanner": "planner",
    "EvidenceLedger": "evidence_ledger",
    "ResearchSynthesizer": "synthesizer",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
