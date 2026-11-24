"""Opt-in durable checkpoint and session memory."""

from importlib import import_module

__all__ = ["SqliteCheckpointSaver", "SessionMemory", "ContextCompactor"]

_EXPORTS = {
    "SqliteCheckpointSaver": ".sqlite_checkpointer",
    "SessionMemory": ".session_memory",
    "ContextCompactor": ".context_compactor",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
