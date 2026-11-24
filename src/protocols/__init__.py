"""Optional FinaRAG protocol bindings."""

from importlib import import_module

_EXPORTS = {
    "MCPToolRegistry": ".tool_registry",
    "ToolPrincipal": ".tool_registry",
    "MCPServer": ".mcp_server",
    "A2ATaskStore": ".a2a_server",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
