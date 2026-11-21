"""Protocol and registry for rule shortcut plugins."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol


class RulePlugin(Protocol):
    name: str

    def execute(self, processor: Any, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Attempt to answer deterministically; return None to fall through."""
        ...


class PluginRegistry:
    """Ordered registry of deterministic rule shortcut plugins."""

    def __init__(self, plugins: Optional[List[RulePlugin]] = None):
        self._plugins: List[RulePlugin] = list(plugins or [])

    def register(self, plugin: RulePlugin) -> None:
        self._plugins.append(plugin)

    def execute_first(self, processor: Any, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for plugin in self._plugins:
            result = plugin.execute(processor, ctx)
            if result is not None:
                return result
        return None
