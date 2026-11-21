"""Rule-based fast-path plugins extracted from QuestionsProcessor."""

from src.graph.plugins.base import PluginRegistry, RulePlugin
from src.graph.plugins.cash_dividend import CashDividendPlugin
from src.graph.plugins.legal_rep import LegalRepresentativePlugin
from src.graph.plugins.metadata_agg import (
    MetadataCountPlugin,
    MetadataMembershipPlugin,
    MetadataNamesPlugin,
)

__all__ = [
    "CashDividendPlugin",
    "LegalRepresentativePlugin",
    "MetadataCountPlugin",
    "MetadataMembershipPlugin",
    "MetadataNamesPlugin",
    "PluginRegistry",
    "RulePlugin",
]
