"""Metric collector implementations.

This module provides the base interface and context for implementing
metric collectors that traverse AST nodes during code analysis.

Example:
    from mcp_vector_search.analysis.collectors import MetricCollector, CollectorContext

    class MyCollector(MetricCollector):
        @property
        def name(self) -> str:
            return "my_collector"

        def collect_node(self, node, context, depth):
            # Process node
            pass

        def finalize_function(self, node, context):
            return {"my_metric": 42}
"""

from .base import CollectorContext, MetricCollector
from .complexity import (
    CognitiveComplexityCollector,
    CyclomaticComplexityCollector,
    MethodCountCollector,
    NestingDepthCollector,
    ParameterCountCollector,
)
from .coupling import (
    AfferentCouplingCollector,
    EfferentCouplingCollector,
    build_import_graph,
)

__all__ = [
    "CollectorContext",
    "MetricCollector",
    "CognitiveComplexityCollector",
    "CyclomaticComplexityCollector",
    "NestingDepthCollector",
    "ParameterCountCollector",
    "MethodCountCollector",
    "EfferentCouplingCollector",
    "AfferentCouplingCollector",
    "build_import_graph",
]
