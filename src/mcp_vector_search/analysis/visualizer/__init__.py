"""Visualization and export schemas for code analysis results.

This module provides the JSON export format for structural code analysis,
enabling integration with visualization tools and external analysis platforms.

The export schema is versioned and designed to be stable across tool updates,
with support for:
- Complete metric snapshots (complexity, coupling, smells)
- Dependency graph visualization
- Historical trend tracking
- Git-aware context for time-series analysis

Example:
    >>> from mcp_vector_search.analysis.visualizer.schemas import (
    ...     AnalysisExport,
    ...     ExportMetadata,
    ...     MetricsSummary,
    ...     generate_json_schema
    ... )
    >>> from datetime import datetime
    >>>
    >>> # Create export
    >>> export = AnalysisExport(
    ...     metadata=ExportMetadata(
    ...         version="1.0.0",
    ...         generated_at=datetime.now(),
    ...         tool_version="0.19.0",
    ...         project_root="/path/to/project"
    ...     ),
    ...     summary=MetricsSummary(...),
    ...     files=[],
    ...     dependencies=DependencyGraph(edges=[], circular_dependencies=[])
    ... )
    >>>
    >>> # Export to JSON
    >>> json_data = export.model_dump_json(indent=2)
    >>>
    >>> # Generate schema for documentation
    >>> schema = generate_json_schema()
"""

from .d3_data import D3Edge, D3Node, transform_for_d3
from .exporter import JSONExporter
from .html_report import HTMLReportGenerator
from .schemas import (
    AnalysisExport,
    ClassMetrics,
    CyclicDependency,
    DependencyEdge,
    DependencyGraph,
    ExportMetadata,
    FileDetail,
    FunctionMetrics,
    MetricsSummary,
    MetricTrend,
    SmellLocation,
    TrendData,
    TrendDataPoint,
    generate_json_schema,
)

__all__ = [
    # Exporters
    "JSONExporter",
    "HTMLReportGenerator",
    # D3 visualization
    "D3Node",
    "D3Edge",
    "transform_for_d3",
    # Main export schema
    "AnalysisExport",
    # Metadata and summary
    "ExportMetadata",
    "MetricsSummary",
    # File-level schemas
    "FileDetail",
    "FunctionMetrics",
    "ClassMetrics",
    "SmellLocation",
    # Dependency analysis
    "DependencyGraph",
    "DependencyEdge",
    "CyclicDependency",
    # Trend tracking
    "TrendData",
    "MetricTrend",
    "TrendDataPoint",
    # Utilities
    "generate_json_schema",
]
