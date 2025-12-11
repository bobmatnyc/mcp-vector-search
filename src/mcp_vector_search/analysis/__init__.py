"""Structural code analysis module.

This module provides dataclasses and interfaces for collecting and storing
code quality metrics during semantic code analysis.

Key Components:
    - ChunkMetrics: Metrics for individual functions/methods/classes
    - FileMetrics: Aggregated metrics for entire files
    - ProjectMetrics: Project-wide metric aggregates
    - MetricCollector: Abstract base class for metric collection
    - CollectorContext: Shared context during AST traversal

Example:
    # Create chunk metrics
    chunk = ChunkMetrics(
        cognitive_complexity=8,
        cyclomatic_complexity=5,
        max_nesting_depth=3,
        parameter_count=2,
        lines_of_code=25
    )

    # Chunk automatically computes grade
    assert chunk.complexity_grade == "B"  # 6-10 range

    # Store in ChromaDB-compatible format
    metadata = chunk.to_metadata()

    # Aggregate file metrics
    file_metrics = FileMetrics(
        file_path="src/module.py",
        chunks=[chunk]
    )
    file_metrics.compute_aggregates()

    # Project-wide analysis
    project = ProjectMetrics(project_root="/path/to/project")
    project.files["src/module.py"] = file_metrics
    project.compute_aggregates()
    hotspots = project.get_hotspots(limit=5)
"""

from .collectors.base import CollectorContext, MetricCollector
from .collectors.cohesion import (
    ClassCohesion,
    FileCohesion,
    LCOM4Calculator,
    MethodAttributeAccess,
    UnionFind,
)
from .collectors.complexity import (
    CognitiveComplexityCollector,
    CyclomaticComplexityCollector,
    MethodCountCollector,
    NestingDepthCollector,
    ParameterCountCollector,
)
from .collectors.coupling import (
    AfferentCouplingCollector,
    EfferentCouplingCollector,
    InstabilityCalculator,
    build_import_graph,
)
from .collectors.smells import CodeSmell, SmellDetector, SmellSeverity
from .metrics import ChunkMetrics, CouplingMetrics, FileMetrics, ProjectMetrics

__all__ = [
    "ChunkMetrics",
    "CouplingMetrics",
    "FileMetrics",
    "ProjectMetrics",
    "CollectorContext",
    "MetricCollector",
    "CognitiveComplexityCollector",
    "CyclomaticComplexityCollector",
    "NestingDepthCollector",
    "ParameterCountCollector",
    "MethodCountCollector",
    "EfferentCouplingCollector",
    "AfferentCouplingCollector",
    "InstabilityCalculator",
    "build_import_graph",
    "SmellDetector",
    "CodeSmell",
    "SmellSeverity",
    "ClassCohesion",
    "FileCohesion",
    "LCOM4Calculator",
    "MethodAttributeAccess",
    "UnionFind",
]
