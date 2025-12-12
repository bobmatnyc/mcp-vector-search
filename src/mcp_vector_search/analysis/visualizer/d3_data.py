"""Transform analysis data to D3.js-friendly format for visualization.

This module converts AnalysisExport schema data into a format optimized for
D3.js force-directed graph visualization. It handles:
- Node transformation (files with metrics)
- Edge transformation (dependencies with coupling strength)
- Circular dependency detection and highlighting
- Module grouping for visual organization

The output format is designed for interactive dependency graphs with:
- Node size based on lines of code
- Node fill based on cognitive complexity (grayscale)
- Node border based on code smell severity (red scale)
- Edge thickness based on coupling strength
- Circular dependencies highlighted in red

Example:
    >>> from mcp_vector_search.analysis.visualizer import JSONExporter
    >>> from pathlib import Path
    >>>
    >>> exporter = JSONExporter(project_root=Path("/path/to/project"))
    >>> export = exporter.export(project_metrics)
    >>>
    >>> from mcp_vector_search.analysis.visualizer.d3_data import transform_for_d3
    >>> d3_data = transform_for_d3(export)
    >>> # Returns: {"nodes": [...], "links": [...], "summary": {...}}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import AnalysisExport, FileDetail


@dataclass
class D3Node:
    """Node data for D3 force graph.

    Represents a single file in the codebase with visual properties
    derived from code metrics.

    Attributes:
        id: Unique file path (relative to project root)
        label: Display name (file name only)
        module: Directory/module name for grouping
        module_path: Full module path for cluster grouping (e.g., 'src/analysis')
        loc: Lines of code (determines node size)
        complexity: Cognitive complexity (determines fill color)
        smell_count: Number of code smells detected
        smell_severity: Worst smell severity level
        cyclomatic_complexity: Cyclomatic complexity score
        function_count: Number of functions in file
        class_count: Number of classes in file
        smells: List of smell details for detail panel
        imports: List of imports (outgoing edges)
    """

    id: str
    label: str
    module: str
    module_path: str
    loc: int
    complexity: float
    smell_count: int
    smell_severity: str
    cyclomatic_complexity: int
    function_count: int
    class_count: int
    smells: list[dict[str, Any]]
    imports: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "label": self.label,
            "module": self.module,
            "module_path": self.module_path,
            "loc": self.loc,
            "complexity": self.complexity,
            "smell_count": self.smell_count,
            "smell_severity": self.smell_severity,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "function_count": self.function_count,
            "class_count": self.class_count,
            "smells": self.smells,
            "imports": self.imports,
        }


@dataclass
class D3Edge:
    """Edge data for D3 force graph.

    Represents a dependency relationship between two files with
    visual properties derived from coupling metrics.

    Attributes:
        source: Source file path (relative to project root)
        target: Target file path
        coupling: Coupling strength (number of imports/dependencies)
        circular: Whether this edge is part of a circular dependency
    """

    source: str
    target: str
    coupling: int
    circular: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "source": self.source,
            "target": self.target,
            "coupling": self.coupling,
            "circular": self.circular,
        }


def transform_for_d3(export: AnalysisExport) -> dict[str, Any]:
    """Transform AnalysisExport to D3-friendly JSON structure.

    Creates a graph structure optimized for D3.js force-directed layout
    with visual properties encoded in node and edge attributes.

    Visual Encodings:
    - Node size: Lines of code (LOC)
    - Node fill: Cognitive complexity (grayscale: light to dark)
        - 0-5: Very light gray (#f3f4f6)
        - 6-10: Light gray (#9ca3af)
        - 11-20: Medium gray (#4b5563)
        - 21-30: Dark gray (#1f2937)
        - 31+: Very dark gray (#111827)
    - Node border: Code smell severity (red scale)
        - none: Light gray (#e5e7eb)
        - info: Light red (#fca5a5)
        - warning: Medium red (#f87171)
        - error: Dark red (#ef4444)
        - critical: Very dark red (#dc2626) with glow
    - Edge thickness: Coupling strength (number of imports)
    - Edge color: Red if circular dependency, gray otherwise

    Args:
        export: Complete analysis export with files and dependencies

    Returns:
        Dictionary containing:
        - nodes: List of node objects with visualization properties
        - links: List of edge objects with source, target, coupling
        - modules: List of module cluster definitions for hulls
        - summary: Project summary statistics for context

    Example:
        >>> d3_data = transform_for_d3(export)
        >>> d3_data.keys()
        dict_keys(['nodes', 'links', 'modules', 'summary'])
        >>> len(d3_data['nodes'])
        42
        >>> d3_data['nodes'][0]
        {'id': 'src/main.py', 'label': 'main.py', ...}
    """
    # Create nodes from files
    nodes = [_create_node(file) for file in export.files]

    # Identify circular dependency paths
    circular_paths = _extract_circular_paths(export)

    # Create edges from dependency graph
    links = _create_edges(export, circular_paths)

    # Group nodes by module for cluster hulls
    modules = _create_module_groups(nodes)

    # Calculate detailed statistics for dashboard panels
    summary = _create_summary_stats(export, nodes, links)

    return {
        "nodes": [n.to_dict() for n in nodes],
        "links": [e.to_dict() for e in links],
        "modules": modules,
        "summary": summary,
    }


def _create_summary_stats(
    export: AnalysisExport, nodes: list[D3Node], links: list[D3Edge]
) -> dict[str, Any]:
    """Create detailed summary statistics for dashboard panels.

    Calculates comprehensive statistics including:
    - Basic counts (files, functions, classes)
    - Complexity metrics with grade distribution
    - Smell breakdown by severity
    - LOC distribution statistics
    - Circular dependency information
    - Complexity level distribution for legend

    Args:
        export: Complete analysis export
        nodes: List of D3Node objects
        links: List of D3Edge objects

    Returns:
        Dictionary containing detailed summary statistics
    """
    # Basic counts from export summary
    basic_stats = {
        "total_files": export.summary.total_files,
        "total_functions": export.summary.total_functions,
        "total_classes": export.summary.total_classes,
        "total_lines": export.summary.total_lines,
        "circular_dependencies": export.summary.circular_dependencies,
    }

    # Complexity statistics with grade
    avg_complexity = export.summary.avg_cognitive_complexity
    complexity_grade = _get_complexity_grade(avg_complexity)

    complexity_stats = {
        "avg_complexity": avg_complexity,
        "avg_cyclomatic_complexity": export.summary.avg_complexity,
        "complexity_grade": complexity_grade,
    }

    # Smell breakdown by severity
    smells_by_severity = export.summary.smells_by_severity or {}
    smell_stats = {
        "total_smells": export.summary.total_smells,
        "error_count": smells_by_severity.get("error", 0),
        "warning_count": smells_by_severity.get("warning", 0),
        "info_count": smells_by_severity.get("info", 0),
    }

    # LOC distribution statistics
    if export.files:
        locs = [f.lines_of_code for f in export.files]
        loc_stats = {
            "min_loc": min(locs),
            "max_loc": max(locs),
            "median_loc": sorted(locs)[len(locs) // 2],
            "total_loc": sum(locs),
        }
    else:
        loc_stats = {"min_loc": 0, "max_loc": 0, "median_loc": 0, "total_loc": 0}

    # Complexity level distribution for legend (count nodes per level)
    complexity_distribution = {
        "low": 0,  # 0-5
        "moderate": 0,  # 6-10
        "high": 0,  # 11-20
        "very_high": 0,  # 21-30
        "critical": 0,  # 31+
    }

    for node in nodes:
        level = get_complexity_class(node.complexity)
        # Map to underscore version for consistency
        level_key = level.replace("-", "_")
        if level_key in complexity_distribution:
            complexity_distribution[level_key] += 1

    # Smell severity distribution for legend (count nodes per severity)
    smell_distribution = {
        "none": 0,
        "info": 0,
        "warning": 0,
        "error": 0,
        "critical": 0,
    }

    for node in nodes:
        severity = node.smell_severity
        if severity in smell_distribution:
            smell_distribution[severity] += 1

    return {
        **basic_stats,
        **complexity_stats,
        **smell_stats,
        **loc_stats,
        "complexity_distribution": complexity_distribution,
        "smell_distribution": smell_distribution,
    }


def _get_complexity_grade(avg_complexity: float) -> str:
    """Get letter grade from average complexity score.

    Args:
        avg_complexity: Average cognitive complexity score

    Returns:
        Letter grade (A, B, C, D, or F)
    """
    if avg_complexity <= 5:
        return "A"
    elif avg_complexity <= 10:
        return "B"
    elif avg_complexity <= 20:
        return "C"
    elif avg_complexity <= 30:
        return "D"
    else:
        return "F"


def _create_node(file: FileDetail) -> D3Node:
    """Create a D3Node from FileDetail.

    Args:
        file: File metrics from analysis export

    Returns:
        D3Node with visual properties derived from metrics
    """
    file_path = Path(file.path)
    label = file_path.name
    module = file_path.parent.name if file_path.parent.name else "root"
    module_path = str(file_path.parent) if file_path.parent.name else "root"

    # Calculate worst smell severity
    smell_severity = _calculate_worst_severity(file)

    # Convert smells to dictionaries for JSON serialization
    smells_data = [
        {
            "type": smell.smell_type,
            "severity": smell.severity,
            "message": smell.message,
            "line": smell.line,
        }
        for smell in file.smells
    ]

    return D3Node(
        id=file.path,
        label=label,
        module=module,
        module_path=module_path,
        loc=file.lines_of_code,
        complexity=file.cognitive_complexity,
        smell_count=len(file.smells),
        smell_severity=smell_severity,
        cyclomatic_complexity=file.cyclomatic_complexity,
        function_count=file.function_count,
        class_count=file.class_count,
        smells=smells_data,
        imports=file.imports or [],
    )


def _calculate_worst_severity(file: FileDetail) -> str:
    """Calculate worst smell severity for a file.

    Severity levels in order of severity (low to high):
    - none: No smells detected
    - info: Informational smells only
    - warning: Warning-level smells
    - error: Error-level smells
    - critical: Critical smells (not in current schema, reserved for future)

    Args:
        file: File detail with smell information

    Returns:
        Worst severity level as string
    """
    if not file.smells:
        return "none"

    severity_order = {"info": 1, "warning": 2, "error": 3}
    worst_severity = "none"
    worst_level = 0

    for smell in file.smells:
        level = severity_order.get(smell.severity, 0)
        if level > worst_level:
            worst_level = level
            worst_severity = smell.severity

    return worst_severity


def _extract_circular_paths(export: AnalysisExport) -> set[tuple[str, str]]:
    """Extract all edges that are part of circular dependencies.

    Creates a set of (source, target) tuples representing edges
    that participate in any circular dependency cycle.

    Args:
        export: Analysis export with circular dependency data

    Returns:
        Set of (source, target) tuples for circular edges
    """
    circular_edges: set[tuple[str, str]] = set()

    for cycle in export.dependencies.circular_dependencies:
        # For each cycle, mark all edges in the cycle as circular
        for i in range(len(cycle.cycle)):
            source = cycle.cycle[i]
            target = cycle.cycle[(i + 1) % len(cycle.cycle)]
            circular_edges.add((source, target))

    return circular_edges


def _create_edges(
    export: AnalysisExport, circular_paths: set[tuple[str, str]]
) -> list[D3Edge]:
    """Create D3Edges from dependency graph.

    Args:
        export: Analysis export with dependency graph
        circular_paths: Set of (source, target) tuples that are circular

    Returns:
        List of D3Edge objects with coupling and circularity info
    """
    edges: list[D3Edge] = []

    # Count coupling strength (number of imports between each pair)
    coupling_counts: dict[tuple[str, str], int] = {}

    for edge in export.dependencies.edges:
        key = (edge.source, edge.target)
        coupling_counts[key] = coupling_counts.get(key, 0) + 1

    # Create D3 edges with coupling strength and circularity flag
    for (source, target), coupling in coupling_counts.items():
        is_circular = (source, target) in circular_paths
        edges.append(
            D3Edge(
                source=source, target=target, coupling=coupling, circular=is_circular
            )
        )

    return edges


def get_complexity_class(complexity: float) -> str:
    """Get CSS class name for complexity level.

    Maps complexity scores to CSS class names for styling.

    Complexity Thresholds:
    - 0-5: low (very light gray)
    - 6-10: moderate (light gray)
    - 11-20: high (medium gray)
    - 21-30: very-high (dark gray)
    - 31+: critical (very dark gray)

    Args:
        complexity: Cognitive complexity score

    Returns:
        CSS class suffix (e.g., "low", "critical")
    """
    if complexity <= 5:
        return "low"
    elif complexity <= 10:
        return "moderate"
    elif complexity <= 20:
        return "high"
    elif complexity <= 30:
        return "very-high"
    else:
        return "critical"


def get_smell_class(severity: str) -> str:
    """Get CSS class name for smell severity.

    Maps smell severity to CSS class names for border styling.

    Args:
        severity: Smell severity level (none, info, warning, error, critical)

    Returns:
        CSS class name (e.g., "smell-none", "smell-error")
    """
    return f"smell-{severity}"


def _create_module_groups(nodes: list[D3Node]) -> list[dict[str, Any]]:
    """Group nodes by module path for cluster visualization.

    Creates module cluster definitions that can be used to draw
    convex hull polygons around related nodes.

    Args:
        nodes: List of D3Node objects

    Returns:
        List of module group dictionaries containing:
        - name: Module path identifier
        - node_ids: List of node IDs belonging to this module
        - color: Hex color for the module cluster
    """
    # Group nodes by module_path
    module_map: dict[str, list[str]] = {}
    for node in nodes:
        if node.module_path not in module_map:
            module_map[node.module_path] = []
        module_map[node.module_path].append(node.id)

    # Assign colors to modules (cycle through a palette)
    colors = [
        "#3b82f6",  # Blue
        "#10b981",  # Green
        "#f59e0b",  # Orange
        "#8b5cf6",  # Purple
        "#ec4899",  # Pink
        "#14b8a6",  # Teal
        "#f97316",  # Orange-red
        "#06b6d4",  # Cyan
    ]

    modules = []
    for idx, (module_path, node_ids) in enumerate(sorted(module_map.items())):
        # Only create module groups with 2+ nodes for meaningful hulls
        if len(node_ids) >= 2:
            modules.append(
                {
                    "name": module_path,
                    "node_ids": node_ids,
                    "color": colors[idx % len(colors)],
                }
            )

    return modules
