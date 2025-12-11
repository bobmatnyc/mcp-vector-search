"""JSON export schema for structural code analysis results.

This module provides Pydantic v2 models that define the JSON export format
for comprehensive code analysis results, including metrics, smells, dependencies,
and trend data.

The schema is designed to be:
- Version-stable: Includes schema version for compatibility
- Complete: Captures all analysis aspects (metrics, smells, dependencies, trends)
- Tool-agnostic: Can be consumed by various visualization and analysis tools
- Git-aware: Tracks commit and branch information for historical analysis

Example:
    >>> from datetime import datetime
    >>> from pathlib import Path
    >>> metadata = ExportMetadata(
    ...     version="1.0.0",
    ...     generated_at=datetime.now(),
    ...     tool_version="0.19.0",
    ...     project_root="/path/to/project",
    ...     git_commit="abc123",
    ...     git_branch="main"
    ... )
    >>> export = AnalysisExport(
    ...     metadata=metadata,
    ...     summary=MetricsSummary(...),
    ...     files=[],
    ...     dependencies=DependencyGraph(edges=[], circular_dependencies=[])
    ... )
    >>> json_output = export.model_dump_json(indent=2)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ExportMetadata(BaseModel):
    """Metadata about the export itself.

    Tracks version information, generation timestamp, and git context
    to enable historical comparison and tool compatibility checks.

    Attributes:
        version: Schema version (e.g., "1.0.0") for compatibility tracking
        generated_at: UTC timestamp when export was generated
        tool_version: mcp-vector-search version that generated the export
        project_root: Absolute path to project root directory
        git_commit: Git commit SHA if available (optional)
        git_branch: Git branch name if available (optional)
    """

    version: str = Field(
        default="1.0.0", description="Schema version for compatibility tracking"
    )
    generated_at: datetime = Field(
        description="UTC timestamp when analysis was performed"
    )
    tool_version: str = Field(
        description="Version of mcp-vector-search that generated this export"
    )
    project_root: str = Field(description="Absolute path to project root directory")
    git_commit: str | None = Field(
        default=None, description="Git commit SHA (if available)"
    )
    git_branch: str | None = Field(
        default=None, description="Git branch name (if available)"
    )


class MetricsSummary(BaseModel):
    """Project-level summary statistics.

    Aggregates key metrics across the entire codebase to provide
    a high-level health overview and identify areas needing attention.

    Attributes:
        total_files: Total number of analyzed files
        total_functions: Total number of functions/methods across all files
        total_classes: Total number of classes
        total_lines: Total lines of code (excluding blank lines)
        avg_complexity: Average cyclomatic complexity across all functions
        avg_cognitive_complexity: Average cognitive complexity
        avg_nesting_depth: Average maximum nesting depth
        total_smells: Total number of detected code smells
        smells_by_severity: Distribution of smells by severity level
        avg_instability: Average instability metric (optional, Phase 3)
        circular_dependencies: Count of circular dependency cycles
        avg_halstead_volume: Average Halstead volume (optional, future)
        avg_halstead_difficulty: Average Halstead difficulty (optional, future)
        estimated_debt_minutes: Estimated technical debt in minutes (optional)
    """

    total_files: int = Field(ge=0, description="Total number of analyzed files")
    total_functions: int = Field(ge=0, description="Total number of functions/methods")
    total_classes: int = Field(ge=0, description="Total number of classes")
    total_lines: int = Field(ge=0, description="Total lines of code")
    avg_complexity: float = Field(ge=0.0, description="Average cyclomatic complexity")
    avg_cognitive_complexity: float = Field(
        ge=0.0, description="Average cognitive complexity"
    )
    avg_nesting_depth: float = Field(
        ge=0.0, description="Average maximum nesting depth"
    )
    total_smells: int = Field(ge=0, description="Total number of code smells detected")
    smells_by_severity: dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of smells by severity (error, warning, info)",
    )

    # Coupling metrics (Phase 3)
    avg_instability: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Average instability metric (0-1)"
    )
    circular_dependencies: int = Field(
        default=0, ge=0, description="Number of circular dependency cycles"
    )

    # Halstead metrics (future, optional)
    avg_halstead_volume: float | None = Field(
        default=None, ge=0.0, description="Average Halstead volume (optional)"
    )
    avg_halstead_difficulty: float | None = Field(
        default=None, ge=0.0, description="Average Halstead difficulty (optional)"
    )

    # Technical debt estimation
    estimated_debt_minutes: int | None = Field(
        default=None, ge=0, description="Estimated technical debt in minutes (optional)"
    )


class FunctionMetrics(BaseModel):
    """Metrics for a single function/method.

    Captures complexity and size metrics for individual functions
    to identify refactoring candidates.

    Attributes:
        name: Function/method name
        line_start: Starting line number
        line_end: Ending line number
        cyclomatic_complexity: Cyclomatic complexity score
        cognitive_complexity: Cognitive complexity score
        nesting_depth: Maximum nesting depth
        parameter_count: Number of parameters
        lines_of_code: Total lines in function body
        halstead_volume: Halstead volume (optional, future)
        halstead_difficulty: Halstead difficulty (optional, future)
        halstead_effort: Halstead effort (optional, future)
    """

    name: str = Field(description="Function or method name")
    line_start: int = Field(ge=1, description="Starting line number")
    line_end: int = Field(ge=1, description="Ending line number")
    cyclomatic_complexity: int = Field(ge=1, description="Cyclomatic complexity score")
    cognitive_complexity: int = Field(ge=0, description="Cognitive complexity score")
    nesting_depth: int = Field(ge=0, description="Maximum nesting depth")
    parameter_count: int = Field(ge=0, description="Number of parameters")
    lines_of_code: int = Field(ge=1, description="Total lines in function")

    # Halstead metrics (optional, future)
    halstead_volume: float | None = Field(
        default=None, ge=0.0, description="Halstead volume (optional)"
    )
    halstead_difficulty: float | None = Field(
        default=None, ge=0.0, description="Halstead difficulty (optional)"
    )
    halstead_effort: float | None = Field(
        default=None, ge=0.0, description="Halstead effort (optional)"
    )

    @field_validator("line_end")
    @classmethod
    def validate_line_range(cls, v: int, info: Any) -> int:
        """Ensure line_end >= line_start."""
        if "line_start" in info.data and v < info.data["line_start"]:
            raise ValueError("line_end must be >= line_start")
        return v


class ClassMetrics(BaseModel):
    """Metrics for a single class.

    Tracks class-level metrics including cohesion and method counts
    to identify classes that violate single responsibility principle.

    Attributes:
        name: Class name
        line_start: Starting line number
        line_end: Ending line number
        method_count: Number of methods in class
        lcom4: Lack of Cohesion of Methods metric (optional, Phase 2)
        methods: List of method metrics
    """

    name: str = Field(description="Class name")
    line_start: int = Field(ge=1, description="Starting line number")
    line_end: int = Field(ge=1, description="Ending line number")
    method_count: int = Field(ge=0, description="Number of methods")
    lcom4: int | None = Field(
        default=None, ge=1, description="Lack of Cohesion metric (LCOM4)"
    )
    methods: list[FunctionMetrics] = Field(
        default_factory=list, description="Metrics for each method"
    )

    @field_validator("line_end")
    @classmethod
    def validate_line_range(cls, v: int, info: Any) -> int:
        """Ensure line_end >= line_start."""
        if "line_start" in info.data and v < info.data["line_start"]:
            raise ValueError("line_end must be >= line_start")
        return v


class SmellLocation(BaseModel):
    """A detected code smell.

    Represents a single code smell instance with location, severity,
    and remediation information for visualization and prioritization.

    Attributes:
        smell_type: Type of smell (e.g., "long_method", "deep_nesting")
        severity: Severity level ("error", "warning", "info")
        message: Human-readable description
        line: Starting line number
        column: Starting column (optional)
        end_line: Ending line number (optional)
        function_name: Function where smell occurs (optional)
        class_name: Class where smell occurs (optional)
        remediation_minutes: Estimated time to fix (optional)
    """

    smell_type: str = Field(description="Type of code smell")
    severity: str = Field(description="Severity level (error, warning, info)")
    message: str = Field(description="Human-readable description")
    line: int = Field(ge=1, description="Starting line number")
    column: int | None = Field(default=None, ge=0, description="Starting column")
    end_line: int | None = Field(default=None, ge=1, description="Ending line number")
    function_name: str | None = Field(
        default=None, description="Function where smell occurs"
    )
    class_name: str | None = Field(default=None, description="Class where smell occurs")
    remediation_minutes: int | None = Field(
        default=None, ge=0, description="Estimated remediation time in minutes"
    )

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Ensure severity is one of the allowed values."""
        allowed = {"error", "warning", "info"}
        if v not in allowed:
            raise ValueError(f"severity must be one of {allowed}, got {v}")
        return v


class FileDetail(BaseModel):
    """Complete metrics for a single file.

    Comprehensive file-level metrics including complexity, coupling,
    and detailed function/class breakdowns for drill-down analysis.

    Attributes:
        path: Relative path from project root
        language: Programming language detected
        lines_of_code: Total lines of code (excluding blanks)
        cyclomatic_complexity: Sum of cyclomatic complexity
        cognitive_complexity: Sum of cognitive complexity
        max_nesting_depth: Maximum nesting depth in file
        function_count: Number of top-level functions
        class_count: Number of classes
        efferent_coupling: Outgoing dependencies (files this depends on)
        afferent_coupling: Incoming dependencies (files depending on this)
        instability: Instability metric (Ce / (Ce + Ca))
        functions: Metrics for each function
        classes: Metrics for each class
        smells: Detected code smells
        imports: List of imported modules/files
    """

    path: str = Field(description="Relative path from project root")
    language: str = Field(description="Programming language")
    lines_of_code: int = Field(ge=0, description="Total lines of code")

    # Aggregate complexity metrics
    cyclomatic_complexity: int = Field(ge=0, description="Sum of cyclomatic complexity")
    cognitive_complexity: int = Field(ge=0, description="Sum of cognitive complexity")
    max_nesting_depth: int = Field(ge=0, description="Maximum nesting depth")
    function_count: int = Field(ge=0, description="Number of functions")
    class_count: int = Field(ge=0, description="Number of classes")

    # Coupling metrics (Phase 3)
    efferent_coupling: int = Field(ge=0, description="Outgoing dependencies")
    afferent_coupling: int = Field(ge=0, description="Incoming dependencies")
    instability: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Instability metric (0-1)"
    )

    # Collections
    functions: list[FunctionMetrics] = Field(
        default_factory=list, description="Function-level metrics"
    )
    classes: list[ClassMetrics] = Field(
        default_factory=list, description="Class-level metrics"
    )
    smells: list[SmellLocation] = Field(
        default_factory=list, description="Detected code smells"
    )
    imports: list[str] = Field(
        default_factory=list, description="Imported modules/files"
    )


class DependencyEdge(BaseModel):
    """An edge in the dependency graph.

    Represents a single import/dependency relationship between files
    to enable dependency analysis and visualization.

    Attributes:
        source: Source file path (relative to project root)
        target: Target file/module path
        import_type: Type of import ("import", "from_import", "dynamic")
    """

    source: str = Field(description="Source file path")
    target: str = Field(description="Target file/module")
    import_type: str = Field(description="Import type (import, from_import, dynamic)")

    @field_validator("import_type")
    @classmethod
    def validate_import_type(cls, v: str) -> str:
        """Ensure import_type is one of the allowed values."""
        allowed = {"import", "from_import", "dynamic"}
        if v not in allowed:
            raise ValueError(f"import_type must be one of {allowed}, got {v}")
        return v


class CyclicDependency(BaseModel):
    """A detected circular dependency.

    Represents a cycle in the dependency graph that should be resolved
    to improve modularity and testability.

    Attributes:
        cycle: Ordered list of file paths forming the cycle
        length: Number of files in the cycle
    """

    cycle: list[str] = Field(
        min_length=2, description="List of files in the cycle (ordered)"
    )
    length: int = Field(ge=2, description="Number of files in cycle")

    @field_validator("length")
    @classmethod
    def validate_length_matches_cycle(cls, v: int, info: Any) -> int:
        """Ensure length matches actual cycle length."""
        if "cycle" in info.data and v != len(info.data["cycle"]):
            raise ValueError(
                f"length {v} does not match cycle length {len(info.data['cycle'])}"
            )
        return v


class DependencyGraph(BaseModel):
    """Project dependency structure.

    Represents the complete dependency graph including edges,
    circular dependencies, and coupling hotspots.

    Attributes:
        edges: All dependency edges in the graph
        circular_dependencies: Detected circular dependency cycles
        most_depended_on: Top files by afferent coupling (file, count)
        most_dependent: Top files by efferent coupling (file, count)
    """

    edges: list[DependencyEdge] = Field(
        default_factory=list, description="All dependency edges"
    )
    circular_dependencies: list[CyclicDependency] = Field(
        default_factory=list, description="Detected circular dependencies"
    )
    most_depended_on: list[tuple[str, int]] = Field(
        default_factory=list,
        description="Top files by afferent coupling (incoming dependencies)",
    )
    most_dependent: list[tuple[str, int]] = Field(
        default_factory=list,
        description="Top files by efferent coupling (outgoing dependencies)",
    )


class TrendDataPoint(BaseModel):
    """A single point in trend history.

    Represents one measurement of a metric at a specific point in time,
    enabling trend analysis and regression detection.

    Attributes:
        timestamp: When the measurement was taken (UTC)
        commit: Git commit SHA (optional)
        value: Metric value at this point
    """

    timestamp: datetime = Field(description="When measurement was taken (UTC)")
    commit: str | None = Field(default=None, description="Git commit SHA")
    value: float = Field(description="Metric value at this point")


class MetricTrend(BaseModel):
    """Trend data for a specific metric.

    Tracks how a metric changes over time with direction indicators
    to highlight improving or worsening trends.

    Attributes:
        metric_name: Name of the metric being tracked
        current_value: Most recent value
        previous_value: Previous measurement value (optional)
        change_percent: Percentage change from previous (optional)
        trend_direction: Direction ("improving", "worsening", "stable")
        history: Time series of historical values
    """

    metric_name: str = Field(description="Name of metric being tracked")
    current_value: float = Field(description="Current metric value")
    previous_value: float | None = Field(
        default=None, description="Previous measurement value"
    )
    change_percent: float | None = Field(
        default=None, description="Percentage change from previous"
    )
    trend_direction: str = Field(
        description="Trend direction (improving, worsening, stable)"
    )
    history: list[TrendDataPoint] = Field(
        default_factory=list, description="Historical data points"
    )

    @field_validator("trend_direction")
    @classmethod
    def validate_trend_direction(cls, v: str) -> str:
        """Ensure trend_direction is one of the allowed values."""
        allowed = {"improving", "worsening", "stable"}
        if v not in allowed:
            raise ValueError(f"trend_direction must be one of {allowed}, got {v}")
        return v


class TrendData(BaseModel):
    """Historical trend information.

    Aggregates trend data for multiple metrics with baseline tracking
    for regression detection and historical comparison.

    Attributes:
        metrics: List of metric trends
        baseline_name: Name of baseline (e.g., "main", "v1.0.0")
        baseline_date: Date baseline was established (optional)
    """

    metrics: list[MetricTrend] = Field(
        default_factory=list, description="Trend data for each tracked metric"
    )
    baseline_name: str | None = Field(
        default=None, description="Name of baseline for comparison"
    )
    baseline_date: datetime | None = Field(
        default=None, description="When baseline was established"
    )


class AnalysisExport(BaseModel):
    """Root schema for complete analysis export.

    Top-level container for all analysis data including metadata,
    summary statistics, file details, dependencies, and trends.

    This is the primary export format for visualization tools and
    external integrations.

    Attributes:
        metadata: Export metadata and version information
        summary: Project-level summary statistics
        files: Detailed metrics for each analyzed file
        dependencies: Dependency graph and coupling analysis
        trends: Historical trend data (optional)
    """

    metadata: ExportMetadata = Field(description="Export metadata and versioning")
    summary: MetricsSummary = Field(description="Project-level summary statistics")
    files: list[FileDetail] = Field(
        default_factory=list, description="File-level metrics"
    )
    dependencies: DependencyGraph = Field(
        description="Dependency graph and coupling analysis"
    )
    trends: TrendData | None = Field(
        default=None, description="Historical trend data (optional)"
    )


def generate_json_schema() -> dict[str, Any]:
    """Generate JSON Schema for documentation and validation.

    Creates a JSON Schema document describing the AnalysisExport format
    for use in documentation, validation tools, and IDE autocomplete.

    Returns:
        Dictionary containing JSON Schema for AnalysisExport

    Example:
        >>> schema = generate_json_schema()
        >>> import json
        >>> print(json.dumps(schema, indent=2))
    """
    return AnalysisExport.model_json_schema()
