"""JSON exporter for structural code analysis results.

This module provides the JSONExporter class that converts analysis metrics
into the standardized JSON export format defined by schemas.py.

The exporter handles:
- Project-level metric aggregation
- File and function/class detail conversion
- Dependency graph construction
- Historical trend data integration
- Git-aware metadata generation

Example:
    >>> from pathlib import Path
    >>> from mcp_vector_search.analysis import ProjectMetrics
    >>> from mcp_vector_search.analysis.visualizer import JSONExporter
    >>>
    >>> exporter = JSONExporter(project_root=Path("/path/to/project"))
    >>> export = exporter.export(project_metrics)
    >>> json_output = export.model_dump_json(indent=2)
    >>>
    >>> # Or export directly to file
    >>> output_path = exporter.export_to_file(
    ...     project_metrics,
    ...     Path("analysis-results.json")
    ... )
"""

from __future__ import annotations

import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

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
    SmellLocation,
    TrendData,
)

if TYPE_CHECKING:
    from ..metrics import ChunkMetrics, FileMetrics, ProjectMetrics
    from ..storage.metrics_store import MetricsStore
    from ..storage.trend_tracker import TrendTracker


class JSONExporter:
    """Exports analysis results to JSON format using the defined schema.

    This exporter converts internal metric dataclasses to the standardized
    Pydantic-based export schema for consumption by visualization tools
    and external analysis platforms.

    Attributes:
        project_root: Root directory of the analyzed project
        metrics_store: Optional metrics store for historical data
        trend_tracker: Optional trend tracker for regression analysis
    """

    def __init__(
        self,
        project_root: Path,
        metrics_store: MetricsStore | None = None,
        trend_tracker: TrendTracker | None = None,
    ):
        """Initialize JSON exporter.

        Args:
            project_root: Root directory of project being analyzed
            metrics_store: Optional store for historical metrics snapshots
            trend_tracker: Optional tracker for trend analysis
        """
        self.project_root = project_root
        self.metrics_store = metrics_store
        self.trend_tracker = trend_tracker

    def export(
        self,
        project_metrics: ProjectMetrics,
        include_trends: bool = True,
        include_dependencies: bool = True,
    ) -> AnalysisExport:
        """Export project metrics to the JSON schema format.

        Args:
            project_metrics: Project-level metrics to export
            include_trends: Whether to include historical trend data
            include_dependencies: Whether to include dependency graph

        Returns:
            Complete analysis export in schema format
        """
        metadata = self._create_metadata()
        summary = self._create_summary(project_metrics)
        files = self._create_file_details(project_metrics)
        dependencies = (
            self._create_dependency_graph(project_metrics)
            if include_dependencies
            else DependencyGraph()
        )
        trends = (
            self._create_trend_data() if include_trends and self.metrics_store else None
        )

        return AnalysisExport(
            metadata=metadata,
            summary=summary,
            files=files,
            dependencies=dependencies,
            trends=trends,
        )

    def export_to_file(
        self,
        project_metrics: ProjectMetrics,
        output_path: Path,
        indent: int = 2,
        **kwargs,
    ) -> Path:
        """Export to a JSON file.

        Args:
            project_metrics: Project metrics to export
            output_path: Path where JSON file will be written
            indent: Number of spaces for JSON indentation
            **kwargs: Additional arguments passed to export()

        Returns:
            Path to the created JSON file
        """
        export = self.export(project_metrics, **kwargs)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(export.model_dump_json(indent=indent))
        logger.info(f"Exported analysis results to {output_path}")
        return output_path

    def _create_metadata(self) -> ExportMetadata:
        """Create export metadata with version and git info.

        Returns:
            Metadata with tool version and git context
        """
        # Get version from package
        from mcp_vector_search import __version__

        # Get git info
        git_commit, git_branch = self._get_git_info()

        return ExportMetadata(
            version="1.0.0",  # Schema version
            generated_at=datetime.now(),
            tool_version=__version__,
            project_root=str(self.project_root),
            git_commit=git_commit,
            git_branch=git_branch,
        )

    def _create_summary(self, project_metrics: ProjectMetrics) -> MetricsSummary:
        """Create project-level summary from metrics.

        Args:
            project_metrics: Project metrics to summarize

        Returns:
            Aggregated summary statistics
        """
        # Get all files and chunks
        all_files = list(project_metrics.files.values())
        all_chunks = [chunk for file in all_files for chunk in file.chunks]

        # Basic counts
        total_files = len(all_files)
        total_functions = sum(file.function_count for file in all_files)
        total_classes = sum(file.class_count for file in all_files)
        total_lines = sum(file.total_lines for file in all_files)

        # Calculate complexity averages
        avg_complexity = (
            sum(chunk.cognitive_complexity for chunk in all_chunks) / len(all_chunks)
            if all_chunks
            else 0.0
        )
        avg_cognitive_complexity = avg_complexity  # Same as above

        avg_nesting_depth = (
            sum(chunk.max_nesting_depth for chunk in all_chunks) / len(all_chunks)
            if all_chunks
            else 0.0
        )

        # Count smells by severity
        smells_by_severity: dict[str, int] = Counter()
        total_smells = 0

        # Note: Smells are currently stored as string lists in ChunkMetrics.smells
        # We'll need to parse them or use a smell detector to get severity
        for chunk in all_chunks:
            total_smells += len(chunk.smells)
            # Default to 'warning' for now since we don't have severity info in ChunkMetrics
            for _ in chunk.smells:
                smells_by_severity["warning"] += 1

        # Calculate coupling metrics
        instabilities = [
            file.coupling.instability
            for file in all_files
            if file.coupling.efferent_coupling + file.coupling.afferent_coupling > 0
        ]
        avg_instability = (
            sum(instabilities) / len(instabilities) if instabilities else None
        )

        # Count circular dependencies
        circular_dependencies = 0
        # TODO: Get this from coupling collectors when available

        # Halstead metrics (optional, from Phase 4)
        halstead_volumes = [
            chunk.halstead_volume
            for chunk in all_chunks
            if chunk.halstead_volume is not None
        ]
        avg_halstead_volume = (
            sum(halstead_volumes) / len(halstead_volumes) if halstead_volumes else None
        )

        halstead_difficulties = [
            chunk.halstead_difficulty
            for chunk in all_chunks
            if chunk.halstead_difficulty is not None
        ]
        avg_halstead_difficulty = (
            sum(halstead_difficulties) / len(halstead_difficulties)
            if halstead_difficulties
            else None
        )

        # Technical debt estimation (optional)
        # TODO: Calculate from DebtEstimator when available
        estimated_debt_minutes = None

        return MetricsSummary(
            total_files=total_files,
            total_functions=total_functions,
            total_classes=total_classes,
            total_lines=total_lines,
            avg_complexity=avg_complexity,
            avg_cognitive_complexity=avg_cognitive_complexity,
            avg_nesting_depth=avg_nesting_depth,
            total_smells=total_smells,
            smells_by_severity=dict(smells_by_severity),
            avg_instability=avg_instability,
            circular_dependencies=circular_dependencies,
            avg_halstead_volume=avg_halstead_volume,
            avg_halstead_difficulty=avg_halstead_difficulty,
            estimated_debt_minutes=estimated_debt_minutes,
        )

    def _create_file_details(self, project_metrics: ProjectMetrics) -> list[FileDetail]:
        """Convert FileMetrics to FileDetail schema.

        Args:
            project_metrics: Project metrics containing file data

        Returns:
            List of file details in schema format
        """
        return [
            self._convert_file(file_metrics)
            for file_metrics in project_metrics.files.values()
        ]

    def _convert_file(self, file_metrics: FileMetrics) -> FileDetail:
        """Convert a single FileMetrics to FileDetail.

        Args:
            file_metrics: File-level metrics to convert

        Returns:
            File detail in schema format
        """
        # Separate functions and classes (methods inside classes)
        # For now, we'll treat all chunks as functions since ChunkMetrics doesn't distinguish
        functions = []
        classes = []

        for chunk in file_metrics.chunks:
            # TODO: Need to add chunk_type or similar field to distinguish
            # For now, assume all are functions
            func_metrics = self._convert_function(chunk, 0)  # Line numbers TBD
            functions.append(func_metrics)

        # Convert smells from chunks
        smells = []
        for chunk in file_metrics.chunks:
            for smell_name in chunk.smells:
                smell = SmellLocation(
                    smell_type=smell_name,
                    severity="warning",  # Default severity
                    message=f"Code smell detected: {smell_name}",
                    line=1,  # TODO: Get actual line numbers from chunk
                )
                smells.append(smell)

        # Calculate aggregated complexity
        cyclomatic_complexity = sum(
            chunk.cyclomatic_complexity for chunk in file_metrics.chunks
        )
        cognitive_complexity = sum(
            chunk.cognitive_complexity for chunk in file_metrics.chunks
        )
        max_nesting_depth = (
            max(chunk.max_nesting_depth for chunk in file_metrics.chunks)
            if file_metrics.chunks
            else 0
        )

        return FileDetail(
            path=file_metrics.file_path,
            language="python",  # TODO: Detect language from file extension
            lines_of_code=file_metrics.total_lines,
            cyclomatic_complexity=cyclomatic_complexity,
            cognitive_complexity=cognitive_complexity,
            max_nesting_depth=max_nesting_depth,
            function_count=file_metrics.function_count,
            class_count=file_metrics.class_count,
            efferent_coupling=file_metrics.coupling.efferent_coupling,
            afferent_coupling=file_metrics.coupling.afferent_coupling,
            instability=file_metrics.coupling.instability,
            functions=functions,
            classes=classes,
            smells=smells,
            imports=file_metrics.coupling.imports,
        )

    def _convert_function(
        self, chunk_metrics: ChunkMetrics, line_start: int
    ) -> FunctionMetrics:
        """Convert a ChunkMetrics to FunctionMetrics.

        Args:
            chunk_metrics: Chunk-level metrics to convert
            line_start: Starting line number of function

        Returns:
            Function metrics in schema format
        """
        # Estimate line_end from lines_of_code
        line_end = line_start + chunk_metrics.lines_of_code

        return FunctionMetrics(
            name="function",  # TODO: Get actual function name from chunk
            line_start=max(1, line_start),
            line_end=max(1, line_end),
            cyclomatic_complexity=chunk_metrics.cyclomatic_complexity,
            cognitive_complexity=chunk_metrics.cognitive_complexity,
            nesting_depth=chunk_metrics.max_nesting_depth,
            parameter_count=chunk_metrics.parameter_count,
            lines_of_code=chunk_metrics.lines_of_code,
            halstead_volume=chunk_metrics.halstead_volume,
            halstead_difficulty=chunk_metrics.halstead_difficulty,
            halstead_effort=chunk_metrics.halstead_effort,
        )

    def _convert_class(self, chunk_metrics: ChunkMetrics) -> ClassMetrics:
        """Convert a ChunkMetrics to ClassMetrics.

        Args:
            chunk_metrics: Chunk-level metrics for class

        Returns:
            Class metrics in schema format
        """
        # TODO: Implement when we have proper class detection
        return ClassMetrics(
            name="Class",
            line_start=1,
            line_end=chunk_metrics.lines_of_code,
            method_count=0,
            lcom4=None,
            methods=[],
        )

    def _create_dependency_graph(
        self, project_metrics: ProjectMetrics
    ) -> DependencyGraph:
        """Create dependency graph from coupling data.

        Args:
            project_metrics: Project metrics with coupling information

        Returns:
            Dependency graph with edges and circular dependencies
        """
        edges: list[DependencyEdge] = []
        all_files = list(project_metrics.files.values())

        # Build edges from coupling data
        for file_metrics in all_files:
            source = file_metrics.file_path
            for target in file_metrics.coupling.imports:
                # Classify import type (simplified for now)
                import_type = "import"  # Could be "from_import" or "dynamic"
                edges.append(
                    DependencyEdge(
                        source=source, target=target, import_type=import_type
                    )
                )

        # Calculate most depended on files (afferent coupling)
        afferent_counts = [
            (file.file_path, file.coupling.afferent_coupling) for file in all_files
        ]
        most_depended_on = sorted(afferent_counts, key=lambda x: x[1], reverse=True)[
            :10
        ]

        # Calculate most dependent files (efferent coupling)
        efferent_counts = [
            (file.file_path, file.coupling.efferent_coupling) for file in all_files
        ]
        most_dependent = sorted(efferent_counts, key=lambda x: x[1], reverse=True)[:10]

        # Circular dependencies (TODO: implement detection)
        circular_dependencies: list[CyclicDependency] = []

        return DependencyGraph(
            edges=edges,
            circular_dependencies=circular_dependencies,
            most_depended_on=most_depended_on,
            most_dependent=most_dependent,
        )

    def _create_trend_data(self) -> TrendData | None:
        """Create trend data from metrics store.

        Returns:
            Historical trend data if available, None otherwise
        """
        if not self.metrics_store or not self.trend_tracker:
            return None

        # TODO: Implement trend data extraction from metrics store
        # This will require querying historical snapshots and formatting as MetricTrends

        return None

    def _get_git_info(self) -> tuple[str | None, str | None]:
        """Get current git commit and branch.

        Returns:
            Tuple of (commit_sha, branch_name) or (None, None) if not in git repo
        """
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],  # nosec B607
                cwd=self.project_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()

            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],  # nosec B607
                cwd=self.project_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()

            return commit, branch
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("Git information not available")
            return None, None
