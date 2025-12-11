"""LLM-friendly interpretation of code analysis results.

This module provides enhanced JSON export with semantic context and natural
language interpretation capabilities for LLM consumption.

Key Features:
- Threshold comparisons with semantic labels (above/below, by how much)
- Code smell classifications with severity and remediation estimates
- Semantic context (callers, callees, purpose hints)
- Natural language interpretation templates

Example:
    >>> from pathlib import Path
    >>> from mcp_vector_search.analysis import ProjectMetrics
    >>> from mcp_vector_search.analysis.interpretation import (
    ...     EnhancedJSONExporter,
    ...     AnalysisInterpreter
    ... )
    >>>
    >>> # Enhanced export with LLM context
    >>> exporter = EnhancedJSONExporter(project_root=Path("/path/to/project"))
    >>> export = exporter.export_with_context(project_metrics)
    >>>
    >>> # Natural language interpretation
    >>> interpreter = AnalysisInterpreter()
    >>> summary = interpreter.interpret(export, focus="summary")
    >>> print(summary)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..config.thresholds import ThresholdConfig
from .collectors.smells import SmellSeverity
from .visualizer.exporter import JSONExporter
from .visualizer.schemas import AnalysisExport, SmellLocation

if TYPE_CHECKING:
    from .metrics import ProjectMetrics


class ThresholdComparison(str, Enum):
    """Comparison status against thresholds."""

    WELL_BELOW = "well_below"  # <50% of threshold
    BELOW = "below"  # 50-100% of threshold
    AT_THRESHOLD = "at_threshold"  # 100-110% of threshold
    ABOVE = "above"  # 110-150% of threshold
    WELL_ABOVE = "well_above"  # >150% of threshold


class RemediationPriority(str, Enum):
    """Priority level for remediation."""

    LOW = "low"  # Info-level smells, cosmetic issues
    MEDIUM = "medium"  # Warning-level, should fix during refactoring
    HIGH = "high"  # Error-level, needs attention soon
    CRITICAL = "critical"  # Severe issues blocking maintainability


@dataclass
class ThresholdContext:
    """Context about metric threshold comparison.

    Attributes:
        metric_name: Name of the metric being compared
        value: Actual metric value
        threshold: Threshold value for this metric
        comparison: How value compares to threshold
        percentage_of_threshold: Value as percentage of threshold
        severity: Severity level based on comparison
    """

    metric_name: str
    value: float
    threshold: float
    comparison: ThresholdComparison
    percentage_of_threshold: float
    severity: SmellSeverity

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "threshold": self.threshold,
            "comparison": self.comparison.value,
            "percentage_of_threshold": round(self.percentage_of_threshold, 1),
            "severity": (
                self.severity.value
                if isinstance(self.severity, SmellSeverity)
                else self.severity
            ),
            "interpretation": self.get_interpretation(),
        }

    def get_interpretation(self) -> str:
        """Get natural language interpretation of threshold comparison."""
        diff_pct = abs(100 - self.percentage_of_threshold)

        if self.comparison == ThresholdComparison.WELL_BELOW:
            return f"{self.metric_name} is {diff_pct:.0f}% below threshold (healthy)"
        elif self.comparison == ThresholdComparison.BELOW:
            return f"{self.metric_name} is within acceptable range"
        elif self.comparison == ThresholdComparison.AT_THRESHOLD:
            return f"{self.metric_name} is at threshold (monitor closely)"
        elif self.comparison == ThresholdComparison.ABOVE:
            return f"{self.metric_name} exceeds threshold by {diff_pct:.0f}% (needs attention)"
        else:  # WELL_ABOVE
            return f"{self.metric_name} significantly exceeds threshold by {diff_pct:.0f}% (urgent)"


class EnhancedSmellLocation(BaseModel):
    """Enhanced smell location with interpretation context.

    Extends SmellLocation with remediation estimates and priority.

    Attributes:
        smell_type: Type of code smell
        severity: Severity level
        message: Human-readable description
        line: Starting line number
        column: Starting column (optional)
        end_line: Ending line number (optional)
        function_name: Function where smell occurs (optional)
        class_name: Class where smell occurs (optional)
        remediation_minutes: Estimated time to fix
        priority: Remediation priority level
        threshold_context: Threshold comparison context
        suggested_actions: List of specific remediation steps
    """

    smell_type: str
    severity: str
    message: str
    line: int
    column: int | None = None
    end_line: int | None = None
    function_name: str | None = None
    class_name: str | None = None
    remediation_minutes: int | None = None
    priority: str = Field(default="medium")
    threshold_context: dict[str, Any] | None = None
    suggested_actions: list[str] = Field(default_factory=list)


class LLMContextExport(BaseModel):
    """Extended analysis export with LLM-consumable context.

    Attributes:
        analysis: Base analysis export
        threshold_comparisons: Threshold context for each metric
        remediation_summary: Summary of remediation priorities
        code_quality_grade: Overall quality grade (A-F)
        interpretation_hints: Natural language hints for LLM
    """

    analysis: AnalysisExport
    threshold_comparisons: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    remediation_summary: dict[str, Any] = Field(default_factory=dict)
    code_quality_grade: str = "C"
    interpretation_hints: list[str] = Field(default_factory=list)


class EnhancedJSONExporter(JSONExporter):
    """Extended JSON exporter with LLM-consumable context.

    Adds semantic context, threshold comparisons, and remediation estimates
    to the base JSON export format.
    """

    def __init__(
        self,
        project_root: Path,
        threshold_config: ThresholdConfig | None = None,
        **kwargs: Any,
    ):
        """Initialize enhanced exporter.

        Args:
            project_root: Root directory of project
            threshold_config: Threshold configuration for comparisons
            **kwargs: Additional arguments passed to JSONExporter
        """
        super().__init__(project_root, **kwargs)
        self.threshold_config = threshold_config or ThresholdConfig()

    def export_with_context(
        self,
        project_metrics: ProjectMetrics,
        include_smells: bool = True,
        **kwargs: Any,
    ) -> LLMContextExport:
        """Export with enhanced LLM context.

        Args:
            project_metrics: Project metrics to export
            include_smells: Whether to detect and include code smells
            **kwargs: Additional arguments passed to export()

        Returns:
            Enhanced export with LLM-friendly context
        """
        # Get base export
        base_export = self.export(project_metrics, **kwargs)

        # Compute threshold comparisons
        threshold_comparisons = self._compute_threshold_comparisons(project_metrics)

        # Generate remediation summary
        remediation_summary = self._generate_remediation_summary(base_export)

        # Calculate quality grade
        quality_grade = self._calculate_quality_grade(project_metrics)

        # Generate interpretation hints
        interpretation_hints = self._generate_interpretation_hints(
            project_metrics, threshold_comparisons
        )

        return LLMContextExport(
            analysis=base_export,
            threshold_comparisons=threshold_comparisons,
            remediation_summary=remediation_summary,
            code_quality_grade=quality_grade,
            interpretation_hints=interpretation_hints,
        )

    def _compute_threshold_comparisons(
        self, project_metrics: ProjectMetrics
    ) -> dict[str, list[dict[str, Any]]]:
        """Compute threshold comparisons for all metrics.

        Args:
            project_metrics: Project metrics to analyze

        Returns:
            Dictionary mapping metric categories to threshold contexts
        """
        comparisons: dict[str, list[dict[str, Any]]] = {
            "complexity": [],
            "size": [],
            "coupling": [],
        }

        # Average complexity comparison
        all_chunks = [
            chunk for file in project_metrics.files.values() for chunk in file.chunks
        ]
        if all_chunks:
            avg_cognitive = sum(c.cognitive_complexity for c in all_chunks) / len(
                all_chunks
            )
            avg_cyclomatic = sum(c.cyclomatic_complexity for c in all_chunks) / len(
                all_chunks
            )

            comparisons["complexity"].append(
                self._create_threshold_context(
                    "avg_cognitive_complexity",
                    avg_cognitive,
                    float(self.threshold_config.complexity.cognitive_b),
                ).to_dict()
            )

            comparisons["complexity"].append(
                self._create_threshold_context(
                    "avg_cyclomatic_complexity",
                    avg_cyclomatic,
                    float(self.threshold_config.complexity.cyclomatic_moderate),
                ).to_dict()
            )

        # Size metrics
        all_files = list(project_metrics.files.values())
        if all_files:
            avg_file_lines = sum(f.total_lines for f in all_files) / len(all_files)
            comparisons["size"].append(
                self._create_threshold_context(
                    "avg_file_lines",
                    avg_file_lines,
                    float(self.threshold_config.smells.god_class_lines),
                ).to_dict()
            )

        return comparisons

    def _create_threshold_context(
        self, metric_name: str, value: float, threshold: float
    ) -> ThresholdContext:
        """Create threshold context for a metric.

        Args:
            metric_name: Name of the metric
            value: Actual value
            threshold: Threshold value

        Returns:
            ThresholdContext with comparison and severity
        """
        percentage = (value / threshold * 100) if threshold > 0 else 0

        # Determine comparison level
        if percentage < 50:
            comparison = ThresholdComparison.WELL_BELOW
            severity = SmellSeverity.INFO
        elif percentage < 100:
            comparison = ThresholdComparison.BELOW
            severity = SmellSeverity.INFO
        elif percentage <= 110:
            comparison = ThresholdComparison.AT_THRESHOLD
            severity = SmellSeverity.WARNING
        elif percentage <= 150:
            comparison = ThresholdComparison.ABOVE
            severity = SmellSeverity.WARNING
        else:
            comparison = ThresholdComparison.WELL_ABOVE
            severity = SmellSeverity.ERROR

        return ThresholdContext(
            metric_name=metric_name,
            value=value,
            threshold=threshold,
            comparison=comparison,
            percentage_of_threshold=percentage,
            severity=severity,
        )

    def _generate_remediation_summary(self, export: AnalysisExport) -> dict[str, Any]:
        """Generate remediation summary from analysis.

        Args:
            export: Base analysis export

        Returns:
            Summary with priorities and estimates
        """
        # Count smells by severity
        smells_by_severity = export.summary.smells_by_severity

        # Estimate total remediation time
        total_minutes = 0
        priority_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}

        # Base estimates (minutes) per smell type
        remediation_estimates = {
            "Long Method": 30,
            "Deep Nesting": 20,
            "Long Parameter List": 15,
            "God Class": 120,
            "Complex Method": 25,
        }

        for file_detail in export.files:
            for smell in file_detail.smells:
                estimate = remediation_estimates.get(smell.smell_type, 20)
                total_minutes += estimate

                # Map severity to priority
                if smell.severity == "error":
                    priority_counts["critical"] += 1
                elif smell.severity == "warning":
                    priority_counts["high"] += 1
                else:
                    priority_counts["medium"] += 1

        return {
            "total_smells": export.summary.total_smells,
            "smells_by_severity": smells_by_severity,
            "priority_counts": priority_counts,
            "estimated_remediation_hours": round(total_minutes / 60, 1),
            "recommended_focus": self._determine_recommended_focus(export),
        }

    def _determine_recommended_focus(self, export: AnalysisExport) -> list[str]:
        """Determine recommended focus areas based on analysis.

        Args:
            export: Analysis export

        Returns:
            List of recommended focus areas
        """
        recommendations = []

        # Check for God Classes
        god_class_count = sum(
            1 for f in export.files for s in f.smells if s.smell_type == "God Class"
        )
        if god_class_count > 0:
            recommendations.append(
                f"Address {god_class_count} God Class smell(s) - highest impact refactoring"
            )

        # Check complexity
        if export.summary.avg_complexity > 15:
            recommendations.append(
                "Reduce overall complexity - average exceeds recommended threshold"
            )

        # Check circular dependencies
        if export.summary.circular_dependencies > 0:
            recommendations.append(
                f"Resolve {export.summary.circular_dependencies} circular dependency cycles"
            )

        # Check instability
        if export.summary.avg_instability and export.summary.avg_instability > 0.7:
            recommendations.append(
                "Improve coupling - high instability indicates fragile architecture"
            )

        if not recommendations:
            recommendations.append(
                "Code quality is good - focus on preventive maintenance"
            )

        return recommendations

    def _calculate_quality_grade(self, project_metrics: ProjectMetrics) -> str:
        """Calculate overall quality grade A-F.

        Args:
            project_metrics: Project metrics

        Returns:
            Grade letter (A-F)
        """
        # Simple scoring based on complexity distribution
        summary = project_metrics.to_summary()
        dist = summary.get("complexity_distribution", {})

        # Calculate weighted score (A=100, B=80, C=60, D=40, F=20)
        total_chunks = sum(dist.values())
        if total_chunks == 0:
            return "C"

        score = (
            dist.get("A", 0) * 100
            + dist.get("B", 0) * 80
            + dist.get("C", 0) * 60
            + dist.get("D", 0) * 40
            + dist.get("F", 0) * 20
        ) / total_chunks

        # Map to grade
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _generate_interpretation_hints(
        self,
        project_metrics: ProjectMetrics,
        threshold_comparisons: dict[str, list[dict[str, Any]]],
    ) -> list[str]:
        """Generate natural language interpretation hints.

        Args:
            project_metrics: Project metrics
            threshold_comparisons: Threshold comparison contexts

        Returns:
            List of interpretation hints for LLM
        """
        hints = []

        # Complexity hints
        for comparison in threshold_comparisons.get("complexity", []):
            if comparison["comparison"] in ["above", "well_above"]:
                hints.append(comparison["interpretation"])

        # File count context
        total_files = len(project_metrics.files)
        if total_files < 10:
            hints.append("Small project - focus on establishing good patterns")
        elif total_files < 50:
            hints.append(
                "Medium project - maintain modularity and separation of concerns"
            )
        else:
            hints.append("Large project - architectural consistency is critical")

        return hints


class AnalysisInterpreter:
    """Natural language interpreter for analysis results.

    Provides LLM-friendly interpretation of analysis exports with
    configurable focus and verbosity levels.
    """

    def __init__(self) -> None:
        """Initialize interpreter."""
        self.prompt_templates = self._load_prompt_templates()

    def interpret(
        self,
        export: LLMContextExport,
        focus: str = "summary",
        verbosity: str = "normal",
    ) -> str:
        """Generate natural language interpretation.

        Args:
            export: Enhanced analysis export
            focus: Focus area - "summary", "recommendations", "priorities"
            verbosity: Verbosity level - "brief", "normal", "detailed"

        Returns:
            Natural language interpretation
        """
        if focus == "summary":
            return self._interpret_summary(export, verbosity)
        elif focus == "recommendations":
            return self._interpret_recommendations(export, verbosity)
        elif focus == "priorities":
            return self._interpret_priorities(export, verbosity)
        else:
            return self._interpret_summary(export, verbosity)

    def _interpret_summary(self, export: LLMContextExport, verbosity: str) -> str:
        """Generate summary interpretation.

        Args:
            export: Analysis export
            verbosity: Verbosity level

        Returns:
            Summary interpretation
        """
        summary = export.analysis.summary
        lines = []

        # Overall assessment
        lines.append(f"# Code Quality Assessment: Grade {export.code_quality_grade}")
        lines.append("")

        # High-level metrics
        lines.append(
            f"**Project Size**: {summary.total_files} files, {summary.total_functions} functions"
        )
        lines.append(
            f"**Average Complexity**: {summary.avg_complexity:.1f} (cognitive: {summary.avg_cognitive_complexity:.1f})"
        )
        lines.append("")

        # Code smells summary
        if summary.total_smells > 0:
            lines.append(f"**Code Smells Detected**: {summary.total_smells} total")
            smells_by_sev = summary.smells_by_severity
            lines.append(f"  - Errors: {smells_by_sev.get('error', 0)}")
            lines.append(f"  - Warnings: {smells_by_sev.get('warning', 0)}")
            lines.append(f"  - Info: {smells_by_sev.get('info', 0)}")
        else:
            lines.append("**Code Smells**: None detected ✓")

        lines.append("")

        # Interpretation hints
        if export.interpretation_hints:
            lines.append("**Key Insights**:")
            for hint in export.interpretation_hints:
                lines.append(f"  - {hint}")

        if verbosity == "detailed":
            # Add threshold comparisons
            lines.append("")
            lines.append("**Threshold Comparisons**:")
            for category, comparisons in export.threshold_comparisons.items():
                if comparisons:
                    lines.append(f"\n_{category.title()}_:")
                    for comp in comparisons:
                        lines.append(f"  - {comp['interpretation']}")

        return "\n".join(lines)

    def _interpret_recommendations(
        self, export: LLMContextExport, verbosity: str
    ) -> str:
        """Generate recommendations interpretation.

        Args:
            export: Analysis export
            verbosity: Verbosity level

        Returns:
            Recommendations interpretation
        """
        lines = ["# Recommended Actions", ""]

        remediation = export.remediation_summary
        focus_areas = remediation.get("recommended_focus", [])

        lines.append(
            f"**Estimated Effort**: {remediation.get('estimated_remediation_hours', 0)} hours"
        )
        lines.append("")

        lines.append("**Priority Focus Areas**:")
        for i, area in enumerate(focus_areas, 1):
            lines.append(f"{i}. {area}")

        if verbosity in ["normal", "detailed"]:
            lines.append("")
            lines.append("**Quick Wins** (low effort, high impact):")
            # Find simple refactorings
            quick_wins = []
            for file_detail in export.analysis.files:
                for smell in file_detail.smells:
                    if smell.smell_type in ["Long Parameter List", "Deep Nesting"]:
                        quick_wins.append(
                            f"  - Fix {smell.smell_type} in {Path(file_detail.path).name}"
                        )
            if quick_wins:
                lines.extend(quick_wins[:5])  # Top 5
            else:
                lines.append("  - No quick wins identified")

        return "\n".join(lines)

    def _interpret_priorities(self, export: LLMContextExport, verbosity: str) -> str:
        """Generate priorities interpretation.

        Args:
            export: Analysis export
            verbosity: Verbosity level

        Returns:
            Priorities interpretation
        """
        lines = ["# Remediation Priorities", ""]

        remediation = export.remediation_summary
        priority_counts = remediation.get("priority_counts", {})

        lines.append("**By Priority Level**:")
        lines.append(f"  - Critical: {priority_counts.get('critical', 0)} issues")
        lines.append(f"  - High: {priority_counts.get('high', 0)} issues")
        lines.append(f"  - Medium: {priority_counts.get('medium', 0)} issues")
        lines.append(f"  - Low: {priority_counts.get('low', 0)} issues")
        lines.append("")

        if verbosity in ["normal", "detailed"]:
            # Group smells by file
            smells_by_file: dict[str, list[SmellLocation]] = {}
            for file_detail in export.analysis.files:
                if file_detail.smells:
                    smells_by_file[file_detail.path] = file_detail.smells

            # Sort files by smell count
            sorted_files = sorted(
                smells_by_file.items(), key=lambda x: len(x[1]), reverse=True
            )

            lines.append("**Files Needing Most Attention**:")
            for file_path, smells in sorted_files[:5]:
                lines.append(f"  - {Path(file_path).name}: {len(smells)} smell(s)")

        return "\n".join(lines)

    def _load_prompt_templates(self) -> dict[str, str]:
        """Load prompt templates for interpretation.

        Returns:
            Dictionary of prompt templates
        """
        return {
            "summary": "Provide a high-level assessment of code quality",
            "recommendations": "Suggest actionable improvements prioritized by impact",
            "priorities": "Prioritize issues by severity and remediation effort",
        }
