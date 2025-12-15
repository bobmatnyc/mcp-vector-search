"""Markdown reporter for code analysis results.

Generates two markdown reports:
1. Full analysis report with metrics and visualizations
2. Agent-actionable fixes report for refactoring tasks
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..collectors.smells import CodeSmell
    from ..metrics import ProjectMetrics


class MarkdownReporter:
    """Markdown reporter for generating analysis reports."""

    def generate_analysis_report(
        self,
        metrics: ProjectMetrics,
        smells: list[CodeSmell] | None = None,
        output_path: Path | None = None,
    ) -> str:
        """Generate full analysis report in markdown format.

        Args:
            metrics: Project metrics to report
            smells: Optional list of detected code smells
            output_path: Optional output file path (defaults to current directory)

        Returns:
            Path to generated markdown file
        """
        if output_path is None:
            output_path = Path.cwd()

        # Determine output directory
        if output_path.is_dir():
            output_dir = output_path
        elif output_path.exists() and output_path.is_file():
            # If file exists, use parent directory
            output_dir = output_path.parent
        else:
            # If path doesn't exist, check if parent exists
            if output_path.parent.exists() and output_path.parent.is_dir():
                output_dir = output_path.parent
            else:
                # Default to current directory
                output_dir = Path.cwd()

        output_file = output_dir / "mcp-vector-search-analysis.md"

        # Generate markdown content
        content = self._build_analysis_markdown(metrics, smells or [])

        # Write to file
        output_file.write_text(content, encoding="utf-8")

        return str(output_file)

    def generate_fixes_report(
        self,
        metrics: ProjectMetrics,
        smells: list[CodeSmell],
        output_path: Path | None = None,
    ) -> str:
        """Generate agent-actionable fixes report.

        Args:
            metrics: Project metrics to analyze
            smells: List of detected code smells
            output_path: Optional output file path (defaults to current directory)

        Returns:
            Path to generated markdown file
        """
        if output_path is None:
            output_path = Path.cwd()

        # Determine output directory
        if output_path.is_dir():
            output_dir = output_path
        elif output_path.exists() and output_path.is_file():
            # If file exists, use parent directory
            output_dir = output_path.parent
        else:
            # If path doesn't exist, check if parent exists
            if output_path.parent.exists() and output_path.parent.is_dir():
                output_dir = output_path.parent
            else:
                # Default to current directory
                output_dir = Path.cwd()

        output_file = output_dir / "mcp-vector-search-analysis-fixes.md"

        # Generate markdown content
        content = self._build_fixes_markdown(metrics, smells)

        # Write to file
        output_file.write_text(content, encoding="utf-8")

        return str(output_file)

    def _build_analysis_markdown(
        self, metrics: ProjectMetrics, smells: list[CodeSmell]
    ) -> str:
        """Build full analysis markdown content.

        Args:
            metrics: Project metrics
            smells: Detected code smells

        Returns:
            Markdown content as string
        """
        lines: list[str] = []

        # Header
        lines.append("# MCP Vector Search - Code Analysis Report")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Summary section
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Files analyzed**: {metrics.total_files}")
        lines.append(f"- **Total lines**: {metrics.total_lines:,}")
        lines.append(f"- **Total functions**: {metrics.total_functions}")
        lines.append(f"- **Total classes**: {metrics.total_classes}")
        lines.append(f"- **Average complexity**: {metrics.avg_file_complexity:.1f}")
        lines.append("")

        # Complexity distribution
        distribution = metrics._compute_grade_distribution()
        total_chunks = sum(distribution.values())

        if total_chunks > 0:
            lines.append("## Complexity Distribution")
            lines.append("")
            lines.append("| Grade | Description | Count | Percentage |")
            lines.append("|-------|------------|-------|------------|")

            grade_info = {
                "A": "Excellent (0-5)",
                "B": "Good (6-10)",
                "C": "Acceptable (11-20)",
                "D": "Needs Improvement (21-30)",
                "F": "Refactor Required (31+)",
            }

            for grade in ["A", "B", "C", "D", "F"]:
                count = distribution.get(grade, 0)
                percentage = (count / total_chunks * 100) if total_chunks > 0 else 0
                description = grade_info[grade]
                lines.append(
                    f"| {grade} | {description} | {count} | {percentage:.1f}% |"
                )

            lines.append("")

        # Code smells summary
        if smells:
            from ..collectors.smells import SmellSeverity

            error_count = sum(1 for s in smells if s.severity == SmellSeverity.ERROR)
            warning_count = sum(
                1 for s in smells if s.severity == SmellSeverity.WARNING
            )
            info_count = sum(1 for s in smells if s.severity == SmellSeverity.INFO)

            lines.append("## Code Smells Summary")
            lines.append("")
            lines.append("| Severity | Count |")
            lines.append("|----------|-------|")
            lines.append(f"| ERROR | {error_count} |")
            lines.append(f"| WARNING | {warning_count} |")
            lines.append(f"| INFO | {info_count} |")
            lines.append(f"| **Total** | **{len(smells)}** |")
            lines.append("")

            # Count by smell type
            smell_types: dict[str, int] = {}
            for smell in smells:
                smell_types[smell.name] = smell_types.get(smell.name, 0) + 1

            lines.append("### By Type")
            lines.append("")
            lines.append("| Type | Count | Severity |")
            lines.append("|------|-------|----------|")

            # Sort by count descending
            for smell_type, count in sorted(
                smell_types.items(), key=lambda x: x[1], reverse=True
            ):
                # Get most common severity for this type
                type_smells = [s for s in smells if s.name == smell_type]
                severities = [s.severity.value for s in type_smells]
                common_severity = max(set(severities), key=severities.count)
                lines.append(f"| {smell_type} | {count} | {common_severity} |")

            lines.append("")

        # Top complexity hotspots
        hotspots = metrics.get_hotspots(limit=10)

        if hotspots:
            lines.append("## Top Complexity Hotspots")
            lines.append("")

            for rank, file_metrics in enumerate(hotspots, 1):
                # Compute average grade
                if file_metrics.chunks:
                    grades = [chunk.complexity_grade for chunk in file_metrics.chunks]
                    avg_grade = max(set(grades), key=grades.count)
                else:
                    avg_grade = "N/A"

                lines.append(
                    f"{rank}. **{file_metrics.file_path}** - "
                    f"Complexity: {file_metrics.avg_complexity:.1f}, "
                    f"Grade: {avg_grade}, "
                    f"Lines: {file_metrics.total_lines}, "
                    f"Functions: {len(file_metrics.chunks)}"
                )

            lines.append("")

        # Detailed smells (top 20)
        if smells:
            from ..collectors.smells import SmellSeverity

            lines.append("## Detailed Code Smells")
            lines.append("")
            lines.append("Showing top 20 most critical issues:")
            lines.append("")

            # Prioritize errors, then warnings, then info
            sorted_smells = sorted(
                smells,
                key=lambda s: (
                    (
                        0
                        if s.severity == SmellSeverity.ERROR
                        else 1
                        if s.severity == SmellSeverity.WARNING
                        else 2
                    ),
                    -s.metric_value,
                ),
            )[:20]

            for i, smell in enumerate(sorted_smells, 1):
                lines.append(f"### {i}. {smell.name} ({smell.severity.value.upper()})")
                lines.append("")
                lines.append(f"- **Location**: `{smell.location}`")
                lines.append(f"- **Description**: {smell.description}")
                lines.append(
                    f"- **Metric**: {smell.metric_value} (threshold: {smell.threshold})"
                )
                if smell.suggestion:
                    lines.append(f"- **Suggestion**: {smell.suggestion}")
                lines.append("")

        # Health metrics
        avg_health = metrics._compute_avg_health_score()
        files_needing_attention = metrics._count_files_needing_attention()

        lines.append("## Health Metrics")
        lines.append("")
        lines.append(f"- **Average health score**: {avg_health:.2f} / 1.00")
        lines.append(
            f"- **Files needing attention**: {files_needing_attention} "
            f"({files_needing_attention / max(metrics.total_files, 1) * 100:.1f}%)"
        )
        lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Generated by mcp-vector-search analyze command*")
        lines.append("")

        return "\n".join(lines)

    def _build_fixes_markdown(
        self, metrics: ProjectMetrics, smells: list[CodeSmell]
    ) -> str:
        """Build agent-actionable fixes markdown content.

        Args:
            metrics: Project metrics
            smells: Detected code smells

        Returns:
            Markdown content as string
        """

        lines: list[str] = []

        # Header
        lines.append("# MCP Vector Search - Refactoring Tasks")
        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append(
            "This report contains actionable refactoring tasks prioritized "
            "by severity and impact."
        )
        lines.append("")

        # Priority 1: God Classes
        god_class_smells = [s for s in smells if s.name == "God Class"]

        if god_class_smells:
            lines.append("## Priority 1: God Classes")
            lines.append("")
            lines.append(
                "Large classes with too many responsibilities. "
                "These should be split into smaller, focused classes."
            )
            lines.append("")

            for i, smell in enumerate(god_class_smells, 1):
                # Extract file path and line info
                location_parts = smell.location.split(":")
                file_path = location_parts[0]

                # Get file metrics for more context
                file_metrics = metrics.files.get(file_path)

                lines.append(f"### {i}. {Path(file_path).name}")
                lines.append("")
                lines.append(f"- **Location**: `{smell.location}`")

                if file_metrics:
                    lines.append(f"- **Lines**: {file_metrics.total_lines}")
                    lines.append(f"- **Methods**: {file_metrics.method_count}")
                    lines.append(f"- **Complexity**: {file_metrics.avg_complexity:.1f}")

                lines.append(f"- **Task**: {smell.suggestion}")
                lines.append("")
                lines.append("**Suggested refactoring:**")
                lines.append("")

                if file_metrics and file_metrics.method_count > 0:
                    # Suggest splitting based on method groups
                    suggested_classes = min(max(file_metrics.method_count // 10, 2), 5)
                    lines.append(f"- Extract into {suggested_classes} separate classes")
                    lines.append("- Group related methods by functionality")
                    lines.append("- Move configuration/setup to separate config class")
                    lines.append("- Consider using composition over inheritance")

                lines.append("")

        # Priority 2: Long Methods (>100 lines)
        long_method_smells = [
            s for s in smells if s.name == "Long Method" and s.metric_value > 100
        ]

        if long_method_smells:
            # Sort by metric value descending
            long_method_smells.sort(key=lambda s: s.metric_value, reverse=True)

            lines.append("## Priority 2: Long Methods (>100 lines)")
            lines.append("")
            lines.append(
                "Methods that are too long and should be broken down into "
                "smaller, focused functions."
            )
            lines.append("")

            for i, smell in enumerate(long_method_smells[:10], 1):
                lines.append(f"### {i}. Method at {smell.location}")
                lines.append("")
                lines.append(f"- **Metric**: {smell.metric_value:.0f} lines/complexity")
                lines.append(f"- **Task**: {smell.suggestion}")
                lines.append("")

        # Priority 3: High Complexity (>20)
        complex_method_smells = [
            s for s in smells if s.name == "Complex Method" and s.metric_value > 20
        ]

        if complex_method_smells:
            # Sort by metric value descending
            complex_method_smells.sort(key=lambda s: s.metric_value, reverse=True)

            lines.append("## Priority 3: High Complexity (>20)")
            lines.append("")
            lines.append(
                "Methods with high cyclomatic complexity that need simplification."
            )
            lines.append("")

            for i, smell in enumerate(complex_method_smells[:10], 1):
                lines.append(f"### {i}. Method at {smell.location}")
                lines.append("")
                lines.append(f"- **Cyclomatic Complexity**: {smell.metric_value:.0f}")
                lines.append(f"- **Task**: {smell.suggestion}")
                lines.append("")

        # Priority 4: Deep Nesting
        nesting_smells = [s for s in smells if s.name == "Deep Nesting"]

        if nesting_smells:
            # Sort by metric value descending
            nesting_smells.sort(key=lambda s: s.metric_value, reverse=True)

            lines.append("## Priority 4: Deep Nesting")
            lines.append("")
            lines.append("Methods with excessive nesting that reduce readability.")
            lines.append("")

            for i, smell in enumerate(nesting_smells[:10], 1):
                lines.append(f"### {i}. Method at {smell.location}")
                lines.append("")
                lines.append(f"- **Nesting Depth**: {smell.metric_value:.0f}")
                lines.append(f"- **Task**: {smell.suggestion}")
                lines.append("")

        # Priority 5: Long Parameter Lists
        param_smells = [s for s in smells if s.name == "Long Parameter List"]

        if param_smells:
            # Sort by metric value descending
            param_smells.sort(key=lambda s: s.metric_value, reverse=True)

            lines.append("## Priority 5: Long Parameter Lists")
            lines.append("")
            lines.append(
                "Functions with too many parameters that should use "
                "parameter objects or builders."
            )
            lines.append("")

            for i, smell in enumerate(param_smells[:10], 1):
                lines.append(f"### {i}. Function at {smell.location}")
                lines.append("")
                lines.append(f"- **Parameter Count**: {smell.metric_value:.0f}")
                lines.append(f"- **Task**: {smell.suggestion}")
                lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total issues**: {len(smells)}")
        lines.append(
            f"- **God Classes**: {len(god_class_smells)} (Priority 1 - highest impact)"
        )
        lines.append(
            f"- **Long Methods**: {len(long_method_smells)} "
            "(Priority 2 - moderate impact)"
        )
        lines.append(
            f"- **Complex Methods**: {len(complex_method_smells)} "
            "(Priority 3 - moderate impact)"
        )
        lines.append(
            f"- **Deep Nesting**: {len(nesting_smells)} (Priority 4 - readability)"
        )
        lines.append(
            f"- **Long Parameters**: {len(param_smells)} (Priority 5 - API design)"
        )
        lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Generated by mcp-vector-search analyze command*")
        lines.append("")
        lines.append(
            "**Note**: Address Priority 1 and 2 items first for maximum impact."
        )
        lines.append("")

        return "\n".join(lines)
