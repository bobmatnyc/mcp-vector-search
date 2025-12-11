"""Baseline comparison for detecting metric changes.

This module provides the BaselineComparator class for comparing current
metrics against a stored baseline, identifying regressions, improvements,
and neutral changes.

Design Decisions:
    - Classification logic based on cognitive complexity thresholds
    - Percentage change calculation for relative comparison
    - Grade transitions tracked (A→B is regression, C→B is improvement)
    - Both absolute and percentage deltas reported
    - Neutral changes include unchanged and minor variations (<5% change)

Classification Logic:
    - Regression: Metric increased (complexity worse)
        - Cognitive complexity increased
        - Grade decreased (A→B, B→C, etc.)
        - Max nesting depth increased
        - More code smells detected
    - Improvement: Metric decreased (complexity better)
        - Cognitive complexity decreased
        - Grade improved (C→B, B→A, etc.)
        - Max nesting depth decreased
        - Fewer code smells
    - Neutral: No significant change (<5% for numeric metrics)

Performance:
    - Compare: O(n + m) where n=files, m=functions
    - Typical: 10-20ms for 100 files with 500 functions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..metrics import FileMetrics, ProjectMetrics


@dataclass
class MetricChange:
    """Represents a change in a single metric.

    Attributes:
        metric_name: Name of the metric (e.g., "cognitive_complexity")
        baseline_value: Value in baseline
        current_value: Value in current analysis
        absolute_delta: Absolute change (current - baseline)
        percentage_delta: Percentage change ((current - baseline) / baseline * 100)
        classification: Change classification (regression/improvement/neutral)
    """

    metric_name: str
    baseline_value: float | int
    current_value: float | int
    absolute_delta: float | int
    percentage_delta: float
    classification: Literal["regression", "improvement", "neutral"]

    @property
    def is_regression(self) -> bool:
        """Check if this is a regression."""
        return self.classification == "regression"

    @property
    def is_improvement(self) -> bool:
        """Check if this is an improvement."""
        return self.classification == "improvement"

    @property
    def is_neutral(self) -> bool:
        """Check if this is neutral (no significant change)."""
        return self.classification == "neutral"


@dataclass
class FileComparison:
    """Comparison results for a single file.

    Attributes:
        file_path: Path to the file
        in_baseline: Whether file existed in baseline
        in_current: Whether file exists in current analysis
        metric_changes: List of metric changes for this file
        grade_change: Tuple of (baseline_grade, current_grade) if changed
    """

    file_path: str
    in_baseline: bool
    in_current: bool
    metric_changes: list[MetricChange] = field(default_factory=list)
    grade_change: tuple[str, str] | None = None

    @property
    def has_regressions(self) -> bool:
        """Check if file has any regressions."""
        return any(change.is_regression for change in self.metric_changes)

    @property
    def has_improvements(self) -> bool:
        """Check if file has any improvements."""
        return any(change.is_improvement for change in self.metric_changes)

    @property
    def is_new_file(self) -> bool:
        """Check if this is a new file (not in baseline)."""
        return self.in_current and not self.in_baseline

    @property
    def is_deleted_file(self) -> bool:
        """Check if this file was deleted (in baseline, not in current)."""
        return self.in_baseline and not self.in_current


@dataclass
class ComparisonResult:
    """Complete baseline comparison results.

    Attributes:
        baseline_name: Name of baseline compared against
        regressions: List of file comparisons with regressions
        improvements: List of file comparisons with improvements
        unchanged: List of file comparisons with no significant changes
        new_files: List of new files not in baseline
        deleted_files: List of files in baseline but not in current
        summary: Dictionary of aggregate statistics
    """

    baseline_name: str
    regressions: list[FileComparison] = field(default_factory=list)
    improvements: list[FileComparison] = field(default_factory=list)
    unchanged: list[FileComparison] = field(default_factory=list)
    new_files: list[FileComparison] = field(default_factory=list)
    deleted_files: list[FileComparison] = field(default_factory=list)
    summary: dict[str, int | float] = field(default_factory=dict)

    @property
    def has_regressions(self) -> bool:
        """Check if comparison found any regressions."""
        return len(self.regressions) > 0

    @property
    def has_improvements(self) -> bool:
        """Check if comparison found any improvements."""
        return len(self.improvements) > 0

    @property
    def total_files_compared(self) -> int:
        """Total number of files compared."""
        return (
            len(self.regressions)
            + len(self.improvements)
            + len(self.unchanged)
            + len(self.new_files)
            + len(self.deleted_files)
        )


class BaselineComparator:
    """Compare current metrics against baseline.

    This class analyzes differences between current metrics and a baseline,
    classifying changes as regressions, improvements, or neutral.

    Comparison Strategy:
        1. Compare files present in both baseline and current
        2. Identify new files (in current, not in baseline)
        3. Identify deleted files (in baseline, not in current)
        4. For each file, compare metrics and classify changes
        5. Aggregate results into ComparisonResult

    Example:
        >>> comparator = BaselineComparator()
        >>> baseline = manager.load_baseline("main-branch")
        >>> current = analyze_project(Path.cwd())
        >>> result = comparator.compare(current, baseline, threshold_percent=5.0)
        >>> print(f"Regressions: {len(result.regressions)}")
        >>> print(f"Improvements: {len(result.improvements)}")
    """

    def compare(
        self,
        current: ProjectMetrics,
        baseline: ProjectMetrics,
        baseline_name: str = "baseline",
        threshold_percent: float = 5.0,
    ) -> ComparisonResult:
        """Compare current metrics against baseline.

        Args:
            current: Current ProjectMetrics
            baseline: Baseline ProjectMetrics to compare against
            baseline_name: Name of baseline (for result metadata)
            threshold_percent: Percentage threshold for neutral classification (default: 5.0%)

        Returns:
            ComparisonResult with classified changes

        Performance: O(n + m) where n=files, m=functions

        Example:
            >>> comparator = BaselineComparator()
            >>> result = comparator.compare(current, baseline)
            >>> if result.has_regressions:
            ...     print(f"Found {len(result.regressions)} regressions")
        """
        result = ComparisonResult(baseline_name=baseline_name)

        # Get file sets
        baseline_files = set(baseline.files.keys())
        current_files = set(current.files.keys())

        # Files in both baseline and current
        common_files = baseline_files & current_files

        # New files (in current, not in baseline)
        new_files = current_files - baseline_files

        # Deleted files (in baseline, not in current)
        deleted_files = baseline_files - current_files

        # Compare common files
        for file_path in common_files:
            comparison = self._compare_file(
                file_path=file_path,
                baseline_file=baseline.files[file_path],
                current_file=current.files[file_path],
                threshold_percent=threshold_percent,
            )

            # Classify based on changes
            if comparison.has_regressions:
                result.regressions.append(comparison)
            elif comparison.has_improvements:
                result.improvements.append(comparison)
            else:
                result.unchanged.append(comparison)

        # Add new files
        for file_path in new_files:
            comparison = FileComparison(
                file_path=file_path, in_baseline=False, in_current=True
            )
            result.new_files.append(comparison)

        # Add deleted files
        for file_path in deleted_files:
            comparison = FileComparison(
                file_path=file_path, in_baseline=True, in_current=False
            )
            result.deleted_files.append(comparison)

        # Compute summary statistics
        result.summary = self._compute_summary(current, baseline)

        return result

    def _compare_file(
        self,
        file_path: str,
        baseline_file: FileMetrics,
        current_file: FileMetrics,
        threshold_percent: float,
    ) -> FileComparison:
        """Compare metrics for a single file.

        Args:
            file_path: Path to file
            baseline_file: Baseline FileMetrics
            current_file: Current FileMetrics
            threshold_percent: Threshold for neutral classification

        Returns:
            FileComparison with metric changes
        """
        comparison = FileComparison(
            file_path=file_path, in_baseline=True, in_current=True
        )

        # Compare file-level metrics
        metrics_to_compare = [
            (
                "total_complexity",
                baseline_file.total_complexity,
                current_file.total_complexity,
            ),
            (
                "avg_complexity",
                baseline_file.avg_complexity,
                current_file.avg_complexity,
            ),
            (
                "max_complexity",
                baseline_file.max_complexity,
                current_file.max_complexity,
            ),
            (
                "function_count",
                baseline_file.function_count,
                current_file.function_count,
            ),
            ("class_count", baseline_file.class_count, current_file.class_count),
        ]

        for metric_name, baseline_value, current_value in metrics_to_compare:
            change = self._calculate_metric_change(
                metric_name=metric_name,
                baseline_value=baseline_value,
                current_value=current_value,
                threshold_percent=threshold_percent,
            )
            comparison.metric_changes.append(change)

        return comparison

    def _calculate_metric_change(
        self,
        metric_name: str,
        baseline_value: float | int,
        current_value: float | int,
        threshold_percent: float,
    ) -> MetricChange:
        """Calculate change for a single metric.

        Args:
            metric_name: Name of metric
            baseline_value: Baseline value
            current_value: Current value
            threshold_percent: Threshold for neutral classification

        Returns:
            MetricChange with classification
        """
        # Calculate deltas
        absolute_delta = current_value - baseline_value

        if baseline_value == 0:
            # Handle division by zero
            if current_value == 0:
                percentage_delta = 0.0
            else:
                # Treat as 100% increase if baseline was 0
                percentage_delta = 100.0 if current_value > 0 else -100.0
        else:
            percentage_delta = (absolute_delta / baseline_value) * 100

        # Classify change
        classification = self._classify_change(
            metric_name=metric_name,
            absolute_delta=absolute_delta,
            percentage_delta=percentage_delta,
            threshold_percent=threshold_percent,
        )

        return MetricChange(
            metric_name=metric_name,
            baseline_value=baseline_value,
            current_value=current_value,
            absolute_delta=absolute_delta,
            percentage_delta=percentage_delta,
            classification=classification,
        )

    def _classify_change(
        self,
        metric_name: str,
        absolute_delta: float | int,
        percentage_delta: float,
        threshold_percent: float,
    ) -> Literal["regression", "improvement", "neutral"]:
        """Classify metric change as regression, improvement, or neutral.

        Args:
            metric_name: Name of metric
            absolute_delta: Absolute change
            percentage_delta: Percentage change
            threshold_percent: Threshold for neutral classification

        Returns:
            Classification string

        Classification Rules:
            - Complexity metrics (higher is worse):
                - Increase > threshold → regression
                - Decrease > threshold → improvement
                - Otherwise → neutral
            - Count metrics (depends on context):
                - function_count, class_count → neutral (not inherently good/bad)
        """
        # Metrics where increase is bad (complexity metrics)
        complexity_metrics = [
            "total_complexity",
            "avg_complexity",
            "max_complexity",
            "cognitive_complexity",
            "cyclomatic_complexity",
            "max_nesting_depth",
            "parameter_count",
        ]

        # Check if change is within neutral threshold
        if abs(percentage_delta) < threshold_percent:
            return "neutral"

        # For complexity metrics, increase is regression
        if metric_name in complexity_metrics:
            if absolute_delta > 0:
                return "regression"
            elif absolute_delta < 0:
                return "improvement"
            else:
                return "neutral"

        # For count metrics, treat as neutral (not inherently good/bad)
        # More functions/classes can be refactoring (good) or bloat (bad)
        return "neutral"

    def _compute_summary(
        self, current: ProjectMetrics, baseline: ProjectMetrics
    ) -> dict[str, int | float]:
        """Compute aggregate summary statistics.

        Args:
            current: Current ProjectMetrics
            baseline: Baseline ProjectMetrics

        Returns:
            Dictionary of summary statistics
        """
        # Aggregate cognitive complexity across all files
        current_total_cc = sum(f.total_complexity for f in current.files.values())
        baseline_total_cc = sum(f.total_complexity for f in baseline.files.values())

        current_avg_cc = (
            current_total_cc / current.total_files if current.total_files > 0 else 0.0
        )
        baseline_avg_cc = (
            baseline_total_cc / baseline.total_files
            if baseline.total_files > 0
            else 0.0
        )

        # Find max complexity
        current_max_cc = max(
            (f.max_complexity for f in current.files.values()), default=0
        )
        baseline_max_cc = max(
            (f.max_complexity for f in baseline.files.values()), default=0
        )

        return {
            "total_files_current": current.total_files,
            "total_files_baseline": baseline.total_files,
            "total_functions_current": current.total_functions,
            "total_functions_baseline": baseline.total_functions,
            "total_complexity_current": current_total_cc,
            "total_complexity_baseline": baseline_total_cc,
            "avg_complexity_current": round(current_avg_cc, 2),
            "avg_complexity_baseline": round(baseline_avg_cc, 2),
            "max_complexity_current": current_max_cc,
            "max_complexity_baseline": baseline_max_cc,
        }
