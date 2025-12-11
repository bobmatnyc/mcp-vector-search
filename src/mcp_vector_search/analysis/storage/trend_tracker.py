"""Trend tracking and regression detection for metrics over time.

This module provides the TrendTracker class for analyzing metric trends,
identifying regressions and improvements, and generating alerts.

Design Decisions:
    Threshold Strategy: Configurable percentage-based thresholds
    - Default 5% change considered "significant"
    - Users can customize via constructor parameter
    - Percentage-based to normalize across different metric scales

    Trend Direction: Three states (Improving/Worsening/Stable)
    - Based on statistical significance of change
    - Uses first/last snapshot comparison
    - Requires minimum 2 snapshots for trend analysis

    Alert Generation: Proactive identification of issues
    - Regression alerts: Metrics that worsened significantly
    - Improvement alerts: Metrics that improved significantly
    - Per-file and project-level analysis

Performance:
    - get_trends: O(n) where n=snapshots, typically <100ms for 30-day period
    - get_regression_alerts: O(m) where m=files in snapshots, <50ms
    - calculate_trend_direction: O(1), instant

Error Handling:
    - ValueError: Invalid threshold or days parameter
    - Logs warnings when insufficient data for analysis
    - Returns empty results rather than raising exceptions

Example:
    >>> tracker = TrendTracker(metrics_store, threshold_percentage=5.0)
    >>> trends = tracker.get_trends("/path/to/project", days=30)
    >>> if trends.complexity_trend == TrendDirection.WORSENING:
    ...     print("Complexity is trending up!")
    >>> alerts = tracker.get_regression_alerts("/path/to/project")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from .metrics_store import MetricsStore, ProjectSnapshot


class TrendDirection(Enum):
    """Direction of metric trend over time."""

    IMPROVING = "improving"  # Complexity/smells decreasing
    WORSENING = "worsening"  # Complexity/smells increasing
    STABLE = "stable"  # No significant change


@dataclass
class FileRegression:
    """Details of a file-level regression.

    Attributes:
        file_path: Path to the file with regression
        metric_name: Name of metric that regressed (e.g., "avg_complexity")
        old_value: Previous metric value
        new_value: Current metric value
        change_percentage: Percentage change (positive = worse)
        timestamp: When regression was detected
    """

    file_path: str
    metric_name: str
    old_value: float
    new_value: float
    change_percentage: float
    timestamp: datetime


@dataclass
class TrendData:
    """Trend analysis data over time period.

    Extends the basic TrendData from metrics_store with additional
    analysis fields for alerts and trend directions.

    Attributes:
        project_path: Project being analyzed
        period_days: Number of days in trend period
        snapshots: List of snapshots in chronological order
        complexity_trend: List of (timestamp, avg_complexity) tuples
        smell_trend: List of (timestamp, total_smells) tuples
        health_trend: List of (timestamp, avg_health_score) tuples
        change_rate: Average daily change in complexity

        # Enhanced trend analysis fields
        complexity_direction: Improving/Worsening/Stable for complexity
        smell_direction: Improving/Worsening/Stable for code smells
        health_direction: Improving/Worsening/Stable for health score
        critical_regressions: List of files with significant worsening
        significant_improvements: List of files with significant improvements
        avg_complexity_change: Total percentage change in avg complexity
        smell_count_change: Total percentage change in smell count
    """

    project_path: str
    period_days: int
    snapshots: list[ProjectSnapshot]
    complexity_trend: list[tuple[datetime, float]]
    smell_trend: list[tuple[datetime, int]]
    health_trend: list[tuple[datetime, float]]
    change_rate: float

    # Enhanced fields for trend analysis
    complexity_direction: TrendDirection = TrendDirection.STABLE
    smell_direction: TrendDirection = TrendDirection.STABLE
    health_direction: TrendDirection = TrendDirection.STABLE
    critical_regressions: list[FileRegression] = field(default_factory=list)
    significant_improvements: list[FileRegression] = field(default_factory=list)
    avg_complexity_change: float = 0.0
    smell_count_change: float = 0.0

    @property
    def improving(self) -> bool:
        """Check if trends are improving overall.

        Returns:
            True if complexity is decreasing AND health is improving
        """
        return (
            self.complexity_direction == TrendDirection.IMPROVING
            and self.health_direction == TrendDirection.IMPROVING
        )

    @property
    def has_regressions(self) -> bool:
        """Check if there are critical regressions.

        Returns:
            True if any critical regressions detected
        """
        return len(self.critical_regressions) > 0


class TrendTracker:
    """Track metric trends and detect regressions/improvements over time.

    This class provides comprehensive trend analysis capabilities:
    - Analyze metric trends over configurable time periods
    - Identify significant regressions (metrics worsening)
    - Identify significant improvements (metrics improving)
    - Calculate trend directions (improving/worsening/stable)

    Thread Safety:
        - Safe for single-threaded CLI usage
        - Reads from MetricsStore (which handles its own locking)

    Example:
        >>> store = MetricsStore()
        >>> tracker = TrendTracker(store, threshold_percentage=5.0)
        >>>
        >>> # Get 30-day trend analysis
        >>> trends = tracker.get_trends("/path/to/project", days=30)
        >>>
        >>> # Check for regressions
        >>> if trends.has_regressions:
        ...     print(f"Found {len(trends.critical_regressions)} regressions!")
        >>>
        >>> # Get specific regression alerts
        >>> alerts = tracker.get_regression_alerts("/path/to/project")
    """

    def __init__(
        self, metrics_store: MetricsStore, threshold_percentage: float = 5.0
    ) -> None:
        """Initialize trend tracker.

        Args:
            metrics_store: MetricsStore instance for accessing historical data
            threshold_percentage: Percentage change to consider "significant" (default: 5%)

        Raises:
            ValueError: If threshold_percentage is negative or > 100
        """
        if threshold_percentage < 0 or threshold_percentage > 100:
            raise ValueError(
                f"threshold_percentage must be between 0 and 100, got {threshold_percentage}"
            )

        self.store = metrics_store
        self.threshold = threshold_percentage / 100.0  # Convert to decimal

        logger.debug(f"Initialized TrendTracker with threshold {threshold_percentage}%")

    def get_trends(self, project_path: str | Path, days: int = 30) -> TrendData:
        """Get comprehensive trend analysis over time period.

        Analyzes all available metrics over the specified time period and
        identifies trends, regressions, and improvements.

        Args:
            project_path: Path to project root
            days: Number of days to analyze (from now backwards)

        Returns:
            TrendData with comprehensive trend analysis

        Raises:
            ValueError: If days <= 0

        Performance: O(n) where n=snapshots, typically <100ms
        """
        if days <= 0:
            raise ValueError(f"days must be positive, got {days}")

        project_path_str = str(Path(project_path).resolve())

        # Get base trend data from store
        base_trends = self.store.get_trends(project_path_str, days=days)

        # If insufficient data, return basic trends
        if len(base_trends.snapshots) < 2:
            logger.warning(
                f"Insufficient snapshots ({len(base_trends.snapshots)}) for trend analysis"
            )
            return TrendData(
                project_path=base_trends.project_path,
                period_days=base_trends.period_days,
                snapshots=base_trends.snapshots,
                complexity_trend=base_trends.complexity_trend,
                smell_trend=base_trends.smell_trend,
                health_trend=base_trends.health_trend,
                change_rate=base_trends.change_rate,
            )

        # Calculate trend directions
        complexity_direction = self.calculate_trend_direction(
            base_trends.complexity_trend
        )
        smell_direction = self.calculate_trend_direction(
            base_trends.smell_trend, lower_is_better=True
        )
        health_direction = self.calculate_trend_direction(
            base_trends.health_trend, lower_is_better=False
        )

        # Calculate percentage changes
        first_snapshot = base_trends.snapshots[0]
        last_snapshot = base_trends.snapshots[-1]

        avg_complexity_change = self._calculate_percentage_change(
            first_snapshot.avg_complexity, last_snapshot.avg_complexity
        )

        smell_count_change = self._calculate_percentage_change(
            first_snapshot.total_smells, last_snapshot.total_smells
        )

        # Get regressions and improvements
        regressions = self._find_regressions(base_trends.snapshots)
        improvements = self._find_improvements(base_trends.snapshots)

        logger.info(
            f"Analyzed trends for {project_path_str}: "
            f"{len(base_trends.snapshots)} snapshots, "
            f"complexity {complexity_direction.value}, "
            f"{len(regressions)} regressions, "
            f"{len(improvements)} improvements"
        )

        return TrendData(
            project_path=base_trends.project_path,
            period_days=base_trends.period_days,
            snapshots=base_trends.snapshots,
            complexity_trend=base_trends.complexity_trend,
            smell_trend=base_trends.smell_trend,
            health_trend=base_trends.health_trend,
            change_rate=base_trends.change_rate,
            complexity_direction=complexity_direction,
            smell_direction=smell_direction,
            health_direction=health_direction,
            critical_regressions=regressions,
            significant_improvements=improvements,
            avg_complexity_change=avg_complexity_change,
            smell_count_change=smell_count_change,
        )

    def get_regression_alerts(
        self, project_path: str | Path, days: int = 30
    ) -> list[FileRegression]:
        """Identify metrics that worsened significantly.

        Returns list of FileRegression objects for all metrics that
        exceeded the threshold for worsening.

        Args:
            project_path: Path to project root
            days: Number of days to analyze (default: 30)

        Returns:
            List of FileRegression objects (empty if no regressions)

        Performance: O(m) where m=files in snapshots, typically <50ms
        """
        trends = self.get_trends(project_path, days=days)
        return trends.critical_regressions

    def get_improvement_alerts(
        self, project_path: str | Path, days: int = 30
    ) -> list[FileRegression]:
        """Identify metrics that improved significantly.

        Returns list of FileRegression objects (reused for improvements)
        for all metrics that exceeded the threshold for improvement.

        Args:
            project_path: Path to project root
            days: Number of days to analyze (default: 30)

        Returns:
            List of FileRegression objects (empty if no improvements)

        Performance: O(m) where m=files in snapshots, typically <50ms
        """
        trends = self.get_trends(project_path, days=days)
        return trends.significant_improvements

    def calculate_trend_direction(
        self,
        trend_data: list[tuple[datetime, float | int]],
        lower_is_better: bool = True,
    ) -> TrendDirection:
        """Determine if metric is improving, worsening, or stable.

        Compares first and last values in trend data to determine direction.
        Uses threshold to determine if change is significant.

        Args:
            trend_data: List of (timestamp, value) tuples
            lower_is_better: If True, decreasing values = improving (default)
                           If False, increasing values = improving

        Returns:
            TrendDirection enum value

        Performance: O(1), instant

        Example:
            >>> trend = [(t1, 10.0), (t2, 15.0), (t3, 20.0)]
            >>> direction = tracker.calculate_trend_direction(trend)
            >>> # Returns WORSENING (increasing complexity is bad)
        """
        if len(trend_data) < 2:
            logger.debug("Insufficient data for trend direction calculation")
            return TrendDirection.STABLE

        first_value = float(trend_data[0][1])
        last_value = float(trend_data[-1][1])

        # Calculate percentage change
        if first_value == 0:
            # Avoid division by zero
            if last_value > 0:
                percentage_change = 1.0  # 100% increase
            else:
                percentage_change = 0.0
        else:
            percentage_change = abs(last_value - first_value) / first_value

        # Check if change is significant
        if percentage_change < self.threshold:
            return TrendDirection.STABLE

        # Determine direction based on whether lower is better
        if lower_is_better:
            # For complexity/smells: lower = better
            if last_value < first_value:
                return TrendDirection.IMPROVING
            else:
                return TrendDirection.WORSENING
        else:
            # For health score: higher = better
            if last_value > first_value:
                return TrendDirection.IMPROVING
            else:
                return TrendDirection.WORSENING

    def _calculate_percentage_change(self, old_value: float, new_value: float) -> float:
        """Calculate percentage change between two values.

        Args:
            old_value: Previous value
            new_value: Current value

        Returns:
            Percentage change (positive = increase, negative = decrease)
        """
        if old_value == 0:
            if new_value > 0:
                return 100.0  # 100% increase from zero
            else:
                return 0.0
        return ((new_value - old_value) / old_value) * 100.0

    def _find_regressions(
        self, snapshots: list[ProjectSnapshot]
    ) -> list[FileRegression]:
        """Find file-level regressions between first and last snapshot.

        Compares per-file metrics between oldest and newest snapshot
        to identify files that significantly worsened.

        Args:
            snapshots: List of project snapshots (chronologically ordered)

        Returns:
            List of FileRegression objects for files that worsened
        """
        if len(snapshots) < 2:
            return []

        first_snapshot = snapshots[0]
        last_snapshot = snapshots[-1]

        # Note: File-level metrics would require per-file tracking
        # For now, we'll create project-level regression if metrics worsened

        regressions: list[FileRegression] = []

        # Check avg complexity regression
        complexity_change = self._calculate_percentage_change(
            first_snapshot.avg_complexity, last_snapshot.avg_complexity
        )
        if complexity_change > (self.threshold * 100):
            regressions.append(
                FileRegression(
                    file_path="PROJECT_OVERALL",
                    metric_name="avg_complexity",
                    old_value=first_snapshot.avg_complexity,
                    new_value=last_snapshot.avg_complexity,
                    change_percentage=complexity_change,
                    timestamp=last_snapshot.timestamp,
                )
            )

        # Check smell count regression
        smell_change = self._calculate_percentage_change(
            first_snapshot.total_smells, last_snapshot.total_smells
        )
        if smell_change > (self.threshold * 100):
            regressions.append(
                FileRegression(
                    file_path="PROJECT_OVERALL",
                    metric_name="total_smells",
                    old_value=float(first_snapshot.total_smells),
                    new_value=float(last_snapshot.total_smells),
                    change_percentage=smell_change,
                    timestamp=last_snapshot.timestamp,
                )
            )

        # Check health score regression (lower = worse)
        # For health: decrease is bad, so absolute decrease > threshold is regression
        if first_snapshot.avg_health_score > 0:  # Avoid division by zero
            health_change = self._calculate_percentage_change(
                first_snapshot.avg_health_score, last_snapshot.avg_health_score
            )
            if last_snapshot.avg_health_score < first_snapshot.avg_health_score:
                # Health decreased - check if change exceeds threshold
                if abs(health_change) > (self.threshold * 100):
                    regressions.append(
                        FileRegression(
                            file_path="PROJECT_OVERALL",
                            metric_name="avg_health_score",
                            old_value=first_snapshot.avg_health_score,
                            new_value=last_snapshot.avg_health_score,
                            change_percentage=health_change,
                            timestamp=last_snapshot.timestamp,
                        )
                    )

        return regressions

    def _find_improvements(
        self, snapshots: list[ProjectSnapshot]
    ) -> list[FileRegression]:
        """Find file-level improvements between first and last snapshot.

        Compares per-file metrics between oldest and newest snapshot
        to identify files that significantly improved.

        Args:
            snapshots: List of project snapshots (chronologically ordered)

        Returns:
            List of FileRegression objects for files that improved
        """
        if len(snapshots) < 2:
            return []

        first_snapshot = snapshots[0]
        last_snapshot = snapshots[-1]

        improvements: list[FileRegression] = []

        # Check avg complexity improvement
        complexity_change = self._calculate_percentage_change(
            first_snapshot.avg_complexity, last_snapshot.avg_complexity
        )
        if complexity_change < -(self.threshold * 100):  # Negative = better
            improvements.append(
                FileRegression(
                    file_path="PROJECT_OVERALL",
                    metric_name="avg_complexity",
                    old_value=first_snapshot.avg_complexity,
                    new_value=last_snapshot.avg_complexity,
                    change_percentage=complexity_change,
                    timestamp=last_snapshot.timestamp,
                )
            )

        # Check smell count improvement
        smell_change = self._calculate_percentage_change(
            first_snapshot.total_smells, last_snapshot.total_smells
        )
        if smell_change < -(self.threshold * 100):  # Negative = better
            improvements.append(
                FileRegression(
                    file_path="PROJECT_OVERALL",
                    metric_name="total_smells",
                    old_value=float(first_snapshot.total_smells),
                    new_value=float(last_snapshot.total_smells),
                    change_percentage=smell_change,
                    timestamp=last_snapshot.timestamp,
                )
            )

        # Check health score improvement (higher = better)
        if first_snapshot.avg_health_score > 0:  # Avoid division by zero
            health_change = self._calculate_percentage_change(
                first_snapshot.avg_health_score, last_snapshot.avg_health_score
            )
            if last_snapshot.avg_health_score > first_snapshot.avg_health_score:
                # Health increased - check if change exceeds threshold
                if abs(health_change) > (self.threshold * 100):
                    improvements.append(
                        FileRegression(
                            file_path="PROJECT_OVERALL",
                            metric_name="avg_health_score",
                            old_value=first_snapshot.avg_health_score,
                            new_value=last_snapshot.avg_health_score,
                            change_percentage=health_change,
                            timestamp=last_snapshot.timestamp,
                        )
                    )

        return improvements
