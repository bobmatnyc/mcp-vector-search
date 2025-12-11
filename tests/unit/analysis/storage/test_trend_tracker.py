"""Unit tests for TrendTracker class."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from mcp_vector_search.analysis.metrics import FileMetrics, ProjectMetrics
from mcp_vector_search.analysis.storage import (
    MetricsStore,
    TrendDirection,
    TrendTracker,
)


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def metrics_store(temp_db):
    """Create MetricsStore with temporary database."""
    store = MetricsStore(db_path=temp_db)
    yield store
    store.close()


@pytest.fixture
def trend_tracker(metrics_store):
    """Create TrendTracker with test store."""
    return TrendTracker(metrics_store, threshold_percentage=5.0)


def create_project_metrics(
    project_root: str,
    avg_complexity: float,
    total_smells: int,
    avg_health_score: float,
    analyzed_at: datetime | None = None,
) -> ProjectMetrics:
    """Helper to create ProjectMetrics with specific values.

    Creates proper ChunkMetrics so complexity values propagate correctly
    through FileMetrics.compute_aggregates() and ProjectMetrics.compute_aggregates().
    """
    from mcp_vector_search.analysis.metrics import ChunkMetrics

    metrics = ProjectMetrics(project_root=project_root)

    if analyzed_at:
        metrics.analyzed_at = analyzed_at

    # Create sample file with specific metrics
    file_metrics = FileMetrics(
        file_path="test.py",
        total_lines=100,
        code_lines=80,
        comment_lines=10,
        blank_lines=10,
        function_count=5,
        class_count=1,
        method_count=3,
    )

    # Create chunks with the desired average complexity
    # Create 5 chunks (matching function_count) with complexities that average to avg_complexity
    num_chunks = 5

    # Distribute smells evenly across chunks
    # If total_smells=10 and num_chunks=5, each chunk gets 2 smells
    smells_per_chunk = total_smells // num_chunks
    remaining_smells = total_smells % num_chunks

    # Calculate chunk complexities that average to target
    # Use floor for most chunks, then add remaining fractional part to last chunk
    base_complexity = int(avg_complexity)
    total_needed = avg_complexity * num_chunks
    total_base = base_complexity * num_chunks
    remainder = round(total_needed - total_base)  # Extra complexity to distribute

    for i in range(num_chunks):
        # Distribute remaining smells to first few chunks
        chunk_smell_count = smells_per_chunk + (1 if i < remaining_smells else 0)
        chunk_smells = [f"smell_{j}" for j in range(chunk_smell_count)]

        # Add remainder to last chunk to achieve exact average
        chunk_complexity = base_complexity + (remainder if i == num_chunks - 1 else 0)

        chunk = ChunkMetrics(
            cognitive_complexity=chunk_complexity,
            cyclomatic_complexity=int(chunk_complexity * 0.8),
            max_nesting_depth=min(3, int(chunk_complexity / 5)),
            parameter_count=2,
            lines_of_code=20,
            smells=chunk_smells,
        )
        file_metrics.chunks.append(chunk)

    # Compute aggregates from chunks (sets avg_complexity, total_complexity, max_complexity)
    file_metrics.compute_aggregates()

    # Add to project
    metrics.files[file_metrics.file_path] = file_metrics

    # Compute project aggregates (sets avg_file_complexity from files with chunks)
    metrics.compute_aggregates()

    return metrics


class TestTrendTrackerInitialization:
    """Test TrendTracker initialization."""

    def test_init_with_valid_threshold(self, metrics_store):
        """Test initialization with valid threshold."""
        tracker = TrendTracker(metrics_store, threshold_percentage=5.0)
        assert tracker.threshold == 0.05
        assert tracker.store == metrics_store

    def test_init_with_custom_threshold(self, metrics_store):
        """Test initialization with custom threshold."""
        tracker = TrendTracker(metrics_store, threshold_percentage=10.0)
        assert tracker.threshold == 0.1

    def test_init_with_negative_threshold_raises(self, metrics_store):
        """Test that negative threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold_percentage must be between"):
            TrendTracker(metrics_store, threshold_percentage=-5.0)

    def test_init_with_threshold_over_100_raises(self, metrics_store):
        """Test that threshold > 100 raises ValueError."""
        with pytest.raises(ValueError, match="threshold_percentage must be between"):
            TrendTracker(metrics_store, threshold_percentage=150.0)


class TestCalculateTrendDirection:
    """Test calculate_trend_direction method."""

    def test_improving_trend_lower_is_better(self, trend_tracker):
        """Test detecting improving trend when values decrease."""
        trend_data = [
            (datetime.now() - timedelta(days=30), 20.0),
            (datetime.now() - timedelta(days=15), 15.0),
            (datetime.now(), 10.0),
        ]
        direction = trend_tracker.calculate_trend_direction(
            trend_data, lower_is_better=True
        )
        assert direction == TrendDirection.IMPROVING

    def test_worsening_trend_lower_is_better(self, trend_tracker):
        """Test detecting worsening trend when values increase."""
        trend_data = [
            (datetime.now() - timedelta(days=30), 10.0),
            (datetime.now() - timedelta(days=15), 15.0),
            (datetime.now(), 20.0),
        ]
        direction = trend_tracker.calculate_trend_direction(
            trend_data, lower_is_better=True
        )
        assert direction == TrendDirection.WORSENING

    def test_stable_trend_below_threshold(self, trend_tracker):
        """Test detecting stable trend when change is below threshold."""
        trend_data = [
            (datetime.now() - timedelta(days=30), 10.0),
            (datetime.now(), 10.3),  # 3% change, below 5% threshold
        ]
        direction = trend_tracker.calculate_trend_direction(
            trend_data, lower_is_better=True
        )
        assert direction == TrendDirection.STABLE

    def test_improving_trend_higher_is_better(self, trend_tracker):
        """Test detecting improving trend for health score (higher is better)."""
        trend_data = [
            (datetime.now() - timedelta(days=30), 0.7),
            (datetime.now(), 0.85),  # Increasing health is good
        ]
        direction = trend_tracker.calculate_trend_direction(
            trend_data, lower_is_better=False
        )
        assert direction == TrendDirection.IMPROVING

    def test_worsening_trend_higher_is_better(self, trend_tracker):
        """Test detecting worsening trend for health score (higher is better)."""
        trend_data = [
            (datetime.now() - timedelta(days=30), 0.85),
            (datetime.now(), 0.7),  # Decreasing health is bad
        ]
        direction = trend_tracker.calculate_trend_direction(
            trend_data, lower_is_better=False
        )
        assert direction == TrendDirection.WORSENING

    def test_insufficient_data_returns_stable(self, trend_tracker):
        """Test that insufficient data returns stable."""
        trend_data = [(datetime.now(), 10.0)]  # Only one data point
        direction = trend_tracker.calculate_trend_direction(trend_data)
        assert direction == TrendDirection.STABLE

    def test_zero_baseline_handled_correctly(self, trend_tracker):
        """Test handling of zero baseline value."""
        trend_data = [
            (datetime.now() - timedelta(days=30), 0.0),
            (datetime.now(), 10.0),
        ]
        direction = trend_tracker.calculate_trend_direction(
            trend_data, lower_is_better=True
        )
        # From 0 to 10 is worsening (for complexity)
        assert direction == TrendDirection.WORSENING


class TestGetTrends:
    """Test get_trends method."""

    def test_get_trends_with_insufficient_snapshots(self, trend_tracker):
        """Test get_trends with less than 2 snapshots."""
        project_path = "/test/project"
        trends = trend_tracker.get_trends(project_path, days=30)

        assert trends.project_path == project_path
        assert trends.period_days == 30
        assert len(trends.snapshots) == 0
        assert trends.complexity_direction == TrendDirection.STABLE

    def test_get_trends_with_improving_complexity(self, metrics_store, trend_tracker):
        """Test get_trends with improving complexity."""
        project_path = "/test/project"

        # Create snapshots with decreasing complexity
        now = datetime.now()
        metrics1 = create_project_metrics(
            project_path,
            avg_complexity=20.0,
            total_smells=10,
            avg_health_score=0.7,
            analyzed_at=now - timedelta(days=30),
        )
        metrics2 = create_project_metrics(
            project_path,
            avg_complexity=15.0,
            total_smells=7,
            avg_health_score=0.8,
            analyzed_at=now - timedelta(days=15),
        )
        metrics3 = create_project_metrics(
            project_path,
            avg_complexity=10.0,
            total_smells=5,
            avg_health_score=0.9,
            analyzed_at=now,
        )

        # Save snapshots
        metrics_store.save_complete_snapshot(metrics1)
        metrics_store.save_complete_snapshot(metrics2)
        metrics_store.save_complete_snapshot(metrics3)

        # Get trends
        trends = trend_tracker.get_trends(project_path, days=35)

        assert len(trends.snapshots) == 3
        assert trends.complexity_direction == TrendDirection.IMPROVING
        assert trends.smell_direction == TrendDirection.IMPROVING
        assert trends.health_direction == TrendDirection.IMPROVING
        assert trends.improving is True

    def test_get_trends_with_worsening_complexity(self, metrics_store, trend_tracker):
        """Test get_trends with worsening complexity."""
        project_path = "/test/project2"

        # Create snapshots with increasing complexity
        now = datetime.now()
        metrics1 = create_project_metrics(
            project_path,
            avg_complexity=10.0,
            total_smells=5,
            avg_health_score=0.9,
            analyzed_at=now - timedelta(days=30),
        )
        metrics2 = create_project_metrics(
            project_path,
            avg_complexity=20.0,
            total_smells=15,
            avg_health_score=0.7,
            analyzed_at=now,
        )

        # Save snapshots
        metrics_store.save_complete_snapshot(metrics1)
        metrics_store.save_complete_snapshot(metrics2)

        # Get trends
        trends = trend_tracker.get_trends(project_path, days=35)

        assert len(trends.snapshots) == 2
        assert trends.complexity_direction == TrendDirection.WORSENING
        assert trends.improving is False
        assert trends.has_regressions is True

    def test_get_trends_with_stable_metrics(self, metrics_store, trend_tracker):
        """Test get_trends with stable metrics (below threshold)."""
        project_path = "/test/project3"

        # Create snapshots with minimal change (3% change, below 5% threshold)
        # Use 9.0 and 9.2 to stay within same health score complexity band (5-10)
        now = datetime.now()
        metrics1 = create_project_metrics(
            project_path,
            avg_complexity=9.0,
            total_smells=5,
            avg_health_score=0.8,
            analyzed_at=now - timedelta(days=30),
        )
        metrics2 = create_project_metrics(
            project_path,
            avg_complexity=9.2,
            total_smells=5,
            avg_health_score=0.81,
            analyzed_at=now,
        )

        # Save snapshots
        metrics_store.save_complete_snapshot(metrics1)
        metrics_store.save_complete_snapshot(metrics2)

        # Get trends
        trends = trend_tracker.get_trends(project_path, days=35)

        assert trends.complexity_direction == TrendDirection.STABLE
        assert trends.smell_direction == TrendDirection.STABLE
        assert trends.health_direction == TrendDirection.STABLE

    def test_get_trends_with_invalid_days(self, trend_tracker):
        """Test get_trends with invalid days parameter."""
        with pytest.raises(ValueError, match="days must be positive"):
            trend_tracker.get_trends("/test/project", days=0)

        with pytest.raises(ValueError, match="days must be positive"):
            trend_tracker.get_trends("/test/project", days=-10)


class TestRegressionAlerts:
    """Test regression alert methods."""

    def test_get_regression_alerts_with_regressions(self, metrics_store, trend_tracker):
        """Test get_regression_alerts identifies regressions."""
        project_path = "/test/project4"

        # Create snapshots with significant regression
        now = datetime.now()
        metrics1 = create_project_metrics(
            project_path,
            avg_complexity=10.0,
            total_smells=5,
            avg_health_score=0.9,
            analyzed_at=now - timedelta(days=30),
        )
        metrics2 = create_project_metrics(
            project_path,
            avg_complexity=20.0,
            total_smells=15,
            avg_health_score=0.7,
            analyzed_at=now,
        )

        metrics_store.save_complete_snapshot(metrics1)
        metrics_store.save_complete_snapshot(metrics2)

        # Get regression alerts
        regressions = trend_tracker.get_regression_alerts(project_path, days=35)

        assert len(regressions) > 0
        # Should have complexity and smell regressions
        regression_metrics = [r.metric_name for r in regressions]
        assert "avg_complexity" in regression_metrics
        assert "total_smells" in regression_metrics

    def test_get_improvement_alerts_with_improvements(
        self, metrics_store, trend_tracker
    ):
        """Test get_improvement_alerts identifies improvements."""
        project_path = "/test/project5"

        # Create snapshots with significant improvement
        now = datetime.now()
        metrics1 = create_project_metrics(
            project_path,
            avg_complexity=20.0,
            total_smells=15,
            avg_health_score=0.7,
            analyzed_at=now - timedelta(days=30),
        )
        metrics2 = create_project_metrics(
            project_path,
            avg_complexity=10.0,
            total_smells=5,
            avg_health_score=0.9,
            analyzed_at=now,
        )

        metrics_store.save_complete_snapshot(metrics1)
        metrics_store.save_complete_snapshot(metrics2)

        # Get improvement alerts
        improvements = trend_tracker.get_improvement_alerts(project_path, days=35)

        assert len(improvements) > 0
        # Should have complexity and smell improvements
        improvement_metrics = [i.metric_name for i in improvements]
        assert "avg_complexity" in improvement_metrics
        assert "total_smells" in improvement_metrics


class TestTrendDataProperties:
    """Test TrendData properties and methods."""

    def test_trend_data_improving_property(self, metrics_store, trend_tracker):
        """Test TrendData.improving property."""
        project_path = "/test/project6"

        # Create improving snapshots
        now = datetime.now()
        metrics1 = create_project_metrics(
            project_path,
            avg_complexity=20.0,
            total_smells=10,
            avg_health_score=0.7,
            analyzed_at=now - timedelta(days=30),
        )
        metrics2 = create_project_metrics(
            project_path,
            avg_complexity=10.0,
            total_smells=5,
            avg_health_score=0.9,
            analyzed_at=now,
        )

        metrics_store.save_complete_snapshot(metrics1)
        metrics_store.save_complete_snapshot(metrics2)

        trends = trend_tracker.get_trends(project_path, days=35)

        assert trends.improving is True

    def test_trend_data_has_regressions_property(self, metrics_store, trend_tracker):
        """Test TrendData.has_regressions property."""
        project_path = "/test/project7"

        # Create regressing snapshots
        now = datetime.now()
        metrics1 = create_project_metrics(
            project_path,
            avg_complexity=10.0,
            total_smells=5,
            avg_health_score=0.9,
            analyzed_at=now - timedelta(days=30),
        )
        metrics2 = create_project_metrics(
            project_path,
            avg_complexity=20.0,
            total_smells=15,
            avg_health_score=0.7,
            analyzed_at=now,
        )

        metrics_store.save_complete_snapshot(metrics1)
        metrics_store.save_complete_snapshot(metrics2)

        trends = trend_tracker.get_trends(project_path, days=35)

        assert trends.has_regressions is True
        assert len(trends.critical_regressions) > 0


class TestPercentageChangeCalculation:
    """Test internal percentage change calculation."""

    def test_calculate_percentage_change_increase(self, trend_tracker):
        """Test percentage change calculation for increase."""
        change = trend_tracker._calculate_percentage_change(10.0, 15.0)
        assert change == 50.0  # 50% increase

    def test_calculate_percentage_change_decrease(self, trend_tracker):
        """Test percentage change calculation for decrease."""
        change = trend_tracker._calculate_percentage_change(20.0, 15.0)
        assert change == -25.0  # 25% decrease

    def test_calculate_percentage_change_from_zero(self, trend_tracker):
        """Test percentage change when old value is zero."""
        change = trend_tracker._calculate_percentage_change(0.0, 10.0)
        assert change == 100.0  # 100% increase from zero

    def test_calculate_percentage_change_to_zero(self, trend_tracker):
        """Test percentage change when new value is zero."""
        change = trend_tracker._calculate_percentage_change(10.0, 0.0)
        assert change == -100.0  # 100% decrease

    def test_calculate_percentage_change_both_zero(self, trend_tracker):
        """Test percentage change when both values are zero."""
        change = trend_tracker._calculate_percentage_change(0.0, 0.0)
        assert change == 0.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_trends_with_single_snapshot(self, metrics_store, trend_tracker):
        """Test trend analysis with only one snapshot."""
        project_path = "/test/project8"

        metrics = create_project_metrics(
            project_path, avg_complexity=10.0, total_smells=5, avg_health_score=0.8
        )
        metrics_store.save_complete_snapshot(metrics)

        trends = trend_tracker.get_trends(project_path, days=30)

        # Should return basic structure with no trend direction
        assert len(trends.snapshots) == 1
        assert trends.complexity_direction == TrendDirection.STABLE
        assert trends.improving is False
        assert trends.has_regressions is False

    def test_trends_with_nonexistent_project(self, trend_tracker):
        """Test trend analysis for project with no snapshots."""
        trends = trend_tracker.get_trends("/nonexistent/project", days=30)

        assert len(trends.snapshots) == 0
        assert trends.complexity_direction == TrendDirection.STABLE

    def test_custom_threshold_affects_trend_direction(self, metrics_store):
        """Test that custom threshold affects trend detection."""
        project_path = "/test/project9"

        # Create snapshots with 4% change (below 5%, above 2%)
        now = datetime.now()
        metrics1 = create_project_metrics(
            project_path,
            avg_complexity=10.0,
            total_smells=10,
            avg_health_score=0.8,
            analyzed_at=now - timedelta(days=30),
        )
        metrics2 = create_project_metrics(
            project_path,
            avg_complexity=10.4,
            total_smells=10,
            avg_health_score=0.8,
            analyzed_at=now,
        )

        metrics_store.save_complete_snapshot(metrics1)
        metrics_store.save_complete_snapshot(metrics2)

        # With 5% threshold, should be stable
        tracker_5pct = TrendTracker(metrics_store, threshold_percentage=5.0)
        trends_5pct = tracker_5pct.get_trends(project_path, days=35)
        assert trends_5pct.complexity_direction == TrendDirection.STABLE

        # With 2% threshold, should detect change
        tracker_2pct = TrendTracker(metrics_store, threshold_percentage=2.0)
        trends_2pct = tracker_2pct.get_trends(project_path, days=35)
        assert trends_2pct.complexity_direction == TrendDirection.WORSENING
