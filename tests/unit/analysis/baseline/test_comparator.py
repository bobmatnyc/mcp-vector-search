"""Unit tests for BaselineComparator.

Tests cover:
- Metric comparison and delta calculation
- Classification (regression/improvement/neutral)
- File comparison logic
- New/deleted file handling
- Summary statistics computation
"""

from __future__ import annotations

from datetime import datetime

import pytest

from mcp_vector_search.analysis.baseline.comparator import (
    BaselineComparator,
    ComparisonResult,
    FileComparison,
    MetricChange,
)
from mcp_vector_search.analysis.metrics import ChunkMetrics, FileMetrics, ProjectMetrics


@pytest.fixture
def comparator() -> BaselineComparator:
    """Create BaselineComparator instance."""
    return BaselineComparator()


@pytest.fixture
def baseline_metrics() -> ProjectMetrics:
    """Create baseline ProjectMetrics for testing."""
    chunks = [
        ChunkMetrics(
            cognitive_complexity=10,
            cyclomatic_complexity=5,
            max_nesting_depth=3,
            parameter_count=2,
            lines_of_code=20,
        ),
        ChunkMetrics(
            cognitive_complexity=5,
            cyclomatic_complexity=3,
            max_nesting_depth=2,
            parameter_count=1,
            lines_of_code=10,
        ),
    ]

    file_metrics = FileMetrics(
        file_path="src/example.py",
        total_lines=100,
        code_lines=80,
        comment_lines=15,
        blank_lines=5,
        function_count=2,
        class_count=1,
        method_count=2,
        chunks=chunks,
    )
    file_metrics.compute_aggregates()

    metrics = ProjectMetrics(
        project_root="/path/to/project",
        analyzed_at=datetime.now(),
        files={"src/example.py": file_metrics},
    )
    metrics.compute_aggregates()

    return metrics


@pytest.fixture
def current_metrics_improved() -> ProjectMetrics:
    """Create current ProjectMetrics with improvements."""
    chunks = [
        ChunkMetrics(
            cognitive_complexity=8,  # Improved from 10
            cyclomatic_complexity=4,  # Improved from 5
            max_nesting_depth=2,  # Improved from 3
            parameter_count=2,
            lines_of_code=20,
        ),
        ChunkMetrics(
            cognitive_complexity=4,  # Improved from 5
            cyclomatic_complexity=2,  # Improved from 3
            max_nesting_depth=2,
            parameter_count=1,
            lines_of_code=10,
        ),
    ]

    file_metrics = FileMetrics(
        file_path="src/example.py",
        total_lines=100,
        code_lines=80,
        comment_lines=15,
        blank_lines=5,
        function_count=2,
        class_count=1,
        method_count=2,
        chunks=chunks,
    )
    file_metrics.compute_aggregates()

    metrics = ProjectMetrics(
        project_root="/path/to/project",
        analyzed_at=datetime.now(),
        files={"src/example.py": file_metrics},
    )
    metrics.compute_aggregates()

    return metrics


@pytest.fixture
def current_metrics_regressed() -> ProjectMetrics:
    """Create current ProjectMetrics with regressions."""
    chunks = [
        ChunkMetrics(
            cognitive_complexity=15,  # Regressed from 10
            cyclomatic_complexity=8,  # Regressed from 5
            max_nesting_depth=4,  # Regressed from 3
            parameter_count=2,
            lines_of_code=20,
        ),
        ChunkMetrics(
            cognitive_complexity=8,  # Regressed from 5
            cyclomatic_complexity=5,  # Regressed from 3
            max_nesting_depth=3,  # Regressed from 2
            parameter_count=1,
            lines_of_code=10,
        ),
    ]

    file_metrics = FileMetrics(
        file_path="src/example.py",
        total_lines=100,
        code_lines=80,
        comment_lines=15,
        blank_lines=5,
        function_count=2,
        class_count=1,
        method_count=2,
        chunks=chunks,
    )
    file_metrics.compute_aggregates()

    metrics = ProjectMetrics(
        project_root="/path/to/project",
        analyzed_at=datetime.now(),
        files={"src/example.py": file_metrics},
    )
    metrics.compute_aggregates()

    return metrics


class TestMetricChange:
    """Test MetricChange dataclass."""

    def test_metric_change_properties(self) -> None:
        """Test MetricChange property methods."""
        regression = MetricChange(
            metric_name="cognitive_complexity",
            baseline_value=10,
            current_value=15,
            absolute_delta=5,
            percentage_delta=50.0,
            classification="regression",
        )

        assert regression.is_regression
        assert not regression.is_improvement
        assert not regression.is_neutral

        improvement = MetricChange(
            metric_name="cognitive_complexity",
            baseline_value=15,
            current_value=10,
            absolute_delta=-5,
            percentage_delta=-33.33,
            classification="improvement",
        )

        assert improvement.is_improvement
        assert not improvement.is_regression
        assert not improvement.is_neutral


class TestFileComparison:
    """Test FileComparison dataclass."""

    def test_has_regressions(self) -> None:
        """Test detecting regressions in file comparison."""
        comparison = FileComparison(
            file_path="test.py",
            in_baseline=True,
            in_current=True,
            metric_changes=[
                MetricChange("cc", 10, 15, 5, 50.0, "regression"),
                MetricChange("cyc", 5, 5, 0, 0.0, "neutral"),
            ],
        )

        assert comparison.has_regressions
        assert not comparison.has_improvements

    def test_has_improvements(self) -> None:
        """Test detecting improvements in file comparison."""
        comparison = FileComparison(
            file_path="test.py",
            in_baseline=True,
            in_current=True,
            metric_changes=[
                MetricChange("cc", 15, 10, -5, -33.33, "improvement"),
                MetricChange("cyc", 5, 5, 0, 0.0, "neutral"),
            ],
        )

        assert comparison.has_improvements
        assert not comparison.has_regressions

    def test_is_new_file(self) -> None:
        """Test detecting new files."""
        comparison = FileComparison(
            file_path="new.py", in_baseline=False, in_current=True
        )

        assert comparison.is_new_file
        assert not comparison.is_deleted_file

    def test_is_deleted_file(self) -> None:
        """Test detecting deleted files."""
        comparison = FileComparison(
            file_path="deleted.py", in_baseline=True, in_current=False
        )

        assert comparison.is_deleted_file
        assert not comparison.is_new_file


class TestBaselineComparator:
    """Test suite for BaselineComparator."""

    def test_compare_returns_result(
        self,
        comparator: BaselineComparator,
        baseline_metrics: ProjectMetrics,
        current_metrics_improved: ProjectMetrics,
    ) -> None:
        """Test compare returns ComparisonResult."""
        result = comparator.compare(current_metrics_improved, baseline_metrics)

        assert isinstance(result, ComparisonResult)
        assert result.baseline_name == "baseline"

    def test_compare_detects_improvements(
        self,
        comparator: BaselineComparator,
        baseline_metrics: ProjectMetrics,
        current_metrics_improved: ProjectMetrics,
    ) -> None:
        """Test compare detects improvements."""
        result = comparator.compare(
            current_metrics_improved, baseline_metrics, threshold_percent=5.0
        )

        assert result.has_improvements
        assert len(result.improvements) > 0
        assert not result.has_regressions

    def test_compare_detects_regressions(
        self,
        comparator: BaselineComparator,
        baseline_metrics: ProjectMetrics,
        current_metrics_regressed: ProjectMetrics,
    ) -> None:
        """Test compare detects regressions."""
        result = comparator.compare(
            current_metrics_regressed, baseline_metrics, threshold_percent=5.0
        )

        assert result.has_regressions
        assert len(result.regressions) > 0
        assert not result.has_improvements

    def test_compare_detects_new_files(
        self, comparator: BaselineComparator, baseline_metrics: ProjectMetrics
    ) -> None:
        """Test compare detects new files."""
        # Create current with new file
        current = ProjectMetrics(
            project_root="/path/to/project",
            analyzed_at=datetime.now(),
            files={
                "src/example.py": baseline_metrics.files["src/example.py"],
                "src/new_file.py": FileMetrics(
                    file_path="src/new_file.py",
                    total_lines=50,
                    code_lines=40,
                    function_count=3,
                    chunks=[
                        ChunkMetrics(
                            cognitive_complexity=5,
                            cyclomatic_complexity=3,
                            lines_of_code=15,
                        )
                    ],
                ),
            },
        )
        current.compute_aggregates()

        result = comparator.compare(current, baseline_metrics)

        assert len(result.new_files) == 1
        assert result.new_files[0].file_path == "src/new_file.py"
        assert result.new_files[0].is_new_file

    def test_compare_detects_deleted_files(
        self, comparator: BaselineComparator, baseline_metrics: ProjectMetrics
    ) -> None:
        """Test compare detects deleted files."""
        # Create current without the file
        current = ProjectMetrics(
            project_root="/path/to/project", analyzed_at=datetime.now(), files={}
        )
        current.compute_aggregates()

        result = comparator.compare(current, baseline_metrics)

        assert len(result.deleted_files) == 1
        assert result.deleted_files[0].file_path == "src/example.py"
        assert result.deleted_files[0].is_deleted_file

    def test_calculate_metric_change_with_increase(
        self, comparator: BaselineComparator
    ) -> None:
        """Test calculating metric change with increase."""
        change = comparator._calculate_metric_change(
            metric_name="cognitive_complexity",
            baseline_value=10,
            current_value=15,
            threshold_percent=5.0,
        )

        assert change.absolute_delta == 5
        assert change.percentage_delta == 50.0
        assert change.classification == "regression"

    def test_calculate_metric_change_with_decrease(
        self, comparator: BaselineComparator
    ) -> None:
        """Test calculating metric change with decrease."""
        change = comparator._calculate_metric_change(
            metric_name="cognitive_complexity",
            baseline_value=15,
            current_value=10,
            threshold_percent=5.0,
        )

        assert change.absolute_delta == -5
        assert change.percentage_delta == pytest.approx(-33.33, rel=0.01)
        assert change.classification == "improvement"

    def test_calculate_metric_change_within_threshold(
        self, comparator: BaselineComparator
    ) -> None:
        """Test metric change within neutral threshold."""
        change = comparator._calculate_metric_change(
            metric_name="cognitive_complexity",
            baseline_value=100,
            current_value=103,  # 3% change
            threshold_percent=5.0,
        )

        assert change.classification == "neutral"

    def test_calculate_metric_change_zero_baseline(
        self, comparator: BaselineComparator
    ) -> None:
        """Test handling zero baseline value."""
        change = comparator._calculate_metric_change(
            metric_name="cognitive_complexity",
            baseline_value=0,
            current_value=10,
            threshold_percent=5.0,
        )

        assert change.absolute_delta == 10
        assert change.percentage_delta == 100.0
        assert change.classification == "regression"

    def test_classify_change_complexity_increase(
        self, comparator: BaselineComparator
    ) -> None:
        """Test classifying complexity increase as regression."""
        classification = comparator._classify_change(
            metric_name="cognitive_complexity",
            absolute_delta=5,
            percentage_delta=50.0,
            threshold_percent=5.0,
        )

        assert classification == "regression"

    def test_classify_change_complexity_decrease(
        self, comparator: BaselineComparator
    ) -> None:
        """Test classifying complexity decrease as improvement."""
        classification = comparator._classify_change(
            metric_name="cognitive_complexity",
            absolute_delta=-5,
            percentage_delta=-33.33,
            threshold_percent=5.0,
        )

        assert classification == "improvement"

    def test_classify_change_count_metrics_neutral(
        self, comparator: BaselineComparator
    ) -> None:
        """Test count metrics are classified as neutral."""
        classification = comparator._classify_change(
            metric_name="function_count",
            absolute_delta=5,
            percentage_delta=50.0,
            threshold_percent=5.0,
        )

        # Function count increase is neutral (could be refactoring)
        assert classification == "neutral"

    def test_compute_summary_statistics(
        self,
        comparator: BaselineComparator,
        baseline_metrics: ProjectMetrics,
        current_metrics_improved: ProjectMetrics,
    ) -> None:
        """Test summary statistics computation."""
        summary = comparator._compute_summary(
            current_metrics_improved, baseline_metrics
        )

        assert "total_files_current" in summary
        assert "total_files_baseline" in summary
        assert "total_functions_current" in summary
        assert "total_functions_baseline" in summary
        assert "total_complexity_current" in summary
        assert "total_complexity_baseline" in summary
        assert "avg_complexity_current" in summary
        assert "avg_complexity_baseline" in summary
        assert "max_complexity_current" in summary
        assert "max_complexity_baseline" in summary

    def test_compare_with_custom_threshold(
        self,
        comparator: BaselineComparator,
        baseline_metrics: ProjectMetrics,
        current_metrics_improved: ProjectMetrics,
    ) -> None:
        """Test comparison with custom threshold."""
        # With high threshold, small improvements should be neutral
        result_high_threshold = comparator.compare(
            current_metrics_improved, baseline_metrics, threshold_percent=50.0
        )

        # With low threshold, same improvements should be detected
        result_low_threshold = comparator.compare(
            current_metrics_improved, baseline_metrics, threshold_percent=1.0
        )

        # More improvements detected with lower threshold
        assert len(result_low_threshold.improvements) >= len(
            result_high_threshold.improvements
        )

    def test_compare_identical_metrics(
        self, comparator: BaselineComparator, baseline_metrics: ProjectMetrics
    ) -> None:
        """Test comparing identical metrics."""
        result = comparator.compare(baseline_metrics, baseline_metrics)

        assert not result.has_regressions
        assert not result.has_improvements
        assert len(result.unchanged) > 0 or result.total_files_compared > 0

    def test_compare_multiple_files(self, comparator: BaselineComparator) -> None:
        """Test comparison with multiple files."""
        # Create file metrics and compute aggregates
        baseline_file1 = FileMetrics(
            file_path="file1.py",
            total_lines=50,
            function_count=5,
            chunks=[ChunkMetrics(cognitive_complexity=10, cyclomatic_complexity=5)],
        )
        baseline_file1.compute_aggregates()

        baseline_file2 = FileMetrics(
            file_path="file2.py",
            total_lines=60,
            function_count=6,
            chunks=[ChunkMetrics(cognitive_complexity=15, cyclomatic_complexity=8)],
        )
        baseline_file2.compute_aggregates()

        baseline_file3 = FileMetrics(
            file_path="file3.py",
            total_lines=40,
            function_count=4,
            chunks=[ChunkMetrics(cognitive_complexity=8, cyclomatic_complexity=4)],
        )
        baseline_file3.compute_aggregates()

        # Create baseline with 3 files
        baseline = ProjectMetrics(
            project_root="/test",
            files={
                "file1.py": baseline_file1,
                "file2.py": baseline_file2,
                "file3.py": baseline_file3,
            },
        )
        baseline.compute_aggregates()

        # Create current file metrics
        current_file1 = FileMetrics(  # Improved
            file_path="file1.py",
            total_lines=50,
            function_count=5,
            chunks=[ChunkMetrics(cognitive_complexity=8, cyclomatic_complexity=4)],
        )
        current_file1.compute_aggregates()

        current_file2 = FileMetrics(  # Regressed
            file_path="file2.py",
            total_lines=60,
            function_count=6,
            chunks=[ChunkMetrics(cognitive_complexity=20, cyclomatic_complexity=12)],
        )
        current_file2.compute_aggregates()

        current_file3 = FileMetrics(  # Unchanged
            file_path="file3.py",
            total_lines=40,
            function_count=4,
            chunks=[ChunkMetrics(cognitive_complexity=8, cyclomatic_complexity=4)],
        )
        current_file3.compute_aggregates()

        # Create current with improvements, regressions, and unchanged
        current = ProjectMetrics(
            project_root="/test",
            files={
                "file1.py": current_file1,
                "file2.py": current_file2,
                "file3.py": current_file3,
            },
        )
        current.compute_aggregates()

        result = comparator.compare(current, baseline, threshold_percent=5.0)

        assert result.total_files_compared == 3
        assert len(result.improvements) > 0
        assert len(result.regressions) > 0

    def test_comparison_result_total_files_count(
        self, comparator: BaselineComparator
    ) -> None:
        """Test total_files_compared property."""
        baseline = ProjectMetrics(project_root="/test", files={})
        current = ProjectMetrics(project_root="/test", files={})

        result = comparator.compare(current, baseline)

        # Should count all categories
        expected_total = (
            len(result.regressions)
            + len(result.improvements)
            + len(result.unchanged)
            + len(result.new_files)
            + len(result.deleted_files)
        )

        assert result.total_files_compared == expected_total
