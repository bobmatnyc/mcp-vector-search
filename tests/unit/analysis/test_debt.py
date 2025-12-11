"""Unit tests for technical debt estimation."""

from __future__ import annotations

import pytest

from mcp_vector_search.analysis.collectors.smells import CodeSmell, SmellSeverity
from mcp_vector_search.analysis.debt import (
    DebtCategory,
    DebtItem,
    DebtSummary,
    RemediationTime,
    TechnicalDebtEstimator,
)
from mcp_vector_search.analysis.metrics import ChunkMetrics, FileMetrics, ProjectMetrics


class TestRemediationTime:
    """Test RemediationTime dataclass."""

    def test_remediation_time_creation(self) -> None:
        """Test creating a RemediationTime instance."""
        rt = RemediationTime(
            smell_type="long_method",
            base_minutes=20,
            category=DebtCategory.MAINTAINABILITY,
            description="Extract method refactoring",
        )

        assert rt.smell_type == "long_method"
        assert rt.base_minutes == 20
        assert rt.category == DebtCategory.MAINTAINABILITY
        assert rt.description == "Extract method refactoring"


class TestDebtItem:
    """Test DebtItem dataclass."""

    def test_debt_item_creation(self) -> None:
        """Test creating a DebtItem instance."""
        item = DebtItem(
            smell_type="long_method",
            file_path="src/example.py",
            line=42,
            severity="warning",
            base_minutes=20,
            adjusted_minutes=20.0,
            category=DebtCategory.MAINTAINABILITY,
            message="Function is too long",
        )

        assert item.smell_type == "long_method"
        assert item.file_path == "src/example.py"
        assert item.line == 42
        assert item.severity == "warning"
        assert item.base_minutes == 20
        assert item.adjusted_minutes == 20.0
        assert item.category == DebtCategory.MAINTAINABILITY


class TestDebtSummary:
    """Test DebtSummary dataclass."""

    def test_debt_summary_to_dict(self) -> None:
        """Test converting DebtSummary to dictionary."""
        # Create sample debt items
        items = [
            DebtItem(
                smell_type="long_method",
                file_path="src/a.py",
                line=10,
                severity="warning",
                base_minutes=20,
                adjusted_minutes=20.0,
                category=DebtCategory.MAINTAINABILITY,
                message="Too long",
            ),
            DebtItem(
                smell_type="deep_nesting",
                file_path="src/b.py",
                line=20,
                severity="error",
                base_minutes=15,
                adjusted_minutes=22.5,  # 15 * 1.5
                category=DebtCategory.COMPLEXITY,
                message="Too nested",
            ),
        ]

        summary = DebtSummary(
            total_minutes=42.5,
            total_hours=0.71,
            total_days=0.09,
            items_by_category={
                DebtCategory.MAINTAINABILITY: [items[0]],
                DebtCategory.COMPLEXITY: [items[1]],
            },
            minutes_by_category={
                DebtCategory.MAINTAINABILITY: 20.0,
                DebtCategory.COMPLEXITY: 22.5,
            },
            items_by_severity={
                "warning": [items[0]],
                "error": [items[1]],
            },
            minutes_by_severity={
                "warning": 20.0,
                "error": 22.5,
            },
            top_files=[("src/a.py", 20.0), ("src/b.py", 22.5)],
            top_smell_types=[("deep_nesting", 22.5), ("long_method", 20.0)],
            item_count=2,
        )

        result = summary.to_dict()

        assert result["total_minutes"] == 42.5
        assert result["total_hours"] == 0.71
        assert result["total_days"] == 0.09
        assert result["item_count"] == 2
        assert "by_category" in result
        assert "by_severity" in result
        assert "top_files" in result
        assert "top_smell_types" in result

        # Check category breakdown
        assert DebtCategory.MAINTAINABILITY.value in result["by_category"]
        assert (
            result["by_category"][DebtCategory.MAINTAINABILITY.value]["minutes"] == 20.0
        )
        assert result["by_category"][DebtCategory.COMPLEXITY.value]["minutes"] == 22.5

        # Check severity breakdown
        assert "warning" in result["by_severity"]
        assert result["by_severity"]["warning"]["minutes"] == 20.0
        assert result["by_severity"]["error"]["minutes"] == 22.5


class TestTechnicalDebtEstimator:
    """Test TechnicalDebtEstimator class."""

    def test_initialization(self) -> None:
        """Test estimator initialization with defaults."""
        estimator = TechnicalDebtEstimator()

        assert estimator.remediation_times is not None
        assert estimator.severity_multipliers is not None
        assert "long_method" in estimator.remediation_times
        assert "error" in estimator.severity_multipliers

    def test_initialization_with_custom_times(self) -> None:
        """Test estimator initialization with custom remediation times."""
        custom_times = {
            "custom_smell": RemediationTime(
                smell_type="custom_smell",
                base_minutes=99,
                category=DebtCategory.SECURITY,
                description="Custom fix",
            )
        }

        estimator = TechnicalDebtEstimator(remediation_times=custom_times)

        assert "custom_smell" in estimator.remediation_times
        assert estimator.remediation_times["custom_smell"].base_minutes == 99

    def test_get_remediation_time_known_smell(self) -> None:
        """Test getting remediation time for known smell type."""
        estimator = TechnicalDebtEstimator()

        time = estimator.get_remediation_time("long_method")

        assert time == 20

    def test_get_remediation_time_unknown_smell(self) -> None:
        """Test getting remediation time for unknown smell type."""
        estimator = TechnicalDebtEstimator()

        time = estimator.get_remediation_time("unknown_smell_type")

        assert time == 15  # Default fallback

    def test_apply_severity_multiplier_error(self) -> None:
        """Test severity multiplier for error level."""
        estimator = TechnicalDebtEstimator()

        adjusted = estimator.apply_severity_multiplier(20, "error")

        assert adjusted == 30.0  # 20 * 1.5

    def test_apply_severity_multiplier_warning(self) -> None:
        """Test severity multiplier for warning level."""
        estimator = TechnicalDebtEstimator()

        adjusted = estimator.apply_severity_multiplier(20, "warning")

        assert adjusted == 20.0  # 20 * 1.0

    def test_apply_severity_multiplier_info(self) -> None:
        """Test severity multiplier for info level."""
        estimator = TechnicalDebtEstimator()

        adjusted = estimator.apply_severity_multiplier(20, "info")

        assert adjusted == 10.0  # 20 * 0.5

    def test_apply_severity_multiplier_unknown(self) -> None:
        """Test severity multiplier for unknown severity."""
        estimator = TechnicalDebtEstimator()

        adjusted = estimator.apply_severity_multiplier(20, "unknown")

        assert adjusted == 20.0  # 20 * 1.0 (default)

    def test_normalize_smell_name(self) -> None:
        """Test smell name normalization."""
        estimator = TechnicalDebtEstimator()

        assert estimator._normalize_smell_name("Long Method") == "long_method"
        assert estimator._normalize_smell_name("Deep Nesting") == "deep_nesting"
        assert estimator._normalize_smell_name("God Class") == "god_class"
        assert estimator._normalize_smell_name("Complex Method") == "complex_method"

    def test_parse_location_with_line(self) -> None:
        """Test parsing location with line number."""
        estimator = TechnicalDebtEstimator()

        file_path, line = estimator._parse_location("src/example.py:42")

        assert file_path == "src/example.py"
        assert line == 42

    def test_parse_location_with_range(self) -> None:
        """Test parsing location with line range."""
        estimator = TechnicalDebtEstimator()

        file_path, line = estimator._parse_location("src/example.py:42-50")

        assert file_path == "src/example.py"
        assert line == 42  # Takes first line of range

    def test_parse_location_without_line(self) -> None:
        """Test parsing location without line number."""
        estimator = TechnicalDebtEstimator()

        file_path, line = estimator._parse_location("src/example.py")

        assert file_path == "src/example.py"
        assert line == 0

    def test_parse_location_invalid_line(self) -> None:
        """Test parsing location with invalid line number."""
        estimator = TechnicalDebtEstimator()

        file_path, line = estimator._parse_location("src/example.py:abc")

        assert file_path == "src/example.py"
        assert line == 0

    def test_estimate_from_smells_empty(self) -> None:
        """Test debt estimation with no smells."""
        estimator = TechnicalDebtEstimator()

        summary = estimator.estimate_from_smells([])

        assert summary.total_minutes == 0.0
        assert summary.total_hours == 0.0
        assert summary.total_days == 0.0
        assert summary.item_count == 0
        assert len(summary.items_by_category) == 0
        assert len(summary.items_by_severity) == 0

    def test_estimate_from_smells_single_smell(self) -> None:
        """Test debt estimation with a single smell."""
        estimator = TechnicalDebtEstimator()

        smells = [
            CodeSmell(
                name="Long Method",
                description="Method is too long: 60 lines",
                severity=SmellSeverity.WARNING,
                location="src/example.py:10",
                metric_value=60.0,
                threshold=50.0,
                suggestion="Extract smaller methods",
            )
        ]

        summary = estimator.estimate_from_smells(smells)

        assert summary.total_minutes == 20.0  # Base time for long_method
        assert summary.total_hours == pytest.approx(20.0 / 60, rel=0.01)
        assert summary.total_days == pytest.approx(20.0 / 60 / 8, rel=0.01)
        assert summary.item_count == 1
        assert DebtCategory.MAINTAINABILITY in summary.items_by_category
        assert "warning" in summary.items_by_severity

    def test_estimate_from_smells_multiple_smells(self) -> None:
        """Test debt estimation with multiple smells."""
        estimator = TechnicalDebtEstimator()

        smells = [
            CodeSmell(
                name="Long Method",
                description="Too long",
                severity=SmellSeverity.WARNING,
                location="src/a.py:10",
                metric_value=60.0,
                threshold=50.0,
            ),
            CodeSmell(
                name="Deep Nesting",
                description="Too nested",
                severity=SmellSeverity.ERROR,
                location="src/b.py:20",
                metric_value=6.0,
                threshold=4.0,
            ),
            CodeSmell(
                name="God Class",
                description="Too many responsibilities",
                severity=SmellSeverity.ERROR,
                location="src/c.py",
                metric_value=30.0,
                threshold=20.0,
            ),
        ]

        summary = estimator.estimate_from_smells(smells)

        # Expected: 20 (long_method) + 15*1.5 (deep_nesting) + 120*1.5 (god_class)
        expected_minutes = 20.0 + 22.5 + 180.0
        assert summary.total_minutes == expected_minutes
        assert summary.item_count == 3

        # Check categories
        assert DebtCategory.MAINTAINABILITY in summary.minutes_by_category
        assert DebtCategory.COMPLEXITY in summary.minutes_by_category
        assert (
            summary.minutes_by_category[DebtCategory.MAINTAINABILITY] == 200.0
        )  # 20 + 180
        assert summary.minutes_by_category[DebtCategory.COMPLEXITY] == 22.5

        # Check severities
        assert summary.minutes_by_severity["warning"] == 20.0
        assert summary.minutes_by_severity["error"] == 202.5  # 22.5 + 180

    def test_estimate_from_smells_severity_multipliers(self) -> None:
        """Test that severity multipliers are applied correctly."""
        estimator = TechnicalDebtEstimator()

        smells = [
            CodeSmell(
                name="Deep Nesting",
                description="Nested",
                severity=SmellSeverity.ERROR,
                location="src/a.py:10",
                metric_value=6.0,
                threshold=4.0,
            ),
            CodeSmell(
                name="Deep Nesting",
                description="Nested",
                severity=SmellSeverity.WARNING,
                location="src/b.py:20",
                metric_value=5.0,
                threshold=4.0,
            ),
            CodeSmell(
                name="Deep Nesting",
                description="Nested",
                severity=SmellSeverity.INFO,
                location="src/c.py:30",
                metric_value=5.0,
                threshold=4.0,
            ),
        ]

        summary = estimator.estimate_from_smells(smells)

        # Expected: 15*1.5 (error) + 15*1.0 (warning) + 15*0.5 (info)
        expected_minutes = 22.5 + 15.0 + 7.5
        assert summary.total_minutes == expected_minutes

    def test_estimate_from_smells_top_files(self) -> None:
        """Test that top files are calculated correctly."""
        estimator = TechnicalDebtEstimator()

        smells = [
            CodeSmell(
                name="Long Method",
                description="Long",
                severity=SmellSeverity.WARNING,
                location="src/a.py:10",
                metric_value=60.0,
                threshold=50.0,
            ),
            CodeSmell(
                name="Deep Nesting",
                description="Nested",
                severity=SmellSeverity.WARNING,
                location="src/a.py:20",
                metric_value=5.0,
                threshold=4.0,
            ),
            CodeSmell(
                name="God Class",
                description="Big",
                severity=SmellSeverity.ERROR,
                location="src/b.py",
                metric_value=30.0,
                threshold=20.0,
            ),
        ]

        summary = estimator.estimate_from_smells(smells)

        # src/a.py: 20 + 15 = 35
        # src/b.py: 120 * 1.5 = 180
        assert len(summary.top_files) == 2
        assert summary.top_files[0] == ("src/b.py", 180.0)
        assert summary.top_files[1] == ("src/a.py", 35.0)

    def test_estimate_from_smells_top_smell_types(self) -> None:
        """Test that top smell types are calculated correctly."""
        estimator = TechnicalDebtEstimator()

        smells = [
            CodeSmell(
                name="Long Method",
                description="Long",
                severity=SmellSeverity.WARNING,
                location="src/a.py:10",
                metric_value=60.0,
                threshold=50.0,
            ),
            CodeSmell(
                name="Long Method",
                description="Long",
                severity=SmellSeverity.WARNING,
                location="src/b.py:10",
                metric_value=60.0,
                threshold=50.0,
            ),
            CodeSmell(
                name="Deep Nesting",
                description="Nested",
                severity=SmellSeverity.WARNING,
                location="src/c.py:20",
                metric_value=5.0,
                threshold=4.0,
            ),
        ]

        summary = estimator.estimate_from_smells(smells)

        # long_method: 20 + 20 = 40
        # deep_nesting: 15
        assert len(summary.top_smell_types) == 2
        assert summary.top_smell_types[0] == ("long_method", 40.0)
        assert summary.top_smell_types[1] == ("deep_nesting", 15.0)

    def test_estimate_from_project_metrics(self) -> None:
        """Test debt estimation from project metrics."""
        estimator = TechnicalDebtEstimator()

        # Create file metrics with chunks that will trigger smells
        chunk1 = ChunkMetrics(
            cognitive_complexity=25,  # High complexity
            cyclomatic_complexity=15,  # High cyclomatic
            max_nesting_depth=5,  # Deep nesting
            parameter_count=6,  # Too many parameters
            lines_of_code=60,  # Long method
        )

        chunk2 = ChunkMetrics(
            cognitive_complexity=8,
            cyclomatic_complexity=5,
            max_nesting_depth=2,
            parameter_count=3,
            lines_of_code=30,
        )

        file1 = FileMetrics(
            file_path="src/file1.py",
            total_lines=100,
            code_lines=80,
            comment_lines=10,
            blank_lines=10,
            function_count=2,
            class_count=0,
            method_count=0,
            chunks=[chunk1, chunk2],
        )
        file1.compute_aggregates()

        # Create project metrics
        project = ProjectMetrics(
            project_root="/test/project",
            files={"src/file1.py": file1},
        )
        project.compute_aggregates()

        summary = estimator.estimate_from_project_metrics(project)

        # Should have detected multiple smells in chunk1
        assert summary.item_count > 0
        assert summary.total_minutes > 0

    def test_estimate_from_project_metrics_empty(self) -> None:
        """Test debt estimation from empty project metrics."""
        estimator = TechnicalDebtEstimator()

        project = ProjectMetrics(project_root="/test/project", files={})

        summary = estimator.estimate_from_project_metrics(project)

        assert summary.total_minutes == 0.0
        assert summary.item_count == 0

    def test_custom_remediation_times(self) -> None:
        """Test using custom remediation times."""
        custom_times = {
            "long_method": RemediationTime(
                smell_type="long_method",
                base_minutes=50,  # Custom time
                category=DebtCategory.MAINTAINABILITY,
                description="Custom fix",
            )
        }

        estimator = TechnicalDebtEstimator(remediation_times=custom_times)

        smells = [
            CodeSmell(
                name="Long Method",
                description="Too long",
                severity=SmellSeverity.WARNING,
                location="src/a.py:10",
                metric_value=60.0,
                threshold=50.0,
            )
        ]

        summary = estimator.estimate_from_smells(smells)

        assert summary.total_minutes == 50.0  # Custom time used

    def test_custom_severity_multipliers(self) -> None:
        """Test using custom severity multipliers."""
        custom_multipliers = {
            "error": 2.0,  # Custom multiplier
            "warning": 1.0,
            "info": 0.25,
        }

        estimator = TechnicalDebtEstimator(severity_multipliers=custom_multipliers)

        smells = [
            CodeSmell(
                name="Deep Nesting",
                description="Nested",
                severity=SmellSeverity.ERROR,
                location="src/a.py:10",
                metric_value=6.0,
                threshold=4.0,
            )
        ]

        summary = estimator.estimate_from_smells(smells)

        assert summary.total_minutes == 30.0  # 15 * 2.0

    def test_get_category_known_smell(self) -> None:
        """Test getting category for known smell type."""
        estimator = TechnicalDebtEstimator()

        category = estimator._get_category("long_method")

        assert category == DebtCategory.MAINTAINABILITY

    def test_get_category_unknown_smell(self) -> None:
        """Test getting category for unknown smell type."""
        estimator = TechnicalDebtEstimator()

        category = estimator._get_category("unknown_smell")

        assert category == DebtCategory.MAINTAINABILITY  # Default

    def test_create_summary_aggregations(self) -> None:
        """Test that summary aggregations are correct."""
        estimator = TechnicalDebtEstimator()

        items = [
            DebtItem(
                smell_type="long_method",
                file_path="src/a.py",
                line=10,
                severity="warning",
                base_minutes=20,
                adjusted_minutes=20.0,
                category=DebtCategory.MAINTAINABILITY,
                message="Long",
            ),
            DebtItem(
                smell_type="deep_nesting",
                file_path="src/a.py",
                line=20,
                severity="error",
                base_minutes=15,
                adjusted_minutes=22.5,
                category=DebtCategory.COMPLEXITY,
                message="Nested",
            ),
            DebtItem(
                smell_type="god_class",
                file_path="src/b.py",
                line=1,
                severity="error",
                base_minutes=120,
                adjusted_minutes=180.0,
                category=DebtCategory.MAINTAINABILITY,
                message="Big",
            ),
        ]

        summary = estimator._create_summary(items)

        # Check totals
        assert summary.total_minutes == 222.5
        assert summary.total_hours == pytest.approx(222.5 / 60, rel=0.01)
        assert summary.total_days == pytest.approx(222.5 / 60 / 8, rel=0.01)
        assert summary.item_count == 3

        # Check category aggregation
        assert summary.minutes_by_category[DebtCategory.MAINTAINABILITY] == 200.0
        assert summary.minutes_by_category[DebtCategory.COMPLEXITY] == 22.5
        assert len(summary.items_by_category[DebtCategory.MAINTAINABILITY]) == 2
        assert len(summary.items_by_category[DebtCategory.COMPLEXITY]) == 1

        # Check severity aggregation
        assert summary.minutes_by_severity["warning"] == 20.0
        assert summary.minutes_by_severity["error"] == 202.5
        assert len(summary.items_by_severity["warning"]) == 1
        assert len(summary.items_by_severity["error"]) == 2

        # Check top files
        assert summary.top_files[0] == ("src/b.py", 180.0)
        assert summary.top_files[1] == ("src/a.py", 42.5)

        # Check top smell types
        assert ("god_class", 180.0) in summary.top_smell_types
        assert ("deep_nesting", 22.5) in summary.top_smell_types
        assert ("long_method", 20.0) in summary.top_smell_types
