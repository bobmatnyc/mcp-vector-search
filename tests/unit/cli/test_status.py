"""Tests for the status CLI command with --metrics flag."""

from datetime import datetime

from typer.testing import CliRunner

from mcp_vector_search.analysis.storage.metrics_store import ProjectSnapshot
from mcp_vector_search.cli.commands.status import (
    _output_metrics_json,
    _print_metrics_summary,
    _status_indicator,
    status_app,
)

runner = CliRunner()


class TestStatusMetricsCommand:
    """Tests for status --metrics command."""

    def test_status_metrics_help(self):
        """Test that status --help includes metrics option."""
        result = runner.invoke(status_app, ["--help"])
        assert result.exit_code == 0
        assert "--metrics" in result.stdout or "-m" in result.stdout

    def test_status_metrics_no_database(self, tmp_path):
        """Test status --metrics when no metrics database exists."""
        result = runner.invoke(
            status_app, ["--metrics", "--project-root", str(tmp_path)]
        )
        assert result.exit_code == 1
        assert "No metrics found" in result.stdout
        assert "analyze" in result.stdout.lower()

    def test_status_metrics_empty_database(self, tmp_path):
        """Test status --metrics when database exists but has no snapshots."""
        # Create empty metrics database
        metrics_dir = tmp_path / ".mcp-vector-search"
        metrics_dir.mkdir()
        db_path = metrics_dir / "metrics.db"

        # Create empty database
        from mcp_vector_search.analysis.storage.metrics_store import MetricsStore

        store = MetricsStore(db_path)
        store.close()

        result = runner.invoke(
            status_app, ["--metrics", "--project-root", str(tmp_path)]
        )
        assert result.exit_code == 1
        assert "No metrics found" in result.stdout

    def test_status_metrics_with_data(self, tmp_path):
        """Test status --metrics with actual metrics data."""
        # Create metrics database with test data
        metrics_dir = tmp_path / ".mcp-vector-search"
        metrics_dir.mkdir()
        db_path = metrics_dir / "metrics.db"

        from mcp_vector_search.analysis.metrics import ProjectMetrics
        from mcp_vector_search.analysis.storage.metrics_store import MetricsStore

        # Create test metrics
        project_metrics = ProjectMetrics(project_root=str(tmp_path))
        project_metrics.total_files = 10
        project_metrics.total_lines = 1000
        project_metrics.total_functions = 50
        project_metrics.total_classes = 5

        # Save to database
        store = MetricsStore(db_path)
        store.save_project_snapshot(project_metrics)
        store.close()

        # Run command
        result = runner.invoke(
            status_app, ["--metrics", "--project-root", str(tmp_path)]
        )

        # Should succeed and show metrics
        assert result.exit_code == 0
        assert "Project Metrics Summary" in result.stdout
        assert "Files: 10" in result.stdout
        assert "Functions: 50" in result.stdout
        assert "Classes: 5" in result.stdout

    def test_status_metrics_json_output(self, tmp_path):
        """Test status --metrics --json output format."""
        import json

        # Create metrics database with test data
        metrics_dir = tmp_path / ".mcp-vector-search"
        metrics_dir.mkdir()
        db_path = metrics_dir / "metrics.db"

        from mcp_vector_search.analysis.metrics import ProjectMetrics
        from mcp_vector_search.analysis.storage.metrics_store import MetricsStore

        # Create test metrics
        project_metrics = ProjectMetrics(project_root=str(tmp_path))
        project_metrics.total_files = 10
        project_metrics.total_lines = 1000
        project_metrics.total_functions = 50
        project_metrics.total_classes = 5

        # Save to database
        store = MetricsStore(db_path)
        store.save_project_snapshot(project_metrics)
        store.close()

        # Run command with JSON output
        result = runner.invoke(
            status_app, ["--metrics", "--json", "--project-root", str(tmp_path)]
        )

        # Should succeed
        assert result.exit_code == 0

        # Parse JSON output
        output = json.loads(result.stdout)
        assert output["status"] == "success"
        assert output["metrics"]["files"]["total"] == 10
        assert output["metrics"]["functions"]["total"] == 50
        assert output["metrics"]["classes"]["total"] == 5
        assert output["metrics"]["lines"]["total"] == 1000


class TestStatusIndicator:
    """Tests for _status_indicator helper function."""

    def test_status_indicator_green(self):
        """Test green status indicator for good values."""
        result = _status_indicator(5.0, 10.0, 20.0)
        assert "green" in result
        assert "●" in result

    def test_status_indicator_yellow(self):
        """Test yellow status indicator for warning values."""
        result = _status_indicator(15.0, 10.0, 20.0)
        assert "yellow" in result
        assert "●" in result

    def test_status_indicator_red(self):
        """Test red status indicator for error values."""
        result = _status_indicator(25.0, 10.0, 20.0)
        assert "red" in result
        assert "●" in result

    def test_status_indicator_boundary_warning(self):
        """Test boundary at warning threshold."""
        result = _status_indicator(10.0, 10.0, 20.0)
        # At threshold should be yellow
        assert "yellow" in result

    def test_status_indicator_boundary_error(self):
        """Test boundary at error threshold."""
        result = _status_indicator(20.0, 10.0, 20.0)
        # At threshold should be red
        assert "red" in result


class TestMetricsOutput:
    """Tests for metrics output functions."""

    def test_output_metrics_json(self, capsys):
        """Test JSON output formatting."""
        # Create mock snapshot
        snapshot = ProjectSnapshot(
            snapshot_id=1,
            project_path="/test/path",
            timestamp=datetime(2024, 12, 11, 12, 0, 0),
            total_files=10,
            total_lines=1000,
            total_functions=50,
            total_classes=5,
            avg_complexity=8.5,
            max_complexity=25,
            total_complexity=425,
            total_smells=3,
            avg_health_score=0.85,
            grade_distribution={"A": 30, "B": 15, "C": 4, "D": 1, "F": 0},
            git_commit="abc123def456",
            git_branch="main",
            tool_version="1.0.0",
        )

        _output_metrics_json(snapshot)

        # Check output contains expected data
        import json

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output["status"] == "success"
        assert output["snapshot_id"] == 1
        assert output["metrics"]["files"]["total"] == 10
        assert output["metrics"]["complexity"]["average"] == 8.5
        assert output["metrics"]["complexity"]["maximum"] == 25
        assert output["metrics"]["health"]["average_score"] == 0.85

    def test_print_metrics_summary(self, capsys):
        """Test console output formatting."""
        # Create mock snapshot
        snapshot = ProjectSnapshot(
            snapshot_id=1,
            project_path="/test/path",
            timestamp=datetime(2024, 12, 11, 12, 0, 0),
            total_files=10,
            total_lines=1000,
            total_functions=50,
            total_classes=5,
            avg_complexity=8.5,
            max_complexity=25,
            total_complexity=425,
            total_smells=3,
            avg_health_score=0.85,
            grade_distribution={"A": 30, "B": 15, "C": 4, "D": 1, "F": 0},
            git_commit="abc123def456",
            git_branch="main",
            tool_version="1.0.0",
        )

        _print_metrics_summary(snapshot)

        # Check output contains key sections
        captured = capsys.readouterr()
        output = captured.out

        assert "Project Metrics Summary" in output
        assert "Files: 10" in output
        assert "Functions: 50" in output
        assert "Classes: 5" in output
        assert "Complexity Metrics" in output
        assert "Grade Distribution" in output
        assert "Health Score" in output

    def test_print_metrics_summary_with_smells(self, capsys):
        """Test console output includes code smells when present."""
        snapshot = ProjectSnapshot(
            snapshot_id=1,
            project_path="/test/path",
            timestamp=datetime(2024, 12, 11, 12, 0, 0),
            total_files=10,
            total_lines=1000,
            total_functions=50,
            total_classes=5,
            avg_complexity=8.5,
            max_complexity=25,
            total_complexity=425,
            total_smells=15,  # Has smells
            avg_health_score=0.65,
            grade_distribution={"A": 30, "B": 15, "C": 4, "D": 1, "F": 0},
        )

        _print_metrics_summary(snapshot)

        captured = capsys.readouterr()
        output = captured.out

        assert "Code Smells" in output
        assert "15 issues detected" in output

    def test_print_metrics_summary_attention_needed(self, capsys):
        """Test console output shows files needing attention."""
        snapshot = ProjectSnapshot(
            snapshot_id=1,
            project_path="/test/path",
            timestamp=datetime(2024, 12, 11, 12, 0, 0),
            total_files=10,
            total_lines=1000,
            total_functions=50,
            total_classes=5,
            avg_complexity=8.5,
            max_complexity=25,
            total_complexity=425,
            total_smells=0,
            avg_health_score=0.85,
            grade_distribution={
                "A": 25,
                "B": 15,
                "C": 5,
                "D": 3,
                "F": 2,
            },  # Has D and F
        )

        _print_metrics_summary(snapshot)

        captured = capsys.readouterr()
        output = captured.out

        assert "need attention" in output.lower()
        # 3 D + 2 F = 5 chunks
        assert "5 code chunks" in output


class TestStatusMetricsIntegration:
    """Integration tests for status --metrics with realistic data."""

    def test_metrics_with_code_smells_and_complexity(self, tmp_path):
        """Test metrics display with code smells and complexity grades."""
        # Create metrics database with realistic test data
        metrics_dir = tmp_path / ".mcp-vector-search"
        metrics_dir.mkdir()
        db_path = metrics_dir / "metrics.db"

        from mcp_vector_search.analysis.metrics import (
            ChunkMetrics,
            FileMetrics,
            ProjectMetrics,
        )
        from mcp_vector_search.analysis.storage.metrics_store import MetricsStore

        # Create test metrics with various complexity levels
        project_metrics = ProjectMetrics(project_root=str(tmp_path))

        # Add file with low complexity
        file1 = FileMetrics(file_path="simple.py")
        file1.total_lines = 50
        file1.code_lines = 40
        file1.function_count = 5
        chunk1 = ChunkMetrics(
            cognitive_complexity=3, cyclomatic_complexity=2, lines_of_code=10
        )
        file1.chunks = [chunk1]
        file1.compute_aggregates()

        # Add file with high complexity and smells
        file2 = FileMetrics(file_path="complex.py")
        file2.total_lines = 200
        file2.code_lines = 150
        file2.function_count = 10
        chunk2 = ChunkMetrics(
            cognitive_complexity=35,
            cyclomatic_complexity=20,
            lines_of_code=50,
            smells=["high_complexity", "too_many_parameters"],
        )
        file2.chunks = [chunk2]
        file2.compute_aggregates()

        project_metrics.files = {
            "simple.py": file1,
            "complex.py": file2,
        }
        project_metrics.compute_aggregates()

        # Save to database
        store = MetricsStore(db_path)
        store.save_project_snapshot(project_metrics)
        store.close()

        # Run status --metrics
        result = runner.invoke(
            status_app, ["--metrics", "--project-root", str(tmp_path)]
        )

        # Should succeed and show detailed metrics
        assert result.exit_code == 0
        assert "Project Metrics Summary" in result.stdout
        assert "Complexity Metrics" in result.stdout
        assert "Grade Distribution" in result.stdout
        assert "Health Score" in result.stdout
        # Should show code smells since we have 2
        assert "Code Smells" in result.stdout
