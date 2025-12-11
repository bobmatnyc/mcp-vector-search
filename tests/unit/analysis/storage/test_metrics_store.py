"""Unit tests for MetricsStore.

Test Coverage:
    - Database initialization and schema creation
    - Saving and retrieving project snapshots
    - Saving and retrieving file metrics
    - Historical queries (file and project)
    - Trend analysis
    - Error handling (duplicates, locked database)
    - Context manager usage
    - Git metadata extraction

Design Principles:
    - Isolated tests: Each test uses temporary database
    - No external dependencies: Mock git commands
    - Fast execution: In-memory SQLite where possible
    - Comprehensive coverage: Test happy path + error cases
"""

import json
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_vector_search.analysis.metrics import ChunkMetrics, FileMetrics, ProjectMetrics
from mcp_vector_search.analysis.storage import (
    DuplicateEntryError,
    GitInfo,
    MetricsStore,
    MetricsStoreError,
)
from mcp_vector_search.analysis.storage.schema import SCHEMA_VERSION


class TestMetricsStoreInitialization:
    """Tests for MetricsStore initialization and schema setup."""

    def test_init_with_default_path(self) -> None:
        """Test initialization with default database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Override home directory for test
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                store = MetricsStore()

                expected_path = (
                    Path(tmpdir) / ".mcp-vector-search" / "metrics.db"
                ).resolve()
                assert store.db_path == expected_path
                assert store.db_path.exists()

                store.close()

    def test_init_with_custom_path(self) -> None:
        """Test initialization with custom database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "custom_metrics.db"

            store = MetricsStore(db_path=db_path)

            assert store.db_path == db_path.resolve()
            assert db_path.exists()

            store.close()

    def test_schema_initialization(self) -> None:
        """Test that database schema is properly initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            store = MetricsStore(db_path=db_path)

            # Check schema version table exists
            cursor = store.conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )
            assert cursor.fetchone() is not None

            # Check schema version
            cursor.execute("SELECT version FROM schema_version")
            version = cursor.fetchone()[0]
            assert version == SCHEMA_VERSION

            # Check required tables exist
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall()]

            assert "schema_version" in tables
            assert "project_snapshots" in tables
            assert "file_metrics" in tables
            assert "code_smells" in tables

            store.close()

    def test_foreign_keys_enabled(self) -> None:
        """Test that foreign key constraints are enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            store = MetricsStore(db_path=db_path)

            cursor = store.conn.cursor()
            cursor.execute("PRAGMA foreign_keys")
            result = cursor.fetchone()[0]

            assert result == 1  # Foreign keys enabled

            store.close()

    def test_context_manager(self) -> None:
        """Test using MetricsStore as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            with MetricsStore(db_path=db_path) as store:
                assert store.conn is not None

            # Connection should be closed after context exit
            with pytest.raises(sqlite3.ProgrammingError):
                store.conn.execute("SELECT 1")


class TestProjectSnapshotSaving:
    """Tests for saving project snapshots."""

    @pytest.fixture
    def store(self) -> MetricsStore:
        """Create temporary metrics store for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = MetricsStore(db_path=db_path)
            yield store
            store.close()

    @pytest.fixture
    def sample_metrics(self) -> ProjectMetrics:
        """Create sample ProjectMetrics for testing."""
        # Create sample file metrics
        chunk1 = ChunkMetrics(
            cognitive_complexity=5,
            cyclomatic_complexity=3,
            max_nesting_depth=2,
            parameter_count=2,
            lines_of_code=20,
            smells=["too_many_parameters"],
        )

        chunk2 = ChunkMetrics(
            cognitive_complexity=15,
            cyclomatic_complexity=8,
            max_nesting_depth=4,
            parameter_count=5,
            lines_of_code=50,
            smells=["high_complexity", "deep_nesting"],
        )

        file1 = FileMetrics(
            file_path="src/module.py",
            total_lines=100,
            code_lines=70,
            comment_lines=20,
            blank_lines=10,
            function_count=2,
            class_count=1,
            method_count=3,
            chunks=[chunk1, chunk2],
        )
        file1.compute_aggregates()

        # Create project metrics
        metrics = ProjectMetrics(
            project_root="/path/to/project",
            analyzed_at=datetime.now(),
            files={"src/module.py": file1},
        )
        metrics.compute_aggregates()

        return metrics

    def test_save_project_snapshot_basic(
        self, store: MetricsStore, sample_metrics: ProjectMetrics
    ) -> None:
        """Test saving a basic project snapshot."""
        git_info = GitInfo(
            commit="abc123def456",
            branch="main",
            remote="origin",
        )

        snapshot_id = store.save_project_snapshot(sample_metrics, git_info)

        assert snapshot_id > 0

        # Verify snapshot was saved correctly
        cursor = store.conn.cursor()
        cursor.execute("SELECT * FROM project_snapshots WHERE id = ?", (snapshot_id,))
        row = cursor.fetchone()

        assert row is not None
        assert row["project_path"] == "/path/to/project"
        assert row["total_files"] == 1
        assert row["total_functions"] == 2
        assert row["git_commit"] == "abc123def456"
        assert row["git_branch"] == "main"

    def test_save_project_snapshot_without_git(
        self, store: MetricsStore, sample_metrics: ProjectMetrics
    ) -> None:
        """Test saving snapshot without git metadata."""
        with patch.object(store, "_get_git_info", return_value=GitInfo()):
            snapshot_id = store.save_project_snapshot(sample_metrics)

            assert snapshot_id > 0

            cursor = store.conn.cursor()
            cursor.execute(
                "SELECT * FROM project_snapshots WHERE id = ?", (snapshot_id,)
            )
            row = cursor.fetchone()

            assert row["git_commit"] is None
            assert row["git_branch"] is None

    def test_save_duplicate_snapshot_raises_error(
        self, store: MetricsStore, sample_metrics: ProjectMetrics
    ) -> None:
        """Test that saving duplicate snapshot raises DuplicateEntryError."""
        git_info = GitInfo(commit="abc123", branch="main")

        # Save first snapshot
        store.save_project_snapshot(sample_metrics, git_info)

        # Attempt to save duplicate (same project + timestamp)
        with pytest.raises(DuplicateEntryError):
            store.save_project_snapshot(sample_metrics, git_info)

    def test_save_project_snapshot_grade_distribution(
        self, store: MetricsStore, sample_metrics: ProjectMetrics
    ) -> None:
        """Test that grade distribution is correctly computed and saved."""
        snapshot_id = store.save_project_snapshot(sample_metrics)

        cursor = store.conn.cursor()
        cursor.execute(
            "SELECT grade_distribution FROM project_snapshots WHERE id = ?",
            (snapshot_id,),
        )
        row = cursor.fetchone()

        grade_dist = json.loads(row["grade_distribution"])

        # chunk1 has complexity 5 (A), chunk2 has complexity 15 (C)
        assert grade_dist["A"] == 1
        assert grade_dist["C"] == 1
        assert grade_dist["B"] == 0


class TestFileMetricsSaving:
    """Tests for saving file metrics."""

    @pytest.fixture
    def store(self) -> MetricsStore:
        """Create temporary metrics store for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = MetricsStore(db_path=db_path)
            yield store
            store.close()

    @pytest.fixture
    def snapshot_id(self, store: MetricsStore) -> int:
        """Create a project snapshot for testing file metrics."""
        metrics = ProjectMetrics(
            project_root="/test/project",
            analyzed_at=datetime.now(),
        )

        return store.save_project_snapshot(metrics)

    def test_save_file_metrics_basic(
        self, store: MetricsStore, snapshot_id: int
    ) -> None:
        """Test saving basic file metrics."""
        chunk = ChunkMetrics(
            cognitive_complexity=10,
            cyclomatic_complexity=5,
            lines_of_code=30,
        )

        file_metrics = FileMetrics(
            file_path="src/test.py",
            total_lines=50,
            code_lines=40,
            comment_lines=5,
            blank_lines=5,
            function_count=2,
            chunks=[chunk],
        )
        file_metrics.compute_aggregates()

        file_id = store.save_file_metrics(file_metrics, snapshot_id)

        assert file_id > 0

        # Verify file metrics were saved
        cursor = store.conn.cursor()
        cursor.execute("SELECT * FROM file_metrics WHERE id = ?", (file_id,))
        row = cursor.fetchone()

        assert row is not None
        assert row["file_path"] == "src/test.py"
        assert row["total_lines"] == 50
        assert row["function_count"] == 2
        assert row["avg_complexity"] == 10.0

    def test_save_file_metrics_with_smells(
        self, store: MetricsStore, snapshot_id: int
    ) -> None:
        """Test saving file metrics with code smells."""
        chunk = ChunkMetrics(
            cognitive_complexity=25,
            cyclomatic_complexity=12,
            lines_of_code=100,
            smells=["high_complexity", "too_long"],
        )

        file_metrics = FileMetrics(
            file_path="src/complex.py",
            total_lines=100,
            chunks=[chunk],
        )
        file_metrics.compute_aggregates()

        file_id = store.save_file_metrics(file_metrics, snapshot_id)

        cursor = store.conn.cursor()
        cursor.execute("SELECT * FROM file_metrics WHERE id = ?", (file_id,))
        row = cursor.fetchone()

        assert row["smell_count"] == 2
        assert row["complexity_grade"] == "D"  # 25 is grade D

    def test_save_duplicate_file_metrics_raises_error(
        self, store: MetricsStore, snapshot_id: int
    ) -> None:
        """Test that duplicate file metrics raise error."""
        file_metrics = FileMetrics(
            file_path="src/test.py",
            total_lines=50,
        )

        # Save first time
        store.save_file_metrics(file_metrics, snapshot_id)

        # Attempt duplicate save
        with pytest.raises(DuplicateEntryError):
            store.save_file_metrics(file_metrics, snapshot_id)


class TestCompleteSnapshot:
    """Tests for save_complete_snapshot (transaction)."""

    @pytest.fixture
    def store(self) -> MetricsStore:
        """Create temporary metrics store for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = MetricsStore(db_path=db_path)
            yield store
            store.close()

    def test_save_complete_snapshot_success(self, store: MetricsStore) -> None:
        """Test saving complete snapshot with multiple files."""
        # Create metrics with multiple files
        file1 = FileMetrics(
            file_path="src/file1.py",
            total_lines=100,
            function_count=3,
            chunks=[ChunkMetrics(cognitive_complexity=5, lines_of_code=20)],
        )
        file1.compute_aggregates()

        file2 = FileMetrics(
            file_path="src/file2.py",
            total_lines=150,
            function_count=5,
            chunks=[ChunkMetrics(cognitive_complexity=8, lines_of_code=30)],
        )
        file2.compute_aggregates()

        metrics = ProjectMetrics(
            project_root="/test/project",
            analyzed_at=datetime.now(),
            files={"src/file1.py": file1, "src/file2.py": file2},
        )
        metrics.compute_aggregates()

        snapshot_id = store.save_complete_snapshot(metrics)

        assert snapshot_id > 0

        # Verify project snapshot
        cursor = store.conn.cursor()
        cursor.execute("SELECT * FROM project_snapshots WHERE id = ?", (snapshot_id,))
        project_row = cursor.fetchone()

        assert project_row["total_files"] == 2
        assert project_row["total_functions"] == 8

        # Verify file metrics
        cursor.execute(
            "SELECT COUNT(*) FROM file_metrics WHERE project_id = ?", (snapshot_id,)
        )
        file_count = cursor.fetchone()[0]

        assert file_count == 2

    def test_save_complete_snapshot_rollback_on_error(
        self, store: MetricsStore
    ) -> None:
        """Test that transaction rolls back on error."""
        # Create invalid metrics that will fail
        metrics = ProjectMetrics(
            project_root="/test/project",
            analyzed_at=datetime.now(),
        )

        # Save once successfully
        store.save_complete_snapshot(metrics)

        # Attempt duplicate save (should rollback)
        with pytest.raises(MetricsStoreError):
            store.save_complete_snapshot(metrics)

        # Verify only one snapshot exists
        cursor = store.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM project_snapshots")
        count = cursor.fetchone()[0]

        assert count == 1


class TestHistoryQueries:
    """Tests for historical data queries."""

    @pytest.fixture
    def store_with_history(self) -> MetricsStore:
        """Create store with historical data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = MetricsStore(db_path=db_path)

            # Create 5 snapshots over time
            for i in range(5):
                timestamp = datetime.now() - timedelta(days=i)

                file_metrics = FileMetrics(
                    file_path="src/test.py",
                    total_lines=100 + i * 10,
                    function_count=5 + i,
                    chunks=[
                        ChunkMetrics(
                            cognitive_complexity=10 + i * 2,
                            lines_of_code=20,
                        )
                    ],
                )
                file_metrics.compute_aggregates()

                metrics = ProjectMetrics(
                    project_root="/test/project",
                    analyzed_at=timestamp,
                    files={"src/test.py": file_metrics},
                )
                metrics.compute_aggregates()

                store.save_complete_snapshot(metrics)

            yield store
            store.close()

    def test_get_project_history(self, store_with_history: MetricsStore) -> None:
        """Test retrieving project history."""
        history = store_with_history.get_project_history("/test/project", limit=10)

        assert len(history) == 5

        # Verify ordered by timestamp (newest first)
        for i in range(len(history) - 1):
            assert history[i].timestamp > history[i + 1].timestamp

        # Verify snapshot data
        assert history[0].total_files == 1
        assert all(s.project_path == "/test/project" for s in history)

    def test_get_project_history_with_limit(
        self, store_with_history: MetricsStore
    ) -> None:
        """Test limiting project history results."""
        history = store_with_history.get_project_history("/test/project", limit=3)

        assert len(history) == 3

    def test_get_file_history(self, store_with_history: MetricsStore) -> None:
        """Test retrieving file history."""
        history = store_with_history.get_file_history("src/test.py", limit=10)

        assert len(history) == 5

        # Verify file metrics
        assert all(fm.file_path == "src/test.py" for fm in history)
        # Note: In fixture, days=i means i=0 is newest (today), i=4 is oldest (4 days ago)
        # Newest (i=0) has function_count=5, oldest (i=4) has function_count=9
        # Query returns DESC order, so history[0] is newest (lowest count)
        assert (
            history[0].function_count <= history[-1].function_count
        )  # Newest has lower count


class TestTrendAnalysis:
    """Tests for trend analysis."""

    @pytest.fixture
    def store_with_trends(self) -> MetricsStore:
        """Create store with trend data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = MetricsStore(db_path=db_path)

            # Create snapshots with increasing complexity over 10 days
            for i in range(10):
                timestamp = datetime.now() - timedelta(days=9 - i)

                file_metrics = FileMetrics(
                    file_path="src/test.py",
                    total_lines=100,
                    chunks=[
                        ChunkMetrics(
                            cognitive_complexity=10 + i * 2,  # Increasing complexity
                            lines_of_code=20,
                        )
                    ],
                )
                file_metrics.compute_aggregates()

                metrics = ProjectMetrics(
                    project_root="/test/project",
                    analyzed_at=timestamp,
                    files={"src/test.py": file_metrics},
                )
                metrics.compute_aggregates()

                store.save_complete_snapshot(metrics)

            yield store
            store.close()

    def test_get_trends_basic(self, store_with_trends: MetricsStore) -> None:
        """Test basic trend analysis."""
        trends = store_with_trends.get_trends("/test/project", days=30)

        assert trends.project_path == "/test/project"
        assert trends.period_days == 30
        assert len(trends.snapshots) == 10

        # Verify trend data
        assert len(trends.complexity_trend) == 10
        assert len(trends.smell_trend) == 10
        assert len(trends.health_trend) == 10

    def test_get_trends_change_rate(self, store_with_trends: MetricsStore) -> None:
        """Test that change rate is computed correctly."""
        trends = store_with_trends.get_trends("/test/project", days=30)

        # Complexity increases by 2 per day
        assert trends.change_rate > 0  # Increasing
        assert not trends.improving  # Degrading

        # Change rate should be approximately 2.0 per day
        assert 1.5 < trends.change_rate < 2.5

    def test_get_trends_improving_code(self) -> None:
        """Test trend analysis for improving code quality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = MetricsStore(db_path=db_path)

            # Create snapshots with DECREASING complexity
            for i in range(10):
                timestamp = datetime.now() - timedelta(days=9 - i)

                file_metrics = FileMetrics(
                    file_path="src/test.py",
                    total_lines=100,
                    chunks=[
                        ChunkMetrics(
                            cognitive_complexity=30 - i * 2,  # Decreasing complexity
                            lines_of_code=20,
                        )
                    ],
                )
                file_metrics.compute_aggregates()

                metrics = ProjectMetrics(
                    project_root="/test/project",
                    analyzed_at=timestamp,
                    files={"src/test.py": file_metrics},
                )
                metrics.compute_aggregates()

                store.save_complete_snapshot(metrics)

            trends = store.get_trends("/test/project", days=30)

            assert trends.change_rate < 0  # Decreasing
            assert trends.improving  # Improving code quality

            store.close()


class TestGitMetadata:
    """Tests for git metadata extraction."""

    def test_get_git_info_success(self) -> None:
        """Test successful git info extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = MetricsStore(db_path=db_path)

            project_root = Path(tmpdir) / "project"
            project_root.mkdir()

            # Mock subprocess.run to simulate git commands
            mock_results = [
                MagicMock(stdout="abc123def456\n"),  # commit hash
                MagicMock(stdout="main\n"),  # branch
                MagicMock(stdout="origin\n"),  # remote
            ]

            with patch("subprocess.run", side_effect=mock_results):
                git_info = store._get_git_info(project_root)

                assert git_info.commit == "abc123def456"
                assert git_info.branch == "main"
                assert git_info.remote == "origin"

            store.close()

    def test_get_git_info_not_a_repo(self) -> None:
        """Test git info extraction for non-git directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = MetricsStore(db_path=db_path)

            project_root = Path(tmpdir) / "project"
            project_root.mkdir()

            # Mock subprocess to raise CalledProcessError
            with patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "git"),
            ):
                git_info = store._get_git_info(project_root)

                # Should return empty GitInfo, not raise exception
                assert git_info.commit is None
                assert git_info.branch is None
                assert git_info.remote is None

            store.close()


class TestErrorHandling:
    """Tests for error handling."""

    def test_database_locked_error(self) -> None:
        """Test handling of database lock errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # Create first connection
            store1 = MetricsStore(db_path=db_path)

            # Simulate lock by starting transaction
            store1.conn.execute("BEGIN EXCLUSIVE")

            # Create second connection with short timeout
            conn2 = sqlite3.connect(str(db_path), timeout=0.1)

            # Attempt exclusive lock - should raise OperationalError
            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                conn2.execute("BEGIN EXCLUSIVE")

            conn2.close()
            store1.conn.rollback()
            store1.close()

    def test_invalid_database_path(self) -> None:
        """Test error when database path is invalid."""
        invalid_path = Path("/invalid/path/that/does/not/exist/db.sqlite")

        with pytest.raises(MetricsStoreError):
            MetricsStore(db_path=invalid_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
