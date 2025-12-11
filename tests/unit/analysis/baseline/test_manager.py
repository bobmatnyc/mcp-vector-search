"""Unit tests for BaselineManager.

Tests cover:
- Baseline saving and loading
- Error handling (not found, exists, corrupted)
- Listing and deleting baselines
- Git information extraction
- JSON serialization/deserialization
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from mcp_vector_search.analysis.baseline.manager import (
    BaselineCorruptedError,
    BaselineExistsError,
    BaselineManager,
    BaselineNotFoundError,
)
from mcp_vector_search.analysis.metrics import ChunkMetrics, FileMetrics, ProjectMetrics


@pytest.fixture
def temp_storage(tmp_path: Path) -> Path:
    """Create temporary storage directory for tests."""
    storage_dir = tmp_path / "baselines"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


@pytest.fixture
def manager(temp_storage: Path) -> BaselineManager:
    """Create BaselineManager with temp storage."""
    return BaselineManager(storage_dir=temp_storage)


@pytest.fixture
def sample_metrics() -> ProjectMetrics:
    """Create sample ProjectMetrics for testing."""
    # Create some chunk metrics
    chunks = [
        ChunkMetrics(
            cognitive_complexity=5,
            cyclomatic_complexity=3,
            max_nesting_depth=2,
            parameter_count=3,
            lines_of_code=20,
            smells=[],
        ),
        ChunkMetrics(
            cognitive_complexity=15,
            cyclomatic_complexity=8,
            max_nesting_depth=4,
            parameter_count=5,
            lines_of_code=50,
            smells=["too_many_parameters"],
        ),
    ]

    # Create file metrics
    file_metrics = FileMetrics(
        file_path="src/example.py",
        total_lines=100,
        code_lines=70,
        comment_lines=20,
        blank_lines=10,
        function_count=2,
        class_count=1,
        method_count=2,
        chunks=chunks,
    )
    file_metrics.compute_aggregates()

    # Create project metrics
    metrics = ProjectMetrics(
        project_root="/path/to/project",
        analyzed_at=datetime.now(),
        files={"src/example.py": file_metrics},
    )
    metrics.compute_aggregates()

    return metrics


class TestBaselineManager:
    """Test suite for BaselineManager."""

    def test_init_creates_storage_dir(self, tmp_path: Path) -> None:
        """Test that manager creates storage directory on init."""
        storage_dir = tmp_path / "new_baselines"
        assert not storage_dir.exists()

        BaselineManager(storage_dir=storage_dir)

        assert storage_dir.exists()
        assert storage_dir.is_dir()

    def test_get_baseline_path_sanitizes_name(self, manager: BaselineManager) -> None:
        """Test that baseline names are sanitized for filesystem safety."""
        # Valid name
        path = manager.get_baseline_path("main-branch")
        assert path.name == "main-branch.json"

        # Name with special characters
        path = manager.get_baseline_path("feature/my-feature")
        assert "/" not in path.name
        assert path.name == "feature_my-feature.json"

        # Name with spaces
        path = manager.get_baseline_path("my baseline")
        assert " " not in path.name
        assert path.name == "my_baseline.json"

    def test_save_baseline_creates_file(
        self, manager: BaselineManager, sample_metrics: ProjectMetrics
    ) -> None:
        """Test saving baseline creates JSON file."""
        baseline_path = manager.save_baseline("test-baseline", sample_metrics)

        assert baseline_path.exists()
        assert baseline_path.suffix == ".json"
        assert baseline_path.parent == manager.storage_dir

    def test_save_baseline_includes_metadata(
        self, manager: BaselineManager, sample_metrics: ProjectMetrics
    ) -> None:
        """Test that saved baseline includes required metadata."""
        baseline_path = manager.save_baseline("test-baseline", sample_metrics)

        with baseline_path.open("r") as f:
            data = json.load(f)

        # Check required fields
        assert data["version"] == "1.0"
        assert data["baseline_name"] == "test-baseline"
        assert "created_at" in data
        assert "tool_version" in data
        assert "git_info" in data
        assert "project" in data
        assert "aggregate_metrics" in data
        assert "files" in data

    def test_save_baseline_raises_if_exists(
        self, manager: BaselineManager, sample_metrics: ProjectMetrics
    ) -> None:
        """Test that saving duplicate baseline raises error."""
        # Save first time
        manager.save_baseline("test-baseline", sample_metrics)

        # Try to save again
        with pytest.raises(BaselineExistsError, match="already exists"):
            manager.save_baseline("test-baseline", sample_metrics)

    def test_save_baseline_overwrites_with_flag(
        self, manager: BaselineManager, sample_metrics: ProjectMetrics
    ) -> None:
        """Test that overwrite flag allows replacing baseline."""
        # Save first time
        baseline_path = manager.save_baseline("test-baseline", sample_metrics)
        original_mtime = baseline_path.stat().st_mtime

        # Wait a bit to ensure different timestamp
        import time

        time.sleep(0.01)

        # Save again with overwrite
        new_path = manager.save_baseline(
            "test-baseline", sample_metrics, overwrite=True
        )

        assert new_path == baseline_path
        assert new_path.stat().st_mtime > original_mtime

    def test_load_baseline_restores_metrics(
        self, manager: BaselineManager, sample_metrics: ProjectMetrics
    ) -> None:
        """Test loading baseline restores ProjectMetrics."""
        # Save baseline
        manager.save_baseline("test-baseline", sample_metrics)

        # Load baseline
        loaded_metrics = manager.load_baseline("test-baseline")

        # Verify structure
        assert isinstance(loaded_metrics, ProjectMetrics)
        assert loaded_metrics.project_root == sample_metrics.project_root
        assert loaded_metrics.total_files == sample_metrics.total_files
        assert loaded_metrics.total_functions == sample_metrics.total_functions

        # Verify file metrics
        assert "src/example.py" in loaded_metrics.files
        loaded_file = loaded_metrics.files["src/example.py"]
        original_file = sample_metrics.files["src/example.py"]

        assert loaded_file.total_lines == original_file.total_lines
        assert loaded_file.function_count == original_file.function_count
        assert len(loaded_file.chunks) == len(original_file.chunks)

        # Verify chunk metrics
        for loaded_chunk, original_chunk in zip(
            loaded_file.chunks, original_file.chunks, strict=False
        ):
            assert (
                loaded_chunk.cognitive_complexity == original_chunk.cognitive_complexity
            )
            assert (
                loaded_chunk.cyclomatic_complexity
                == original_chunk.cyclomatic_complexity
            )
            assert loaded_chunk.smells == original_chunk.smells

    def test_load_baseline_raises_if_not_found(self, manager: BaselineManager) -> None:
        """Test loading nonexistent baseline raises error."""
        with pytest.raises(BaselineNotFoundError, match="not found"):
            manager.load_baseline("nonexistent-baseline")

    def test_load_baseline_raises_if_corrupted(
        self, manager: BaselineManager, temp_storage: Path
    ) -> None:
        """Test loading corrupted JSON raises error."""
        # Create corrupted baseline file
        baseline_path = temp_storage / "corrupted.json"
        baseline_path.write_text("{invalid json")

        with pytest.raises(BaselineCorruptedError, match="corrupted"):
            manager.load_baseline("corrupted")

    def test_load_baseline_raises_if_missing_fields(
        self, manager: BaselineManager, temp_storage: Path
    ) -> None:
        """Test loading baseline with missing fields raises error."""
        # Create baseline missing required fields
        baseline_path = temp_storage / "incomplete.json"
        baseline_path.write_text(json.dumps({"version": "1.0"}))

        with pytest.raises(BaselineCorruptedError, match="Missing required field"):
            manager.load_baseline("incomplete")

    def test_list_baselines_returns_metadata(
        self, manager: BaselineManager, sample_metrics: ProjectMetrics
    ) -> None:
        """Test listing baselines returns metadata."""
        # Save some baselines
        manager.save_baseline("baseline-1", sample_metrics)
        manager.save_baseline("baseline-2", sample_metrics)

        # List baselines
        baselines = manager.list_baselines()

        assert len(baselines) == 2
        assert any(b.baseline_name == "baseline-1" for b in baselines)
        assert any(b.baseline_name == "baseline-2" for b in baselines)

        # Check metadata fields
        baseline = baselines[0]
        assert baseline.baseline_name
        assert baseline.created_at
        assert baseline.tool_version
        assert baseline.file_count > 0

    def test_list_baselines_sorted_by_time(
        self, manager: BaselineManager, sample_metrics: ProjectMetrics
    ) -> None:
        """Test baselines are sorted by creation time (newest first)."""
        import time

        # Save baselines with delays
        manager.save_baseline("old", sample_metrics)
        time.sleep(0.01)
        manager.save_baseline("new", sample_metrics)

        baselines = manager.list_baselines()

        # Newest should be first
        assert baselines[0].baseline_name == "new"
        assert baselines[1].baseline_name == "old"

    def test_list_baselines_skips_corrupted(
        self,
        manager: BaselineManager,
        sample_metrics: ProjectMetrics,
        temp_storage: Path,
    ) -> None:
        """Test listing baselines skips corrupted files."""
        # Save valid baseline
        manager.save_baseline("valid", sample_metrics)

        # Create corrupted baseline
        corrupted_path = temp_storage / "corrupted.json"
        corrupted_path.write_text("{invalid json")

        # List should only include valid baseline
        baselines = manager.list_baselines()

        assert len(baselines) == 1
        assert baselines[0].baseline_name == "valid"

    def test_delete_baseline_removes_file(
        self, manager: BaselineManager, sample_metrics: ProjectMetrics
    ) -> None:
        """Test deleting baseline removes file."""
        # Save baseline
        baseline_path = manager.save_baseline("test-baseline", sample_metrics)
        assert baseline_path.exists()

        # Delete baseline
        manager.delete_baseline("test-baseline")

        assert not baseline_path.exists()

    def test_delete_baseline_raises_if_not_found(
        self, manager: BaselineManager
    ) -> None:
        """Test deleting nonexistent baseline raises error."""
        with pytest.raises(BaselineNotFoundError, match="not found"):
            manager.delete_baseline("nonexistent-baseline")

    def test_git_info_extraction(
        self, manager: BaselineManager, tmp_path: Path
    ) -> None:
        """Test extracting git information from repository."""
        # Create a git repo
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()

        import subprocess

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True)
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo_dir,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repo_dir,
            capture_output=True,
        )

        # Create a commit
        test_file = repo_dir / "test.txt"
        test_file.write_text("test")
        subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"], cwd=repo_dir, capture_output=True
        )

        # Extract git info
        git_info = manager._get_git_info(repo_dir)

        assert git_info.commit is not None
        assert len(git_info.commit) == 40  # SHA-1 hash
        assert git_info.branch is not None

    def test_git_info_handles_non_git_repo(
        self, manager: BaselineManager, tmp_path: Path
    ) -> None:
        """Test git info extraction handles non-git repositories gracefully."""
        non_git_dir = tmp_path / "not_a_repo"
        non_git_dir.mkdir()

        git_info = manager._get_git_info(non_git_dir)

        # Should return GitInfo with None values
        assert git_info.commit is None
        assert git_info.branch is None
        assert git_info.remote is None

    def test_serialize_deserialize_roundtrip(
        self, manager: BaselineManager, sample_metrics: ProjectMetrics
    ) -> None:
        """Test that serialize -> deserialize is lossless."""
        # Save and load
        manager.save_baseline("roundtrip", sample_metrics)
        loaded = manager.load_baseline("roundtrip")

        # Compare key metrics
        assert loaded.total_files == sample_metrics.total_files
        assert loaded.total_functions == sample_metrics.total_functions
        assert loaded.avg_file_complexity == sample_metrics.avg_file_complexity

        # Compare file-level metrics
        for file_path, original_file in sample_metrics.files.items():
            assert file_path in loaded.files
            loaded_file = loaded.files[file_path]

            assert loaded_file.total_complexity == original_file.total_complexity
            assert loaded_file.avg_complexity == original_file.avg_complexity
            assert loaded_file.max_complexity == original_file.max_complexity

    def test_baseline_with_description(
        self, manager: BaselineManager, sample_metrics: ProjectMetrics
    ) -> None:
        """Test saving baseline with custom description."""
        description = "Baseline after PR #123 merge"
        baseline_path = manager.save_baseline(
            "test-baseline", sample_metrics, description=description
        )

        with baseline_path.open("r") as f:
            data = json.load(f)

        assert data["description"] == description

    def test_multiple_files_baseline(
        self, manager: BaselineManager, sample_metrics: ProjectMetrics
    ) -> None:
        """Test baseline with multiple files."""
        # Add more files to metrics
        for i in range(5):
            file_metrics = FileMetrics(
                file_path=f"src/file_{i}.py",
                total_lines=50,
                code_lines=40,
                comment_lines=5,
                blank_lines=5,
                function_count=3,
                class_count=1,
                method_count=3,
                chunks=[
                    ChunkMetrics(
                        cognitive_complexity=10,
                        cyclomatic_complexity=5,
                        max_nesting_depth=3,
                        parameter_count=2,
                        lines_of_code=15,
                    )
                ],
            )
            file_metrics.compute_aggregates()
            sample_metrics.files[f"src/file_{i}.py"] = file_metrics

        sample_metrics.compute_aggregates()

        # Save and load
        manager.save_baseline("multi-file", sample_metrics)
        loaded = manager.load_baseline("multi-file")

        assert len(loaded.files) == len(sample_metrics.files)
        assert loaded.total_files == sample_metrics.total_files

    def test_empty_project_baseline(self, manager: BaselineManager) -> None:
        """Test baseline with no files (edge case)."""
        empty_metrics = ProjectMetrics(project_root="/empty/project")
        empty_metrics.compute_aggregates()

        # Save and load
        manager.save_baseline("empty", empty_metrics)
        loaded = manager.load_baseline("empty")

        assert loaded.total_files == 0
        assert loaded.total_functions == 0
        assert len(loaded.files) == 0
