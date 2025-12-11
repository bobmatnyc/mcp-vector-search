"""Baseline storage and retrieval manager.

This module provides the BaselineManager class for persisting and loading
metric snapshots (baselines) to/from JSON files.

Design Decisions:
    - JSON format for human readability and simplicity (Phase 2)
    - Storage location: ~/.mcp-vector-search/baselines/ by default
    - Includes git metadata (commit, branch) for traceability
    - Includes tool version for compatibility validation
    - Atomic writes with temp file + rename for data integrity

Storage Format:
    Baselines are stored as JSON files with structure:
    {
        "version": "1.0",
        "baseline_name": "main-branch",
        "created_at": "2025-12-11T15:30:00Z",
        "tool_version": "v0.18.0",
        "git_info": {"commit": "abc123", "branch": "main"},
        "project": {"path": "/path/to/project", "file_count": 42},
        "aggregate_metrics": {...},
        "files": {...}
    }

Error Handling:
    - BaselineNotFoundError: Baseline doesn't exist
    - BaselineExistsError: Baseline already exists (use overwrite=True)
    - BaselineCorruptedError: JSON parsing failed or invalid structure
    - OSError: Filesystem permission issues (propagated with clear message)

Performance:
    - Save: O(n) where n=files, typically 50-100ms for 100 files
    - Load: O(n), typically 20-50ms for 100 files
    - List: O(k) where k=number of baselines, <10ms typical
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from ...utils.version import get_version_string
from ..metrics import ProjectMetrics


class BaselineError(Exception):
    """Base exception for baseline-related errors."""

    pass


class BaselineNotFoundError(BaselineError):
    """Baseline file does not exist."""

    pass


class BaselineExistsError(BaselineError):
    """Baseline file already exists."""

    pass


class BaselineCorruptedError(BaselineError):
    """Baseline file is corrupted or invalid."""

    pass


@dataclass
class GitInfo:
    """Git repository information for baseline traceability.

    Attributes:
        commit: Git commit hash (full SHA-1)
        branch: Current branch name (None if detached HEAD)
        remote: Remote repository name (e.g., "origin")
    """

    commit: str | None = None
    branch: str | None = None
    remote: str | None = None


@dataclass
class BaselineMetadata:
    """Metadata for a baseline snapshot.

    Attributes:
        baseline_name: Human-readable identifier
        created_at: ISO timestamp when baseline was created
        tool_version: Version of mcp-vector-search used
        git_info: Git repository information
        project_path: Absolute path to project root
        file_count: Number of files in baseline
        function_count: Total number of functions analyzed
    """

    baseline_name: str
    created_at: str
    tool_version: str
    git_info: GitInfo
    project_path: str
    file_count: int
    function_count: int


class BaselineManager:
    """Manage baseline snapshot storage and retrieval.

    This class handles persisting ProjectMetrics to JSON files and loading
    them back for comparison. Baselines are stored in a user-specific
    directory for easy access across projects.

    Storage Strategy:
        - Primary: ~/.mcp-vector-search/baselines/
        - File naming: {baseline_name}.json
        - Atomic writes: temp file + rename

    Example:
        >>> manager = BaselineManager()
        >>> metrics = ProjectMetrics(project_root="/path/to/project")
        >>> manager.save_baseline("main-branch", metrics)
        >>> baseline = manager.load_baseline("main-branch")
        >>> print(f"Baseline has {baseline.total_files} files")
    """

    BASELINE_VERSION = "1.0"

    def __init__(self, storage_dir: Path | None = None):
        """Initialize baseline manager.

        Args:
            storage_dir: Optional custom storage directory.
                        Defaults to ~/.mcp-vector-search/baselines/
        """
        if storage_dir is None:
            # Default storage location
            storage_dir = Path.home() / ".mcp-vector-search" / "baselines"

        self.storage_dir = storage_dir.resolve()

        # Ensure storage directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Baseline storage directory: {self.storage_dir}")

    def get_baseline_path(self, baseline_name: str) -> Path:
        """Get path to baseline file.

        Args:
            baseline_name: Baseline identifier

        Returns:
            Path to baseline JSON file
        """
        # Sanitize baseline name (alphanumeric + hyphens/underscores)
        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in baseline_name
        )
        return self.storage_dir / f"{safe_name}.json"

    def save_baseline(
        self,
        baseline_name: str,
        metrics: ProjectMetrics,
        overwrite: bool = False,
        description: str | None = None,
    ) -> Path:
        """Save metrics as a baseline snapshot.

        Args:
            baseline_name: Human-readable identifier (e.g., "main-branch", "v1.2.0")
            metrics: ProjectMetrics to save
            overwrite: Allow overwriting existing baseline (default: False)
            description: Optional description for baseline

        Returns:
            Path to saved baseline file

        Raises:
            BaselineExistsError: If baseline exists and overwrite=False
            OSError: If filesystem write fails

        Performance: O(n) where n is number of files, typically 50-100ms

        Example:
            >>> manager = BaselineManager()
            >>> metrics = ProjectMetrics(project_root="/path/to/project")
            >>> path = manager.save_baseline("main-branch", metrics)
            >>> print(f"Saved to {path}")
        """
        baseline_path = self.get_baseline_path(baseline_name)

        # Check if baseline exists
        if baseline_path.exists() and not overwrite:
            raise BaselineExistsError(
                f"Baseline '{baseline_name}' already exists at {baseline_path}. "
                f"Use overwrite=True to replace it."
            )

        # Collect git information
        git_info = self._get_git_info(Path(metrics.project_root))

        # Build baseline data structure
        baseline_data = {
            "version": self.BASELINE_VERSION,
            "baseline_name": baseline_name,
            "created_at": datetime.now().isoformat(),
            "tool_version": get_version_string(include_build=True),
            "description": description,
            "git_info": asdict(git_info),
            "project": {
                "path": metrics.project_root,
                "file_count": metrics.total_files,
                "function_count": metrics.total_functions,
                "class_count": metrics.total_classes,
            },
            # Serialize ProjectMetrics
            "aggregate_metrics": self._serialize_aggregate_metrics(metrics),
            "files": self._serialize_files(metrics),
        }

        # Atomic write: write to temp file, then rename
        temp_path = baseline_path.with_suffix(".tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as f:
                json.dump(baseline_data, f, indent=2, ensure_ascii=False)

            # Atomic rename (POSIX guarantees atomicity)
            temp_path.replace(baseline_path)

            file_size = baseline_path.stat().st_size
            logger.info(
                f"Saved baseline '{baseline_name}' to {baseline_path} "
                f"({file_size // 1024} KB)"
            )

            return baseline_path

        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            logger.error(f"Failed to save baseline: {e}")
            raise

    def load_baseline(self, baseline_name: str) -> ProjectMetrics:
        """Load baseline from storage.

        Args:
            baseline_name: Baseline identifier

        Returns:
            ProjectMetrics restored from baseline

        Raises:
            BaselineNotFoundError: If baseline doesn't exist
            BaselineCorruptedError: If JSON is invalid or missing required fields

        Performance: O(n) where n is number of files, typically 20-50ms

        Example:
            >>> manager = BaselineManager()
            >>> baseline = manager.load_baseline("main-branch")
            >>> print(f"Baseline from {baseline.analyzed_at}")
        """
        baseline_path = self.get_baseline_path(baseline_name)

        if not baseline_path.exists():
            # Provide helpful error with available baselines
            available = self.list_baselines()
            available_str = ", ".join(b.baseline_name for b in available[:5])
            raise BaselineNotFoundError(
                f"Baseline '{baseline_name}' not found at {baseline_path}. "
                f"Available baselines: {available_str or 'none'}"
            )

        try:
            with baseline_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            # Validate baseline structure
            self._validate_baseline(data)

            # Deserialize back to ProjectMetrics
            metrics = self._deserialize_project_metrics(data)

            logger.info(
                f"Loaded baseline '{baseline_name}' "
                f"({metrics.total_files} files, {metrics.total_functions} functions)"
            )

            return metrics

        except json.JSONDecodeError as e:
            logger.error(f"Baseline file is corrupted: {e}")
            raise BaselineCorruptedError(
                f"Baseline '{baseline_name}' is corrupted: {e}"
            )
        except KeyError as e:
            logger.error(f"Baseline missing required field: {e}")
            raise BaselineCorruptedError(
                f"Baseline '{baseline_name}' is missing required field: {e}"
            )

    def list_baselines(self) -> list[BaselineMetadata]:
        """List all available baselines.

        Returns:
            List of baseline metadata sorted by creation time (newest first)

        Performance: O(k) where k is number of baselines, typically <10ms

        Example:
            >>> manager = BaselineManager()
            >>> baselines = manager.list_baselines()
            >>> for baseline in baselines:
            ...     print(f"{baseline.baseline_name}: {baseline.file_count} files")
        """
        baselines = []

        # Scan storage directory for .json files
        for baseline_path in self.storage_dir.glob("*.json"):
            try:
                with baseline_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                # Extract metadata
                metadata = BaselineMetadata(
                    baseline_name=data.get("baseline_name", baseline_path.stem),
                    created_at=data.get("created_at", "unknown"),
                    tool_version=data.get("tool_version", "unknown"),
                    git_info=GitInfo(**data.get("git_info", {})),
                    project_path=data.get("project", {}).get("path", "unknown"),
                    file_count=data.get("project", {}).get("file_count", 0),
                    function_count=data.get("project", {}).get("function_count", 0),
                )

                baselines.append(metadata)

            except (json.JSONDecodeError, KeyError) as e:
                # Skip corrupted baselines
                logger.warning(f"Skipping corrupted baseline {baseline_path}: {e}")
                continue

        # Sort by creation time (newest first)
        baselines.sort(key=lambda b: b.created_at, reverse=True)

        logger.debug(f"Found {len(baselines)} baselines")
        return baselines

    def delete_baseline(self, baseline_name: str) -> None:
        """Delete a baseline.

        Args:
            baseline_name: Baseline identifier

        Raises:
            BaselineNotFoundError: If baseline doesn't exist

        Example:
            >>> manager = BaselineManager()
            >>> manager.delete_baseline("old-baseline")
        """
        baseline_path = self.get_baseline_path(baseline_name)

        if not baseline_path.exists():
            raise BaselineNotFoundError(
                f"Baseline '{baseline_name}' not found at {baseline_path}"
            )

        baseline_path.unlink()
        logger.info(f"Deleted baseline '{baseline_name}' from {baseline_path}")

    def _get_git_info(self, project_root: Path) -> GitInfo:
        """Extract git information from project repository.

        Args:
            project_root: Project root directory

        Returns:
            GitInfo with commit, branch, and remote (if available)

        Note: Does not raise exceptions. Returns GitInfo with None values if git unavailable.
        """
        git_info = GitInfo()

        try:
            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            git_info.commit = result.stdout.strip()

            # Get branch name
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            branch = result.stdout.strip()
            git_info.branch = branch if branch != "HEAD" else None

            # Get remote name (if exists)
            result = subprocess.run(
                ["git", "remote"],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            remotes = result.stdout.strip().split("\n")
            git_info.remote = remotes[0] if remotes and remotes[0] else None

        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            # Git not available or not a git repo
            logger.debug("Git information unavailable")

        return git_info

    def _serialize_aggregate_metrics(self, metrics: ProjectMetrics) -> dict[str, Any]:
        """Serialize project-level aggregate metrics.

        Args:
            metrics: ProjectMetrics to serialize

        Returns:
            Dictionary of aggregate metrics
        """
        # Compute grade distribution
        grade_dist = dict.fromkeys(["A", "B", "C", "D", "F"], 0)
        for file_metrics in metrics.files.values():
            for chunk in file_metrics.chunks:
                grade_dist[chunk.complexity_grade] += 1

        # Collect all complexity values for statistics
        all_cc = [
            chunk.cognitive_complexity
            for file_metrics in metrics.files.values()
            for chunk in file_metrics.chunks
        ]

        all_cyc = [
            chunk.cyclomatic_complexity
            for file_metrics in metrics.files.values()
            for chunk in file_metrics.chunks
        ]

        all_nesting = [
            chunk.max_nesting_depth
            for file_metrics in metrics.files.values()
            for chunk in file_metrics.chunks
        ]

        all_params = [
            chunk.parameter_count
            for file_metrics in metrics.files.values()
            for chunk in file_metrics.chunks
        ]

        return {
            "cognitive_complexity": {
                "sum": sum(all_cc),
                "avg": sum(all_cc) / len(all_cc) if all_cc else 0.0,
                "max": max(all_cc) if all_cc else 0,
                "grade_distribution": grade_dist,
            },
            "cyclomatic_complexity": {
                "sum": sum(all_cyc),
                "avg": sum(all_cyc) / len(all_cyc) if all_cyc else 0.0,
                "max": max(all_cyc) if all_cyc else 0,
            },
            "nesting_depth": {
                "max": max(all_nesting) if all_nesting else 0,
                "avg": sum(all_nesting) / len(all_nesting) if all_nesting else 0.0,
            },
            "parameter_count": {
                "max": max(all_params) if all_params else 0,
                "avg": sum(all_params) / len(all_params) if all_params else 0.0,
            },
        }

    def _serialize_files(self, metrics: ProjectMetrics) -> dict[str, Any]:
        """Serialize file-level metrics.

        Args:
            metrics: ProjectMetrics to serialize

        Returns:
            Dictionary mapping file paths to serialized FileMetrics
        """
        files_data = {}

        for file_path, file_metrics in metrics.files.items():
            files_data[file_path] = {
                "file_path": file_metrics.file_path,
                "total_lines": file_metrics.total_lines,
                "code_lines": file_metrics.code_lines,
                "comment_lines": file_metrics.comment_lines,
                "blank_lines": file_metrics.blank_lines,
                "function_count": file_metrics.function_count,
                "class_count": file_metrics.class_count,
                "method_count": file_metrics.method_count,
                "total_complexity": file_metrics.total_complexity,
                "avg_complexity": file_metrics.avg_complexity,
                "max_complexity": file_metrics.max_complexity,
                "chunks": [
                    {
                        "cognitive_complexity": chunk.cognitive_complexity,
                        "cyclomatic_complexity": chunk.cyclomatic_complexity,
                        "max_nesting_depth": chunk.max_nesting_depth,
                        "parameter_count": chunk.parameter_count,
                        "lines_of_code": chunk.lines_of_code,
                        "smells": chunk.smells,
                        "complexity_grade": chunk.complexity_grade,
                    }
                    for chunk in file_metrics.chunks
                ],
            }

        return files_data

    def _deserialize_project_metrics(self, data: dict[str, Any]) -> ProjectMetrics:
        """Deserialize JSON data back to ProjectMetrics.

        Args:
            data: JSON data from baseline file

        Returns:
            ProjectMetrics instance
        """
        from ..metrics import ChunkMetrics, FileMetrics

        # Deserialize files
        files = {}
        for file_path, file_data in data["files"].items():
            # Deserialize chunks
            chunks = [
                ChunkMetrics(
                    cognitive_complexity=chunk_data["cognitive_complexity"],
                    cyclomatic_complexity=chunk_data["cyclomatic_complexity"],
                    max_nesting_depth=chunk_data["max_nesting_depth"],
                    parameter_count=chunk_data["parameter_count"],
                    lines_of_code=chunk_data["lines_of_code"],
                    smells=chunk_data.get("smells", []),
                )
                for chunk_data in file_data["chunks"]
            ]

            file_metrics = FileMetrics(
                file_path=file_data["file_path"],
                total_lines=file_data["total_lines"],
                code_lines=file_data["code_lines"],
                comment_lines=file_data["comment_lines"],
                blank_lines=file_data["blank_lines"],
                function_count=file_data["function_count"],
                class_count=file_data["class_count"],
                method_count=file_data["method_count"],
                total_complexity=file_data["total_complexity"],
                avg_complexity=file_data["avg_complexity"],
                max_complexity=file_data["max_complexity"],
                chunks=chunks,
            )

            files[file_path] = file_metrics

        # Create ProjectMetrics
        metrics = ProjectMetrics(
            project_root=data["project"]["path"],
            analyzed_at=datetime.fromisoformat(data["created_at"]),
            total_files=data["project"]["file_count"],
            total_functions=data["project"]["function_count"],
            total_classes=data["project"].get("class_count", 0),
            files=files,
        )

        # Recompute aggregates
        metrics.compute_aggregates()

        return metrics

    def _validate_baseline(self, data: dict[str, Any]) -> None:
        """Validate baseline data structure.

        Args:
            data: JSON data from baseline file

        Raises:
            BaselineCorruptedError: If required fields are missing
        """
        required_fields = ["version", "baseline_name", "created_at", "project", "files"]

        for field in required_fields:
            if field not in data:
                raise BaselineCorruptedError(f"Missing required field: {field}")

        # Validate version compatibility
        if data["version"] != self.BASELINE_VERSION:
            logger.warning(
                f"Baseline version mismatch: {data['version']} vs {self.BASELINE_VERSION}"
            )
