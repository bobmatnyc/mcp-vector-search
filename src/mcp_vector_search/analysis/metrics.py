"""Metric dataclasses for structural code analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ChunkMetrics:
    """Metrics for a single code chunk (function/class/method).

    Tracks complexity metrics, code smells, and computes quality grades
    for individual code chunks.

    Attributes:
        cognitive_complexity: Cognitive complexity score (how hard to understand)
        cyclomatic_complexity: Cyclomatic complexity (number of decision paths)
        max_nesting_depth: Maximum nesting level (if/for/while/try depth)
        parameter_count: Number of function parameters
        lines_of_code: Total lines in the chunk
        smells: List of detected code smells (e.g., "too_many_parameters")
        complexity_grade: Computed A-F grade based on cognitive complexity
    """

    cognitive_complexity: int = 0
    cyclomatic_complexity: int = 0
    max_nesting_depth: int = 0
    parameter_count: int = 0
    lines_of_code: int = 0

    # Code smells detected
    smells: list[str] = field(default_factory=list)

    # Computed grades (A-F scale)
    complexity_grade: str = field(init=False, default="A")

    def __post_init__(self) -> None:
        """Initialize computed fields after dataclass initialization."""
        self.complexity_grade = self._compute_grade()

    def _compute_grade(self) -> str:
        """Compute A-F grade based on cognitive complexity.

        Grade thresholds:
        - A: 0-5 (excellent)
        - B: 6-10 (good)
        - C: 11-20 (acceptable)
        - D: 21-30 (needs improvement)
        - F: 31+ (refactor recommended)

        Returns:
            Letter grade from A to F
        """
        if self.cognitive_complexity <= 5:
            return "A"
        elif self.cognitive_complexity <= 10:
            return "B"
        elif self.cognitive_complexity <= 20:
            return "C"
        elif self.cognitive_complexity <= 30:
            return "D"
        else:
            return "F"

    def to_metadata(self) -> dict[str, Any]:
        """Flatten metrics for ChromaDB metadata storage.

        ChromaDB supports: str, int, float, bool.
        Lists are converted to JSON strings for compatibility.

        Returns:
            Dictionary of flattened metrics compatible with ChromaDB
        """
        import json

        return {
            "cognitive_complexity": self.cognitive_complexity,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "max_nesting_depth": self.max_nesting_depth,
            "parameter_count": self.parameter_count,
            "lines_of_code": self.lines_of_code,
            "complexity_grade": self.complexity_grade,
            "code_smells": json.dumps(self.smells),  # Convert list to JSON string
            "smell_count": len(self.smells),
        }


@dataclass
class FileMetrics:
    """Aggregated metrics for an entire file.

    Tracks file-level statistics and aggregates chunk metrics for all
    functions/classes within the file.

    Attributes:
        file_path: Relative or absolute path to the file
        total_lines: Total lines in file (including blank/comments)
        code_lines: Lines containing code
        comment_lines: Lines containing comments
        blank_lines: Blank lines
        function_count: Number of top-level functions
        class_count: Number of classes
        method_count: Number of methods (functions inside classes)
        total_complexity: Sum of cognitive complexity across all chunks
        avg_complexity: Average cognitive complexity per chunk
        max_complexity: Maximum cognitive complexity in any chunk
        chunks: List of chunk metrics for each function/class
    """

    file_path: str
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0

    function_count: int = 0
    class_count: int = 0
    method_count: int = 0

    # Aggregated complexity
    total_complexity: int = 0
    avg_complexity: float = 0.0
    max_complexity: int = 0

    # Chunk metrics for each function/class
    chunks: list[ChunkMetrics] = field(default_factory=list)

    def compute_aggregates(self) -> None:
        """Compute aggregate metrics from chunk metrics.

        Calculates total_complexity, avg_complexity, and max_complexity
        by aggregating values from all chunks.
        """
        if not self.chunks:
            self.total_complexity = 0
            self.avg_complexity = 0.0
            self.max_complexity = 0
            return

        # Compute complexity aggregates
        complexities = [chunk.cognitive_complexity for chunk in self.chunks]
        self.total_complexity = sum(complexities)
        self.avg_complexity = self.total_complexity / len(self.chunks)
        self.max_complexity = max(complexities)

    @property
    def health_score(self) -> float:
        """Calculate 0.0-1.0 health score based on metrics.

        Health score considers:
        - Average complexity (lower is better)
        - Code smells count (fewer is better)
        - Comment ratio (balanced is better)

        Returns:
            Health score from 0.0 (poor) to 1.0 (excellent)
        """
        score = 1.0

        # Penalty for high average complexity (A=0%, B=-10%, C=-20%, D=-30%, F=-50%)
        if self.avg_complexity > 30:
            score -= 0.5
        elif self.avg_complexity > 20:
            score -= 0.3
        elif self.avg_complexity > 10:
            score -= 0.2
        elif self.avg_complexity > 5:
            score -= 0.1

        # Penalty for code smells (up to -30%)
        total_smells = sum(len(chunk.smells) for chunk in self.chunks)
        smell_penalty = min(0.3, total_smells * 0.05)  # 5% per smell, max 30%
        score -= smell_penalty

        # Penalty for poor comment ratio (ideal: 10-30%)
        if self.total_lines > 0:
            comment_ratio = self.comment_lines / self.total_lines
            if comment_ratio < 0.1:  # Too few comments
                score -= 0.1
            elif comment_ratio > 0.5:  # Too many comments (suspicious)
                score -= 0.1

        return max(0.0, score)  # Clamp to 0.0 minimum


@dataclass
class ProjectMetrics:
    """Project-wide metric aggregates.

    Tracks project-level statistics and identifies complexity hotspots
    across the entire codebase.

    Attributes:
        project_root: Root directory of the project
        analyzed_at: Timestamp when analysis was performed
        total_files: Total number of analyzed files
        total_lines: Total lines across all files
        total_functions: Total number of functions
        total_classes: Total number of classes
        files: Dictionary mapping file paths to FileMetrics
        avg_file_complexity: Average complexity across all files
        hotspots: List of file paths with highest complexity (top 10)
    """

    project_root: str
    analyzed_at: datetime = field(default_factory=datetime.now)

    total_files: int = 0
    total_lines: int = 0
    total_functions: int = 0
    total_classes: int = 0

    # File metrics indexed by path
    files: dict[str, FileMetrics] = field(default_factory=dict)

    # Project-wide aggregates
    avg_file_complexity: float = 0.0
    hotspots: list[str] = field(default_factory=list)  # Top 10 complex files

    def compute_aggregates(self) -> None:
        """Compute project-wide aggregates from file metrics.

        Calculates:
        - Total files, lines, functions, classes
        - Average file complexity
        - Identifies complexity hotspots
        """
        if not self.files:
            self.total_files = 0
            self.total_lines = 0
            self.total_functions = 0
            self.total_classes = 0
            self.avg_file_complexity = 0.0
            self.hotspots = []
            return

        # Compute totals
        self.total_files = len(self.files)
        self.total_lines = sum(f.total_lines for f in self.files.values())
        self.total_functions = sum(f.function_count for f in self.files.values())
        self.total_classes = sum(f.class_count for f in self.files.values())

        # Compute average file complexity
        file_complexities = [f.avg_complexity for f in self.files.values() if f.chunks]
        if file_complexities:
            self.avg_file_complexity = sum(file_complexities) / len(file_complexities)
        else:
            self.avg_file_complexity = 0.0

        # Identify hotspots (top 10 most complex files)
        hotspot_files = self.get_hotspots(limit=10)
        self.hotspots = [f.file_path for f in hotspot_files]

    def get_hotspots(self, limit: int = 10) -> list[FileMetrics]:
        """Return top N most complex files.

        Complexity is determined by average cognitive complexity per chunk.
        Files with no chunks are excluded.

        Args:
            limit: Maximum number of hotspots to return

        Returns:
            List of FileMetrics sorted by complexity (highest first)
        """
        # Filter files with chunks and sort by avg complexity
        files_with_complexity = [f for f in self.files.values() if f.chunks]
        sorted_files = sorted(
            files_with_complexity, key=lambda f: f.avg_complexity, reverse=True
        )
        return sorted_files[:limit]

    def to_summary(self) -> dict[str, Any]:
        """Generate summary dict for reporting.

        Returns:
            Dictionary containing project summary with key metrics
        """
        return {
            "project_root": self.project_root,
            "analyzed_at": self.analyzed_at.isoformat(),
            "total_files": self.total_files,
            "total_lines": self.total_lines,
            "total_functions": self.total_functions,
            "total_classes": self.total_classes,
            "avg_file_complexity": round(self.avg_file_complexity, 2),
            "hotspots": self.hotspots,
            "complexity_distribution": self._compute_grade_distribution(),
            "health_metrics": {
                "avg_health_score": self._compute_avg_health_score(),
                "files_needing_attention": self._count_files_needing_attention(),
            },
        }

    def _compute_grade_distribution(self) -> dict[str, int]:
        """Compute distribution of complexity grades across all chunks.

        Returns:
            Dictionary mapping grade (A-F) to count of chunks
        """
        distribution: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}

        for file_metrics in self.files.values():
            for chunk in file_metrics.chunks:
                distribution[chunk.complexity_grade] += 1

        return distribution

    def _compute_avg_health_score(self) -> float:
        """Compute average health score across all files.

        Returns:
            Average health score from 0.0 to 1.0
        """
        if not self.files:
            return 1.0

        health_scores = [f.health_score for f in self.files.values()]
        return sum(health_scores) / len(health_scores)

    def _count_files_needing_attention(self) -> int:
        """Count files with health score below 0.7.

        Returns:
            Number of files that need attention
        """
        return sum(1 for f in self.files.values() if f.health_score < 0.7)
