"""Code smell detection based on structural metrics.

This module provides smell detection functionality that identifies common
code smells based on complexity metrics and structural analysis.

Supported Smells:
- Long Method: Functions/methods with too many lines or complexity
- Deep Nesting: Excessive nesting depth
- Long Parameter List: Too many function parameters
- God Class: Classes with too many methods and lines
- Complex Method: High cyclomatic complexity
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config.thresholds import ThresholdConfig
    from ..metrics import ChunkMetrics, FileMetrics


class SmellSeverity(Enum):
    """Severity level for code smells.

    Attributes:
        INFO: Informational smell, no immediate action needed
        WARNING: Warning smell, should be addressed during refactoring
        ERROR: Error-level smell, requires immediate attention
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    def __str__(self) -> str:
        """Return string representation of severity."""
        return self.value


@dataclass
class CodeSmell:
    """Represents a detected code smell.

    Attributes:
        name: Human-readable smell name (e.g., "Long Method")
        description: Detailed description of the smell
        severity: Severity level (INFO, WARNING, ERROR)
        location: File location in format "file:line" or "file:line-range"
        metric_value: Actual metric value that triggered the smell
        threshold: Threshold value that was exceeded
        suggestion: Optional suggestion for fixing the smell
    """

    name: str
    description: str
    severity: SmellSeverity
    location: str
    metric_value: float
    threshold: float
    suggestion: str = ""

    def __str__(self) -> str:
        """Return string representation of code smell."""
        return (
            f"[{self.severity.value.upper()}] {self.name} at {self.location}: "
            f"{self.description} (value: {self.metric_value}, threshold: {self.threshold})"
        )


class SmellDetector:
    """Detects code smells based on structural metrics.

    This detector analyzes ChunkMetrics and FileMetrics to identify common
    code smells that indicate maintainability issues. Detection rules are
    configurable via ThresholdConfig.

    Detection Rules:
    - Long Method: lines > 50 OR cognitive_complexity > 15
    - Deep Nesting: max_nesting_depth > 4
    - Long Parameter List: parameter_count > 5
    - God Class: method_count > 20 AND lines > 500
    - Complex Method: cyclomatic_complexity > 10

    Example:
        detector = SmellDetector()
        smells = detector.detect(chunk_metrics)

        for smell in smells:
            print(f"{smell.severity}: {smell.name} at {smell.location}")
    """

    def __init__(self, thresholds: ThresholdConfig | None = None) -> None:
        """Initialize smell detector.

        Args:
            thresholds: Optional custom threshold configuration.
                       If None, uses default thresholds from ThresholdConfig.
        """
        if thresholds is None:
            # Import here to avoid circular dependency
            from ...config.thresholds import ThresholdConfig

            thresholds = ThresholdConfig()

        self.thresholds = thresholds

    def detect(
        self, metrics: ChunkMetrics, file_path: str = "", start_line: int = 0
    ) -> list[CodeSmell]:
        """Detect code smells in a single chunk (function/method/class).

        Analyzes the provided metrics and returns a list of detected smells.
        Each smell includes location information and severity level.

        Args:
            metrics: Chunk metrics to analyze
            file_path: Path to the file containing this chunk
            start_line: Starting line number of the chunk

        Returns:
            List of detected CodeSmell objects (empty if no smells found)
        """
        smells: list[CodeSmell] = []

        # Location string for reporting
        location = f"{file_path}:{start_line}" if file_path else f"line {start_line}"

        # 1. Long Method Detection
        # Rule: lines > 50 OR cognitive_complexity > 15
        long_method_lines_threshold = self.thresholds.smells.long_method_lines
        high_complexity_threshold = self.thresholds.smells.high_complexity

        is_long_by_lines = metrics.lines_of_code > long_method_lines_threshold
        is_long_by_complexity = metrics.cognitive_complexity > high_complexity_threshold

        if is_long_by_lines or is_long_by_complexity:
            description_parts = []
            if is_long_by_lines:
                description_parts.append(
                    f"{metrics.lines_of_code} lines (threshold: {long_method_lines_threshold})"
                )
            if is_long_by_complexity:
                description_parts.append(
                    f"cognitive complexity {metrics.cognitive_complexity} (threshold: {high_complexity_threshold})"
                )

            smells.append(
                CodeSmell(
                    name="Long Method",
                    description=f"Method/function is too long: {', '.join(description_parts)}",
                    severity=SmellSeverity.WARNING,
                    location=location,
                    metric_value=float(
                        max(metrics.lines_of_code, metrics.cognitive_complexity)
                    ),
                    threshold=float(
                        max(long_method_lines_threshold, high_complexity_threshold)
                    ),
                    suggestion="Consider breaking this method into smaller, focused functions",
                )
            )

        # 2. Deep Nesting Detection
        # Rule: max_nesting_depth > 4
        nesting_threshold = self.thresholds.smells.deep_nesting_depth

        if metrics.max_nesting_depth > nesting_threshold:
            smells.append(
                CodeSmell(
                    name="Deep Nesting",
                    description=f"Excessive nesting depth: {metrics.max_nesting_depth} levels (threshold: {nesting_threshold})",
                    severity=SmellSeverity.WARNING,
                    location=location,
                    metric_value=float(metrics.max_nesting_depth),
                    threshold=float(nesting_threshold),
                    suggestion="Consider extracting nested logic into separate functions or using early returns",
                )
            )

        # 3. Long Parameter List Detection
        # Rule: parameter_count > 5
        param_threshold = self.thresholds.smells.too_many_parameters

        if metrics.parameter_count > param_threshold:
            smells.append(
                CodeSmell(
                    name="Long Parameter List",
                    description=f"Too many parameters: {metrics.parameter_count} parameters (threshold: {param_threshold})",
                    severity=SmellSeverity.WARNING,
                    location=location,
                    metric_value=float(metrics.parameter_count),
                    threshold=float(param_threshold),
                    suggestion="Consider introducing a parameter object or using builder pattern",
                )
            )

        # 4. Complex Method Detection
        # Rule: cyclomatic_complexity > 10
        cyclomatic_threshold = self.thresholds.complexity.cyclomatic_moderate

        if metrics.cyclomatic_complexity > cyclomatic_threshold:
            smells.append(
                CodeSmell(
                    name="Complex Method",
                    description=f"High cyclomatic complexity: {metrics.cyclomatic_complexity} (threshold: {cyclomatic_threshold})",
                    severity=SmellSeverity.WARNING,
                    location=location,
                    metric_value=float(metrics.cyclomatic_complexity),
                    threshold=float(cyclomatic_threshold),
                    suggestion="Simplify control flow by extracting complex conditions into separate functions",
                )
            )

        return smells

    def detect_god_class(
        self, file_metrics: FileMetrics, file_path: str = ""
    ) -> list[CodeSmell]:
        """Detect God Class smell at file level.

        A God Class has too many responsibilities, indicated by high method
        count and large file size.

        Rule: method_count > 20 AND lines > 500

        Args:
            file_metrics: File-level metrics to analyze
            file_path: Path to the file being analyzed

        Returns:
            List containing CodeSmell if God Class detected (empty otherwise)
        """
        smells: list[CodeSmell] = []

        # God Class Detection
        # Rule: method_count > 20 AND total_lines > 500
        method_threshold = self.thresholds.smells.god_class_methods
        lines_threshold = self.thresholds.smells.god_class_lines

        is_god_class = (
            file_metrics.method_count > method_threshold
            and file_metrics.total_lines > lines_threshold
        )

        if is_god_class:
            location = file_path if file_path else "file"
            smells.append(
                CodeSmell(
                    name="God Class",
                    description=(
                        f"Class has too many responsibilities: "
                        f"{file_metrics.method_count} methods (threshold: {method_threshold}), "
                        f"{file_metrics.total_lines} lines (threshold: {lines_threshold})"
                    ),
                    severity=SmellSeverity.ERROR,
                    location=location,
                    metric_value=float(file_metrics.method_count),
                    threshold=float(method_threshold),
                    suggestion="Split this class into smaller, focused classes following Single Responsibility Principle",
                )
            )

        return smells

    def detect_all(
        self, file_metrics: FileMetrics, file_path: str = ""
    ) -> list[CodeSmell]:
        """Detect all code smells across an entire file.

        Analyzes both chunk-level smells (Long Method, Deep Nesting, etc.)
        and file-level smells (God Class).

        Args:
            file_metrics: File metrics containing all chunks
            file_path: Path to the file being analyzed

        Returns:
            List of all detected CodeSmell objects
        """
        all_smells: list[CodeSmell] = []

        # Detect chunk-level smells
        for i, chunk in enumerate(file_metrics.chunks):
            # Estimate start line for each chunk
            # This is a rough approximation - ideally we'd have actual line numbers
            estimated_start_line = (
                1 + i * 20
            )  # Rough estimate assuming 20 lines per chunk

            chunk_smells = self.detect(chunk, file_path, estimated_start_line)
            all_smells.extend(chunk_smells)

        # Detect file-level smells (God Class)
        file_smells = self.detect_god_class(file_metrics, file_path)
        all_smells.extend(file_smells)

        return all_smells

    def get_smell_summary(self, smells: list[CodeSmell]) -> dict[str, int]:
        """Generate summary statistics for detected smells.

        Args:
            smells: List of detected code smells

        Returns:
            Dictionary with counts by severity and smell type
        """
        summary: dict[str, int] = {
            "total": len(smells),
            "error": sum(1 for s in smells if s.severity == SmellSeverity.ERROR),
            "warning": sum(1 for s in smells if s.severity == SmellSeverity.WARNING),
            "info": sum(1 for s in smells if s.severity == SmellSeverity.INFO),
        }

        # Count by smell type
        smell_types = {}
        for smell in smells:
            smell_types[smell.name] = smell_types.get(smell.name, 0) + 1

        summary["by_type"] = smell_types

        return summary
