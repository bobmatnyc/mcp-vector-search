"""Dead code detection using reachability analysis."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from .entry_points import EntryPoint, EntryPointDetector, EntryPointType

if TYPE_CHECKING:
    pass


class Confidence(str, Enum):
    """Confidence level for dead code findings.

    Attributes:
        HIGH: Private function, definitely not called externally
        MEDIUM: Public function, might be API but not called internally
        LOW: Might be dynamically called or used as callback
    """

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    def __str__(self) -> str:
        """Return string representation."""
        return self.value


@dataclass
class DeadCodeFinding:
    """Represents a potentially dead code finding.

    Attributes:
        function_name: Name of the potentially dead function
        file_path: Path to file containing the function
        start_line: Starting line number
        end_line: Ending line number
        confidence: Confidence level (HIGH, MEDIUM, LOW)
        reason: Human-readable explanation
        caveats: List of caveats that might affect analysis accuracy
    """

    function_name: str
    file_path: str
    start_line: int
    end_line: int
    confidence: Confidence
    reason: str
    caveats: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Return string representation."""
        location = f"{self.file_path}:{self.start_line}-{self.end_line}"
        return f"[{self.confidence.value}] {self.function_name} at {location}: {self.reason}"


@dataclass
class DeadCodeReport:
    """Complete dead code analysis report.

    Attributes:
        entry_points: List of detected entry points
        findings: List of dead code findings
        total_functions: Total number of functions analyzed
        reachable_count: Number of reachable functions
        unreachable_count: Number of unreachable functions
    """

    entry_points: list[EntryPoint]
    findings: list[DeadCodeFinding]
    total_functions: int
    reachable_count: int
    unreachable_count: int

    @property
    def reachable_percentage(self) -> float:
        """Calculate percentage of reachable functions.

        Returns:
            Percentage (0-100) of functions that are reachable
        """
        if self.total_functions == 0:
            return 100.0
        return (self.reachable_count / self.total_functions) * 100

    def get_findings_by_confidence(
        self, confidence: Confidence
    ) -> list[DeadCodeFinding]:
        """Filter findings by confidence level.

        Args:
            confidence: Confidence level to filter by

        Returns:
            List of findings matching the confidence level
        """
        return [f for f in self.findings if f.confidence == confidence]


class DeadCodeAnalyzer:
    """Analyzes codebase for dead/unreachable code.

    Uses reachability analysis to identify functions that are never called
    from any entry point. Entry points include:
    - Main blocks (if __name__ == "__main__")
    - CLI commands (@click.command, etc.)
    - HTTP routes (@app.get, etc.)
    - Test functions (test_*, @pytest.fixture)
    - Module exports (__all__)
    - Optionally, public functions

    Example:
        analyzer = DeadCodeAnalyzer(
            include_public_entry_points=False,
            min_confidence=Confidence.MEDIUM
        )
        report = analyzer.analyze(project_path, chunks)

        for finding in report.findings:
            print(f"{finding.confidence}: {finding.function_name} - {finding.reason}")
    """

    def __init__(
        self,
        include_public_entry_points: bool = False,
        custom_entry_points: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        min_confidence: Confidence = Confidence.LOW,
    ) -> None:
        """Initialize dead code analyzer.

        Args:
            include_public_entry_points: If True, treat all public functions as entry points
            custom_entry_points: List of custom function names to treat as entry points
            exclude_patterns: List of file path patterns to exclude from analysis
            min_confidence: Minimum confidence level for reporting findings
        """
        self.include_public = include_public_entry_points
        self.custom_entry_points = custom_entry_points or []
        self.exclude_patterns = exclude_patterns or []
        self.min_confidence = min_confidence
        self.entry_detector = EntryPointDetector(
            include_public=include_public_entry_points
        )

    def analyze(self, project_path: Path, chunks: list[dict]) -> DeadCodeReport:
        """Perform dead code analysis on the project.

        Args:
            project_path: Root path of the project
            chunks: List of code chunks from the index (with function metadata)

        Returns:
            DeadCodeReport with findings
        """
        logger.info(f"Starting dead code analysis on {project_path}")

        # 1. Detect entry points
        entry_points = self._collect_entry_points(project_path)
        logger.info(f"Detected {len(entry_points)} entry points")

        # 2. Build call graph (function -> list of functions it calls)
        call_graph = self._build_call_graph(chunks)
        logger.debug(f"Built call graph with {len(call_graph)} nodes")

        # 3. Compute reachability from entry points
        reachable = self._compute_reachability(entry_points, call_graph, chunks)
        logger.info(f"Found {len(reachable)} reachable functions")

        # 4. Find unreachable functions
        findings = self._find_dead_code(chunks, reachable)

        # 5. Filter by confidence
        findings = [f for f in findings if self._meets_confidence(f.confidence)]

        # Count total functions (functions and methods only)
        total_functions = len(
            [c for c in chunks if c.get("type") in ("function", "method")]
        )

        logger.info(
            f"Analysis complete: {len(findings)} potential dead code findings "
            f"({len(reachable)} reachable, {len(findings)} unreachable out of {total_functions} total)"
        )

        return DeadCodeReport(
            entry_points=entry_points,
            findings=findings,
            total_functions=total_functions,
            reachable_count=len(reachable),
            unreachable_count=len(findings),
        )

    def _collect_entry_points(self, project_path: Path) -> list[EntryPoint]:
        """Collect all entry points including custom ones.

        Args:
            project_path: Root path of the project

        Returns:
            List of all entry points
        """
        # Detect entry points from code
        entry_points = self.entry_detector.detect_from_directory(project_path)

        # Add custom entry points
        for custom_name in self.custom_entry_points:
            entry_points.append(
                EntryPoint(
                    name=custom_name,
                    file_path="<custom>",
                    line_number=0,
                    type=EntryPointType.CUSTOM,
                    confidence=1.0,
                )
            )

        return entry_points

    def _build_call_graph(self, chunks: list[dict]) -> dict[str, set[str]]:
        """Build caller -> callees mapping from chunks.

        Extracts function calls from each chunk's code and builds a directed graph
        where edges represent "calls" relationships.

        Args:
            chunks: List of code chunks with function metadata

        Returns:
            Dictionary mapping function names to sets of called function names
        """
        from ..core.relationships import extract_function_calls

        call_graph: dict[str, set[str]] = defaultdict(set)

        for chunk in chunks:
            # Skip non-function chunks
            chunk_type = chunk.get("type", "")
            if chunk_type not in ("function", "method"):
                continue

            # Get function name
            function_name = chunk.get("function_name") or chunk.get("class_name")
            if not function_name:
                continue

            # Extract function calls from the code
            code = chunk.get("content", "")
            called_functions = extract_function_calls(code)

            # Add to call graph
            call_graph[function_name].update(called_functions)

        return call_graph

    def _compute_reachability(
        self,
        entry_points: list[EntryPoint],
        call_graph: dict[str, set[str]],
        chunks: list[dict],
    ) -> set[str]:
        """BFS from entry points to find all reachable functions.

        Performs breadth-first search starting from all entry points,
        following call relationships to identify all reachable code.

        Args:
            entry_points: List of entry point functions
            call_graph: Function call graph (caller -> callees)
            chunks: List of all chunks (for validation)

        Returns:
            Set of reachable function names
        """
        reachable: set[str] = set()
        queue: deque[str] = deque()

        # Start with all entry point names
        for entry_point in entry_points:
            queue.append(entry_point.name)
            reachable.add(entry_point.name)

        # BFS traversal
        while queue:
            current = queue.popleft()

            # Get all functions called by current function
            callees = call_graph.get(current, set())

            for callee in callees:
                if callee not in reachable:
                    reachable.add(callee)
                    queue.append(callee)

        return reachable

    def _find_dead_code(
        self, chunks: list[dict], reachable: set[str]
    ) -> list[DeadCodeFinding]:
        """Identify unreachable functions and assign confidence levels.

        Args:
            chunks: List of all code chunks
            reachable: Set of reachable function names

        Returns:
            List of dead code findings
        """
        findings: list[DeadCodeFinding] = []

        for chunk in chunks:
            # Only analyze functions and methods
            chunk_type = chunk.get("type", "")
            if chunk_type not in ("function", "method"):
                continue

            # Get function name
            function_name = chunk.get("function_name") or chunk.get("class_name")
            if not function_name:
                continue

            # Skip if reachable
            if function_name in reachable:
                continue

            # Check if excluded by pattern
            file_path = chunk.get("file_path", "")
            if self._is_excluded(file_path):
                continue

            # Assign confidence and create finding
            confidence = self._assign_confidence(chunk)
            reason = self._generate_reason(chunk, confidence)
            caveats = self._generate_caveats(chunk)

            findings.append(
                DeadCodeFinding(
                    function_name=function_name,
                    file_path=file_path,
                    start_line=chunk.get("start_line", 0),
                    end_line=chunk.get("end_line", 0),
                    confidence=confidence,
                    reason=reason,
                    caveats=caveats,
                )
            )

        return findings

    def _assign_confidence(self, chunk: dict) -> Confidence:
        """Assign confidence level based on function characteristics.

        Confidence heuristics:
        - HIGH: Private functions (_name) with no decorators
        - MEDIUM: Public functions with no decorators
        - LOW: Functions with decorators (might be registered callbacks)

        Args:
            chunk: Code chunk dictionary

        Returns:
            Confidence level
        """
        function_name = chunk.get("function_name") or chunk.get("class_name", "")
        decorators = chunk.get("decorators", [])

        # LOW confidence if has decorators (might be registered as callback)
        if decorators:
            return Confidence.LOW

        # HIGH confidence if private (starts with _)
        if function_name.startswith("_"):
            return Confidence.HIGH

        # MEDIUM confidence for public functions
        return Confidence.MEDIUM

    def _generate_reason(self, chunk: dict, confidence: Confidence) -> str:
        """Generate human-readable reason for dead code finding.

        Args:
            chunk: Code chunk dictionary
            confidence: Confidence level

        Returns:
            Reason string
        """
        function_name = chunk.get("function_name") or chunk.get("class_name", "")

        if confidence == Confidence.HIGH:
            return (
                f"Private function '{function_name}' is not called from any entry point"
            )
        elif confidence == Confidence.MEDIUM:
            return f"Public function '{function_name}' is not called from any entry point (might be external API)"
        else:
            return f"Function '{function_name}' with decorators is not called (might be registered callback)"

    def _generate_caveats(self, chunk: dict) -> list[str]:
        """Generate caveats for dead code finding.

        Caveats are warnings about potential false positives.

        Args:
            chunk: Code chunk dictionary

        Returns:
            List of caveat strings
        """
        caveats: list[str] = []

        decorators = chunk.get("decorators", [])
        if decorators:
            caveats.append(
                f"Has decorators {decorators} - might be dynamically registered"
            )

        # Check for common callback patterns in function name
        function_name = chunk.get("function_name") or chunk.get("class_name", "")
        callback_patterns = ["callback", "handler", "on_", "handle_"]
        if any(pattern in function_name.lower() for pattern in callback_patterns):
            caveats.append(
                "Name suggests callback/handler - might be called dynamically"
            )

        return caveats

    def _is_excluded(self, file_path: str) -> bool:
        """Check if file path matches exclusion patterns.

        Args:
            file_path: Path to check

        Returns:
            True if excluded
        """
        for pattern in self.exclude_patterns:
            if pattern in file_path:
                return True
        return False

    def _meets_confidence(self, confidence: Confidence) -> bool:
        """Check if finding meets minimum confidence threshold.

        Confidence ordering: HIGH > MEDIUM > LOW

        Args:
            confidence: Confidence level to check

        Returns:
            True if meets or exceeds minimum confidence
        """
        confidence_order = {Confidence.HIGH: 3, Confidence.MEDIUM: 2, Confidence.LOW: 1}

        return confidence_order[confidence] >= confidence_order[self.min_confidence]
