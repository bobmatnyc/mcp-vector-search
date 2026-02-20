"""Code quality analysis for engineer profiling.

This module provides AST-based analysis of Python code to extract
complexity metrics, code smells, and quality indicators for each function.
"""

import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass
class FunctionMetrics:
    """Metrics for a single function."""

    name: str
    file_path: str
    start_line: int
    end_line: int
    lines: int
    cyclomatic_complexity: int
    max_nesting: int
    parameter_count: int
    author: str | None = None

    @property
    def is_complex(self) -> bool:
        """Check if function exceeds complexity threshold."""
        return self.cyclomatic_complexity > 10

    @property
    def is_very_complex(self) -> bool:
        """Check if function exceeds very high complexity threshold."""
        return self.cyclomatic_complexity > 20

    @property
    def is_long(self) -> bool:
        """Check if function exceeds line count threshold."""
        return self.lines > 50

    @property
    def is_very_long(self) -> bool:
        """Check if function exceeds very high line count threshold."""
        return self.lines > 100

    @property
    def is_deeply_nested(self) -> bool:
        """Check if function exceeds nesting depth threshold."""
        return self.max_nesting > 4

    @property
    def has_many_parameters(self) -> bool:
        """Check if function has too many parameters."""
        return self.parameter_count > 5


class ComplexityAnalyzer(ast.NodeVisitor):
    """Calculate cyclomatic complexity and nesting depth.

    Cyclomatic complexity counts decision points in code:
    - Each if, for, while, except adds +1
    - Each and/or in boolean expressions adds +1
    - Each comprehension adds +1
    - Base complexity starts at 1

    Nesting depth tracks maximum indentation level.
    """

    def __init__(self):
        self.complexity = 1  # Base complexity
        self.max_nesting = 0
        self._current_nesting = 0

    def visit_If(self, node):
        """Count if statements and track nesting."""
        self.complexity += 1
        self._visit_nested(node)

    def visit_For(self, node):
        """Count for loops and track nesting."""
        self.complexity += 1
        self._visit_nested(node)

    def visit_While(self, node):
        """Count while loops and track nesting."""
        self.complexity += 1
        self._visit_nested(node)

    def visit_ExceptHandler(self, node):
        """Count except handlers and track nesting."""
        self.complexity += 1
        self._visit_nested(node)

    def visit_With(self, node):
        """Track nesting for with statements."""
        self._visit_nested(node)

    def visit_BoolOp(self, node):
        """Count and/or operators in boolean expressions."""
        # Each and/or adds to complexity
        self.complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_comprehension(self, node):
        """Count list/dict/set comprehensions and their conditions."""
        self.complexity += 1
        if node.ifs:
            self.complexity += len(node.ifs)
        self.generic_visit(node)

    def _visit_nested(self, node):
        """Visit nested block with nesting tracking."""
        self._current_nesting += 1
        self.max_nesting = max(self.max_nesting, self._current_nesting)
        self.generic_visit(node)
        self._current_nesting -= 1


def analyze_python_file(file_path: Path) -> list[FunctionMetrics]:
    """Analyze a Python file for function metrics.

    Args:
        file_path: Path to Python file to analyze

    Returns:
        List of FunctionMetrics for each function/method in file
    """
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError) as e:
        logger.debug(f"Failed to parse {file_path}: {e}")
        return []

    metrics = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            analyzer = ComplexityAnalyzer()
            analyzer.visit(node)

            # Calculate function length
            lines = (node.end_lineno or node.lineno) - node.lineno + 1

            # Count parameters (args + kwargs)
            param_count = len(node.args.args) + len(node.args.kwonlyargs)

            metrics.append(
                FunctionMetrics(
                    name=node.name,
                    file_path=str(file_path),
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    lines=lines,
                    cyclomatic_complexity=analyzer.complexity,
                    max_nesting=analyzer.max_nesting,
                    parameter_count=param_count,
                )
            )

    return metrics


def get_function_author(
    repo_root: Path, file_path: Path, start_line: int
) -> str | None:
    """Get the primary author of a function using git blame.

    Args:
        repo_root: Root directory of git repository
        file_path: Path to file containing function
        start_line: Starting line number of function

    Returns:
        Author name or None if git blame fails
    """
    try:
        # Use --porcelain for machine-readable output
        result = subprocess.run(  # nosec B607
            [
                "git",
                "blame",
                "-L",
                f"{start_line},{start_line}",
                "--porcelain",
                str(file_path),
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        if result.returncode == 0:
            # Parse porcelain format to extract author
            for line in result.stdout.split("\n"):
                if line.startswith("author "):
                    return line[7:].strip()

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.debug(f"git blame failed for {file_path}:{start_line}: {e}")

    return None


def get_commit_stats(repo_root: Path) -> dict[str, dict[str, str | int]]:
    """Get commit statistics per author.

    Args:
        repo_root: Root directory of git repository

    Returns:
        Dict mapping author name to commit stats (commits, email)
    """
    try:
        result = subprocess.run(  # nosec B607
            ["git", "shortlog", "-sne", "--all"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        if result.returncode != 0:
            logger.warning(f"git shortlog failed: {result.stderr}")
            return {}

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"Failed to get commit stats: {e}")
        return {}

    stats = {}
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue

        parts = line.strip().split("\t", 1)
        if len(parts) < 2:
            continue

        count = int(parts[0].strip())
        author_part = parts[1]

        # Extract name and email from "Name <email>" format
        if "<" in author_part and ">" in author_part:
            name = author_part.split("<")[0].strip()
            email = author_part.split("<")[1].rstrip(">")
        else:
            name = author_part
            email = ""

        stats[name] = {"commits": count, "email": email}

    return stats


def get_total_lines_by_author(repo_root: Path) -> dict[str, int]:
    """Get total lines authored by each developer.

    Args:
        repo_root: Root directory of git repository

    Returns:
        Dict mapping author name to total lines written
    """
    try:
        result = subprocess.run(  # nosec B607
            [
                "git",
                "log",
                "--format=%an",
                "--numstat",
                "--no-merges",
                "--all",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        if result.returncode != 0:
            logger.warning(f"git log failed: {result.stderr}")
            return {}

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"Failed to get line stats: {e}")
        return {}

    # Parse git log output
    # Format alternates between:
    # - Author name line
    # - Empty line
    # - Numstat lines (additions, deletions, filename)
    author_lines: dict[str, int] = {}
    current_author = None

    for line in result.stdout.split("\n"):
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Check if line is numstat (starts with numbers or dashes)
        if line[0].isdigit() or line.startswith("-"):
            # Numstat line: additions deletions filename
            parts = line.split(None, 2)
            if len(parts) >= 2 and current_author:
                try:
                    additions = int(parts[0]) if parts[0] != "-" else 0
                    author_lines[current_author] = (
                        author_lines.get(current_author, 0) + additions
                    )
                except ValueError:
                    pass
        else:
            # Author name line
            current_author = line

    return author_lines
