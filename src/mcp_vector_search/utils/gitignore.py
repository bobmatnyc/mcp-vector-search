"""Gitignore parsing and matching utilities."""

import fnmatch
import re
from pathlib import Path

from loguru import logger


class GitignorePattern:
    """Represents a single gitignore pattern with its matching logic."""

    def __init__(
        self, pattern: str, is_negation: bool = False, is_directory_only: bool = False
    ):
        """Initialize a gitignore pattern.

        Args:
            pattern: The pattern string
            is_negation: Whether this is a negation pattern (starts with !)
            is_directory_only: Whether this pattern only matches directories (ends with /)
        """
        self.original_pattern = pattern
        self.is_negation = is_negation
        self.is_directory_only = is_directory_only
        self.pattern = self._normalize_pattern(pattern)

        # Pre-compile regex pattern if needed (cache for performance)
        self._regex = None
        if "**" in self.pattern:
            regex_pattern = self.pattern.replace("**", ".*")
            regex_pattern = regex_pattern.replace("*", "[^/]*")
            regex_pattern = regex_pattern.replace("?", "[^/]")
            regex_pattern = f"^{regex_pattern}$"
            try:
                self._regex = re.compile(regex_pattern)
            except re.error:
                self._regex = None

    def _normalize_pattern(self, pattern: str) -> str:
        """Normalize the pattern for matching."""
        # Remove leading ! for negation patterns
        if pattern.startswith("!"):
            pattern = pattern[1:]

        # Remove trailing / for directory-only patterns
        if pattern.endswith("/"):
            pattern = pattern[:-1]

        # Handle leading slash (absolute from repo root)
        if pattern.startswith("/"):
            pattern = pattern[1:]

        return pattern

    def matches(self, path: str, is_directory: bool = False) -> bool:
        """Check if this pattern matches the given path.

        PERFORMANCE: Optimized to minimize string operations and regex compilations.

        Args:
            path: Relative path from repository root
            is_directory: Whether the path is a directory

        Returns:
            True if the pattern matches
        """
        # Convert path separators for consistent matching (once)
        path = path.replace("\\", "/")
        pattern = self.pattern.replace("\\", "/")

        # FAST PATH: Try exact match first (cheapest operation)
        if fnmatch.fnmatch(path, pattern):
            return True

        # FAST PATH: Use pre-compiled regex for ** patterns
        if self._regex:
            if self._regex.match(path):
                return True

        # Split path once for reuse
        path_parts = path.split("/")

        # For directory-only patterns, check if any parent directory matches
        # This implements Git's behavior where "dir/" excludes both the directory
        # AND all files within it recursively
        if self.is_directory_only:
            # Check each parent directory component
            for i in range(1, len(path_parts) + 1):
                parent = "/".join(path_parts[:i])
                if fnmatch.fnmatch(parent, pattern):
                    return True

        # Try matching any parent directory (suffix matching)
        for i in range(len(path_parts)):
            subpath = "/".join(path_parts[i:])
            if fnmatch.fnmatch(subpath, pattern):
                return True

        return False


class GitignoreParser:
    """Parser for .gitignore files with proper pattern matching."""

    def __init__(self, project_root: Path):
        """Initialize gitignore parser.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.patterns: list[GitignorePattern] = []
        self._load_gitignore_files()

    def _load_gitignore_files(self) -> None:
        """Load .gitignore file from project root only.

        Note: Only the root .gitignore is loaded to avoid performance issues
        with rglob traversing large directory trees (e.g., node_modules with
        250K+ files). Subdirectory .gitignore files are intentionally skipped
        as they would add significant overhead without much benefit for
        semantic code search indexing.
        """
        # Load root .gitignore only
        root_gitignore = self.project_root / ".gitignore"
        if root_gitignore.exists():
            self._parse_gitignore_file(root_gitignore)

    def _parse_gitignore_file(self, gitignore_path: Path) -> None:
        """Parse a single .gitignore file.

        Args:
            gitignore_path: Path to the .gitignore file
        """
        try:
            with open(gitignore_path, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            for _line_num, line in enumerate(lines, 1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Check for negation pattern
                is_negation = line.startswith("!")

                # Check for directory-only pattern
                is_directory_only = line.endswith("/")

                # Create pattern (all patterns are from root .gitignore)
                pattern = GitignorePattern(line, is_negation, is_directory_only)
                self.patterns.append(pattern)

        except Exception as e:
            logger.warning(f"Failed to parse {gitignore_path}: {e}")

    def is_ignored(self, path: Path, is_directory: bool | None = None) -> bool:
        """Check if a path should be ignored according to .gitignore rules.

        Args:
            path: Path to check (can be absolute or relative to project root)
            is_directory: Optional hint if path is a directory.
                         If None, will check filesystem (slower).
                         If provided, skips filesystem check (faster).

        Returns:
            True if the path should be ignored
        """
        try:
            # SHORT-CIRCUIT: If no patterns, nothing is ignored
            # This prevents 200k+ unnecessary filesystem stat() calls on projects
            # without .gitignore files
            if not self.patterns:
                return False

            # Convert to relative path from project root
            if path.is_absolute():
                relative_path = path.relative_to(self.project_root)
            else:
                relative_path = path

            path_str = str(relative_path).replace("\\", "/")

            # Only check if directory when needed and not provided as hint
            # PERFORMANCE: Passing is_directory hint from caller (e.g., os.walk)
            # avoids hundreds of thousands of stat() calls on large repositories
            if is_directory is None:
                is_directory = path.is_dir() if path.exists() else False

            # Apply patterns in order, with later patterns overriding earlier ones
            ignored = False

            for pattern in self.patterns:
                if pattern.matches(path_str, is_directory):
                    ignored = not pattern.is_negation

            return ignored

        except ValueError:
            # Path is not relative to project root
            return False
        except Exception as e:
            logger.debug(f"Error checking gitignore for {path}: {e}")
            return False

    def get_ignored_patterns(self) -> list[str]:
        """Get list of all ignore patterns.

        Returns:
            List of pattern strings
        """
        return [p.original_pattern for p in self.patterns if not p.is_negation]

    def get_negation_patterns(self) -> list[str]:
        """Get list of all negation patterns.

        Returns:
            List of negation pattern strings
        """
        return [p.original_pattern for p in self.patterns if p.is_negation]


def create_gitignore_parser(project_root: Path) -> GitignoreParser:
    """Create a gitignore parser for the given project.

    Args:
        project_root: Root directory of the project

    Returns:
        GitignoreParser instance
    """
    return GitignoreParser(project_root)


def is_path_gitignored(
    path: Path, project_root: Path, is_directory: bool | None = None
) -> bool:
    """Quick function to check if a path is gitignored.

    Args:
        path: Path to check
        project_root: Root directory of the project
        is_directory: Optional hint if path is a directory (avoids filesystem check)

    Returns:
        True if the path should be ignored
    """
    parser = create_gitignore_parser(project_root)
    return parser.is_ignored(path, is_directory=is_directory)
