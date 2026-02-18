"""Git blame integration for tracking code authorship.

This module provides utilities to extract git blame information for code chunks,
showing who last modified each line and when.
"""

import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger


@dataclass
class BlameInfo:
    """Git blame information for a code region."""

    author: str  # Author name
    timestamp: str  # ISO 8601 timestamp
    commit_hash: str  # Full commit hash


class GitBlameCache:
    """Caches git blame results per-file to avoid repeated git calls.

    Key optimization: Call git blame once per file, not per chunk.
    Stores mapping of line_number -> BlameInfo.
    """

    def __init__(self, repo_root: Path):
        """Initialize blame cache.

        Args:
            repo_root: Root directory of git repository
        """
        self.repo_root = repo_root
        self._cache: dict[str, dict[int, BlameInfo]] = {}

    def get_blame_for_range(
        self, file_path: Path, start_line: int, end_line: int
    ) -> BlameInfo | None:
        """Get git blame info for a line range.

        Returns blame info for the MOST RECENT modification in the range.
        This tells us who last touched this code.

        Args:
            file_path: Absolute path to file
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (1-indexed)

        Returns:
            BlameInfo for most recent change, or None if not in git/error
        """
        try:
            # Make path relative to repo root
            try:
                rel_path = file_path.relative_to(self.repo_root)
            except ValueError:
                # File not in repo
                logger.debug(f"File not in repository: {file_path}")
                return None

            # Get or populate cache for this file
            file_key = str(rel_path)
            if file_key not in self._cache:
                self._populate_file_blame(rel_path)

            # Find most recent blame in range
            file_blame = self._cache.get(file_key, {})
            if not file_blame:
                return None

            # Collect all blame entries in range
            range_blames: list[BlameInfo] = []
            for line_num in range(start_line, end_line + 1):
                if line_num in file_blame:
                    range_blames.append(file_blame[line_num])

            if not range_blames:
                return None

            # Return most recent (latest timestamp)
            return max(range_blames, key=lambda b: b.timestamp)

        except Exception as e:
            logger.debug(
                f"Failed to get blame for {file_path}:{start_line}-{end_line}: {e}"
            )
            return None

    def _populate_file_blame(self, rel_path: Path) -> None:
        """Populate cache with git blame data for entire file.

        Uses `git blame --porcelain` for structured output.

        Args:
            rel_path: Path relative to repo root
        """
        file_key = str(rel_path)

        try:
            # Run git blame in porcelain format
            result = subprocess.run(  # nosec B603, B607 - safe git command
                [
                    "git",
                    "blame",
                    "--porcelain",
                    str(rel_path),
                ],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=False,  # Don't raise on non-zero exit
                timeout=10,  # Prevent hanging on large files
            )

            if result.returncode != 0:
                # File not in git, binary file, or other error
                logger.debug(
                    f"git blame failed for {rel_path}: {result.stderr.strip()}"
                )
                self._cache[file_key] = {}
                return

            # Parse porcelain output
            file_blame = self._parse_porcelain_blame(result.stdout)
            self._cache[file_key] = file_blame

            logger.debug(f"Cached blame for {rel_path}: {len(file_blame)} lines")

        except subprocess.TimeoutExpired:
            logger.warning(f"git blame timeout for {rel_path}")
            self._cache[file_key] = {}
        except Exception as e:
            logger.debug(f"Failed to populate blame for {rel_path}: {e}")
            self._cache[file_key] = {}

    def _parse_porcelain_blame(self, output: str) -> dict[int, BlameInfo]:
        """Parse git blame --porcelain output.

        Format:
        <commit_hash> <original_line> <final_line> <num_lines>
        author <author>
        author-time <timestamp>
        ...
        \t<line content>

        Args:
            output: Raw porcelain output from git blame

        Returns:
            Mapping of line_number (1-indexed) -> BlameInfo
        """
        lines = output.split("\n")
        blame_map: dict[int, BlameInfo] = {}

        current_commit = None
        current_author = None
        current_time = None
        current_line_num = None

        for line in lines:
            if not line:
                continue

            # Commit line: <hash> <orig_line> <final_line> <num_lines>
            if not line.startswith("\t") and not line.startswith(
                (
                    "author ",
                    "author-time ",
                    "committer ",
                    "summary ",
                    "filename ",
                    "boundary",
                    "previous ",
                )
            ):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        current_commit = parts[0]
                        current_line_num = int(parts[2])  # final_line is 3rd field
                        # Reset metadata (will be populated by subsequent lines)
                        current_author = None
                        current_time = None
                    except ValueError:
                        # Not a commit line (might be a filename or other metadata)
                        continue

            # Author line
            elif line.startswith("author "):
                current_author = line[7:].strip()  # Skip "author "

            # Timestamp line
            elif line.startswith("author-time "):
                try:
                    timestamp_unix = int(line[12:].strip())
                    # Convert to ISO 8601
                    current_time = datetime.fromtimestamp(
                        timestamp_unix, tz=UTC
                    ).isoformat()
                except (ValueError, OSError):
                    current_time = None

            # Tab-prefixed line = actual code content (marks end of blame block)
            elif line.startswith("\t"):
                # Store blame info if we have all required fields
                if (
                    current_commit
                    and current_author
                    and current_time
                    and current_line_num
                ):
                    blame_map[current_line_num] = BlameInfo(
                        author=current_author,
                        timestamp=current_time,
                        commit_hash=current_commit,
                    )

        return blame_map

    def clear(self) -> None:
        """Clear the blame cache."""
        self._cache = {}
