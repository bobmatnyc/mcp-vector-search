"""Git integration for diff-aware analysis.

This module provides the GitManager class for detecting changed files in a git
repository, enabling diff-aware analysis that focuses only on modified code.

Design Decisions:
    - Uses subprocess to call git commands (standard approach, no dependencies)
    - Returns absolute Paths for consistency with rest of codebase
    - Robust error handling with custom exceptions
    - Supports both uncommitted changes and baseline comparisons

Performance:
    - Git operations are typically fast (<100ms for most repos)
    - File path resolution is O(n) where n is number of changed files
    - Subprocess overhead is minimal compared to parsing/analysis time

Error Handling:
    All git operations are wrapped with proper exception handling:
    - GitNotAvailableError: Git binary not found in PATH
    - GitNotRepoError: Not a git repository
    - GitReferenceError: Invalid branch/commit reference
    - GitError: General git operation failures
"""

import subprocess
from pathlib import Path

from loguru import logger


class GitError(Exception):
    """Base exception for git-related errors."""

    pass


class GitNotAvailableError(GitError):
    """Git binary is not available in PATH."""

    pass


class GitNotRepoError(GitError):
    """Directory is not a git repository."""

    pass


class GitReferenceError(GitError):
    """Git reference (branch, tag, commit) does not exist."""

    pass


class GitManager:
    """Manage git operations for diff-aware analysis.

    This class provides methods to detect changed files in a git repository,
    supporting both uncommitted changes and baseline comparisons.

    Design Pattern: Simple wrapper around git commands with error handling.
    No caching to ensure always-fresh results (git is fast enough).

    Example:
        >>> manager = GitManager(Path("/path/to/repo"))
        >>> changed = manager.get_changed_files()
        >>> print(f"Found {len(changed)} changed files")
    """

    def __init__(self, project_root: Path):
        """Initialize git manager.

        Args:
            project_root: Root directory of the project

        Raises:
            GitNotAvailableError: If git binary is not available
            GitNotRepoError: If project_root is not a git repository
        """
        self.project_root = project_root.resolve()

        # Check git availability first
        if not self.is_git_available():
            raise GitNotAvailableError(
                "Git binary not found. Install git or run without --changed-only"
            )

        # Check if this is a git repository
        if not self.is_git_repo():
            raise GitNotRepoError(
                f"Not a git repository: {self.project_root}. "
                "Initialize git with: git init"
            )

    def is_git_available(self) -> bool:
        """Check if git command is available in PATH.

        Returns:
            True if git is available, False otherwise

        Performance: O(1), cached by OS after first call
        """
        try:
            subprocess.run(  # nosec B607 - git is intentionally called via PATH
                ["git", "--version"],
                capture_output=True,
                check=True,
                timeout=5,
            )
            return True
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            return False

    def is_git_repo(self) -> bool:
        """Check if project directory is a git repository.

        Returns:
            True if directory is a git repository

        Performance: O(1), filesystem check
        """
        try:
            subprocess.run(  # nosec B607 - git is intentionally called via PATH
                ["git", "rev-parse", "--git-dir"],
                cwd=self.project_root,
                capture_output=True,
                check=True,
                timeout=5,
            )
            # Successfully ran, so it's a git repo
            return True
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            return False

    def get_changed_files(self, include_untracked: bool = True) -> list[Path]:
        """Get list of changed files in working directory.

        Detects uncommitted changes using `git status --porcelain`.
        Includes both staged and unstaged modifications.

        Args:
            include_untracked: Include untracked files (default: True)

        Returns:
            List of changed file paths (absolute paths)

        Raises:
            GitError: If git status command fails

        Performance: O(n) where n is number of files in working tree

        Git Status Format:
            XY filename
            X = index status (staged)
            Y = working tree status (unstaged)
            ?? = untracked
            D = deleted
            R  old -> new = renamed

        Example:
            >>> manager = GitManager(Path.cwd())
            >>> changed = manager.get_changed_files()
            >>> for file in changed:
            ...     print(f"Modified: {file}")
        """
        cmd = ["git", "status", "--porcelain"]

        try:
            result = subprocess.run(  # nosec B607 - git is intentionally called via PATH
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )

            changed_files = []
            for line in result.stdout.splitlines():
                if not line.strip():
                    continue

                # Parse git status porcelain format
                # Format: XY filename (X=index, Y=working tree)
                status = line[:2]
                filename = line[3:].strip()

                # Handle renamed files: "R  old -> new"
                if " -> " in filename:
                    filename = filename.split(" -> ")[1]

                # Skip deleted files (they don't exist to analyze)
                if "D" in status:
                    logger.debug(f"Skipping deleted file: {filename}")
                    continue

                # Skip untracked if not requested
                if not include_untracked and status.startswith("??"):
                    logger.debug(f"Skipping untracked file: {filename}")
                    continue

                # Convert to absolute path and verify existence
                file_path = self.project_root / filename
                if file_path.exists() and file_path.is_file():
                    changed_files.append(file_path)
                else:
                    logger.debug(f"Skipping non-existent file: {file_path}")

            logger.info(
                f"Found {len(changed_files)} changed files "
                f"(untracked={'included' if include_untracked else 'excluded'})"
            )
            return changed_files

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else "Unknown error"
            logger.error(f"Git status failed: {error_msg}")
            raise GitError(f"Failed to get changed files: {error_msg}")
        except subprocess.TimeoutExpired:
            logger.error("Git status command timed out")
            raise GitError("Git status command timed out after 10 seconds")

    def get_diff_files(self, baseline: str = "main") -> list[Path]:
        """Get list of files that differ from baseline branch.

        Compares current branch against baseline using `git diff --name-only`.

        Args:
            baseline: Baseline branch or commit (default: "main")

        Returns:
            List of changed file paths (absolute paths)

        Raises:
            GitReferenceError: If baseline reference doesn't exist
            GitError: If git diff command fails

        Performance: O(n) where n is number of files in diff

        Baseline Fallback Strategy:
            1. Try requested baseline (e.g., "main")
            2. If not found, try "master"
            3. If not found, try "develop"
            4. If not found, try "HEAD~1"
            5. If still not found, raise GitReferenceError

        Example:
            >>> manager = GitManager(Path.cwd())
            >>> diff_files = manager.get_diff_files("main")
            >>> print(f"Changed vs main: {len(diff_files)} files")
        """
        # First, check if baseline exists
        if not self.ref_exists(baseline):
            # Try common alternatives
            alternatives = ["master", "develop", "HEAD~1"]
            for alt in alternatives:
                if self.ref_exists(alt):
                    logger.warning(
                        f"Baseline '{baseline}' not found, using '{alt}' instead"
                    )
                    baseline = alt
                    break
            else:
                raise GitReferenceError(
                    f"Baseline '{baseline}' not found. "
                    f"Try: main, master, develop, or HEAD~1. "
                    f"Check available branches with: git branch -a"
                )

        # Get list of changed files
        cmd = ["git", "diff", "--name-only", baseline]

        try:
            result = subprocess.run(  # nosec B607 - git is intentionally called via PATH
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )

            changed_files = []
            for line in result.stdout.splitlines():
                if not line.strip():
                    continue

                # Convert to absolute path and verify existence
                file_path = self.project_root / line.strip()
                if file_path.exists() and file_path.is_file():
                    changed_files.append(file_path)
                else:
                    # File may have been deleted in current branch
                    logger.debug(f"Skipping non-existent diff file: {file_path}")

            logger.info(f"Found {len(changed_files)} files different from {baseline}")
            return changed_files

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else "Unknown error"
            logger.error(f"Git diff failed: {error_msg}")
            raise GitError(f"Failed to get diff files: {error_msg}")
        except subprocess.TimeoutExpired:
            logger.error("Git diff command timed out")
            raise GitError("Git diff command timed out after 10 seconds")

    def ref_exists(self, ref: str) -> bool:
        """Check if a git ref (branch, tag, commit) exists.

        Uses `git rev-parse --verify` to check reference validity.

        Args:
            ref: Git reference to check (branch, tag, commit hash)

        Returns:
            True if ref exists and is valid

        Performance: O(1), fast git operation

        Example:
            >>> manager = GitManager(Path.cwd())
            >>> if manager.ref_exists("main"):
            ...     print("Main branch exists")
        """
        cmd = ["git", "rev-parse", "--verify", ref]

        try:
            subprocess.run(  # nosec B607 - git is intentionally called via PATH
                cmd,
                cwd=self.project_root,
                capture_output=True,
                check=True,
                timeout=5,
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    def get_current_branch(self) -> str | None:
        """Get name of current branch.

        Returns:
            Branch name or None if detached HEAD

        Performance: O(1), fast git operation

        Example:
            >>> manager = GitManager(Path.cwd())
            >>> branch = manager.get_current_branch()
            >>> if branch:
            ...     print(f"Current branch: {branch}")
            ... else:
            ...     print("Detached HEAD state")
        """
        cmd = ["git", "rev-parse", "--abbrev-ref", "HEAD"]

        try:
            result = subprocess.run(  # nosec B607 - git is intentionally called via PATH
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )

            branch = result.stdout.strip()
            # "HEAD" means detached HEAD state
            return branch if branch != "HEAD" else None

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
