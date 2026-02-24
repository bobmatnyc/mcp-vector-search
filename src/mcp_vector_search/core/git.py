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

    def get_diff(
        self, base_ref: str = "main", head_ref: str = "HEAD", context_lines: int = 3
    ) -> str:
        """Get unified diff between two git refs.

        Args:
            base_ref: Base reference (e.g., "main", "develop")
            head_ref: Head reference (default: "HEAD")
            context_lines: Number of context lines around changes (default: 3)

        Returns:
            Unified diff text

        Raises:
            GitReferenceError: If references don't exist
            GitError: If git diff command fails

        Example:
            >>> manager = GitManager(Path.cwd())
            >>> diff = manager.get_diff("main", "HEAD")
            >>> print(diff)
        """
        # Verify refs exist
        if not self.ref_exists(base_ref):
            raise GitReferenceError(f"Base ref '{base_ref}' does not exist")
        if not self.ref_exists(head_ref):
            raise GitReferenceError(f"Head ref '{head_ref}' does not exist")

        cmd = [
            "git",
            "diff",
            f"-U{context_lines}",  # Unified diff with N context lines
            f"{base_ref}...{head_ref}",  # Three-dot diff (merge base)
        ]

        try:
            result = subprocess.run(  # nosec B607
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )

            return result.stdout

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else "Unknown error"
            logger.error(f"Git diff failed: {error_msg}")
            raise GitError(f"Failed to get diff: {error_msg}")
        except subprocess.TimeoutExpired:
            logger.error("Git diff command timed out")
            raise GitError("Git diff command timed out after 30 seconds")

    def get_file_at_ref(self, file_path: str, ref: str) -> str | None:
        """Get file content at a specific git reference.

        Args:
            file_path: Path to file relative to project root
            ref: Git reference (branch, tag, commit)

        Returns:
            File content as string, or None if file doesn't exist at ref

        Example:
            >>> manager = GitManager(Path.cwd())
            >>> old_content = manager.get_file_at_ref("src/auth.py", "main")
        """
        cmd = ["git", "show", f"{ref}:{file_path}"]

        try:
            result = subprocess.run(  # nosec B607
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )

            return result.stdout

        except subprocess.CalledProcessError:
            # File doesn't exist at this ref (could be new file or deleted)
            return None
        except subprocess.TimeoutExpired:
            logger.error(f"Git show timed out for {file_path} at {ref}")
            return None

    def parse_diff_stats(self, diff_text: str) -> dict[str, tuple[int, int]]:
        """Parse diff text to extract per-file addition/deletion stats.

        Args:
            diff_text: Unified diff text from git diff

        Returns:
            Dictionary mapping file paths to (additions, deletions) tuples

        Example:
            >>> manager = GitManager(Path.cwd())
            >>> diff = manager.get_diff("main", "HEAD")
            >>> stats = manager.parse_diff_stats(diff)
            >>> stats["src/auth.py"]
            (15, 3)  # 15 additions, 3 deletions
        """
        stats: dict[str, tuple[int, int]] = {}
        current_file: str | None = None
        additions = 0
        deletions = 0

        for line in diff_text.splitlines():
            # Detect file header: diff --git a/path b/path
            if line.startswith("diff --git "):
                # Save previous file stats
                if current_file:
                    stats[current_file] = (additions, deletions)

                # Parse new file path (from b/ path)
                parts = line.split()
                if len(parts) >= 4:
                    # b/path is the new file path
                    current_file = parts[3][2:]  # Remove "b/" prefix
                    additions = 0
                    deletions = 0

            # Count additions/deletions
            elif current_file:
                if line.startswith("+") and not line.startswith("+++"):
                    additions += 1
                elif line.startswith("-") and not line.startswith("---"):
                    deletions += 1

        # Save last file stats
        if current_file:
            stats[current_file] = (additions, deletions)

        return stats

    def get_staged_diff(self) -> str:
        """Get the diff of staged changes.

        Returns:
            Unified diff text of staged changes

        Raises:
            GitError: If git operation fails
        """
        try:
            result = subprocess.run(  # nosec B607 - git is intentionally called via PATH
                ["git", "diff", "--staged"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                # No error for empty diff (no staged changes)
                if result.returncode == 128 and "no changes" in result.stderr.lower():
                    return ""
                raise GitError(f"git diff --staged failed: {result.stderr}")

            return result.stdout

        except subprocess.TimeoutExpired:
            raise GitError("git diff --staged timed out")
        except FileNotFoundError:
            raise GitNotAvailableError("git binary not found")

    def get_staged_files(self) -> list[Path]:
        """Get list of staged files.

        Returns:
            List of staged file paths relative to project root

        Raises:
            GitError: If git operation fails
        """
        try:
            result = subprocess.run(  # nosec B607 - git is intentionally called via PATH
                ["git", "diff", "--staged", "--name-only"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                # No error for empty list (no staged changes)
                if "no changes" in result.stderr.lower():
                    return []
                raise GitError(f"git diff --staged --name-only failed: {result.stderr}")

            file_paths = []
            for line in result.stdout.strip().split("\n"):
                if line.strip():  # Skip empty lines
                    file_path = self.project_root / line.strip()
                    if file_path.exists():  # Only include existing files
                        file_paths.append(file_path)

            logger.debug(f"Found {len(file_paths)} staged files")
            return file_paths

        except subprocess.TimeoutExpired:
            raise GitError("git diff --staged --name-only timed out")
        except FileNotFoundError:
            raise GitNotAvailableError("git binary not found")

    def get_staged_diff_stats(self) -> dict[str, int]:
        """Get statistics of staged changes.

        Returns:
            Dictionary with files_changed, insertions, deletions

        Raises:
            GitError: If git operation fails
        """
        try:
            result = subprocess.run(  # nosec B607 - git is intentionally called via PATH
                ["git", "diff", "--staged", "--numstat"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                # No error for empty stats (no staged changes)
                if "no changes" in result.stderr.lower():
                    return {"files_changed": 0, "insertions": 0, "deletions": 0}
                raise GitError(f"git diff --staged --numstat failed: {result.stderr}")

            total_insertions = 0
            total_deletions = 0
            files_changed = 0

            for line in result.stdout.strip().split("\n"):
                if line.strip():  # Skip empty lines
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        try:
                            insertions = int(parts[0]) if parts[0] != "-" else 0
                            deletions = int(parts[1]) if parts[1] != "-" else 0
                            total_insertions += insertions
                            total_deletions += deletions
                            files_changed += 1
                        except ValueError:
                            # Skip lines that don't have numeric stats (binary files)
                            files_changed += 1

            return {
                "files_changed": files_changed,
                "insertions": total_insertions,
                "deletions": total_deletions,
            }

        except subprocess.TimeoutExpired:
            raise GitError("git diff --staged --numstat timed out")
        except FileNotFoundError:
            raise GitNotAvailableError("git binary not found")

    def get_file_diff(self, file_path: str, staged: bool = False) -> str:
        """Get diff for a specific file.

        Args:
            file_path: Path to the file (relative to project root)
            staged: If True, get staged changes; if False, get working directory changes

        Returns:
            Unified diff text for the file

        Raises:
            GitError: If git operation fails
        """
        try:
            cmd = ["git", "diff"]
            if staged:
                cmd.append("--staged")
            cmd.append("--")
            cmd.append(file_path)

            result = subprocess.run(  # nosec B607 - git is intentionally called via PATH
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                # No error for empty diff (no changes)
                if "no changes" in result.stderr.lower():
                    return ""
                raise GitError(f"git diff failed for {file_path}: {result.stderr}")

            return result.stdout

        except subprocess.TimeoutExpired:
            raise GitError(f"git diff timed out for {file_path}")
        except FileNotFoundError:
            raise GitNotAvailableError("git binary not found")
