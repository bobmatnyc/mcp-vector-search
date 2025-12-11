"""Unit tests for GitManager.

Tests cover all methods and error conditions using mocked subprocess calls.
No real git operations are performed to ensure test isolation and speed.
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from mcp_vector_search.core.git import (
    GitError,
    GitManager,
    GitNotAvailableError,
    GitNotRepoError,
    GitReferenceError,
)

# Fixtures


@pytest.fixture
def mock_git_available():
    """Mock git being available in PATH."""
    with patch("subprocess.run") as mock_run:
        # Mock successful git --version
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="git version 2.39.0",
            stderr="",
        )
        yield mock_run


@pytest.fixture
def mock_git_repo(tmp_path, mock_git_available):
    """Mock a valid git repository."""
    with patch("subprocess.run") as mock_run:

        def run_side_effect(cmd, *args, **kwargs):
            # Mock git --version
            if cmd == ["git", "--version"]:
                return MagicMock(returncode=0, stdout="git version 2.39.0")
            # Mock git rev-parse --git-dir (checks if git repo)
            elif cmd == ["git", "rev-parse", "--git-dir"]:
                return MagicMock(returncode=0, stdout=".git")
            # Default success
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = run_side_effect
        yield tmp_path, mock_run


# Tests for GitManager.__init__


def test_git_manager_init_success(mock_git_repo):
    """Test GitManager initialization with valid git repo."""
    repo_path, _ = mock_git_repo
    manager = GitManager(repo_path)
    assert manager.project_root == repo_path.resolve()


def test_git_manager_init_git_not_available(tmp_path):
    """Test GitManager fails when git binary not available."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError()

        with pytest.raises(GitNotAvailableError) as exc_info:
            GitManager(tmp_path)

        assert "Git binary not found" in str(exc_info.value)


def test_git_manager_init_not_a_repo(tmp_path, mock_git_available):
    """Test GitManager fails when directory is not a git repo."""
    with patch("subprocess.run") as mock_run:

        def run_side_effect(cmd, *args, **kwargs):
            if cmd == ["git", "--version"]:
                return MagicMock(returncode=0)
            elif cmd == ["git", "rev-parse", "--git-dir"]:
                raise subprocess.CalledProcessError(128, cmd)
            return MagicMock(returncode=0)

        mock_run.side_effect = run_side_effect

        with pytest.raises(GitNotRepoError) as exc_info:
            GitManager(tmp_path)

        assert "Not a git repository" in str(exc_info.value)


# Tests for is_git_available()


def test_is_git_available_success():
    """Test is_git_available returns True when git is installed."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)

        # Need to test the method directly, not via __init__
        manager = object.__new__(GitManager)
        assert manager.is_git_available() is True


def test_is_git_available_not_found():
    """Test is_git_available returns False when git not in PATH."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError()

        manager = object.__new__(GitManager)
        assert manager.is_git_available() is False


def test_is_git_available_timeout():
    """Test is_git_available returns False on timeout."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired("git", 5)

        manager = object.__new__(GitManager)
        assert manager.is_git_available() is False


# Tests for is_git_repo()


def test_is_git_repo_true(mock_git_repo):
    """Test is_git_repo returns True for valid git repo."""
    repo_path, _ = mock_git_repo
    manager = GitManager(repo_path)
    assert manager.is_git_repo() is True


def test_is_git_repo_false(mock_git_repo):
    """Test is_git_repo returns False for non-git directory."""
    repo_path, mock_run = mock_git_repo

    # Change mock to fail on rev-parse
    def run_side_effect(cmd, *args, **kwargs):
        if cmd == ["git", "--version"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--git-dir"]:
            raise subprocess.CalledProcessError(128, cmd)
        return MagicMock(returncode=0)

    mock_run.side_effect = run_side_effect

    # Initialize manager first (will succeed)
    manager = GitManager.__new__(GitManager)
    manager.project_root = repo_path

    # Now test the method
    assert manager.is_git_repo() is False


# Tests for get_changed_files()


def test_get_changed_files_empty(mock_git_repo):
    """Test get_changed_files with no changes."""
    repo_path, mock_run = mock_git_repo

    def run_side_effect(cmd, *args, **kwargs):
        if cmd == ["git", "--version"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--git-dir"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "status", "--porcelain"]:
            return MagicMock(returncode=0, stdout="", stderr="")
        return MagicMock(returncode=0)

    mock_run.side_effect = run_side_effect
    manager = GitManager(repo_path)

    changed = manager.get_changed_files()
    assert changed == []


def test_get_changed_files_modified(mock_git_repo):
    """Test get_changed_files with modified files."""
    repo_path, mock_run = mock_git_repo

    # Create test files
    test_file = repo_path / "test.py"
    test_file.write_text("print('hello')")

    def run_side_effect(cmd, *args, **kwargs):
        if cmd == ["git", "--version"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--git-dir"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "status", "--porcelain"]:
            return MagicMock(
                returncode=0,
                stdout=" M test.py\n",  # Modified in working tree
                stderr="",
            )
        return MagicMock(returncode=0)

    mock_run.side_effect = run_side_effect
    manager = GitManager(repo_path)

    changed = manager.get_changed_files()
    assert len(changed) == 1
    assert changed[0] == test_file


def test_get_changed_files_staged_and_unstaged(mock_git_repo):
    """Test get_changed_files with both staged and unstaged changes."""
    repo_path, mock_run = mock_git_repo

    # Create test files
    staged = repo_path / "staged.py"
    unstaged = repo_path / "unstaged.py"
    staged.write_text("staged")
    unstaged.write_text("unstaged")

    def run_side_effect(cmd, *args, **kwargs):
        if cmd == ["git", "--version"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--git-dir"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "status", "--porcelain"]:
            return MagicMock(
                returncode=0,
                stdout="M  staged.py\n M unstaged.py\n",
                stderr="",
            )
        return MagicMock(returncode=0)

    mock_run.side_effect = run_side_effect
    manager = GitManager(repo_path)

    changed = manager.get_changed_files()
    assert len(changed) == 2
    assert staged in changed
    assert unstaged in changed


def test_get_changed_files_with_untracked(mock_git_repo):
    """Test get_changed_files includes untracked files by default."""
    repo_path, mock_run = mock_git_repo

    # Create test file
    untracked = repo_path / "untracked.py"
    untracked.write_text("untracked")

    def run_side_effect(cmd, *args, **kwargs):
        if cmd == ["git", "--version"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--git-dir"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "status", "--porcelain"]:
            return MagicMock(
                returncode=0,
                stdout="?? untracked.py\n",
                stderr="",
            )
        return MagicMock(returncode=0)

    mock_run.side_effect = run_side_effect
    manager = GitManager(repo_path)

    changed = manager.get_changed_files(include_untracked=True)
    assert len(changed) == 1
    assert untracked in changed


def test_get_changed_files_exclude_untracked(mock_git_repo):
    """Test get_changed_files excludes untracked when requested."""
    repo_path, mock_run = mock_git_repo

    def run_side_effect(cmd, *args, **kwargs):
        if cmd == ["git", "--version"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--git-dir"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "status", "--porcelain"]:
            return MagicMock(
                returncode=0,
                stdout="?? untracked.py\n",
                stderr="",
            )
        return MagicMock(returncode=0)

    mock_run.side_effect = run_side_effect
    manager = GitManager(repo_path)

    changed = manager.get_changed_files(include_untracked=False)
    assert changed == []


def test_get_changed_files_renamed(mock_git_repo):
    """Test get_changed_files handles renamed files correctly."""
    repo_path, mock_run = mock_git_repo

    # Create new file (the renamed target)
    new_file = repo_path / "new.py"
    new_file.write_text("renamed")

    def run_side_effect(cmd, *args, **kwargs):
        if cmd == ["git", "--version"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--git-dir"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "status", "--porcelain"]:
            return MagicMock(
                returncode=0,
                stdout="R  old.py -> new.py\n",
                stderr="",
            )
        return MagicMock(returncode=0)

    mock_run.side_effect = run_side_effect
    manager = GitManager(repo_path)

    changed = manager.get_changed_files()
    assert len(changed) == 1
    assert changed[0] == new_file


def test_get_changed_files_deleted_excluded(mock_git_repo):
    """Test get_changed_files excludes deleted files."""
    repo_path, mock_run = mock_git_repo

    def run_side_effect(cmd, *args, **kwargs):
        if cmd == ["git", "--version"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--git-dir"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "status", "--porcelain"]:
            return MagicMock(
                returncode=0,
                stdout=" D deleted.py\n",
                stderr="",
            )
        return MagicMock(returncode=0)

    mock_run.side_effect = run_side_effect
    manager = GitManager(repo_path)

    changed = manager.get_changed_files()
    assert changed == []


def test_get_changed_files_nonexistent_skipped(mock_git_repo):
    """Test get_changed_files skips files that don't exist."""
    repo_path, mock_run = mock_git_repo

    def run_side_effect(cmd, *args, **kwargs):
        if cmd == ["git", "--version"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--git-dir"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "status", "--porcelain"]:
            return MagicMock(
                returncode=0,
                stdout=" M nonexistent.py\n",
                stderr="",
            )
        return MagicMock(returncode=0)

    mock_run.side_effect = run_side_effect
    manager = GitManager(repo_path)

    changed = manager.get_changed_files()
    assert changed == []


def test_get_changed_files_error(mock_git_repo):
    """Test get_changed_files raises GitError on command failure."""
    repo_path, mock_run = mock_git_repo

    def run_side_effect(cmd, *args, **kwargs):
        if cmd == ["git", "--version"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--git-dir"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "status", "--porcelain"]:
            raise subprocess.CalledProcessError(1, cmd, stderr="fatal: error")
        return MagicMock(returncode=0)

    mock_run.side_effect = run_side_effect
    manager = GitManager(repo_path)

    with pytest.raises(GitError) as exc_info:
        manager.get_changed_files()

    assert "Failed to get changed files" in str(exc_info.value)


def test_get_changed_files_timeout(mock_git_repo):
    """Test get_changed_files raises GitError on timeout."""
    repo_path, mock_run = mock_git_repo

    def run_side_effect(cmd, *args, **kwargs):
        if cmd == ["git", "--version"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--git-dir"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "status", "--porcelain"]:
            raise subprocess.TimeoutExpired(cmd, 10)
        return MagicMock(returncode=0)

    mock_run.side_effect = run_side_effect
    manager = GitManager(repo_path)

    with pytest.raises(GitError) as exc_info:
        manager.get_changed_files()

    assert "timed out" in str(exc_info.value)


# Tests for get_diff_files()


def test_get_diff_files_success(mock_git_repo):
    """Test get_diff_files with valid baseline."""
    repo_path, mock_run = mock_git_repo

    # Create test file
    diff_file = repo_path / "changed.py"
    diff_file.write_text("changed")

    def run_side_effect(cmd, *args, **kwargs):
        if cmd == ["git", "--version"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--git-dir"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--verify", "main"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "diff", "--name-only", "main"]:
            return MagicMock(returncode=0, stdout="changed.py\n", stderr="")
        return MagicMock(returncode=0)

    mock_run.side_effect = run_side_effect
    manager = GitManager(repo_path)

    diff_files = manager.get_diff_files("main")
    assert len(diff_files) == 1
    assert diff_files[0] == diff_file


def test_get_diff_files_multiple(mock_git_repo):
    """Test get_diff_files with multiple changed files."""
    repo_path, mock_run = mock_git_repo

    # Create test files
    file1 = repo_path / "file1.py"
    file2 = repo_path / "file2.py"
    file1.write_text("file1")
    file2.write_text("file2")

    def run_side_effect(cmd, *args, **kwargs):
        if cmd == ["git", "--version"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--git-dir"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--verify", "main"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "diff", "--name-only", "main"]:
            return MagicMock(
                returncode=0,
                stdout="file1.py\nfile2.py\n",
                stderr="",
            )
        return MagicMock(returncode=0)

    mock_run.side_effect = run_side_effect
    manager = GitManager(repo_path)

    diff_files = manager.get_diff_files("main")
    assert len(diff_files) == 2
    assert file1 in diff_files
    assert file2 in diff_files


def test_get_diff_files_baseline_not_found_fallback(mock_git_repo):
    """Test get_diff_files falls back to master when main not found."""
    repo_path, mock_run = mock_git_repo

    # Create test file
    diff_file = repo_path / "changed.py"
    diff_file.write_text("changed")

    def run_side_effect(cmd, *args, **kwargs):
        if cmd == ["git", "--version"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--git-dir"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--verify", "main"]:
            raise subprocess.CalledProcessError(1, cmd)
        elif cmd == ["git", "rev-parse", "--verify", "master"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "diff", "--name-only", "master"]:
            return MagicMock(returncode=0, stdout="changed.py\n", stderr="")
        return MagicMock(returncode=0)

    mock_run.side_effect = run_side_effect
    manager = GitManager(repo_path)

    diff_files = manager.get_diff_files("main")
    assert len(diff_files) == 1


def test_get_diff_files_no_valid_baseline(mock_git_repo):
    """Test get_diff_files raises GitReferenceError when no baseline found."""
    repo_path, mock_run = mock_git_repo

    def run_side_effect(cmd, *args, **kwargs):
        if cmd == ["git", "--version"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--git-dir"]:
            return MagicMock(returncode=0)
        elif cmd[0:3] == ["git", "rev-parse", "--verify"]:
            raise subprocess.CalledProcessError(1, cmd)
        return MagicMock(returncode=0)

    mock_run.side_effect = run_side_effect
    manager = GitManager(repo_path)

    with pytest.raises(GitReferenceError) as exc_info:
        manager.get_diff_files("nonexistent")

    assert "not found" in str(exc_info.value)


def test_get_diff_files_empty(mock_git_repo):
    """Test get_diff_files with no differences."""
    repo_path, mock_run = mock_git_repo

    def run_side_effect(cmd, *args, **kwargs):
        if cmd == ["git", "--version"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--git-dir"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--verify", "main"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "diff", "--name-only", "main"]:
            return MagicMock(returncode=0, stdout="", stderr="")
        return MagicMock(returncode=0)

    mock_run.side_effect = run_side_effect
    manager = GitManager(repo_path)

    diff_files = manager.get_diff_files("main")
    assert diff_files == []


def test_get_diff_files_error(mock_git_repo):
    """Test get_diff_files raises GitError on command failure."""
    repo_path, mock_run = mock_git_repo

    def run_side_effect(cmd, *args, **kwargs):
        if cmd == ["git", "--version"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--git-dir"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--verify", "main"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "diff", "--name-only", "main"]:
            raise subprocess.CalledProcessError(1, cmd, stderr="fatal: error")
        return MagicMock(returncode=0)

    mock_run.side_effect = run_side_effect
    manager = GitManager(repo_path)

    with pytest.raises(GitError) as exc_info:
        manager.get_diff_files("main")

    assert "Failed to get diff files" in str(exc_info.value)


# Tests for ref_exists()


def test_ref_exists_true(mock_git_repo):
    """Test ref_exists returns True for valid reference."""
    repo_path, mock_run = mock_git_repo

    def run_side_effect(cmd, *args, **kwargs):
        if cmd == ["git", "--version"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--git-dir"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--verify", "main"]:
            return MagicMock(returncode=0)
        return MagicMock(returncode=0)

    mock_run.side_effect = run_side_effect
    manager = GitManager(repo_path)

    assert manager.ref_exists("main") is True


def test_ref_exists_false(mock_git_repo):
    """Test ref_exists returns False for invalid reference."""
    repo_path, mock_run = mock_git_repo

    def run_side_effect(cmd, *args, **kwargs):
        if cmd == ["git", "--version"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--git-dir"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--verify", "nonexistent"]:
            raise subprocess.CalledProcessError(1, cmd)
        return MagicMock(returncode=0)

    mock_run.side_effect = run_side_effect
    manager = GitManager(repo_path)

    assert manager.ref_exists("nonexistent") is False


# Tests for get_current_branch()


def test_get_current_branch_success(mock_git_repo):
    """Test get_current_branch returns branch name."""
    repo_path, mock_run = mock_git_repo

    def run_side_effect(cmd, *args, **kwargs):
        if cmd == ["git", "--version"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--git-dir"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return MagicMock(returncode=0, stdout="main\n", stderr="")
        return MagicMock(returncode=0)

    mock_run.side_effect = run_side_effect
    manager = GitManager(repo_path)

    branch = manager.get_current_branch()
    assert branch == "main"


def test_get_current_branch_detached_head(mock_git_repo):
    """Test get_current_branch returns None for detached HEAD."""
    repo_path, mock_run = mock_git_repo

    def run_side_effect(cmd, *args, **kwargs):
        if cmd == ["git", "--version"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--git-dir"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return MagicMock(returncode=0, stdout="HEAD\n", stderr="")
        return MagicMock(returncode=0)

    mock_run.side_effect = run_side_effect
    manager = GitManager(repo_path)

    branch = manager.get_current_branch()
    assert branch is None


def test_get_current_branch_error(mock_git_repo):
    """Test get_current_branch returns None on error."""
    repo_path, mock_run = mock_git_repo

    def run_side_effect(cmd, *args, **kwargs):
        if cmd == ["git", "--version"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--git-dir"]:
            return MagicMock(returncode=0)
        elif cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            raise subprocess.CalledProcessError(1, cmd)
        return MagicMock(returncode=0)

    mock_run.side_effect = run_side_effect
    manager = GitManager(repo_path)

    branch = manager.get_current_branch()
    assert branch is None
