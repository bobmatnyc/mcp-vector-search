"""Tests for git integration in the analyze CLI command."""

from unittest.mock import patch

import pytest

from mcp_vector_search.cli.commands.analyze import _find_analyzable_files
from mcp_vector_search.core.git import (
    GitManager,
    GitNotAvailableError,
    GitNotRepoError,
)
from mcp_vector_search.parsers.registry import ParserRegistry


class TestAnalyzeGitIntegration:
    """Tests for git integration in analyze command."""

    @pytest.mark.asyncio
    async def test_find_analyzable_files_with_git_filter(self, tmp_path):
        """Test finding files filtered by git changed files."""
        # Create test files
        file1 = tmp_path / "changed.py"
        file1.write_text("def foo(): pass")
        file2 = tmp_path / "unchanged.py"
        file2.write_text("def bar(): pass")
        file3 = tmp_path / "changed.js"
        file3.write_text("function baz() {}")

        registry = ParserRegistry()

        # Simulate git changed files (only changed.py and changed.js)
        git_changed_files = [file1, file3]

        files = _find_analyzable_files(
            tmp_path, None, None, registry, git_changed_files
        )

        # Should only find the changed files
        assert len(files) == 2
        assert file1 in files
        assert file3 in files
        assert file2 not in files

    @pytest.mark.asyncio
    async def test_find_analyzable_files_git_filter_with_language(self, tmp_path):
        """Test git filter combined with language filter."""
        # Create test files
        file1 = tmp_path / "changed.py"
        file1.write_text("def foo(): pass")
        file2 = tmp_path / "changed.js"
        file2.write_text("function bar() {}")

        registry = ParserRegistry()

        # Simulate git changed files
        git_changed_files = [file1, file2]

        # Filter for Python only
        files = _find_analyzable_files(
            tmp_path, "python", None, registry, git_changed_files
        )

        # Should only find Python files
        assert len(files) == 1
        assert file1 in files
        assert file2 not in files

    @pytest.mark.asyncio
    async def test_find_analyzable_files_git_filter_with_path(self, tmp_path):
        """Test git filter combined with path filter."""
        # Create directory structure
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        test_dir = tmp_path / "tests"
        test_dir.mkdir()

        # Create test files
        file1 = src_dir / "changed.py"
        file1.write_text("def foo(): pass")
        file2 = test_dir / "changed_test.py"
        file2.write_text("def test_bar(): pass")

        registry = ParserRegistry()

        # Simulate git changed files (both files)
        git_changed_files = [file1, file2]

        # Filter for src directory only
        files = _find_analyzable_files(
            tmp_path, None, src_dir, registry, git_changed_files
        )

        # Should only find files in src directory
        assert len(files) == 1
        assert file1 in files
        assert file2 not in files

    @pytest.mark.asyncio
    async def test_find_analyzable_files_git_filter_unsupported_extension(
        self, tmp_path
    ):
        """Test that unsupported file types are filtered out from git changes."""
        # Create test files
        file1 = tmp_path / "changed.py"
        file1.write_text("def foo(): pass")
        file2 = tmp_path / "changed.txt"
        file2.write_text("This is text")
        file3 = tmp_path / "changed.xyz"
        file3.write_text("Unknown format")

        registry = ParserRegistry()

        # Simulate git changed files including unsupported types
        git_changed_files = [file1, file2, file3]

        files = _find_analyzable_files(
            tmp_path, None, None, registry, git_changed_files
        )

        # Should only find supported file types
        # Note: .txt might be supported by TextParser, so we check for .xyz exclusion
        assert file1 in files
        assert file3 not in files  # .xyz is definitely not supported

    @pytest.mark.asyncio
    async def test_find_analyzable_files_git_filter_empty(self, tmp_path):
        """Test behavior when git changed files list is empty."""
        # Create test files
        file1 = tmp_path / "file.py"
        file1.write_text("def foo(): pass")

        registry = ParserRegistry()

        # Simulate empty git changed files
        git_changed_files = []

        files = _find_analyzable_files(
            tmp_path, None, None, registry, git_changed_files
        )

        # Should return empty list
        assert len(files) == 0

    @pytest.mark.asyncio
    async def test_find_analyzable_files_no_git_filter_fallback(self, tmp_path):
        """Test fallback to full directory scan when no git filter."""
        # Create test files
        file1 = tmp_path / "file1.py"
        file1.write_text("def foo(): pass")
        file2 = tmp_path / "file2.py"
        file2.write_text("def bar(): pass")

        registry = ParserRegistry()

        # No git filter (None)
        files = _find_analyzable_files(tmp_path, None, None, registry, None)

        # Should find all files
        assert len(files) == 2
        assert file1 in files
        assert file2 in files

    @pytest.mark.asyncio
    async def test_find_analyzable_files_git_filter_specific_file(self, tmp_path):
        """Test git filter with path_filter pointing to specific file."""
        # Create test files
        file1 = tmp_path / "changed1.py"
        file1.write_text("def foo(): pass")
        file2 = tmp_path / "changed2.py"
        file2.write_text("def bar(): pass")

        registry = ParserRegistry()

        # Simulate git changed files
        git_changed_files = [file1, file2]

        # Filter for specific file only
        files = _find_analyzable_files(
            tmp_path, None, file1, registry, git_changed_files
        )

        # Should only find the specified file
        assert len(files) == 1
        assert file1 in files
        assert file2 not in files


class TestGitManagerErrorHandling:
    """Tests for GitManager error handling in analyze command."""

    def test_git_not_available_error(self, tmp_path):
        """Test error handling when git is not available."""
        with patch("subprocess.run") as mock_run:
            # Simulate git not available
            mock_run.side_effect = FileNotFoundError()

            with pytest.raises(GitNotAvailableError):
                GitManager(tmp_path)

    def test_git_not_repo_error(self, tmp_path):
        """Test error handling when directory is not a git repo."""
        from subprocess import CalledProcessError

        with patch.object(GitManager, "is_git_available", return_value=True):
            with patch("subprocess.run") as mock_run:
                # Simulate not a git repo: git rev-parse fails
                mock_run.side_effect = CalledProcessError(128, "git rev-parse")

                with pytest.raises(GitNotRepoError):
                    GitManager(tmp_path)


class TestGitIntegrationEndToEnd:
    """End-to-end tests for git integration (requires git)."""

    @pytest.mark.integration
    def test_git_changed_files_in_real_repo(self, tmp_path):
        """Test getting changed files in a real git repo.

        This test requires git to be available and is marked as integration test.
        """
        import subprocess

        # Initialize a git repo
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        # Create and commit a file
        committed_file = tmp_path / "committed.py"
        committed_file.write_text("def old(): pass")
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        # Create new files (changed)
        new_file = tmp_path / "new.py"
        new_file.write_text("def new(): pass")

        # Modify committed file
        committed_file.write_text("def old(): pass\ndef modified(): pass")

        # Get changed files
        git_manager = GitManager(tmp_path)
        changed_files = git_manager.get_changed_files(include_untracked=True)

        # Should detect both new and modified files
        assert len(changed_files) >= 1
        assert any(f.name == "new.py" for f in changed_files)

    @pytest.mark.integration
    def test_git_diff_files_baseline(self, tmp_path):
        """Test getting diff files against baseline branch."""
        import subprocess

        # Initialize a git repo
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        # Create initial commit on main
        old_file = tmp_path / "old.py"
        old_file.write_text("def old(): pass")
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        # Rename current branch to main if needed
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout.strip() != "main":
            subprocess.run(
                ["git", "branch", "-M", "main"],
                cwd=tmp_path,
                check=True,
                capture_output=True,
            )

        # Create feature branch
        subprocess.run(
            ["git", "checkout", "-b", "feature"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        # Create new file on feature branch
        new_file = tmp_path / "feature.py"
        new_file.write_text("def feature(): pass")
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Add feature"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        # Get diff files vs main
        git_manager = GitManager(tmp_path)
        diff_files = git_manager.get_diff_files("main")

        # Should detect the new file
        assert len(diff_files) == 1
        assert any(f.name == "feature.py" for f in diff_files)
