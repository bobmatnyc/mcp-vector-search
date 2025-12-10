"""Tests for gitignore updater utility."""

from pathlib import Path

import pytest

from mcp_vector_search.utils.gitignore_updater import ensure_gitignore_entry


class TestEnsureGitignoreEntry:
    """Test suite for ensure_gitignore_entry function."""

    @pytest.fixture
    def git_repo(self, tmp_path: Path) -> Path:
        """Create a temporary git repository."""
        # Create .git directory to simulate git repo
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        return tmp_path

    @pytest.fixture
    def non_git_repo(self, tmp_path: Path) -> Path:
        """Create a temporary non-git directory."""
        return tmp_path

    def test_create_gitignore_in_git_repo(self, git_repo: Path):
        """Test creating new .gitignore in a git repository."""
        result = ensure_gitignore_entry(git_repo)

        assert result is True
        gitignore_path = git_repo / ".gitignore"
        assert gitignore_path.exists()

        content = gitignore_path.read_text()
        assert ".mcp-vector-search/" in content
        assert "MCP Vector Search index directory" in content

    def test_skip_creation_in_non_git_repo(self, non_git_repo: Path):
        """Test skipping .gitignore creation in non-git directory."""
        result = ensure_gitignore_entry(non_git_repo)

        assert result is False
        gitignore_path = non_git_repo / ".gitignore"
        assert not gitignore_path.exists()

    def test_add_to_existing_gitignore(self, git_repo: Path):
        """Test adding entry to existing .gitignore."""
        gitignore_path = git_repo / ".gitignore"
        gitignore_path.write_text("# Existing content\n*.pyc\n__pycache__/\n")

        result = ensure_gitignore_entry(git_repo)

        assert result is True
        content = gitignore_path.read_text()
        assert "*.pyc" in content  # Original content preserved
        assert ".mcp-vector-search/" in content

    def test_pattern_already_exists_exact(self, git_repo: Path):
        """Test detecting exact pattern match."""
        gitignore_path = git_repo / ".gitignore"
        gitignore_path.write_text(".mcp-vector-search/\n")

        result = ensure_gitignore_entry(git_repo)

        assert result is True
        content = gitignore_path.read_text()
        # Should not duplicate the entry
        assert content.count(".mcp-vector-search/") == 1

    def test_pattern_already_exists_without_slash(self, git_repo: Path):
        """Test detecting pattern without trailing slash."""
        gitignore_path = git_repo / ".gitignore"
        gitignore_path.write_text(".mcp-vector-search\n")

        result = ensure_gitignore_entry(git_repo)

        assert result is True
        content = gitignore_path.read_text()
        # Should not add duplicate (patterns are equivalent)
        assert ".mcp-vector-search/" not in content
        assert content.count(".mcp-vector-search") == 1

    def test_pattern_already_exists_with_wildcard(self, git_repo: Path):
        """Test detecting pattern with wildcard."""
        gitignore_path = git_repo / ".gitignore"
        gitignore_path.write_text(".mcp-vector-search/*\n")

        result = ensure_gitignore_entry(git_repo)

        assert result is True
        content = gitignore_path.read_text()
        # Should not add duplicate
        assert content.count("mcp-vector-search") == 1

    def test_pattern_already_exists_root_relative(self, git_repo: Path):
        """Test detecting root-relative pattern."""
        gitignore_path = git_repo / ".gitignore"
        gitignore_path.write_text("/.mcp-vector-search/\n")

        result = ensure_gitignore_entry(git_repo)

        assert result is True
        content = gitignore_path.read_text()
        # Should not add duplicate
        assert content.count("mcp-vector-search") == 1

    def test_negation_pattern_detected(self, git_repo: Path):
        """Test detecting and respecting negation pattern."""
        gitignore_path = git_repo / ".gitignore"
        gitignore_path.write_text(".*\n!.mcp-vector-search/\n")

        result = ensure_gitignore_entry(git_repo)

        # Should return False and not modify file
        assert result is False
        content = gitignore_path.read_text()
        # Original negation pattern should remain
        assert "!.mcp-vector-search/" in content
        # Should not add duplicate entry
        assert content.count("mcp-vector-search") == 1

    def test_empty_gitignore(self, git_repo: Path):
        """Test handling empty .gitignore file."""
        gitignore_path = git_repo / ".gitignore"
        gitignore_path.write_text("")

        result = ensure_gitignore_entry(git_repo)

        assert result is True
        content = gitignore_path.read_text()
        assert ".mcp-vector-search/" in content

    def test_empty_gitignore_with_whitespace(self, git_repo: Path):
        """Test handling .gitignore with only whitespace."""
        gitignore_path = git_repo / ".gitignore"
        gitignore_path.write_text("   \n\n  \n")

        result = ensure_gitignore_entry(git_repo)

        assert result is True
        content = gitignore_path.read_text()
        assert ".mcp-vector-search/" in content

    def test_preserves_structure(self, git_repo: Path):
        """Test preserving file structure and newlines."""
        gitignore_path = git_repo / ".gitignore"
        original_content = "*.pyc\n__pycache__/\n"
        gitignore_path.write_text(original_content)

        result = ensure_gitignore_entry(git_repo)

        assert result is True
        content = gitignore_path.read_text()
        # Original content should be preserved
        assert "*.pyc\n" in content
        assert "__pycache__/\n" in content
        # New entry should be added with proper formatting
        assert ".mcp-vector-search/\n" in content

    def test_custom_pattern(self, git_repo: Path):
        """Test adding custom pattern."""
        result = ensure_gitignore_entry(
            git_repo, pattern=".custom-dir/", comment="Custom directory"
        )

        assert result is True
        gitignore_path = git_repo / ".gitignore"
        content = gitignore_path.read_text()
        assert ".custom-dir/" in content
        assert "Custom directory" in content

    def test_no_comment(self, git_repo: Path):
        """Test adding pattern without comment."""
        result = ensure_gitignore_entry(git_repo, comment=None)

        assert result is True
        gitignore_path = git_repo / ".gitignore"
        content = gitignore_path.read_text()
        assert ".mcp-vector-search/\n" in content
        assert "# MCP Vector Search" not in content

    def test_create_if_missing_false(self, git_repo: Path):
        """Test not creating .gitignore when create_if_missing=False."""
        result = ensure_gitignore_entry(git_repo, create_if_missing=False)

        assert result is False
        gitignore_path = git_repo / ".gitignore"
        assert not gitignore_path.exists()

    def test_utf8_encoding(self, git_repo: Path):
        """Test UTF-8 encoding handling."""
        gitignore_path = git_repo / ".gitignore"
        # Write file with UTF-8 characters
        gitignore_path.write_text("# コメント\n*.pyc\n", encoding="utf-8")

        result = ensure_gitignore_entry(git_repo)

        assert result is True
        content = gitignore_path.read_text(encoding="utf-8")
        assert "コメント" in content  # UTF-8 preserved
        assert ".mcp-vector-search/" in content

    def test_permission_error_handling(self, git_repo: Path):
        """Test graceful handling of permission errors."""
        gitignore_path = git_repo / ".gitignore"
        gitignore_path.write_text("*.pyc\n")

        # Make file read-only
        gitignore_path.chmod(0o444)

        try:
            result = ensure_gitignore_entry(git_repo)

            # Should return False but not raise exception
            assert result is False
        finally:
            # Restore permissions for cleanup
            gitignore_path.chmod(0o644)

    def test_handles_comments_in_gitignore(self, git_repo: Path):
        """Test that comments are properly skipped when checking patterns."""
        gitignore_path = git_repo / ".gitignore"
        gitignore_path.write_text(
            "# This is a comment about .mcp-vector-search/\n*.pyc\n"
        )

        result = ensure_gitignore_entry(git_repo)

        assert result is True
        content = gitignore_path.read_text()
        # Should add pattern because comment doesn't count as pattern
        assert content.count(".mcp-vector-search") >= 1

    def test_multiple_calls_idempotent(self, git_repo: Path):
        """Test that multiple calls are idempotent."""
        # First call
        result1 = ensure_gitignore_entry(git_repo)
        assert result1 is True

        gitignore_path = git_repo / ".gitignore"
        content_after_first = gitignore_path.read_text()

        # Second call
        result2 = ensure_gitignore_entry(git_repo)
        assert result2 is True

        content_after_second = gitignore_path.read_text()

        # Content should be identical (no duplicate entries)
        assert content_after_first == content_after_second
        assert content_after_second.count(".mcp-vector-search") == 1

    def test_blank_lines_preserved(self, git_repo: Path):
        """Test that blank lines in original file are preserved."""
        gitignore_path = git_repo / ".gitignore"
        gitignore_path.write_text("*.pyc\n\n__pycache__/\n")

        result = ensure_gitignore_entry(git_repo)

        assert result is True
        content = gitignore_path.read_text()
        # Original blank line should be preserved
        assert "*.pyc\n\n__pycache__/" in content

    def test_handles_trailing_whitespace(self, git_repo: Path):
        """Test handling patterns with trailing whitespace."""
        gitignore_path = git_repo / ".gitignore"
        gitignore_path.write_text(".mcp-vector-search/  \n")

        result = ensure_gitignore_entry(git_repo)

        assert result is True
        content = gitignore_path.read_text()
        # Should detect existing pattern despite whitespace
        # Should not add duplicate
        lines_with_pattern = [
            line for line in content.split("\n") if "mcp-vector-search" in line
        ]
        assert len(lines_with_pattern) == 1
