"""Integration tests for .gitignore automation during project initialization."""

from pathlib import Path

import pytest

from mcp_vector_search.core.project import ProjectManager


class TestProjectInitGitignore:
    """Integration tests for automatic .gitignore updates during init."""

    @pytest.fixture
    def git_project_dir(self, tmp_path: Path) -> Path:
        """Create a temporary git project directory."""
        # Create .git directory to simulate git repo
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        # Create some sample files
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "README.md").write_text("# Test Project")

        return tmp_path

    @pytest.fixture
    def non_git_project_dir(self, tmp_path: Path) -> Path:
        """Create a temporary non-git project directory."""
        # Create sample files but no .git directory
        (tmp_path / "main.py").write_text("print('hello')")
        return tmp_path

    def test_init_creates_gitignore_entry_in_git_repo(self, git_project_dir: Path):
        """Test that init creates .gitignore entry in git repository."""
        pm = ProjectManager(project_root=git_project_dir)

        # Initialize project
        config = pm.initialize(force=True)

        assert config is not None
        assert pm.is_initialized()

        # Check that .gitignore was created/updated
        gitignore_path = git_project_dir / ".gitignore"
        assert gitignore_path.exists()

        content = gitignore_path.read_text()
        assert ".mcp-vector-search/" in content
        assert "MCP Vector Search index directory" in content

    def test_init_skips_gitignore_in_non_git_repo(self, non_git_project_dir: Path):
        """Test that init skips .gitignore creation in non-git directory."""
        pm = ProjectManager(project_root=non_git_project_dir)

        # Initialize project
        config = pm.initialize(force=True)

        assert config is not None
        assert pm.is_initialized()

        # .gitignore should not be created in non-git projects
        gitignore_path = non_git_project_dir / ".gitignore"
        assert not gitignore_path.exists()

    def test_init_with_existing_gitignore(self, git_project_dir: Path):
        """Test that init updates existing .gitignore without corruption."""
        # Create existing .gitignore with some content
        gitignore_path = git_project_dir / ".gitignore"
        original_content = "# Existing gitignore\n*.pyc\n__pycache__/\n.env\n"
        gitignore_path.write_text(original_content)

        pm = ProjectManager(project_root=git_project_dir)
        config = pm.initialize(force=True)

        assert config is not None

        # Check that original content is preserved
        content = gitignore_path.read_text()
        assert "*.pyc" in content
        assert "__pycache__/" in content
        assert ".env" in content

        # Check that new entry was added
        assert ".mcp-vector-search/" in content
        assert "MCP Vector Search index directory" in content

    def test_init_force_with_existing_gitignore_entry(self, git_project_dir: Path):
        """Test that re-init doesn't duplicate .gitignore entry."""
        pm = ProjectManager(project_root=git_project_dir)

        # First initialization
        pm.initialize(force=True)

        gitignore_path = git_project_dir / ".gitignore"
        content_after_first = gitignore_path.read_text()
        first_count = content_after_first.count(".mcp-vector-search")

        # Force re-initialization
        pm.initialize(force=True)

        content_after_second = gitignore_path.read_text()
        second_count = content_after_second.count(".mcp-vector-search")

        # Count should be the same (no duplication)
        assert first_count == second_count
        assert first_count >= 1  # At least one entry exists

    def test_init_respects_negation_pattern(self, git_project_dir: Path):
        """Test that init respects existing negation patterns."""
        # User explicitly wants to track .mcp-vector-search/
        gitignore_path = git_project_dir / ".gitignore"
        gitignore_path.write_text(".*\n!.mcp-vector-search/\n")

        pm = ProjectManager(project_root=git_project_dir)
        config = pm.initialize(force=True)

        assert config is not None

        # Negation pattern should be preserved
        content = gitignore_path.read_text()
        assert "!.mcp-vector-search/" in content

        # Should not add duplicate entry
        non_negation_entries = [
            line
            for line in content.split("\n")
            if "mcp-vector-search" in line and not line.strip().startswith("!")
        ]
        assert len(non_negation_entries) == 0

    def test_init_handles_permission_error_gracefully(self, git_project_dir: Path):
        """Test that init continues even if .gitignore update fails."""
        # Create read-only .gitignore
        gitignore_path = git_project_dir / ".gitignore"
        gitignore_path.write_text("*.pyc\n")
        gitignore_path.chmod(0o444)

        try:
            pm = ProjectManager(project_root=git_project_dir)
            config = pm.initialize(force=True)

            # Initialization should succeed despite .gitignore error
            assert config is not None
            assert pm.is_initialized()

            # .mcp-vector-search directory should still be created
            index_dir = git_project_dir / ".mcp-vector-search"
            assert index_dir.exists()

        finally:
            # Restore permissions for cleanup
            gitignore_path.chmod(0o644)

    def test_init_with_complex_gitignore(self, git_project_dir: Path):
        """Test init with complex existing .gitignore file."""
        gitignore_path = git_project_dir / ".gitignore"
        complex_content = """# OS files
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp

# Python
*.pyc
__pycache__/
.pytest_cache/
.venv/

# Build
dist/
build/
*.egg-info/
"""
        gitignore_path.write_text(complex_content)

        pm = ProjectManager(project_root=git_project_dir)
        config = pm.initialize(force=True)

        assert config is not None

        content = gitignore_path.read_text()

        # All original content should be preserved
        assert ".DS_Store" in content
        assert ".vscode/" in content
        assert "*.pyc" in content
        assert ".pytest_cache/" in content

        # New entry should be added
        assert ".mcp-vector-search/" in content

    def test_init_with_utf8_gitignore(self, git_project_dir: Path):
        """Test init with UTF-8 characters in .gitignore."""
        gitignore_path = git_project_dir / ".gitignore"
        gitignore_path.write_text(
            "# コメント (Japanese comment)\n*.pyc\n", encoding="utf-8"
        )

        pm = ProjectManager(project_root=git_project_dir)
        config = pm.initialize(force=True)

        assert config is not None

        content = gitignore_path.read_text(encoding="utf-8")
        # UTF-8 content should be preserved
        assert "コメント" in content
        assert ".mcp-vector-search/" in content
