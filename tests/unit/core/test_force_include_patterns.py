"""Test force_include_patterns feature."""

from pathlib import Path

from mcp_vector_search.config.settings import ProjectConfig
from mcp_vector_search.core.project import ProjectManager


class TestForceIncludePatterns:
    """Test force_include_patterns configuration."""

    def test_force_include_overrides_gitignore(self, tmp_path: Path) -> None:
        """Test that force_include_patterns override gitignore rules."""
        # Create test project structure
        project_root = tmp_path / "test_project"
        project_root.mkdir()

        # Create .gitignore that excludes repos/
        gitignore_file = project_root / ".gitignore"
        gitignore_file.write_text("repos/\n")

        # Create repos directory with files
        repos_dir = project_root / "repos"
        repos_dir.mkdir()
        (repos_dir / "test.java").write_text("class Test {}")
        (repos_dir / "test.py").write_text("print('hello')")

        # Initialize project manager
        pm = ProjectManager(project_root)

        # Initialize config with force_include_patterns
        config = ProjectConfig(
            project_root=project_root,
            index_path=project_root / ".mcp-vector-search",
            respect_gitignore=True,
            force_include_patterns=["repos/**/*.java"],
        )
        pm._config = config

        # Create index directory
        config.index_path.mkdir(exist_ok=True)

        # Test: Java file should NOT be ignored (force_include)
        java_file = repos_dir / "test.java"
        assert not pm._should_ignore_path(java_file, is_directory=False), (
            "Java file should be force-included despite gitignore"
        )

        # Test: Python file should still be ignored (not in force_include)
        py_file = repos_dir / "test.py"
        assert pm._should_ignore_path(py_file, is_directory=False), (
            "Python file should be ignored (not in force_include pattern)"
        )

    def test_force_include_with_multiple_patterns(self, tmp_path: Path) -> None:
        """Test force_include with multiple glob patterns."""
        project_root = tmp_path / "test_project2"
        project_root.mkdir()

        # Create .gitignore
        gitignore_file = project_root / ".gitignore"
        gitignore_file.write_text("repos/\nvendor/\n")

        # Create directories
        repos_dir = project_root / "repos"
        repos_dir.mkdir()
        vendor_dir = project_root / "vendor"
        vendor_dir.mkdir()

        # Create files
        (repos_dir / "app.java").write_text("class App {}")
        (vendor_dir / "lib.kt").write_text("class Lib")

        # Initialize project manager
        pm = ProjectManager(project_root)

        # Config with multiple patterns
        config = ProjectConfig(
            project_root=project_root,
            index_path=project_root / ".mcp-vector-search",
            respect_gitignore=True,
            force_include_patterns=["repos/**/*.java", "vendor/**/*.kt"],
        )
        pm._config = config
        config.index_path.mkdir(exist_ok=True)

        # Both files should be force-included
        assert not pm._should_ignore_path(repos_dir / "app.java", is_directory=False)
        assert not pm._should_ignore_path(vendor_dir / "lib.kt", is_directory=False)

    def test_empty_force_include_patterns(self, tmp_path: Path) -> None:
        """Test that empty force_include_patterns behaves normally."""
        project_root = tmp_path / "test_project3"
        project_root.mkdir()

        # Create .gitignore
        gitignore_file = project_root / ".gitignore"
        gitignore_file.write_text("repos/\n")

        # Create directory
        repos_dir = project_root / "repos"
        repos_dir.mkdir()
        (repos_dir / "test.java").write_text("class Test {}")

        # Initialize project manager
        pm = ProjectManager(project_root)

        # Config with empty force_include_patterns
        config = ProjectConfig(
            project_root=project_root,
            index_path=project_root / ".mcp-vector-search",
            respect_gitignore=True,
            force_include_patterns=[],  # Empty list
        )
        pm._config = config
        config.index_path.mkdir(exist_ok=True)

        # File should be ignored (normal gitignore behavior)
        java_file = repos_dir / "test.java"
        assert pm._should_ignore_path(java_file, is_directory=False), (
            "File should be ignored when force_include_patterns is empty"
        )

    def test_respect_gitignore_false_with_force_include(self, tmp_path: Path) -> None:
        """Test that force_include works even when respect_gitignore is False."""
        project_root = tmp_path / "test_project4"
        project_root.mkdir()

        # Create .gitignore (should be ignored since respect_gitignore=False)
        gitignore_file = project_root / ".gitignore"
        gitignore_file.write_text("repos/\n")

        # Create directory
        repos_dir = project_root / "repos"
        repos_dir.mkdir()
        (repos_dir / "test.java").write_text("class Test {}")

        # Initialize project manager
        pm = ProjectManager(project_root)

        # Config with respect_gitignore=False
        config = ProjectConfig(
            project_root=project_root,
            index_path=project_root / ".mcp-vector-search",
            respect_gitignore=False,  # Don't respect gitignore
            force_include_patterns=["repos/**/*.java"],
        )
        pm._config = config
        config.index_path.mkdir(exist_ok=True)

        # File should NOT be ignored (gitignore is disabled)
        java_file = repos_dir / "test.java"
        assert not pm._should_ignore_path(java_file, is_directory=False), (
            "File should not be ignored when respect_gitignore is False"
        )
