"""Tests for MCP installation auto-detection functionality."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from mcp_vector_search.cli.commands.install import (
    detect_project_root,
    find_git_root,
)


class TestProjectRootDetection:
    """Test project root auto-detection functionality."""

    def test_detect_with_mcp_vector_search_directory(self, tmp_path: Path):
        """Should detect project root via .mcp-vector-search directory."""
        # Create .mcp-vector-search directory
        mcp_dir = tmp_path / ".mcp-vector-search"
        mcp_dir.mkdir()

        # Detect from project root
        result = detect_project_root(tmp_path)
        assert result == tmp_path

    def test_detect_with_git_and_mcp_directory(self, tmp_path: Path):
        """Should detect git root when .mcp-vector-search exists there."""
        # Create git repo structure
        git_root = tmp_path / "project"
        git_root.mkdir()
        (git_root / ".git").mkdir()
        (git_root / ".mcp-vector-search").mkdir()

        # Create subdirectory
        subdir = git_root / "src" / "components"
        subdir.mkdir(parents=True)

        # Detect from subdirectory - should return git root
        result = detect_project_root(subdir)
        assert result == git_root

    def test_detect_fallback_to_current_directory(self, tmp_path: Path):
        """Should fallback to current directory if no markers found."""
        # No .git or .mcp-vector-search
        result = detect_project_root(tmp_path)
        assert result == tmp_path

    def test_detect_prefers_mcp_directory_over_git(self, tmp_path: Path):
        """Should prefer .mcp-vector-search in current dir over git root."""
        # Create git repo
        git_root = tmp_path / "git-project"
        git_root.mkdir()
        (git_root / ".git").mkdir()

        # Create subdirectory with .mcp-vector-search
        subdir = git_root / "sub-project"
        subdir.mkdir()
        (subdir / ".mcp-vector-search").mkdir()

        # Detect from subdirectory - should return subdirectory, not git root
        result = detect_project_root(subdir)
        assert result == subdir


class TestGitRootDetection:
    """Test git repository root detection."""

    def test_find_git_root_in_current_directory(self, tmp_path: Path):
        """Should find .git in current directory."""
        (tmp_path / ".git").mkdir()
        result = find_git_root(tmp_path)
        assert result == tmp_path

    def test_find_git_root_in_parent_directory(self, tmp_path: Path):
        """Should walk up to find .git in parent."""
        git_root = tmp_path / "project"
        git_root.mkdir()
        (git_root / ".git").mkdir()

        subdir = git_root / "src" / "components"
        subdir.mkdir(parents=True)

        result = find_git_root(subdir)
        assert result == git_root

    def test_find_git_root_returns_none_when_not_in_repo(self, tmp_path: Path):
        """Should return None when not in a git repository."""
        result = find_git_root(tmp_path)
        assert result is None

    def test_find_git_root_stops_at_filesystem_root(self, tmp_path: Path):
        """Should stop searching at filesystem root."""
        # Use a path that definitely won't have .git
        # tmp_path is already isolated
        deep_path = tmp_path / "a" / "b" / "c" / "d" / "e"
        deep_path.mkdir(parents=True)

        result = find_git_root(deep_path)
        assert result is None


class TestProjectRootEnvironmentVariables:
    """Test project root detection from environment variables."""

    def test_mcp_server_uses_mcp_project_root_env(self, tmp_path: Path):
        """MCP server should use MCP_PROJECT_ROOT environment variable."""
        from mcp_vector_search.mcp.server import MCPVectorSearchServer

        project_path = str(tmp_path / "my-project")

        with patch.dict(os.environ, {"MCP_PROJECT_ROOT": project_path}):
            server = MCPVectorSearchServer()
            assert str(server.project_root) == str(Path(project_path).resolve())

    def test_mcp_server_uses_legacy_project_root_env(self, tmp_path: Path):
        """MCP server should fallback to legacy PROJECT_ROOT env var."""
        from mcp_vector_search.mcp.server import MCPVectorSearchServer

        project_path = str(tmp_path / "legacy-project")

        with patch.dict(os.environ, {"PROJECT_ROOT": project_path}):
            server = MCPVectorSearchServer()
            assert str(server.project_root) == str(Path(project_path).resolve())

    def test_mcp_server_prefers_mcp_project_root_over_legacy(self, tmp_path: Path):
        """MCP_PROJECT_ROOT should take priority over PROJECT_ROOT."""
        from mcp_vector_search.mcp.server import MCPVectorSearchServer

        new_path = str(tmp_path / "new-project")
        legacy_path = str(tmp_path / "legacy-project")

        with patch.dict(
            os.environ, {"MCP_PROJECT_ROOT": new_path, "PROJECT_ROOT": legacy_path}
        ):
            server = MCPVectorSearchServer()
            assert str(server.project_root) == str(Path(new_path).resolve())

    def test_mcp_server_uses_cwd_when_no_env_set(self, tmp_path: Path, monkeypatch):
        """MCP server should use current directory when no env vars set."""
        from mcp_vector_search.mcp.server import MCPVectorSearchServer

        # Change to tmp_path
        monkeypatch.chdir(tmp_path)

        # Ensure no env vars are set
        with patch.dict(os.environ, {}, clear=True):
            server = MCPVectorSearchServer()
            assert server.project_root == Path.cwd()


class TestEndToEndScenarios:
    """Test realistic usage scenarios."""

    def test_scenario_fresh_project_installation(self, tmp_path: Path):
        """Scenario: User runs 'install mcp' in a fresh project."""
        # Setup: User has a git repo but no .mcp-vector-search yet
        project = tmp_path / "my-app"
        project.mkdir()
        (project / ".git").mkdir()

        # User is in project root
        detected = detect_project_root(project)
        assert detected == project

    def test_scenario_nested_subdirectory_installation(self, tmp_path: Path):
        """Scenario: User runs command from nested subdirectory."""
        # Setup: Project initialized at root
        project = tmp_path / "my-app"
        project.mkdir()
        (project / ".git").mkdir()
        (project / ".mcp-vector-search").mkdir()

        # User is in nested directory
        nested = project / "src" / "components" / "ui"
        nested.mkdir(parents=True)

        # Should still find project root
        detected = detect_project_root(nested)
        assert detected == project

    def test_scenario_monorepo_with_multiple_projects(self, tmp_path: Path):
        """Scenario: Monorepo with multiple sub-projects."""
        # Setup: Monorepo with two projects
        monorepo = tmp_path / "monorepo"
        monorepo.mkdir()
        (monorepo / ".git").mkdir()

        # Project A has its own mcp-vector-search
        project_a = monorepo / "project-a"
        project_a.mkdir()
        (project_a / ".mcp-vector-search").mkdir()

        # Project B has its own mcp-vector-search
        project_b = monorepo / "project-b"
        project_b.mkdir()
        (project_b / ".mcp-vector-search").mkdir()

        # From project A, should detect project A (not monorepo root)
        detected_a = detect_project_root(project_a)
        assert detected_a == project_a

        # From project B, should detect project B
        detected_b = detect_project_root(project_b)
        assert detected_b == project_b

    def test_scenario_non_git_project(self, tmp_path: Path):
        """Scenario: Project without git (e.g., simple script)."""
        # Setup: Simple project without git
        project = tmp_path / "simple-script"
        project.mkdir()
        (project / ".mcp-vector-search").mkdir()

        # Should still detect via .mcp-vector-search
        detected = detect_project_root(project)
        assert detected == project


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
