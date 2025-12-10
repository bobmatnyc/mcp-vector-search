"""Integration tests for the setup command.

This module provides end-to-end integration tests for the setup command,
testing the complete workflow from detection through configuration and indexing.
"""

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mcp_vector_search.cli.commands.setup import _run_smart_setup
from mcp_vector_search.core.project import ProjectManager

# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def cli_runner():
    """Create CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def python_project(tmp_path):
    """Create a realistic Python project."""
    # Create project structure
    (tmp_path / ".git").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()

    # Main application files
    (tmp_path / "src" / "__init__.py").write_text("")
    (tmp_path / "src" / "main.py").write_text(
        """
def main():
    '''Main application entry point.'''
    print('Hello, World!')
    return 0

if __name__ == '__main__':
    main()
"""
    )

    (tmp_path / "src" / "utils.py").write_text(
        """
def calculate(a, b):
    '''Calculate sum of two numbers.'''
    return a + b

class Helper:
    '''Helper class for utilities.'''
    def __init__(self):
        self.data = []

    def process(self, item):
        '''Process an item.'''
        self.data.append(item)
        return len(self.data)
"""
    )

    # Test files
    (tmp_path / "tests" / "__init__.py").write_text("")
    (tmp_path / "tests" / "test_main.py").write_text(
        """
import pytest
from src.main import main

def test_main():
    '''Test main function.'''
    assert main() == 0
"""
    )

    # Configuration files
    (tmp_path / "README.md").write_text("# Test Project")
    (tmp_path / "requirements.txt").write_text("pytest\nrequests\n")
    (tmp_path / ".gitignore").write_text("__pycache__/\n*.pyc\n")

    return tmp_path


@pytest.fixture
def javascript_project(tmp_path):
    """Create a realistic JavaScript project."""
    (tmp_path / ".git").mkdir()

    (tmp_path / "package.json").write_text(
        json.dumps({"name": "test-project", "version": "1.0.0", "main": "index.js"})
    )

    (tmp_path / "index.js").write_text(
        """
function main() {
  console.log('Hello, World!');
  return 0;
}

module.exports = { main };
"""
    )

    (tmp_path / "utils.js").write_text(
        """
class Helper {
  constructor() {
    this.data = [];
  }

  process(item) {
    this.data.push(item);
    return this.data.length;
  }
}

module.exports = { Helper };
"""
    )

    return tmp_path


@pytest.fixture
def mixed_project(tmp_path):
    """Create a project with multiple languages."""
    (tmp_path / ".git").mkdir()

    # Python files
    (tmp_path / "app.py").write_text("print('Python')")

    # JavaScript files
    (tmp_path / "app.js").write_text("console.log('JavaScript');")

    # TypeScript files
    (tmp_path / "app.ts").write_text("console.log('TypeScript');")

    # Configuration
    (tmp_path / "README.md").write_text("# Mixed Project")

    return tmp_path


@pytest.fixture
def mock_typer_context(tmp_path):
    """Create mock Typer context."""
    from unittest.mock import Mock

    context = Mock()
    context.obj = {"project_root": tmp_path}
    context.invoked_subcommand = None
    return context


# ==============================================================================
# A. Full Workflow Tests
# ==============================================================================


@pytest.mark.skip(
    reason="Tests need update for py_mcp_installer PlatformInfo data structure"
)
class TestSetupCompleteWorkflow:
    """Test complete setup workflow end-to-end."""

    @pytest.mark.asyncio
    async def test_setup_complete_workflow(self, python_project, mock_typer_context):
        """Test complete setup workflow from start to finish."""
        # Arrange
        mock_typer_context.obj = {"project_root": python_project}

        # Mock external dependencies
        with (
            patch(
                "mcp_vector_search.cli.commands.setup.detect_all_platforms"
            ) as mock_detect,
            patch(
                "mcp_vector_search.cli.commands.setup.check_claude_cli_available"
            ) as mock_check_claude,
            patch(
                "mcp_vector_search.cli.commands.setup.check_uv_available"
            ) as mock_check_uv,
            patch("subprocess.run") as mock_subprocess,
            patch("mcp_vector_search.cli.commands.index.run_indexing") as mock_index,
        ):
            mock_detect.return_value = [
                {"name": "claude-code", "path": Path(".mcp.json")}
            ]
            mock_check_claude.return_value = True
            mock_check_uv.return_value = True
            mock_subprocess.return_value = type(
                "obj", (object,), {"returncode": 0, "stdout": "", "stderr": ""}
            )()
            mock_index.return_value = None

            # Act
            await _run_smart_setup(mock_typer_context, force=False, verbose=False)

            # Assert - Verify all phases completed
            # 1. Project initialized
            project_manager = ProjectManager(python_project)
            assert project_manager.is_initialized()

            # 2. Configuration created
            config_path = python_project / ".mcp-vector-search" / "config.json"
            assert config_path.exists()

            # 3. .gitignore updated
            gitignore_path = python_project / ".gitignore"
            assert gitignore_path.exists()
            gitignore_content = gitignore_path.read_text()
            assert ".mcp-vector-search/" in gitignore_content

            # 4. Indexing called
            mock_index.assert_called_once()

            # 5. MCP platform configured via subprocess calls
            # Should have called remove and then add
            subprocess_calls = mock_subprocess.call_args_list
            assert len(subprocess_calls) >= 2
            # First call should be remove
            remove_call = subprocess_calls[0][0][0]
            assert "claude" in remove_call
            assert "mcp" in remove_call
            assert "remove" in remove_call
            # Second call should be add
            add_call = subprocess_calls[1][0][0]
            assert "claude" in add_call
            assert "mcp" in add_call
            assert "add" in add_call

    @pytest.mark.asyncio
    async def test_setup_with_existing_project(
        self, python_project, mock_typer_context
    ):
        """Test setup on already initialized project."""
        # Arrange
        mock_typer_context.obj = {"project_root": python_project}

        # Initialize project first
        project_manager = ProjectManager(python_project)
        project_manager.initialize(
            file_extensions=[".py"],
            embedding_model="test-model",
            similarity_threshold=0.5,
        )

        with (
            patch(
                "mcp_vector_search.cli.commands.setup.detect_installed_platforms"
            ) as mock_detect,
            patch(
                "mcp_vector_search.cli.commands.setup.check_claude_cli_available"
            ) as mock_check_claude,
            patch(
                "mcp_vector_search.cli.commands.setup.check_uv_available"
            ) as mock_check_uv,
            patch("subprocess.run") as mock_subprocess,
            patch("mcp_vector_search.cli.commands.index.run_indexing") as mock_index,
        ):
            mock_detect.return_value = {"claude-code": Path(".mcp.json")}
            mock_check_claude.return_value = True
            mock_check_uv.return_value = True
            mock_subprocess.return_value = type(
                "obj", (object,), {"returncode": 0, "stdout": "", "stderr": ""}
            )()

            # Act
            await _run_smart_setup(mock_typer_context, force=False, verbose=False)

            # Assert
            # Should skip re-initialization but still configure MCP
            mock_index.assert_not_called()
            # Should have called subprocess for MCP registration
            assert mock_subprocess.call_count >= 2

    @pytest.mark.asyncio
    async def test_setup_multi_language_project(
        self, mixed_project, mock_typer_context
    ):
        """Test setup on multi-language project."""
        # Arrange
        mock_typer_context.obj = {"project_root": mixed_project}

        with (
            patch(
                "mcp_vector_search.cli.commands.setup.detect_installed_platforms"
            ) as mock_detect,
            patch(
                "mcp_vector_search.cli.commands.setup.configure_platform"
            ) as mock_configure,
            patch("mcp_vector_search.cli.commands.index.run_indexing") as mock_index,
        ):
            mock_detect.return_value = {}
            mock_configure.return_value = True
            mock_index.return_value = None

            # Act
            await _run_smart_setup(mock_typer_context, force=False, verbose=False)

            # Assert
            project_manager = ProjectManager(mixed_project)
            languages = project_manager.detect_languages()

            # Should detect multiple languages
            assert len(languages) > 0

            # Configuration should include extensions for all languages
            config_path = mixed_project / ".mcp-vector-search" / "config.json"
            config = json.loads(config_path.read_text())

            # Should have detected .py, .js, .ts extensions
            extensions = config.get("file_extensions", [])
            assert any(ext in extensions for ext in [".py", ".js", ".ts"])

    @pytest.mark.asyncio
    async def test_setup_verbose_mode(self, python_project, mock_typer_context):
        """Test setup with verbose output enabled."""
        # Arrange
        mock_typer_context.obj = {"project_root": python_project}

        with (
            patch(
                "mcp_vector_search.cli.commands.setup.detect_installed_platforms"
            ) as mock_detect,
            patch(
                "mcp_vector_search.cli.commands.setup.configure_platform"
            ) as mock_configure,
            patch("mcp_vector_search.cli.commands.index.run_indexing") as mock_index,
        ):
            mock_detect.return_value = {"claude-code": Path(".mcp.json")}
            mock_configure.return_value = True
            mock_index.return_value = None

            # Act
            await _run_smart_setup(mock_typer_context, force=False, verbose=True)

            # Assert - Should complete successfully with verbose output
            project_manager = ProjectManager(python_project)
            assert project_manager.is_initialized()


# ==============================================================================
# B. MCP Integration Tests
# ==============================================================================


@pytest.mark.skip(
    reason="Tests need update for py_mcp_installer PlatformInfo data structure"
)
class TestSetupMCPIntegration:
    """Test MCP platform integration in setup."""

    @pytest.mark.asyncio
    async def test_setup_creates_all_mcp_configs(
        self, python_project, mock_typer_context
    ):
        """Test that setup creates configs for all detected platforms."""
        # Arrange
        mock_typer_context.obj = {"project_root": python_project}

        with (
            patch(
                "mcp_vector_search.cli.commands.setup.detect_installed_platforms"
            ) as mock_detect,
            patch(
                "mcp_vector_search.cli.commands.setup.check_claude_cli_available"
            ) as mock_check_claude,
            patch(
                "mcp_vector_search.cli.commands.setup.check_uv_available"
            ) as mock_check_uv,
            patch("subprocess.run") as mock_subprocess,
            patch(
                "mcp_vector_search.cli.commands.setup.configure_platform"
            ) as mock_configure,
            patch("mcp_vector_search.cli.commands.index.run_indexing") as mock_index,
        ):
            # Simulate multiple platforms detected
            mock_detect.return_value = {
                "claude-code": Path(".mcp.json"),
                "cursor": Path("~/.cursor/mcp.json"),
            }
            mock_check_claude.return_value = True
            mock_check_uv.return_value = True
            mock_subprocess.return_value = type(
                "obj", (object,), {"returncode": 0, "stdout": "", "stderr": ""}
            )()
            mock_configure.return_value = True
            mock_index.return_value = None

            # Act
            await _run_smart_setup(mock_typer_context, force=False, verbose=False)

            # Assert - Claude Code should use subprocess, cursor should use configure_platform
            # Subprocess should be called for claude-code (remove + add)
            assert mock_subprocess.call_count >= 2
            # configure_platform should be called for cursor
            mock_configure.assert_called_once()
            call_args = mock_configure.call_args
            assert call_args[1]["platform"] == "cursor" or call_args[0][0] == "cursor"

    @pytest.mark.asyncio
    async def test_setup_project_scoped_mcp(self, python_project, mock_typer_context):
        """Test project-scoped MCP configuration (.mcp.json)."""
        # Arrange
        mock_typer_context.obj = {"project_root": python_project}

        with (
            patch(
                "mcp_vector_search.cli.commands.setup.detect_installed_platforms"
            ) as mock_detect,
            patch(
                "mcp_vector_search.cli.commands.setup.check_claude_cli_available"
            ) as mock_check_claude,
            patch(
                "mcp_vector_search.cli.commands.setup.check_uv_available"
            ) as mock_check_uv,
            patch("subprocess.run") as mock_subprocess,
            patch("mcp_vector_search.cli.commands.index.run_indexing") as mock_index,
        ):
            mock_detect.return_value = {"claude-code": Path(".mcp.json")}
            mock_check_claude.return_value = True
            mock_check_uv.return_value = True
            mock_subprocess.return_value = type(
                "obj", (object,), {"returncode": 0, "stdout": "", "stderr": ""}
            )()
            mock_index.return_value = None

            # Act
            await _run_smart_setup(mock_typer_context, force=False, verbose=False)

            # Assert - Should have called subprocess for MCP registration
            assert mock_subprocess.call_count >= 2
            # Verify remove and add commands were called
            subprocess_calls = mock_subprocess.call_args_list
            remove_call = subprocess_calls[0][0][0]
            assert "remove" in remove_call
            add_call = subprocess_calls[1][0][0]
            assert "add" in add_call

    @pytest.mark.asyncio
    async def test_setup_global_mcp_configs(self, python_project, mock_typer_context):
        """Test global MCP configuration includes cwd."""
        # Arrange
        mock_typer_context.obj = {"project_root": python_project}

        with (
            patch(
                "mcp_vector_search.cli.commands.setup.detect_installed_platforms"
            ) as mock_detect,
            patch(
                "mcp_vector_search.cli.commands.setup.configure_platform"
            ) as mock_configure,
            patch("mcp_vector_search.cli.commands.index.run_indexing") as mock_index,
        ):
            mock_detect.return_value = {"cursor": Path("~/.cursor/mcp.json")}
            mock_index.return_value = None

            # Capture configure_platform call to verify cwd
            configured_platforms = []

            def capture_configure(*args, **kwargs):
                configured_platforms.append((args, kwargs))
                return True

            mock_configure.side_effect = capture_configure

            # Act
            await _run_smart_setup(mock_typer_context, force=False, verbose=False)

            # Assert - Cursor is global, should be configured
            assert len(configured_platforms) > 0
            # Extract platform name from first call (positional or keyword)
            args, kwargs = configured_platforms[0]
            if args:
                platform_name = args[0]
            elif "platform" in kwargs:
                platform_name = kwargs["platform"]
            else:
                platform_name = None
            assert platform_name == "cursor"


# ==============================================================================
# C. Edge Case Tests
# ==============================================================================


@pytest.mark.skip(
    reason="Tests need update for py_mcp_installer PlatformInfo data structure"
)
class TestSetupEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_setup_interrupted_and_resumed(
        self, python_project, mock_typer_context
    ):
        """Test that setup can be safely interrupted and resumed."""
        # Arrange
        mock_typer_context.obj = {"project_root": python_project}

        # First run - simulate interruption during indexing
        with (
            patch(
                "mcp_vector_search.cli.commands.setup.detect_installed_platforms"
            ) as mock_detect,
            patch(
                "mcp_vector_search.cli.commands.setup.configure_platform"
            ) as mock_configure,
            patch("mcp_vector_search.cli.commands.index.run_indexing") as mock_index,
        ):
            mock_detect.return_value = {}
            mock_configure.return_value = True
            mock_index.side_effect = KeyboardInterrupt()

            # Act - First run interrupted
            with pytest.raises(KeyboardInterrupt):
                await _run_smart_setup(mock_typer_context, force=False, verbose=False)

        # Second run - complete setup
        with (
            patch(
                "mcp_vector_search.cli.commands.setup.detect_installed_platforms"
            ) as mock_detect,
            patch(
                "mcp_vector_search.cli.commands.setup.configure_platform"
            ) as mock_configure,
            patch("mcp_vector_search.cli.commands.index.run_indexing") as mock_index,
        ):
            mock_detect.return_value = {}
            mock_configure.return_value = True
            mock_index.return_value = None

            # Act - Second run completes
            await _run_smart_setup(mock_typer_context, force=False, verbose=False)

            # Assert - Should work correctly (idempotent)
            project_manager = ProjectManager(python_project)
            assert project_manager.is_initialized()

    @pytest.mark.asyncio
    async def test_setup_no_mcp_platforms_detected(
        self, python_project, mock_typer_context
    ):
        """Test setup when no MCP platforms are detected."""
        # Arrange
        mock_typer_context.obj = {"project_root": python_project}

        with (
            patch(
                "mcp_vector_search.cli.commands.setup.detect_installed_platforms"
            ) as mock_detect,
            patch(
                "mcp_vector_search.cli.commands.setup.check_claude_cli_available"
            ) as mock_check_claude,
            patch(
                "mcp_vector_search.cli.commands.setup.check_uv_available"
            ) as mock_check_uv,
            patch("subprocess.run") as mock_subprocess,
            patch("mcp_vector_search.cli.commands.index.run_indexing") as mock_index,
        ):
            mock_detect.return_value = {}  # No platforms
            mock_check_claude.return_value = True
            mock_check_uv.return_value = True
            mock_subprocess.return_value = type(
                "obj", (object,), {"returncode": 0, "stdout": "", "stderr": ""}
            )()
            mock_index.return_value = None

            # Act
            await _run_smart_setup(mock_typer_context, force=False, verbose=False)

            # Assert - Should still work and configure claude-code as fallback
            project_manager = ProjectManager(python_project)
            assert project_manager.is_initialized()

            # Should configure at least claude-code via subprocess
            assert mock_subprocess.call_count >= 2

    @pytest.mark.asyncio
    async def test_setup_large_project_timeout(self, tmp_path, mock_typer_context):
        """Test setup on large project with timeout protection."""
        # Arrange - Create a large project
        mock_typer_context.obj = {"project_root": tmp_path}
        (tmp_path / ".git").mkdir()

        # Create many files
        for i in range(500):
            (tmp_path / f"file_{i}.py").write_text(f"def func_{i}(): pass")

        with (
            patch(
                "mcp_vector_search.cli.commands.setup.detect_installed_platforms"
            ) as mock_detect,
            patch(
                "mcp_vector_search.cli.commands.setup.configure_platform"
            ) as mock_configure,
            patch("mcp_vector_search.cli.commands.index.run_indexing") as mock_index,
        ):
            mock_detect.return_value = {}
            mock_configure.return_value = True
            mock_index.return_value = None

            # Act
            start_time = time.time()
            await _run_smart_setup(mock_typer_context, force=False, verbose=False)
            elapsed = time.time() - start_time

            # Assert - Should complete in reasonable time due to timeout
            assert elapsed < 60.0  # Should not hang indefinitely

            # Project should still be initialized even if scan times out
            project_manager = ProjectManager(tmp_path)
            assert project_manager.is_initialized()

    @pytest.mark.asyncio
    async def test_setup_preserves_existing_mcp_servers(
        self, python_project, mock_typer_context
    ):
        """Test that setup preserves existing MCP server configurations."""
        # Arrange
        mock_typer_context.obj = {"project_root": python_project}

        # Create existing .mcp.json with other servers
        mcp_json = python_project / ".mcp.json"
        existing_config = {
            "mcpServers": {
                "other-server": {
                    "command": "other-command",
                    "args": ["--flag"],
                }
            }
        }
        mcp_json.write_text(json.dumps(existing_config, indent=2))

        with (
            patch(
                "mcp_vector_search.cli.commands.setup.detect_installed_platforms"
            ) as mock_detect,
            patch(
                "mcp_vector_search.cli.commands.setup.check_claude_cli_available"
            ) as mock_check_claude,
            patch(
                "mcp_vector_search.cli.commands.setup.check_uv_available"
            ) as mock_check_uv,
            patch("subprocess.run") as mock_subprocess,
            patch("mcp_vector_search.cli.commands.index.run_indexing") as mock_index,
        ):
            mock_detect.return_value = {"claude-code": Path(".mcp.json")}
            mock_check_claude.return_value = True
            mock_check_uv.return_value = True
            mock_subprocess.return_value = type(
                "obj", (object,), {"returncode": 0, "stdout": "", "stderr": ""}
            )()
            mock_index.return_value = None

            # Act
            await _run_smart_setup(mock_typer_context, force=False, verbose=False)

            # Assert - Should have called subprocess for MCP registration
            # The actual file preservation is handled by claude CLI, not our code
            assert mock_subprocess.call_count >= 2

    @pytest.mark.asyncio
    async def test_setup_handles_corrupted_config(
        self, python_project, mock_typer_context
    ):
        """Test setup handles corrupted existing configuration."""
        # Arrange
        mock_typer_context.obj = {"project_root": python_project}

        # Create corrupted .mcp.json
        mcp_json = python_project / ".mcp.json"
        mcp_json.write_text("{ invalid json content")

        with (
            patch(
                "mcp_vector_search.cli.commands.setup.detect_installed_platforms"
            ) as mock_detect,
            patch(
                "mcp_vector_search.cli.commands.setup.check_claude_cli_available"
            ) as mock_check_claude,
            patch(
                "mcp_vector_search.cli.commands.setup.check_uv_available"
            ) as mock_check_uv,
            patch("subprocess.run") as mock_subprocess,
            patch("mcp_vector_search.cli.commands.index.run_indexing") as mock_index,
        ):
            mock_detect.return_value = {"claude-code": Path(".mcp.json")}
            mock_check_claude.return_value = True
            mock_check_uv.return_value = True
            mock_subprocess.return_value = type(
                "obj", (object,), {"returncode": 0, "stdout": "", "stderr": ""}
            )()
            mock_index.return_value = None

            # Act - Should handle corrupted file gracefully
            await _run_smart_setup(mock_typer_context, force=False, verbose=False)

            # Assert - Should have attempted MCP registration
            # The corrupted file handling is done by claude CLI
            assert mock_subprocess.call_count >= 2


# ==============================================================================
# D. Performance Tests
# ==============================================================================


@pytest.mark.skip(
    reason="Tests need update for py_mcp_installer PlatformInfo data structure"
)
class TestSetupPerformance:
    """Test setup command performance characteristics."""

    @pytest.mark.asyncio
    async def test_setup_completes_quickly_small_project(
        self, python_project, mock_typer_context
    ):
        """Test that setup completes quickly on small projects."""
        # Arrange
        mock_typer_context.obj = {"project_root": python_project}

        with (
            patch(
                "mcp_vector_search.cli.commands.setup.detect_installed_platforms"
            ) as mock_detect,
            patch(
                "mcp_vector_search.cli.commands.setup.configure_platform"
            ) as mock_configure,
            patch("mcp_vector_search.cli.commands.index.run_indexing") as mock_index,
        ):
            mock_detect.return_value = {}
            mock_configure.return_value = True
            mock_index.return_value = None

            # Act
            start_time = time.time()
            await _run_smart_setup(mock_typer_context, force=False, verbose=False)
            elapsed = time.time() - start_time

            # Assert - Should complete quickly (< 5 seconds typical)
            assert elapsed < 10.0

    @pytest.mark.asyncio
    async def test_setup_file_scan_respects_timeout(self, tmp_path, mock_typer_context):
        """Test that file scanning respects timeout parameter."""
        # Arrange
        mock_typer_context.obj = {"project_root": tmp_path}
        (tmp_path / ".git").mkdir()

        # Create moderate number of files
        for i in range(50):
            (tmp_path / f"file_{i}.py").write_text("pass")

        # Mock time to force timeout
        with (
            patch(
                "mcp_vector_search.cli.commands.setup.detect_installed_platforms"
            ) as mock_detect,
            patch(
                "mcp_vector_search.cli.commands.setup.configure_platform"
            ) as mock_configure,
            patch("mcp_vector_search.cli.commands.index.run_indexing") as mock_index,
            patch("time.time") as mock_time,
        ):
            mock_detect.return_value = {}
            mock_configure.return_value = True
            mock_index.return_value = None

            # Force timeout on file scan
            mock_time.side_effect = [0.0, 3.0, 3.1, 3.2]  # First scan times out

            # Act
            await _run_smart_setup(mock_typer_context, force=False, verbose=False)

            # Assert - Should complete despite timeout
            project_manager = ProjectManager(tmp_path)
            assert project_manager.is_initialized()


# ==============================================================================
# E. Configuration Validation Tests
# ==============================================================================


@pytest.mark.skip(
    reason="Tests need update for py_mcp_installer PlatformInfo data structure"
)
class TestSetupConfiguration:
    """Test configuration generation and validation."""

    @pytest.mark.asyncio
    async def test_setup_creates_valid_config_structure(
        self, python_project, mock_typer_context
    ):
        """Test that setup creates valid configuration structure."""
        # Arrange
        mock_typer_context.obj = {"project_root": python_project}

        with (
            patch(
                "mcp_vector_search.cli.commands.setup.detect_installed_platforms"
            ) as mock_detect,
            patch(
                "mcp_vector_search.cli.commands.setup.configure_platform"
            ) as mock_configure,
            patch("mcp_vector_search.cli.commands.index.run_indexing") as mock_index,
        ):
            mock_detect.return_value = {}
            mock_configure.return_value = True
            mock_index.return_value = None

            # Act
            await _run_smart_setup(mock_typer_context, force=False, verbose=False)

            # Assert
            config_path = python_project / ".mcp-vector-search" / "config.json"
            assert config_path.exists()

            config = json.loads(config_path.read_text())

            # Verify required fields
            assert "file_extensions" in config
            assert "embedding_model" in config
            assert "similarity_threshold" in config

            # Verify types
            assert isinstance(config["file_extensions"], list)
            assert isinstance(config["embedding_model"], str)
            assert isinstance(config["similarity_threshold"], int | float)

    @pytest.mark.asyncio
    async def test_setup_mcp_config_structure(self, python_project, mock_typer_context):
        """Test that MCP configuration has correct structure."""
        # Arrange
        mock_typer_context.obj = {"project_root": python_project}

        with (
            patch(
                "mcp_vector_search.cli.commands.setup.detect_installed_platforms"
            ) as mock_detect,
            patch(
                "mcp_vector_search.cli.commands.setup.check_claude_cli_available"
            ) as mock_check_claude,
            patch(
                "mcp_vector_search.cli.commands.setup.check_uv_available"
            ) as mock_check_uv,
            patch("subprocess.run") as mock_subprocess,
            patch("mcp_vector_search.cli.commands.index.run_indexing") as mock_index,
        ):
            mock_detect.return_value = {"claude-code": Path(".mcp.json")}
            mock_check_claude.return_value = True
            mock_check_uv.return_value = True
            mock_subprocess.return_value = type(
                "obj", (object,), {"returncode": 0, "stdout": "", "stderr": ""}
            )()
            mock_index.return_value = None

            # Act
            await _run_smart_setup(mock_typer_context, force=False, verbose=False)

            # Assert - Verify subprocess was called correctly
            assert mock_subprocess.call_count >= 2
            # Verify the structure of the add command
            add_call = mock_subprocess.call_args_list[1][0][0]
            assert "claude" in add_call
            assert "mcp" in add_call
            assert "add" in add_call
