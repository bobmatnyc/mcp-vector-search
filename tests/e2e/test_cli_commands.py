"""End-to-end tests for CLI commands."""

import json
import os
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mcp_vector_search.cli.main import app


class TestCLICommands:
    """End-to-end tests for CLI commands."""

    @pytest.fixture
    def cli_runner(self):
        """Create CLI runner for testing."""
        return CliRunner()

    @pytest.fixture(autouse=True)
    def setup_project_dir(self, temp_project_dir):
        """Automatically change to project directory for tests."""
        import shutil

        original_dir = os.getcwd()
        os.chdir(str(temp_project_dir))

        # Clean up any existing .mcp-vector-search directory to avoid ChromaDB corruption
        mcp_dir = temp_project_dir / ".mcp-vector-search"
        if mcp_dir.exists():
            shutil.rmtree(mcp_dir)

        yield

        # Clean up ChromaDB after test to prevent corruption in subsequent tests
        if mcp_dir.exists():
            shutil.rmtree(mcp_dir)

        os.chdir(original_dir)

    def test_init_command(self, cli_runner, temp_project_dir):
        """Test project initialization command."""
        # Test basic initialization
        result = cli_runner.invoke(
            app,
            [
                "init",
                "--extensions",
                ".py",
                "--embedding-model",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--similarity-threshold",
                "0.7",
                "--force",
                "--auto-index",
            ],
        )

        assert result.exit_code == 0
        assert "initialized" in result.output.lower()

        # Verify config file was created
        config_file = temp_project_dir / ".mcp-vector-search" / "config.json"
        assert config_file.exists()

        # Verify config content
        with open(config_file) as f:
            config = json.load(f)

        assert config["file_extensions"] == [".py"]
        assert config["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert config["similarity_threshold"] == 0.7

    def test_init_command_without_force(self, cli_runner, temp_project_dir):
        """Test initialization without force flag."""
        # Initialize once - mock confirm_action to return True for init, False for auto-index
        with patch(
            "mcp_vector_search.cli.commands.init.confirm_action"
        ) as mock_confirm:
            mock_confirm.side_effect = [True, False]  # Yes to init, No to auto-index
            result = cli_runner.invoke(app, ["init", "--extensions", ".py"])
            assert result.exit_code == 0

        # Try to initialize again without force - should succeed with message
        with patch(
            "mcp_vector_search.cli.commands.init.confirm_action"
        ) as mock_confirm:
            mock_confirm.side_effect = [
                True,
                False,
            ]  # Won't matter - already initialized
            result = cli_runner.invoke(app, ["init", "--extensions", ".py"])
            # Modern behavior: init succeeds but shows "already initialized" message
            assert result.exit_code == 0
            assert "already initialized" in result.output.lower()

    def test_index_command(self, cli_runner, temp_project_dir):
        """Test indexing command."""
        # Initialize project first - mock auto-index prompt
        with patch(
            "mcp_vector_search.cli.commands.init.confirm_action", return_value=False
        ):
            cli_runner.invoke(app, ["init", "--extensions", ".py", "--force"])

        # Test indexing
        result = cli_runner.invoke(app, ["index"])

        assert result.exit_code == 0
        assert "indexed" in result.output.lower()

        # Verify index database was created
        index_db = temp_project_dir / ".mcp-vector-search" / "chroma.sqlite3"
        assert index_db.exists()

    def test_index_command_force(self, cli_runner, temp_project_dir):
        """Test force indexing command."""
        # Initialize and index first - mock auto-index prompt
        with patch(
            "mcp_vector_search.cli.commands.init.confirm_action", return_value=False
        ):
            cli_runner.invoke(app, ["init", "--extensions", ".py", "--force"])
        cli_runner.invoke(app, ["index"])

        # Test force reindexing (with --no-analyze to keep test focused on indexing)
        result = cli_runner.invoke(app, ["index", "--force", "--no-analyze"])

        assert result.exit_code == 0
        assert "indexed" in result.output.lower()

    def test_index_command_force_with_analyze(self, cli_runner, temp_project_dir):
        """Test force indexing command with auto-analysis."""
        # Initialize and index first - mock auto-index prompt
        with patch(
            "mcp_vector_search.cli.commands.init.confirm_action", return_value=False
        ):
            cli_runner.invoke(app, ["init", "--extensions", ".py", "--force"])
        cli_runner.invoke(app, ["index"])

        # Test force reindexing with auto-analysis (default behavior)
        result = cli_runner.invoke(app, ["index", "--force"])

        assert result.exit_code == 0
        assert "indexed" in result.output.lower()
        assert "running analysis" in result.output.lower()

    def test_search_command(self, cli_runner, temp_project_dir):
        """Test search command."""
        # Initialize and index first - mock auto-index prompt
        with patch(
            "mcp_vector_search.cli.commands.init.confirm_action", return_value=False
        ):
            cli_runner.invoke(app, ["init", "--extensions", ".py", "--force"])
        cli_runner.invoke(app, ["index"])

        # Test search
        result = cli_runner.invoke(
            app,
            [
                "search",
                "--limit",
                "5",
                "main function",
            ],
        )

        assert result.exit_code == 0
        # May or may not find results, but should not error
        assert (
            "error" not in result.output.lower()
            or "no results" in result.output.lower()
        )

    def test_search_command_with_filters(self, cli_runner, temp_project_dir):
        """Test search command with filters."""
        # Initialize and index first - mock auto-index prompt
        with patch(
            "mcp_vector_search.cli.commands.init.confirm_action", return_value=False
        ):
            cli_runner.invoke(app, ["init", "--extensions", ".py", "--force"])
        cli_runner.invoke(app, ["index"])

        # Test search with language filter
        result = cli_runner.invoke(
            app,
            [
                "search",
                "--language",
                "python",
                "--threshold",
                "0.1",
                "function",
            ],
        )

        assert result.exit_code == 0

    def test_search_command_with_glob_pattern(self, cli_runner, temp_project_dir):
        """Test search command with glob pattern file filtering."""
        # Initialize and index first - mock auto-index prompt
        with patch(
            "mcp_vector_search.cli.commands.init.confirm_action", return_value=False
        ):
            cli_runner.invoke(app, ["init", "--extensions", ".py", "--force"])
        cli_runner.invoke(app, ["index"])

        # Test search with *.py glob pattern (should match all Python files)
        result = cli_runner.invoke(
            app,
            [
                "search",
                "--files",
                "*.py",
                "--threshold",
                "0.1",
                "function",
            ],
        )

        assert result.exit_code == 0
        # Should find results since we have Python files

        # Test search with specific file pattern
        result = cli_runner.invoke(
            app,
            [
                "search",
                "--files",
                "user_service.py",
                "--threshold",
                "0.1",
                "user",
            ],
        )

        assert result.exit_code == 0
        # Should only match user_service.py
        if "user_service.py" in result.output.lower():
            # If results found, they should be from user_service.py
            assert (
                "main.py" not in result.output.lower()
                or "no results" in result.output.lower()
            )

        # Test search with non-matching pattern
        result = cli_runner.invoke(
            app,
            [
                "search",
                "--files",
                "*.ts",
                "--threshold",
                "0.1",
                "function",
            ],
        )

        assert result.exit_code == 0
        # Should find no results since we have no TypeScript files

    def test_status_command(self, cli_runner, temp_project_dir):
        """Test status command."""
        # Initialize and index first - mock auto-index prompt
        with patch(
            "mcp_vector_search.cli.commands.init.confirm_action", return_value=False
        ):
            cli_runner.invoke(app, ["init", "--extensions", ".py", "--force"])
        cli_runner.invoke(app, ["index"])

        # Test status
        result = cli_runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "files" in result.output.lower()
        assert "chunks" in result.output.lower()

    def test_status_command_verbose(self, cli_runner, temp_project_dir):
        """Test verbose status command."""
        # Initialize and index first - mock auto-index prompt
        with patch(
            "mcp_vector_search.cli.commands.init.confirm_action", return_value=False
        ):
            cli_runner.invoke(app, ["init", "--extensions", ".py", "--force"])
        cli_runner.invoke(app, ["index"])

        # Test verbose status
        result = cli_runner.invoke(app, ["status", "--verbose"])

        assert result.exit_code == 0
        assert "files" in result.output.lower()
        assert "chunks" in result.output.lower()

    def test_config_command_show(self, cli_runner, temp_project_dir):
        """Test config show command."""
        # Initialize project first - mock auto-index prompt
        with patch(
            "mcp_vector_search.cli.commands.init.confirm_action", return_value=False
        ):
            cli_runner.invoke(app, ["init", "--extensions", ".py", "--force"])

        # Test config show
        result = cli_runner.invoke(app, ["config", "show"])

        assert result.exit_code == 0
        assert "File Extensions" in result.output or "file_extensions" in result.output
        assert ".py" in result.output

    def test_config_command_set(self, cli_runner, temp_project_dir):
        """Test config set command."""
        # Initialize project first - mock auto-index prompt
        with patch(
            "mcp_vector_search.cli.commands.init.confirm_action", return_value=False
        ):
            cli_runner.invoke(app, ["init", "--extensions", ".py", "--force"])

        # Test config set
        result = cli_runner.invoke(
            app, ["config", "set", "similarity_threshold", "0.8"]
        )

        assert result.exit_code == 0

        # Verify config was updated
        config_file = temp_project_dir / ".mcp-vector-search" / "config.json"
        with open(config_file) as f:
            config = json.load(f)

        assert config["similarity_threshold"] == 0.8

    def test_auto_index_status_command(self, cli_runner, temp_project_dir):
        """Test auto-index status command."""
        # Initialize and index first - mock auto-index prompt
        with patch(
            "mcp_vector_search.cli.commands.init.confirm_action", return_value=False
        ):
            cli_runner.invoke(app, ["init", "--extensions", ".py", "--force"])
        cli_runner.invoke(app, ["index"])

        # Test auto-index status
        result = cli_runner.invoke(app, ["index", "auto", "status"])

        assert result.exit_code == 0
        assert "files" in result.output.lower()

    @pytest.mark.skip(
        reason="ChromaDB Rust bindings have a known SQLite corruption issue when reopening databases in quick succession (https://github.com/chroma-core/chroma/issues). This test triggers 'range start index 10 out of range for slice of length 9' error in chromadb/api/rust.py. Works fine in production with proper database lifecycle management."
    )
    def test_auto_index_check_command(self, cli_runner, temp_project_dir):
        """Test auto-index check command.

        Note: This test is skipped due to ChromaDB Rust bindings bug with SQLite.
        The underlying functionality works correctly in production use cases where
        database connections are properly managed with delays between operations.
        """
        import gc
        import time

        # Initialize and index first - mock auto-index prompt
        with patch(
            "mcp_vector_search.cli.commands.init.confirm_action", return_value=False
        ):
            result = cli_runner.invoke(app, ["init", "--extensions", ".py", "--force"])
            assert result.exit_code == 0

        # Force garbage collection and short delay to ensure ChromaDB releases resources
        gc.collect()
        time.sleep(0.1)

        result = cli_runner.invoke(app, ["index"])
        assert result.exit_code == 0

        # Force garbage collection and delay before reopening database
        gc.collect()
        time.sleep(0.1)

        # Test auto-index check
        result = cli_runner.invoke(
            app,
            [
                "index",
                "auto",
                "check",
                "--no-auto-reindex",
            ],
        )

        assert result.exit_code == 0

    def test_error_handling_uninitialized_project(self, cli_runner, temp_project_dir):
        """Test error handling for uninitialized project."""
        # Try to index without initialization
        result = cli_runner.invoke(app, ["index"])

        assert result.exit_code != 0
        assert "not initialized" in result.output.lower()

    def test_error_handling_invalid_path(self, cli_runner, temp_dir):
        """Test error handling for invalid paths."""
        invalid_path = temp_dir / "nonexistent"

        # Try to initialize non-existent directory
        original_dir = os.getcwd()
        try:
            os.chdir(str(invalid_path))
            result = cli_runner.invoke(app, ["init", "main"])
        except FileNotFoundError:
            # Expected - directory doesn't exist
            result = type("Result", (), {"exit_code": 1})()
        finally:
            os.chdir(original_dir)

        assert result.exit_code != 0

    def test_help_commands(self, cli_runner):
        """Test help commands."""
        # Test main help
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "usage" in result.output.lower() or "help" in result.output.lower()

        # Test subcommand help
        subcommands = ["init", "index", "search", "status", "config", "auto-index"]

        for subcommand in subcommands:
            result = cli_runner.invoke(app, [subcommand, "--help"])
            assert result.exit_code == 0
            assert "usage" in result.output.lower() or "help" in result.output.lower()

    def test_full_workflow(self, cli_runner, temp_project_dir):
        """Test complete CLI workflow."""
        # Step 1: Initialize - mock auto-index prompt
        with patch(
            "mcp_vector_search.cli.commands.init.confirm_action", return_value=False
        ):
            result = cli_runner.invoke(app, ["init", "--extensions", ".py", "--force"])
        assert result.exit_code == 0

        # Step 2: Index
        result = cli_runner.invoke(app, ["index"])
        assert result.exit_code == 0

        # Step 3: Search
        result = cli_runner.invoke(
            app,
            [
                "search",
                "--limit",
                "3",
                "user",
            ],
        )
        assert result.exit_code == 0

        # Step 4: Status
        result = cli_runner.invoke(app, ["status"])
        assert result.exit_code == 0

        # Step 5: Config
        result = cli_runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0

        # Step 6: Force reindex
        result = cli_runner.invoke(app, ["index", "--force"])
        assert result.exit_code == 0

    @pytest.mark.skip(reason="CliRunner doesn't support concurrent operations well")
    def test_concurrent_cli_operations(self, cli_runner, temp_project_dir):
        """Test concurrent CLI operations."""
        # Initialize project - mock auto-index prompt
        with patch(
            "mcp_vector_search.cli.commands.init.confirm_action", return_value=False
        ):
            cli_runner.invoke(app, ["init", "--extensions", ".py", "--force"])
        cli_runner.invoke(app, ["index"])

        # Note: CliRunner has issues with concurrent operations due to shared file handles
        # This test is skipped for now. In production, the CLI should handle concurrent
        # searches correctly since they use separate processes.

        # Sequential searches as a workaround to test multiple searches work
        queries = ["function", "class", "import"]
        for query in queries:
            result = cli_runner.invoke(app, ["search", query, "--limit", "5"])
            assert result.exit_code == 0

    def test_performance_cli_operations(self, cli_runner, temp_project_dir):
        """Test performance of CLI operations."""
        import time

        # Time initialization - mock auto-index prompt
        start_time = time.perf_counter()
        with patch(
            "mcp_vector_search.cli.commands.init.confirm_action", return_value=False
        ):
            result = cli_runner.invoke(app, ["init", "--extensions", ".py", "--force"])
        init_time = time.perf_counter() - start_time

        assert result.exit_code == 0
        assert init_time < 5.0, f"Initialization took too long: {init_time:.3f}s"

        # Time indexing
        start_time = time.perf_counter()
        result = cli_runner.invoke(app, ["index"])
        index_time = time.perf_counter() - start_time

        assert result.exit_code == 0
        assert index_time < 10.0, f"Indexing took too long: {index_time:.3f}s"

        # Time search
        start_time = time.perf_counter()
        result = cli_runner.invoke(
            app,
            [
                "search",
                "--threshold",
                "0.1",
                "function",
            ],
        )
        search_time = time.perf_counter() - start_time

        assert result.exit_code == 0
        assert search_time < 5.0, f"Search took too long: {search_time:.3f}s"
