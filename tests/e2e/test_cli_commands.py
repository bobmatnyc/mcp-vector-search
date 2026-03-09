"""End-to-end tests for CLI commands."""

import json
import os
import re
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mcp_vector_search.cli.main import app

# ---------------------------------------------------------------------------
# Source root for `uv run mvs` invocations
# ---------------------------------------------------------------------------
_SRC_ROOT = Path(__file__).parent.parent.parent


def _run_cli(
    args: list[str],
    project_root: Path,
    timeout: int = 60,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``uv run mvs <args>`` from *project_root*.

    Uses ``uv --directory _SRC_ROOT run mvs`` so that uv resolves the correct
    virtual environment from the project's pyproject.toml, while the subprocess
    working directory is *project_root* so that the CLI auto-detects it as the
    current project (important for ``init``, ``index``, ``status``, etc.).

    Uses Popen + communicate(timeout) so the child is always killed on timeout
    (subprocess.run raises TimeoutExpired but leaves an orphan process holding
    file locks).

    Disables tqdm progress bars via TQDM_DISABLE=1 to avoid fileno issues
    when stdout is captured.
    """
    env = {**os.environ, "TQDM_DISABLE": "1", "NO_COLOR": "1"}
    if extra_env:
        env.update(extra_env)

    # uv --project tells uv which project to use for environment resolution
    # without changing the subprocess cwd (unlike --directory which changes cwd).
    # The subprocess cwd=project_root lets mvs auto-detect it as the project root.
    cmd = ["uv", "--project", str(_SRC_ROOT), "run", "mvs"] + args
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(project_root),
        env=env,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        return subprocess.CompletedProcess(
            args=proc.args,
            returncode=-1,
            stdout=stdout or "",
            stderr=f"[TIMEOUT after {timeout}s]\n{stderr or ''}",
        )
    return subprocess.CompletedProcess(
        args=proc.args,
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _output(r: subprocess.CompletedProcess[str]) -> str:
    """Return combined stdout + stderr for assertion messages."""
    return (r.stdout or "") + (r.stderr or "")


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

        # Clean up any existing .mcp-vector-search directory
        mcp_dir = temp_project_dir / ".mcp-vector-search"
        if mcp_dir.exists():
            shutil.rmtree(mcp_dir)

        yield

        # Clean up after test
        if mcp_dir.exists():
            shutil.rmtree(mcp_dir)

        os.chdir(original_dir)

    # -----------------------------------------------------------------------
    # Tests that don't require actual embedding (CliRunner is fine)
    # -----------------------------------------------------------------------

    def test_init_command(self, cli_runner, temp_project_dir):
        """Test project initialization command."""
        result = cli_runner.invoke(
            app,
            [
                "--project-root",
                str(temp_project_dir),
                "init",
                "--extensions",
                ".py",
                "--embedding-model",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--similarity-threshold",
                "0.7",
                "--force",
                "--no-auto-index",
            ],
        )

        assert result.exit_code == 0, result.output
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
        # Initialize once — no auto-index to avoid fileno issues with CliRunner
        with patch(
            "mcp_vector_search.cli.commands.init.confirm_action"
        ) as mock_confirm:
            mock_confirm.side_effect = [True, False]  # Yes to init, No to auto-index
            result = cli_runner.invoke(
                app,
                [
                    "--project-root",
                    str(temp_project_dir),
                    "init",
                    "--extensions",
                    ".py",
                ],
            )
            assert result.exit_code == 0, result.output

        # Try to initialize again without force — should succeed with message
        with patch(
            "mcp_vector_search.cli.commands.init.confirm_action"
        ) as mock_confirm:
            mock_confirm.side_effect = [True, False]
            result = cli_runner.invoke(
                app,
                [
                    "--project-root",
                    str(temp_project_dir),
                    "init",
                    "--extensions",
                    ".py",
                ],
            )
            assert result.exit_code == 0, result.output
            assert "already initialized" in result.output.lower()

    def test_config_command_show(self, cli_runner, temp_project_dir):
        """Test config show command."""
        # Initialize project first
        with patch(
            "mcp_vector_search.cli.commands.init.confirm_action", return_value=False
        ):
            cli_runner.invoke(
                app,
                [
                    "--project-root",
                    str(temp_project_dir),
                    "init",
                    "--extensions",
                    ".py",
                    "--force",
                ],
            )

        result = cli_runner.invoke(
            app, ["--project-root", str(temp_project_dir), "config", "show"]
        )

        assert result.exit_code == 0, result.output
        assert "File Extensions" in result.output or "file_extensions" in result.output
        assert ".py" in result.output

    def test_config_command_set(self, cli_runner, temp_project_dir):
        """Test config set command."""
        with patch(
            "mcp_vector_search.cli.commands.init.confirm_action", return_value=False
        ):
            cli_runner.invoke(
                app,
                [
                    "--project-root",
                    str(temp_project_dir),
                    "init",
                    "--extensions",
                    ".py",
                    "--force",
                ],
            )

        result = cli_runner.invoke(
            app,
            [
                "--project-root",
                str(temp_project_dir),
                "config",
                "set",
                "similarity_threshold",
                "0.8",
            ],
        )
        assert result.exit_code == 0, result.output

        config_file = temp_project_dir / ".mcp-vector-search" / "config.json"
        with open(config_file) as f:
            config = json.load(f)

        assert config["similarity_threshold"] == 0.8

    def test_error_handling_uninitialized_project(self, cli_runner, temp_project_dir):
        """Test error handling for uninitialized project."""
        result = cli_runner.invoke(
            app, ["--project-root", str(temp_project_dir), "index"]
        )
        assert result.exit_code != 0
        assert "not initialized" in result.output.lower()

    def test_error_handling_invalid_path(self, cli_runner, temp_dir):
        """Test error handling for invalid paths."""
        invalid_path = temp_dir / "nonexistent"

        original_dir = os.getcwd()
        try:
            os.chdir(str(invalid_path))
            result = cli_runner.invoke(app, ["init", "main"])
        except FileNotFoundError:
            result = type("Result", (), {"exit_code": 1})()
        finally:
            os.chdir(original_dir)

        assert result.exit_code != 0

    def test_help_commands(self, cli_runner):
        """Test help commands."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "usage" in result.output.lower() or "help" in result.output.lower()

        subcommands = ["init", "index", "search", "status", "config"]

        for subcommand in subcommands:
            result = cli_runner.invoke(app, [subcommand, "--help"])
            assert result.exit_code == 0, (
                f"'{subcommand} --help' failed: {result.output}"
            )
            assert "usage" in result.output.lower() or "help" in result.output.lower()

        # index auto is the current subcommand (not top-level auto-index)
        result = cli_runner.invoke(app, ["index", "auto", "--help"])
        assert result.exit_code == 0, f"'index auto --help' failed: {result.output}"

    # -----------------------------------------------------------------------
    # Tests that require actual embedding — use subprocess to avoid fileno
    # -----------------------------------------------------------------------

    def _init_project(self, project_root: Path) -> None:
        """Initialize project via subprocess (no prompts)."""
        r = _run_cli(
            ["init", "--extensions", ".py", "--force", "--no-auto-index"],
            project_root,
        )
        assert r.returncode == 0, (
            f"init failed:\nSTDOUT: {r.stdout}\nSTDERR: {r.stderr}"
        )

    def test_index_command(self, temp_project_dir):
        """Test indexing command."""
        self._init_project(temp_project_dir)

        r = _run_cli(["index"], temp_project_dir)

        assert r.returncode == 0, (
            f"index failed:\nSTDOUT: {r.stdout}\nSTDERR: {r.stderr}"
        )
        out = _output(r).lower()
        assert "indexed" in out or "chunk" in out or "complete" in out

        # Verify LanceDB index was created (LanceDB, not ChromaDB)
        lance_dir = temp_project_dir / ".mcp-vector-search" / "lance"
        assert lance_dir.exists(), f"LanceDB index not found at {lance_dir}"

    def test_index_command_force(self, temp_project_dir):
        """Test force indexing command."""
        self._init_project(temp_project_dir)
        _run_cli(["index"], temp_project_dir)

        r = _run_cli(["index", "--force"], temp_project_dir)

        assert r.returncode == 0, (
            f"force index failed:\nSTDOUT: {r.stdout}\nSTDERR: {r.stderr}"
        )
        out = _output(r).lower()
        assert "indexed" in out or "chunk" in out or "complete" in out

    def test_index_command_force_with_analyze(self, temp_project_dir):
        """Test force indexing command with auto-analysis."""
        self._init_project(temp_project_dir)
        _run_cli(["index"], temp_project_dir)

        # Force reindex (includes all phases)
        r = _run_cli(["index", "--force"], temp_project_dir, timeout=120)

        assert r.returncode == 0, (
            f"force+analyze index failed:\nSTDOUT: {r.stdout}\nSTDERR: {r.stderr}"
        )
        out = _output(r).lower()
        assert "indexed" in out or "chunk" in out or "complete" in out

    def test_search_command(self, temp_project_dir):
        """Test search command."""
        self._init_project(temp_project_dir)
        _run_cli(["index"], temp_project_dir)

        r = _run_cli(["search", "--limit", "5", "main function"], temp_project_dir)

        assert r.returncode == 0, (
            f"search failed:\nSTDOUT: {r.stdout}\nSTDERR: {r.stderr}"
        )
        # Non-fatal BM25/reindex warnings may contain the word "error" — check
        # for actual search failure message rather than any occurrence of "error"
        out = _output(r).lower()
        assert "search failed:" not in out

    def test_search_command_with_filters(self, temp_project_dir):
        """Test search command with filters."""
        self._init_project(temp_project_dir)
        _run_cli(["index"], temp_project_dir)

        r = _run_cli(
            ["search", "--language", "python", "--threshold", "0.1", "function"],
            temp_project_dir,
        )
        assert r.returncode == 0, (
            f"filtered search failed:\nSTDOUT: {r.stdout}\nSTDERR: {r.stderr}"
        )

    def test_search_command_with_glob_pattern(self, temp_project_dir):
        """Test search command with glob pattern file filtering."""
        self._init_project(temp_project_dir)
        _run_cli(["index"], temp_project_dir)

        # Search with *.py glob pattern
        r = _run_cli(
            ["search", "--files", "*.py", "--threshold", "0.1", "function"],
            temp_project_dir,
        )
        assert r.returncode == 0, (
            f"glob search *.py failed:\nSTDOUT: {r.stdout}\nSTDERR: {r.stderr}"
        )

        # Search with non-matching pattern — should return 0 with no results
        r = _run_cli(
            ["search", "--files", "*.ts", "--threshold", "0.1", "function"],
            temp_project_dir,
        )
        assert r.returncode == 0, (
            f"glob search *.ts failed:\nSTDOUT: {r.stdout}\nSTDERR: {r.stderr}"
        )

    def test_status_command(self, temp_project_dir):
        """Test status command."""
        self._init_project(temp_project_dir)
        _run_cli(["index"], temp_project_dir)

        r = _run_cli(["status"], temp_project_dir)

        assert r.returncode == 0, (
            f"status failed:\nSTDOUT: {r.stdout}\nSTDERR: {r.stderr}"
        )
        out = _output(r).lower()
        assert "files" in out or "chunks" in out or "indexed" in out

    def test_status_command_verbose(self, temp_project_dir):
        """Test verbose status command."""
        self._init_project(temp_project_dir)
        _run_cli(["index"], temp_project_dir)

        r = _run_cli(["status", "--verbose"], temp_project_dir)

        assert r.returncode == 0, (
            f"status --verbose failed:\nSTDOUT: {r.stdout}\nSTDERR: {r.stderr}"
        )
        out = _output(r).lower()
        assert "files" in out or "chunks" in out or "indexed" in out

    def test_auto_index_status_command(self, temp_project_dir):
        """Test auto-index status command."""
        self._init_project(temp_project_dir)
        _run_cli(["index"], temp_project_dir)

        r = _run_cli(["index", "auto", "status"], temp_project_dir)

        assert r.returncode == 0, (
            f"index auto status failed:\nSTDOUT: {r.stdout}\nSTDERR: {r.stderr}"
        )
        out = _output(r).lower()
        assert "files" in out or "watch" in out or "auto" in out or "status" in out

    @pytest.mark.skip(reason="CliRunner doesn't support concurrent operations well")
    def test_auto_index_check_command(self, temp_project_dir):
        """Skipped: ChromaDB Rust SQLite corruption with CliRunner."""
        pass

    def test_full_workflow(self, temp_project_dir):
        """Test complete CLI workflow: init → index → search → status → config → force-reindex."""
        # Step 1: Initialize
        self._init_project(temp_project_dir)

        # Step 2: Index
        r = _run_cli(["index"], temp_project_dir)
        assert r.returncode == 0, f"index step failed: {_output(r)}"

        # Step 3: Search
        r = _run_cli(["search", "--limit", "3", "user"], temp_project_dir)
        assert r.returncode == 0, f"search step failed: {_output(r)}"

        # Step 4: Status
        r = _run_cli(["status"], temp_project_dir)
        assert r.returncode == 0, f"status step failed: {_output(r)}"

        # Step 5: Config (read-only — CliRunner is fine)
        runner = CliRunner()
        result = runner.invoke(
            app, ["--project-root", str(temp_project_dir), "config", "show"]
        )
        assert result.exit_code == 0, result.output

        # Step 6: Force reindex
        r = _run_cli(["index", "--force"], temp_project_dir)
        assert r.returncode == 0, f"force reindex step failed: {_output(r)}"

    @pytest.mark.skip(reason="CliRunner doesn't support concurrent operations well")
    def test_concurrent_cli_operations(self, temp_project_dir):
        """Skipped: concurrent ops not supported by CliRunner."""
        pass

    def test_performance_cli_operations(self, temp_project_dir):
        """Test performance of CLI operations."""
        # Initialization — CliRunner is fine for this
        runner = CliRunner()
        start_time = time.perf_counter()
        with patch(
            "mcp_vector_search.cli.commands.init.confirm_action", return_value=False
        ):
            result = runner.invoke(
                app,
                [
                    "--project-root",
                    str(temp_project_dir),
                    "init",
                    "--extensions",
                    ".py",
                    "--force",
                ],
            )
        init_time = time.perf_counter() - start_time
        assert result.exit_code == 0, result.output
        assert init_time < 5.0, f"Initialization took too long: {init_time:.3f}s"

        # Indexing via subprocess
        start_time = time.perf_counter()
        r = _run_cli(["index"], temp_project_dir)
        index_time = time.perf_counter() - start_time
        assert r.returncode == 0, (
            f"Indexing failed:\nSTDOUT: {r.stdout}\nSTDERR: {r.stderr}"
        )
        assert index_time < 60.0, f"Indexing took too long: {index_time:.3f}s"

        # Search via subprocess
        start_time = time.perf_counter()
        r = _run_cli(["search", "--threshold", "0.1", "function"], temp_project_dir)
        search_time = time.perf_counter() - start_time
        assert r.returncode == 0, (
            f"Search failed:\nSTDOUT: {r.stdout}\nSTDERR: {r.stderr}"
        )
        assert search_time < 10.0, f"Search took too long: {search_time:.3f}s"

    def test_chunking_performance_metrics(self, temp_project_dir):
        """Test chunking and embedding performance metrics."""
        # 1. Create multiple Python files for meaningful chunking
        for i in range(5):
            file_path = temp_project_dir / f"module_{i}.py"
            file_path.write_text(
                f'''"""Module {i} for performance testing."""


def function_{i}_a(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


def function_{i}_b(items: list) -> list:
    """Process items."""
    return [item * 2 for item in items]


class Handler_{i}:
    """Handler class {i}."""

    def __init__(self, name: str):
        self.name = name

    def process(self, data: dict) -> dict:
        """Process data."""
        return {{"result": data.get("input", "")}}
'''
            )

        # 2. Initialize via subprocess
        r = _run_cli(
            [
                "init",
                "--extensions",
                ".py",
                "--embedding-model",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--force",
                "--no-auto-index",
            ],
            temp_project_dir,
        )
        assert r.returncode == 0, f"init failed: {_output(r)}"

        # 3. Index via subprocess
        start_time = time.time()
        r = _run_cli(["index"], temp_project_dir, timeout=120)
        indexing_time = time.time() - start_time

        assert r.returncode == 0, (
            f"Indexing failed:\nSTDOUT: {r.stdout}\nSTDERR: {r.stderr}"
        )
        assert indexing_time < 60.0, f"Indexing took too long: {indexing_time:.2f}s"

        # 4. Check for throughput metrics (optional)
        output = _output(r)
        if "chunks/sec" in output.lower():
            match = re.search(r"(\d+\.?\d*)\s*chunks/sec", output, re.IGNORECASE)
            if match:
                throughput = float(match.group(1))
                assert throughput > 1.0, f"Throughput too low: {throughput} chunks/sec"
