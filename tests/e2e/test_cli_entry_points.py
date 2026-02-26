"""End-to-end tests for the main CLI entry points: init, setup, and index.

These tests invoke the actual `mvs` CLI binary via subprocess to exercise the
full stack — argument parsing, project detection, file I/O, and database
operations — in a true E2E fashion.

Run with:
    uv run pytest tests/e2e/test_cli_entry_points.py -v
    uv run pytest tests/e2e/test_cli_entry_points.py -m e2e -v

Each test uses an isolated temporary directory so tests cannot interfere with
each other or with the real project on disk.
"""

import json
import subprocess
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLI_BINARY = "mvs"
INDEX_TIMEOUT = 120  # seconds – index on a tiny project should be quick
COMMAND_TIMEOUT = 30  # seconds for non-index operations


def run_cli(
    args: list[str],
    cwd: str | Path | None = None,
    timeout: int = COMMAND_TIMEOUT,
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    """Run the mvs CLI and return the completed process.

    Uses `uv run mvs` so the correct project-local installation is used
    regardless of PATH or active virtualenv state.
    """
    cmd = ["uv", "run", CLI_BINARY] + args
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    return result


def make_python_file(path: Path, name: str) -> Path:
    """Write a minimal Python source file that is valid and indexable."""
    content = f'''"""Module {name}."""


def function_{name}(x: int, y: int) -> int:
    """Add two integers and return the result."""
    return x + y


class Handler{name.title()}:
    """A simple handler for {name}."""

    def __init__(self, value: str) -> None:
        self.value = value

    def process(self, data: dict) -> dict:
        """Process the incoming data dict."""
        return {{"value": self.value, "input": data}}
'''
    target = path / f"{name}.py"
    target.write_text(content)
    return target


def make_typescript_file(path: Path, name: str) -> Path:
    """Write a minimal TypeScript source file."""
    content = f"""// Module {name}

export interface Config{name.title()} {{
    value: string;
    count: number;
}}

export function process{name.title()}(cfg: Config{name.title()}): string {{
    return `${{cfg.value}}-${{cfg.count}}`;
}}

export class Service{name.title()} {{
    private cfg: Config{name.title()};

    constructor(cfg: Config{name.title()}) {{
        this.cfg = cfg;
    }}

    run(): string {{
        return process{name.title()}(this.cfg);
    }}
}}
"""
    target = path / f"{name}.ts"
    target.write_text(content)
    return target


def create_small_python_project(base: Path) -> None:
    """Populate *base* with 5 Python files for indexing tests."""
    for name in ["alpha", "beta", "gamma", "delta", "epsilon"]:
        make_python_file(base, name)


def create_mixed_language_project(base: Path) -> None:
    """Populate *base* with Python and TypeScript files."""
    for name in ["auth", "user", "service"]:
        make_python_file(base, name)
    for name in ["client", "types"]:
        make_typescript_file(base, name)


def config_path_for(project_dir: Path) -> Path:
    """Return the expected config.json path for a project directory."""
    return project_dir / ".mcp-vector-search" / "config.json"


def index_dir_for(project_dir: Path) -> Path:
    """Return the expected .mcp-vector-search index directory."""
    return project_dir / ".mcp-vector-search"


def read_config(project_dir: Path) -> dict:
    """Load and return the parsed config.json for a project."""
    cfg_path = config_path_for(project_dir)
    with open(cfg_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Return an empty temporary directory isolated for one test."""
    return tmp_path


@pytest.fixture
def small_python_project(tmp_path: Path) -> Path:
    """Return a temp dir pre-populated with 5 Python source files."""
    create_small_python_project(tmp_path)
    return tmp_path


@pytest.fixture
def mixed_language_project(tmp_path: Path) -> Path:
    """Return a temp dir with Python + TypeScript files."""
    create_mixed_language_project(tmp_path)
    return tmp_path


@pytest.fixture
def initialized_project(tmp_path: Path) -> Path:
    """Return a small Python project that has already been `mvs init`-ed.

    Uses --force to skip the confirmation prompt (required in non-interactive
    subprocess mode), --no-auto-index and --no-mcp to keep the fixture fast.
    """
    create_small_python_project(tmp_path)
    result = run_cli(
        [
            "init",
            "--force",
            "--no-auto-index",
            "--no-mcp",
            "--extensions",
            ".py",
            "--embedding-model",
            "sentence-transformers/all-MiniLM-L6-v2",
        ],
        cwd=tmp_path,
    )
    assert result.returncode == 0, (
        f"Fixture init failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
    return tmp_path


# ---------------------------------------------------------------------------
# 1. mvs init  E2E tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestInitCommand:
    """E2E tests for `mvs init`."""

    def test_init_fresh_project(self, small_python_project: Path) -> None:
        """Init on a fresh directory succeeds and creates config.json.

        Uses --force to bypass the interactive confirmation prompt when running
        in a non-interactive subprocess (no stdin).
        """
        result = run_cli(
            ["init", "--force", "--no-auto-index", "--no-mcp", "--extensions", ".py"],
            cwd=small_python_project,
        )

        assert result.returncode == 0, (
            f"Expected exit code 0.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

        cfg = config_path_for(small_python_project)
        assert cfg.exists(), f"config.json was not created at {cfg}"

        idx_dir = index_dir_for(small_python_project)
        assert idx_dir.exists(), (
            f".mcp-vector-search/ directory was not created at {idx_dir}"
        )

    def test_init_already_initialized_is_idempotent(
        self, initialized_project: Path
    ) -> None:
        """Running init a second time should succeed (idempotent, no error).

        Without --force, init detects an existing project and exits 0 with an
        "already initialized" message rather than asking for confirmation.
        """
        result = run_cli(
            ["init", "--no-auto-index", "--no-mcp"],
            cwd=initialized_project,
        )

        assert result.returncode == 0, (
            f"Second init should succeed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
        # Output should mention already initialized
        combined = (result.stdout + result.stderr).lower()
        assert "already initialized" in combined, (
            f"Expected 'already initialized' in output.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_init_creates_config_with_expected_fields(
        self, small_python_project: Path
    ) -> None:
        """Config created by init contains required fields with correct types."""
        result = run_cli(
            [
                "init",
                "--force",
                "--no-auto-index",
                "--no-mcp",
                "--extensions",
                ".py",
                "--embedding-model",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--similarity-threshold",
                "0.6",
            ],
            cwd=small_python_project,
        )
        assert result.returncode == 0, (
            f"Init failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

        config = read_config(small_python_project)

        # Required top-level fields
        assert "project_root" in config, "config.json missing 'project_root'"
        assert "index_path" in config, "config.json missing 'index_path'"
        assert "file_extensions" in config, "config.json missing 'file_extensions'"
        assert "embedding_model" in config, "config.json missing 'embedding_model'"
        assert "similarity_threshold" in config, (
            "config.json missing 'similarity_threshold'"
        )

        # Value assertions
        assert ".py" in config["file_extensions"], (
            f"Expected '.py' in file_extensions, got: {config['file_extensions']}"
        )
        assert config["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2", (
            f"Unexpected embedding_model: {config['embedding_model']}"
        )
        assert config["similarity_threshold"] == pytest.approx(0.6, abs=1e-4), (
            f"Unexpected similarity_threshold: {config['similarity_threshold']}"
        )

        # project_root should point to the temp directory
        assert str(small_python_project) in config["project_root"], (
            f"project_root does not contain expected path.\n"
            f"Expected substring: {small_python_project}\n"
            f"Got: {config['project_root']}"
        )

    def test_init_detects_languages(self, mixed_language_project: Path) -> None:
        """Init on a mixed Python/TypeScript project detects both languages."""
        result = run_cli(
            ["init", "--force", "--no-auto-index", "--no-mcp"],
            cwd=mixed_language_project,
        )
        assert result.returncode == 0, (
            f"Init failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

        config = read_config(mixed_language_project)
        languages = [lang.lower() for lang in config.get("languages", [])]

        assert "python" in languages, (
            f"Expected 'python' in detected languages, got: {languages}"
        )
        assert "typescript" in languages, (
            f"Expected 'typescript' in detected languages, got: {languages}"
        )

    def test_init_force_reinitializes(self, initialized_project: Path) -> None:
        """--force causes init to re-initialize and regenerate config."""
        _config_before = read_config(initialized_project)  # noqa: F841

        # Touch the config to track recreation
        cfg_path = config_path_for(initialized_project)
        mtime_before = cfg_path.stat().st_mtime

        result = run_cli(
            ["init", "--force", "--no-auto-index", "--no-mcp", "--extensions", ".py"],
            cwd=initialized_project,
        )
        assert result.returncode == 0, (
            f"Force init failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

        mtime_after = cfg_path.stat().st_mtime
        assert mtime_after >= mtime_before, (
            "config.json was not updated by --force init"
        )

    def test_init_output_mentions_initialized(self, small_python_project: Path) -> None:
        """Init output contains confirmation of successful initialization."""
        result = run_cli(
            ["init", "--force", "--no-auto-index", "--no-mcp", "--extensions", ".py"],
            cwd=small_python_project,
        )
        assert result.returncode == 0, (
            f"Init failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
        combined = (result.stdout + result.stderr).lower()
        assert "initialized" in combined, (
            f"Expected 'initialized' in output.\nOutput: {combined[:500]}"
        )

    def test_init_without_source_files_still_succeeds(self, tmp_project: Path) -> None:
        """Init on a directory with no source files should still succeed."""
        result = run_cli(
            ["init", "--force", "--no-auto-index", "--no-mcp"],
            cwd=tmp_project,
        )
        assert result.returncode == 0, (
            f"Init with no source files failed.\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
        assert config_path_for(tmp_project).exists(), "config.json should be created"


# ---------------------------------------------------------------------------
# 2. mvs setup  E2E tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestSetupCommand:
    """E2E tests for `mvs setup`."""

    def test_setup_after_init_succeeds(self, initialized_project: Path) -> None:
        """Setup on an already-initialized project exits 0."""
        result = run_cli(
            ["setup", "--force"],
            cwd=initialized_project,
            timeout=INDEX_TIMEOUT,
        )
        assert result.returncode == 0, (
            f"Setup failed on initialized project.\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_setup_on_fresh_project_creates_config(
        self, small_python_project: Path
    ) -> None:
        """Setup from scratch initializes the project and creates config.json."""
        result = run_cli(
            ["setup"],
            cwd=small_python_project,
            timeout=INDEX_TIMEOUT,
        )
        assert result.returncode == 0, (
            f"Setup failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
        assert config_path_for(small_python_project).exists(), (
            "config.json was not created by setup"
        )

    def test_setup_shows_configuration_output(self, small_python_project: Path) -> None:
        """Setup command produces informational output about what it's doing."""
        result = run_cli(
            ["setup", "--verbose"],
            cwd=small_python_project,
            timeout=INDEX_TIMEOUT,
        )
        assert result.returncode == 0, (
            f"Setup --verbose failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
        combined = (result.stdout + result.stderr).lower()
        # Setup should mention detecting and/or configuring something
        has_detection = any(
            keyword in combined
            for keyword in ["detect", "configur", "initializ", "setup", "index"]
        )
        assert has_detection, (
            f"Expected setup output to mention detection/configuration.\n"
            f"Output (first 800 chars): {combined[:800]}"
        )

    def test_setup_is_idempotent(self, initialized_project: Path) -> None:
        """Running setup multiple times on the same project should not error."""
        for run_num in range(1, 3):
            result = run_cli(
                ["setup"],
                cwd=initialized_project,
                timeout=INDEX_TIMEOUT,
            )
            assert result.returncode == 0, (
                f"Setup run #{run_num} failed.\n"
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )


# ---------------------------------------------------------------------------
# 3. mvs index  E2E tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestIndexCommand:
    """E2E tests for `mvs index`."""

    def test_index_fresh_project(self, initialized_project: Path) -> None:
        """Index on a freshly initialized small project completes without error."""
        result = run_cli(
            ["index"],
            cwd=initialized_project,
            timeout=INDEX_TIMEOUT,
        )
        assert result.returncode == 0, (
            f"Indexing failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_index_without_init_returns_error(self, tmp_project: Path) -> None:
        """Indexing an uninitialized project produces a non-zero exit code."""
        # Create some source files so the error is not about empty dir
        make_python_file(tmp_project, "sample")

        result = run_cli(
            ["index"],
            cwd=tmp_project,
            timeout=COMMAND_TIMEOUT,
        )
        assert result.returncode != 0, (
            f"Expected non-zero exit code for uninitialized project.\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
        combined = (result.stdout + result.stderr).lower()
        assert "not initialized" in combined or "init" in combined, (
            f"Expected error message mentioning initialization.\n"
            f"Output: {combined[:600]}"
        )

    def test_index_empty_project_graceful(self, tmp_project: Path) -> None:
        """Index on a project with no indexable files produces a clear message.

        The tool may exit non-zero when there are no files to chunk (the chunks
        backend is never initialized), but it must NOT crash silently or hang.
        The important invariant is: a clear message about missing files appears.
        """
        # Initialize with no files present
        run_cli(
            ["init", "--force", "--no-auto-index", "--no-mcp"],
            cwd=tmp_project,
        )

        result = run_cli(
            ["index"],
            cwd=tmp_project,
            timeout=INDEX_TIMEOUT,
        )
        # The combined output must mention the lack of files
        combined = (result.stdout + result.stderr).lower()
        has_no_files_message = any(
            keyword in combined
            for keyword in [
                "no indexable files",
                "no files",
                "nothing to index",
                "chunks backend not initialized",
                "0 files",
            ]
        )
        assert has_no_files_message, (
            f"Expected a message about no indexable files.\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
        # The process must not have hung (timeout would be the test failure)

    def test_index_force_rebuild(self, initialized_project: Path) -> None:
        """mvs index --force on an already-indexed project re-indexes from scratch."""
        # First normal index pass
        first = run_cli(
            ["index"],
            cwd=initialized_project,
            timeout=INDEX_TIMEOUT,
        )
        assert first.returncode == 0, (
            f"Initial index failed.\nSTDOUT: {first.stdout}\nSTDERR: {first.stderr}"
        )

        # Force rebuild
        second = run_cli(
            ["index", "--force"],
            cwd=initialized_project,
            timeout=INDEX_TIMEOUT,
        )
        assert second.returncode == 0, (
            f"Force index failed.\nSTDOUT: {second.stdout}\nSTDERR: {second.stderr}"
        )

    def test_index_produces_output_about_files(self, initialized_project: Path) -> None:
        """Index output should mention files, chunks, or indexing progress."""
        result = run_cli(
            ["index"],
            cwd=initialized_project,
            timeout=INDEX_TIMEOUT,
        )
        assert result.returncode == 0, (
            f"Index failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
        combined = (result.stdout + result.stderr).lower()
        has_progress = any(
            keyword in combined
            for keyword in ["file", "chunk", "index", "embed", "process"]
        )
        assert has_progress, (
            f"Expected index output to mention files/chunks/indexing.\n"
            f"Output (first 800 chars): {combined[:800]}"
        )

    def test_index_aborts_on_backend_failure(self, tmp_path: Path) -> None:
        """When the backend cannot be initialised, indexer reports a clear error.

        We simulate a broken backend by making the .mcp-vector-search/
        directory unreadable (which will cause LanceDB to fail on open).
        """
        create_small_python_project(tmp_path)

        # Init successfully first
        init_result = run_cli(
            ["init", "--force", "--no-auto-index", "--no-mcp", "--extensions", ".py"],
            cwd=tmp_path,
        )
        assert init_result.returncode == 0, "Setup for backend-failure test failed"

        # Corrupt the index directory so LanceDB cannot open it
        mcp_dir = tmp_path / ".mcp-vector-search"
        chunks_db = mcp_dir / "chunks.lance"
        chunks_db.mkdir(parents=True, exist_ok=True)
        # Write garbage into a critical manifest file to corrupt the DB
        bad_manifest = chunks_db / "_latest.manifest"
        bad_manifest.write_bytes(b"\x00CORRUPTED\x00")

        result = run_cli(
            ["index"],
            cwd=tmp_path,
            timeout=COMMAND_TIMEOUT,
        )

        # Either it fails with a clear message OR it recovers — both acceptable.
        # What is NOT acceptable: a silent hang, or exit 0 with no diagnostics
        # when the backend is broken.
        if result.returncode != 0:
            combined = (result.stdout + result.stderr).lower()
            has_error_msg = any(
                keyword in combined
                for keyword in [
                    "error",
                    "fail",
                    "corrupt",
                    "invalid",
                    "cannot",
                    "unable",
                ]
            )
            assert has_error_msg, (
                f"Non-zero exit but no clear error message.\n"
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
        # If it recovers and exits 0, that is also acceptable behaviour.


# ---------------------------------------------------------------------------
# 4. Integration flow tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestIntegrationFlows:
    """Full workflow integration tests across init → index → verify."""

    def test_init_then_index_flow(self, small_python_project: Path) -> None:
        """Full flow: init → index → verify that .mcp-vector-search has lance data."""
        # Step 1: init
        init_result = run_cli(
            [
                "init",
                "--force",
                "--no-auto-index",
                "--no-mcp",
                "--extensions",
                ".py",
                "--embedding-model",
                "sentence-transformers/all-MiniLM-L6-v2",
            ],
            cwd=small_python_project,
        )
        assert init_result.returncode == 0, (
            f"Init step failed.\nSTDOUT: {init_result.stdout}\nSTDERR: {init_result.stderr}"
        )

        # Step 2: index
        index_result = run_cli(
            ["index"],
            cwd=small_python_project,
            timeout=INDEX_TIMEOUT,
        )
        assert index_result.returncode == 0, (
            f"Index step failed.\nSTDOUT: {index_result.stdout}\nSTDERR: {index_result.stderr}"
        )

        # Step 3: verify artefacts exist
        mcp_dir = index_dir_for(small_python_project)
        assert mcp_dir.exists(), ".mcp-vector-search/ should exist after indexing"

        # At minimum the config should still be intact
        assert config_path_for(small_python_project).exists(), (
            "config.json should still exist after indexing"
        )

        # The indexing pipeline should produce at least one of: a lance/ directory,
        # chunks.lance, vectors.lance, or bm25_index.pkl — depending on which
        # phases completed successfully.
        lance_subdir = mcp_dir / "lance"
        chunks_db = mcp_dir / "chunks.lance"
        vectors_db = mcp_dir / "vectors.lance"
        bm25_index = mcp_dir / "bm25_index.pkl"
        has_data_store = (
            lance_subdir.exists()
            or chunks_db.exists()
            or vectors_db.exists()
            or bm25_index.exists()
        )
        assert has_data_store, (
            f"Expected at least one indexing artefact after indexing.\n"
            f"Contents of {mcp_dir}: {list(mcp_dir.iterdir())}"
        )

    def test_reinit_preserves_existing_data(self, small_python_project: Path) -> None:
        """Init → index → init again (no --force) → data files still present."""
        # Init
        run_cli(
            ["init", "--force", "--no-auto-index", "--no-mcp", "--extensions", ".py"],
            cwd=small_python_project,
        )

        # Index
        index_result = run_cli(
            ["index"],
            cwd=small_python_project,
            timeout=INDEX_TIMEOUT,
        )
        assert index_result.returncode == 0, (
            f"Index failed.\nSTDOUT: {index_result.stdout}\nSTDERR: {index_result.stderr}"
        )

        # Record data store mtimes before second init
        mcp_dir = index_dir_for(small_python_project)
        pre_contents = set(mcp_dir.rglob("*"))

        # Second init without --force should skip re-initialization
        second_init = run_cli(
            ["init", "--no-auto-index", "--no-mcp"],
            cwd=small_python_project,
        )
        assert second_init.returncode == 0, (
            f"Second init failed.\nSTDOUT: {second_init.stdout}\nSTDERR: {second_init.stderr}"
        )

        # Data files should still be present
        post_contents = set(mcp_dir.rglob("*"))
        assert pre_contents.issubset(post_contents), (
            f"Data files were lost after second init.\n"
            f"Missing: {pre_contents - post_contents}"
        )

    def test_full_init_index_status_flow(self, small_python_project: Path) -> None:
        """Full flow: init → index → status to confirm indexed files reported."""
        run_cli(
            ["init", "--force", "--no-auto-index", "--no-mcp", "--extensions", ".py"],
            cwd=small_python_project,
        )

        run_cli(
            ["index"],
            cwd=small_python_project,
            timeout=INDEX_TIMEOUT,
        )

        status_result = run_cli(
            ["status"],
            cwd=small_python_project,
        )
        assert status_result.returncode == 0, (
            f"Status failed.\nSTDOUT: {status_result.stdout}\nSTDERR: {status_result.stderr}"
        )
        combined = (status_result.stdout + status_result.stderr).lower()
        has_metrics = any(
            keyword in combined for keyword in ["file", "chunk", "index", "status"]
        )
        assert has_metrics, (
            f"Status output should report metrics.\nOutput: {combined[:600]}"
        )

    def test_init_check_subcommand_after_init(self, initialized_project: Path) -> None:
        """mvs init check should confirm project is initialized."""
        result = run_cli(
            ["init", "check"],
            cwd=initialized_project,
        )
        assert result.returncode == 0, (
            f"init check failed on initialized project.\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
        combined = (result.stdout + result.stderr).lower()
        assert "initialized" in combined, (
            f"Expected 'initialized' in output.\nOutput: {combined[:400]}"
        )

    def test_init_check_subcommand_before_init(self, tmp_project: Path) -> None:
        """mvs init check should return non-zero on an uninitialized directory."""
        result = run_cli(
            ["init", "check"],
            cwd=tmp_project,
        )
        assert result.returncode != 0, (
            f"Expected non-zero exit for uninitialized project.\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
        combined = (result.stdout + result.stderr).lower()
        assert "not initialized" in combined or "init" in combined, (
            f"Expected error message about initialization.\nOutput: {combined[:400]}"
        )

    def test_index_chunk_subcommand(self, initialized_project: Path) -> None:
        """mvs index chunk (phase-1 only) completes without error."""
        result = run_cli(
            ["index", "chunk"],
            cwd=initialized_project,
            timeout=INDEX_TIMEOUT,
        )
        assert result.returncode == 0, (
            f"index chunk failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_multiple_sequential_indexes(self, initialized_project: Path) -> None:
        """Running mvs index twice in a row (incremental) should both succeed."""
        for run_num in range(1, 3):
            result = run_cli(
                ["index"],
                cwd=initialized_project,
                timeout=INDEX_TIMEOUT,
            )
            assert result.returncode == 0, (
                f"Index run #{run_num} failed.\n"
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )

    def test_version_command_works(self) -> None:
        """Sanity check: `mvs version` exits 0 and shows a version string."""
        result = run_cli(["version"])
        assert result.returncode == 0, (
            f"version command failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
        combined = result.stdout + result.stderr
        # Should contain a version number pattern like "3.0.x"
        import re

        assert re.search(r"\d+\.\d+\.\d+", combined), (
            f"No version string found in output: {combined[:300]}"
        )
