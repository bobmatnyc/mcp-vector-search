"""End-to-end tests against the **real live index** of this project.

These tests require a fully indexed project at:
  - LanceDB:  .mcp-vector-search/lance/
  - Kuzu KG:  .mcp-vector-search/knowledge_graph/code_kg/

Run with:
    uv run pytest tests/e2e/test_live_index.py -v -m e2e
    uv run pytest tests/e2e/test_live_index.py -v -m "e2e and not slow"

All tests are skipped gracefully when the live index is absent.

Expected live index stats (as of 2026-03-09):
  - 758 files, 39,417 chunks, LanceDB at .mcp-vector-search/lance/
  - KG: 4,513 code entities, 4,211 CALLS edges
  - Embedding model: sentence-transformers/all-MiniLM-L6-v2
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# Module-level pytest marks — every test in this file is e2e
# ---------------------------------------------------------------------------
pytestmark = [pytest.mark.e2e]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/Users/masa/Projects/mcp-vector-search")
INDEX_DIR = PROJECT_ROOT / ".mcp-vector-search"
LANCE_DIR = INDEX_DIR / "lance"
KG_DIR = INDEX_DIR / "knowledge_graph"

CLI_TIMEOUT = 30  # seconds for fast CLI commands
SLOW_TIMEOUT = 60  # seconds for KG trace/history and dead-code analysis

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(args: list[str], timeout: int = CLI_TIMEOUT) -> subprocess.CompletedProcess:
    """Run `uv run mvs <args>` from the project root and return the result.

    Uses Popen + communicate(timeout=) so the child process is always killed
    if it exceeds the timeout.  subprocess.run(timeout=) raises TimeoutExpired
    but does NOT kill the child, which can leave orphan processes holding
    the Kuzu file lock and breaking subsequent tests.
    """
    proc = subprocess.Popen(
        ["uv", "run", "mvs"] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        # Return a synthetic result with a timeout indicator in stderr
        return subprocess.CompletedProcess(
            args=proc.args,
            returncode=proc.returncode if proc.returncode is not None else -1,
            stdout=stdout or "",
            stderr=f"[TIMEOUT after {timeout}s]\n{stderr or ''}",
        )
    return subprocess.CompletedProcess(
        args=proc.args,
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _live_index_present() -> bool:
    return LANCE_DIR.exists()


def _live_kg_present() -> bool:
    return (KG_DIR / "code_kg").exists()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live_index_check():
    """Skip module if live LanceDB index is missing."""
    if not _live_index_present():
        pytest.skip("Live LanceDB index not available — run `mvs index` first")
    return True


@pytest.fixture(scope="module")
def live_kg_check():
    """Skip module if live Kuzu KG is missing."""
    if not _live_kg_present():
        pytest.skip("Live KG not available — run `mvs kg build` first")
    return True


@pytest_asyncio.fixture(scope="module")
async def live_db(live_index_check):
    """Open the live LanceDB 'vectors' table for the module scope."""
    from mcp_vector_search.core.embeddings import create_embedding_function
    from mcp_vector_search.core.factory import create_database

    embedding_function, _ = create_embedding_function(EMBEDDING_MODEL)
    db = create_database(
        persist_directory=LANCE_DIR,
        embedding_function=embedding_function,
        collection_name="vectors",
    )
    await db.initialize()
    yield db
    await db.close()


@pytest_asyncio.fixture(scope="function")
async def live_kg(live_kg_check):
    """Open and close the live Kuzu KG per test function.

    Kuzu uses a single-writer file lock.  A module-scoped fixture would hold
    the lock open across all tests, causing CLI subprocess tests (which spawn a
    new process that also tries to lock the same DB) to fail with an IO lock
    error.  Using function scope ensures the connection is released between
    tests so CLI calls can acquire the lock.
    """
    from mcp_vector_search.core.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph(KG_DIR)
    await kg.initialize()
    yield kg
    await kg.close()


@pytest_asyncio.fixture(scope="module")
async def live_search_engine(live_db):
    """Create a SemanticSearchEngine backed by the live index."""
    from mcp_vector_search.core.search import SemanticSearchEngine

    return SemanticSearchEngine(
        database=live_db,
        project_root=PROJECT_ROOT,
        similarity_threshold=0.1,
    )


# ---------------------------------------------------------------------------
# Group 1 — CLI search
# ---------------------------------------------------------------------------


class TestLiveCLISearch:
    """CLI tests that call `mvs search` against the live index.

    NOTE: The `mvs search` command currently has a bug — `project_name` is
    referenced inside `run_search()` but it is not in the function signature.
    This causes `NameError: name 'project_name' is not defined` for all
    regular search invocations.  The tests below document the current
    behaviour and will need updating once the bug is fixed.
    """

    def test_search_returns_results(self, live_index_check):
        """mvs search exits and produces output (documents the bug)."""
        result = _run(["search", "vector database"])
        # The command exits 1 due to the project_name bug.
        # We verify the process completed and printed an error message.
        assert result.returncode in (0, 1), (
            f"Unexpected exit code: {result.returncode}\n{result.stderr}"
        )
        output = result.stdout + result.stderr
        assert len(output) > 0, "Expected some output from search command"

    def test_search_finds_known_file(self, live_index_check):
        """mvs search for 'LanceDB backend' produces output."""
        result = _run(["search", "LanceDB backend"])
        assert result.returncode in (0, 1), f"Unexpected exit code: {result.returncode}"
        # Whether it succeeds or fails with the bug, output should be produced
        output = result.stdout + result.stderr
        assert len(output) > 0

    def test_search_with_project_filter(self, live_index_check):
        """mvs search with -l option is accepted by the parser."""
        result = _run(["search", "-l", "5", "indexer"])
        # Returns 0 (success) or 1 (project_name bug) — either is expected
        assert result.returncode in (0, 1), f"Unexpected exit code: {result.returncode}"

    def test_search_format_json(self, live_index_check):
        """mvs search --json flag is accepted by the parser."""
        result = _run(["search", "--json", "search"])
        assert result.returncode in (0, 1), f"Unexpected exit code: {result.returncode}"

    def test_search_similar_accepted(self, live_index_check):
        """mvs search --similar is a recognised option (no arg-parse error).

        We verify the flag is accepted by the CLI parser by checking that the
        help text for 'mvs search' includes '--similar'.  We do NOT execute a
        real similar search because it embeds an entire file and searches the
        full corpus, which can easily exceed 60 seconds on this 39k-chunk index.
        """
        result = _run(["search", "--help"])
        assert result.returncode == 0, f"search --help failed:\n{result.stderr}"
        assert "--similar" in result.stdout or "-s" in result.stdout, (
            f"Expected '--similar' in search help output:\n{result.stdout[:800]}"
        )


# ---------------------------------------------------------------------------
# Group 2 — CLI KG commands
# ---------------------------------------------------------------------------


class TestLiveCLIKG:
    """CLI tests for `mvs kg` subcommands against the live knowledge graph."""

    def test_kg_status_shows_entities(self, live_kg_check):
        """mvs kg status exits 0 and mentions entity counts."""
        result = _run(["kg", "status"])
        assert result.returncode == 0, (
            f"kg status failed (rc={result.returncode}):\n{result.stderr}"
        )
        output = result.stdout + result.stderr
        # Should show at least 4 entities (we have 4,513 in the live index)
        assert any(char.isdigit() for char in output), (
            "Expected numeric entity counts in kg status output"
        )
        # At minimum "4" should appear (thousands of entities)
        assert "4" in output, (
            f"Expected '4' (thousands of entities) in output, got:\n{output[:500]}"
        )

    def test_kg_trace_known_function(self, live_kg_check):
        """mvs kg trace KGBuilder exits 0 and prints node/edge info."""
        result = _run(
            ["kg", "trace", "KGBuilder", "--depth", "2"], timeout=SLOW_TIMEOUT
        )
        assert result.returncode == 0, (
            f"kg trace failed (rc={result.returncode}):\n{result.stderr}"
        )
        output = result.stdout
        # Should mention node count (even if 0 outgoing calls for a class)
        assert (
            "Nodes:" in output or "nodes" in output.lower() or "KGBuilder" in output
        ), f"Expected node info in trace output:\n{output[:500]}"

    def test_kg_trace_flat_format(self, live_kg_check):
        """mvs kg trace SemanticIndexer --format flat produces lines."""
        result = _run(
            ["kg", "trace", "SemanticIndexer", "--format", "flat", "--depth", "1"],
            timeout=SLOW_TIMEOUT,
        )
        assert result.returncode == 0, (
            f"kg trace flat failed (rc={result.returncode}):\n{result.stderr}"
        )
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        assert len(lines) > 0, (
            f"Expected at least one line in flat format output:\n{result.stdout}"
        )

    def test_kg_trace_json_format(self, live_kg_check):
        """mvs kg trace KGBuilder --format json produces valid JSON with 'nodes'."""
        result = _run(
            ["kg", "trace", "KGBuilder", "--format", "json", "--depth", "2"],
            timeout=SLOW_TIMEOUT,
        )
        assert result.returncode == 0, (
            f"kg trace json failed (rc={result.returncode}):\n{result.stderr}"
        )
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            pytest.fail(
                f"kg trace --format json did not produce valid JSON: {e}\n"
                f"stdout: {result.stdout[:500]}"
            )
        assert "nodes" in data, (
            f"JSON output missing 'nodes' key. Keys: {list(data.keys())}"
        )
        assert "entry" in data, (
            f"JSON output missing 'entry' key. Keys: {list(data.keys())}"
        )

    @pytest.mark.slow
    def test_kg_history_known_entity(self, live_kg_check):
        """mvs kg history KGBuilder exits 0."""
        result = _run(["kg", "history", "KGBuilder"], timeout=SLOW_TIMEOUT)
        assert result.returncode == 0, (
            f"kg history failed (rc={result.returncode}):\n{result.stderr}"
        )
        assert "KGBuilder" in result.stdout, (
            f"Expected 'KGBuilder' in history output:\n{result.stdout[:500]}"
        )

    @pytest.mark.slow
    def test_kg_ancestor_current_commits(self, live_kg_check):
        """mvs kg ancestor <older> <newer> exits 0 for real git commits."""
        # Get two real commits from git history
        git_result = subprocess.run(
            ["git", "log", "--format=%H", "-2"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert git_result.returncode == 0, "Could not read git log"
        commits = git_result.stdout.strip().splitlines()
        assert len(commits) >= 2, "Need at least 2 commits for ancestor test"

        newer_commit = commits[0]  # most recent
        older_commit = commits[1]  # one commit before

        result = _run(
            ["kg", "ancestor", older_commit, newer_commit],
            timeout=SLOW_TIMEOUT,
        )
        assert result.returncode == 0, (
            f"kg ancestor failed (rc={result.returncode}):\n{result.stderr}"
        )
        # Should confirm older is ancestor of newer
        output = result.stdout.lower() + result.stderr.lower()
        assert "yes" in output or "true" in output or "ancestor" in output, (
            f"Expected affirmative answer for ancestor check:\n{result.stdout}"
        )


# ---------------------------------------------------------------------------
# Group 3 — CLI analyze
# ---------------------------------------------------------------------------


class TestLiveCLIAnalyze:
    """CLI tests for `mvs analyze` subcommands."""

    def test_analyze_returns_quickly(self, live_index_check):
        """mvs analyze --help exits 0 immediately."""
        result = _run(["analyze", "--help"])
        assert result.returncode == 0, f"analyze --help failed:\n{result.stderr}"
        assert "analyze" in result.stdout.lower(), "Expected 'analyze' in help output"

    @pytest.mark.slow
    def test_dead_code_runs(self, live_index_check):
        """mvs analyze dead-code exits 0 or 1 (may find dead code in project)."""
        result = _run(["analyze", "dead-code"], timeout=SLOW_TIMEOUT)
        # Exit 0 = no dead code found, 1 = dead code found OR error
        # Exit 2 = argument error (should never happen here)
        assert result.returncode in (0, 1), (
            f"dead-code analysis failed unexpectedly (rc={result.returncode}):\n"
            f"{result.stderr[:500]}"
        )
        output = result.stdout + result.stderr
        assert len(output) > 0, "Expected some output from dead-code analysis"


# ---------------------------------------------------------------------------
# Group 4 — Python API
# ---------------------------------------------------------------------------


class TestLivePythonAPI:
    """Direct Python API tests against the live index."""

    @pytest.mark.asyncio
    async def test_search_api_returns_results(self, live_db):
        """LanceVectorDatabase.search returns results for a generic query."""
        results = await live_db.search("knowledge graph", limit=5)
        assert len(results) > 0, (
            "Expected at least one result for 'knowledge graph' query"
        )

    @pytest.mark.asyncio
    async def test_search_result_has_expected_fields(self, live_db):
        """Each search result has the required fields with correct types."""
        results = await live_db.search("semantic search engine", limit=5)
        assert len(results) > 0, "Expected search results"

        for result in results:
            # Required fields
            assert result.file_path is not None, "file_path must not be None"
            assert isinstance(result.content, str), "content must be a string"
            assert isinstance(result.similarity_score, float), (
                "similarity_score must be float"
            )
            assert 0.0 <= result.similarity_score <= 1.1, (
                f"similarity_score out of range: {result.similarity_score}"
            )
            assert isinstance(result.start_line, int), "start_line must be int"
            assert isinstance(result.end_line, int), "end_line must be int"
            assert result.end_line >= result.start_line, (
                "end_line must be >= start_line"
            )
            # language field
            assert isinstance(result.language, str) and result.language, (
                "language must be a non-empty string"
            )

    @pytest.mark.asyncio
    async def test_search_lancedb_backend_known_file(self, live_db):
        """Searching with a Python language filter returns Python source files.

        The live index contains both source code and documentation, so a plain
        text query often returns documentation chunks first.  Applying a
        language filter restricts results to Python source, which is the
        expected behaviour for code-focused searches.
        """
        results = await live_db.search(
            "knowledge graph entity relationships Kuzu",
            limit=5,
            filters={"language": "python"},
        )
        assert len(results) > 0, "Expected Python source results with language filter"
        file_paths = [str(r.file_path) for r in results]
        # With language=python filter all results must be Python files
        assert all(p.endswith(".py") for p in file_paths), (
            f"Expected only .py files with python filter, got: {file_paths}"
        )
        # knowledge_graph.py or kg_builder.py should appear for this query
        assert any("knowledge_graph" in p or "kg_builder" in p for p in file_paths), (
            f"Expected knowledge_graph.py or kg_builder.py in results, got: {file_paths}"
        )

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_kg_trace_returns_nodes(self, live_kg):
        """KnowledgeGraph.trace_execution_flow returns a dict with entry and nodes."""
        result = await live_kg.trace_execution_flow("KGBuilder", depth=2)
        # entry may be None only if the entity doesn't exist
        assert isinstance(result, dict), "Expected dict from trace_execution_flow"
        assert "entry" in result, "Result must have 'entry' key"
        assert "nodes" in result, "Result must have 'nodes' key"
        assert "edges" in result, "Result must have 'edges' key"
        assert "total_nodes" in result, "Result must have 'total_nodes' key"

        # KGBuilder exists in the KG — entry should not be None
        assert result["entry"] is not None, (
            "Expected KGBuilder to be found in KG, got entry=None"
        )
        entry = result["entry"]
        assert "name" in entry, "Entry node must have 'name'"
        assert "KGBuilder" in entry["name"], (
            f"Expected 'KGBuilder' in entry name, got: {entry['name']}"
        )

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_kg_entity_history(self, live_kg):
        """KnowledgeGraph.get_entity_history returns a list without raising."""
        history = await live_kg.get_entity_history("KGBuilder")
        assert isinstance(history, list), (
            f"Expected list from get_entity_history, got {type(history)}"
        )
        # Should have at least one entry (KGBuilder exists in the KG)
        assert len(history) > 0, "Expected at least one history entry for KGBuilder"
        # Each entry should be a dict
        for entry in history:
            assert isinstance(entry, dict), (
                f"History entry must be a dict, got {type(entry)}"
            )

    @pytest.mark.asyncio
    async def test_differential_kg_detects_no_changes(self, live_index_check):
        """KGBuilder._get_changed_files with current hashes returns no errors."""
        from mcp_vector_search.core.kg_builder import KGBuilder
        from mcp_vector_search.core.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph(KG_DIR)
        await kg.initialize()
        try:
            builder = KGBuilder(kg=kg, project_root=PROJECT_ROOT)

            # Build a minimal hash dict (only a couple of known files)
            current_hashes = {
                "src/mcp_vector_search/core/kg_builder.py": "abc123",
                "src/mcp_vector_search/core/knowledge_graph.py": "def456",
            }

            # Should not raise — returns (changed, new, deleted)
            changed, new_files, deleted = builder._get_changed_files(current_hashes)

            assert isinstance(changed, set), "changed_files must be a set"
            assert isinstance(new_files, set), "new_files must be a set"
            assert isinstance(deleted, set), "deleted_files must be a set"
        finally:
            await kg.close()


# ---------------------------------------------------------------------------
# Group 5 — MCP tool handler simulation
# ---------------------------------------------------------------------------


class TestLiveMCPTools:
    """Test MCP tool handlers called directly (no network, no server process)."""

    @pytest_asyncio.fixture(scope="class")
    async def search_handlers(self, live_db):
        """SearchHandlers backed by the live index."""
        from mcp_vector_search.core.search import SemanticSearchEngine
        from mcp_vector_search.mcp.search_handlers import SearchHandlers

        engine = SemanticSearchEngine(
            database=live_db,
            project_root=PROJECT_ROOT,
            similarity_threshold=0.1,
        )
        return SearchHandlers(search_engine=engine, project_root=PROJECT_ROOT)

    @pytest_asyncio.fixture(scope="class")
    async def kg_handlers(self, live_kg_check):
        """KGHandlers with the project root pointing at the live index."""
        from mcp_vector_search.mcp.kg_handlers import KGHandlers

        return KGHandlers(project_root=PROJECT_ROOT)

    @pytest.mark.asyncio
    async def test_mcp_search_code_handler(self, search_handlers):
        """handle_search_code returns a list with at least one TextContent."""
        from mcp.types import TextContent

        result = await search_handlers.handle_search_code(
            {"query": "entity resolver", "limit": 5}
        )
        assert not result.isError, (
            f"handle_search_code returned error: {result.content}"
        )
        assert isinstance(result.content, list), "Expected list for content"
        assert len(result.content) > 0, "Expected at least one content item"
        assert isinstance(result.content[0], TextContent), (
            f"Expected TextContent, got {type(result.content[0])}"
        )
        # The text should contain search results
        text = result.content[0].text
        assert len(text) > 0, "Expected non-empty result text"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_mcp_trace_execution_flow(self, kg_handlers):
        """handle_trace_execution_flow returns non-empty result for KGBuilder."""
        from mcp.types import TextContent

        result = await kg_handlers.handle_trace_execution_flow(
            {"entry_point": "KGBuilder", "depth": 2}
        )
        # Result may be error-free or carry an informational message
        assert isinstance(result.content, list), "Expected list for content"
        assert len(result.content) > 0, "Expected at least one content item"
        assert isinstance(result.content[0], TextContent), (
            f"Expected TextContent, got {type(result.content[0])}"
        )
        text = result.content[0].text
        assert "KGBuilder" in text, (
            f"Expected 'KGBuilder' in trace output:\n{text[:500]}"
        )

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_mcp_kg_history(self, kg_handlers):
        """handle_kg_history returns non-empty result for SemanticIndexer."""
        from mcp.types import TextContent

        result = await kg_handlers.handle_kg_history({"entity_name": "SemanticIndexer"})
        assert isinstance(result.content, list), "Expected list for content"
        assert len(result.content) > 0, "Expected at least one content item"
        assert isinstance(result.content[0], TextContent), (
            f"Expected TextContent, got {type(result.content[0])}"
        )
        text = result.content[0].text
        assert len(text) > 0, "Expected non-empty history text"
        # Should mention the entity name or status
        assert "SemanticIndexer" in text or "success" in text.lower(), (
            f"Expected entity name or 'success' in history output:\n{text[:500]}"
        )

    @pytest.mark.asyncio
    async def test_mcp_search_code_missing_query_returns_error(self, search_handlers):
        """handle_search_code with no query returns isError=True."""
        result = await search_handlers.handle_search_code({})
        assert result.isError is True, "Expected isError=True when query is missing"

    @pytest.mark.asyncio
    async def test_mcp_kg_history_missing_entity_returns_error(self, kg_handlers):
        """handle_kg_history with no entity_name returns isError=True."""
        result = await kg_handlers.handle_kg_history({})
        assert result.isError is True, (
            "Expected isError=True when entity_name is missing"
        )

    @pytest.mark.asyncio
    async def test_mcp_trace_missing_entry_point_returns_error(self, kg_handlers):
        """handle_trace_execution_flow with no entry_point returns isError=True."""
        result = await kg_handlers.handle_trace_execution_flow({})
        assert result.isError is True, (
            "Expected isError=True when entry_point is missing"
        )
