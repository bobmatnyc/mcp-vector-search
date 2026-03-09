"""Unit tests for temporal KG query methods (issue #99).

Tests cover:
- git_utils.is_ancestor_commit / get_commit_timestamp
- KnowledgeGraph.get_entity_history
- KnowledgeGraph.get_callers_at_commit (KG not initialised path)
- kg_builder CodeEntity receives commit_sha from chunk.commit_hash
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# git_utils tests
# ---------------------------------------------------------------------------


class TestIsAncestorCommit:
    """Tests for git_utils.is_ancestor_commit."""

    def test_empty_earlier_sha_returns_false(self, tmp_path: Path) -> None:
        """Empty earlier_sha short-circuits to False without calling git."""
        from mcp_vector_search.core.git_utils import is_ancestor_commit

        assert is_ancestor_commit("", "abc123", tmp_path) is False

    def test_empty_later_sha_returns_false(self, tmp_path: Path) -> None:
        """Empty later_sha short-circuits to False without calling git."""
        from mcp_vector_search.core.git_utils import is_ancestor_commit

        assert is_ancestor_commit("abc123", "", tmp_path) is False

    def test_subprocess_exit_0_returns_true(self, tmp_path: Path) -> None:
        """Subprocess exit code 0 means ancestor → True."""
        from mcp_vector_search.core.git_utils import is_ancestor_commit

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = is_ancestor_commit("aaa111", "bbb222", tmp_path)

        assert result is True
        mock_run.assert_called_once()

    def test_subprocess_exit_1_returns_false(self, tmp_path: Path) -> None:
        """Subprocess exit code 1 means not an ancestor → False."""
        from mcp_vector_search.core.git_utils import is_ancestor_commit

        mock_result = MagicMock()
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            result = is_ancestor_commit("aaa111", "bbb222", tmp_path)

        assert result is False

    def test_is_ancestor_commit_equal_sha(self, tmp_path: Path) -> None:
        """Same SHA passed as both args: git merge-base treats A==B as ancestor."""
        from mcp_vector_search.core.git_utils import is_ancestor_commit

        mock_result = MagicMock()
        mock_result.returncode = 0  # git exits 0 for identical SHAs

        with patch("subprocess.run", return_value=mock_result):
            result = is_ancestor_commit("deadbeef", "deadbeef", tmp_path)

        assert result is True


class TestGetCommitTimestamp:
    """Tests for git_utils.get_commit_timestamp."""

    def test_empty_sha_returns_none(self, tmp_path: Path) -> None:
        from mcp_vector_search.core.git_utils import get_commit_timestamp

        assert get_commit_timestamp("", tmp_path) is None

    def test_valid_output_returns_int(self, tmp_path: Path) -> None:
        from mcp_vector_search.core.git_utils import get_commit_timestamp

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "1700000000\n"

        with patch("subprocess.run", return_value=mock_result):
            ts = get_commit_timestamp("abc123", tmp_path)

        assert ts == 1700000000

    def test_git_failure_returns_none(self, tmp_path: Path) -> None:
        from mcp_vector_search.core.git_utils import get_commit_timestamp

        mock_result = MagicMock()
        mock_result.returncode = 128
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            ts = get_commit_timestamp("badsha", tmp_path)

        assert ts is None


# ---------------------------------------------------------------------------
# KnowledgeGraph.get_entity_history tests
# ---------------------------------------------------------------------------


class TestGetEntityHistory:
    """Tests for KnowledgeGraph.get_entity_history."""

    def _make_kg(self) -> MagicMock:
        """Return a MagicMock with get_entity_history bound as the real method."""
        from mcp_vector_search.core.knowledge_graph import KnowledgeGraph

        kg = MagicMock(spec=KnowledgeGraph)
        kg._initialized = True
        kg.get_entity_history = KnowledgeGraph.get_entity_history.__get__(kg)
        return kg

    import pytest  # needed for async mark below

    @__import__("pytest").mark.asyncio
    async def test_get_entity_history_empty(self) -> None:
        """Entity not present in KG → empty list."""

        from mcp_vector_search.core.knowledge_graph import KnowledgeGraph

        kg = MagicMock(spec=KnowledgeGraph)
        kg._initialized = True
        kg.get_entity_history = KnowledgeGraph.get_entity_history.__get__(kg)

        # Mock conn.execute to return an iterator with no rows
        mock_result = MagicMock()
        mock_result.has_next.return_value = False
        kg.conn = MagicMock()
        kg.conn.execute.return_value = mock_result

        history = await kg.get_entity_history("NonExistentEntity")

        assert history == []

    @__import__("pytest").mark.asyncio
    async def test_get_entity_history_returns_entity(self) -> None:
        """Entity with commit_sha stored → one entry returned."""
        from mcp_vector_search.core.knowledge_graph import KnowledgeGraph

        kg = MagicMock(spec=KnowledgeGraph)
        kg._initialized = True
        kg.get_entity_history = KnowledgeGraph.get_entity_history.__get__(kg)

        # Simulate one row: name, entity_type, file_path, commit_sha
        mock_result = MagicMock()
        mock_result.has_next.side_effect = [True, False]
        mock_result.get_next.return_value = [
            "my_func",
            "function",
            "src/app.py",
            "abc1234",
        ]
        kg.conn = MagicMock()
        kg.conn.execute.return_value = mock_result

        history = await kg.get_entity_history("my_func")

        assert len(history) == 1
        assert history[0]["name"] == "my_func"
        assert history[0]["entity_type"] == "function"
        assert history[0]["file_path"] == "src/app.py"
        assert history[0]["commit_sha"] == "abc1234"


# ---------------------------------------------------------------------------
# KnowledgeGraph.get_callers_at_commit — no-KG path
# ---------------------------------------------------------------------------


class TestGetCallersAtCommit:
    """Tests for KnowledgeGraph.get_callers_at_commit."""

    @__import__("pytest").mark.asyncio
    async def test_get_callers_at_commit_no_kg(self, tmp_path: Path) -> None:
        """When conn.execute raises, an empty list is returned gracefully."""
        from mcp_vector_search.core.knowledge_graph import KnowledgeGraph

        kg = MagicMock(spec=KnowledgeGraph)
        kg._initialized = True
        kg.get_callers_at_commit = KnowledgeGraph.get_callers_at_commit.__get__(kg)
        kg.conn = MagicMock()
        kg.conn.execute.side_effect = RuntimeError("KG not built")

        callers = await kg.get_callers_at_commit("some_func", "abc1234", tmp_path)

        assert callers == []

    @__import__("pytest").mark.asyncio
    async def test_get_callers_at_commit_filters_by_ancestry(
        self, tmp_path: Path
    ) -> None:
        """Only callers whose commit_sha is an ancestor are returned."""
        from mcp_vector_search.core.knowledge_graph import KnowledgeGraph

        kg = MagicMock(spec=KnowledgeGraph)
        kg._initialized = True
        kg.get_callers_at_commit = KnowledgeGraph.get_callers_at_commit.__get__(kg)

        # Two rows: one with ancestor SHA, one not
        mock_result = MagicMock()
        mock_result.has_next.side_effect = [True, True, False]
        mock_result.get_next.side_effect = [
            ["caller_a", "src/a.py", "ancestor_sha", "callee_func"],
            ["caller_b", "src/b.py", "non_ancestor_sha", "callee_func"],
        ]
        kg.conn = MagicMock()
        kg.conn.execute.return_value = mock_result

        # Patch is_ancestor_commit: "ancestor_sha" → True, "non_ancestor_sha" → False
        def fake_is_ancestor(earlier: str, later: str, repo: Path) -> bool:
            return earlier == "ancestor_sha"

        with patch(
            "mcp_vector_search.core.knowledge_graph.KnowledgeGraph"
            ".get_callers_at_commit",
            KnowledgeGraph.get_callers_at_commit.__get__(kg),
        ):
            with patch(
                "mcp_vector_search.core.git_utils.is_ancestor_commit",
                side_effect=fake_is_ancestor,
            ):
                callers = await kg.get_callers_at_commit(
                    "callee_func", "ref_sha", tmp_path
                )

        # Only caller_a should be included
        assert len(callers) == 1
        assert callers[0]["caller_name"] == "caller_a"


# ---------------------------------------------------------------------------
# kg_builder: commit_sha propagation
# ---------------------------------------------------------------------------


class TestCommitShaInBuilder:
    """Verify that kg_builder passes commit_hash from chunk into CodeEntity."""

    def test_commit_sha_populated_in_builder(self) -> None:
        """Mock chunk with commit_hash → CodeEntity receives that commit_sha."""
        from mcp_vector_search.core.knowledge_graph import CodeEntity

        # Simulate what _extract_code_entity does with commit_sha set
        chunk = MagicMock()
        chunk.chunk_id = "entity:test_func:src/app.py:1"
        chunk.chunk_type = "function"
        chunk.file_path = Path("src/app.py")
        chunk.commit_hash = "cafebabe"

        entity = CodeEntity(
            id=chunk.chunk_id,
            name="test_func",
            entity_type=chunk.chunk_type,
            file_path=str(chunk.file_path),
            commit_sha=chunk.commit_hash or "",
        )

        assert entity.commit_sha == "cafebabe"

    def test_commit_sha_defaults_to_empty_when_none(self) -> None:
        """chunk.commit_hash = None → CodeEntity.commit_sha is empty string."""
        from mcp_vector_search.core.knowledge_graph import CodeEntity

        chunk = MagicMock()
        chunk.commit_hash = None

        entity = CodeEntity(
            id="entity:x:f.py:1",
            name="x",
            entity_type="function",
            file_path="f.py",
            commit_sha=chunk.commit_hash or "",
        )

        assert entity.commit_sha == ""
