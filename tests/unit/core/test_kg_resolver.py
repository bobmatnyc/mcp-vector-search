"""Unit tests for KG entity resolver — unique-name ambiguity detection."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mcp_vector_search.core.kg_builder import KGBuilder
from mcp_vector_search.core.models import CodeChunk


def make_chunk(
    name: str, file_path: str = "/repo/src/a.py", chunk_id: str | None = None
) -> CodeChunk:
    """Create a minimal CodeChunk for testing."""
    cid = chunk_id or f"chunk_{name}_{file_path}"
    return CodeChunk(
        content=f"def {name}(): pass",
        file_path=Path(file_path),
        start_line=1,
        end_line=1,
        language="python",
        chunk_type="function",
        function_name=name,
        chunk_id=cid,
        calls=[],
    )


@pytest.fixture
def builder():
    """Create a KGBuilder with a mock KG."""
    mock_kg = MagicMock()
    return KGBuilder(kg=mock_kg, project_root=Path("/repo"))


class TestPrescanEntityNames:
    def test_populates_entity_map(self, builder):
        chunks = [make_chunk("authenticate", chunk_id="c1")]
        builder._prescan_entity_names(chunks)
        assert "authenticate" in builder._entity_map
        assert builder._entity_map["authenticate"] == "c1"

    def test_counts_unique_name(self, builder):
        chunks = [make_chunk("authenticate", chunk_id="c1")]
        builder._prescan_entity_names(chunks)
        assert builder._entity_name_counts["authenticate"] == 1

    def test_counts_duplicate_names(self, builder):
        chunks = [
            make_chunk("search", file_path="/repo/a.py", chunk_id="c1"),
            make_chunk("search", file_path="/repo/b.py", chunk_id="c2"),
            make_chunk("search", file_path="/repo/c.py", chunk_id="c3"),
        ]
        builder._prescan_entity_names(chunks)
        assert builder._entity_name_counts["search"] == 3

    def test_skips_generic_names(self, builder):
        chunks = [
            make_chunk("data", chunk_id="c1")
        ]  # "data" is in GENERIC_ENTITY_NAMES
        builder._prescan_entity_names(chunks)
        assert "data" not in builder._entity_name_counts

    def test_clears_previous_state(self, builder):
        builder._entity_map["stale"] = "old_id"
        builder._entity_name_counts["stale"] = 1
        chunks = [make_chunk("fresh", chunk_id="c1")]
        builder._prescan_entity_names(chunks)
        assert "stale" not in builder._entity_map


class TestResolveEntity:
    def test_resolves_unique_name(self, builder):
        builder._entity_map["WikiPublisher"] = "c1"
        builder._entity_name_counts["WikiPublisher"] = 1
        assert builder._resolve_entity("WikiPublisher") == "c1"

    def test_skips_ambiguous_name(self, builder):
        builder._entity_map["search"] = "c_last"
        builder._entity_name_counts["search"] = 5
        assert builder._resolve_entity("search") is None

    def test_resolves_dotted_unique(self, builder):
        builder._entity_map["authenticate"] = "c1"
        builder._entity_name_counts["authenticate"] = 1
        assert builder._resolve_entity("AuthService.authenticate") == "c1"

    def test_skips_dotted_ambiguous(self, builder):
        builder._entity_map["search"] = "c2"
        builder._entity_name_counts["search"] = 3
        assert builder._resolve_entity("engine.search") is None

    def test_returns_none_for_unknown(self, builder):
        assert builder._resolve_entity("nonexistent_function_xyz") is None

    def test_returns_none_for_empty(self, builder):
        assert builder._resolve_entity("") is None


class TestEndToEndPrescan:
    def test_calls_resolved_after_prescan(self, builder):
        """Calls to later-defined functions resolve after prescan."""
        callee = make_chunk("WikiPublisher", file_path="/repo/b.py", chunk_id="c_wiki")
        callee_with_calls = CodeChunk(
            content="def publish(): WikiPublisher()",
            file_path=Path("/repo/a.py"),
            start_line=1,
            end_line=1,
            language="python",
            chunk_type="function",
            function_name="publish",
            chunk_id="c_publish",
            calls=["WikiPublisher"],
        )

        # Prescan with both — callee registered before caller's calls are resolved
        builder._prescan_entity_names([callee_with_calls, callee])
        # Now resolve — should find WikiPublisher even though it was "second"
        resolved = builder._resolve_entity("WikiPublisher")
        assert resolved == "c_wiki"

    def test_no_self_loops_with_ambiguous_names(self, builder):
        """Ambiguous callee name produces no CALLS edge, preventing self-loops."""
        chunks = [
            make_chunk("search", file_path="/repo/a.py", chunk_id="c1"),
            make_chunk("search", file_path="/repo/b.py", chunk_id="c2"),
        ]
        builder._prescan_entity_names(chunks)
        # "search" is ambiguous — must not resolve
        assert builder._resolve_entity("search") is None
