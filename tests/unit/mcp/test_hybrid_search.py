"""Unit tests for the search_hybrid MCP tool handler.

These tests focus on the RRF fusion logic and the handler's contract
(input validation, output schema, partial-failure resilience) without
requiring a live index or knowledge graph.  All I/O-heavy dependencies
are replaced with lightweight mocks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import TextContent

from mcp_vector_search.core.models import SearchResult
from mcp_vector_search.mcp.hybrid_search_handler import (
    _RRF_K,
    _STRATEGY_KG,
    _STRATEGY_SEMANTIC,
    _STRATEGY_TEXT,
    HybridSearchHandlers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    file_path: str,
    start_line: int,
    end_line: int,
    score: float,
    content: str = "code content",
    chunk_type: str = "function",
) -> SearchResult:
    """Create a minimal SearchResult for testing."""
    return SearchResult(
        file_path=Path(file_path),
        content=content,
        start_line=start_line,
        end_line=end_line,
        language="python",
        similarity_score=score,
        rank=1,
        chunk_type=chunk_type,
    )


def _make_handler(project_root: Path | None = None) -> HybridSearchHandlers:
    """Create a HybridSearchHandlers with a mock SemanticSearchEngine."""
    engine = MagicMock()
    engine.search = AsyncMock(return_value=[])
    engine.database = MagicMock()
    root = project_root or Path("/tmp/fake_project")
    return HybridSearchHandlers(search_engine=engine, project_root=root)


# ---------------------------------------------------------------------------
# _fuse_rrf unit tests
# ---------------------------------------------------------------------------


class TestFuseRRF:
    """Tests for the pure RRF fusion logic."""

    def test_single_strategy_scores(self) -> None:
        """A result found only at rank 1 by one strategy gets 1/(60+1)."""
        handler = _make_handler()
        result = _make_result("a.py", 1, 10, 0.9)
        fused = handler._fuse_rrf([result], [], [])

        assert len(fused) == 1
        expected_score = 1.0 / (_RRF_K + 1)
        assert abs(fused[0].rrf_score - expected_score) < 1e-9

    def test_multi_strategy_score_accumulates(self) -> None:
        """The same chunk found by two strategies gets the sum of their RRF contributions."""
        handler = _make_handler()
        result_sem = _make_result("shared.py", 5, 20, 0.9)
        result_txt = _make_result("shared.py", 5, 20, 0.7)

        fused = handler._fuse_rrf([result_sem], [result_txt], [])

        assert len(fused) == 1
        # Both at rank 1 → 2 × 1/(60+1)
        expected = 2.0 / (_RRF_K + 1)
        assert abs(fused[0].rrf_score - expected) < 1e-9
        assert set(fused[0].strategies_found) == {_STRATEGY_SEMANTIC, _STRATEGY_TEXT}

    def test_deduplication_by_file_and_line(self) -> None:
        """Two results with the same (file_path, start_line) are merged into one."""
        handler = _make_handler()
        r1 = _make_result("dup.py", 10, 20, 0.8)
        r2 = _make_result("dup.py", 10, 20, 0.6)

        fused = handler._fuse_rrf([r1, r2], [], [])
        assert len(fused) == 1

    def test_different_files_not_merged(self) -> None:
        """Results from different files are kept separate."""
        handler = _make_handler()
        r1 = _make_result("file_a.py", 1, 10, 0.9)
        r2 = _make_result("file_b.py", 1, 10, 0.8)

        fused = handler._fuse_rrf([r1], [r2], [])
        assert len(fused) == 2

    def test_result_ranked_by_rrf_score_descending(self) -> None:
        """Higher-ranked results (lower rank index) get a higher RRF score."""
        handler = _make_handler()
        r_rank1 = _make_result("best.py", 1, 10, 0.99)
        r_rank2 = _make_result("ok.py", 1, 10, 0.50)

        # r_rank1 at position 0 → rank 1; r_rank2 at position 1 → rank 2
        fused = handler._fuse_rrf([r_rank1, r_rank2], [], [])

        assert fused[0].file_path == "best.py"
        assert fused[0].rrf_score > fused[1].rrf_score

    def test_rich_content_wins_over_kg_stub(self) -> None:
        """When a KG stub (empty content) and a semantic result share the same
        (file_path, start_line), the semantic result's content is kept."""
        handler = _make_handler()
        semantic = _make_result("impl.py", 1, 50, 0.9, content="def foo(): pass")
        kg_stub = _make_result(
            "impl.py", 1, 50, 1.0, content="", chunk_type="kg_entity"
        )

        fused = handler._fuse_rrf([semantic], [], [kg_stub])

        assert len(fused) == 1
        assert fused[0].content == "def foo(): pass"
        assert set(fused[0].strategies_found) == {_STRATEGY_SEMANTIC, _STRATEGY_KG}

    def test_score_attributes_stored_per_strategy(self) -> None:
        """semantic_score, text_score, and kg_score are stored on each candidate."""
        handler = _make_handler()
        sem = _make_result("x.py", 1, 5, 0.8)
        txt = _make_result("x.py", 1, 5, 0.6)
        kg = _make_result("x.py", 1, 5, 1.0, content="")

        fused = handler._fuse_rrf([sem], [txt], [kg])
        assert len(fused) == 1
        c = fused[0]
        assert c.semantic_score == 0.8
        assert c.text_score == 0.6
        assert c.kg_score == 1.0


# ---------------------------------------------------------------------------
# handle_search_hybrid integration tests (mocked strategies)
# ---------------------------------------------------------------------------


class TestHandleSearchHybrid:
    """Tests for the MCP tool entry point."""

    @pytest.mark.asyncio
    async def test_returns_error_when_query_missing(self) -> None:
        handler = _make_handler()
        result = await handler.handle_search_hybrid({})
        assert result.isError is True
        content = result.content[0]
        assert isinstance(content, TextContent)
        assert "query" in content.text.lower()

    @pytest.mark.asyncio
    async def test_returns_error_when_query_blank(self) -> None:
        handler = _make_handler()
        result = await handler.handle_search_hybrid({"query": "   "})
        assert result.isError is True

    @pytest.mark.asyncio
    async def test_returns_error_for_unknown_strategies(self) -> None:
        handler = _make_handler()
        result = await handler.handle_search_hybrid(
            {"query": "test", "strategies": ["bogus"]}
        )
        assert result.isError is True

    @pytest.mark.asyncio
    async def test_successful_response_schema(self) -> None:
        """A successful call returns valid JSON matching the expected schema."""
        handler = _make_handler()

        sem_result = _make_result("src/core/search.py", 10, 30, 0.85)
        txt_result = _make_result("src/core/search.py", 10, 30, 0.70)

        # Wire the engine mock to return results for semantic + BM25 modes.
        from mcp_vector_search.core.search import SearchMode

        async def fake_search(**kwargs: Any) -> list[SearchResult]:
            mode = kwargs.get("search_mode")
            if mode == SearchMode.VECTOR:
                return [sem_result]
            if mode == SearchMode.BM25:
                return [txt_result]
            return []

        handler.search_engine.search = fake_search  # type: ignore[method-assign]

        # KG strategy is disabled so we don't need a real DB.
        result = await handler.handle_search_hybrid(
            {"query": "semantic search", "limit": 5, "strategies": ["semantic", "text"]}
        )

        assert result.isError is False
        _content = result.content[0]
        assert isinstance(_content, TextContent)
        payload = json.loads(_content.text)

        # Top-level keys
        assert "results" in payload
        assert "query" in payload
        assert "strategy_counts" in payload

        # Strategy counts
        sc = payload["strategy_counts"]
        assert sc["semantic"] == 1
        assert sc["text"] == 1
        assert sc["kg"] == 0
        assert sc["total_before_dedup"] == 2
        assert sc["total_after_dedup"] == 1  # same chunk merged

        # Result item keys
        item = payload["results"][0]
        for key in (
            "file_path",
            "content",
            "score",
            "strategies",
            "semantic_score",
            "text_score",
            "kg_score",
            "line_start",
            "line_end",
            "chunk_type",
        ):
            assert key in item, f"Missing key: {key}"

        assert item["score"] > 0
        assert set(item["strategies"]) == {"semantic", "text"}

    @pytest.mark.asyncio
    async def test_limit_is_respected(self) -> None:
        """The response never exceeds the requested limit."""
        handler = _make_handler()

        # Produce 5 distinct results from semantic strategy.
        sem_results = [
            _make_result(f"file_{i}.py", 1, 10, 1.0 - i * 0.05) for i in range(5)
        ]

        from mcp_vector_search.core.search import SearchMode

        async def fake_search(**kwargs: Any) -> list[SearchResult]:
            if kwargs.get("search_mode") == SearchMode.VECTOR:
                return sem_results
            return []

        handler.search_engine.search = fake_search  # type: ignore[method-assign]

        result = await handler.handle_search_hybrid(
            {"query": "anything", "limit": 3, "strategies": ["semantic"]}
        )

        assert result.isError is False
        _content = result.content[0]
        assert isinstance(_content, TextContent)
        payload = json.loads(_content.text)
        assert len(payload["results"]) <= 3

    @pytest.mark.asyncio
    async def test_partial_failure_still_returns_results(self) -> None:
        """If the BM25 strategy raises, semantic results are still returned."""
        handler = _make_handler()

        sem_result = _make_result("good.py", 1, 10, 0.9)

        from mcp_vector_search.core.search import SearchMode

        async def fake_search(**kwargs: Any) -> list[SearchResult]:
            mode = kwargs.get("search_mode")
            if mode == SearchMode.VECTOR:
                return [sem_result]
            raise RuntimeError("BM25 index not available")

        handler.search_engine.search = fake_search  # type: ignore[method-assign]

        result = await handler.handle_search_hybrid(
            {"query": "good", "strategies": ["semantic", "text"]}
        )

        assert result.isError is False
        _content = result.content[0]
        assert isinstance(_content, TextContent)
        payload = json.loads(_content.text)
        assert len(payload["results"]) == 1
        assert payload["results"][0]["file_path"] == "good.py"
        # Text strategy contributed 0 results.
        assert payload["strategy_counts"]["text"] == 0

    @pytest.mark.asyncio
    async def test_kg_skipped_when_not_built(self, tmp_path: Path) -> None:
        """KG strategy silently returns [] when the DB directory is absent."""
        # tmp_path has no .mcp-vector-search/knowledge_graph/code_kg sub-dir.
        handler = _make_handler(project_root=tmp_path)

        from mcp_vector_search.core.search import SearchMode

        async def fake_search(**kwargs: Any) -> list[SearchResult]:
            if kwargs.get("search_mode") == SearchMode.VECTOR:
                return [_make_result("a.py", 1, 5, 0.9)]
            return []

        handler.search_engine.search = fake_search  # type: ignore[method-assign]

        result = await handler.handle_search_hybrid(
            {"query": "something", "strategies": ["semantic", "kg"]}
        )

        assert result.isError is False
        _content = result.content[0]
        assert isinstance(_content, TextContent)
        payload = json.loads(_content.text)
        assert payload["strategy_counts"]["kg"] == 0
        assert len(payload["results"]) == 1

    @pytest.mark.asyncio
    async def test_only_semantic_strategy(self) -> None:
        """Requesting only 'semantic' runs a single strategy."""
        handler = _make_handler()
        sem_result = _make_result("solo.py", 1, 20, 0.95)

        from mcp_vector_search.core.search import SearchMode

        call_log: list[SearchMode] = []

        async def fake_search(**kwargs: Any) -> list[SearchResult]:
            mode = kwargs.get("search_mode")
            if mode is not None:
                call_log.append(mode)
            if mode == SearchMode.VECTOR:
                return [sem_result]
            return []

        handler.search_engine.search = fake_search  # type: ignore[method-assign]

        result = await handler.handle_search_hybrid(
            {"query": "solo", "strategies": ["semantic"]}
        )

        assert result.isError is False
        # BM25 search must NOT have been invoked.
        assert SearchMode.BM25 not in call_log
        _content = result.content[0]
        assert isinstance(_content, TextContent)
        payload = json.loads(_content.text)
        assert payload["strategy_counts"]["text"] == 0


# ---------------------------------------------------------------------------
# Tool schema tests
# ---------------------------------------------------------------------------


class TestSearchHybridSchema:
    """Validate that the tool schema is correctly wired."""

    def test_schema_included_in_tool_list(self) -> None:
        from mcp_vector_search.mcp.tool_schemas import get_tool_schemas

        schemas = get_tool_schemas()
        names = [t.name for t in schemas]
        assert "search_hybrid" in names

    def test_schema_has_required_query(self) -> None:
        from mcp_vector_search.mcp.tool_schemas import get_tool_schemas

        schemas = get_tool_schemas()
        hybrid = next(t for t in schemas if t.name == "search_hybrid")
        assert "query" in hybrid.inputSchema["required"]

    def test_schema_properties(self) -> None:
        from mcp_vector_search.mcp.tool_schemas import get_tool_schemas

        schemas = get_tool_schemas()
        hybrid = next(t for t in schemas if t.name == "search_hybrid")
        props = hybrid.inputSchema["properties"]
        for expected_prop in ("query", "limit", "strategies", "project_path"):
            assert expected_prop in props, f"Missing property: {expected_prop}"

    def test_schema_documents_warnings_field(self) -> None:
        """The response schema documents the 'warnings' output field."""
        from mcp_vector_search.mcp.tool_schemas import get_tool_schemas

        schemas = get_tool_schemas()
        hybrid = next(t for t in schemas if t.name == "search_hybrid")
        response_schema = hybrid.inputSchema.get("x-response-schema", {})
        assert "warnings" in response_schema.get("properties", {}), (
            "warnings field must be documented in x-response-schema"
        )


# ---------------------------------------------------------------------------
# New fix tests
# ---------------------------------------------------------------------------


class TestKGWarning:
    """Tests for Fix 1: KG-not-built warning in response."""

    @pytest.mark.asyncio
    async def test_kg_warning_when_not_built(self, tmp_path: Path) -> None:
        """When KG is requested but the DB dir is absent, a warning is emitted."""
        # tmp_path has no .mcp-vector-search/knowledge_graph/code_kg directory.
        handler = _make_handler(project_root=tmp_path)

        from mcp_vector_search.core.search import SearchMode

        async def fake_search(**kwargs: Any) -> list[SearchResult]:
            if kwargs.get("search_mode") == SearchMode.VECTOR:
                return [_make_result("src/core/search.py", 1, 10, 0.9)]
            return []

        handler.search_engine.search = fake_search  # type: ignore[method-assign]

        result = await handler.handle_search_hybrid(
            {"query": "search engine", "strategies": ["semantic", "kg"]}
        )

        assert result.isError is False
        _content = result.content[0]
        assert isinstance(_content, TextContent)
        payload = json.loads(_content.text)

        assert "warnings" in payload, "Response must contain 'warnings' key"
        assert isinstance(payload["warnings"], list)
        assert len(payload["warnings"]) == 1
        assert "kg_build" in payload["warnings"][0].lower()
        assert "not built" in payload["warnings"][0].lower()

    @pytest.mark.asyncio
    async def test_no_warnings_when_kg_not_requested(self, tmp_path: Path) -> None:
        """When KG strategy is not requested, warnings list is empty."""
        handler = _make_handler(project_root=tmp_path)

        from mcp_vector_search.core.search import SearchMode

        async def fake_search(**kwargs: Any) -> list[SearchResult]:
            if kwargs.get("search_mode") == SearchMode.VECTOR:
                return [_make_result("src/core/search.py", 1, 10, 0.9)]
            return []

        handler.search_engine.search = fake_search  # type: ignore[method-assign]

        result = await handler.handle_search_hybrid(
            {"query": "search engine", "strategies": ["semantic", "text"]}
        )

        assert result.isError is False
        _content = result.content[0]
        assert isinstance(_content, TextContent)
        payload = json.loads(_content.text)
        assert "warnings" in payload
        assert payload["warnings"] == []

    @pytest.mark.asyncio
    async def test_warnings_empty_on_successful_response(self) -> None:
        """A fully successful call includes 'warnings' as an empty list."""
        handler = _make_handler()

        from mcp_vector_search.core.search import SearchMode

        async def fake_search(**kwargs: Any) -> list[SearchResult]:
            if kwargs.get("search_mode") == SearchMode.VECTOR:
                return [_make_result("src/a.py", 1, 5, 0.9)]
            return []

        handler.search_engine.search = fake_search  # type: ignore[method-assign]

        result = await handler.handle_search_hybrid(
            {"query": "something", "strategies": ["semantic"]}
        )

        _content = result.content[0]
        assert isinstance(_content, TextContent)
        payload = json.loads(_content.text)
        assert "warnings" in payload
        assert payload["warnings"] == []


class TestMinScoreCutoff:
    """Tests for Fix 3: Minimum RRF score cutoff."""

    @pytest.mark.asyncio
    async def test_min_score_cutoff_filters_low_scores(self) -> None:
        """Results with RRF score below 0.01 are excluded from the response."""
        handler = _make_handler()

        from mcp_vector_search.core.search import SearchMode

        # Produce 100 distinct results — the 100th result gets rank 100.
        # RRF score at rank 100 = 1/(60+100) = 0.00625, which is below 0.01.
        many_results = [
            _make_result(f"file_{i}.py", 1, 10, 1.0 - i * 0.005) for i in range(100)
        ]

        async def fake_search(**kwargs: Any) -> list[SearchResult]:
            if kwargs.get("search_mode") == SearchMode.VECTOR:
                return many_results
            return []

        handler.search_engine.search = fake_search  # type: ignore[method-assign]

        result = await handler.handle_search_hybrid(
            {"query": "test cutoff", "limit": 100, "strategies": ["semantic"]}
        )

        assert result.isError is False
        _content = result.content[0]
        assert isinstance(_content, TextContent)
        payload = json.loads(_content.text)

        # All returned results must be above the 0.01 threshold.
        for item in payload["results"]:
            assert item["score"] >= 0.01, (
                f"Result {item['file_path']} has score {item['score']} below 0.01 cutoff"
            )

        # The 100-result list should have fewer items after the cutoff.
        # rank 39 → 1/(60+39) = 0.0101 (passes), rank 40 → 1/(60+40) = 0.01 (passes),
        # rank 41 → 1/(60+41) = 0.00990 (filtered out).
        assert len(payload["results"]) < 100

    @pytest.mark.asyncio
    async def test_high_score_results_pass_cutoff(self) -> None:
        """Results with good ranks always survive the min-score filter."""
        handler = _make_handler()

        from mcp_vector_search.core.search import SearchMode

        top_results = [
            _make_result(f"src/module_{i}.py", 1, 10, 0.95 - i * 0.01) for i in range(5)
        ]

        async def fake_search(**kwargs: Any) -> list[SearchResult]:
            if kwargs.get("search_mode") == SearchMode.VECTOR:
                return top_results
            return []

        handler.search_engine.search = fake_search  # type: ignore[method-assign]

        result = await handler.handle_search_hybrid(
            {"query": "module search", "limit": 10, "strategies": ["semantic"]}
        )

        assert result.isError is False
        _content = result.content[0]
        assert isinstance(_content, TextContent)
        payload = json.loads(_content.text)
        # All 5 high-quality results should survive.
        assert len(payload["results"]) == 5
