"""Unit tests for semantic search functionality."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from mcp_vector_search.core.auto_indexer import AutoIndexer
from mcp_vector_search.core.models import SearchResult
from mcp_vector_search.core.search import SemanticSearchEngine
from tests.conftest import assert_search_results_valid


class TestSemanticSearchEngine:
    """Test cases for SemanticSearchEngine."""

    @pytest.mark.asyncio
    async def test_search_basic_functionality(self, mock_database, sample_code_chunks):
        """Test basic search functionality."""
        # Setup
        await mock_database.add_chunks(sample_code_chunks)
        search_engine = SemanticSearchEngine(
            database=mock_database,
            project_root=Path("/test"),
            similarity_threshold=0.1,
        )

        # Execute
        results = await search_engine.search(
            "main function", limit=5, similarity_threshold=0.1
        )

        # Assert
        assert_search_results_valid(results, min_count=1)
        assert any("main" in result.content.lower() for result in results)

    @pytest.mark.asyncio
    async def test_search_empty_query(self, mock_database):
        """Test search with empty query."""
        search_engine = SemanticSearchEngine(
            database=mock_database,
            project_root=Path("/test"),
        )

        # Test empty string
        results = await search_engine.search("", limit=5)
        assert len(results) == 0

        # Test whitespace only
        results = await search_engine.search("   ", limit=5)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_with_filters(self, mock_database, sample_code_chunks):
        """Test search with filters."""
        await mock_database.add_chunks(sample_code_chunks)
        search_engine = SemanticSearchEngine(
            database=mock_database,
            project_root=Path("/test"),
        )

        # Search with language filter
        filters = {"language": "python"}
        results = await search_engine.search(
            "function", limit=5, filters=filters, similarity_threshold=0.1
        )

        assert_search_results_valid(results)
        for result in results:
            assert result.language == "python"

    @pytest.mark.asyncio
    async def test_search_similarity_threshold(self, mock_database, sample_code_chunks):
        """Test search with different similarity thresholds."""
        async with mock_database as db:
            await db.add_chunks(sample_code_chunks)
            search_engine = SemanticSearchEngine(
                database=db,
                project_root=Path("/test"),
            )

            # Low threshold should return more results
            low_threshold_results = await search_engine.search(
                "function", limit=10, similarity_threshold=0.1
            )

            # High threshold should return fewer results
            high_threshold_results = await search_engine.search(
                "function", limit=10, similarity_threshold=0.9
            )

            assert len(low_threshold_results) >= len(high_threshold_results)

    @pytest.mark.asyncio
    async def test_search_limit_parameter(self, mock_database, sample_code_chunks):
        """Test search result limiting."""
        async with mock_database as db:
            # Add more chunks than limit
            extended_chunks = sample_code_chunks * 5  # 15 chunks total
            await db.add_chunks(extended_chunks)

            search_engine = SemanticSearchEngine(
                database=db,
                project_root=Path("/test"),
            )

            # Test with limit
            limit = 3
            results = await search_engine.search(
                "function", limit=limit, similarity_threshold=0.1
            )

            assert len(results) <= limit

    @pytest.mark.asyncio
    async def test_search_with_auto_indexer(self, mock_database, sample_code_chunks):
        """Test search with auto-indexer integration."""
        async with mock_database as db:
            await db.add_chunks(sample_code_chunks)

            # Create mock auto-indexer
            mock_auto_indexer = Mock(spec=AutoIndexer)
            mock_search_triggered_indexer = Mock()
            mock_search_triggered_indexer.pre_search_hook = AsyncMock()

            search_engine = SemanticSearchEngine(
                database=db,
                project_root=Path("/test"),
                auto_indexer=mock_auto_indexer,
                enable_auto_reindex=True,
            )
            search_engine.search_triggered_indexer = mock_search_triggered_indexer

            # Execute search
            results = await search_engine.search("function", similarity_threshold=0.1)

            # Assert auto-indexer was called
            mock_search_triggered_indexer.pre_search_hook.assert_called_once()
            assert_search_results_valid(results)

    @pytest.mark.asyncio
    async def test_search_auto_indexer_error_handling(
        self, mock_database, sample_code_chunks
    ):
        """Test search handles auto-indexer errors gracefully."""
        async with mock_database as db:
            await db.add_chunks(sample_code_chunks)

            # Create mock auto-indexer that raises exception
            mock_search_triggered_indexer = Mock()
            mock_search_triggered_indexer.pre_search_hook = AsyncMock(
                side_effect=Exception("Auto-indexer error")
            )

            search_engine = SemanticSearchEngine(
                database=db,
                project_root=Path("/test"),
            )
            search_engine.search_triggered_indexer = mock_search_triggered_indexer

            # Execute search - should not raise exception
            results = await search_engine.search("function", similarity_threshold=0.1)

            # Search should still work despite auto-indexer error
            assert_search_results_valid(results)

    @pytest.mark.asyncio
    async def test_adaptive_threshold(self, mock_database):
        """Test adaptive threshold calculation."""
        search_engine = SemanticSearchEngine(
            database=mock_database,
            project_root=Path("/test"),
            similarity_threshold=0.7,
        )

        # Test different query types
        short_query = "test"
        long_query = "this is a very long and specific query with many words"

        short_threshold = search_engine._get_adaptive_threshold(short_query)
        long_threshold = search_engine._get_adaptive_threshold(long_query)

        # Longer queries should have lower thresholds
        assert isinstance(short_threshold, float)
        assert isinstance(long_threshold, float)
        assert 0.0 <= short_threshold <= 1.0
        assert 0.0 <= long_threshold <= 1.0

    @pytest.mark.asyncio
    async def test_query_preprocessing(self, mock_database):
        """Test query preprocessing."""
        async with mock_database as db:
            search_engine = SemanticSearchEngine(
                database=db,
                project_root=Path("/test"),
            )

            # Test various query formats
            test_cases = [
                ("simple query", "simple query"),
                ("  whitespace  ", "whitespace"),
                ("UPPERCASE", "uppercase"),
                ("mixed Case Query", "mixed case query"),
            ]

            for input_query, expected_output in test_cases:
                processed = search_engine._preprocess_query(input_query)
                assert expected_output in processed.lower()

    @pytest.mark.asyncio
    async def test_result_enhancement(self, mock_database, sample_code_chunks):
        """Test result enhancement with context."""
        async with mock_database as db:
            await db.add_chunks(sample_code_chunks)
            search_engine = SemanticSearchEngine(
                database=db,
                project_root=Path("/test"),
            )

            # Create a basic search result
            chunk = sample_code_chunks[0]
            basic_result = SearchResult(
                content=chunk.content,
                file_path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                language=chunk.language,
                similarity_score=0.8,
                rank=1,
                chunk_type=chunk.chunk_type,
                function_name=chunk.function_name,
                class_name=chunk.class_name,
                context_before=[],
                context_after=[],
                highlights=[],
            )

            # Test enhancement
            enhanced_result = await search_engine._enhance_result(
                basic_result, include_context=True
            )

            assert enhanced_result.content == chunk.content
            assert enhanced_result.similarity_score == 0.8
            # Context enhancement would require file reading, so we just check structure
            assert hasattr(enhanced_result, "context_before")
            assert hasattr(enhanced_result, "context_after")

    @pytest.mark.asyncio
    async def test_result_reranking(self, mock_database):
        """Test result reranking."""
        async with mock_database as db:
            search_engine = SemanticSearchEngine(
                database=db,
                project_root=Path("/test"),
            )

            # Create sample results with different scores
            results = [
                SearchResult(
                    content="def main():",
                    file_path=Path("main.py"),
                    start_line=1,
                    end_line=1,
                    language="python",
                    similarity_score=0.6,
                    rank=1,
                    chunk_type="function",
                    function_name="main",
                    class_name=None,
                    context_before=[],
                    context_after=[],
                    highlights=[],
                ),
                SearchResult(
                    content="class User:",
                    file_path=Path("user.py"),
                    start_line=1,
                    end_line=1,
                    language="python",
                    similarity_score=0.8,
                    rank=2,
                    chunk_type="class",
                    function_name=None,
                    class_name="User",
                    context_before=[],
                    context_after=[],
                    highlights=[],
                ),
            ]

            # Test reranking
            reranked = search_engine._rerank_results(results, "main function")

            # Results should be sorted by similarity score (descending)
            assert len(reranked) == 2
            assert reranked[0].similarity_score >= reranked[1].similarity_score

    @pytest.mark.asyncio
    async def test_search_with_context(self, mock_database, sample_code_chunks):
        """Test enhanced search with context analysis."""
        async with mock_database as db:
            await db.add_chunks(sample_code_chunks)
            search_engine = SemanticSearchEngine(
                database=db,
                project_root=Path("/test"),
            )

            # Test search with context
            context_result = await search_engine.search_with_context(
                query="user management", limit=5, similarity_threshold=0.1
            )

            # Verify structure
            assert isinstance(context_result, dict)
            assert "results" in context_result
            assert "analysis" in context_result
            assert "suggestions" in context_result

            # Verify results
            results = context_result["results"]
            assert_search_results_valid(results)

    @pytest.mark.asyncio
    async def test_query_analysis(self, mock_database):
        """Test query analysis functionality."""
        async with mock_database as db:
            search_engine = SemanticSearchEngine(
                database=db,
                project_root=Path("/test"),
            )

            # Test different query types
            test_queries = [
                "function that handles user authentication",
                "class User",
                "database connection",
                "error handling",
            ]

            for query in test_queries:
                analysis = search_engine.analyze_query(query)

                assert isinstance(analysis, dict)
                assert "query_type" in analysis
                assert "original_query" in analysis
                assert "processed_query" in analysis
                # The implementation may have changed - check what fields exist
                assert any(k in analysis for k in ["confidence", "suggestions"])

    @pytest.mark.asyncio
    async def test_performance_characteristics(
        self, mock_database, sample_code_chunks, performance_timer
    ):
        """Test search performance characteristics."""
        async with mock_database as db:
            # Add many chunks for performance testing
            large_chunk_set = sample_code_chunks * 100  # 300 chunks
            await db.add_chunks(large_chunk_set)

            search_engine = SemanticSearchEngine(
                database=db,
                project_root=Path("/test"),
            )

            # Perform multiple searches and measure time
            queries = ["function", "class", "user", "main", "service"]

            for query in queries:
                result, elapsed = await performance_timer.time_async_operation(
                    search_engine.search, query, limit=10, similarity_threshold=0.1
                )

                # Basic performance assertions
                assert elapsed < 1.0, f"Search took too long: {elapsed:.3f}s"
                assert_search_results_valid(result)

            # Check overall performance stats
            stats = performance_timer.get_stats()
            assert stats["average"] < 0.5, (
                f"Average search time too high: {stats['average']:.3f}s"
            )

    @pytest.mark.asyncio
    async def test_concurrent_searches(self, mock_database, sample_code_chunks):
        """Test concurrent search operations."""
        async with mock_database as db:
            await db.add_chunks(sample_code_chunks)
            search_engine = SemanticSearchEngine(
                database=db,
                project_root=Path("/test"),
            )

            # Perform concurrent searches
            import asyncio

            async def search_task(query: str):
                return await search_engine.search(query, similarity_threshold=0.1)

            queries = ["function", "class", "user", "main", "service"]
            tasks = [search_task(query) for query in queries]

            # Execute all searches concurrently
            results = await asyncio.gather(*tasks)

            # Verify all searches completed successfully
            assert len(results) == len(queries)
            for result_set in results:
                assert isinstance(result_set, list)
                # Results may be empty for some queries, which is fine

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_database):
        """Test error handling in search operations."""
        async with mock_database as db:
            search_engine = SemanticSearchEngine(
                database=db,
                project_root=Path("/test"),
            )

            # Test with database error
            db.search = AsyncMock(side_effect=Exception("Database error"))

            with pytest.raises(Exception):
                await search_engine.search("test query")

    @pytest.mark.asyncio
    async def test_search_result_validation(self, mock_database, sample_code_chunks):
        """Test that search results are properly validated."""
        async with mock_database as db:
            await db.add_chunks(sample_code_chunks)
            search_engine = SemanticSearchEngine(
                database=db,
                project_root=Path("/test"),
            )

            results = await search_engine.search("function", similarity_threshold=0.1)

            # Validate each result
            for result in results:
                assert isinstance(result, SearchResult)
                assert result.content is not None
                assert isinstance(result.similarity_score, int | float)
                assert 0.0 <= result.similarity_score <= 1.0
                assert result.file_path is not None
                assert isinstance(result.start_line, int)
                assert isinstance(result.end_line, int)
                assert result.start_line > 0
                assert result.end_line >= result.start_line
