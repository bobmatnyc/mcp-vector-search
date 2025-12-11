"""Unit tests for corruption recovery and retry logic."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from mcp_vector_search.core.exceptions import RustPanicError, SearchError
from mcp_vector_search.core.search import SemanticSearchEngine


class TestCorruptionRecovery:
    """Test cases for corruption detection and recovery."""

    @pytest.mark.asyncio
    async def test_rust_panic_detection(self):
        """Test that Rust panic errors are correctly identified."""
        # Test the static method with various error messages
        rust_panic_msg = "range start index 10 out of range for slice of length 9"
        assert SemanticSearchEngine._is_rust_panic_error(Exception(rust_panic_msg))

        pyo3_panic_msg = "pyo3_runtime.PanicException: thread panicked"
        assert SemanticSearchEngine._is_rust_panic_error(Exception(pyo3_panic_msg))

        tokio_panic_msg = (
            "thread 'tokio-runtime-worker' panicked at rust/sqlite/src/db.rs"
        )
        assert SemanticSearchEngine._is_rust_panic_error(Exception(tokio_panic_msg))

        # Test non-panic error
        normal_error = "Connection timeout"
        assert not SemanticSearchEngine._is_rust_panic_error(Exception(normal_error))

    @pytest.mark.asyncio
    async def test_corruption_detection(self):
        """Test that corruption errors are correctly identified."""
        # Test pickle corruption
        pickle_error = "EOFError: Ran out of input while unpickling"
        assert SemanticSearchEngine._is_corruption_error(Exception(pickle_error))

        # Test HNSW corruption
        hnsw_error = "Failed to deserialize HNSW index"
        assert SemanticSearchEngine._is_corruption_error(Exception(hnsw_error))

        # Test non-corruption error
        normal_error = "Network timeout"
        assert not SemanticSearchEngine._is_corruption_error(Exception(normal_error))

    @pytest.mark.asyncio
    async def test_search_with_retry_success_on_first_attempt(self):
        """Test successful search on first attempt without retries."""
        # Setup mock database
        mock_database = AsyncMock()
        mock_results = [
            Mock(
                file_path=Path("/test/file.py"),
                start_line=1,
                end_line=5,
                content="test content",
                similarity_score=0.9,
            )
        ]
        mock_database.search = AsyncMock(return_value=mock_results)

        # Create search engine
        search_engine = SemanticSearchEngine(
            database=mock_database,
            project_root=Path("/test"),
        )

        # Execute search with retry
        results = await search_engine._search_with_retry(
            query="test query",
            limit=10,
            filters=None,
            threshold=0.5,
        )

        # Assert
        assert len(results) == 1
        assert mock_database.search.call_count == 1  # Only one call, no retries

    @pytest.mark.asyncio
    async def test_search_with_retry_transient_failure(self):
        """Test retry logic with transient Rust panic that succeeds on retry."""
        # Setup mock database that fails once then succeeds
        mock_database = AsyncMock()
        mock_results = [
            Mock(
                file_path=Path("/test/file.py"),
                start_line=1,
                end_line=5,
                content="test content",
                similarity_score=0.9,
            )
        ]

        # First call raises Rust panic, second call succeeds
        mock_database.search = AsyncMock(
            side_effect=[
                Exception("range start index 10 out of range for slice of length 9"),
                mock_results,
            ]
        )

        # Create search engine
        search_engine = SemanticSearchEngine(
            database=mock_database,
            project_root=Path("/test"),
        )

        # Execute search with retry
        results = await search_engine._search_with_retry(
            query="test query",
            limit=10,
            filters=None,
            threshold=0.5,
        )

        # Assert
        assert len(results) == 1
        assert mock_database.search.call_count == 2  # One failure, one success

    @pytest.mark.asyncio
    async def test_search_with_retry_persistent_rust_panic(self):
        """Test that persistent Rust panic raises RustPanicError after max retries."""
        # Setup mock database that always fails
        mock_database = AsyncMock()
        rust_panic_error = Exception(
            "range start index 10 out of range for slice of length 9"
        )
        mock_database.search = AsyncMock(side_effect=rust_panic_error)

        # Create search engine
        search_engine = SemanticSearchEngine(
            database=mock_database,
            project_root=Path("/test"),
        )

        # Execute and expect RustPanicError
        with pytest.raises(RustPanicError) as exc_info:
            await search_engine._search_with_retry(
                query="test query",
                limit=10,
                filters=None,
                threshold=0.5,
            )

        # Assert
        assert "HNSW index may be corrupted" in str(exc_info.value)
        assert "mcp-vector-search reset" in str(exc_info.value)
        assert mock_database.search.call_count == 3  # Max retries

    @pytest.mark.asyncio
    async def test_search_with_retry_corruption_error(self):
        """Test that corruption errors are raised immediately without retries."""
        # Setup mock database that returns corruption error
        mock_database = AsyncMock()
        corruption_error = Exception("EOFError: Ran out of input while unpickling")
        mock_database.search = AsyncMock(side_effect=corruption_error)

        # Create search engine
        search_engine = SemanticSearchEngine(
            database=mock_database,
            project_root=Path("/test"),
        )

        # Execute and expect SearchError (not RustPanicError)
        with pytest.raises(SearchError) as exc_info:
            await search_engine._search_with_retry(
                query="test query",
                limit=10,
                filters=None,
                threshold=0.5,
            )

        # Assert
        assert "Index corruption detected" in str(exc_info.value)
        assert "mcp-vector-search reset" in str(exc_info.value)
        assert mock_database.search.call_count == 1  # No retries for corruption

    @pytest.mark.asyncio
    async def test_search_with_retry_unknown_error(self):
        """Test that unknown errors are raised immediately as SearchError."""
        # Setup mock database that returns unknown error
        mock_database = AsyncMock()
        unknown_error = Exception("Unknown database error")
        mock_database.search = AsyncMock(side_effect=unknown_error)

        # Create search engine
        search_engine = SemanticSearchEngine(
            database=mock_database,
            project_root=Path("/test"),
        )

        # Execute and expect SearchError
        with pytest.raises(SearchError) as exc_info:
            await search_engine._search_with_retry(
                query="test query",
                limit=10,
                filters=None,
                threshold=0.5,
            )

        # Assert
        assert "Search failed" in str(exc_info.value)
        assert mock_database.search.call_count == 1  # No retries for unknown errors

    @pytest.mark.asyncio
    async def test_search_integration_with_retry(self):
        """Test that the main search method uses retry logic."""
        # Setup mock database with transient failure
        mock_database = AsyncMock()
        mock_results = [
            Mock(
                file_path=Path("/test/file.py"),
                start_line=1,
                end_line=5,
                content="test content",
                similarity_score=0.9,
                context_before=[],
                context_after=[],
                chunk_type="function",
                function_name="test_func",
                class_name=None,
                language="python",
                file_missing=False,
                rank=1,
            )
        ]

        # First call fails with Rust panic, second succeeds
        mock_database.search = AsyncMock(
            side_effect=[
                Exception("range start index 10 out of range for slice of length 9"),
                mock_results,
            ]
        )

        # Create search engine
        search_engine = SemanticSearchEngine(
            database=mock_database,
            project_root=Path("/test"),
        )

        # Mock file reading to avoid FileNotFoundError
        with patch("mcp_vector_search.core.search.aiofiles.open"):
            # Execute main search method
            results = await search_engine.search(
                query="test query",
                limit=10,
                include_context=False,  # Avoid file I/O
            )

        # Assert
        assert len(results) == 1
        assert mock_database.search.call_count == 2  # Retry worked

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        """Test that exponential backoff delays are applied correctly."""
        import time

        # Setup mock database with multiple failures
        mock_database = AsyncMock()
        mock_results = [Mock()]

        call_times = []

        async def track_call_time(*args, **kwargs):
            call_times.append(time.time())
            if len(call_times) < 3:
                # Fail first 2 attempts
                raise Exception(
                    "range start index 10 out of range for slice of length 9"
                )
            return mock_results

        mock_database.search = AsyncMock(side_effect=track_call_time)

        # Create search engine
        search_engine = SemanticSearchEngine(
            database=mock_database,
            project_root=Path("/test"),
        )

        # Execute search with retry
        await search_engine._search_with_retry(
            query="test query",
            limit=10,
            filters=None,
            threshold=0.5,
        )

        # Assert timing
        assert len(call_times) == 3

        # Check delays between calls
        # backoff_delays = [0, 0.1, 0.5]
        # - First attempt: immediate (no delay)
        # - Second attempt: 0.1s delay after first failure
        # - Third attempt: 0.5s delay after second failure

        # First retry (second call) should have ~100ms delay
        delay1 = call_times[1] - call_times[0]
        assert 0.05 < delay1 < 0.2  # ~100ms with tolerance

        # Second retry (third call) should have ~500ms delay
        delay2 = call_times[2] - call_times[1]
        assert 0.4 < delay2 < 0.7  # ~500ms with tolerance
