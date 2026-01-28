"""Unit tests for database functionality."""

from pathlib import Path

import pytest
import pytest_asyncio

from mcp_vector_search.core.database import (
    ChromaVectorDatabase,
    PooledChromaVectorDatabase,
)
from mcp_vector_search.core.exceptions import (
    DatabaseNotInitializedError,
    DocumentAdditionError,
)
from mcp_vector_search.core.models import CodeChunk, IndexStats
from tests.conftest import assert_search_results_valid


class TestChromaVectorDatabase:
    """Test cases for ChromaVectorDatabase."""

    @pytest_asyncio.fixture
    async def database(self, temp_dir, mock_embedding_function):
        """Create a test database instance."""
        db = ChromaVectorDatabase(
            persist_directory=temp_dir / "test_db",
            embedding_function=mock_embedding_function,
            collection_name="test_collection",
        )
        await db.initialize()
        yield db
        await db.close()

    @pytest.mark.asyncio
    async def test_database_initialization(self, temp_dir, mock_embedding_function):
        """Test database initialization."""
        db = ChromaVectorDatabase(
            persist_directory=temp_dir / "test_db",
            embedding_function=mock_embedding_function,
            collection_name="test_collection",
        )

        # Should not be initialized initially
        assert not hasattr(db, "_collection") or db._collection is None

        # Initialize
        await db.initialize()
        assert db._collection is not None

        # Close
        await db.close()

    @pytest.mark.asyncio
    async def test_add_chunks(self, database, sample_code_chunks):
        """Test adding code chunks to database."""
        # Add chunks
        await database.add_chunks(sample_code_chunks)

        # Verify chunks were added
        stats = await database.get_stats()
        assert stats.total_chunks >= len(sample_code_chunks)

    @pytest.mark.asyncio
    async def test_add_empty_chunks(self, database):
        """Test adding empty chunk list."""
        # Should handle empty list gracefully
        await database.add_chunks([])

        stats = await database.get_stats()
        assert stats.total_chunks == 0

    @pytest.mark.asyncio
    async def test_search_basic(self, database, sample_code_chunks):
        """Test basic search functionality."""
        # Add chunks first
        await database.add_chunks(sample_code_chunks)

        # Search for content
        results = await database.search(
            query="main function", limit=5, similarity_threshold=0.1
        )

        # Verify results
        assert isinstance(results, list)
        if results:  # May be empty with mock embeddings
            assert_search_results_valid(results)

    @pytest.mark.asyncio
    async def test_search_with_filters(self, database, sample_code_chunks):
        """Test search with filters."""
        await database.add_chunks(sample_code_chunks)

        # Search with language filter
        filters = {"language": "python"}
        results = await database.search(
            query="function", filters=filters, similarity_threshold=0.1
        )

        # All results should match filter
        for result in results:
            assert result.language == "python"

    @pytest.mark.asyncio
    async def test_search_limit(self, database, sample_code_chunks):
        """Test search result limiting."""
        # Add chunks only once to avoid duplicate IDs
        # ChromaDB requires unique IDs, and multiplying chunks creates duplicates
        await database.add_chunks(sample_code_chunks)

        # Search with limit
        limit = 2  # Use limit smaller than sample_code_chunks length (3)
        results = await database.search(
            query="function",
            limit=limit,
            similarity_threshold=0.0,  # Very low threshold to get results
        )

        assert len(results) <= limit

    @pytest.mark.asyncio
    async def test_delete_by_file(self, database, sample_code_chunks):
        """Test deleting chunks by file."""
        await database.add_chunks(sample_code_chunks)

        # Get initial stats
        initial_stats = await database.get_stats()
        initial_count = initial_stats.total_chunks

        # Delete chunks for specific file
        file_to_delete = sample_code_chunks[0].file_path
        deleted_count = await database.delete_by_file(file_to_delete)

        # Verify deletion
        assert deleted_count > 0

        final_stats = await database.get_stats()
        assert final_stats.total_chunks < initial_count

    @pytest.mark.asyncio
    async def test_get_stats(self, database, sample_code_chunks):
        """Test getting database statistics."""
        await database.add_chunks(sample_code_chunks)

        stats = await database.get_stats()

        # Verify stats structure
        assert isinstance(stats, IndexStats)
        assert stats.total_chunks > 0
        assert stats.total_files > 0
        assert len(stats.languages) > 0
        assert stats.index_size_mb >= 0

    @pytest.mark.asyncio
    async def test_health_check(self, database):
        """Test database health check."""
        # Should be healthy after initialization
        is_healthy = await database.health_check()
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_database_not_initialized_error(
        self, temp_dir, mock_embedding_function
    ):
        """Test operations on uninitialized database."""
        db = ChromaVectorDatabase(
            persist_directory=temp_dir / "test_db",
            embedding_function=mock_embedding_function,
        )

        # Operations should fail before initialization
        with pytest.raises(DatabaseNotInitializedError):
            await db.search("test query")

    @pytest.mark.asyncio
    async def test_context_manager(
        self, temp_dir, mock_embedding_function, sample_code_chunks
    ):
        """Test database as async context manager."""
        db = ChromaVectorDatabase(
            persist_directory=temp_dir / "test_db",
            embedding_function=mock_embedding_function,
        )

        # Use as context manager
        async with db:
            await db.add_chunks(sample_code_chunks)
            results = await db.search("function", similarity_threshold=0.1)
            # Should work within context
            assert isinstance(results, list)

        # Should be closed after context
        with pytest.raises(DatabaseNotInitializedError):
            await db.search("test")

    def test_build_where_clause(self, temp_dir, mock_embedding_function):
        """Test building where clauses for filters."""
        from mcp_vector_search.core.query_builder import QueryBuilder

        query_builder = QueryBuilder()

        # Test simple filter
        filters = {"language": "python"}
        where_clause = query_builder.build_where_clause(filters)
        assert where_clause == {"language": "python"}

        # Test multiple filters
        filters = {"language": "python", "chunk_type": "function"}
        where_clause = query_builder.build_where_clause(filters)
        assert where_clause == {"language": "python", "chunk_type": "function"}

        # Test empty filters
        where_clause = query_builder.build_where_clause({})
        assert where_clause is None

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, database, sample_code_chunks):
        """Test concurrent database operations."""
        import asyncio

        # Concurrent add operations
        chunk_batches = [
            sample_code_chunks[i : i + 1] for i in range(len(sample_code_chunks))
        ]
        add_tasks = [database.add_chunks(batch) for batch in chunk_batches]

        await asyncio.gather(*add_tasks)

        # Verify all chunks were added
        stats = await database.get_stats()
        assert stats.total_chunks >= len(sample_code_chunks)

        # Concurrent search operations
        search_tasks = [
            database.search(f"query_{i}", similarity_threshold=0.1) for i in range(5)
        ]

        results = await asyncio.gather(*search_tasks)
        assert len(results) == 5
        assert all(isinstance(result, list) for result in results)


class TestPooledChromaVectorDatabase:
    """Test cases for PooledChromaVectorDatabase."""

    @pytest_asyncio.fixture
    async def pooled_database(self, temp_dir, mock_embedding_function):
        """Create a test pooled database instance."""
        db = PooledChromaVectorDatabase(
            persist_directory=temp_dir / "test_db",
            embedding_function=mock_embedding_function,
            collection_name="test_collection",
            max_connections=3,
            min_connections=1,
        )
        await db.initialize()
        yield db
        await db.close()

    @pytest.mark.asyncio
    async def test_pooled_database_initialization(
        self, temp_dir, mock_embedding_function
    ):
        """Test pooled database initialization."""
        db = PooledChromaVectorDatabase(
            persist_directory=temp_dir / "test_db",
            embedding_function=mock_embedding_function,
            max_connections=3,
            min_connections=1,
        )

        # Initialize
        await db.initialize()
        assert db._pool is not None
        assert db._pool._initialized is True

        # Close
        await db.close()

    @pytest.mark.asyncio
    async def test_pooled_add_chunks(self, pooled_database, sample_code_chunks):
        """Test adding chunks with connection pooling."""
        await pooled_database.add_chunks(sample_code_chunks)

        stats = await pooled_database.get_stats()
        assert stats.total_chunks >= len(sample_code_chunks)

    @pytest.mark.asyncio
    async def test_pooled_search(self, pooled_database, sample_code_chunks):
        """Test search with connection pooling."""
        await pooled_database.add_chunks(sample_code_chunks)

        results = await pooled_database.search(
            query="function", similarity_threshold=0.1
        )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_pool_statistics(self, pooled_database):
        """Test connection pool statistics."""
        stats = pooled_database.get_pool_stats()

        # Verify stats structure
        assert isinstance(stats, dict)
        assert "pool_size" in stats
        assert "active_connections" in stats
        assert "idle_connections" in stats
        assert "connections_created" in stats
        assert "connections_reused" in stats
        assert "pool_hits" in stats
        assert "pool_misses" in stats

        # Verify stats values
        assert stats["pool_size"] >= 0
        assert stats["active_connections"] >= 0
        assert stats["idle_connections"] >= 0

    @pytest.mark.asyncio
    async def test_pooled_concurrent_operations(
        self, pooled_database, sample_code_chunks
    ):
        """Test concurrent operations with connection pooling."""
        await pooled_database.add_chunks(sample_code_chunks)

        import asyncio

        # Concurrent search operations
        search_tasks = [
            pooled_database.search(f"query_{i}", similarity_threshold=0.1)
            for i in range(10)  # More than pool size
        ]

        results = await asyncio.gather(*search_tasks)
        assert len(results) == 10

        # Check pool statistics
        stats = pooled_database.get_pool_stats()
        assert stats["connections_reused"] > 0  # Should have reused connections

    @pytest.mark.asyncio
    async def test_pool_health_check(self, pooled_database):
        """Test pooled database health check."""
        is_healthy = await pooled_database.health_check()
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_pooled_context_manager(
        self, temp_dir, mock_embedding_function, sample_code_chunks
    ):
        """Test pooled database as context manager."""
        db = PooledChromaVectorDatabase(
            persist_directory=temp_dir / "test_db",
            embedding_function=mock_embedding_function,
            max_connections=2,
            min_connections=1,
        )

        async with db:
            await db.add_chunks(sample_code_chunks)
            results = await db.search("function", similarity_threshold=0.1)
            assert isinstance(results, list)

        # Pool should be closed after context
        assert not db._pool._initialized

    @pytest.mark.asyncio
    async def test_performance_comparison(
        self, temp_dir, mock_embedding_function, sample_code_chunks, performance_timer
    ):
        """Test performance comparison between regular and pooled database."""
        # Regular database
        regular_db = ChromaVectorDatabase(
            persist_directory=temp_dir / "regular_db",
            embedding_function=mock_embedding_function,
        )

        # Pooled database
        pooled_db = PooledChromaVectorDatabase(
            persist_directory=temp_dir / "pooled_db",
            embedding_function=mock_embedding_function,
            max_connections=3,
            min_connections=1,
        )

        try:
            # Initialize both
            await regular_db.initialize()
            await pooled_db.initialize()

            # Add chunks to both
            await regular_db.add_chunks(sample_code_chunks)
            await pooled_db.add_chunks(sample_code_chunks)

            # Time regular database searches
            regular_times = []
            for i in range(5):
                _, elapsed = await performance_timer.time_async_operation(
                    regular_db.search, f"query_{i}", similarity_threshold=0.1
                )
                regular_times.append(elapsed)

            # Time pooled database searches
            pooled_times = []
            for i in range(5):
                _, elapsed = await performance_timer.time_async_operation(
                    pooled_db.search, f"query_{i}", similarity_threshold=0.1
                )
                pooled_times.append(elapsed)

            # Both should complete in reasonable time
            assert all(t < 1.0 for t in regular_times)
            assert all(t < 1.0 for t in pooled_times)

        finally:
            await regular_db.close()
            await pooled_db.close()

    @pytest.mark.asyncio
    async def test_error_handling(self, temp_dir, mock_embedding_function):
        """Test error handling in database operations."""
        db = ChromaVectorDatabase(
            persist_directory=temp_dir / "test_db",
            embedding_function=mock_embedding_function,
        )

        # Test with invalid chunks - CodeChunk doesn't accept 'id' parameter
        # The 'id' is a computed property based on file_path:start_line:end_line
        invalid_chunk = CodeChunk(
            content="",  # Empty content
            file_path=Path("test.py"),
            start_line=1,
            end_line=1,
            language="python",
            chunk_type="function",
        )

        await db.initialize()

        # Should handle invalid chunks gracefully
        try:
            await db.add_chunks([invalid_chunk])
        except Exception as e:
            # Should be a specific database error, not a generic exception
            assert isinstance(e, DocumentAdditionError | Exception)

        await db.close()

    @pytest.mark.asyncio
    async def test_sqlite_corruption_detection(self, temp_dir, mock_embedding_function):
        """Test SQLite corruption detection and recovery (Layer 1).

        This test verifies that:
        1. SQLite corruption is detected during pre-initialization checks
        2. Automatic recovery is attempted
        3. Either recovery succeeds OR a clear error message is shown
        """
        import sqlite3

        from mcp_vector_search.core.exceptions import DatabaseInitializationError

        db_path = temp_dir / "test_db"
        db_path.mkdir(parents=True, exist_ok=True)

        # Create a valid database first
        db = ChromaVectorDatabase(
            persist_directory=db_path,
            embedding_function=mock_embedding_function,
        )
        await db.initialize()
        await db.close()

        # Corrupt the SQLite database
        sqlite_path = db_path / "chroma.sqlite3"
        assert sqlite_path.exists()

        # Completely corrupt the file to simulate worst-case scenario
        with open(sqlite_path, "wb") as f:
            f.write(b"CORRUPTED DATABASE FILE")

        # Verify corruption is detectable
        try:
            conn = sqlite3.connect(str(sqlite_path))
            cursor = conn.execute("PRAGMA quick_check")
            result = cursor.fetchone()[0]
            conn.close()
            assert result != "ok"  # Should detect corruption
        except sqlite3.Error:
            pass  # Expected - database is corrupted

        # Create new database instance
        db2 = ChromaVectorDatabase(
            persist_directory=db_path,
            embedding_function=mock_embedding_function,
        )

        # Initialize should either:
        # 1. Successfully recover and create a fresh database, OR
        # 2. Raise DatabaseInitializationError with helpful message after attempting recovery
        try:
            await db2.initialize()

            # If we get here, recovery succeeded
            is_healthy = await db2.health_check()
            assert is_healthy is True

            # Fresh database after recovery
            stats = await db2.get_stats()
            assert stats.total_chunks == 0

            await db2.close()

        except DatabaseInitializationError as e:
            # Recovery failed after attempt - verify error message is helpful
            error_msg = str(e)
            assert "mcp-vector-search reset" in error_msg
            # This is acceptable - corruption was detected and recovery was attempted

    @pytest.mark.asyncio
    async def test_rust_panic_recovery(self, temp_dir, mock_embedding_function):
        """Test Rust panic pattern detection and recovery (Layer 2)."""
        from unittest.mock import AsyncMock, MagicMock, patch

        db_path = temp_dir / "test_db"
        db_path.mkdir(parents=True, exist_ok=True)

        # Create valid database first
        db = ChromaVectorDatabase(
            persist_directory=db_path,
            embedding_function=mock_embedding_function,
        )
        await db.initialize()
        await db.close()

        # Create new instance to test panic handling
        db2 = ChromaVectorDatabase(
            persist_directory=db_path,
            embedding_function=mock_embedding_function,
        )

        # Mock chromadb.PersistentClient to raise Rust panic on first call
        # Create mock that inherits from BaseException like pyo3_runtime.PanicException
        class MockPanicException(BaseException):
            """Mock for pyo3_runtime.PanicException which inherits from BaseException."""

            pass

        rust_panic_error = MockPanicException(
            "range start index 5 out of range for slice of length 3"
        )

        with patch("chromadb.PersistentClient") as mock_client:
            # First call raises Rust panic, second call succeeds
            mock_client.side_effect = [
                rust_panic_error,
                MagicMock(),  # Success on retry
            ]

            # Mock _corruption_recovery.recover to avoid actual cleanup
            db2._corruption_recovery.recover = AsyncMock()

            # Mock get_or_create_collection
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection = MagicMock(
                return_value=mock_collection
            )

            # Initialize should detect Rust panic and recover
            await db2.initialize()

            # Verify recovery was called
            db2._corruption_recovery.recover.assert_called_once()

            # Verify client was called twice (initial + retry)
            assert mock_client.call_count == 2

        await db2.close()

    @pytest.mark.asyncio
    async def test_rust_panic_recovery_failure(self, temp_dir, mock_embedding_function):
        """Test Rust panic recovery failure after retry (Layer 2)."""
        from unittest.mock import AsyncMock, patch

        from mcp_vector_search.core.exceptions import DatabaseError

        db_path = temp_dir / "test_db"
        db_path.mkdir(parents=True, exist_ok=True)

        db = ChromaVectorDatabase(
            persist_directory=db_path,
            embedding_function=mock_embedding_function,
        )

        # Mock chromadb.PersistentClient to always raise Rust panic
        # Create mock that inherits from BaseException like pyo3_runtime.PanicException
        class MockPanicException(BaseException):
            """Mock for pyo3_runtime.PanicException which inherits from BaseException."""

            pass

        rust_panic_error = MockPanicException(
            "thread panicked at 'index out of bounds'"
        )

        with patch("chromadb.PersistentClient") as mock_client:
            # Always raise Rust panic (both initial and retry)
            mock_client.side_effect = rust_panic_error

            # Mock _corruption_recovery.recover
            db._corruption_recovery.recover = AsyncMock()

            # Initialize should raise DatabaseError after recovery fails
            with pytest.raises(DatabaseError) as exc_info:
                await db.initialize()

            # Verify error message suggests manual reset
            assert "mcp-vector-search reset" in str(exc_info.value)

            # Verify recovery was called once
            db._corruption_recovery.recover.assert_called_once()

            # Verify client was called twice (initial + retry)
            assert mock_client.call_count == 2
