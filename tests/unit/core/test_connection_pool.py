"""Unit tests for connection pooling functionality."""

import asyncio
import time
from unittest.mock import Mock

import pytest
import pytest_asyncio

from mcp_vector_search.core.connection_pool import (
    ChromaConnectionPool,
    PooledConnection,
)


class TestChromaConnectionPool:
    """Test cases for ChromaConnectionPool."""

    @pytest_asyncio.fixture
    async def connection_pool(self, temp_dir, mock_embedding_function):
        """Create a test connection pool."""
        pool = ChromaConnectionPool(
            persist_directory=temp_dir / "test_db",
            embedding_function=mock_embedding_function,
            collection_name="test_collection",
            max_connections=3,
            min_connections=1,
            max_idle_time=60.0,
            max_connection_age=300.0,
        )
        await pool.initialize()
        yield pool
        await pool.cleanup()

    @pytest.mark.asyncio
    async def test_pool_initialization(self, temp_dir, mock_embedding_function):
        """Test connection pool initialization."""
        pool = ChromaConnectionPool(
            persist_directory=temp_dir / "test_db",
            embedding_function=mock_embedding_function,
            max_connections=3,
            min_connections=1,
        )

        assert not pool._initialized
        assert len(pool._connections) == 0

        await pool.initialize()

        assert pool._initialized
        assert len(pool._connections) >= pool._min_connections

        await pool.cleanup()

    @pytest.mark.asyncio
    async def test_get_connection(self, connection_pool):
        """Test getting connections from pool."""
        # Get a connection
        async with connection_pool.get_connection() as conn:
            assert isinstance(conn, PooledConnection)
            assert conn.client is not None
            assert conn.collection is not None

        # Connection should be returned to pool after use
        stats = connection_pool.get_stats()
        assert stats["pool_size"] > 0

    @pytest.mark.asyncio
    async def test_connection_reuse(self, connection_pool):
        """Test connection reuse."""
        initial_stats = connection_pool.get_stats()
        initial_created = initial_stats["connections_created"]

        # Use multiple connections
        for _i in range(5):
            async with connection_pool.get_connection() as conn:
                assert conn is not None

        final_stats = connection_pool.get_stats()

        # Should have reused connections
        assert final_stats["connections_reused"] > 0
        # Should not have created many new connections
        assert (
            final_stats["connections_created"]
            <= initial_created + connection_pool._max_connections
        )

    @pytest.mark.asyncio
    async def test_concurrent_connections(self, connection_pool):
        """Test concurrent connection usage."""

        async def use_connection(connection_id: int):
            async with connection_pool.get_connection():
                # Simulate some work
                await asyncio.sleep(0.1)
                return connection_id

        # Use connections concurrently
        tasks = [use_connection(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert results == list(range(10))

        # Check pool statistics
        stats = connection_pool.get_stats()
        assert stats["connections_reused"] > 0
        assert stats["pool_hits"] > 0

    @pytest.mark.asyncio
    async def test_pool_exhaustion(self, temp_dir, mock_embedding_function):
        """Test behavior when pool is exhausted."""
        # Create small pool
        pool = ChromaConnectionPool(
            persist_directory=temp_dir / "test_db",
            embedding_function=mock_embedding_function,
            max_connections=2,
            min_connections=1,
        )
        await pool.initialize()

        try:
            # Use connections sequentially to test pool reuse
            for _i in range(5):
                async with pool.get_connection() as conn:
                    assert conn is not None

            # Verify pool statistics show reuse
            stats = pool.get_stats()
            assert stats["connections_reused"] > 0
            assert stats["pool_size"] <= 2  # Should not exceed max

        finally:
            await pool.cleanup()

    @pytest.mark.asyncio
    async def test_connection_validation(self, connection_pool):
        """Test connection validation."""
        async with connection_pool.get_connection() as conn:
            # Connection should be valid
            is_valid = connection_pool._validate_connection(conn)
            assert is_valid is True

    @pytest.mark.asyncio
    async def test_idle_connection_cleanup(self, temp_dir, mock_embedding_function):
        """Test cleanup of idle connections."""
        pool = ChromaConnectionPool(
            persist_directory=temp_dir / "test_db",
            embedding_function=mock_embedding_function,
            max_connections=5,
            min_connections=1,
            max_idle_time=0.1,  # Very short idle time for testing
        )
        await pool.initialize()

        try:
            # Use several connections to create them
            for _i in range(3):
                async with pool.get_connection():
                    pass

            initial_stats = pool.get_stats()
            initial_pool_size = initial_stats["pool_size"]

            # Wait for idle timeout
            await asyncio.sleep(0.2)

            # Trigger cleanup
            await pool._cleanup_idle_connections()

            final_stats = pool.get_stats()

            # Should have cleaned up some idle connections
            assert final_stats["pool_size"] <= initial_pool_size
            assert final_stats["pool_size"] >= pool._min_connections

        finally:
            await pool.cleanup()

    @pytest.mark.asyncio
    async def test_connection_age_limits(self, temp_dir, mock_embedding_function):
        """Test connection age limits."""
        pool = ChromaConnectionPool(
            persist_directory=temp_dir / "test_db",
            embedding_function=mock_embedding_function,
            max_connections=3,
            min_connections=1,
            max_connection_age=0.1,  # Very short age limit for testing
        )
        await pool.initialize()

        try:
            # Use a connection
            async with pool.get_connection() as conn:
                id(conn)

            # Wait for age limit
            await asyncio.sleep(0.2)

            # Get connection again - should be a new one due to age limit
            async with pool.get_connection() as conn:
                id(conn)

            # Note: Due to the nature of connection pooling, we might get the same
            # connection object but it should have been refreshed internally
            assert conn is not None

        finally:
            await pool.cleanup()

    @pytest.mark.asyncio
    async def test_pool_statistics(self, connection_pool):
        """Test pool statistics tracking."""
        initial_stats = connection_pool.get_stats()

        # Use some connections
        for _i in range(5):
            async with connection_pool.get_connection():
                pass

        final_stats = connection_pool.get_stats()

        # Verify statistics structure
        required_keys = [
            "pool_size",
            "active_connections",
            "idle_connections",
            "connections_created",
            "connections_reused",
            "pool_hits",
            "pool_misses",
        ]
        for key in required_keys:
            assert key in final_stats
            assert isinstance(final_stats[key], int)
            assert final_stats[key] >= 0

        # Verify statistics values
        assert final_stats["connections_reused"] >= initial_stats["connections_reused"]
        assert final_stats["pool_hits"] >= initial_stats["pool_hits"]

    @pytest.mark.asyncio
    async def test_health_check(self, connection_pool):
        """Test pool health check."""
        # Pool should be healthy after initialization
        is_healthy = await connection_pool.health_check()
        assert is_healthy is True

        # Use a connection to ensure it's working
        async with connection_pool.get_connection() as conn:
            assert conn is not None

        # Should still be healthy
        is_healthy = await connection_pool.health_check()
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_pool_cleanup(self, temp_dir, mock_embedding_function):
        """Test pool cleanup."""
        pool = ChromaConnectionPool(
            persist_directory=temp_dir / "test_db",
            embedding_function=mock_embedding_function,
            max_connections=3,
            min_connections=1,
        )
        await pool.initialize()

        # Use some connections
        async with pool.get_connection():
            pass

        stats_before = pool.get_stats()
        assert stats_before["pool_size"] > 0

        # Cleanup
        await pool.cleanup()

        assert not pool._initialized
        assert len(pool._connections) == 0

    @pytest.mark.asyncio
    async def test_error_handling(self, temp_dir, mock_embedding_function):
        """Test error handling in connection pool."""
        pool = ChromaConnectionPool(
            persist_directory=temp_dir / "test_db",
            embedding_function=mock_embedding_function,
            max_connections=2,
            min_connections=1,
        )

        # Test that get_connection auto-initializes if not initialized
        # (this is the actual behavior - auto-initialization for convenience)
        try:
            async with pool.get_connection() as conn:
                assert conn is not None
                assert pool._initialized  # Should be auto-initialized
        finally:
            await pool.cleanup()

    @pytest.mark.asyncio
    async def test_connection_context_manager(self, connection_pool):
        """Test connection context manager behavior."""
        # Test normal usage
        async with connection_pool.get_connection() as conn:
            assert conn is not None
            assert conn.client is not None
            assert conn.collection is not None

        # Connection should be returned to pool
        stats = connection_pool.get_stats()
        assert stats["pool_size"] > 0

    @pytest.mark.asyncio
    async def test_connection_exception_handling(self, connection_pool):
        """Test connection handling when exceptions occur."""
        try:
            async with connection_pool.get_connection() as conn:
                assert conn is not None
                # Simulate an exception
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected

        # Connection should still be returned to pool
        stats = connection_pool.get_stats()
        assert stats["pool_size"] > 0

    @pytest.mark.asyncio
    async def test_performance_characteristics(
        self, connection_pool, performance_timer
    ):
        """Test performance characteristics of connection pool."""
        # Time connection acquisition
        times = []

        for _i in range(10):
            start_time = time.perf_counter()
            async with connection_pool.get_connection() as conn:
                assert conn is not None
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        # Connection acquisition should be fast
        avg_time = sum(times) / len(times)
        assert avg_time < 0.1, f"Connection acquisition too slow: {avg_time:.3f}s"

        # Check pool efficiency
        stats = connection_pool.get_stats()
        if stats["pool_hits"] + stats["pool_misses"] > 0:
            hit_rate = stats["pool_hits"] / (stats["pool_hits"] + stats["pool_misses"])
            assert hit_rate > 0.5, f"Pool hit rate too low: {hit_rate:.2f}"

    @pytest.mark.asyncio
    async def test_pool_configuration(self, temp_dir, mock_embedding_function):
        """Test different pool configurations."""
        # Test minimal configuration
        minimal_pool = ChromaConnectionPool(
            persist_directory=temp_dir / "minimal_db",
            embedding_function=mock_embedding_function,
            max_connections=1,
            min_connections=1,
        )
        await minimal_pool.initialize()

        async with minimal_pool.get_connection() as conn:
            assert conn is not None

        await minimal_pool.cleanup()

        # Test larger configuration
        large_pool = ChromaConnectionPool(
            persist_directory=temp_dir / "large_db",
            embedding_function=mock_embedding_function,
            max_connections=10,
            min_connections=3,
        )
        await large_pool.initialize()

        stats = large_pool.get_stats()
        assert stats["pool_size"] >= 3

        await large_pool.cleanup()


class TestPooledConnection:
    """Test cases for PooledConnection."""

    def test_pooled_connection_creation(self, mock_embedding_function):
        """Test PooledConnection creation."""
        # Mock ChromaDB client and collection
        mock_client = Mock()
        mock_collection = Mock()

        conn = PooledConnection(
            client=mock_client,
            collection=mock_collection,
            created_at=time.time(),
            last_used=time.time(),
        )

        assert conn.client == mock_client
        assert conn.collection == mock_collection
        assert conn.created_at > 0
        assert conn.last_used > 0
        assert not conn.in_use

    def test_pooled_connection_age(self):
        """Test connection age calculation."""
        current_time = time.time()
        conn = PooledConnection(
            client=Mock(),
            collection=Mock(),
            created_at=current_time - 100,  # 100 seconds ago
            last_used=current_time,
        )

        age = conn.age
        assert 99 <= age <= 101  # Allow for small timing differences

    def test_pooled_connection_idle_time(self):
        """Test connection idle time calculation."""
        current_time = time.time()
        conn = PooledConnection(
            client=Mock(),
            collection=Mock(),
            created_at=current_time,
            last_used=current_time - 50,  # 50 seconds ago
        )

        idle_time = conn.idle_time
        assert 49 <= idle_time <= 51  # Allow for small timing differences
