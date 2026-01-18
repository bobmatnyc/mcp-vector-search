"""Connection pooling for vector database operations."""

import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from .exceptions import DatabaseError


@dataclass
class PooledConnection:
    """Represents a pooled database connection."""

    client: Any
    collection: Any
    created_at: float
    last_used: float
    in_use: bool = False
    use_count: int = 0

    @property
    def age(self) -> float:
        """Get the age of this connection in seconds."""
        return time.time() - self.created_at

    @property
    def idle_time(self) -> float:
        """Get the idle time of this connection in seconds."""
        return time.time() - self.last_used


class ChromaConnectionPool:
    """Connection pool for ChromaDB operations."""

    def __init__(
        self,
        persist_directory: Path,
        embedding_function: Any,
        collection_name: str = "code_search",
        max_connections: int = 10,
        min_connections: int = 2,
        max_idle_time: float = 300.0,  # 5 minutes
        max_connection_age: float = 3600.0,  # 1 hour
    ):
        """Initialize connection pool.

        Args:
            persist_directory: Directory to persist database
            embedding_function: Function to generate embeddings
            collection_name: Name of the collection
            max_connections: Maximum number of connections in pool
            min_connections: Minimum number of connections to maintain
            max_idle_time: Maximum time a connection can be idle (seconds)
            max_connection_age: Maximum age of a connection (seconds)
        """
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.max_idle_time = max_idle_time
        self.max_connection_age = max_connection_age

        self._pool: list[PooledConnection] = []
        self._lock = asyncio.Lock()
        self._initialized = False
        self._cleanup_task: asyncio.Task | None = None

        # Statistics
        self._stats = {
            "connections_created": 0,
            "connections_reused": 0,
            "connections_expired": 0,
            "pool_hits": 0,
            "pool_misses": 0,
        }

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            # Create minimum number of connections
            for _ in range(self.min_connections):
                conn = await self._create_connection()
                self._pool.append(conn)

            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._initialized = True

            logger.info(
                f"Connection pool initialized with {len(self._pool)} connections"
            )

    async def close(self) -> None:
        """Close all connections and cleanup."""
        if not self._initialized:
            return

        async with self._lock:
            # Cancel cleanup task
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            # Close all connections
            for conn in self._pool:
                await self._close_connection(conn)

            self._pool.clear()
            self._initialized = False

            logger.info("Connection pool closed")

    async def _create_connection(self) -> PooledConnection:
        """Create a new database connection."""
        try:
            import chromadb

            # Ensure directory exists
            self.persist_directory.mkdir(parents=True, exist_ok=True)

            # Create client
            client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            # Configure SQLite busy_timeout to prevent indefinite waits on locked database
            # This gives SQLite 30 seconds to acquire a lock before raising an error
            try:
                # Access the underlying SQLite connection through ChromaDB's internal API
                # ChromaDB uses DuckDB by default, but falls back to SQLite for persistence
                chroma_db_path = self.persist_directory / "chroma.sqlite3"
                if chroma_db_path.exists():
                    import sqlite3

                    # Open a direct connection to configure busy_timeout
                    temp_conn = sqlite3.connect(str(chroma_db_path))
                    temp_conn.execute("PRAGMA busy_timeout = 30000")  # 30 seconds
                    temp_conn.close()
                    logger.debug(
                        "Configured SQLite busy_timeout=30000ms for deadlock protection"
                    )
            except Exception as e:
                logger.warning(f"Failed to configure SQLite busy_timeout: {e}")

            # Create or get collection
            collection = client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={
                    "description": "Semantic code search collection",
                },
            )

            conn = PooledConnection(
                client=client,
                collection=collection,
                created_at=time.time(),
                last_used=time.time(),
            )

            self._stats["connections_created"] += 1
            logger.debug(
                f"Created new database connection (total: {self._stats['connections_created']})"
            )

            return conn

        except Exception as e:
            logger.error(f"Failed to create database connection: {e}")
            raise DatabaseError(f"Connection creation failed: {e}") from e

    async def _close_connection(self, conn: PooledConnection) -> None:
        """Close a database connection."""
        try:
            # ChromaDB doesn't require explicit closing
            conn.client = None
            conn.collection = None
            logger.debug("Closed database connection")
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[PooledConnection, None]:
        """Get a connection from the pool."""
        if not self._initialized:
            await self.initialize()

        conn = None
        try:
            # Get connection from pool
            conn = await self._acquire_connection()
            yield conn
        finally:
            # Return connection to pool
            if conn:
                await self._release_connection(conn)

    async def _acquire_connection(self) -> PooledConnection:
        """Acquire a connection from the pool."""
        async with self._lock:
            # Try to find an available connection
            for conn in self._pool:
                if not conn.in_use and self._is_connection_valid(conn):
                    conn.in_use = True
                    conn.last_used = time.time()
                    conn.use_count += 1
                    self._stats["pool_hits"] += 1
                    self._stats["connections_reused"] += 1
                    logger.debug(f"Reused connection (use count: {conn.use_count})")
                    return conn

            # No available connection, create new one if under limit
            if len(self._pool) < self.max_connections:
                conn = await self._create_connection()
                conn.in_use = True
                self._pool.append(conn)
                self._stats["pool_misses"] += 1
                logger.debug(f"Created new connection (pool size: {len(self._pool)})")
                return conn

        # Pool is full, wait for a connection to become available (outside lock)
        self._stats["pool_misses"] += 1
        logger.warning("Connection pool exhausted, waiting for available connection")

        # Wait for a connection (with timeout) - release lock during wait
        timeout = 30.0  # 30 seconds
        start_time = time.time()

        while time.time() - start_time < timeout:
            await asyncio.sleep(0.1)
            # Re-acquire lock to check for available connections
            async with self._lock:
                for conn in self._pool:
                    if not conn.in_use and self._is_connection_valid(conn):
                        conn.in_use = True
                        conn.last_used = time.time()
                        conn.use_count += 1
                        self._stats["connections_reused"] += 1
                        return conn

        raise DatabaseError("Connection pool timeout: no connections available")

    async def _release_connection(self, conn: PooledConnection) -> None:
        """Release a connection back to the pool."""
        async with self._lock:
            conn.in_use = False
            conn.last_used = time.time()
            logger.debug(f"Released connection (use count: {conn.use_count})")

    def _is_connection_valid(self, conn: PooledConnection) -> bool:
        """Check if a connection is still valid."""
        now = time.time()

        # Check age
        if now - conn.created_at > self.max_connection_age:
            return False

        # Check if idle too long
        if now - conn.last_used > self.max_idle_time:
            return False

        # Check if client/collection are still valid
        if not conn.client or not conn.collection:
            return False

        return True

    async def _cleanup_loop(self) -> None:
        """Background task to cleanup expired connections."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_expired_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in connection cleanup: {e}")

    async def _cleanup_expired_connections(self) -> None:
        """Remove expired connections from the pool."""
        async with self._lock:
            expired_connections = []

            for conn in self._pool:
                if not conn.in_use and not self._is_connection_valid(conn):
                    expired_connections.append(conn)

            # Remove expired connections
            for conn in expired_connections:
                self._pool.remove(conn)
                await self._close_connection(conn)
                self._stats["connections_expired"] += 1

            if expired_connections:
                logger.debug(
                    f"Cleaned up {len(expired_connections)} expired connections"
                )

            # Ensure minimum connections
            while len(self._pool) < self.min_connections:
                try:
                    conn = await self._create_connection()
                    self._pool.append(conn)
                except Exception as e:
                    logger.error(f"Failed to create minimum connection: {e}")
                    break

    def get_stats(self) -> dict[str, Any]:
        """Get connection pool statistics."""
        active_connections = sum(1 for conn in self._pool if conn.in_use)
        idle_connections = len(self._pool) - active_connections

        return {
            **self._stats,
            "pool_size": len(self._pool),
            "active_connections": active_connections,
            "idle_connections": idle_connections,
            "max_connections": self.max_connections,
            "min_connections": self.min_connections,
        }

    async def health_check(self) -> bool:
        """Perform a health check on the connection pool."""
        try:
            async with self.get_connection() as conn:
                # Try a simple operation
                conn.collection.count()
                return True
        except Exception as e:
            logger.error(f"Connection pool health check failed: {e}")
            return False

    # Backward compatibility aliases for old test API
    async def cleanup(self) -> None:
        """Alias for close() method (backward compatibility)."""
        await self.close()

    def _validate_connection(self, conn: PooledConnection) -> bool:
        """Alias for _is_connection_valid() method (backward compatibility)."""
        return self._is_connection_valid(conn)

    async def _cleanup_idle_connections(self) -> None:
        """Alias for _cleanup_expired_connections() method (backward compatibility)."""
        await self._cleanup_expired_connections()

    @property
    def _connections(self) -> list[PooledConnection]:
        """Alias for _pool attribute (backward compatibility)."""
        return self._pool

    @property
    def _max_connections(self) -> int:
        """Alias for max_connections attribute (backward compatibility)."""
        return self.max_connections

    @property
    def _min_connections(self) -> int:
        """Alias for min_connections attribute (backward compatibility)."""
        return self.min_connections
