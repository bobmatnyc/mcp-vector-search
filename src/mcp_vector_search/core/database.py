"""Database abstraction and ChromaDB implementation for MCP Vector Search."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from .collection_manager import CollectionManager
from .connection_pool import ChromaConnectionPool
from .corruption_recovery import CorruptionRecovery
from .dimension_checker import DimensionChecker
from .exceptions import (
    DatabaseError,
    DatabaseInitializationError,
    DatabaseNotInitializedError,
    DocumentAdditionError,
)
from .metadata_converter import MetadataConverter
from .models import CodeChunk, IndexStats, SearchResult
from .query_builder import QueryBuilder
from .search_handler import SearchHandler
from .statistics_collector import StatisticsCollector


@runtime_checkable
class EmbeddingFunction(Protocol):
    """Protocol for embedding functions."""

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for input texts."""
        ...


class VectorDatabase(ABC):
    """Abstract interface for vector database operations."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the database connection and collections."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close database connections and cleanup resources."""
        ...

    @abstractmethod
    async def add_chunks(
        self, chunks: list[CodeChunk], metrics: dict[str, Any] | None = None
    ) -> None:
        """Add code chunks to the database with optional structural metrics.

        Args:
            chunks: List of code chunks to add
            metrics: Optional dict mapping chunk IDs to ChunkMetrics.to_metadata() dicts
        """
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        similarity_threshold: float = 0.7,
    ) -> list[SearchResult]:
        """Search for similar code chunks.

        Args:
            query: Search query
            limit: Maximum number of results
            filters: Optional filters to apply
            similarity_threshold: Minimum similarity score

        Returns:
            List of search results
        """
        ...

    @abstractmethod
    async def delete_by_file(self, file_path: Path) -> int:
        """Delete all chunks for a specific file.

        Args:
            file_path: Path to the file

        Returns:
            Number of deleted chunks
        """
        ...

    @abstractmethod
    async def get_stats(self) -> IndexStats:
        """Get database statistics.

        Returns:
            Index statistics
        """
        ...

    @abstractmethod
    async def reset(self) -> None:
        """Reset the database (delete all data)."""
        ...

    @abstractmethod
    async def get_all_chunks(self) -> list[CodeChunk]:
        """Get all chunks from the database.

        Returns:
            List of all code chunks with metadata
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check database health and integrity.

        Returns:
            True if database is healthy, False otherwise
        """
        ...

    async def __aenter__(self) -> "VectorDatabase":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


class ChromaVectorDatabase(VectorDatabase):
    """ChromaDB implementation of vector database.

    This class coordinates multiple specialized helper classes:
    - CollectionManager: Collection lifecycle management
    - CorruptionRecovery: Corruption detection and recovery
    - MetadataConverter: CodeChunk <-> ChromaDB metadata conversion
    - QueryBuilder: Build ChromaDB where clauses
    - SearchHandler: Execute searches and process results
    - StatisticsCollector: Gather database statistics
    - DimensionChecker: Check embedding dimension compatibility
    """

    def __init__(
        self,
        persist_directory: Path,
        embedding_function: EmbeddingFunction,
        collection_name: str = "code_search",
    ) -> None:
        """Initialize ChromaDB vector database.

        Args:
            persist_directory: Directory to persist database
            embedding_function: Function to generate embeddings
            collection_name: Name of the collection
        """
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self._client = None
        self._collection = None

        # Initialize helper classes
        self._collection_manager = CollectionManager(collection_name)
        self._corruption_recovery = CorruptionRecovery(persist_directory)
        self._metadata_converter = MetadataConverter()
        self._query_builder = QueryBuilder()
        self._search_handler = SearchHandler()
        self._statistics_collector = StatisticsCollector(persist_directory)
        self._dimension_checker = DimensionChecker()

    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection with corruption recovery."""
        try:
            import chromadb

            # Ensure directory exists
            self.persist_directory.mkdir(parents=True, exist_ok=True)

            # LAYER 1: Proactive corruption detection (SQLite + HNSW .pkl/.bin files)
            # This MUST run before any ChromaDB operations to prevent bus errors
            if await self._corruption_recovery.detect_corruption():
                logger.info("Corruption detected, initiating automatic recovery...")
                await self._corruption_recovery.recover()

            # LAYER 2: Wrap ChromaDB initialization with Rust panic detection
            try:
                # Create client
                self._client = chromadb.PersistentClient(
                    path=str(self.persist_directory),
                    settings=chromadb.Settings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                    ),
                )

                # Configure SQLite timeout
                self._collection_manager.configure_sqlite_timeout(
                    self.persist_directory
                )

                # Create or get collection with optimized HNSW parameters
                self._collection = self._collection_manager.get_or_create_collection(
                    self._client, self.embedding_function
                )

                # LAYER 3: Check for dimension mismatch (migration detection)
                # This uses subprocess-isolated count() to survive bus errors
                await self._dimension_checker.check_compatibility(
                    self._collection, self.embedding_function
                )

                logger.debug(f"ChromaDB initialized at {self.persist_directory}")

            except BaseException as init_error:
                # Re-raise system exceptions we should never catch
                if isinstance(
                    init_error, KeyboardInterrupt | SystemExit | GeneratorExit
                ):
                    raise

                # LAYER 2: Detect Rust panic patterns during initialization
                if self._corruption_recovery.is_rust_panic_error(init_error):
                    logger.warning(
                        f"Rust panic detected during ChromaDB initialization: {init_error}"
                    )
                    await self._handle_initialization_recovery()
                else:
                    # Not a Rust panic, re-raise original exception
                    raise

        except (DatabaseError, DatabaseInitializationError):
            # Re-raise our own errors without re-processing
            raise
        except Exception as e:
            # Check if this is a corruption error (legacy detection)
            if self._corruption_recovery.is_corruption_error(e):
                # Prevent infinite recursion
                if self._corruption_recovery.recovery_attempted:
                    logger.error(
                        f"Recovery already attempted but corruption persists: {e}"
                    )
                    raise DatabaseInitializationError(
                        f"Failed to recover from database corruption. "
                        f"Please run 'mcp-vector-search reset index' to clear and rebuild the database. Error: {e}"
                    ) from e

                logger.warning(f"Detected index corruption: {e}")
                await self._corruption_recovery.recover()

                # Retry initialization ONE TIME
                await self.initialize()
            else:
                logger.debug(f"Failed to initialize ChromaDB: {e}")
                raise DatabaseInitializationError(
                    f"ChromaDB initialization failed: {e}"
                ) from e

    async def _handle_initialization_recovery(self) -> None:
        """Handle recovery during initialization and retry.

        Raises:
            DatabaseError: If recovery fails
        """
        logger.info("Attempting automatic recovery from database corruption...")
        await self._corruption_recovery.recover()

        # Retry initialization ONCE after recovery
        try:
            import chromadb

            logger.info("Retrying ChromaDB initialization after recovery...")
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            self._collection = self._collection_manager.get_or_create_collection(
                self._client, self.embedding_function
            )

            logger.info("ChromaDB successfully initialized after recovery")

        except BaseException as retry_error:
            # Re-raise system exceptions
            if isinstance(retry_error, KeyboardInterrupt | SystemExit | GeneratorExit):
                raise

            logger.error(f"Failed to recover from database corruption: {retry_error}")
            raise DatabaseError(
                f"Failed to recover from database corruption. "
                f"Please run 'mcp-vector-search reset index' to clear the database. "
                f"Error: {retry_error}"
            ) from retry_error

    async def remove_file_chunks(self, file_path: str) -> int:
        """Remove all chunks for a specific file.

        Args:
            file_path: Relative path to the file

        Returns:
            Number of chunks removed
        """
        if not self._collection:
            raise DatabaseNotInitializedError("Database not initialized")

        try:
            # Get all chunks for this file
            results = self._collection.get(where={"file_path": file_path})

            if not results["ids"]:
                return 0

            # Delete the chunks
            self._collection.delete(ids=results["ids"])

            removed_count = len(results["ids"])
            logger.debug(f"Removed {removed_count} chunks for file: {file_path}")
            return removed_count

        except Exception as e:
            logger.error(f"Failed to remove chunks for file {file_path}: {e}")
            return 0

    async def close(self) -> None:
        """Close database connections."""
        if self._client:
            # ChromaDB doesn't require explicit closing
            self._client = None
            self._collection = None
            logger.debug("ChromaDB connections closed")

    async def add_chunks(
        self, chunks: list[CodeChunk], metrics: dict[str, Any] | None = None
    ) -> None:
        """Add code chunks to the database with optional structural metrics.

        Args:
            chunks: List of code chunks to add
            metrics: Optional dict mapping chunk IDs to ChunkMetrics.to_metadata() dicts
        """
        if not self._collection:
            raise DatabaseNotInitializedError("Database not initialized")

        if not chunks:
            return

        try:
            documents = []
            metadatas = []
            ids = []

            for chunk in chunks:
                # Store original content directly in documents
                documents.append(chunk.content)

                # Convert chunk to metadata format
                metadata = self._metadata_converter.chunk_to_metadata(chunk, metrics)
                metadatas.append(metadata)

                # Use chunk ID
                ids.append(chunk.chunk_id or chunk.id)

            # Add to collection
            self._collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )

            logger.debug(f"Added {len(chunks)} chunks to database")

        except Exception as e:
            logger.error(f"Failed to add chunks: {e}")
            raise DocumentAdditionError(f"Failed to add chunks: {e}") from e

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        similarity_threshold: float = 0.7,
    ) -> list[SearchResult]:
        """Search for similar code chunks."""
        if not self._collection:
            raise DatabaseNotInitializedError("Database not initialized")

        # Build where clause
        where_clause = self._query_builder.build_where_clause(filters)

        # Execute search
        results = self._search_handler.execute_search(
            self._collection, query, limit, where_clause
        )

        # Process and return results
        return self._search_handler.process_results(
            results, query, similarity_threshold
        )

    async def delete_by_file(self, file_path: Path) -> int:
        """Delete all chunks for a specific file."""
        if not self._collection:
            raise DatabaseNotInitializedError("Database not initialized")

        try:
            # Get all chunks for this file
            results = self._collection.get(
                where={"file_path": str(file_path)},
                include=["metadatas"],
            )

            if results["ids"]:
                self._collection.delete(ids=results["ids"])
                count = len(results["ids"])
                logger.debug(f"Deleted {count} chunks for {file_path}")
                return count

            return 0

        except Exception as e:
            logger.error(f"Failed to delete chunks for {file_path}: {e}")
            raise DatabaseError(f"Failed to delete chunks: {e}") from e

    async def get_stats(self, skip_stats: bool = False) -> IndexStats:
        """Get database statistics with optimized chunked queries.

        Args:
            skip_stats: If True, skip detailed statistics collection
        """
        if not self._collection:
            raise DatabaseNotInitializedError("Database not initialized")

        return await self._statistics_collector.collect_stats(
            self._collection, skip_stats
        )

    async def reset(self) -> None:
        """Reset the database."""
        if self._client:
            try:
                self._collection_manager.reset_collection(self._client)
                # Recreate collection
                await self.initialize()
                logger.info("Database reset successfully")
            except Exception as e:
                logger.error(f"Failed to reset database: {e}")
                raise DatabaseError(f"Failed to reset database: {e}") from e

    async def get_all_chunks(self) -> list[CodeChunk]:
        """Get all chunks from the database.

        Returns:
            List of all code chunks with metadata
        """
        if not self._collection:
            raise DatabaseNotInitializedError("Database not initialized")

        try:
            # Get all documents from collection
            results = self._collection.get(include=["metadatas", "documents"])

            chunks = []
            if results and results.get("ids"):
                for i, _chunk_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i]
                    content = results["documents"][i]

                    # Convert metadata back to CodeChunk
                    chunk = self._metadata_converter.metadata_to_chunk(
                        metadata, content
                    )
                    chunks.append(chunk)

            logger.debug(f"Retrieved {len(chunks)} chunks from database")
            return chunks

        except Exception as e:
            logger.error(f"Failed to get all chunks: {e}")
            raise DatabaseError(f"Failed to get all chunks: {e}") from e

    async def health_check(self) -> bool:
        """Check database health and integrity.

        Returns:
            True if database is healthy, False otherwise
        """
        try:
            # First check if client is initialized
            if not self._client or not self._collection:
                logger.warning("Database not initialized")
                return False

            # Try a simple operation to test the connection
            try:
                # Attempt to get count - this will fail if index is corrupted
                count = self._collection.count()
                logger.debug(f"Health check passed: {count} chunks in database")

                # Try a minimal query to ensure search works
                self._collection.query(
                    query_texts=["test"], n_results=1, include=["metadatas"]
                )

                return True

            except Exception as e:
                if self._corruption_recovery.is_corruption_error(e):
                    logger.error(f"Index corruption detected during health check: {e}")
                    return False
                else:
                    # Some other error
                    logger.warning(f"Health check failed: {e}")
                    return False

        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False


class PooledChromaVectorDatabase(VectorDatabase):
    """ChromaDB implementation with connection pooling for improved performance.

    Uses the same helper classes as ChromaVectorDatabase for consistency:
    - MetadataConverter: CodeChunk <-> ChromaDB metadata conversion
    - QueryBuilder: Build ChromaDB where clauses
    - SearchHandler: Execute searches and process results
    - CorruptionRecovery: Corruption detection and recovery
    """

    def __init__(
        self,
        persist_directory: Path,
        embedding_function: EmbeddingFunction,
        collection_name: str = "code_search",
        max_connections: int = 10,
        min_connections: int = 2,
        max_idle_time: float = 300.0,
        max_connection_age: float = 3600.0,
    ) -> None:
        """Initialize pooled ChromaDB vector database.

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

        self._pool = ChromaConnectionPool(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            collection_name=collection_name,
            max_connections=max_connections,
            min_connections=min_connections,
            max_idle_time=max_idle_time,
            max_connection_age=max_connection_age,
        )

        # Initialize helper classes
        self._metadata_converter = MetadataConverter()
        self._query_builder = QueryBuilder()
        self._search_handler = SearchHandler()
        self._corruption_recovery = CorruptionRecovery(persist_directory)

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        await self._pool.initialize()
        logger.debug(f"Pooled ChromaDB initialized at {self.persist_directory}")

    async def close(self) -> None:
        """Close the connection pool."""
        await self._pool.close()
        logger.debug("Pooled ChromaDB connections closed")

    async def add_chunks(
        self, chunks: list[CodeChunk], metrics: dict[str, Any] | None = None
    ) -> None:
        """Add code chunks to the database using pooled connection with optional metrics.

        Args:
            chunks: List of code chunks to add
            metrics: Optional dict mapping chunk IDs to ChunkMetrics.to_metadata() dicts
        """
        if not chunks:
            return

        # Ensure pool is initialized
        if not self._pool._initialized:
            await self._pool.initialize()

        try:
            async with self._pool.get_connection() as conn:
                documents = []
                metadatas = []
                ids = []

                for chunk in chunks:
                    documents.append(chunk.content)
                    # Use helper to convert chunk to metadata
                    metadata = self._metadata_converter.chunk_to_metadata(
                        chunk, metrics
                    )
                    metadatas.append(metadata)
                    ids.append(chunk.chunk_id or chunk.id)

                # Add to collection
                conn.collection.add(documents=documents, metadatas=metadatas, ids=ids)
                logger.debug(f"Added {len(chunks)} chunks to database")

        except Exception as e:
            logger.error(f"Failed to add chunks: {e}")
            raise DocumentAdditionError(f"Failed to add chunks: {e}") from e

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        similarity_threshold: float = 0.7,
    ) -> list[SearchResult]:
        """Search for similar code chunks using pooled connection."""
        # Ensure pool is initialized
        if not self._pool._initialized:
            await self._pool.initialize()

        async with self._pool.get_connection() as conn:
            # Build where clause
            where_clause = self._query_builder.build_simple_where_clause(filters)

            # Execute search using helper
            results = self._search_handler.execute_search(
                conn.collection, query, limit, where_clause
            )

            # Process and return results
            return self._search_handler.process_results(
                results, query, similarity_threshold
            )

    async def delete_by_file(self, file_path: Path) -> int:
        """Delete all chunks for a specific file using pooled connection."""
        try:
            async with self._pool.get_connection() as conn:
                # Get all chunks for this file
                results = conn.collection.get(
                    where={"file_path": str(file_path)}, include=["metadatas"]
                )

                if not results["ids"]:
                    return 0

                # Delete the chunks
                conn.collection.delete(ids=results["ids"])

                deleted_count = len(results["ids"])
                logger.debug(f"Deleted {deleted_count} chunks for file: {file_path}")
                return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete chunks for file {file_path}: {e}")
            raise DatabaseError(f"Failed to delete chunks: {e}") from e

    async def get_stats(self) -> IndexStats:
        """Get database statistics with connection pooling and chunked queries."""
        async with self._pool.get_connection() as conn:
            # Create a statistics collector for pooled operations
            stats_collector = StatisticsCollector(self.persist_directory)
            return await stats_collector.collect_stats(conn.collection)

    async def remove_file_chunks(self, file_path: str) -> int:
        """Remove all chunks for a specific file using pooled connection."""
        try:
            async with self._pool.get_connection() as conn:
                # Get all chunks for this file
                results = conn.collection.get(where={"file_path": file_path})

                if not results["ids"]:
                    return 0

                # Delete the chunks
                conn.collection.delete(ids=results["ids"])

                return len(results["ids"])

        except Exception as e:
            logger.error(f"Failed to remove chunks for file {file_path}: {e}")
            return 0

    async def reset(self) -> None:
        """Reset the database using pooled connection."""
        try:
            async with self._pool.get_connection() as conn:
                conn.client.reset()
                # Reinitialize the pool after reset
                await self._pool.close()
                await self._pool.initialize()
                logger.info("Database reset successfully")
        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
            raise DatabaseError(f"Failed to reset database: {e}") from e

    async def get_all_chunks(self) -> list[CodeChunk]:
        """Get all chunks from the database using pooled connection.

        Returns:
            List of all code chunks with metadata
        """
        try:
            async with self._pool.get_connection() as conn:
                # Get all documents from collection
                results = conn.collection.get(include=["metadatas", "documents"])

                chunks = []
                if results and results.get("ids"):
                    for i, _chunk_id in enumerate(results["ids"]):
                        metadata = results["metadatas"][i]
                        content = results["documents"][i]

                        # Use helper to convert metadata back to CodeChunk
                        chunk = self._metadata_converter.metadata_to_chunk(
                            metadata, content
                        )
                        chunks.append(chunk)

                logger.debug(f"Retrieved {len(chunks)} chunks from database")
                return chunks

        except Exception as e:
            logger.error(f"Failed to get all chunks: {e}")
            raise DatabaseError(f"Failed to get all chunks: {e}") from e

    def get_pool_stats(self) -> dict[str, Any]:
        """Get connection pool statistics."""
        return self._pool.get_stats()

    async def health_check(self) -> bool:
        """Perform a health check on the database and connection pool."""
        try:
            # Check pool health
            pool_healthy = await self._pool.health_check()
            if not pool_healthy:
                return False

            # Try a simple query to verify database integrity
            try:
                async with self._pool.get_connection() as conn:
                    # Test basic operations
                    conn.collection.count()
                    conn.collection.query(
                        query_texts=["test"], n_results=1, include=["metadatas"]
                    )
                return True
            except Exception as e:
                if self._corruption_recovery.is_corruption_error(e):
                    logger.error(f"Index corruption detected: {e}")
                    # Attempt recovery
                    await self._recover_from_corruption()
                    return False
                else:
                    logger.warning(f"Health check failed: {e}")
                    return False
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False

    async def _recover_from_corruption(self) -> None:
        """Recover from index corruption by rebuilding the index."""
        logger.info("Attempting to recover from index corruption...")

        # Close the pool first
        await self._pool.close()

        # Use corruption recovery helper
        await self._corruption_recovery.recover()

        # Reinitialize the pool
        await self._pool.initialize()
        logger.info("Index recovered. Please re-index your codebase.")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
