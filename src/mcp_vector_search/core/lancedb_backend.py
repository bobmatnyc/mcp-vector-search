"""LanceDB vector database backend - simpler, more stable alternative to ChromaDB.

LanceDB provides:
- Serverless architecture (no separate server process)
- Built on Apache Arrow for fast columnar operations
- Native support for vector search with ANN indices
- Simple file-based storage (no complex HNSW corruption issues)
- Better performance for large-scale operations
"""

import hashlib
from pathlib import Path
from typing import Any

import lancedb
import orjson
from loguru import logger

from .exceptions import (
    DatabaseError,
    DatabaseInitializationError,
    DatabaseNotInitializedError,
    DocumentAdditionError,
)
from .models import CodeChunk, IndexStats, SearchResult


class LanceVectorDatabase:
    """LanceDB implementation of vector database.

    This class provides a drop-in replacement for ChromaVectorDatabase
    with the same async interface and methods.

    Features:
    - Async context manager support (__aenter__, __aexit__)
    - Vector similarity search with metadata filtering
    - Automatic schema inference from first batch
    - File-based persistence (no corruption issues like ChromaDB)
    - Simple migration path from ChromaDB

    Example:
        async with LanceVectorDatabase(persist_directory, embedding_function) as db:
            await db.add_chunks(chunks)
            results = await db.search("query", limit=10)
    """

    def __init__(
        self,
        persist_directory: Path,
        embedding_function: Any,  # EmbeddingFunction protocol
        collection_name: str = "code_search",
    ) -> None:
        """Initialize LanceDB vector database.

        Args:
            persist_directory: Directory to persist database
            embedding_function: Function to generate embeddings
            collection_name: Name of the table (equivalent to ChromaDB collection)
        """
        self.persist_directory = (
            Path(persist_directory)
            if isinstance(persist_directory, str)
            else persist_directory
        )
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self._db = None
        self._table = None

        # LRU cache for search results (same as ChromaDB implementation)
        import os

        cache_size = int(os.environ.get("MCP_VECTOR_SEARCH_CACHE_SIZE", "100"))
        self._search_cache: dict[str, list[SearchResult]] = {}
        self._search_cache_order: list[str] = []
        self._search_cache_max_size = cache_size

    async def initialize(self) -> None:
        """Initialize LanceDB database and table.

        Creates directory if needed and opens/creates the table.
        LanceDB uses lazy initialization - table is created on first add_chunks.
        """
        try:
            # Ensure directory exists
            self.persist_directory.mkdir(parents=True, exist_ok=True)

            # Connect to LanceDB (creates if doesn't exist)
            self._db = lancedb.connect(str(self.persist_directory))

            # Check if table exists, open if it does
            if self.collection_name in self._db.table_names():
                self._table = self._db.open_table(self.collection_name)
                logger.debug(
                    f"LanceDB table '{self.collection_name}' opened at {self.persist_directory}"
                )
            else:
                # Table will be created on first add_chunks
                self._table = None
                logger.debug(
                    f"LanceDB table '{self.collection_name}' will be created on first add"
                )

        except Exception as e:
            logger.error(f"Failed to initialize LanceDB: {e}")
            raise DatabaseInitializationError(
                f"LanceDB initialization failed: {e}"
            ) from e

    async def close(self) -> None:
        """Close database connections.

        LanceDB doesn't require explicit closing, but we set references to None
        for consistency with ChromaDB interface.
        """
        self._table = None
        self._db = None
        logger.debug("LanceDB connections closed")

    async def add_chunks(
        self, chunks: list[CodeChunk], metrics: dict[str, Any] | None = None
    ) -> None:
        """Add code chunks to the database with optional structural metrics.

        Args:
            chunks: List of code chunks to add
            metrics: Optional dict mapping chunk IDs to ChunkMetrics.to_metadata() dicts

        Raises:
            DatabaseNotInitializedError: If database not initialized
            DocumentAdditionError: If adding chunks fails
        """
        if self._db is None:
            raise DatabaseNotInitializedError("Database not initialized")

        if not chunks:
            return

        try:
            # Generate embeddings for all chunks
            contents = [chunk.content for chunk in chunks]
            embeddings = self.embedding_function(contents)

            # Convert chunks to LanceDB records
            records = []
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                # Build metadata dict (same as ChromaDB)
                metadata = {
                    "file_path": str(chunk.file_path),
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "language": chunk.language,
                    "chunk_type": chunk.chunk_type,
                    "function_name": chunk.function_name or "",
                    "class_name": chunk.class_name or "",
                    "docstring": chunk.docstring or "",
                    "imports": ",".join(chunk.imports) if chunk.imports else "",
                    "complexity_score": chunk.complexity_score,
                    "chunk_id": chunk.chunk_id or chunk.id,
                    "parent_chunk_id": chunk.parent_chunk_id or "",
                    "child_chunk_ids": (
                        ",".join(chunk.child_chunk_ids) if chunk.child_chunk_ids else ""
                    ),
                    "chunk_depth": chunk.chunk_depth,
                    "decorators": (
                        ",".join(chunk.decorators) if chunk.decorators else ""
                    ),
                    "return_type": chunk.return_type or "",
                    "subproject_name": chunk.subproject_name or "",
                    "subproject_path": chunk.subproject_path or "",
                }

                # Add structural metrics if provided
                if metrics and chunk.chunk_id in metrics:
                    chunk_metrics = metrics[chunk.chunk_id]
                    metadata.update(chunk_metrics)

                # Create record with embedding vector
                record = {
                    "id": chunk.chunk_id or chunk.id,
                    "vector": embedding,
                    "content": chunk.content,
                    **metadata,
                }
                records.append(record)

            # Create or append to table
            if self._table is None:
                # Create table with first batch
                self._table = self._db.create_table(self.collection_name, records)
                logger.debug(
                    f"Created LanceDB table '{self.collection_name}' with {len(records)} chunks"
                )
            else:
                # Append to existing table
                self._table.add(records)
                logger.debug(f"Added {len(records)} chunks to LanceDB table")

            # Invalidate search cache
            self._invalidate_search_cache()

        except Exception as e:
            logger.error(f"Failed to add chunks to LanceDB: {e}")
            raise DocumentAdditionError(f"Failed to add chunks: {e}") from e

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        similarity_threshold: float = 0.7,
    ) -> list[SearchResult]:
        """Search for similar code chunks with LRU caching.

        Args:
            query: Search query string
            limit: Maximum number of results
            filters: Optional metadata filters (file_path, language, chunk_type, etc.)
            similarity_threshold: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of search results sorted by similarity

        Raises:
            DatabaseNotInitializedError: If database not initialized
        """
        if self._table is None:
            # Empty database - return empty results
            return []

        # Check cache
        cache_key = self._generate_search_cache_key(
            query, limit, filters, similarity_threshold
        )
        if cache_key in self._search_cache:
            # LRU update
            self._search_cache_order.remove(cache_key)
            self._search_cache_order.append(cache_key)
            logger.debug(f"Search cache hit for query: {query[:50]}...")
            return self._search_cache[cache_key]

        try:
            # Generate query embedding
            query_embedding = self.embedding_function([query])[0]

            # Build LanceDB query
            search = self._table.search(query_embedding).limit(limit)

            # Apply metadata filters if provided
            if filters:
                filter_clauses = []
                for key, value in filters.items():
                    if value is not None:
                        # Handle different filter types
                        if isinstance(value, str):
                            filter_clauses.append(f"{key} = '{value}'")
                        elif isinstance(value, list):
                            # Support IN queries
                            values_str = ", ".join(f"'{v}'" for v in value)
                            filter_clauses.append(f"{key} IN ({values_str})")
                        else:
                            filter_clauses.append(f"{key} = {value}")

                if filter_clauses:
                    where_clause = " AND ".join(filter_clauses)
                    search = search.where(where_clause)

            # Execute search
            results = search.to_list()

            # Convert to SearchResult format
            search_results = []
            for rank, result in enumerate(results):
                # LanceDB returns _distance (L2 distance)
                # Convert to similarity score (cosine similarity approximation)
                # For unit vectors: similarity ≈ 1 - (distance² / 2)
                distance = result.get("_distance", 0.0)
                similarity = 1.0 - (distance * distance / 2.0)

                # Filter by similarity threshold
                if similarity < similarity_threshold:
                    continue

                # Create SearchResult
                search_result = SearchResult(
                    content=result["content"],
                    file_path=Path(result["file_path"]),
                    start_line=result["start_line"],
                    end_line=result["end_line"],
                    language=result["language"],
                    similarity_score=similarity,
                    rank=rank + 1,
                    chunk_type=result.get("chunk_type", "code"),
                    function_name=result.get("function_name") or None,
                    class_name=result.get("class_name") or None,
                )
                search_results.append(search_result)

            # Cache results
            self._add_to_search_cache(cache_key, search_results)

            logger.debug(
                f"LanceDB search returned {len(search_results)} results for query: {query[:50]}..."
            )
            return search_results

        except Exception as e:
            logger.error(f"LanceDB search failed: {e}")
            raise DatabaseError(f"Search failed: {e}") from e

    async def delete_by_file(self, file_path: Path) -> int:
        """Delete all chunks for a specific file.

        Args:
            file_path: Path to the file

        Returns:
            Number of chunks deleted

        Raises:
            DatabaseNotInitializedError: If database not initialized
        """
        if self._table is None:
            return 0

        try:
            # Count chunks before deletion
            file_path_str = str(file_path)
            count_df = (
                self._table.to_pandas()
                .query(f"file_path == '{file_path_str}'")
                .shape[0]
            )

            if count_df == 0:
                return 0

            # Delete matching rows
            self._table.delete(f"file_path = '{file_path_str}'")

            # Invalidate search cache
            self._invalidate_search_cache()

            logger.debug(f"Deleted {count_df} chunks for file: {file_path}")
            return count_df

        except Exception as e:
            logger.error(f"Failed to delete chunks for {file_path}: {e}")
            raise DatabaseError(f"Failed to delete chunks: {e}") from e

    async def get_stats(self, skip_stats: bool = False) -> IndexStats:
        """Get database statistics.

        Args:
            skip_stats: If True, skip detailed statistics collection

        Returns:
            Index statistics including total chunks and indexed files
        """
        if self._table is None:
            return IndexStats(
                total_chunks=0,
                indexed_files=0,
                total_size_bytes=0,
                languages={},
                chunk_types={},
                backend="lancedb",
            )

        try:
            total_chunks = self._table.count_rows()

            if skip_stats or total_chunks == 0:
                return IndexStats(
                    total_chunks=total_chunks,
                    indexed_files=0,
                    total_size_bytes=0,
                    languages={},
                    chunk_types={},
                    backend="lancedb",
                )

            # Get detailed statistics
            df = self._table.to_pandas()

            # Count unique files
            indexed_files = df["file_path"].nunique()

            # Language distribution
            language_counts = df["language"].value_counts().to_dict()

            # Chunk type distribution
            chunk_type_counts = df["chunk_type"].value_counts().to_dict()

            # Calculate storage size (approximate)
            total_size_bytes = len(df) * df.memory_usage(deep=True).sum()

            return IndexStats(
                total_chunks=total_chunks,
                indexed_files=indexed_files,
                total_size_bytes=total_size_bytes,
                languages=language_counts,
                chunk_types=chunk_type_counts,
                backend="lancedb",
            )

        except Exception as e:
            logger.error(f"Failed to get LanceDB stats: {e}")
            # Return minimal stats on error
            return IndexStats(
                total_chunks=0,
                indexed_files=0,
                total_size_bytes=0,
                languages={},
                chunk_types={},
                backend="lancedb",
            )

    async def reset(self) -> None:
        """Reset the database (delete all data).

        Drops the table and recreates it empty.
        """
        if self._db is None:
            raise DatabaseNotInitializedError("Database not initialized")

        try:
            # Drop table if exists
            if self.collection_name in self._db.table_names():
                self._db.drop_table(self.collection_name)
                logger.info(f"Dropped LanceDB table '{self.collection_name}'")

            # Set table to None (will be recreated on next add_chunks)
            self._table = None

            # Clear cache
            self._invalidate_search_cache()

            logger.info("LanceDB database reset successfully")

        except Exception as e:
            logger.error(f"Failed to reset LanceDB: {e}")
            raise DatabaseError(f"Failed to reset database: {e}") from e

    async def get_all_chunks(self) -> list[CodeChunk]:
        """Get all chunks from the database.

        Returns:
            List of all code chunks with metadata
        """
        if self._table is None:
            return []

        try:
            df = self._table.to_pandas()

            chunks = []
            for _, row in df.iterrows():
                # Parse list fields (stored as comma-separated strings)
                imports = row["imports"].split(",") if row["imports"] else []
                child_chunk_ids = (
                    row["child_chunk_ids"].split(",") if row["child_chunk_ids"] else []
                )
                decorators = row["decorators"].split(",") if row["decorators"] else []

                chunk = CodeChunk(
                    content=row["content"],
                    file_path=Path(row["file_path"]),
                    start_line=row["start_line"],
                    end_line=row["end_line"],
                    language=row["language"],
                    chunk_type=row.get("chunk_type", "code"),
                    function_name=row.get("function_name") or None,
                    class_name=row.get("class_name") or None,
                    docstring=row.get("docstring") or None,
                    imports=imports,
                    complexity_score=row.get("complexity_score", 0.0),
                    chunk_id=row.get("chunk_id"),
                    parent_chunk_id=row.get("parent_chunk_id") or None,
                    child_chunk_ids=child_chunk_ids,
                    chunk_depth=row.get("chunk_depth", 0),
                    decorators=decorators,
                    return_type=row.get("return_type") or None,
                    subproject_name=row.get("subproject_name") or None,
                    subproject_path=row.get("subproject_path") or None,
                )
                chunks.append(chunk)

            logger.debug(f"Retrieved {len(chunks)} chunks from LanceDB")
            return chunks

        except Exception as e:
            logger.error(f"Failed to get all chunks from LanceDB: {e}")
            raise DatabaseError(f"Failed to get all chunks: {e}") from e

    async def health_check(self) -> bool:
        """Check database health and integrity.

        Returns:
            True if database is healthy, False otherwise
        """
        try:
            # Check if database is initialized
            if not self._db:
                logger.warning("Database not initialized")
                return False

            # If table doesn't exist yet, that's OK (not an error)
            if self._table is None:
                logger.debug("Table not created yet (health check passed)")
                return True

            # Try a simple operation
            count = self._table.count_rows()
            logger.debug(f"Health check passed: {count} chunks in database")
            return True

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def _generate_search_cache_key(
        self,
        query: str,
        limit: int,
        filters: dict[str, Any] | None,
        similarity_threshold: float,
    ) -> str:
        """Generate cache key for search parameters.

        Args:
            query: Search query
            limit: Result limit
            filters: Search filters
            similarity_threshold: Similarity threshold

        Returns:
            Cache key string (16-char hash)
        """
        params = {
            "query": query,
            "limit": limit,
            "filters": filters or {},
            "threshold": similarity_threshold,
        }
        params_bytes = orjson.dumps(params, option=orjson.OPT_SORT_KEYS)
        return hashlib.sha256(params_bytes).hexdigest()[:16]

    def _add_to_search_cache(self, cache_key: str, results: list[SearchResult]) -> None:
        """Add search results to cache with LRU eviction.

        Args:
            cache_key: Cache key
            results: Search results to cache
        """
        # LRU eviction if cache is full
        if len(self._search_cache) >= self._search_cache_max_size:
            lru_key = self._search_cache_order.pop(0)
            del self._search_cache[lru_key]

        # Add to cache
        self._search_cache[cache_key] = results
        self._search_cache_order.append(cache_key)

    def _invalidate_search_cache(self) -> None:
        """Invalidate search cache when database is modified."""
        self._search_cache.clear()
        self._search_cache_order.clear()
        logger.debug("Search cache invalidated")

    async def __aenter__(self) -> "LanceVectorDatabase":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
