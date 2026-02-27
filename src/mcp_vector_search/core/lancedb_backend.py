"""LanceDB vector database backend.

LanceDB provides:
- Serverless architecture (no separate server process)
- Built on Apache Arrow for fast columnar operations
- Native support for vector search with ANN indices
- Simple file-based storage with excellent data integrity
- High performance for large-scale operations
"""

import hashlib
import json
import os
import platform
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import orjson
import pyarrow as pa
from loguru import logger

from .context_builder import build_contextual_text
from .exceptions import (
    DatabaseError,
    DatabaseInitializationError,
    DatabaseNotInitializedError,
    DocumentAdditionError,
)
from .models import CodeChunk, IndexStats, SearchResult


# Explicit PyArrow schema for main vector search table
# This ensures consistent schema across all batches, preventing
# "Field not found in target schema" errors when adding new fields
#
# NOTE: The vector dimension is set dynamically based on the embedding model.
# Common dimensions: 384 (MiniLM), 768 (CodeBERT), 1024 (CodeXEmbed)
def _create_lance_schema(vector_dim: int) -> pa.Schema:
    """Create PyArrow schema with dynamic vector dimension.

    Args:
        vector_dim: Embedding vector dimension (e.g., 384, 768, 1024)

    Returns:
        PyArrow schema for LanceDB table
    """
    return pa.schema(
        [
            # Identity
            pa.field("id", pa.string()),
            pa.field("chunk_id", pa.string()),
            # Vector embedding (dimension varies by model)
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
            # Content and metadata
            pa.field("content", pa.string()),
            pa.field("file_path", pa.string()),
            pa.field("start_line", pa.int32()),
            pa.field("end_line", pa.int32()),
            pa.field("language", pa.string()),
            pa.field("chunk_type", pa.string()),
            pa.field("function_name", pa.string()),
            pa.field("class_name", pa.string()),
            pa.field("docstring", pa.string()),
            pa.field("imports", pa.string()),  # JSON-encoded list
            pa.field("calls", pa.string()),  # Comma-separated
            pa.field("inherits_from", pa.string()),  # Comma-separated
            pa.field("complexity_score", pa.float64()),
            pa.field("parent_chunk_id", pa.string()),
            pa.field("child_chunk_ids", pa.string()),  # Comma-separated
            pa.field("chunk_depth", pa.int32()),
            pa.field("decorators", pa.string()),  # Comma-separated
            pa.field("return_type", pa.string()),
            pa.field("subproject_name", pa.string()),
            pa.field("subproject_path", pa.string()),
            # NLP-extracted entities
            pa.field("nlp_keywords", pa.string()),  # Comma-separated
            pa.field("nlp_code_refs", pa.string()),  # Comma-separated
            pa.field("nlp_technical_terms", pa.string()),  # Comma-separated
            # Git blame metadata
            pa.field("last_author", pa.string()),
            pa.field("last_modified", pa.string()),  # ISO timestamp
            pa.field("commit_hash", pa.string()),
            # Quality metrics (added dynamically, nullable)
            pa.field("cognitive_complexity", pa.int32()),
            pa.field("cyclomatic_complexity", pa.int32()),
            pa.field("max_nesting_depth", pa.int32()),
            pa.field("parameter_count", pa.int32()),
            pa.field("lines_of_code", pa.int32()),
            pa.field("complexity_grade", pa.string()),
            pa.field("code_smells", pa.string()),  # JSON-encoded list
            pa.field("smell_count", pa.int32()),
            pa.field("quality_score", pa.int32()),
        ]
    )


# Default schema for 768-dimensional embeddings (GraphCodeBERT default)
LANCEDB_SCHEMA = _create_lance_schema(768)


def _detect_optimal_write_buffer_size() -> int:
    """Detect optimal write buffer size based on available RAM.

    Returns:
        Optimal buffer size for batch writes:
        - 10000 for 64GB+ RAM (M4 Max/Ultra, high-end workstations)
        - 5000 for 32GB RAM (M4 Pro, mid-tier systems)
        - 2000 for 16GB RAM (M4 base, standard systems)
        - 1000 for <16GB RAM or detection failure (safe default)

    Environment Variables:
        MCP_VECTOR_SEARCH_WRITE_BUFFER_SIZE: Override auto-detection
    """
    env_size = os.environ.get("MCP_VECTOR_SEARCH_WRITE_BUFFER_SIZE")
    if env_size:
        return int(env_size)

    try:
        import subprocess

        result = subprocess.run(  # nosec B607 - safe system call for RAM detection
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            total_ram_gb = int(result.stdout.strip()) / (1024**3)

            if total_ram_gb >= 64:
                logger.debug(
                    f"Detected {total_ram_gb:.1f}GB RAM: using write buffer size 10000"
                )
                return 10000  # 64GB+ RAM
            elif total_ram_gb >= 32:
                logger.debug(
                    f"Detected {total_ram_gb:.1f}GB RAM: using write buffer size 5000"
                )
                return 5000  # 32GB RAM
            elif total_ram_gb >= 16:
                logger.debug(
                    f"Detected {total_ram_gb:.1f}GB RAM: using write buffer size 2000"
                )
                return 2000  # 16GB RAM
            else:
                logger.debug(
                    f"Detected {total_ram_gb:.1f}GB RAM: using write buffer size 1000"
                )
                return 1000  # <16GB RAM
    except Exception as e:
        logger.debug(f"RAM detection failed: {e}, using default write buffer size 1000")

    return 1000  # Safe default


class LanceVectorDatabase:
    """LanceDB implementation of vector database.

    Features:
    - Async context manager support (__aenter__, __aexit__)
    - Vector similarity search with metadata filtering
    - Automatic schema inference from first batch
    - File-based persistence with excellent data integrity
    - High performance for large datasets

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
        vector_dim: int | None = None,  # Optional: specify vector dimension
    ) -> None:
        """Initialize LanceDB vector database.

        Args:
            persist_directory: Directory to persist database
            embedding_function: Function to generate embeddings
            collection_name: Name of the table
            vector_dim: Vector dimension (auto-detected if not provided)
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

        # Detect vector dimension from embedding function or use provided value
        if vector_dim is None:
            # Try to get dimension from embedding function
            if hasattr(embedding_function, "dimension"):
                self.vector_dim = embedding_function.dimension
            else:
                # Fallback: detect by generating a test embedding
                try:
                    test_embedding = embedding_function(["test"])[0]
                    self.vector_dim = len(test_embedding)
                    logger.debug(f"Detected vector dimension: {self.vector_dim}")
                except Exception as e:
                    logger.warning(
                        f"Failed to detect vector dimension: {e}, using default 768"
                    )
                    self.vector_dim = 768
        else:
            self.vector_dim = vector_dim

        # Create schema with correct vector dimension
        self._schema = _create_lance_schema(self.vector_dim)

        # LRU cache for search results (same as ChromaDB implementation)
        import os

        cache_size = int(os.environ.get("MCP_VECTOR_SEARCH_CACHE_SIZE", "100"))
        self._search_cache: dict[str, list[SearchResult]] = {}
        self._search_cache_order: list[str] = []
        self._search_cache_max_size = cache_size

        # Write buffer for batching database inserts (2-4x speedup)
        # Auto-detect optimal buffer size based on available RAM
        self._write_buffer: list[dict] = []
        self._write_buffer_size = _detect_optimal_write_buffer_size()

    # ------------------------------------------------------------------
    # Helpers: idempotent table operations
    # ------------------------------------------------------------------

    def _list_table_names(self) -> list[str]:
        """Return table names from LanceDB, handling API variations."""
        tables_response = self._db.list_tables()
        if hasattr(tables_response, "tables"):
            return tables_response.tables
        return tables_response

    def _is_corruption_error(self, error: Exception) -> bool:
        """Check if error indicates corrupted LanceDB data fragments.

        Args:
            error: Exception to check

        Returns:
            True if error indicates corruption (missing data fragments), False otherwise

        Note:
            This checks for ACTUAL corruption (missing data fragment files),
            NOT schema mismatches or other operational errors. Schema mismatches
            should be handled by the caller, not treated as corruption.
        """
        error_msg = str(error).lower()

        # Check for genuine corruption: missing data fragment files
        # These errors mention specific fragment files like "data/abc123.lance"
        # Example: "NotFound: data fragment 'data/abc123.lance' not found"
        is_fragment_error = (
            "not found" in error_msg or "no such file" in error_msg
        ) and ("fragment" in error_msg or "data/" in error_msg)

        # Schema errors are NOT corruption - they indicate wrong collection_name
        is_schema_error = (
            "schema" in error_msg
            or "field" in error_msg
            or "column" in error_msg
            or "type mismatch" in error_msg
        )

        # Only treat as corruption if it's a fragment error AND NOT a schema error
        return is_fragment_error and not is_schema_error

    def _handle_corrupt_table(self, error: Exception, table_name: str) -> bool:
        """Handle corrupted LanceDB table by deleting and resetting.

        Args:
            error: The corruption error
            table_name: Name of the corrupted table

        Returns:
            True if recovery successful, False if unrecoverable
        """
        if not self._is_corruption_error(error):
            return False

        try:
            table_path = self.persist_directory / f"{table_name}.lance"
            if table_path.exists():
                logger.warning(
                    f"Detected corrupted {table_name} table (missing data fragment). "
                    f"Auto-recovering by deleting: {table_path}"
                )
                shutil.rmtree(table_path)
                logger.info(f"Deleted corrupted table: {table_path}")

            # Reset internal table reference
            self._table = None
            return True

        except Exception as e:
            logger.error(f"Failed to recover from corruption: {e}")
            return False

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
            table_names = self._list_table_names()

            if self.collection_name in table_names:
                try:
                    self._table = self._db.open_table(self.collection_name)
                    logger.debug(
                        f"LanceDB table '{self.collection_name}' opened at {self.persist_directory}"
                    )
                except Exception as e:
                    # Check for corruption and auto-recover
                    if self._handle_corrupt_table(e, self.collection_name):
                        self._table = None
                        logger.info(
                            f"Table '{self.collection_name}' corrupted and deleted. Will be recreated on next index."
                        )
                    else:
                        raise
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

    async def _flush_write_buffer(self) -> None:
        """Flush accumulated chunks to database in a single bulk write.

        This method is called automatically when the buffer reaches its size limit,
        or manually when closing the database or during explicit flush operations.
        """
        if not self._write_buffer:
            return

        try:
            # Create or append to table with buffered records
            if self._table is None:
                # Create table with explicit schema (uses instance schema with correct dimension)
                self._table = self._db.create_table(
                    self.collection_name,
                    self._write_buffer,
                    schema=self._schema,
                    exist_ok=True,
                )
                logger.debug(
                    f"Created LanceDB table '{self.collection_name}' with {len(self._write_buffer)} chunks"
                )
            else:
                # Append to existing table
                self._table.add(self._write_buffer)
                logger.debug(
                    f"Flushed {len(self._write_buffer)} chunks to LanceDB table"
                )

            # Invalidate search cache after buffer flush
            self._invalidate_search_cache()

            # Clear buffer after successful flush
            self._write_buffer = []

        except Exception as e:
            logger.error(f"Failed to flush write buffer: {e}")
            # Keep buffer intact on error for retry
            raise

    async def optimize(self) -> None:
        """Optimize table by compacting fragments and cleaning up old versions.

        This should be called periodically (e.g., after batch indexing) to:
        - Merge small data fragments into larger files
        - Remove old transaction files
        - Improve query performance

        Note: This is an expensive operation and should not be called after every add_chunks.

        WORKAROUND: Skipped on macOS to avoid SIGBUS crash caused by memory conflict
        between PyTorch MPS memory-mapped model files and LanceDB compaction operations.
        """
        if platform.system() == "Darwin":
            logger.debug(
                "Skipping LanceDB optimize on macOS to avoid SIGBUS crash "
                "(PyTorch MPS + LanceDB compaction memory conflict)"
            )
            return

        if self._table is None:
            logger.debug("No table to optimize")
            return

        # Linux guard: skip compaction for large tables to prevent arrow offset overflow
        # documented in lance issue #3330. optimize() on tables > 100k rows can trigger
        # an int32 overflow in the offset buffer on Linux.
        try:
            row_count = self._table.count_rows()
            if row_count > 100_000:
                logger.debug(
                    "Skipping compaction: table has %d rows (lance#3330 safety)",
                    row_count,
                )
                return
        except Exception:
            pass

        try:
            from datetime import timedelta

            # Optimize with immediate cleanup of old versions
            # This compacts fragments and removes transaction files
            self._table.optimize(cleanup_older_than=timedelta(seconds=0))
            logger.info("LanceDB table optimized successfully")
        except Exception as e:
            # Check for corruption and auto-recover
            if self._handle_corrupt_table(e, self.collection_name):
                logger.warning(
                    f"Table '{self.collection_name}' corrupted during optimize. Skipped. "
                    "Run 'mcp-vector-search index' to rebuild."
                )
                return

            # Non-fatal - optimization failure doesn't affect data integrity
            logger.warning(f"Failed to optimize LanceDB table: {e}")

    async def close(self) -> None:
        """Close database connections.

        Flushes any remaining buffered writes and optimizes table before closing.
        LanceDB doesn't require explicit closing, but we set references to None
        for consistency with ChromaDB interface.
        """
        # Flush any remaining buffered writes
        await self._flush_write_buffer()

        # Optimize table to compact fragments and cleanup transaction files
        await self.optimize()

        # Save schema version after successful operation
        try:
            from .schema import save_schema_version

            save_schema_version(self.persist_directory)
        except Exception as e:
            # Non-fatal - don't fail close() if schema version save fails
            logger.warning(f"Failed to save schema version: {e}")

        self._table = None
        self._db = None
        logger.debug("LanceDB connections closed")

    async def add_chunks(
        self,
        chunks: list[CodeChunk],
        metrics: dict[str, Any] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Add code chunks to the database with optional structural metrics.

        Args:
            chunks: List of code chunks to add
            metrics: Optional dict mapping chunk IDs to ChunkMetrics.to_metadata() dicts
            embeddings: Optional pre-computed embeddings (if None, will be generated)

        Raises:
            DatabaseNotInitializedError: If database not initialized
            DocumentAdditionError: If adding chunks fails
        """
        if self._db is None:
            raise DatabaseNotInitializedError("Database not initialized")

        if not chunks:
            return

        try:
            # Build context-enriched texts for embedding.
            # build_contextual_text() prepends file path, language, class/function
            # context, imports, and docstring to improve retrieval quality by
            # 35–49%.  The stored chunk.content field is NOT modified — only the
            # text sent to the embedding model is enriched.
            embedding_texts = [build_contextual_text(chunk) for chunk in chunks]
            if embeddings is None:
                # Run embedding generation in thread pool to avoid blocking event loop
                # This allows other async operations to proceed during CPU-intensive embedding
                import asyncio

                try:
                    embeddings = await asyncio.to_thread(
                        self.embedding_function, embedding_texts
                    )
                except BaseException as e:
                    # PyO3 panics inherit from BaseException, not Exception
                    if "Python interpreter is not initialized" in str(e):
                        logger.warning("Embedding interrupted during shutdown")
                        raise RuntimeError(
                            "Embedding interrupted during Python shutdown"
                        ) from e
                    raise

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
                    "imports": (
                        orjson.dumps(chunk.imports).decode() if chunk.imports else "[]"
                    ),
                    "calls": (
                        ",".join(chunk.calls)
                        if hasattr(chunk, "calls") and chunk.calls
                        else ""
                    ),
                    "inherits_from": (
                        ",".join(chunk.inherits_from)
                        if hasattr(chunk, "inherits_from") and chunk.inherits_from
                        else ""
                    ),
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
                    # NLP-extracted entities
                    "nlp_keywords": (
                        ",".join(chunk.nlp_keywords)
                        if hasattr(chunk, "nlp_keywords") and chunk.nlp_keywords
                        else ""
                    ),
                    "nlp_code_refs": (
                        ",".join(chunk.nlp_code_refs)
                        if hasattr(chunk, "nlp_code_refs") and chunk.nlp_code_refs
                        else ""
                    ),
                    "nlp_technical_terms": (
                        ",".join(chunk.nlp_technical_terms)
                        if hasattr(chunk, "nlp_technical_terms")
                        and chunk.nlp_technical_terms
                        else ""
                    ),
                    # Git blame metadata
                    "last_author": chunk.last_author or "",
                    "last_modified": chunk.last_modified or "",
                    "commit_hash": chunk.commit_hash or "",
                }

                # Add structural metrics if provided
                if metrics and chunk.chunk_id in metrics:
                    chunk_metrics = metrics[chunk.chunk_id]
                    metadata.update(chunk_metrics)

                # Ensure all schema fields are present with defaults
                # This prevents "Field not found in target schema" errors
                schema_defaults = {
                    # Quality metrics (nullable in schema)
                    "cognitive_complexity": None,
                    "cyclomatic_complexity": None,
                    "max_nesting_depth": None,
                    "parameter_count": None,
                    "lines_of_code": None,
                    "complexity_grade": "",
                    "code_smells": "[]",
                    "smell_count": 0,
                    "quality_score": 0,
                }

                # Create record with embedding vector
                record = {
                    "id": chunk.chunk_id or chunk.id,
                    "vector": embedding,
                    "content": chunk.content,
                    **schema_defaults,  # Add defaults first
                    **metadata,  # Override with actual values
                }
                records.append(record)

            # Add to write buffer instead of immediate insertion
            self._write_buffer.extend(records)

            # Always flush to prevent transaction accumulation
            # The buffer is for batching within a single add_chunks call, not across calls
            await self._flush_write_buffer()

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

            # Build LanceDB query with cosine metric.
            # nprobes and refine_factor enable two-stage ANN retrieval:
            #   - nprobes=20: scan 20 IVF partitions (higher = better recall, slower)
            #   - refine_factor=5: re-rank 5x candidates with exact distances
            # LanceDB silently ignores these when no ANN index exists (brute-force).
            search = (
                self._table.search(query_embedding)
                .metric("cosine")
                .nprobes(20)
                .refine_factor(5)
                .limit(limit)
            )

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
                # LanceDB returns _distance (cosine distance, 0-2 range)
                # Convert to similarity: 0 distance -> 1.0 similarity, 2 distance -> 0.0
                distance = result.get("_distance", 0.0)
                similarity = max(0.0, 1.0 - (distance / 2.0))

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
                    # Git blame metadata
                    last_author=result.get("last_author") or None,
                    last_modified=result.get("last_modified") or None,
                    commit_hash=result.get("commit_hash") or None,
                )
                search_results.append(search_result)

            # Cache results
            self._add_to_search_cache(cache_key, search_results)

            logger.debug(
                f"LanceDB search returned {len(search_results)} results for query: {query[:50]}..."
            )
            return search_results

        except Exception as e:
            # Check for corruption and auto-recover
            if self._handle_corrupt_table(e, self.collection_name):
                logger.error(
                    f"Table '{self.collection_name}' corrupted. Search unavailable. "
                    "Run 'mcp-vector-search index' to rebuild."
                )
                raise DatabaseError(
                    "Index corrupted. Run 'mcp-vector-search index' to rebuild."
                ) from e

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
            # Handle LanceDB "Not found" errors gracefully (file not in index)
            error_msg = str(e).lower()
            if "not found" in error_msg:
                logger.debug(f"No chunks to delete for {file_path} (not in index)")
                return 0
            logger.error(f"Failed to delete chunks for {file_path}: {e}")
            raise DatabaseError(f"Failed to delete chunks: {e}") from e

    async def get_stats(self, skip_stats: bool = False) -> IndexStats:
        """Get database statistics.

        Args:
            skip_stats: If True, skip detailed statistics collection

        Returns:
            Index statistics including total chunks and indexed files
        """
        # If table is not open, try to open it
        if self._table is None:
            if self._db is None:
                # Database not initialized at all
                return IndexStats(
                    total_files=0,
                    total_chunks=0,
                    languages={},
                    file_types={},
                    index_size_mb=0.0,
                    last_updated="N/A",
                    embedding_model="unknown",
                    database_size_bytes=0,
                )

            # Try to open the table if it exists
            try:
                tables_response = self._db.list_tables()
                table_names = (
                    tables_response.tables
                    if hasattr(tables_response, "tables")
                    else tables_response
                )

                if self.collection_name in table_names:
                    self._table = self._db.open_table(self.collection_name)
                    logger.debug(f"Opened table '{self.collection_name}' for stats")
                else:
                    # Table doesn't exist, return zeros
                    return IndexStats(
                        total_files=0,
                        total_chunks=0,
                        languages={},
                        file_types={},
                        index_size_mb=0.0,
                        last_updated="N/A",
                        embedding_model="unknown",
                        database_size_bytes=0,
                    )
            except Exception as e:
                logger.warning(f"Failed to open table for stats: {e}")
                return IndexStats(
                    total_files=0,
                    total_chunks=0,
                    languages={},
                    file_types={},
                    index_size_mb=0.0,
                    last_updated="N/A",
                    embedding_model="unknown",
                    database_size_bytes=0,
                )

        try:
            total_chunks = self._table.count_rows()

            if skip_stats or total_chunks == 0:
                # Calculate database size even for empty DB
                db_size_bytes = self._get_database_size()
                db_size_mb = db_size_bytes / (1024 * 1024)

                return IndexStats(
                    total_files=0,
                    total_chunks=total_chunks,
                    languages={},
                    file_types={},
                    index_size_mb=db_size_mb,
                    last_updated="N/A" if total_chunks == 0 else "unknown",
                    embedding_model="unknown",
                    database_size_bytes=db_size_bytes,
                )

            # Get detailed statistics using pandas
            df = self._table.to_pandas()

            # Count unique files
            total_files = df["file_path"].nunique()

            # Language distribution
            language_counts = df["language"].value_counts().to_dict()

            # File type distribution (extract extensions)
            file_types: dict[str, int] = {}
            for file_path in df["file_path"].unique():
                ext = Path(file_path).suffix or "no_extension"
                file_types[ext] = file_types.get(ext, 0) + 1

            # Calculate storage size
            db_size_bytes = self._get_database_size()
            index_size_mb = db_size_bytes / (1024 * 1024)

            return IndexStats(
                total_files=total_files,
                total_chunks=total_chunks,
                languages=language_counts,
                file_types=file_types,
                index_size_mb=index_size_mb,
                last_updated="unknown",  # LanceDB doesn't track modification time
                embedding_model="unknown",  # Would need to be passed in or stored
                database_size_bytes=db_size_bytes,
            )

        except Exception as e:
            # Check for corruption and auto-recover
            if self._handle_corrupt_table(e, self.collection_name):
                logger.warning(
                    f"Table '{self.collection_name}' corrupted. Stats unavailable. "
                    "Run 'mcp-vector-search index' to rebuild."
                )
                # Return minimal stats indicating corruption
                return IndexStats(
                    total_files=0,
                    total_chunks=0,
                    languages={},
                    file_types={},
                    index_size_mb=0.0,
                    last_updated="corrupted",
                    embedding_model="unknown",
                    database_size_bytes=0,
                )

            logger.error(f"Failed to get LanceDB stats: {e}")
            # Return minimal stats on error
            return IndexStats(
                total_files=0,
                total_chunks=0,
                languages={},
                file_types={},
                index_size_mb=0.0,
                last_updated="error",
                embedding_model="unknown",
                database_size_bytes=0,
            )

    def _get_database_size(self) -> int:
        """Get total database directory size in bytes.

        Returns:
            Total size of all files in database directory (bytes)
        """
        total_size = 0
        try:
            # LanceDB stores data in multiple files in the persist directory
            for file_path in self.persist_directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.warning(f"Failed to calculate database size: {e}")
            return 0
        return total_size

    async def reset(self) -> None:
        """Reset the database (delete all data).

        Drops the table and recreates it empty.
        """
        if self._db is None:
            raise DatabaseNotInitializedError("Database not initialized")

        try:
            # Clear write buffer (discard unflushed data)
            self._write_buffer = []

            # Drop table if exists
            if self.collection_name in self._db.list_tables():
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

    def iter_chunks_batched(
        self,
        batch_size: int = 10000,
        file_path: str | None = None,
        language: str | None = None,
    ) -> Any:  # Returns Iterator[List[CodeChunk]]
        """Stream chunks from database in batches to avoid memory explosion.

        This method provides two strategies:
        1. **Optimal (with pylance)**: Uses to_lance() + scanner for true streaming
        2. **Fallback (without pylance)**: Uses chunked Pandas iteration

        The method automatically falls back to Pandas if pylance is not installed.

        Args:
            batch_size: Number of chunks per batch (default 10000)
            file_path: Optional filter by file path
            language: Optional filter by language

        Yields:
            List of CodeChunk objects per batch

        Example:
            >>> db = LanceVectorDatabase("/path/to/db")
            >>> total = 0
            >>> for batch in db.iter_chunks_batched(batch_size=1000):
            ...     total += len(batch)
            ...     print(f"Processed {total} chunks")
        """
        if self._table is None:
            return

        # Try optimal strategy first (requires pylance)
        try:
            yield from self._iter_chunks_lance_scanner(batch_size, file_path, language)
            return
        except Exception as e:
            # Check if error is due to missing pylance
            error_msg = str(e).lower()
            if "pylance" in error_msg or "lance library" in error_msg:
                logger.debug(
                    "pylance not installed, falling back to chunked Pandas iteration"
                )
            else:
                # Other error - log but try fallback anyway
                logger.warning(f"Lance scanner failed: {e}, trying fallback")

        # Fallback strategy (works without pylance)
        yield from self._iter_chunks_pandas_chunked(batch_size, file_path, language)

    def _iter_chunks_lance_scanner(
        self,
        batch_size: int,
        file_path: str | None,
        language: str | None,
    ) -> Iterator[list[CodeChunk]]:
        """Optimal batch iteration using Lance scanner (requires pylance)."""
        # Build filter expression
        filter_expr = None
        if file_path:
            filter_expr = f"file_path = '{file_path}'"
        if language:
            lang_filter = f"language = '{language}'"
            filter_expr = (
                f"{filter_expr} AND {lang_filter}" if filter_expr else lang_filter
            )

        # Get Lance dataset (requires pylance)
        lance_dataset = self._table.to_lance()

        # Create scanner with batch iteration
        scanner = lance_dataset.scanner(
            filter=filter_expr,
            batch_size=batch_size,
        )

        # Iterate over Arrow batches
        for batch in scanner.to_reader():
            chunks = []
            batch_dict = batch.to_pydict()
            num_rows = len(batch_dict["content"])

            for i in range(num_rows):
                chunk = self._batch_dict_to_chunk(batch_dict, i, num_rows)
                chunks.append(chunk)

            yield chunks

    def _iter_chunks_pandas_chunked(
        self,
        batch_size: int,
        file_path: str | None,
        language: str | None,
    ) -> Iterator[list[CodeChunk]]:
        """Fallback batch iteration using chunked Pandas DataFrames."""
        # Load table to Pandas (this is the memory-intensive step)
        df = self._table.to_pandas()

        # Apply filters if provided
        if file_path:
            df = df[df["file_path"] == file_path]
        if language:
            df = df[df["language"] == language]

        # Iterate in chunks
        total_rows = len(df)
        offset = 0

        while offset < total_rows:
            # Get batch slice
            batch_df = df.iloc[offset : offset + batch_size]

            chunks = []
            for _, row in batch_df.iterrows():
                chunk = self._row_to_chunk(row)
                chunks.append(chunk)

            yield chunks
            offset += batch_size

    def _batch_dict_to_chunk(
        self, batch_dict: dict, i: int, num_rows: int
    ) -> CodeChunk:
        """Convert Arrow batch dictionary row to CodeChunk."""
        # Handle imports field (list/numpy array of JSON strings or legacy comma-separated string)
        imports_raw = batch_dict.get("imports", [[]] * num_rows)[i]

        if isinstance(imports_raw, (list, np.ndarray)):
            # New format: list or numpy array of JSON strings
            imports = []
            for item in imports_raw:
                try:
                    imports.append(json.loads(item))
                except (json.JSONDecodeError, TypeError):
                    imports.append(item)  # Keep as string if not valid JSON
        elif isinstance(imports_raw, str):
            # Legacy format: comma-separated string
            imports = imports_raw.split(",") if imports_raw else []
        else:
            # Unknown type - default to empty
            imports = []

        # Handle child_chunk_ids field (list/numpy array or legacy comma-separated string)
        child_ids_raw = batch_dict.get("child_chunk_ids", [[]] * num_rows)[i]
        if isinstance(child_ids_raw, (list, np.ndarray)):
            child_chunk_ids = list(child_ids_raw)
        elif isinstance(child_ids_raw, str):
            child_chunk_ids = child_ids_raw.split(",") if child_ids_raw else []
        else:
            child_chunk_ids = []

        # Handle decorators field (list/numpy array or legacy comma-separated string)
        decorators_raw = batch_dict.get("decorators", [[]] * num_rows)[i]
        if isinstance(decorators_raw, (list, np.ndarray)):
            decorators = list(decorators_raw)
        elif isinstance(decorators_raw, str):
            decorators = decorators_raw.split(",") if decorators_raw else []
        else:
            decorators = []

        # Handle calls field (list/numpy array or legacy comma-separated string)
        calls_raw = batch_dict.get("calls", [[]] * num_rows)[i]
        if isinstance(calls_raw, (list, np.ndarray)):
            calls = list(calls_raw)
        elif isinstance(calls_raw, str):
            calls = calls_raw.split(",") if calls_raw else []
        else:
            calls = []

        # Handle inherits_from field (list/numpy array or legacy comma-separated string)
        inherits_raw = batch_dict.get("inherits_from", [[]] * num_rows)[i]
        if isinstance(inherits_raw, (list, np.ndarray)):
            inherits_from = list(inherits_raw)
        elif isinstance(inherits_raw, str):
            inherits_from = inherits_raw.split(",") if inherits_raw else []
        else:
            inherits_from = []

        return CodeChunk(
            content=batch_dict["content"][i],
            file_path=Path(batch_dict["file_path"][i]),
            start_line=batch_dict["start_line"][i],
            end_line=batch_dict["end_line"][i],
            language=batch_dict["language"][i],
            chunk_type=batch_dict.get("chunk_type", ["code"] * num_rows)[i],
            function_name=batch_dict.get("function_name", [None] * num_rows)[i] or None,
            class_name=batch_dict.get("class_name", [None] * num_rows)[i] or None,
            docstring=batch_dict.get("docstring", [None] * num_rows)[i] or None,
            imports=imports,
            calls=calls,
            inherits_from=inherits_from,
            complexity_score=float(batch_dict.get("complexity", [0] * num_rows)[i]),
            chunk_id=batch_dict.get("chunk_id", [None] * num_rows)[i],
            parent_chunk_id=batch_dict.get("parent_chunk_id", [None] * num_rows)[i]
            or None,
            child_chunk_ids=child_chunk_ids,
            chunk_depth=batch_dict.get("chunk_depth", [0] * num_rows)[i],
            decorators=decorators,
            return_type=batch_dict.get("return_type", [None] * num_rows)[i] or None,
            subproject_name=batch_dict.get("subproject_name", [None] * num_rows)[i]
            or None,
            subproject_path=batch_dict.get("subproject_path", [None] * num_rows)[i]
            or None,
        )

    def _row_to_chunk(self, row: Any) -> CodeChunk:
        """Convert Pandas DataFrame row to CodeChunk."""
        # Handle imports field (list/numpy array of JSON strings or legacy comma-separated string)
        imports_raw = row.get("imports", [])
        if isinstance(imports_raw, (list, np.ndarray)):
            # New format: list or numpy array of JSON strings
            imports = []
            for item in imports_raw:
                try:
                    imports.append(json.loads(item))
                except (json.JSONDecodeError, TypeError):
                    imports.append(item)  # Keep as string if not valid JSON
        elif isinstance(imports_raw, str):
            # Legacy format: comma-separated string
            imports = imports_raw.split(",") if imports_raw else []
        else:
            # Unknown type - default to empty
            imports = []

        # Handle child_chunk_ids field (list/numpy array or legacy comma-separated string)
        child_ids_raw = row.get("child_chunk_ids", [])
        if isinstance(child_ids_raw, (list, np.ndarray)):
            child_chunk_ids = list(child_ids_raw)
        elif isinstance(child_ids_raw, str):
            child_chunk_ids = child_ids_raw.split(",") if child_ids_raw else []
        else:
            child_chunk_ids = []

        # Handle decorators field (list/numpy array or legacy comma-separated string)
        decorators_raw = row.get("decorators", [])
        if isinstance(decorators_raw, (list, np.ndarray)):
            decorators = list(decorators_raw)
        elif isinstance(decorators_raw, str):
            decorators = decorators_raw.split(",") if decorators_raw else []
        else:
            decorators = []

        # Handle calls field (list/numpy array or legacy comma-separated string)
        calls_raw = row.get("calls", [])
        if isinstance(calls_raw, (list, np.ndarray)):
            calls = list(calls_raw)
        elif isinstance(calls_raw, str):
            calls = calls_raw.split(",") if calls_raw else []
        else:
            calls = []

        # Handle inherits_from field (list/numpy array or legacy comma-separated string)
        inherits_raw = row.get("inherits_from", [])
        if isinstance(inherits_raw, (list, np.ndarray)):
            inherits_from = list(inherits_raw)
        elif isinstance(inherits_raw, str):
            inherits_from = inherits_raw.split(",") if inherits_raw else []
        else:
            inherits_from = []

        return CodeChunk(
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
            calls=calls,
            inherits_from=inherits_from,
            complexity_score=float(row.get("complexity", 0)),
            chunk_id=row.get("chunk_id"),
            parent_chunk_id=row.get("parent_chunk_id") or None,
            child_chunk_ids=child_chunk_ids,
            chunk_depth=row.get("chunk_depth", 0),
            decorators=decorators,
            return_type=row.get("return_type") or None,
            subproject_name=row.get("subproject_name") or None,
            subproject_path=row.get("subproject_path") or None,
        )

    def get_chunk_count(
        self, file_path: str | None = None, language: str | None = None
    ) -> int:
        """Get total chunk count without loading all data.

        Args:
            file_path: Optional filter by file path
            language: Optional filter by language

        Returns:
            Total number of chunks matching the filter criteria
        """
        if self._table is None:
            return 0

        try:
            # If no filters, use count_rows() for efficiency
            if not file_path and not language:
                return self._table.count_rows()

            # With filters, build filter expression and count via scanner
            filter_expr = None
            if file_path:
                filter_expr = f"file_path = '{file_path}'"
            if language:
                lang_filter = f"language = '{language}'"
                filter_expr = (
                    f"{filter_expr} AND {lang_filter}" if filter_expr else lang_filter
                )

            # Use scanner to get filtered data and count length
            # Note: count_rows() on filtered scanner is not supported in newer pylance
            try:
                lance_dataset = self._table.to_lance()
                scanner = lance_dataset.scanner(
                    filter=filter_expr,
                    columns=[],  # Empty column list for counting only
                )
                result = scanner.to_table()
                return len(result)
            except Exception as e:
                # pylance not available or scanner fails, fall back to Pandas
                error_msg = str(e).lower()
                if "pylance" not in error_msg and "lance library" not in error_msg:
                    # Not a pylance error - re-raise
                    raise

            # Fallback: Load and filter with Pandas
            df = self._table.to_pandas()
            if file_path:
                df = df[df["file_path"] == file_path]
            if language:
                df = df[df["language"] == language]
            return len(df)

        except Exception as e:
            logger.error(f"Failed to get chunk count: {e}")
            return 0

    async def get_all_chunks(self) -> list[CodeChunk]:
        """Get all chunks from the database.

        WARNING: This loads the entire table into memory. For large databases
        (576K+ chunks), use iter_chunks_batched() instead to avoid OOM.

        Returns:
            List of all code chunks with metadata
        """
        if self._table is None:
            return []

        try:
            # Use streaming iterator to collect all chunks
            # This is more memory-efficient than to_pandas()
            chunks = []
            for batch in self.iter_chunks_batched(batch_size=10000):
                chunks.extend(batch)

            logger.debug(f"Retrieved {len(chunks)} chunks from LanceDB")
            return chunks

        except Exception as e:
            logger.error(f"Failed to get all chunks from LanceDB: {e}")
            raise DatabaseError(f"Failed to get all chunks: {e}") from e

    async def health_check(self) -> bool:
        """Check database health and integrity.

        Auto-initializes the database if not already initialized.

        Returns:
            True if database is healthy, False otherwise
        """
        try:
            # Auto-initialize if not already initialized
            if not self._db:
                logger.debug("Database not initialized, initializing now")
                await self.initialize()

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
