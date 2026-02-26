"""Vectors backend for Phase 2 storage (post-embedding).

This module manages the vectors.lance table which stores embedded vectors
for semantic search. Enables:
- Vector similarity search
- Resumable embedding (picks up from chunks.lance)
- Search without JOINs (denormalized data)
- Efficient vector indexing with IVF_SQ
"""

import platform
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
from loguru import logger

from .exceptions import (
    DatabaseError,
    DatabaseInitializationError,
    DatabaseNotInitializedError,
    SearchError,
)

# Maximum number of file paths per SQL IN clause.
# DataFusion's recursive descent parser stack-overflows on thousands of OR clauses,
# causing SEGV (signal 139) on Linux. Bounded IN batches keep the parse tree shallow.
DELETE_BATCH_LIMIT = 500


# PyArrow schema for vectors table (dynamic dimension)
def _create_vectors_schema(vector_dim: int) -> pa.Schema:
    """Create vectors schema with dynamic vector dimension.

    Args:
        vector_dim: Embedding vector dimension (e.g., 384 for MiniLM, 768 for CodeBERT)

    Returns:
        PyArrow schema for vectors table
    """
    return pa.schema(
        [
            # Identity (links to chunks table)
            pa.field("chunk_id", pa.string()),
            # Vector embedding (dimension varies by model)
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
            # Denormalized fields for search (avoid JOINs)
            pa.field("file_path", pa.string()),
            pa.field("content", pa.string()),
            pa.field("language", pa.string()),
            pa.field("start_line", pa.int32()),
            pa.field("end_line", pa.int32()),
            pa.field("chunk_type", pa.string()),
            pa.field("name", pa.string()),
            pa.field("function_name", pa.string()),  # Separate function name field
            pa.field("class_name", pa.string()),  # Separate class name field
            pa.field(
                "project_name", pa.string()
            ),  # Monorepo subproject name (empty for single projects)
            pa.field("hierarchy_path", pa.string()),
            # Metadata
            pa.field("embedded_at", pa.string()),  # ISO timestamp
            pa.field("model_version", pa.string()),  # Embedding model used
        ]
    )


# Default schema for backward compatibility (768-dimensional for GraphCodeBERT)
# NOTE: This is rarely used due to dimension auto-detection in VectorsBackend.__init__
VECTORS_SCHEMA = _create_vectors_schema(768)


class VectorsBackend:
    """Manages vectors.lance table for Phase 2 storage and search.

    This backend stores embedded vectors with denormalized metadata for fast
    semantic search without requiring JOINs to the chunks table.

    Features:
    - Vector similarity search with IVF_SQ index
    - Incremental embedding (detect which chunks need vectors)
    - Metadata filtering (language, file_path, chunk_type)
    - Resumable embedding workflow
    - Index rebuilding after batch inserts

    Example:
        backend = VectorsBackend(db_path)
        await backend.initialize()

        # Add vectors for chunks
        chunks_with_vectors = [
            {"chunk_id": "abc", "vector": [...], "content": "...", ...}
        ]
        count = await backend.add_vectors(chunks_with_vectors)

        # Search by vector similarity
        results = await backend.search(query_vector, limit=10)

        # Find unembedded chunks
        all_chunk_ids = ["abc", "def", "ghi"]
        pending = await backend.get_unembedded_chunk_ids(all_chunk_ids)
    """

    TABLE_NAME = "vectors"
    DEFAULT_VECTOR_DIMENSION = 768  # GraphCodeBERT (default model)

    def __init__(
        self, db_path: Path, vector_dim: int | None = None, table_name: str = "vectors"
    ) -> None:
        """Initialize vectors backend.

        Args:
            db_path: Directory for LanceDB database (same as chunks backend)
            vector_dim: Expected vector dimension (auto-detected if not provided)
            table_name: Name of the vectors table (default: "vectors")
        """
        self.db_path = Path(db_path) if isinstance(db_path, str) else db_path
        self.table_name = table_name
        self._db = None
        self._table = None
        # Vector dimension is auto-detected from first batch or set explicitly
        self.vector_dim = vector_dim
        self._append_count = 0  # Track appends for periodic compaction

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
            table_path = self.db_path / f"{table_name}.lance"
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

    def _delete_chunk_ids(self, chunk_ids: list[str]) -> int:
        """Delete existing vectors with specified chunk_ids to prevent duplicates.

        This method is called before appending new vectors to ensure no duplicates
        exist in the vectors table. Uses batched deletes to avoid SQL length limits.

        Args:
            chunk_ids: List of chunk_ids to delete

        Returns:
            Number of chunks deleted (estimated, not exact)

        Note:
            Deletion failures are logged but non-fatal. If deletion fails, append
            will still proceed, which may result in duplicates. This is acceptable
            as search will still work (just with some redundancy).
        """
        if self._table is None or not chunk_ids:
            return 0

        deleted_count = 0
        batch_size = 500  # Limit SQL query length

        try:
            for i in range(0, len(chunk_ids), batch_size):
                batch = chunk_ids[i : i + batch_size]
                # Build SQL IN clause with proper escaping
                id_list = ", ".join(
                    f"'{cid.replace(chr(39), chr(39) * 2)}'" for cid in batch
                )

                try:
                    # LanceDB delete returns None, so we can't get exact count
                    self._table.delete(f"chunk_id IN ({id_list})")
                    deleted_count += len(batch)
                except Exception as e:
                    # Log but don't fail - dedup is best-effort
                    logger.debug(
                        f"Dedup delete failed for batch {i // batch_size + 1} "
                        f"(non-fatal, may result in duplicates): {e}"
                    )

            if deleted_count > 0:
                logger.debug(
                    f"Deleted {deleted_count} existing vectors to prevent duplicates"
                )

        except Exception as e:
            logger.warning(f"Dedup deletion failed (non-fatal): {e}")

        return deleted_count

    async def initialize(self) -> None:
        """Create table if not exists with proper schema and vector index.

        Raises:
            DatabaseInitializationError: If initialization fails
        """
        try:
            # Ensure directory exists
            self.db_path.mkdir(parents=True, exist_ok=True)

            # Connect to LanceDB
            logger.info(f"[VectorsBackend] Connecting to LanceDB at: {self.db_path}")
            self._db = lancedb.connect(str(self.db_path))

            # Check if table exists
            # list_tables() returns a response object with .tables attribute
            tables_response = self._db.list_tables()
            table_names = (
                tables_response.tables
                if hasattr(tables_response, "tables")
                else tables_response
            )

            if self.table_name in table_names:
                try:
                    self._table = self._db.open_table(self.table_name)
                    logger.debug(
                        f"Opened existing vectors table '{self.table_name}' at {self.db_path}"
                    )
                except Exception as e:
                    # Check for stale table entry (listed but not actually openable)
                    if (
                        "not found" in str(e).lower()
                        and "fragment" not in str(e).lower()
                    ):
                        logger.warning(
                            f"Stale table entry '{self.table_name}' detected "
                            f"(listed but not openable: {e}). "
                            f"Cleaning up for fresh creation."
                        )
                        try:
                            self._db.drop_table(self.table_name)
                        except Exception:
                            pass
                        self._table = None
                    # Check for corruption and auto-recover
                    elif self._handle_corrupt_table(e, self.table_name):
                        self._table = None
                        logger.info(
                            f"Vectors table '{self.table_name}' corrupted and deleted. Will be recreated on next index."
                        )
                    else:
                        raise
            else:
                # Table will be created on first add_vectors with schema
                self._table = None
                logger.debug("Vectors table will be created on first write")

        except Exception as e:
            logger.error(f"Failed to initialize vectors backend: {e}")
            raise DatabaseInitializationError(
                f"Vectors backend initialization failed: {e}"
            ) from e

    async def check_dimension_mismatch(self, expected_dim: int) -> bool:
        """Check if vectors table has mismatched dimensions.

        Args:
            expected_dim: Expected vector dimension from embedding model

        Returns:
            True if dimension mismatch detected, False otherwise
        """
        if self._table is None:
            # No table exists yet - no mismatch
            return False

        try:
            # Get schema from existing table
            schema = self._table.schema
            vector_field = schema.field("vector")

            # Extract dimension from list type (e.g., list<item: float>[384])
            # The list_size property gives the fixed dimension
            if hasattr(vector_field.type, "list_size"):
                actual_dim = vector_field.type.list_size
                if actual_dim != expected_dim:
                    logger.warning(
                        f"Vector dimension mismatch: table has {actual_dim}D, "
                        f"model expects {expected_dim}D"
                    )
                    return True
            return False
        except Exception as e:
            logger.warning(f"Failed to check dimension mismatch: {e}")
            return False

    async def recreate_table_with_new_dimensions(self, new_dim: int) -> None:
        """Drop and recreate vectors table with new dimensions.

        This is needed when switching embedding models with different dimensions
        (e.g., 384D -> 768D). Preserves chunk data - only vectors need re-embedding.

        Args:
            new_dim: New vector dimension

        Raises:
            DatabaseNotInitializedError: If backend not initialized
            DatabaseError: If recreation fails
        """
        if self._db is None:
            raise DatabaseNotInitializedError("Vectors backend not initialized")

        try:
            # Check if table exists
            tables_response = self._db.list_tables()
            table_names = (
                tables_response.tables
                if hasattr(tables_response, "tables")
                else tables_response
            )

            if self.table_name in table_names:
                # Drop existing table
                self._db.drop_table(self.table_name)
                logger.info(
                    f"Dropped vectors table (dimension change: {self.vector_dim}D -> {new_dim}D)"
                )

            # Update dimension
            self.vector_dim = new_dim
            self._table = None
            logger.info(
                f"Vectors table will be recreated with {new_dim}D on first write"
            )

        except Exception as e:
            logger.error(f"Failed to recreate vectors table: {e}")
            raise DatabaseError(f"Failed to recreate vectors table: {e}") from e

    async def add_vectors(
        self,
        chunks_with_vectors: list[dict[str, Any]],
        model_version: str = "all-MiniLM-L6-v2",
    ) -> int:
        """Add embedded chunks to vectors table.

        All chunks must include a 'vector' field with the embedding.
        Other required fields are denormalized from chunks table for fast search.

        Args:
            chunks_with_vectors: List of dicts with chunk data + 'vector' field:
                - chunk_id (str): Unique identifier linking to chunks.lance
                - vector (list[float]): Embedding vector (dimension varies by model)
                - file_path (str): Source file path
                - content (str): Code content for display
                - language (str): Programming language
                - start_line (int): Starting line number
                - end_line (int): Ending line number
                - chunk_type (str): function, class, method, etc.
                - name (str): Function/class name
                - hierarchy_path (str): Dotted path (e.g., "MyClass.my_method")
            model_version: Embedding model identifier (default: all-MiniLM-L6-v2)

        Returns:
            Number of vectors added

        Raises:
            DatabaseNotInitializedError: If backend not initialized
            DatabaseError: If adding vectors fails
        """
        if self._db is None:
            raise DatabaseNotInitializedError("Vectors backend not initialized")

        if not chunks_with_vectors:
            return 0

        try:
            # Normalize and validate vectors
            normalized_vectors = []
            timestamp = datetime.utcnow().isoformat()

            for chunk in chunks_with_vectors:
                # Validate required fields
                if "chunk_id" not in chunk:
                    raise ValueError("Missing required field: chunk_id")
                if "vector" not in chunk:
                    raise ValueError(f"Missing vector for chunk: {chunk['chunk_id']}")

                # Auto-detect vector dimension from first vector
                vector = chunk["vector"]
                if self.vector_dim is None:
                    self.vector_dim = len(vector)
                    logger.info(
                        f"Auto-detected vector dimension: {self.vector_dim}D from first chunk"
                    )

                # Validate vector dimensions match detected/expected dimension
                if len(vector) != self.vector_dim:
                    raise ValueError(
                        f"Invalid vector dimension for chunk {chunk['chunk_id']}: "
                        f"expected {self.vector_dim}, got {len(vector)}"
                    )

                # Required fields
                normalized = {
                    "chunk_id": chunk["chunk_id"],
                    "vector": vector,
                    "file_path": chunk["file_path"],
                    "content": chunk["content"],
                    "language": chunk["language"],
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "chunk_type": chunk["chunk_type"],
                    "name": chunk["name"],
                }

                # Optional fields with defaults
                normalized["function_name"] = chunk.get("function_name", "")
                normalized["class_name"] = chunk.get("class_name", "")
                normalized["project_name"] = chunk.get("project_name", "")
                normalized["hierarchy_path"] = chunk.get("hierarchy_path", "")

                # Metadata
                normalized["embedded_at"] = timestamp
                normalized["model_version"] = model_version

                normalized_vectors.append(normalized)

            # Convert to PyArrow Table with explicit schema to avoid type inference issues
            # This ensures vectors are treated as fixed-size lists, not variable-length lists
            schema = _create_vectors_schema(self.vector_dim)

            # Convert normalized vectors to PyArrow Table with explicit schema
            # This prevents LanceDB from inferring variable-length list types
            import pyarrow as pa_table_module

            pa_table = pa_table_module.Table.from_pylist(
                normalized_vectors, schema=schema
            )

            # Deduplicate: Extract chunk_ids to remove before adding new vectors
            chunk_ids_to_add = [row["chunk_id"] for row in normalized_vectors]

            # Create or append to table
            if self._table is None:
                # Check if table exists (it might exist but self._table wasn't opened)
                tables_response = self._db.list_tables()
                table_names = (
                    tables_response.tables
                    if hasattr(tables_response, "tables")
                    else tables_response
                )

                if self.table_name in table_names:
                    # Table exists, check if dimensions match before appending
                    try:
                        existing_table = self._db.open_table(self.table_name)
                    except Exception as open_err:
                        if "not found" in str(open_err).lower():
                            logger.warning(
                                f"Stale table entry '{self.table_name}' detected in add_vectors "
                                f"(listed but not openable: {open_err}). "
                                f"Dropping stale entry and creating fresh table."
                            )
                            try:
                                self._db.drop_table(self.table_name)
                            except Exception:
                                pass
                            self._table = self._db.create_table(
                                self.table_name, pa_table, schema=schema
                            )
                            logger.debug(
                                f"Created fresh vectors table with {len(normalized_vectors)} vectors (dimension: {self.vector_dim})"
                            )
                        else:
                            raise
                    else:
                        existing_schema = existing_table.schema
                        vector_field = existing_schema.field("vector")

                        # Extract existing dimension from fixed-size list type
                        existing_dim = None
                        if hasattr(vector_field.type, "list_size"):
                            existing_dim = vector_field.type.list_size

                        if existing_dim == self.vector_dim:
                            # Dimensions match, now check schema compatibility
                            # Check if all required columns exist in the existing table
                            existing_field_names = set(existing_schema.names)
                            new_field_names = set(schema.names)
                            missing_columns = new_field_names - existing_field_names

                            if missing_columns:
                                # Schema evolution: existing table is missing new columns
                                # Add missing columns with null values (preserves existing embeddings)
                                logger.info(
                                    f"Schema evolution detected: adding missing columns {missing_columns} to existing table"
                                )

                                # Build PyArrow fields for missing columns
                                missing_fields = [
                                    schema.field(col_name)
                                    for col_name in missing_columns
                                ]

                                # Add columns with null/empty defaults
                                existing_table.add_columns(missing_fields)

                                # Reopen table to get updated schema
                                self._table = self._db.open_table(self.table_name)
                                logger.info(
                                    f"Added {len(missing_columns)} new columns to vectors table (preserving {existing_table.count_rows()} existing embeddings)"
                                )
                            else:
                                # Schema compatible, open existing table
                                self._table = existing_table

                            # Deduplicate: Delete existing vectors with same chunk_ids before appending
                            self._delete_chunk_ids(chunk_ids_to_add)

                            self._table.add(pa_table, mode="append")
                            logger.debug(
                                f"Opened existing table and added {len(normalized_vectors)} vectors (append mode)"
                            )
                        else:
                            # Dimension mismatch - drop and recreate table
                            logger.warning(
                                f"Vector dimension mismatch: existing table has {existing_dim}D, "
                                f"new data has {self.vector_dim}D. Dropping and recreating table."
                            )
                            self._db.drop_table(self.table_name)
                            self._table = self._db.create_table(
                                self.table_name, pa_table, schema=schema
                            )
                            logger.info(
                                f"Recreated vectors table with {len(normalized_vectors)} vectors (dimension: {self.vector_dim})"
                            )
                else:
                    # Create table with first batch using explicit schema
                    self._table = self._db.create_table(
                        self.table_name, pa_table, schema=schema
                    )
                    logger.debug(
                        f"Created vectors table with {len(normalized_vectors)} vectors (dimension: {self.vector_dim})"
                    )
            else:
                # Check dimension mismatch before appending
                # This handles cases where the table was opened with stale dimension
                existing_schema = self._table.schema
                vector_field = existing_schema.field("vector")
                existing_dim = None
                if hasattr(vector_field.type, "list_size"):
                    existing_dim = vector_field.type.list_size

                if existing_dim != self.vector_dim:
                    # Dimension mismatch detected - drop and recreate table
                    logger.warning(
                        f"Vector dimension mismatch detected during append: "
                        f"table has {existing_dim}D, new data has {self.vector_dim}D. "
                        f"Dropping and recreating table."
                    )
                    self._db.drop_table(self.table_name)
                    self._table = self._db.create_table(
                        self.table_name, pa_table, schema=schema
                    )
                    logger.info(
                        f"Recreated vectors table with {len(normalized_vectors)} vectors "
                        f"(dimension: {self.vector_dim})"
                    )
                else:
                    # Dimensions match, now check schema compatibility
                    # Check if all required columns exist in the existing table
                    existing_field_names = set(existing_schema.names)
                    new_field_names = set(schema.names)
                    missing_columns = new_field_names - existing_field_names

                    if missing_columns:
                        # Schema evolution: existing table is missing new columns
                        # Add missing columns with null values (preserves existing embeddings)
                        logger.info(
                            f"Schema evolution detected during append: adding missing columns {missing_columns} to existing table"
                        )

                        # Build PyArrow fields for missing columns
                        missing_fields = [
                            schema.field(col_name) for col_name in missing_columns
                        ]

                        # Add columns with null/empty defaults
                        self._table.add_columns(missing_fields)

                        # Reopen table to get updated schema
                        self._table = self._db.open_table(self.table_name)
                        logger.info(
                            f"Added {len(missing_columns)} new columns to vectors table (preserving existing embeddings)"
                        )

                    # Schema compatible (either was already compatible or we just added missing columns)
                    # Deduplicate: Delete existing vectors with same chunk_ids before appending
                    self._delete_chunk_ids(chunk_ids_to_add)

                    # OPTIMIZATION: Use mode='append' for faster bulk inserts
                    # This defers index updates until later, improving write throughput
                    self._table.add(pa_table, mode="append")
                    logger.debug(
                        f"Added {len(normalized_vectors)} vectors to table (append mode)"
                    )

            # Track appends and compact periodically to prevent file descriptor exhaustion
            self._append_count += 1
            if self._append_count % 500 == 0:
                self._compact_table()

            return len(normalized_vectors)

        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            raise DatabaseError(f"Failed to add vectors: {e}") from e

    def _compact_table(self) -> None:
        """Compact LanceDB table to merge small fragments.

        LanceDB creates one file per append operation. During large reindexing,
        this accumulates thousands of small files, exhausting file descriptors.
        Periodic compaction merges fragments to reduce file count.

        This is called every 500 appends to keep file count manageable.
        Failures are non-fatal (best-effort optimization).

        WORKAROUND: Skipped on macOS to avoid SIGBUS crash caused by memory conflict
        between PyTorch MPS memory-mapped model files and LanceDB compaction operations.
        """
        if platform.system() == "Darwin":
            logger.debug(
                "Skipping vectors table compaction on macOS to avoid SIGBUS crash "
                "(PyTorch MPS + LanceDB compaction memory conflict)"
            )
            return

        if self._table is None:
            return

        # Linux guard: skip compaction for large tables to prevent arrow offset overflow
        # documented in lance issue #3330. compact_files() on tables > 100k rows can
        # trigger an int32 overflow in the offset buffer on Linux.
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
            # LanceDB 0.5+ API: compact_files() merges data fragments
            self._table.compact_files()
            logger.debug(
                f"Compacted vectors table after {self._append_count} appends (reduced fragment count)"
            )
        except AttributeError:
            # Fallback for older LanceDB versions without compact_files()
            try:
                self._table.cleanup_old_versions()
                logger.debug("Cleaned up old versions (compaction not available)")
            except Exception as e:
                logger.debug(f"Compaction/cleanup not available: {e}")
        except Exception as e:
            # Compaction is best-effort, don't fail on errors
            logger.debug(f"Compaction failed (non-fatal): {e}")

    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic vector search.

        Performs similarity search using LanceDB's IVF_SQ index for fast
        approximate nearest neighbor (ANN) search.

        Args:
            query_vector: Query embedding (dimension must match table)
            limit: Max results to return
            filters: Optional metadata filters:
                - language (str): Filter by programming language
                - file_path (str): Filter by file path
                - chunk_type (str): Filter by chunk type (function, class, etc.)
                - function_name (str): Filter by function name
                - class_name (str): Filter by class name

        Returns:
            List of matching chunks with similarity scores, sorted by relevance:
            [
                {
                    "chunk_id": "abc",
                    "content": "def foo()...",
                    "file_path": "src/main.py",
                    "start_line": 10,
                    "end_line": 20,
                    "language": "python",
                    "chunk_type": "function",
                    "name": "foo",
                    "function_name": "foo",
                    "class_name": "",
                    "hierarchy_path": "MyClass.foo",
                    "similarity": 0.95,
                    "_distance": 0.05
                }
            ]

        Raises:
            DatabaseNotInitializedError: If backend not initialized
            SearchError: If search fails
        """
        if self._table is None:
            # Table doesn't exist - raise error instead of silently returning []
            raise SearchError(
                "Vectors table not found. Index may be corrupted or not built yet. "
                "Run 'mcp-vector-search index' to rebuild the index."
            )

        # Validate query vector dimensions (if dimension is known)
        if self.vector_dim is not None and len(query_vector) != self.vector_dim:
            raise SearchError(
                f"Invalid query vector dimension: "
                f"expected {self.vector_dim}, got {len(query_vector)}"
            )

        try:
            # Build LanceDB query with cosine metric.
            # nprobes and refine_factor enable two-stage ANN retrieval:
            #   - nprobes=20: scan 20 IVF partitions (higher = better recall, slower)
            #   - refine_factor=5: re-rank 5x candidates with exact distances
            # LanceDB silently ignores these when no ANN index exists (brute-force).
            search = (
                self._table.search(query_vector)
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
                            # Escape single quotes in value
                            escaped_value = value.replace("'", "''")
                            filter_clauses.append(f"{key} = '{escaped_value}'")
                        elif isinstance(value, list):
                            # Support IN queries
                            escaped_values = [v.replace("'", "''") for v in value]
                            values_str = ", ".join(f"'{v}'" for v in escaped_values)
                            filter_clauses.append(f"{key} IN ({values_str})")
                        else:
                            filter_clauses.append(f"{key} = {value}")

                if filter_clauses:
                    where_clause = " AND ".join(filter_clauses)
                    search = search.where(where_clause)

            # Execute search
            results = search.to_list()

            # Convert LanceDB results to standard format
            search_results = []
            for result in results:
                # LanceDB returns _distance (L2 distance for vectors)
                # Since vectors aren't normalized, we get large L2 distances
                # Convert cosine distance to similarity score
                # Cosine distance ranges from 0 (identical) to 2 (opposite)
                # Convert to similarity: 0 distance -> 1.0 similarity, 2 distance -> 0.0
                distance = result.get("_distance", 0.0)
                similarity = max(0.0, 1.0 - (distance / 2.0))

                search_result = {
                    "chunk_id": result["chunk_id"],
                    "content": result["content"],
                    "file_path": result["file_path"],
                    "start_line": result["start_line"],
                    "end_line": result["end_line"],
                    "language": result["language"],
                    "chunk_type": result["chunk_type"],
                    "name": result["name"],
                    "function_name": result.get("function_name", ""),
                    "class_name": result.get("class_name", ""),
                    "project_name": result.get("project_name", ""),
                    "hierarchy_path": result.get("hierarchy_path", ""),
                    "similarity": similarity,
                    "_distance": distance,
                }
                search_results.append(search_result)

            logger.debug(
                f"Vector search returned {len(search_results)} results (limit: {limit})"
            )
            return search_results

        except Exception as e:
            # Check for corruption and auto-recover
            if self._handle_corrupt_table(e, self.table_name):
                logger.error(
                    "Vectors table corrupted. Search unavailable. Run 'mcp-vector-search index' to rebuild."
                )
                raise SearchError(
                    "Index corrupted. Run 'mcp-vector-search index' to rebuild."
                ) from e

            logger.error(f"Vector search failed: {e}")
            raise SearchError(f"Vector search failed: {e}") from e

    async def search_by_file(
        self, file_path: str, query_vector: list[float], limit: int = 10
    ) -> list[dict[str, Any]]:
        """Search within a specific file.

        Convenience method that applies file_path filter to search.

        Args:
            file_path: Path to file to search within
            query_vector: Query embedding (dimension must match table)
            limit: Max results to return

        Returns:
            List of matching chunks from specified file
        """
        return await self.search(
            query_vector, limit=limit, filters={"file_path": file_path}
        )

    async def delete_file_vectors(self, file_path: str) -> int:
        """Delete all vectors for a file (for re-indexing).

        Args:
            file_path: Path to file whose vectors should be deleted

        Returns:
            Number of vectors deleted

        Raises:
            DatabaseNotInitializedError: If backend not initialized
            DatabaseError: If deletion fails
        """
        if self._table is None:
            return 0

        try:
            # Count vectors before deletion
            df = self._table.to_pandas().query(f"file_path == '{file_path}'")
            count = len(df)

            if count == 0:
                return 0

            # Delete matching rows
            self._table.delete(f"file_path = '{file_path}'")

            logger.debug(f"Deleted {count} vectors for file: {file_path}")
            return count

        except Exception as e:
            logger.error(f"Failed to delete vectors for {file_path}: {e}")
            raise DatabaseError(f"Failed to delete vectors: {e}") from e

    async def delete_files_batch(self, file_paths: list[str]) -> int:
        """Delete vectors for multiple files in batched IN expressions.

        Uses IN clauses capped at DELETE_BATCH_LIMIT per batch to prevent
        DataFusion's recursive descent parser from stack-overflowing on large
        OR chains, which causes SEGV (signal 139) on Linux.

        Args:
            file_paths: List of relative file paths whose vectors should be deleted

        Returns:
            Total number of file-path batches processed (not exact vector count,
            since LanceDB delete() returns None)
        """
        if self._table is None or not file_paths:
            return 0

        deleted_count = 0
        total_batches = (len(file_paths) + DELETE_BATCH_LIMIT - 1) // DELETE_BATCH_LIMIT

        for batch_num, i in enumerate(
            range(0, len(file_paths), DELETE_BATCH_LIMIT), start=1
        ):
            batch = file_paths[i : i + DELETE_BATCH_LIMIT]

            # Escape single quotes and build SQL IN clause.
            # IN instead of chained OR keeps DataFusion's parse tree shallow,
            # preventing stack overflow (SEGV signal 139) on Linux with large batches.
            escaped = [fp.replace("'", "''") for fp in batch]
            values_sql = ", ".join(f"'{fp}'" for fp in escaped)
            filter_expr = f"file_path IN ({values_sql})"

            try:
                self._table.delete(filter_expr)
                deleted_count += len(batch)
                logger.debug(
                    "Deleted vectors batch %d/%d (%d files, running total: %d)",
                    batch_num,
                    total_batches,
                    len(batch),
                    deleted_count,
                )
            except Exception as e:
                # Handle LanceDB "Not found" errors gracefully; continue other batches
                error_msg = str(e).lower()
                if "not found" in error_msg:
                    logger.debug(
                        "No vectors to delete for batch %d/%d (not in index)",
                        batch_num,
                        total_batches,
                    )
                else:
                    logger.error(
                        "Failed to delete vectors batch %d/%d: %s",
                        batch_num,
                        total_batches,
                        e,
                    )

        logger.debug("Batch delete complete: %d files processed", deleted_count)
        return deleted_count

    async def update_file_path(self, old_path: str, new_path: str) -> int:
        """Update file_path for all vectors of a moved/renamed file.

        Uses LanceDB table.update() for in-place metadata update — no re-embedding needed.
        Safe on macOS (update doesn't trigger mmap compaction like delete does).

        Args:
            old_path: Current (old) relative file path stored in the table
            new_path: New relative file path to set

        Returns:
            Number of rows updated (0 if table not initialized or update fails)
        """
        if self._table is None:
            return 0
        try:
            escaped_old = old_path.replace("'", "''")
            result = self._table.update(
                where=f"file_path = '{escaped_old}'",
                values={"file_path": new_path},
            )
            rows = getattr(result, "rows_updated", 0) if result else 0
            logger.debug(
                "Updated vector file_path: %s → %s (%d vectors)",
                old_path,
                new_path,
                rows,
            )
            return rows
        except Exception as e:
            logger.error(
                "Failed to update vector file_path %s → %s: %s", old_path, new_path, e
            )
            return 0

    async def get_chunk_vector(self, chunk_id: str) -> list[float] | None:
        """Get vector for a specific chunk.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Vector embedding if found, None otherwise
        """
        if self._table is None:
            return None

        try:
            df = self._table.to_pandas().query(f"chunk_id == '{chunk_id}'")

            if df.empty:
                return None

            # Return first matching vector
            # Vector might be a numpy array, pyarrow array, or list
            vector = df.iloc[0]["vector"]
            if isinstance(vector, list):
                return vector
            # Convert numpy/pyarrow arrays to list
            try:
                return vector.tolist()
            except AttributeError:
                # If it's already iterable but not a list
                return list(vector)

        except Exception as e:
            logger.warning(f"Failed to get vector for chunk {chunk_id}: {e}")
            return None

    async def get_chunk_vectors_batch(
        self, chunk_ids: list[str]
    ) -> dict[str, list[float]]:
        """Get vectors for multiple chunks in a single query.

        More efficient than calling get_chunk_vector() in a loop.

        Args:
            chunk_ids: List of chunk identifiers

        Returns:
            Dictionary mapping chunk_id to vector embedding
            Only includes chunk_ids that were found in the table
        """
        if self._table is None or not chunk_ids:
            return {}

        try:
            # Build SQL IN clause with proper escaping
            # Batch into chunks of 500 to avoid SQL query length limits
            batch_size = 500
            all_vectors: dict[str, list[float]] = {}

            for i in range(0, len(chunk_ids), batch_size):
                batch = chunk_ids[i : i + batch_size]
                id_list = ", ".join(
                    f"'{cid.replace(chr(39), chr(39) * 2)}'" for cid in batch
                )

                # Query for this batch
                df = self._table.to_pandas().query(f"chunk_id in [{id_list}]")

                # Extract vectors from results
                for _, row in df.iterrows():
                    chunk_id = row["chunk_id"]
                    vector = row["vector"]

                    # Convert to list if needed
                    if isinstance(vector, list):
                        all_vectors[chunk_id] = vector
                    else:
                        try:
                            all_vectors[chunk_id] = vector.tolist()
                        except AttributeError:
                            all_vectors[chunk_id] = list(vector)

            logger.debug(
                f"Retrieved {len(all_vectors)} vectors from {len(chunk_ids)} requested chunk_ids"
            )
            return all_vectors

        except Exception as e:
            logger.warning(f"Failed to get vectors for batch: {e}")
            return {}

    async def has_vector(self, chunk_id: str) -> bool:
        """Check if chunk already has a vector.

        Args:
            chunk_id: Chunk identifier

        Returns:
            True if vector exists, False otherwise
        """
        if self._table is None:
            return False

        try:
            df = self._table.to_pandas().query(f"chunk_id == '{chunk_id}'")
            return not df.empty

        except Exception as e:
            logger.warning(f"Failed to check vector for chunk {chunk_id}: {e}")
            return False

    async def get_stats(self) -> dict[str, Any]:
        """Get vector statistics.

        Returns:
            Dictionary with vector counts and metadata:
            {
                "total": 1000,
                "files": 50,
                "languages": {"python": 30, "javascript": 20, ...},
                "chunk_types": {"function": 40, "class": 10, ...},
                "models": {"all-MiniLM-L6-v2": 1000},
                "avg_vector_norm": 1.0
            }
        """
        if self._table is None:
            return {
                "total": 0,
                "files": 0,
                "languages": {},
                "chunk_types": {},
                "models": {},
                "avg_vector_norm": 0.0,
            }

        try:
            df = self._table.to_pandas()

            # Count total vectors
            total = len(df)

            # Count unique files
            file_count = df["file_path"].nunique()

            # Count by language
            language_counts = df["language"].value_counts().to_dict()

            # Count by chunk type
            chunk_type_counts = df["chunk_type"].value_counts().to_dict()

            # Count by model version
            model_counts = df["model_version"].value_counts().to_dict()

            # Calculate average vector norm (for debugging)
            try:
                import numpy as np

                vectors = np.array(list(df["vector"].values))
                norms = np.linalg.norm(vectors, axis=1)
                avg_norm = float(np.mean(norms))
            except Exception:
                avg_norm = 0.0

            return {
                "total": total,
                "files": file_count,
                "languages": language_counts,
                "chunk_types": chunk_type_counts,
                "models": model_counts,
                "avg_vector_norm": avg_norm,
            }

        except Exception as e:
            # Check for corruption and auto-recover
            if self._handle_corrupt_table(e, self.table_name):
                logger.warning(
                    "Vectors table corrupted. Stats unavailable. Run 'mcp-vector-search index' to rebuild."
                )
                return {
                    "total": 0,
                    "files": 0,
                    "languages": {},
                    "chunk_types": {},
                    "models": {},
                    "avg_vector_norm": 0.0,
                }

            logger.error(f"Failed to get vector stats: {e}")
            return {
                "total": 0,
                "files": 0,
                "languages": {},
                "chunk_types": {},
                "models": {},
                "avg_vector_norm": 0.0,
            }

    async def rebuild_index(self) -> None:
        """Rebuild the vector index for ANN search.

        Creates an IVF_SQ approximate nearest neighbor index after bulk inserts
        to enable fast similarity search at scale.  Skipped for small datasets
        (< 256 rows) where brute-force is faster.  Non-fatal: if index creation
        fails search continues with exact brute-force scan.

        Note: This is an expensive operation and should not be called after every add.
        """
        if self._table is None:
            logger.warning("Cannot rebuild index: table not initialized")
            return

        try:
            import math

            row_count = self._table.count_rows()

            # Skip index for very small datasets — brute-force is faster and
            # IVF training requires at least sample_rate * num_partitions vectors.
            # With 16 partitions and sample_rate=256 that's ~4 K minimum, but we
            # use a conservative 256-row floor to match actual LanceDB behaviour.
            if row_count < 256:
                logger.info(
                    f"Skipping vector index creation: {row_count} rows "
                    f"(minimum 256 required)"
                )
                return

            # Determine vector dimension from a sample row
            sample = self._table.search().limit(1).to_list()
            if not sample or "vector" not in sample[0]:
                logger.warning("Cannot determine vector dimension for index")
                return

            vector_dim = len(sample[0]["vector"])

            # Adaptive partition count: sqrt(N) clamped to [16, 512]
            # Rule of thumb from LanceDB docs: good recall vs. latency balance
            num_partitions = min(512, max(16, int(math.sqrt(row_count))))

            logger.info(
                f"Creating IVF_SQ vector index: {row_count:,} rows, {vector_dim}d, "
                f"{num_partitions} partitions (int8 scalar quantization)"
            )

            self._table.create_index(
                metric="cosine",
                num_partitions=num_partitions,
                index_type="IVF_SQ",
                replace=True,
            )

            logger.info("Vector index created successfully")

        except Exception as e:
            # Check for corruption and auto-recover
            if self._handle_corrupt_table(e, self.table_name):
                logger.warning(
                    "Vectors table corrupted during index rebuild. Skipped. "
                    "Run 'mcp-vector-search index' to rebuild."
                )
                return

            # Non-fatal — search falls back to brute-force automatically
            logger.warning(f"Vector index creation failed (non-fatal): {e}")

    async def get_unembedded_chunk_ids(self, all_chunk_ids: list[str]) -> list[str]:
        """Find which chunks don't have vectors yet.

        This enables incremental embedding - only embed chunks that don't
        already have vectors in the table.

        Args:
            all_chunk_ids: List of all chunk IDs from chunks table

        Returns:
            List of chunk IDs that don't have vectors yet
        """
        if self._table is None or not all_chunk_ids:
            # No vectors table yet, all chunks need embedding
            return all_chunk_ids

        try:
            # Get all chunk IDs that have vectors
            df = self._table.to_pandas()
            embedded_ids = set(df["chunk_id"].values)

            # Find chunks without vectors
            unembedded = [cid for cid in all_chunk_ids if cid not in embedded_ids]

            logger.debug(
                f"Found {len(unembedded)} unembedded chunks out of {len(all_chunk_ids)} total"
            )
            return unembedded

        except Exception as e:
            logger.error(f"Failed to get unembedded chunk IDs: {e}")
            # On error, assume all chunks need embedding
            return all_chunk_ids

    async def close(self) -> None:
        """Close database connections.

        LanceDB doesn't require explicit closing, but we set references to None
        for consistency with other backends.
        """
        self._table = None
        self._db = None
        logger.debug("Vectors backend closed")

    async def __aenter__(self) -> "VectorsBackend":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
