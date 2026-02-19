"""Vectors backend for Phase 2 storage (post-embedding).

This module manages the vectors.lance table which stores embedded vectors
for semantic search. Enables:
- Vector similarity search
- Resumable embedding (picks up from chunks.lance)
- Search without JOINs (denormalized data)
- Efficient vector indexing with IVF_PQ
"""

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
            pa.field("hierarchy_path", pa.string()),
            # Metadata
            pa.field("embedded_at", pa.string()),  # ISO timestamp
            pa.field("model_version", pa.string()),  # Embedding model used
        ]
    )


# Default schema for backward compatibility (384-dimensional for all-MiniLM-L6-v2)
VECTORS_SCHEMA = _create_vectors_schema(384)


class VectorsBackend:
    """Manages vectors.lance table for Phase 2 storage and search.

    This backend stores embedded vectors with denormalized metadata for fast
    semantic search without requiring JOINs to the chunks table.

    Features:
    - Vector similarity search with IVF_PQ index
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
    DEFAULT_VECTOR_DIMENSION = (
        384  # all-MiniLM-L6-v2 (local CPU default); 768 for GraphCodeBERT on CUDA
    )

    def __init__(self, db_path: Path, vector_dim: int | None = None) -> None:
        """Initialize vectors backend.

        Args:
            db_path: Directory for LanceDB database (same as chunks backend)
            vector_dim: Expected vector dimension (auto-detected if not provided)
        """
        self.db_path = Path(db_path) if isinstance(db_path, str) else db_path
        self._db = None
        self._table = None
        # Vector dimension is auto-detected from first batch or set explicitly
        self.vector_dim = vector_dim

    async def initialize(self) -> None:
        """Create table if not exists with proper schema and vector index.

        Raises:
            DatabaseInitializationError: If initialization fails
        """
        try:
            # Ensure directory exists
            self.db_path.mkdir(parents=True, exist_ok=True)

            # Connect to LanceDB
            self._db = lancedb.connect(str(self.db_path))

            # Check if table exists
            # list_tables() returns a response object with .tables attribute
            tables_response = self._db.list_tables()
            table_names = (
                tables_response.tables
                if hasattr(tables_response, "tables")
                else tables_response
            )

            if self.TABLE_NAME in table_names:
                self._table = self._db.open_table(self.TABLE_NAME)
                logger.debug(f"Opened existing vectors table at {self.db_path}")
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

            if self.TABLE_NAME in table_names:
                # Drop existing table
                self._db.drop_table(self.TABLE_NAME)
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
        model_version: str = "graphcodebert-base",
    ) -> int:
        """Add embedded chunks to vectors table.

        All chunks must include a 'vector' field with the embedding.
        Other required fields are denormalized from chunks table for fast search.

        Args:
            chunks_with_vectors: List of dicts with chunk data + 'vector' field:
                - chunk_id (str): Unique identifier linking to chunks.lance
                - vector (list[float]): Embedding vector (384D)
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
                    logger.debug(f"Auto-detected vector dimension: {self.vector_dim}")

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
                normalized["hierarchy_path"] = chunk.get("hierarchy_path", "")

                # Metadata
                normalized["embedded_at"] = timestamp
                normalized["model_version"] = model_version

                normalized_vectors.append(normalized)

            # Create or append to table
            if self._table is None:
                # Check if table exists (it might exist but self._table wasn't opened)
                tables_response = self._db.list_tables()
                table_names = (
                    tables_response.tables
                    if hasattr(tables_response, "tables")
                    else tables_response
                )

                if self.TABLE_NAME in table_names:
                    # Table exists, open it
                    self._table = self._db.open_table(self.TABLE_NAME)
                    self._table.add(normalized_vectors, mode="append")
                    logger.debug(
                        f"Opened existing table and added {len(normalized_vectors)} vectors (append mode)"
                    )
                else:
                    # Create table with first batch using detected dimension
                    schema = _create_vectors_schema(self.vector_dim)
                    self._table = self._db.create_table(
                        self.TABLE_NAME, normalized_vectors, schema=schema
                    )
                    logger.debug(
                        f"Created vectors table with {len(normalized_vectors)} vectors (dimension: {self.vector_dim})"
                    )
            else:
                # Append to existing table
                # OPTIMIZATION: Use mode='append' for faster bulk inserts
                # This defers index updates until later, improving write throughput
                self._table.add(normalized_vectors, mode="append")
                logger.debug(
                    f"Added {len(normalized_vectors)} vectors to table (append mode)"
                )

            return len(normalized_vectors)

        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            raise DatabaseError(f"Failed to add vectors: {e}") from e

    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic vector search.

        Performs similarity search using LanceDB's IVF_PQ index for fast
        approximate nearest neighbor (ANN) search.

        Args:
            query_vector: Query embedding (384D)
            limit: Max results to return
            filters: Optional metadata filters:
                - language (str): Filter by programming language
                - file_path (str): Filter by file path
                - chunk_type (str): Filter by chunk type (function, class, etc.)

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
            # Empty database - return empty results
            return []

        # Validate query vector dimensions (if dimension is known)
        if self.vector_dim is not None and len(query_vector) != self.vector_dim:
            raise SearchError(
                f"Invalid query vector dimension: "
                f"expected {self.vector_dim}, got {len(query_vector)}"
            )

        try:
            # Build LanceDB query
            search = self._table.search(query_vector).limit(limit)

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
                # Convert to similarity: similarity = 1 / (1 + distance)
                # This ensures similarity is in range [0, 1] where 1 is most similar
                distance = result.get("_distance", 0.0)
                similarity = 1.0 / (1.0 + distance)

                search_result = {
                    "chunk_id": result["chunk_id"],
                    "content": result["content"],
                    "file_path": result["file_path"],
                    "start_line": result["start_line"],
                    "end_line": result["end_line"],
                    "language": result["language"],
                    "chunk_type": result["chunk_type"],
                    "name": result["name"],
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
            logger.error(f"Vector search failed: {e}")
            raise SearchError(f"Vector search failed: {e}") from e

    async def search_by_file(
        self, file_path: str, query_vector: list[float], limit: int = 10
    ) -> list[dict[str, Any]]:
        """Search within a specific file.

        Convenience method that applies file_path filter to search.

        Args:
            file_path: Path to file to search within
            query_vector: Query embedding (384D)
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
        """Rebuild the vector index (after large batch inserts).

        This should be called after bulk inserts to optimize search performance.
        LanceDB creates an IVF_PQ index for fast approximate nearest neighbor search.

        Note: This is an expensive operation and should not be called after every add.
        """
        if self._table is None:
            logger.debug("No table to rebuild index")
            return

        try:
            # Create IVF_PQ index for fast vector search
            # This uses Inverted File Index with Product Quantization
            self._table.create_index(
                metric="L2",  # L2 distance for cosine similarity approximation
                num_partitions=256,  # Number of IVF partitions (good default)
                num_sub_vectors=96,  # PQ sub-vectors (384 / 4 = 96)
            )
            logger.info("Vector index rebuilt successfully")

        except Exception as e:
            # Non-fatal - search will still work with brute force
            logger.warning(f"Failed to rebuild vector index: {e}")

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
