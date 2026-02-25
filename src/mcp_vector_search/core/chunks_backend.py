"""Chunks backend for Phase 1 storage (pre-embedding).

This module manages the chunks.lance table which stores parsed code chunks
before embedding. Enables:
- Fast parse/chunk phase (no embedding bottleneck)
- Change detection via file_hash
- Resumable embedding (track embedding_status)
- Incremental re-indexing (skip unchanged files)
"""

import hashlib
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
)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of file content.

    Args:
        file_path: Path to file to hash

    Returns:
        SHA-256 hash as hexadecimal string

    Raises:
        FileNotFoundError: If file does not exist
        PermissionError: If file cannot be read
    """
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


# PyArrow schema for chunks table
CHUNKS_SCHEMA = pa.schema(
    [
        # Identity
        pa.field("chunk_id", pa.string()),
        pa.field("file_path", pa.string()),
        pa.field("file_hash", pa.string()),
        # Content
        pa.field("content", pa.string()),
        pa.field("language", pa.string()),
        # Position
        pa.field("start_line", pa.int32()),
        pa.field("end_line", pa.int32()),
        pa.field("start_char", pa.int32()),
        pa.field("end_char", pa.int32()),
        # Hierarchy
        pa.field("chunk_type", pa.string()),
        pa.field("name", pa.string()),
        pa.field("parent_name", pa.string()),
        pa.field("hierarchy_path", pa.string()),
        # Metadata
        pa.field("docstring", pa.string()),
        pa.field("signature", pa.string()),
        pa.field("complexity", pa.int32()),
        pa.field("token_count", pa.int32()),
        # Code relationships (for KG)
        pa.field("calls", pa.list_(pa.string())),  # Function/method calls
        pa.field("imports", pa.list_(pa.string())),  # Import statements (JSON strings)
        pa.field("inherits_from", pa.list_(pa.string())),  # Base classes
        # Git blame metadata
        pa.field("last_author", pa.string()),
        pa.field("last_modified", pa.string()),
        pa.field("commit_hash", pa.string()),
        # Phase tracking
        pa.field(
            "embedding_status", pa.string()
        ),  # pending, processing, complete, error
        pa.field("embedding_batch_id", pa.int32()),  # For resume logic (nullable)
        pa.field("created_at", pa.string()),  # ISO timestamp
        pa.field("updated_at", pa.string()),  # ISO timestamp
        pa.field("error_message", pa.string()),  # Error details if status=error
    ]
)


class ChunksBackend:
    """Manages chunks.lance table for Phase 1 storage.

    This backend stores parsed code chunks WITHOUT embeddings to enable:
    - Fast Phase 1 indexing (no embedding overhead)
    - Incremental updates (skip unchanged files via file_hash)
    - Resumable Phase 2 embedding (track embedding_status)
    - Crash recovery (identify stuck "processing" chunks)

    Example:
        backend = ChunksBackend(db_path)
        await backend.initialize()

        # Add chunks for a file
        chunks = [{"chunk_id": "abc", "content": "...", ...}]
        file_hash = compute_file_hash(file_path)
        count = await backend.add_chunks(chunks, file_hash)

        # Check if file changed
        if await backend.file_changed(file_path, current_hash):
            print("File needs re-indexing")

        # Get pending chunks for Phase 2
        pending = await backend.get_pending_chunks(batch_size=1000)
    """

    TABLE_NAME = "chunks"

    def __init__(self, db_path: Path) -> None:
        """Initialize chunks backend.

        Args:
            db_path: Directory for LanceDB database
        """
        self.db_path = Path(db_path) if isinstance(db_path, str) else db_path
        self._db = None
        self._table = None
        self._append_count = 0  # Track appends for periodic compaction

    async def initialize(self) -> None:
        """Create table if not exists with proper schema.

        Raises:
            DatabaseInitializationError: If initialization fails
        """
        try:
            # Ensure directory exists
            self.db_path.mkdir(parents=True, exist_ok=True)

            # Connect to LanceDB
            self._db = lancedb.connect(str(self.db_path))

            # Check if table exists
            # list_tables() returns a response object with .tables attribute or is iterable
            tables_response = self._db.list_tables()
            table_names = (
                tables_response.tables
                if hasattr(tables_response, "tables")
                else tables_response
            )

            if self.TABLE_NAME in table_names:
                self._table = self._db.open_table(self.TABLE_NAME)

                # Validate schema compatibility (handles atomic rebuild stale dirs + schema upgrades)
                existing_fields = set(self._table.schema.names)
                required_fields = {field.name for field in CHUNKS_SCHEMA}

                if not required_fields.issubset(existing_fields):
                    missing = required_fields - existing_fields
                    logger.warning(
                        f"Schema mismatch: missing fields {missing}. "
                        f"Recreating table with current schema."
                    )
                    self._db.drop_table(self.TABLE_NAME)
                    self._table = None
                    logger.debug("Table dropped, will be recreated on first write")
                else:
                    logger.debug(f"Opened existing chunks table at {self.db_path}")
            else:
                # Table will be created on first add_chunks with schema
                self._table = None
                logger.debug("Chunks table will be created on first write")

        except Exception as e:
            logger.error(f"Failed to initialize chunks backend: {e}")
            raise DatabaseInitializationError(
                f"Chunks backend initialization failed: {e}"
            ) from e

    async def add_chunks(self, chunks: list[dict[str, Any]], file_hash: str) -> int:
        """Add parsed chunks for a file.

        All chunks from the same file should have the same file_hash
        to enable change detection and incremental updates.

        Args:
            chunks: List of chunk dicts with required fields:
                - chunk_id (str): Unique chunk identifier
                - file_path (str): Relative path to source file
                - content (str): Code content
                - language (str): Programming language
                - start_line (int): Starting line number
                - end_line (int): Ending line number
                - chunk_type (str): function, class, method, etc.
                - name (str): Function/class name
                Optional fields will use defaults if not provided.
            file_hash: SHA-256 hash of source file content

        Returns:
            Number of chunks added

        Raises:
            DatabaseNotInitializedError: If backend not initialized
            DatabaseError: If adding chunks fails
        """
        if self._db is None:
            raise DatabaseNotInitializedError("Chunks backend not initialized")

        if not chunks:
            return 0

        try:
            # Normalize and validate chunks
            normalized_chunks = []
            timestamp = datetime.utcnow().isoformat()

            for chunk in chunks:
                # Required fields
                normalized = {
                    "chunk_id": chunk["chunk_id"],
                    "file_path": chunk["file_path"],
                    "file_hash": file_hash,
                    "content": chunk["content"],
                    "language": chunk["language"],
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "chunk_type": chunk["chunk_type"],
                    "name": chunk["name"],
                }

                # Optional fields with defaults
                normalized["start_char"] = chunk.get("start_char", 0)
                normalized["end_char"] = chunk.get("end_char", 0)
                normalized["parent_name"] = chunk.get("parent_name", "")
                normalized["hierarchy_path"] = chunk.get("hierarchy_path", "")
                normalized["docstring"] = chunk.get("docstring", "")
                normalized["signature"] = chunk.get("signature", "")
                normalized["complexity"] = chunk.get("complexity", 0)
                normalized["token_count"] = chunk.get("token_count", 0)

                # Git blame metadata with defaults
                normalized["last_author"] = chunk.get("last_author", "")
                normalized["last_modified"] = chunk.get("last_modified", "")
                normalized["commit_hash"] = chunk.get("commit_hash", "")

                # Code relationships (for KG) with defaults
                normalized["calls"] = chunk.get("calls", [])
                normalized["imports"] = chunk.get("imports", [])
                normalized["inherits_from"] = chunk.get("inherits_from", [])

                # Phase tracking
                normalized["embedding_status"] = "pending"
                normalized["embedding_batch_id"] = 0
                normalized["created_at"] = timestamp
                normalized["updated_at"] = timestamp
                normalized["error_message"] = ""

                normalized_chunks.append(normalized)

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
                    self._table.add(normalized_chunks, mode="append")
                    logger.debug(
                        f"Opened existing table and added {len(normalized_chunks)} chunks (append mode)"
                    )
                else:
                    # Create table with first batch
                    self._table = self._db.create_table(
                        self.TABLE_NAME, normalized_chunks, schema=CHUNKS_SCHEMA
                    )
                    logger.debug(
                        f"Created chunks table with {len(normalized_chunks)} chunks"
                    )
            else:
                # Append to existing table
                # OPTIMIZATION: Use mode='append' for faster bulk inserts
                # This defers index updates until later, improving write throughput
                self._table.add(normalized_chunks, mode="append")
                logger.debug(
                    f"Added {len(normalized_chunks)} chunks to table (append mode)"
                )

            # Track appends and compact periodically to prevent file descriptor exhaustion
            self._append_count += 1
            if self._append_count % 500 == 0:
                self._compact_table()

            return len(normalized_chunks)

        except Exception as e:
            logger.error(f"Failed to add chunks: {e}")
            raise DatabaseError(f"Failed to add chunks: {e}") from e

    async def add_chunks_batch(self, chunks: list[dict[str, Any]]) -> int:
        """Add parsed chunks from multiple files in a single write.

        Each chunk dict must already include the file_hash field.
        This method is optimized for batch processing where chunks from
        multiple files are accumulated and written once.

        Args:
            chunks: List of chunk dicts with required fields including:
                - chunk_id (str): Unique chunk identifier
                - file_path (str): Relative path to source file
                - file_hash (str): SHA-256 hash of source file
                - content (str): Code content
                - language (str): Programming language
                - start_line (int): Starting line number
                - end_line (int): Ending line number
                - chunk_type (str): function, class, method, etc.
                - name (str): Function/class name
                Optional fields will use defaults if not provided.

        Returns:
            Number of chunks added

        Raises:
            DatabaseNotInitializedError: If backend not initialized
            DatabaseError: If adding chunks fails
        """
        if self._db is None:
            raise DatabaseNotInitializedError("Chunks backend not initialized")

        if not chunks:
            return 0

        try:
            # Normalize and validate chunks
            normalized_chunks = []
            timestamp = datetime.utcnow().isoformat()

            for chunk in chunks:
                # Required fields (including file_hash)
                normalized = {
                    "chunk_id": chunk["chunk_id"],
                    "file_path": chunk["file_path"],
                    "file_hash": chunk["file_hash"],
                    "content": chunk["content"],
                    "language": chunk["language"],
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "chunk_type": chunk["chunk_type"],
                    "name": chunk["name"],
                }

                # Optional fields with defaults
                normalized["start_char"] = chunk.get("start_char", 0)
                normalized["end_char"] = chunk.get("end_char", 0)
                normalized["parent_name"] = chunk.get("parent_name", "")
                normalized["hierarchy_path"] = chunk.get("hierarchy_path", "")
                normalized["docstring"] = chunk.get("docstring", "")
                normalized["signature"] = chunk.get("signature", "")
                normalized["complexity"] = chunk.get("complexity", 0)
                normalized["token_count"] = chunk.get("token_count", 0)

                # Git blame metadata with defaults
                normalized["last_author"] = chunk.get("last_author", "")
                normalized["last_modified"] = chunk.get("last_modified", "")
                normalized["commit_hash"] = chunk.get("commit_hash", "")

                # Code relationships (for KG) with defaults
                normalized["calls"] = chunk.get("calls", [])
                normalized["imports"] = chunk.get("imports", [])
                normalized["inherits_from"] = chunk.get("inherits_from", [])

                # Phase tracking
                normalized["embedding_status"] = "pending"
                normalized["embedding_batch_id"] = 0
                normalized["created_at"] = timestamp
                normalized["updated_at"] = timestamp
                normalized["error_message"] = ""

                normalized_chunks.append(normalized)

            # Create or append to table
            if self._table is None:
                # Check if table exists
                tables_response = self._db.list_tables()
                table_names = (
                    tables_response.tables
                    if hasattr(tables_response, "tables")
                    else tables_response
                )

                if self.TABLE_NAME in table_names:
                    # Table exists, open it
                    self._table = self._db.open_table(self.TABLE_NAME)
                    self._table.add(normalized_chunks, mode="append")
                    logger.debug(
                        f"Opened existing table and added {len(normalized_chunks)} chunks (append mode)"
                    )
                else:
                    # Create table with first batch
                    self._table = self._db.create_table(
                        self.TABLE_NAME, normalized_chunks, schema=CHUNKS_SCHEMA
                    )
                    logger.debug(
                        f"Created chunks table with {len(normalized_chunks)} chunks"
                    )
            else:
                # Append to existing table
                # OPTIMIZATION: Use mode='append' for faster bulk inserts
                # This defers index updates until later, improving write throughput
                self._table.add(normalized_chunks, mode="append")
                logger.debug(
                    f"Added {len(normalized_chunks)} chunks to table (append mode)"
                )

            # Track appends and compact periodically to prevent file descriptor exhaustion
            self._append_count += 1
            if self._append_count % 500 == 0:
                self._compact_table()

            return len(normalized_chunks)

        except Exception as e:
            logger.error(f"Failed to add chunks batch: {e}")
            raise DatabaseError(f"Failed to add chunks batch: {e}") from e

    async def add_chunks_raw(self, chunks: list[dict[str, Any]]) -> int:
        """Add pre-normalized chunks directly without re-normalization.

        OPTIMIZATION: Use this when chunks are already fully normalized to avoid
        redundant dict building. Each chunk must include all required fields:
        chunk_id, file_path, file_hash, content, language, start_line, end_line,
        chunk_type, name, and all optional fields.

        Args:
            chunks: List of pre-normalized chunk dicts

        Returns:
            Number of chunks added
        """
        if self._db is None:
            raise DatabaseNotInitializedError("Chunks backend not initialized")

        if not chunks:
            return 0

        try:
            # Add timestamp and status fields to each chunk
            timestamp = datetime.utcnow().isoformat()
            for chunk in chunks:
                # Ensure git blame fields exist with defaults
                chunk.setdefault("last_author", "")
                chunk.setdefault("last_modified", "")
                chunk.setdefault("commit_hash", "")

                chunk["embedding_status"] = "pending"
                chunk["embedding_batch_id"] = 0
                chunk["created_at"] = timestamp
                chunk["updated_at"] = timestamp
                chunk["error_message"] = ""

            # Create or append to table
            if self._table is None:
                tables_response = self._db.list_tables()
                table_names = (
                    tables_response.tables
                    if hasattr(tables_response, "tables")
                    else tables_response
                )

                if self.TABLE_NAME in table_names:
                    self._table = self._db.open_table(self.TABLE_NAME)
                    self._table.add(chunks, mode="append")
                else:
                    self._table = self._db.create_table(
                        self.TABLE_NAME, chunks, schema=CHUNKS_SCHEMA
                    )
            else:
                self._table.add(chunks, mode="append")

            # Track appends and compact periodically to prevent file descriptor exhaustion
            self._append_count += 1
            if self._append_count % 500 == 0:
                self._compact_table()

            return len(chunks)

        except Exception as e:
            logger.error(f"Failed to add raw chunks: {e}")
            raise DatabaseError(f"Failed to add raw chunks: {e}") from e

    def _compact_table(self) -> None:
        """Compact LanceDB table to merge small fragments.

        LanceDB creates one file per append operation. During large reindexing,
        this accumulates thousands of small files, exhausting file descriptors.
        Periodic compaction merges fragments to reduce file count.

        This is called every 500 appends to keep file count manageable.
        Failures are non-fatal (best-effort optimization).
        """
        if self._table is None:
            return

        try:
            # LanceDB 0.5+ API: compact_files() merges data fragments
            self._table.compact_files()
            logger.debug(
                f"Compacted chunks table after {self._append_count} appends (reduced fragment count)"
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

    async def get_all_indexed_file_hashes(self) -> dict[str, str]:
        """Get all indexed file paths and their hashes in one scan.

        OPTIMIZATION: Load all file_path→hash mappings once for O(1) per-file lookup.
        This replaces 39K per-file get_file_hash() queries (each a full table scan)
        with a single scan. Critical for large projects.

        Returns:
            Dict mapping file_path to file_hash for all indexed files
        """
        if self._table is None:
            logger.warning(
                "Chunks table not initialized — change detection unavailable, "
                "all files will be processed"
            )
            return {}

        try:
            # Single full scan, return just file_path and file_hash columns
            scanner = self._table.to_lance().scanner(columns=["file_path", "file_hash"])
            result = scanner.to_table()

            if len(result) == 0:
                return {}

            # Convert to pandas for groupby (get first hash per file, handles duplicates)
            df = result.to_pandas()
            # Group by file_path and take first hash (all chunks from same file have same hash)
            file_hashes = df.groupby("file_path")["file_hash"].first().to_dict()

            logger.debug(
                f"Loaded {len(file_hashes)} indexed file paths for change detection"
            )
            return file_hashes

        except Exception as e:
            logger.warning(
                f"Failed to load indexed file hashes ({e}) — "
                "all files will be treated as changed"
            )
            return {}

    async def get_file_hash(self, file_path: str) -> str | None:
        """Get stored hash for a file (for change detection).

        Args:
            file_path: Relative path to file

        Returns:
            SHA-256 hash of file content, or None if file not indexed
        """
        if self._table is None:
            return None

        try:
            # OPTIMIZATION: Use LanceDB scanner for O(1) filtered query instead of to_pandas()
            # This avoids loading entire table into memory (1.2GB+ on large projects)
            scanner = self._table.to_lance().scanner(
                filter=f"file_path = '{file_path}'",
                columns=["file_hash"],  # Only fetch the column we need
                limit=1,
            )

            # Get the first matching row (no count_rows() - incompatible with selected columns)
            result = scanner.to_table()
            if len(result) == 0:
                return None

            return result["file_hash"][0].as_py()

        except Exception as e:
            logger.warning(f"Failed to get file hash for {file_path}: {e}")
            return None

    async def file_changed(self, file_path: str, current_hash: str) -> bool:
        """Check if file has changed since last index.

        Args:
            file_path: Relative path to file
            current_hash: Current SHA-256 hash of file content

        Returns:
            True if file has changed (or not indexed), False if unchanged
        """
        stored_hash = await self.get_file_hash(file_path)

        if stored_hash is None:
            # File not indexed yet, so it "changed"
            return True

        return stored_hash != current_hash

    async def get_pending_chunks(
        self, batch_size: int = 1000, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Get chunks that need embedding (status=pending).

        Args:
            batch_size: Maximum number of chunks to return
            offset: Offset for pagination

        Returns:
            List of chunk dictionaries with all fields
        """
        if self._table is None:
            return []

        try:
            # OPTIMIZATION: Use LanceDB scanner for O(1) filtered query instead of to_pandas()
            scanner = self._table.to_lance().scanner(
                filter="embedding_status = 'pending'", limit=batch_size, offset=offset
            )

            # Convert to pandas for dict conversion (only selected rows, not entire table)
            result = scanner.to_table()
            if len(result) == 0:
                return []

            # Convert PyArrow table to list of dicts
            chunks = result.to_pylist()
            logger.debug(f"Retrieved {len(chunks)} pending chunks")
            return chunks

        except Exception as e:
            logger.error(f"Failed to get pending chunks: {e}")
            return []

    async def mark_chunks_processing(self, chunk_ids: list[str], batch_id: int) -> None:
        """Mark chunks as being processed (for resume logic).

        Args:
            chunk_ids: List of chunk IDs to mark
            batch_id: Batch identifier for tracking

        Raises:
            DatabaseError: If update fails
        """
        if self._table is None or not chunk_ids:
            return

        try:
            # Build filter for chunk_ids
            ids_str = "', '".join(chunk_ids)
            filter_expr = f"chunk_id IN ('{ids_str}')"

            # OPTIMIZATION: Use LanceDB scanner for O(1) filtered query instead of to_pandas()
            scanner = self._table.to_lance().scanner(filter=filter_expr)

            # Convert to PyArrow table (only selected rows, not entire table)
            result = scanner.to_table()
            if len(result) == 0:
                return

            # Convert to pandas for manipulation
            df = result.to_pandas()

            # Update fields
            df["embedding_status"] = "processing"
            df["embedding_batch_id"] = batch_id
            df["updated_at"] = datetime.utcnow().isoformat()

            # Delete old rows
            self._table.delete(filter_expr)

            # Add updated rows
            updated_chunks = df.to_dict("records")
            self._table.add(updated_chunks)

            logger.debug(f"Marked {len(chunk_ids)} chunks as processing")

        except Exception as e:
            logger.error(f"Failed to mark chunks processing: {e}")
            raise DatabaseError(f"Failed to update chunk status: {e}") from e

    async def mark_chunks_complete(self, chunk_ids: list[str]) -> None:
        """Mark chunks as successfully embedded.

        Args:
            chunk_ids: List of chunk IDs to mark

        Raises:
            DatabaseError: If update fails
        """
        if self._table is None or not chunk_ids:
            return

        try:
            # Build filter
            ids_str = "', '".join(chunk_ids)
            filter_expr = f"chunk_id IN ('{ids_str}')"

            # OPTIMIZATION: Use LanceDB scanner for O(1) filtered query instead of to_pandas()
            scanner = self._table.to_lance().scanner(filter=filter_expr)

            # Convert to PyArrow table (only selected rows, not entire table)
            result = scanner.to_table()
            if len(result) == 0:
                return

            # Convert to pandas for manipulation
            df = result.to_pandas()

            df["embedding_status"] = "complete"
            df["updated_at"] = datetime.utcnow().isoformat()
            df["error_message"] = ""

            self._table.delete(filter_expr)
            self._table.add(df.to_dict("records"))

            logger.debug(f"Marked {len(chunk_ids)} chunks as complete")

        except Exception as e:
            logger.error(f"Failed to mark chunks complete: {e}")
            raise DatabaseError(f"Failed to update chunk status: {e}") from e

    async def mark_chunks_pending(self, chunk_ids: list[str]) -> None:
        """Mark chunks as pending (revert from processing status).

        Used when batch needs to be retried due to memory pressure or errors.

        Args:
            chunk_ids: List of chunk IDs to mark as pending

        Raises:
            DatabaseError: If update fails
        """
        if self._table is None or not chunk_ids:
            return

        try:
            # Build filter
            ids_str = "', '".join(chunk_ids)
            filter_expr = f"chunk_id IN ('{ids_str}')"

            # OPTIMIZATION: Use LanceDB scanner for O(1) filtered query instead of to_pandas()
            scanner = self._table.to_lance().scanner(filter=filter_expr)

            # Convert to PyArrow table (only selected rows, not entire table)
            result = scanner.to_table()
            if len(result) == 0:
                return

            # Convert to pandas for manipulation
            df = result.to_pandas()

            df["embedding_status"] = "pending"
            df["updated_at"] = datetime.utcnow().isoformat()
            df["batch_id"] = 0  # Reset batch assignment
            df["error_message"] = ""

            self._table.delete(filter_expr)
            self._table.add(df.to_dict("records"))

            logger.debug(
                f"Marked {len(chunk_ids)} chunks as pending (reverted from processing)"
            )

        except Exception as e:
            logger.error(f"Failed to mark chunks pending: {e}")
            raise DatabaseError(f"Failed to update chunk status: {e}") from e

    async def mark_chunks_error(self, chunk_ids: list[str], error: str) -> None:
        """Mark chunks as failed embedding.

        Args:
            chunk_ids: List of chunk IDs to mark
            error: Error message to store

        Raises:
            DatabaseError: If update fails
        """
        if self._table is None or not chunk_ids:
            return

        try:
            # Build filter
            ids_str = "', '".join(chunk_ids)
            filter_expr = f"chunk_id IN ('{ids_str}')"

            # OPTIMIZATION: Use LanceDB scanner for O(1) filtered query instead of to_pandas()
            scanner = self._table.to_lance().scanner(filter=filter_expr)

            # Convert to PyArrow table (only selected rows, not entire table)
            result = scanner.to_table()
            if len(result) == 0:
                return

            # Convert to pandas for manipulation
            df = result.to_pandas()

            df["embedding_status"] = "error"
            df["updated_at"] = datetime.utcnow().isoformat()
            df["error_message"] = error[:500]  # Truncate long errors

            self._table.delete(filter_expr)
            self._table.add(df.to_dict("records"))

            logger.debug(f"Marked {len(chunk_ids)} chunks as error")

        except Exception as e:
            logger.error(f"Failed to mark chunks error: {e}")
            raise DatabaseError(f"Failed to update chunk status: {e}") from e

    async def delete_file_chunks(self, file_path: str) -> int:
        """Delete all chunks for a file (for re-indexing).

        Args:
            file_path: Relative path to file

        Returns:
            Number of chunks deleted

        Raises:
            DatabaseError: If deletion fails
        """
        if self._table is None:
            return 0

        try:
            # Count chunks before deletion to return accurate count
            filter_expr = f"file_path = '{file_path}'"
            df = (
                self._table.search()
                .where(filter_expr, prefilter=True)
                .limit(10000)
                .to_pandas()
            )
            count = len(df)

            if count > 0:
                # Delete matching rows
                self._table.delete(filter_expr)
                logger.debug(f"Deleted {count} chunks for file: {file_path}")

            return count

        except Exception as e:
            # Handle LanceDB "Not found" errors gracefully (file not in index)
            error_msg = str(e).lower()
            if "not found" in error_msg:
                logger.debug(f"No chunks to delete for {file_path} (not in index)")
                return 0
            logger.error(f"Failed to delete chunks for {file_path}: {e}")
            raise DatabaseError(f"Failed to delete chunks: {e}") from e

    async def delete_files_batch(self, file_paths: list[str]) -> int:
        """Delete chunks for multiple files in a single operation.

        This is more efficient than calling delete_file_chunks() for each file,
        especially on large databases where to_pandas() would load the entire
        table into memory for each call.

        Args:
            file_paths: List of relative file paths to delete

        Returns:
            Total number of chunks deleted

        Raises:
            DatabaseError: If deletion fails
        """
        if self._table is None or not file_paths:
            return 0

        try:
            # Build filter for all files at once: file_path = 'a' OR file_path = 'b' OR ...
            filter_clauses = [f"file_path = '{fp}'" for fp in file_paths]
            filter_expr = " OR ".join(filter_clauses)

            # Single delete operation for all files (delete is idempotent)
            self._table.delete(filter_expr)

            logger.debug(f"Deleted chunks for {len(file_paths)} files in batch")
            return len(file_paths)  # Return count of files processed

        except Exception as e:
            # Handle LanceDB "Not found" errors gracefully
            error_msg = str(e).lower()
            if "not found" in error_msg:
                logger.debug("No chunks to delete for batch (not in index)")
                return 0
            logger.error(f"Failed to delete chunks batch: {e}")
            raise DatabaseError(f"Failed to delete chunks batch: {e}") from e

    async def get_stats(self) -> dict[str, Any]:
        """Get chunk statistics by status.

        Returns:
            Dictionary with counts by status and total metrics:
            {
                "total": 1000,
                "pending": 500,
                "processing": 50,
                "complete": 400,
                "error": 50,
                "files": 100,
                "languages": {"python": 50, "javascript": 30, ...}
            }
        """
        if self._table is None:
            return {
                "total": 0,
                "pending": 0,
                "processing": 0,
                "complete": 0,
                "error": 0,
                "files": 0,
                "languages": {},
            }

        try:
            # OPTIMIZATION: Use LanceDB scanner for efficient aggregations
            # However, for stats we need to read all rows to count by status/language
            # Since this is a reporting operation (not per-file), to_pandas() is acceptable
            # But we can optimize by reading only needed columns
            scanner = self._table.to_lance().scanner(
                columns=["embedding_status", "file_path", "language"]
            )
            result = scanner.to_table()
            df = result.to_pandas()

            # Count by status
            status_counts = df["embedding_status"].value_counts().to_dict()

            # Count unique files
            file_count = df["file_path"].nunique()

            # Count by language
            language_counts = df["language"].value_counts().to_dict()

            return {
                "total": len(df),
                "pending": status_counts.get("pending", 0),
                "processing": status_counts.get("processing", 0),
                "complete": status_counts.get("complete", 0),
                "error": status_counts.get("error", 0),
                "files": file_count,
                "languages": language_counts,
            }

        except Exception as e:
            logger.error(f"Failed to get chunk stats: {e}")
            return {
                "total": 0,
                "pending": 0,
                "processing": 0,
                "complete": 0,
                "error": 0,
                "files": 0,
                "languages": {},
            }

    async def cleanup_stale_processing(self, older_than_minutes: int = 30) -> int:
        """Reset chunks stuck in 'processing' state.

        This handles crash recovery where embedding process was interrupted
        and chunks are stuck in "processing" status.

        Args:
            older_than_minutes: Reset chunks older than this many minutes

        Returns:
            Number of chunks reset to pending

        Raises:
            DatabaseError: If update fails
        """
        if self._table is None:
            return 0

        try:
            from datetime import timedelta

            # Calculate cutoff time
            cutoff = datetime.utcnow() - timedelta(minutes=older_than_minutes)
            cutoff_str = cutoff.isoformat()

            # OPTIMIZATION: Use LanceDB scanner with filter for processing status
            # Note: LanceDB doesn't support complex filters like "updated_at < cutoff"
            # So we need to filter by status first, then check timestamps in memory
            scanner = self._table.to_lance().scanner(
                filter="embedding_status = 'processing'"
            )
            result = scanner.to_table()

            if len(result) == 0:
                return 0

            df = result.to_pandas()

            # Filter by timestamp (in-memory filtering needed for date comparison)
            stale_df = df[df["updated_at"] < cutoff_str]

            if stale_df.empty:
                return 0

            # Get chunk IDs to reset
            chunk_ids = stale_df["chunk_id"].tolist()

            # Reset to pending
            stale_df["embedding_status"] = "pending"
            stale_df["embedding_batch_id"] = 0
            stale_df["updated_at"] = datetime.utcnow().isoformat()

            # Delete old and add updated
            ids_str = "', '".join(chunk_ids)
            filter_expr = f"chunk_id IN ('{ids_str}')"
            self._table.delete(filter_expr)
            self._table.add(stale_df.to_dict("records"))

            logger.info(f"Reset {len(chunk_ids)} stale processing chunks")
            return len(chunk_ids)

        except Exception as e:
            logger.error(f"Failed to cleanup stale chunks: {e}")
            raise DatabaseError(f"Failed to cleanup stale chunks: {e}") from e

    async def count_chunks(self) -> int:
        """Get total count of chunks in database.

        Returns:
            Total number of chunks

        Raises:
            DatabaseNotInitializedError: If backend not initialized
        """
        if self._table is None:
            raise DatabaseNotInitializedError("Chunks backend not initialized")

        try:
            return self._table.count_rows()
        except Exception as e:
            logger.error(f"Failed to count chunks: {e}")
            raise DatabaseError(f"Failed to count chunks: {e}") from e

    async def count_pending_chunks(self) -> int:
        """Get count of chunks with embedding_status='pending'.

        Returns:
            Number of pending chunks

        Raises:
            DatabaseNotInitializedError: If backend not initialized
        """
        if self._table is None:
            raise DatabaseNotInitializedError("Chunks backend not initialized")

        try:
            # Use LanceDB scanner with filter to count only pending chunks
            scanner = self._table.to_lance().scanner(
                filter="embedding_status = 'pending'"
            )
            result = scanner.to_table()
            count = len(result)
            logger.debug(f"Found {count} pending chunks")
            return count
        except Exception as e:
            logger.error(f"Failed to count pending chunks: {e}")
            raise DatabaseError(f"Failed to count pending chunks: {e}") from e

    async def reset_all_to_pending(self) -> int:
        """Reset all chunks to pending status for re-embedding.

        This is used when changing embedding models or re-embedding with
        a different model. It resets embedding_status to "pending" for
        all chunks regardless of current status.

        Returns:
            Number of chunks reset

        Raises:
            DatabaseNotInitializedError: If backend not initialized
            DatabaseError: If reset fails
        """
        if self._table is None:
            raise DatabaseNotInitializedError("Chunks backend not initialized")

        try:
            # OPTIMIZATION: Use LanceDB scanner to get all chunks efficiently
            scanner = self._table.to_lance().scanner()
            result = scanner.to_table()

            if len(result) == 0:
                logger.info("No chunks to reset")
                return 0

            # Convert to pandas for manipulation
            df = result.to_pandas()
            chunk_count = len(df)

            # Update status fields (only columns that exist in the schema)
            df["embedding_status"] = "pending"
            df["updated_at"] = datetime.utcnow().isoformat()
            if "embedding_batch_id" in df.columns:
                df["embedding_batch_id"] = 0  # Reset batch assignment
            if "error_message" in df.columns:
                df["error_message"] = ""  # Clear any error messages

            # Delete all rows and re-add with updated status
            # This is more efficient than updating each row individually
            self._table.delete("chunk_id IS NOT NULL")  # Delete all rows
            self._table.add(df.to_dict("records"))

            logger.info(
                f"Reset {chunk_count:,} chunks to pending status for re-embedding"
            )
            return chunk_count

        except Exception as e:
            logger.error(f"Failed to reset chunks to pending: {e}")
            raise DatabaseError(f"Failed to reset chunks: {e}") from e

    async def close(self) -> None:
        """Close database connections.

        LanceDB doesn't require explicit closing, but we set references to None
        for consistency with other backends.
        """
        self._table = None
        self._db = None
        logger.debug("Chunks backend closed")

    async def __aenter__(self) -> "ChunksBackend":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
