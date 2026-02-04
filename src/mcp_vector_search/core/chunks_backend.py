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
            if self.TABLE_NAME in self._db.list_tables():
                self._table = self._db.open_table(self.TABLE_NAME)
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

                # Phase tracking
                normalized["embedding_status"] = "pending"
                normalized["embedding_batch_id"] = 0
                normalized["created_at"] = timestamp
                normalized["updated_at"] = timestamp
                normalized["error_message"] = ""

                normalized_chunks.append(normalized)

            # Create or append to table
            if self._table is None:
                # Create table with first batch
                self._table = self._db.create_table(
                    self.TABLE_NAME, normalized_chunks, schema=CHUNKS_SCHEMA
                )
                logger.debug(
                    f"Created chunks table with {len(normalized_chunks)} chunks"
                )
            else:
                # Append to existing table
                self._table.add(normalized_chunks)
                logger.debug(f"Added {len(normalized_chunks)} chunks to table")

            return len(normalized_chunks)

        except Exception as e:
            logger.error(f"Failed to add chunks: {e}")
            raise DatabaseError(f"Failed to add chunks: {e}") from e

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
            # Query for any chunk from this file
            df = self._table.to_pandas().query(f"file_path == '{file_path}'").head(1)

            if df.empty:
                return None

            return df.iloc[0]["file_hash"]

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
            # Query pending chunks with pagination
            df = (
                self._table.to_pandas()
                .query("embedding_status == 'pending'")
                .iloc[offset : offset + batch_size]
            )

            # Convert to list of dicts
            chunks = df.to_dict("records")
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

            # LanceDB doesn't have a direct update method, so we need to:
            # 1. Read matching rows
            # 2. Update them in memory
            # 3. Delete old rows
            # 4. Add updated rows

            df = self._table.to_pandas().query(f"chunk_id in {chunk_ids}")

            if df.empty:
                return

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

            # Read, update, delete, add pattern
            df = self._table.to_pandas().query(f"chunk_id in {chunk_ids}")

            if df.empty:
                return

            df["embedding_status"] = "complete"
            df["updated_at"] = datetime.utcnow().isoformat()
            df["error_message"] = ""

            self._table.delete(filter_expr)
            self._table.add(df.to_dict("records"))

            logger.debug(f"Marked {len(chunk_ids)} chunks as complete")

        except Exception as e:
            logger.error(f"Failed to mark chunks complete: {e}")
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

            # Read, update, delete, add pattern
            df = self._table.to_pandas().query(f"chunk_id in {chunk_ids}")

            if df.empty:
                return

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
            # Count chunks before deletion
            df = self._table.to_pandas().query(f"file_path == '{file_path}'")
            count = len(df)

            if count == 0:
                return 0

            # Delete matching rows
            self._table.delete(f"file_path = '{file_path}'")

            logger.debug(f"Deleted {count} chunks for file: {file_path}")
            return count

        except Exception as e:
            logger.error(f"Failed to delete chunks for {file_path}: {e}")
            raise DatabaseError(f"Failed to delete chunks: {e}") from e

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
            df = self._table.to_pandas()

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

            # Find stale processing chunks
            df = self._table.to_pandas()
            stale_df = df[
                (df["embedding_status"] == "processing")
                & (df["updated_at"] < cutoff_str)
            ]

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
