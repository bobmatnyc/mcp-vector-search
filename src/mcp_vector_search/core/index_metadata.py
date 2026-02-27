"""Index metadata management for tracking file modifications and versions."""

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from packaging import version

from .. import __version__


class IndexMetadata:
    """Manages index metadata including file modification times and version tracking.

    This class encapsulates all logic related to tracking which files have been
    indexed, when they were modified, and what version of the indexer created them.
    """

    def __init__(self, project_root: Path) -> None:
        """Initialize index metadata manager.

        Args:
            project_root: Project root directory
        """
        self.project_root = project_root
        self._metadata_file = (
            project_root / ".mcp-vector-search" / "index_metadata.json"
        )

    def load(
        self, progress_callback: Callable[[int, int], None] | None = None
    ) -> dict[str, float]:
        """Load file modification times from metadata file.

        Args:
            progress_callback: Optional callback(bytes_read, total_bytes) for progress tracking

        Returns:
            Dictionary mapping file paths to modification times
        """
        if not self._metadata_file.exists():
            return {}

        try:
            file_size = self._metadata_file.stat().st_size

            # Read file with progress tracking
            with open(self._metadata_file, "rb") as f:
                chunks = []
                bytes_read = 0
                chunk_size = 8192  # 8KB chunks

                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break

                    chunks.append(chunk)
                    bytes_read += len(chunk)

                    if progress_callback:
                        progress_callback(bytes_read, file_size)

                content = b"".join(chunks)

            # Parse JSON
            data = json.loads(content.decode("utf-8"))

            # Handle legacy format (just file_mtimes dict) and new format
            if "file_mtimes" in data:
                return data["file_mtimes"]
            else:
                # Legacy format - just return as-is
                return data
        except Exception as e:
            logger.warning(f"Failed to load index metadata: {e}")
            return {}

    def save(
        self,
        metadata: dict[str, float],
        embedding_model: str | None = None,
        embedding_dimensions: int | None = None,
    ) -> None:
        """Save file modification times to metadata file.

        Args:
            metadata: Dictionary mapping file paths to modification times
            embedding_model: Name of the embedding model used (e.g.,
                "sentence-transformers/all-MiniLM-L6-v2").  When provided,
                the field is written to the metadata file so callers can
                detect model changes without opening the vector database.
            embedding_dimensions: Vector dimension of the embedding model
                (e.g. 384 for MiniLM, 768 for GraphCodeBERT).
        """
        try:
            # Ensure directory exists
            self._metadata_file.parent.mkdir(parents=True, exist_ok=True)

            # Preserve existing embedding_model / dimensions when not supplied
            existing = self._read_raw()
            resolved_model = embedding_model or existing.get("embedding_model")
            resolved_dims = (
                embedding_dimensions
                if embedding_dimensions is not None
                else existing.get("embedding_dimensions")
            )

            # New metadata format with version and embedding tracking
            now = datetime.now(UTC).isoformat()
            data: dict[str, object] = {
                "index_version": __version__,
                "embedding_model": resolved_model,
                "embedding_dimensions": resolved_dims,
                # created_at is set once on first write and never overwritten
                "created_at": existing.get("created_at", now),
                "updated_at": now,
                "file_mtimes": metadata,
            }

            with open(self._metadata_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save index metadata: {e}")

    def _read_raw(self) -> dict:
        """Read the raw metadata JSON dict (empty dict on any error)."""
        if not self._metadata_file.exists():
            return {}
        try:
            with open(self._metadata_file) as f:
                return json.load(f)
        except Exception:
            return {}

    def get_index_metadata(self) -> dict[str, object]:
        """Return the full index metadata dict.

        Returns a standardised dict with the following keys (all optional
        fields default to ``None`` if not yet recorded):

        * ``embedding_model`` – model name, e.g.
          ``"sentence-transformers/all-MiniLM-L6-v2"``
        * ``embedding_dimensions`` – integer vector dimension
        * ``index_version`` – tool version that built the index
        * ``created_at`` – ISO-8601 UTC timestamp of first index build
        * ``updated_at`` – ISO-8601 UTC timestamp of last index update

        Returns:
            dict with index metadata fields
        """
        raw = self._read_raw()
        return {
            "embedding_model": raw.get("embedding_model"),
            "embedding_dimensions": raw.get("embedding_dimensions"),
            "index_version": raw.get("index_version"),
            "created_at": raw.get("created_at"),
            "updated_at": raw.get("updated_at"),
        }

    def needs_reindexing(self, file_path: Path, metadata: dict[str, float]) -> bool:
        """Check if a file needs reindexing based on modification time.

        Args:
            file_path: Path to the file
            metadata: Current metadata dictionary

        Returns:
            True if file needs reindexing
        """
        try:
            current_mtime = os.path.getmtime(file_path)
            stored_mtime = metadata.get(str(file_path), 0)
            return current_mtime > stored_mtime
        except OSError:
            # File doesn't exist or can't be accessed
            return False

    def get_index_version(self) -> str | None:
        """Get the version of the tool that created the current index.

        Returns:
            Version string or None if not available
        """
        if not self._metadata_file.exists():
            return None

        try:
            with open(self._metadata_file) as f:
                data = json.load(f)
                return data.get("index_version")
        except Exception as e:
            logger.warning(f"Failed to read index version: {e}")
            return None

    def needs_reindex_for_version(self) -> bool:
        """Check if reindex is needed due to version upgrade.

        Returns:
            True if reindex is needed for version compatibility
        """
        index_version = self.get_index_version()

        if not index_version:
            # No version recorded - this is either a new index or legacy format
            # Reindex to establish version tracking
            return True

        try:
            current = version.parse(__version__)
            indexed = version.parse(index_version)

            # Reindex on major or minor version change
            # Patch versions (0.5.1 -> 0.5.2) don't require reindex
            needs_reindex = (
                current.major != indexed.major or current.minor != indexed.minor
            )

            if needs_reindex:
                logger.info(
                    f"Version upgrade detected: {index_version} -> {__version__} "
                    f"(reindex recommended)"
                )

            return needs_reindex

        except Exception as e:
            logger.warning(f"Failed to compare versions: {e}")
            # If we can't parse versions, be safe and reindex
            return True

    def write_indexing_run_header(self) -> None:
        """Write version and timestamp header to error log at start of indexing run."""
        try:
            error_log_path = (
                self.project_root / ".mcp-vector-search" / "indexing_errors.log"
            )
            error_log_path.parent.mkdir(parents=True, exist_ok=True)

            with open(error_log_path, "a", encoding="utf-8") as f:
                timestamp = datetime.now(UTC).isoformat()
                separator = "=" * 80
                f.write(f"\n{separator}\n")
                f.write(
                    f"[{timestamp}] Indexing run started - mcp-vector-search v{__version__}\n"
                )
                f.write(f"{separator}\n")
        except Exception as e:
            logger.debug(f"Failed to write indexing run header: {e}")

    def log_indexing_error(self, error_msg: str) -> None:
        """Log an indexing error to the error log file.

        Args:
            error_msg: Error message to log
        """
        try:
            error_log_path = (
                self.project_root / ".mcp-vector-search" / "indexing_errors.log"
            )
            with open(error_log_path, "a", encoding="utf-8") as f:
                timestamp = datetime.now().isoformat()
                f.write(f"[{timestamp}] {error_msg}\n")
        except Exception as e:
            logger.debug(f"Failed to write error log: {e}")

    def cleanup_stale_entries(self, valid_files: set[str]) -> int:
        """Remove metadata entries for files that no longer exist.

        Args:
            valid_files: Set of relative file paths that currently exist

        Returns:
            Number of stale entries removed
        """
        metadata = self.load()
        original_count = len(metadata)

        cleaned = {
            path: mtime for path, mtime in metadata.items() if path in valid_files
        }

        removed = original_count - len(cleaned)
        if removed > 0:
            self.save(cleaned)
            logger.info(f"Cleaned up {removed} stale metadata entries")

        return removed
