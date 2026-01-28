"""Index metadata management for tracking file modifications and versions."""

import json
import os
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

    def load(self) -> dict[str, float]:
        """Load file modification times from metadata file.

        Returns:
            Dictionary mapping file paths to modification times
        """
        if not self._metadata_file.exists():
            return {}

        try:
            with open(self._metadata_file) as f:
                data = json.load(f)
                # Handle legacy format (just file_mtimes dict) and new format
                if "file_mtimes" in data:
                    return data["file_mtimes"]
                else:
                    # Legacy format - just return as-is
                    return data
        except Exception as e:
            logger.warning(f"Failed to load index metadata: {e}")
            return {}

    def save(self, metadata: dict[str, float]) -> None:
        """Save file modification times to metadata file.

        Args:
            metadata: Dictionary mapping file paths to modification times
        """
        try:
            # Ensure directory exists
            self._metadata_file.parent.mkdir(parents=True, exist_ok=True)

            # New metadata format with version tracking
            data = {
                "index_version": __version__,
                "indexed_at": datetime.now(UTC).isoformat(),
                "file_mtimes": metadata,
            }

            with open(self._metadata_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save index metadata: {e}")

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
