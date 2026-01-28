"""Statistics collection for ChromaDB vector database."""

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger

from .models import IndexStats


class StatisticsCollector:
    """Collects and aggregates statistics from ChromaDB collections.

    Handles large databases efficiently by:
    - Processing data in chunks to avoid memory issues
    - Detecting and skipping stats for very large databases (>500MB)
    - Yielding to event loop to prevent blocking
    """

    def __init__(
        self,
        persist_directory: Path,
        batch_size: int = 1000,
        large_db_threshold_mb: float = 500.0,
    ) -> None:
        """Initialize statistics collector.

        Args:
            persist_directory: Path to ChromaDB persistence directory
            batch_size: Number of records to process per batch
            large_db_threshold_mb: Database size threshold for skipping stats
        """
        self.persist_directory = persist_directory
        self.batch_size = batch_size
        self.large_db_threshold_mb = large_db_threshold_mb

    async def collect_stats(
        self, collection: Any, skip_stats: bool = False
    ) -> IndexStats:
        """Collect database statistics with optimized chunked queries.

        Args:
            collection: ChromaDB collection instance
            skip_stats: If True, skip detailed statistics collection

        Returns:
            Index statistics
        """
        # Check database size and auto-enable skip_stats for large databases
        db_size_mb, db_size_bytes = self._get_database_size()

        if db_size_mb > self.large_db_threshold_mb and not skip_stats:
            logger.warning(
                f"Large database detected ({db_size_mb:.1f} MB). "
                "Skipping detailed statistics to prevent potential crashes."
            )
            skip_stats = True

        # Return minimal stats if skipping
        if skip_stats:
            return IndexStats(
                total_files=0,
                total_chunks="Large DB (count skipped for safety)",
                languages={},
                file_types={},
                index_size_mb=db_size_mb,
                last_updated="Skipped (large database)",
                embedding_model="unknown",
                database_size_bytes=db_size_bytes,
            )

        try:
            # Get total count
            count = collection.count()

            if count == 0:
                return IndexStats(
                    total_files=0,
                    total_chunks=0,
                    languages={},
                    file_types={},
                    index_size_mb=0.0,
                    last_updated="N/A",
                    embedding_model="unknown",
                )

            # Process in chunks
            files = set()
            language_counts: dict[str, int] = {}
            file_type_counts: dict[str, int] = {}

            offset = 0
            while offset < count:
                # Fetch batch
                batch_size = min(self.batch_size, count - offset)
                logger.debug(
                    f"Processing database stats: batch {offset // self.batch_size + 1}, "
                    f"{offset}-{offset + batch_size} of {count} chunks"
                )

                results = collection.get(
                    include=["metadatas"],
                    limit=batch_size,
                    offset=offset,
                )

                # Process batch metadata
                self._process_batch_metadata(
                    results.get("metadatas", []),
                    files,
                    language_counts,
                    file_type_counts,
                )

                offset += batch_size

                # Yield to event loop periodically to prevent blocking
                await asyncio.sleep(0)

            # Estimate index size (rough approximation: ~1KB per chunk)
            index_size_mb = count * 0.001

            return IndexStats(
                total_files=len(files),
                total_chunks=count,
                languages=language_counts,
                file_types=file_type_counts,
                index_size_mb=index_size_mb,
                last_updated="unknown",
                embedding_model="unknown",
            )

        except Exception as e:
            logger.error(f"Failed to collect database statistics: {e}")
            # Return empty stats instead of raising
            return IndexStats(
                total_files=0,
                total_chunks=0,
                languages={},
                file_types={},
                index_size_mb=0.0,
                last_updated="error",
                embedding_model="unknown",
            )

    def _get_database_size(self) -> tuple[float, int]:
        """Get database file size.

        Returns:
            Tuple of (size in MB, size in bytes)
        """
        chroma_db_path = self.persist_directory / "chroma.sqlite3"
        db_size_mb = 0.0
        db_size_bytes = 0

        if chroma_db_path.exists():
            db_size_bytes = chroma_db_path.stat().st_size
            db_size_mb = db_size_bytes / (1024 * 1024)

        return db_size_mb, db_size_bytes

    def _process_batch_metadata(
        self,
        metadatas: list[dict[str, Any]],
        files: set[str],
        language_counts: dict[str, int],
        file_type_counts: dict[str, int],
    ) -> None:
        """Process a batch of metadata to aggregate statistics.

        Args:
            metadatas: List of metadata dictionaries
            files: Set of unique file paths (mutated)
            language_counts: Dictionary of language counts (mutated)
            file_type_counts: Dictionary of file type counts (mutated)
        """
        for metadata in metadatas:
            # Language stats
            lang = metadata.get("language", "unknown")
            language_counts[lang] = language_counts.get(lang, 0) + 1

            # File stats
            file_path = metadata.get("file_path", "")
            if file_path:
                files.add(file_path)
                ext = Path(file_path).suffix or "no_extension"
                file_type_counts[ext] = file_type_counts.get(ext, 0) + 1
