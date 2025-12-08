"""Automatic indexing strategies without daemon processes."""

import asyncio
import os
import time
from pathlib import Path

from loguru import logger

from .database import VectorDatabase
from .indexer import SemanticIndexer


class AutoIndexer:
    """Handles automatic reindexing without daemon processes."""

    def __init__(
        self,
        indexer: SemanticIndexer,
        database: VectorDatabase,
        auto_reindex_threshold: int = 5,  # Max files to auto-reindex
        staleness_threshold: float = 300.0,  # 5 minutes
    ):
        """Initialize auto-indexer.

        Args:
            indexer: Semantic indexer instance
            database: Vector database instance
            auto_reindex_threshold: Max files to auto-reindex without asking
            staleness_threshold: Time in seconds before considering index stale
        """
        self.indexer = indexer
        self.database = database
        self.auto_reindex_threshold = auto_reindex_threshold
        self.staleness_threshold = staleness_threshold
        self._last_check_time = 0.0
        self._check_interval = 30.0  # Check at most every 30 seconds

    async def check_and_reindex_if_needed(
        self, force_check: bool = False, interactive: bool = True
    ) -> tuple[bool, int]:
        """Check if reindexing is needed and optionally perform it.

        Args:
            force_check: Skip time-based check throttling
            interactive: Whether to prompt user for large reindexes

        Returns:
            Tuple of (reindexed, files_updated)
        """
        current_time = time.time()

        # Throttle checks to avoid excessive filesystem scanning
        if (
            not force_check
            and (current_time - self._last_check_time) < self._check_interval
        ):
            return False, 0

        self._last_check_time = current_time

        try:
            # Get files that need reindexing
            stale_files = await self._find_stale_files()

            if not stale_files:
                logger.debug("No files need reindexing")
                return False, 0

            logger.info(f"Found {len(stale_files)} files that need reindexing")

            # Decide whether to auto-reindex
            should_reindex = await self._should_auto_reindex(stale_files, interactive)

            if should_reindex:
                updated_count = await self._reindex_files(stale_files)
                logger.info(f"Auto-reindexed {updated_count} files")
                return True, updated_count
            else:
                logger.info("Skipping auto-reindex (user choice or too many files)")
                return False, len(stale_files)

        except Exception as e:
            logger.error(f"Auto-reindex check failed: {e}")
            return False, 0

    async def _find_stale_files(self) -> list[Path]:
        """Find files that need reindexing."""
        try:
            # Load existing metadata
            metadata = self.indexer._load_index_metadata()

            # Find all indexable files
            all_files = self.indexer._find_indexable_files()

            stale_files = []
            for file_path in all_files:
                if self.indexer._needs_reindexing(file_path, metadata):
                    stale_files.append(file_path)

            return stale_files

        except Exception as e:
            logger.error(f"Failed to find stale files: {e}")
            return []

    async def _should_auto_reindex(
        self, stale_files: list[Path], interactive: bool
    ) -> bool:
        """Determine if we should automatically reindex."""
        file_count = len(stale_files)

        # Always auto-reindex small numbers of files
        if file_count <= self.auto_reindex_threshold:
            logger.debug(f"Auto-reindexing {file_count} files (under threshold)")
            return True

        # For larger numbers, check if interactive mode is enabled
        if not interactive:
            logger.debug(
                f"Skipping auto-reindex of {file_count} files (non-interactive)"
            )
            return False

        # In interactive mode, we could prompt the user
        # For now, we'll be conservative and skip large reindexes
        logger.info(f"Skipping auto-reindex of {file_count} files (over threshold)")
        logger.info("Run 'mcp-vector-search index' to update manually")
        return False

    async def _reindex_files(self, files: list[Path]) -> int:
        """Reindex the specified files."""
        updated_count = 0

        try:
            # Process files in small batches to avoid overwhelming the system
            batch_size = min(self.auto_reindex_threshold, 10)

            for i in range(0, len(files), batch_size):
                batch = files[i : i + batch_size]

                # Process batch
                results = await self.indexer._process_file_batch(
                    batch, force_reindex=False
                )

                # Count successful updates
                updated_count += sum(1 for success in results if success)

                # Small delay between batches to be nice to the system
                if i + batch_size < len(files):
                    await asyncio.sleep(0.1)

            return updated_count

        except Exception as e:
            logger.error(f"Failed to reindex files: {e}")
            return updated_count

    def get_staleness_info(self) -> dict[str, any]:
        """Get information about index staleness."""
        try:
            metadata = self.indexer._load_index_metadata()
            all_files = self.indexer._find_indexable_files()

            stale_count = 0
            newest_file_time = 0.0
            oldest_index_time = float("inf")

            for file_path in all_files:
                file_mtime = os.path.getmtime(file_path)
                newest_file_time = max(newest_file_time, file_mtime)

                stored_mtime = metadata.get(str(file_path), 0)
                if stored_mtime > 0:
                    oldest_index_time = min(oldest_index_time, stored_mtime)

                if self.indexer._needs_reindexing(file_path, metadata):
                    stale_count += 1

            current_time = time.time()
            staleness_seconds = (
                current_time - oldest_index_time
                if oldest_index_time != float("inf")
                else 0
            )

            return {
                "total_files": len(all_files),
                "indexed_files": len(metadata),
                "stale_files": stale_count,
                "staleness_seconds": staleness_seconds,
                "is_stale": staleness_seconds > self.staleness_threshold,
                "newest_file_time": newest_file_time,
                "oldest_index_time": (
                    oldest_index_time if oldest_index_time != float("inf") else 0
                ),
            }

        except Exception as e:
            logger.error(f"Failed to get staleness info: {e}")
            return {
                "total_files": 0,
                "indexed_files": 0,
                "stale_files": 0,
                "staleness_seconds": 0,
                "is_stale": False,
                "newest_file_time": 0,
                "oldest_index_time": 0,
            }


class SearchTriggeredIndexer:
    """Automatically reindex when searches are performed."""

    def __init__(self, auto_indexer: AutoIndexer):
        self.auto_indexer = auto_indexer
        self._search_count = 0
        self._searches_since_check = 0
        self._check_every_n_searches = 10  # Check every 10 searches

    async def pre_search_hook(self) -> bool:
        """Hook to run before search operations.

        Returns:
            True if reindexing occurred, False otherwise
        """
        self._search_count += 1
        self._searches_since_check += 1

        # Only check periodically to avoid slowing down searches
        if self._searches_since_check >= self._check_every_n_searches:
            self._searches_since_check = 0

            logger.debug("Checking for stale files before search")
            reindexed, file_count = await self.auto_indexer.check_and_reindex_if_needed(
                force_check=False,
                interactive=False,  # Non-interactive during search
            )

            if reindexed:
                logger.info(f"Auto-reindexed {file_count} files before search")

            return reindexed

        return False

    def get_search_stats(self) -> dict[str, int]:
        """Get search-related statistics."""
        return {
            "total_searches": self._search_count,
            "searches_since_check": self._searches_since_check,
            "check_interval": self._check_every_n_searches,
        }


class PeriodicIndexChecker:
    """Check for stale index periodically during operations."""

    def __init__(self, auto_indexer: AutoIndexer, check_interval: float = 3600.0):
        """Initialize periodic checker.

        Args:
            auto_indexer: AutoIndexer instance
            check_interval: Check interval in seconds (default: 1 hour)
        """
        self.auto_indexer = auto_indexer
        self.check_interval = check_interval
        self._last_periodic_check = 0.0

    async def maybe_check_and_reindex(self) -> bool:
        """Check if it's time for a periodic reindex check.

        Returns:
            True if reindexing occurred, False otherwise
        """
        current_time = time.time()

        if (current_time - self._last_periodic_check) >= self.check_interval:
            self._last_periodic_check = current_time

            logger.debug("Performing periodic index staleness check")
            reindexed, file_count = await self.auto_indexer.check_and_reindex_if_needed(
                force_check=True, interactive=False
            )

            if reindexed:
                logger.info(f"Periodic auto-reindex updated {file_count} files")

            return reindexed

        return False

    def time_until_next_check(self) -> float:
        """Get time in seconds until next periodic check."""
        current_time = time.time()
        elapsed = current_time - self._last_periodic_check
        return max(0, self.check_interval - elapsed)
