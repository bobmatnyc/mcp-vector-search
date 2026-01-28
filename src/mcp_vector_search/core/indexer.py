"""Semantic indexer for MCP Vector Search."""

import asyncio
import os
import time
from pathlib import Path
from typing import Any

from loguru import logger

from ..analysis.collectors.base import MetricCollector
from ..analysis.trends import TrendTracker
from ..config.settings import ProjectConfig
from ..parsers.registry import get_parser_registry
from ..utils.monorepo import MonorepoDetector
from .chunk_processor import ChunkProcessor
from .database import VectorDatabase
from .directory_index import DirectoryIndex
from .exceptions import ParsingError
from .file_discovery import FileDiscovery
from .index_metadata import IndexMetadata
from .metrics_collector import IndexerMetricsCollector
from .models import CodeChunk, IndexStats
from .relationships import RelationshipStore


def cleanup_stale_locks(project_dir: Path) -> None:
    """Remove stale SQLite journal files that indicate interrupted transactions.

    Journal files (-journal, -wal, -shm) can be left behind if indexing is
    interrupted or crashes, preventing future database access. This function
    safely removes stale lock files at index startup.

    Args:
        project_dir: Project root directory containing .mcp-vector-search/
    """
    mcp_dir = project_dir / ".mcp-vector-search"
    if not mcp_dir.exists():
        return

    # SQLite journal file extensions that indicate locks/transactions
    lock_extensions = ["-journal", "-wal", "-shm"]

    removed_count = 0
    for ext in lock_extensions:
        lock_path = mcp_dir / f"chroma.sqlite3{ext}"
        if lock_path.exists():
            try:
                lock_path.unlink()
                logger.warning(f"Removed stale database lock file: {lock_path.name}")
                removed_count += 1
            except Exception as e:
                logger.error(f"Failed to remove stale lock file {lock_path}: {e}")

    if removed_count > 0:
        logger.info(
            f"Cleaned up {removed_count} stale lock files (indexing can now proceed)"
        )


class SemanticIndexer:
    """Semantic indexer for parsing and indexing code files."""

    def __init__(
        self,
        database: VectorDatabase,
        project_root: Path,
        file_extensions: list[str] | None = None,
        config: ProjectConfig | None = None,
        max_workers: int | None = None,
        batch_size: int = 10,
        debug: bool = False,
        collectors: list[MetricCollector] | None = None,
        use_multiprocessing: bool = True,
    ) -> None:
        """Initialize semantic indexer.

        Args:
            database: Vector database instance
            project_root: Project root directory
            file_extensions: File extensions to index (deprecated, use config)
            config: Project configuration (preferred over file_extensions)
            max_workers: Maximum number of worker processes for parallel parsing (ignored if use_multiprocessing=False)
            batch_size: Number of files to process in each batch
            debug: Enable debug output for hierarchy building
            collectors: Metric collectors to run during indexing (defaults to all complexity collectors)
            use_multiprocessing: Enable multiprocess parallel parsing (default: True, disable for debugging)
        """
        self.database = database
        self.project_root = project_root
        self.config = config
        self.batch_size = batch_size

        # Handle backward compatibility: use config.file_extensions or fallback to parameter
        if config is not None:
            file_extensions_set = {ext.lower() for ext in config.file_extensions}
        elif file_extensions is not None:
            file_extensions_set = {ext.lower() for ext in file_extensions}
        else:
            raise ValueError("Either config or file_extensions must be provided")

        # Initialize helper classes
        self.file_discovery = FileDiscovery(
            project_root=project_root,
            file_extensions=file_extensions_set,
            config=config,
        )

        self.metadata = IndexMetadata(project_root)

        self.metrics_collector = IndexerMetricsCollector(collectors)

        # Initialize monorepo detector
        self.monorepo_detector = MonorepoDetector(project_root)
        if self.monorepo_detector.is_monorepo():
            subprojects = self.monorepo_detector.detect_subprojects()
            logger.info(f"Detected monorepo with {len(subprojects)} subprojects")
            for sp in subprojects:
                logger.debug(f"  - {sp.name} ({sp.relative_path})")

        # Initialize parser registry
        self.parser_registry = get_parser_registry()

        self.chunk_processor = ChunkProcessor(
            parser_registry=self.parser_registry,
            monorepo_detector=self.monorepo_detector,
            max_workers=max_workers,
            use_multiprocessing=use_multiprocessing,
            debug=debug,
        )

        # Store use_multiprocessing for _process_file_batch
        self.use_multiprocessing = use_multiprocessing

        # Initialize directory index
        self.directory_index = DirectoryIndex(
            project_root / ".mcp-vector-search" / "directory_index.json"
        )
        # Load existing directory index
        self.directory_index.load()

        # Initialize relationship store for pre-computing visualization relationships
        self.relationship_store = RelationshipStore(project_root)

        # Initialize trend tracker for historical metrics
        self.trend_tracker = TrendTracker(project_root)

    async def index_project(
        self,
        force_reindex: bool = False,
        show_progress: bool = True,
        skip_relationships: bool = False,
    ) -> int:
        """Index all files in the project.

        Args:
            force_reindex: Whether to reindex existing files
            show_progress: Whether to show progress information
            skip_relationships: Skip computing relationships for visualization (faster, but visualize will be slower)

        Returns:
            Number of files indexed
        """
        logger.info(f"Starting indexing of project: {self.project_root}")

        # Clean up stale lock files from previous interrupted indexing runs
        cleanup_stale_locks(self.project_root)

        # Find all indexable files
        all_files = self.file_discovery.find_indexable_files()

        if not all_files:
            logger.warning("No indexable files found")
            return 0

        # Load existing metadata for incremental indexing
        metadata_dict = self.metadata.load()

        # Filter files that need indexing
        if force_reindex:
            files_to_index = all_files
            logger.info(f"Force reindex: processing all {len(files_to_index)} files")
        else:
            files_to_index = [
                f for f in all_files if self.metadata.needs_reindexing(f, metadata_dict)
            ]
            logger.info(
                f"Incremental index: {len(files_to_index)} of {len(all_files)} files need updating"
            )

        if not files_to_index:
            logger.info("All files are up to date")
            return 0

        # Index files in parallel batches
        indexed_count = 0
        failed_count = 0

        # Heartbeat logging to detect stuck indexing

        heartbeat_interval = 60  # Log every 60 seconds
        last_heartbeat = time.time()

        # Process files in batches for better memory management
        for i in range(0, len(files_to_index), self.batch_size):
            batch = files_to_index[i : i + self.batch_size]

            # Heartbeat logging
            now = time.time()
            if now - last_heartbeat >= heartbeat_interval:
                percentage = ((i + len(batch)) / len(files_to_index)) * 100
                logger.info(
                    f"Indexing heartbeat: {i + len(batch)}/{len(files_to_index)} files "
                    f"({percentage:.1f}%), {indexed_count} indexed, {failed_count} failed"
                )
                last_heartbeat = now

            if show_progress:
                logger.info(
                    f"Processing batch {i // self.batch_size + 1}/{(len(files_to_index) + self.batch_size - 1) // self.batch_size} ({len(batch)} files)"
                )

            # Process batch in parallel
            batch_results = await self._process_file_batch(batch, force_reindex)

            # Count results
            for success in batch_results:
                if success:
                    indexed_count += 1
                else:
                    failed_count += 1

        # Update metadata for successfully indexed files
        if indexed_count > 0:
            for file_path in files_to_index:
                try:
                    metadata_dict[str(file_path)] = os.path.getmtime(file_path)
                except OSError:
                    pass  # File might have been deleted during indexing

            self.metadata.save(metadata_dict)

            # Rebuild directory index from successfully indexed files
            try:
                logger.debug("Rebuilding directory index...")
                # We don't have chunk counts here, but we have file modification times
                # Build a simple stats dict with file mod times for recency tracking
                chunk_stats = {}
                for file_path in files_to_index:
                    try:
                        mtime = os.path.getmtime(file_path)
                        # For now, just track modification time
                        # Chunk counts will be aggregated from the database later if needed
                        chunk_stats[str(file_path)] = {
                            "modified": mtime,
                            "chunks": 1,  # Placeholder - real count from chunks
                        }
                    except OSError:
                        pass

                self.directory_index.rebuild_from_files(
                    files_to_index, self.project_root, chunk_stats=chunk_stats
                )
                self.directory_index.save()
                dir_stats = self.directory_index.get_stats()
                logger.info(
                    f"Directory index updated: {dir_stats['total_directories']} directories, "
                    f"{dir_stats['total_files']} files"
                )
            except Exception as e:
                logger.error(f"Failed to update directory index: {e}")
                import traceback

                logger.debug(traceback.format_exc())

        logger.info(
            f"Indexing complete: {indexed_count} files indexed, {failed_count} failed"
        )

        # Mark relationships for background computation (unless skipped)
        # Default behavior: skip blocking computation, mark for background processing
        if not skip_relationships and indexed_count > 0:
            try:
                logger.info("Marking relationships for background computation...")
                # Get all chunks from database for relationship computation
                all_chunks = await self.database.get_all_chunks()

                if len(all_chunks) > 0:
                    # Mark for background computation (non-blocking)
                    await self.relationship_store.compute_and_store(
                        all_chunks, self.database, background=True
                    )
                    logger.info("✓ Relationships marked for background computation")
                    logger.info(
                        "  Use 'mcp-vector-search index relationships' to compute now or wait for background task"
                    )
                else:
                    logger.warning("No chunks found for relationship computation")
            except Exception as e:
                logger.warning(f"Failed to mark relationships: {e}")
                logger.debug("Visualization will compute relationships on demand")

        # Save trend snapshot after successful indexing
        if indexed_count > 0:
            try:
                logger.info("Saving metrics snapshot for trend tracking...")
                # Get database stats
                stats = await self.database.get_stats()
                # Get all chunks for detailed metrics
                all_chunks = await self.database.get_all_chunks()
                # Compute metrics from stats and chunks
                metrics = self.trend_tracker.compute_metrics_from_stats(
                    stats.to_dict(), all_chunks
                )
                # Save snapshot (updates today's entry if exists)
                self.trend_tracker.save_snapshot(metrics)
                logger.info(
                    f"✓ Saved trend snapshot: {metrics['total_files']} files, "
                    f"{metrics['total_chunks']} chunks, health score {metrics['health_score']}"
                )
            except Exception as e:
                logger.warning(f"Failed to save trend snapshot: {e}")

        return indexed_count

    async def _parse_and_prepare_file(
        self, file_path: Path, force_reindex: bool = False
    ) -> tuple[list[CodeChunk], dict[str, Any] | None]:
        """Parse file and prepare chunks with metrics (no database insertion).

        This method extracts the parsing and metric collection logic from index_file()
        to enable batch processing across multiple files.

        Args:
            file_path: Path to the file to parse
            force_reindex: Whether to force reindexing (always deletes existing chunks)

        Returns:
            Tuple of (chunks_with_hierarchy, chunk_metrics)

        Raises:
            ParsingError: If file parsing fails
        """
        # Check if file should be indexed
        if not self.file_discovery.should_index_file(file_path):
            return ([], None)

        # Always remove existing chunks when reindexing a file
        # This prevents duplicate chunks and ensures consistency
        await self.database.delete_by_file(file_path)

        # Parse file into chunks
        chunks = await self.chunk_processor.parse_file(file_path)

        if not chunks:
            logger.debug(f"No chunks extracted from {file_path}")
            return ([], None)

        # Build hierarchical relationships between chunks
        chunks_with_hierarchy = self.chunk_processor.build_chunk_hierarchy(chunks)

        # Debug: Check if hierarchy was built
        methods_with_parents = sum(
            1
            for c in chunks_with_hierarchy
            if c.chunk_type in ("method", "function") and c.parent_chunk_id
        )
        logger.debug(
            f"After hierarchy build: {methods_with_parents}/{len([c for c in chunks_with_hierarchy if c.chunk_type in ('method', 'function')])} methods have parents"
        )

        # Collect metrics for chunks (if collectors are enabled)
        chunk_metrics = self.metrics_collector.collect_metrics_for_chunks(
            chunks_with_hierarchy, file_path
        )

        return (chunks_with_hierarchy, chunk_metrics)

    async def _process_file_batch(
        self, file_paths: list[Path], force_reindex: bool = False
    ) -> list[bool]:
        """Process a batch of files and accumulate chunks for batch embedding.

        This method processes multiple files in parallel (using multiprocessing for
        CPU-bound parsing) and then performs a single database insertion for all chunks,
        enabling efficient batch embedding generation.

        Args:
            file_paths: List of file paths to process
            force_reindex: Whether to force reindexing

        Returns:
            List of success flags for each file
        """
        all_chunks: list[CodeChunk] = []
        all_metrics: dict[str, Any] = {}
        file_to_chunks_map: dict[str, tuple[int, int]] = {}
        success_flags: list[bool] = []

        # Filter files that should be indexed and delete old chunks
        files_to_parse = []
        for file_path in file_paths:
            if not self.file_discovery.should_index_file(file_path):
                success_flags.append(True)  # Skipped file is not an error
                continue
            # Delete old chunks before parsing
            await self.database.delete_by_file(file_path)
            files_to_parse.append(file_path)

        if not files_to_parse:
            return success_flags

        # Parse files using multiprocessing if enabled
        if self.use_multiprocessing and len(files_to_parse) > 1:
            # Use ProcessPoolExecutor for CPU-bound parsing
            parse_results = await self.chunk_processor.parse_files_multiprocess(
                files_to_parse
            )
        else:
            # Fall back to async processing (for single file or disabled multiprocessing)
            parse_results = await self.chunk_processor.parse_files_async(files_to_parse)

        # Accumulate chunks from all successfully parsed files
        metadata_dict = self.metadata.load()
        for file_path, chunks, error in parse_results:
            if error:
                logger.error(f"Failed to parse {file_path}: {error}")
                success_flags.append(False)
                continue

            if chunks:
                # Build hierarchy and collect metrics for parsed chunks
                chunks_with_hierarchy = self.chunk_processor.build_chunk_hierarchy(
                    chunks
                )

                # Collect metrics if enabled
                chunk_metrics = self.metrics_collector.collect_metrics_for_chunks(
                    chunks_with_hierarchy, file_path
                )

                # Accumulate chunks
                start_idx = len(all_chunks)
                all_chunks.extend(chunks_with_hierarchy)
                end_idx = len(all_chunks)
                file_to_chunks_map[str(file_path)] = (start_idx, end_idx)

                # Merge metrics
                if chunk_metrics:
                    all_metrics.update(chunk_metrics)

                # Update metadata for successfully parsed file
                metadata_dict[str(file_path)] = os.path.getmtime(file_path)
                success_flags.append(True)
            else:
                # Empty file is not an error
                metadata_dict[str(file_path)] = os.path.getmtime(file_path)
                success_flags.append(True)

        # Single database insertion for entire batch
        if all_chunks:
            logger.info(
                f"Batch inserting {len(all_chunks)} chunks from {len(file_paths)} files"
            )
            try:
                await self.database.add_chunks(all_chunks, metrics=all_metrics)
                logger.debug(
                    f"Successfully indexed {len(all_chunks)} chunks from {sum(success_flags)} files"
                )
            except Exception as e:
                logger.error(f"Failed to insert batch of chunks: {e}")
                # Mark all files in this batch as failed
                return [False] * len(file_paths)

        # Save updated metadata after successful batch
        self.metadata.save(metadata_dict)

        return success_flags

    async def _index_file_safe(
        self, file_path: Path, force_reindex: bool = False
    ) -> bool:
        """Safely index a single file with error handling.

        Args:
            file_path: Path to the file to index
            force_reindex: Whether to force reindexing

        Returns:
            True if successful, False otherwise
        """
        try:
            return await self.index_file(file_path, force_reindex)
        except Exception as e:
            logger.error(f"Error indexing {file_path}: {e}")
            return False

    async def index_file(
        self,
        file_path: Path,
        force_reindex: bool = False,
    ) -> bool:
        """Index a single file.

        Args:
            file_path: Path to the file to index
            force_reindex: Whether to reindex if already indexed

        Returns:
            True if file was successfully indexed
        """
        try:
            # Check if file should be indexed
            if not self.file_discovery.should_index_file(file_path):
                return False

            # Always remove existing chunks when reindexing a file
            # This prevents duplicate chunks and ensures consistency
            await self.database.delete_by_file(file_path)

            # Parse file into chunks
            chunks = await self.chunk_processor.parse_file(file_path)

            if not chunks:
                logger.debug(f"No chunks extracted from {file_path}")
                return True  # Not an error, just empty file

            # Build hierarchical relationships between chunks
            chunks_with_hierarchy = self.chunk_processor.build_chunk_hierarchy(chunks)

            # Debug: Check if hierarchy was built
            methods_with_parents = sum(
                1
                for c in chunks_with_hierarchy
                if c.chunk_type in ("method", "function") and c.parent_chunk_id
            )
            logger.debug(
                f"After hierarchy build: {methods_with_parents}/{len([c for c in chunks_with_hierarchy if c.chunk_type in ('method', 'function')])} methods have parents"
            )

            # Collect metrics for chunks (if collectors are enabled)
            chunk_metrics = self.metrics_collector.collect_metrics_for_chunks(
                chunks_with_hierarchy, file_path
            )

            # Add chunks to database with metrics
            await self.database.add_chunks(chunks_with_hierarchy, metrics=chunk_metrics)

            # Update metadata after successful indexing
            metadata_dict = self.metadata.load()
            metadata_dict[str(file_path)] = os.path.getmtime(file_path)
            self.metadata.save(metadata_dict)

            logger.debug(f"Indexed {len(chunks)} chunks from {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to index file {file_path}: {e}")
            raise ParsingError(f"Failed to index file {file_path}: {e}") from e

    async def reindex_file(self, file_path: Path) -> bool:
        """Reindex a single file (removes existing chunks first).

        Args:
            file_path: Path to the file to reindex

        Returns:
            True if file was successfully reindexed
        """
        return await self.index_file(file_path, force_reindex=True)

    async def remove_file(self, file_path: Path) -> int:
        """Remove all chunks for a file from the index.

        Args:
            file_path: Path to the file to remove

        Returns:
            Number of chunks removed
        """
        try:
            count = await self.database.delete_by_file(file_path)
            logger.debug(f"Removed {count} chunks for {file_path}")
            return count
        except Exception as e:
            logger.error(f"Failed to remove file {file_path}: {e}")
            return 0

    def add_ignore_pattern(self, pattern: str) -> None:
        """Add a pattern to ignore during indexing.

        Args:
            pattern: Pattern to ignore (directory or file name)
        """
        self.file_discovery.add_ignore_pattern(pattern)

    def remove_ignore_pattern(self, pattern: str) -> None:
        """Remove an ignore pattern.

        Args:
            pattern: Pattern to remove
        """
        self.file_discovery.remove_ignore_pattern(pattern)

    def get_ignore_patterns(self) -> set[str]:
        """Get current ignore patterns.

        Returns:
            Set of ignore patterns
        """
        return self.file_discovery.get_ignore_patterns()

    def get_index_version(self) -> str | None:
        """Get the version of the tool that created the current index.

        Returns:
            Version string or None if not available
        """
        return self.metadata.get_index_version()

    def needs_reindex_for_version(self) -> bool:
        """Check if reindex is needed due to version upgrade.

        Returns:
            True if reindex is needed for version compatibility
        """
        return self.metadata.needs_reindex_for_version()

    # Backward compatibility methods for tests (delegate to helper classes)
    def _find_indexable_files(self) -> list[Path]:
        """Find all indexable files (backward compatibility)."""
        return self.file_discovery.find_indexable_files()

    def _should_index_file(
        self, file_path: Path, skip_file_check: bool = False
    ) -> bool:
        """Check if file should be indexed (backward compatibility)."""
        return self.file_discovery.should_index_file(file_path, skip_file_check)

    def _should_ignore_path(
        self, file_path: Path, is_directory: bool | None = None
    ) -> bool:
        """Check if path should be ignored (backward compatibility)."""
        return self.file_discovery.should_ignore_path(file_path, is_directory)

    def _needs_reindexing(self, file_path: Path, metadata: dict[str, float]) -> bool:
        """Check if file needs reindexing (backward compatibility)."""
        return self.metadata.needs_reindexing(file_path, metadata)

    def _load_index_metadata(self) -> dict[str, float]:
        """Load index metadata (backward compatibility)."""
        return self.metadata.load()

    def _save_index_metadata(self, metadata: dict[str, float]) -> None:
        """Save index metadata (backward compatibility)."""
        self.metadata.save(metadata)

    async def _parse_file(self, file_path: Path) -> list[CodeChunk]:
        """Parse a file (backward compatibility)."""
        return await self.chunk_processor.parse_file(file_path)

    def _build_chunk_hierarchy(self, chunks: list[CodeChunk]) -> list[CodeChunk]:
        """Build chunk hierarchy (backward compatibility)."""
        return self.chunk_processor.build_chunk_hierarchy(chunks)

    async def get_indexing_stats(self, db_stats: IndexStats | None = None) -> dict:
        """Get statistics about the indexing process.

        Args:
            db_stats: Optional pre-fetched database stats to avoid duplicate queries

        Returns:
            Dictionary with indexing statistics

        Note:
            Uses database statistics only for performance on large projects.
            Filesystem scanning would timeout on 100K+ file projects.
            Pass db_stats parameter to avoid calling database.get_stats() twice.
        """
        try:
            # Get database stats if not provided (fast, no filesystem scan)
            if db_stats is None:
                db_stats = await self.database.get_stats()

            # Use database stats for all file counts
            # This avoids expensive filesystem scans on large projects
            return {
                "total_indexable_files": db_stats.total_files,
                "indexed_files": db_stats.total_files,
                "total_files": db_stats.total_files,  # For backward compatibility
                "total_chunks": db_stats.total_chunks,
                "languages": db_stats.languages,
                "file_types": db_stats.file_types,  # Include file type distribution
                "file_extensions": list(self.file_discovery.file_extensions),
                "ignore_patterns": list(self.file_discovery.get_ignore_patterns()),
                "parser_info": self.parser_registry.get_parser_info(),
            }

        except Exception as e:
            logger.error(f"Failed to get indexing stats: {e}")
            return {
                "error": str(e),
                "total_indexable_files": 0,
                "indexed_files": 0,
                "total_files": 0,
                "total_chunks": 0,
            }

    async def get_files_to_index(
        self, force_reindex: bool = False
    ) -> tuple[list[Path], list[Path]]:
        """Get all indexable files and those that need indexing.

        Args:
            force_reindex: Whether to force reindex of all files

        Returns:
            Tuple of (all_indexable_files, files_to_index)
        """
        # Find all indexable files
        all_files = await self.file_discovery.find_indexable_files_async()

        if not all_files:
            return [], []

        # Load existing metadata for incremental indexing
        metadata_dict = self.metadata.load()

        # Filter files that need indexing
        if force_reindex:
            files_to_index = all_files
            logger.info(f"Force reindex: processing all {len(files_to_index)} files")
        else:
            files_to_index = [
                f for f in all_files if self.metadata.needs_reindexing(f, metadata_dict)
            ]
            logger.info(
                f"Incremental index: {len(files_to_index)} of {len(all_files)} files need updating"
            )

        return all_files, files_to_index

    async def index_files_with_progress(
        self,
        files_to_index: list[Path],
        force_reindex: bool = False,
    ):
        """Index files and yield progress updates for each file.

        This method processes files in batches and accumulates chunks across files
        before performing a single database insertion per batch for better performance.

        Args:
            files_to_index: List of file paths to index
            force_reindex: Whether to force reindexing

        Yields:
            Tuple of (file_path, chunks_added, success) for each processed file
        """
        # Write version header to error log at start of indexing run
        self.metadata.write_indexing_run_header()

        # Process files in batches for better memory management and embedding efficiency
        for i in range(0, len(files_to_index), self.batch_size):
            batch = files_to_index[i : i + self.batch_size]

            # Accumulate chunks from all files in batch
            all_chunks: list[CodeChunk] = []
            all_metrics: dict[str, Any] = {}
            file_to_chunks_map: dict[str, tuple[int, int]] = {}
            file_results: dict[Path, tuple[int, bool]] = {}

            # Parse all files in parallel
            tasks = []
            for file_path in batch:
                task = asyncio.create_task(
                    self._parse_and_prepare_file(file_path, force_reindex)
                )
                tasks.append(task)

            parse_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Accumulate chunks from successfully parsed files
            metadata_dict = self.metadata.load()
            for file_path, result in zip(batch, parse_results, strict=True):
                if isinstance(result, Exception):
                    error_msg = f"Failed to index file {file_path}: {type(result).__name__}: {str(result)}"
                    logger.error(error_msg)
                    file_results[file_path] = (0, False)

                    # Save error to error log file
                    self.metadata.log_indexing_error(error_msg)
                    continue

                chunks, metrics = result
                if chunks:
                    start_idx = len(all_chunks)
                    all_chunks.extend(chunks)
                    end_idx = len(all_chunks)
                    file_to_chunks_map[str(file_path)] = (start_idx, end_idx)

                    # Merge metrics
                    if metrics:
                        all_metrics.update(metrics)

                    # Update metadata for successfully parsed file
                    metadata_dict[str(file_path)] = os.path.getmtime(file_path)
                    file_results[file_path] = (len(chunks), True)
                    logger.debug(f"Prepared {len(chunks)} chunks from {file_path}")
                else:
                    # Empty file is not an error
                    metadata_dict[str(file_path)] = os.path.getmtime(file_path)
                    file_results[file_path] = (0, True)

            # Single database insertion for entire batch
            if all_chunks:
                logger.info(
                    f"Batch inserting {len(all_chunks)} chunks from {len(batch)} files"
                )
                try:
                    await self.database.add_chunks(all_chunks, metrics=all_metrics)
                    logger.debug(
                        f"Successfully indexed {len(all_chunks)} chunks from batch"
                    )
                except Exception as e:
                    error_msg = f"Failed to insert batch of chunks: {e}"
                    logger.error(error_msg)
                    # Mark all files with chunks in this batch as failed
                    for file_path in file_to_chunks_map.keys():
                        file_results[Path(file_path)] = (0, False)

                    # Save error to error log file
                    self.metadata.log_indexing_error(error_msg)

            # Save metadata after batch
            self.metadata.save(metadata_dict)

            # Yield progress updates for each file in batch
            for file_path in batch:
                chunks_added, success = file_results.get(file_path, (0, False))
                yield (file_path, chunks_added, success)
