"""Semantic indexer for MCP Vector Search with two-phase architecture.

Phase 1: Parse and chunk files, store to chunks.lance (fast, durable)
Phase 2: Embed chunks, store to vectors.lance (resumable, incremental)
"""

import asyncio
import gc
import json
import os
import platform
import shutil
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from ..analysis.collectors.base import MetricCollector
from ..analysis.trends import TrendTracker
from ..config.settings import ProjectConfig
from ..parsers.registry import get_parser_registry
from ..utils.monorepo import MonorepoDetector
from .bm25_backend import BM25Backend
from .chunk_processor import ChunkProcessor
from .chunks_backend import ChunksBackend, compute_file_hash
from .context_builder import build_contextual_text
from .database import VectorDatabase
from .directory_index import DirectoryIndex
from .exceptions import DatabaseError, DatabaseInitializationError, ParsingError
from .file_discovery import FileDiscovery
from .index_metadata import IndexMetadata
from .memory_monitor import MemoryMonitor
from .metrics import get_metrics_tracker
from .metrics_collector import IndexerMetricsCollector
from .models import CodeChunk, IndexStats
from .relationships import RelationshipStore
from .resource_manager import calculate_optimal_workers
from .vectors_backend import VectorsBackend


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
        batch_size: int | None = None,
        debug: bool = False,
        collectors: list[MetricCollector] | None = None,
        use_multiprocessing: bool = True,
        auto_optimize: bool = True,
        ignore_patterns: set[str] | None = None,
        skip_blame: bool = False,
        progress_tracker: Any = None,
    ) -> None:
        """Initialize semantic indexer.

        Args:
            database: Vector database instance
            project_root: Project root directory
            file_extensions: File extensions to index (deprecated, use config)
            config: Project configuration (preferred over file_extensions)
            max_workers: Maximum number of worker processes for parallel parsing (ignored if use_multiprocessing=False)
            batch_size: Number of files to process in each batch (default: 256, override with MCP_VECTOR_SEARCH_BATCH_SIZE or auto-optimization)
            debug: Enable debug output for hierarchy building
            collectors: Metric collectors to run during indexing (defaults to all complexity collectors)
            use_multiprocessing: Enable multiprocess parallel parsing (default: True, disable for debugging)
            auto_optimize: Enable automatic optimization based on codebase profile (default: True)
            ignore_patterns: Additional patterns to ignore (merged with defaults, e.g., vendor patterns)
            skip_blame: Skip git blame tracking for faster indexing (default: False)
            progress_tracker: Optional ProgressTracker instance for displaying progress bars

        Environment Variables:
            MCP_VECTOR_SEARCH_BATCH_SIZE: Override batch size (default: 256)
            MCP_VECTOR_SEARCH_SKIP_BLAME: Skip git blame tracking (true/1/yes)
        """
        self.database = database
        self.project_root = project_root
        self.config = config
        self.auto_optimize = auto_optimize
        self._applied_optimizations: dict[str, Any] | None = None
        self.cancellation_flag: threading.Event | None = (
            None  # Set externally for cancellation support
        )
        self.progress_tracker = progress_tracker  # Optional progress bar display

        # Set batch size with environment variable override
        if batch_size is None:
            # Check for FILE_BATCH_SIZE first (new name for clarity)
            env_batch_size = os.environ.get(
                "MCP_VECTOR_SEARCH_FILE_BATCH_SIZE"
            ) or os.environ.get("MCP_VECTOR_SEARCH_BATCH_SIZE")
            if env_batch_size:
                try:
                    self.batch_size = int(env_batch_size)
                    logger.info(
                        f"Using file batch size from environment: {self.batch_size}"
                    )
                except ValueError:
                    logger.warning(
                        f"Invalid batch size value: {env_batch_size}, using default 512"
                    )
                    self.batch_size = 512
            else:
                # Increased default from 256 to 512 for better throughput
                # PERFORMANCE: Larger batches reduce per-batch overhead
                # 32K files @ 256/batch = 125 batches, @ 512/batch = 62 batches (50% reduction)
                # 512 files per batch is optimal for GPU utilization
                self.batch_size = 512
        else:
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
            ignore_patterns=ignore_patterns,
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

        # Auto-configure workers if not provided
        if max_workers is None:
            # Check environment variable first for explicit override
            env_workers = os.environ.get("MCP_VECTOR_SEARCH_WORKERS")
            if env_workers:
                try:
                    max_workers = int(env_workers)
                    logger.info(f"Using worker count from environment: {max_workers}")
                except ValueError:
                    logger.warning(
                        f"Invalid MCP_VECTOR_SEARCH_WORKERS value: {env_workers}, using auto-detection"
                    )
                    max_workers = None

            # Auto-detect based on CPU cores and memory if not overridden
            if max_workers is None:
                # Use memory-aware worker calculation
                # Remove hard-coded max_workers=4 limit - let calculate_optimal_workers decide
                limits = calculate_optimal_workers(
                    memory_per_worker_mb=800,  # Embedding models use more memory
                    # max_workers defaults to 8 in calculate_optimal_workers, but CPU count is also considered
                )
                max_workers = limits.max_workers
                logger.info(
                    f"Auto-configured {max_workers} workers based on available memory and CPU cores"
                )

        # Conditionally enable git blame based on skip_blame flag
        # When skip_blame=True, pass repo_root=None to disable GitBlameCache
        repo_root_for_blame = None if skip_blame else project_root

        self.chunk_processor = ChunkProcessor(
            parser_registry=self.parser_registry,
            monorepo_detector=self.monorepo_detector,
            max_workers=max_workers,
            use_multiprocessing=use_multiprocessing,
            debug=debug,
            repo_root=repo_root_for_blame,  # Enable git blame only if not skipped
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

        # Initialize two-phase backends
        # Both use same db_path directory for LanceDB
        index_path = project_root / ".mcp-vector-search" / "lance"
        self.chunks_backend = ChunksBackend(index_path)
        self.vectors_backend = VectorsBackend(index_path)

        # Background KG build tracking
        self._kg_build_task: asyncio.Task | None = None
        self._kg_build_status: str = "not_started"
        self._enable_background_kg: bool = os.environ.get(
            "MCP_VECTOR_SEARCH_AUTO_KG", "false"
        ).lower() in ("true", "1", "yes")

        # Track atomic rebuild state (fresh database doesn't need deletes)
        self._atomic_rebuild_active: bool = False

        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor()

    async def _build_kg_background(self) -> None:
        """Build knowledge graph in background (non-blocking).

        This method runs after indexing completes to build the KG without
        blocking search availability. KG enhancement will be available
        once this completes.
        """
        self._kg_build_status = "building"
        try:
            from .kg_builder import KGBuilder
            from .knowledge_graph import KnowledgeGraph

            kg_path = self.project_root / ".mcp-vector-search" / "knowledge_graph"
            kg = KnowledgeGraph(kg_path)
            await kg.initialize()

            builder = KGBuilder(kg, self.project_root)

            logger.info(
                "🔗 Phase 3: Building knowledge graph in background (relationship extraction)..."
            )

            # Use the same database connection
            async with self.database:
                await builder.build_from_database(
                    self.database,
                    show_progress=False,  # No progress for background
                    skip_documents=True,  # Fast mode for background
                )

            await kg.close()
            self._kg_build_status = "complete"
            logger.info("✓ Phase 3 complete: Knowledge graph built successfully")

        except Exception as e:
            self._kg_build_status = f"error: {e}"
            logger.error(f"Background KG build failed: {e}")

    def get_kg_status(self) -> str:
        """Get current KG build status.

        Returns:
            Status string: "not_started", "building", "complete", or "error: <message>"
        """
        return self._kg_build_status

    def get_embedding_model_name(self) -> str:
        """Get the current embedding model name from the database.

        Returns:
            Model name (e.g., "all-MiniLM-L6-v2", "graphcodebert-base")
        """
        # Try multiple paths to get the model name
        if hasattr(self.database, "_embedding_function") and hasattr(
            self.database._embedding_function, "model_name"
        ):
            return self.database._embedding_function.model_name
        elif hasattr(self.database, "_collection") and hasattr(
            self.database._collection, "_embedding_function"
        ):
            return self.database._collection._embedding_function.model_name
        elif hasattr(self.database, "embedding_function") and hasattr(
            self.database.embedding_function, "model_name"
        ):
            return self.database.embedding_function.model_name

        # Fallback: extract from model name if available
        # This handles cases where the embedding function doesn't expose model_name
        return "all-MiniLM-L6-v2"  # Default fallback

    def apply_auto_optimizations(self) -> tuple[Any, Any] | None:
        """Apply automatic optimizations based on codebase profile.

        Profiles the codebase and applies optimal settings for batch size,
        file extensions, and other performance parameters.

        Returns:
            Tuple of (profile, preset) if optimizations applied, None otherwise
        """
        if not self.auto_optimize:
            return None

        from .codebase_profiler import CodebaseProfiler

        try:
            # Profile codebase (fast sampling)
            profiler = CodebaseProfiler(self.project_root)
            profile = profiler.profile()

            # Get optimization preset
            preset = profiler.get_optimization_preset(profile)

            # Store previous batch size for comparison
            previous_batch_size = self.batch_size

            # Apply optimizations (only if not already overridden)
            if os.environ.get("MCP_VECTOR_SEARCH_BATCH_SIZE") is None:
                self.batch_size = preset.batch_size

            # Apply file extension filtering (only for large/enterprise)
            if preset.file_extensions and self.config:
                # Merge preset extensions with user-configured extensions
                original_extensions = set(self.config.file_extensions)
                filtered_extensions = original_extensions & preset.file_extensions
                if filtered_extensions:
                    self.config.file_extensions = list(filtered_extensions)
                    logger.info(
                        f"Filtered to {len(filtered_extensions)} code extensions "
                        f"(from {len(original_extensions)})"
                    )

            # Store applied optimizations for reporting
            self._applied_optimizations = {
                "profile": profile,
                "preset": preset,
                "previous_batch_size": previous_batch_size,
            }

            # Log profile and optimizations
            logger.info(profiler.format_profile_summary(profile))
            logger.info("")
            logger.info(
                profiler.format_optimization_summary(
                    profile, preset, previous_batch_size
                )
            )

            return profile, preset

        except Exception as e:
            logger.warning(f"Auto-optimization failed, using defaults: {e}")
            return None

    async def _run_auto_migrations(self) -> None:
        """Check and run pending database migrations automatically.

        This method is called during indexer initialization to ensure
        the database schema is up-to-date before indexing begins.

        Only runs migrations that are needed (check_needed() returns True).
        Logs warnings if migrations fail but doesn't stop indexing.
        """
        try:
            from ..migrations import MigrationRunner
            from ..migrations.v2_3_0_two_phase import TwoPhaseArchitectureMigration

            # Create migration runner
            runner = MigrationRunner(self.project_root)

            # Register migrations that might be needed
            runner.register_migrations([TwoPhaseArchitectureMigration()])

            # Get pending migrations
            pending = runner.get_pending_migrations()

            if not pending:
                logger.debug("No pending migrations, schema is up-to-date")
                return

            logger.info(
                f"Found {len(pending)} pending migration(s), running automatically..."
            )

            # Run pending migrations
            for migration in pending:
                logger.info(
                    f"Running migration: {migration.name} (v{migration.version})"
                )

                result = runner.run_migration(migration, dry_run=False, force=False)

                if result.status.value == "success":
                    logger.info(f"✓ Migration {migration.name} completed successfully")
                    if result.metadata:
                        for key, value in result.metadata.items():
                            logger.debug(f"  {key}: {value}")
                elif result.status.value == "skipped":
                    logger.debug(
                        f"⊘ Migration {migration.name} skipped: {result.message}"
                    )
                elif result.status.value == "failed":
                    logger.warning(
                        f"✗ Migration {migration.name} failed: {result.message}\n"
                        "  Continuing with indexing, but you may encounter issues.\n"
                        "  Run 'mcp-vector-search migrate' to fix manually."
                    )

        except Exception as e:
            logger.warning(
                f"Auto-migration check failed: {e}\n"
                "  Continuing with indexing, but you may encounter schema issues.\n"
                "  Run 'mcp-vector-search migrate' to fix manually."
            )

    async def _index_with_pipeline(
        self, force_reindex: bool = False
    ) -> tuple[int, int, int]:
        """Index files using pipeline parallelism to overlap Phase 1 and Phase 2.

        This method implements a producer-consumer pattern where:
        - Producer: Chunks files in batches and puts them on a queue
        - Consumer: Embeds chunks from the queue concurrently
        - Result: Parsing and embedding overlap for 30-50% speedup

        Args:
            force_reindex: Whether to reindex all files

        Returns:
            Tuple of (files_indexed, chunks_created, chunks_embedded)
        """
        # Find all indexable files
        all_files = self.file_discovery.find_indexable_files()

        # Clean up stale metadata entries
        if force_reindex:
            logger.info("Force reindex: clearing all metadata entries")
            self.metadata.save({})
        elif all_files:
            valid_files = {str(f) for f in all_files}
            removed = self.metadata.cleanup_stale_entries(valid_files)
            if removed > 0:
                logger.info(
                    f"Removed {removed} stale metadata entries for deleted files"
                )

        if not all_files:
            logger.warning("No indexable files found")
            return 0, 0, 0

        # Filter files that need indexing
        files_to_index = all_files
        if not force_reindex:
            logger.info(
                f"Incremental change detection: checking {len(all_files)} files..."
            )

            # OPTIMIZATION: Load all indexed file hashes ONCE for O(1) per-file lookup
            indexed_file_hashes = (
                await self.chunks_backend.get_all_indexed_file_hashes()
            )
            logger.info(
                f"Loaded {len(indexed_file_hashes)} indexed files for change detection"
            )

            # Detect file moves/renames — update metadata instead of re-chunking
            detected_moves, _moved_old_paths = self._detect_file_moves(
                all_files, indexed_file_hashes
            )
            if detected_moves:
                logger.info(
                    f"Detected {len(detected_moves)} file move(s), updating metadata..."
                )
                for old_path, new_path, _file_hash in detected_moves:
                    chunks_updated = await self.chunks_backend.update_file_path(
                        old_path, new_path
                    )
                    vectors_updated = await self.vectors_backend.update_file_path(
                        old_path, new_path
                    )
                    logger.info(
                        f"  Moved: {old_path} -> {new_path} "
                        f"({chunks_updated} chunks, {vectors_updated} vectors)"
                    )
                # Reload so change detection sees updated paths
                indexed_file_hashes = (
                    await self.chunks_backend.get_all_indexed_file_hashes()
                )

            filtered_files = []
            for idx, f in enumerate(all_files, start=1):
                try:
                    file_hash = compute_file_hash(f)
                    rel_path = str(f.relative_to(self.project_root))

                    # OPTIMIZATION: O(1) dict lookup instead of per-file database query
                    stored_hash = indexed_file_hashes.get(rel_path)
                    if stored_hash is None or stored_hash != file_hash:
                        filtered_files.append(f)
                except Exception as e:
                    logger.warning(f"Error checking file {f}: {e}, will re-index")
                    filtered_files.append(f)

                if idx % 500 == 0:
                    logger.info(
                        f"Change detection progress: {idx}/{len(all_files)} files checked"
                    )

            files_to_index = filtered_files
            logger.info(
                f"Incremental index: {len(files_to_index)} of {len(all_files)} files need updating"
            )
        else:
            logger.info(f"Force reindex: processing all {len(files_to_index)} files")

        if not files_to_index:
            logger.info("All files are up to date")
            # Still build BM25 index if it doesn't exist
            if self.config and self.config.index_path:
                bm25_path = self.config.index_path / "bm25_index.pkl"
                if not bm25_path.exists():
                    logger.info(f"BM25 index not found at {bm25_path}, building now...")
                    await self._build_bm25_index()
                else:
                    logger.info(f"BM25 index already exists at {bm25_path}")
            return 0, 0, 0

        # PIPELINE IMPLEMENTATION: Producer-consumer pattern
        # PERFORMANCE: Increased queue buffer from 2 to 10 to allow more parsed batches ahead
        # This prevents GPU starvation when parsing batches take varying amounts of time
        chunk_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        files_indexed = 0
        chunks_created = 0
        chunks_embedded = 0

        # Get metrics tracker
        metrics_tracker = get_metrics_tracker()

        # Batch size for file processing (can be different from embedding batch size)
        file_batch_size = 256  # Process files in groups of 256

        async def chunk_producer():
            """Producer: Parse and chunk files in batches, put on queue."""
            nonlocal files_indexed, chunks_created

            logger.info(
                f"📄 Phase 1: Chunking {len(files_to_index)} files (parsing and extracting code structure)..."
            )

            # Track start time for ETA calculation
            phase_start_time = time.time()

            with metrics_tracker.phase("parsing") as parsing_metrics:
                # Batch delete optimization (like _phase1_chunk_files)
                files_to_delete = []
                files_to_process = []

                # OPTIMIZATION: Load all indexed file hashes ONCE for O(1) per-file lookup
                indexed_file_hashes = {}
                if not force_reindex:
                    logger.info("Loading indexed file hashes for change detection...")
                    indexed_file_hashes = (
                        await self.chunks_backend.get_all_indexed_file_hashes()
                    )
                    logger.info(
                        f"Loaded {len(indexed_file_hashes)} indexed files for change detection"
                    )

                # Detect file moves/renames — update metadata instead of re-chunking
                if not force_reindex:
                    detected_moves, _moved_old_paths = self._detect_file_moves(
                        files_to_index, indexed_file_hashes
                    )
                    if detected_moves:
                        logger.info(
                            f"Detected {len(detected_moves)} file move(s), updating metadata..."
                        )
                        for old_path, new_path, _file_hash in detected_moves:
                            chunks_updated = await self.chunks_backend.update_file_path(
                                old_path, new_path
                            )
                            vectors_updated = (
                                await self.vectors_backend.update_file_path(
                                    old_path, new_path
                                )
                            )
                            logger.info(
                                f"  Moved: {old_path} -> {new_path} "
                                f"({chunks_updated} chunks, {vectors_updated} vectors)"
                            )
                        # Reload so change detection sees updated paths
                        indexed_file_hashes = (
                            await self.chunks_backend.get_all_indexed_file_hashes()
                        )

                # Compute file hashes
                if force_reindex:
                    logger.info("Force mode enabled, skipping file hash computation...")
                else:
                    logger.info(
                        f"Computing file hashes for change detection ({len(files_to_index)} files)..."
                    )

                for idx, file_path in enumerate(files_to_index):
                    try:
                        if idx > 0 and idx % 1000 == 0:
                            logger.info(
                                f"Computing file hashes: {idx}/{len(files_to_index)}"
                            )

                        if force_reindex:
                            file_hash = ""
                        else:
                            file_hash = compute_file_hash(file_path)

                        rel_path = str(file_path.relative_to(self.project_root))

                        # OPTIMIZATION: O(1) dict lookup instead of per-file database query
                        if not force_reindex:
                            stored_hash = indexed_file_hashes.get(rel_path)
                            if stored_hash is not None and stored_hash == file_hash:
                                logger.debug(f"Skipping unchanged file: {rel_path}")
                                continue

                        files_to_delete.append(rel_path)
                        files_to_process.append((file_path, rel_path, file_hash))

                    except Exception as e:
                        logger.error(f"Failed to check file {file_path}: {e}")
                        continue

                if not force_reindex:
                    logger.info(
                        f"File hash computation complete: {len(files_to_process)} files changed, "
                        f"{len(files_to_index) - len(files_to_process)} unchanged"
                    )

                # Batch delete old chunks
                # WORKAROUND: Skip delete on macOS to avoid SIGBUS crash
                # LanceDB delete() triggers memory-mapped file compaction which conflicts
                # with PyTorch's memory-mapped model files on Apple Silicon.
                # Duplicate chunks will be handled by deduplication logic during query.
                if (
                    not self._atomic_rebuild_active
                    and files_to_delete
                    and platform.system() != "Darwin"
                ):
                    deleted_count = await self.chunks_backend.delete_files_batch(
                        files_to_delete
                    )
                    if deleted_count > 0:
                        logger.info(
                            f"Batch deleted {deleted_count} old chunks for {len(files_to_delete)} files"
                        )
                elif files_to_delete and platform.system() == "Darwin":
                    logger.debug(
                        f"Skipping batch delete on macOS for {len(files_to_delete)} files "
                        "(defer cleanup to avoid SIGBUS)"
                    )

                # Process files in batches and put on queue
                for batch_start in range(0, len(files_to_process), file_batch_size):
                    # Check if consumer is still alive
                    if consumer_task.done():
                        exc = consumer_task.exception()
                        logger.error(
                            f"Consumer died unexpectedly: {exc}. Stopping producer."
                        )
                        break

                    batch_end = min(
                        batch_start + file_batch_size, len(files_to_process)
                    )
                    batch = files_to_process[batch_start:batch_end]

                    batch_chunks = []
                    batch_files_processed = 0

                    for file_path, rel_path, file_hash in batch:
                        # Check memory before processing
                        is_ok, usage_pct, status = (
                            self.memory_monitor.check_memory_limit()
                        )
                        if not is_ok:
                            logger.warning(
                                f"Memory limit exceeded during chunking "
                                f"({usage_pct * 100:.1f}% of {self.memory_monitor.max_memory_gb:.1f}GB), "
                                "waiting for memory to free up..."
                            )
                            await self.memory_monitor.wait_for_memory_available(
                                target_pct=self.memory_monitor.warn_threshold
                            )

                        try:
                            # Parse file (runs in thread pool to avoid blocking event loop)
                            chunks = await self.chunk_processor.parse_file(file_path)

                            if not chunks:
                                # Use TRACE to avoid cluttering progress displays
                                logger.trace(f"No chunks extracted from {file_path}")
                                continue

                            # Build hierarchical relationships (CPU-bound, run in thread pool)
                            chunks_with_hierarchy = await asyncio.to_thread(
                                self.chunk_processor.build_chunk_hierarchy, chunks
                            )

                            # Convert CodeChunk objects to dicts for storage
                            chunk_dicts = []
                            for chunk in chunks_with_hierarchy:
                                chunk_dict = {
                                    "chunk_id": chunk.chunk_id,
                                    "file_path": rel_path,
                                    "content": chunk.content,
                                    "language": chunk.language,
                                    "start_line": chunk.start_line,
                                    "end_line": chunk.end_line,
                                    "start_char": 0,
                                    "end_char": 0,
                                    "chunk_type": chunk.chunk_type,
                                    "name": chunk.function_name
                                    or chunk.class_name
                                    or "",
                                    "parent_name": "",
                                    "hierarchy_path": self._build_hierarchy_path(chunk),
                                    "docstring": chunk.docstring or "",
                                    "signature": "",
                                    "complexity": int(chunk.complexity_score),
                                    "token_count": len(chunk.content.split()),
                                    "last_author": chunk.last_author or "",
                                    "last_modified": chunk.last_modified or "",
                                    "commit_hash": chunk.commit_hash or "",
                                    "calls": chunk.calls or [],
                                    "imports": [
                                        json.dumps(imp)
                                        if isinstance(imp, dict)
                                        else imp
                                        for imp in (chunk.imports or [])
                                    ],
                                    "inherits_from": chunk.inherits_from or [],
                                }
                                chunk_dicts.append(chunk_dict)

                            # Store chunks to chunks.lance
                            if chunk_dicts:
                                count = await self.chunks_backend.add_chunks(
                                    chunk_dicts, file_hash
                                )
                                batch_chunks.extend(chunk_dicts)
                                chunks_created += count
                                batch_files_processed += 1
                                # Use TRACE to avoid cluttering progress displays
                                logger.trace(f"Chunked {count} chunks from {rel_path}")

                        except Exception as e:
                            logger.error(f"Failed to chunk file {file_path}: {e}")
                            continue

                    files_indexed += batch_files_processed

                    # Put batch on queue for embedding (blocks if queue is full - backpressure)
                    if batch_chunks:
                        # Show progress bar if tracker is available
                        if self.progress_tracker:
                            self.progress_tracker.progress_bar_with_eta(
                                current=files_indexed,
                                total=len(files_to_process),
                                prefix="Parsing files",
                                start_time=phase_start_time,
                            )
                        else:
                            logger.info(
                                f"Phase 1 progress: {files_indexed}/{len(files_to_process)} files, "
                                f"{chunks_created} chunks | Queuing {len(batch_chunks)} chunks for embedding"
                            )
                        await chunk_queue.put(
                            {"chunks": batch_chunks, "batch_size": len(batch_chunks)}
                        )
                        # Yield to event loop to let consumer process the batch
                        await asyncio.sleep(0)

                # Update metrics
                parsing_metrics.item_count = files_indexed

            # Track chunking separately
            with metrics_tracker.phase("chunking") as chunking_metrics:
                chunking_metrics.item_count = chunks_created

            # Log completion
            self.memory_monitor.log_memory_summary()
            logger.info(
                f"✓ Phase 1 complete: {files_indexed} files processed, {chunks_created} chunks created"
            )

            # Signal completion
            await chunk_queue.put(None)

        async def embed_consumer():
            """Consumer: Take chunks from queue, embed, and store to vectors.lance."""
            nonlocal chunks_embedded

            logger.info(
                "🧠 Phase 2: Embedding pending chunks (GPU processing for semantic search)..."
            )

            embedding_batch_size = self.batch_size  # Use configured batch size
            consecutive_errors = 0  # Track consecutive errors to detect fatal issues
            # Track start time for ETA calculation
            embed_start_time = time.time()

            with metrics_tracker.phase("embedding") as embedding_metrics:
                while True:
                    # Get next batch from queue (blocks until available)
                    batch_data = await chunk_queue.get()

                    # Check for completion signal
                    if batch_data is None:
                        break

                    chunks = batch_data["chunks"]

                    if not chunks:
                        continue

                    # Process chunks in embedding batches
                    for emb_batch_start in range(0, len(chunks), embedding_batch_size):
                        emb_batch_end = min(
                            emb_batch_start + embedding_batch_size, len(chunks)
                        )
                        emb_batch = chunks[emb_batch_start:emb_batch_end]

                        # Check memory before embedding
                        usage_pct = self.memory_monitor.get_memory_usage_pct()

                        # Proactively reduce batch size if memory is high
                        while usage_pct > 0.90 and embedding_batch_size > 100:
                            embedding_batch_size = embedding_batch_size // 2
                            logger.warning(
                                f"⚠️  Memory at {usage_pct * 100:.1f}%, proactively reducing batch to {embedding_batch_size}"
                            )
                            usage_pct = self.memory_monitor.get_memory_usage_pct()

                        if usage_pct > 0.90:
                            logger.warning(
                                f"Memory usage at {usage_pct * 100:.1f}%, waiting for memory to drop..."
                            )
                            try:
                                await asyncio.wait_for(
                                    self.memory_monitor.wait_for_memory_available(
                                        target_pct=0.80
                                    ),
                                    timeout=120.0,  # 2 minute timeout
                                )
                            except TimeoutError:
                                logger.warning(
                                    "Memory wait timed out after 120s, proceeding anyway"
                                )

                        try:
                            # Generate embeddings using context-enriched text.
                            # build_contextual_text() prepends file path, language,
                            # class/function context, imports, and docstring to each
                            # chunk before embedding, improving retrieval quality by
                            # 35–49% (contextual RAG research).  The stored
                            # chunk["content"] field is NOT modified — only the text
                            # sent to the embedding model is enriched.
                            contents = [build_contextual_text(c) for c in emb_batch]

                            # Check memory before expensive embedding
                            is_ok, usage_pct, status = (
                                self.memory_monitor.check_memory_limit()
                            )
                            if not is_ok:
                                logger.warning(
                                    f"Memory limit exceeded before embedding "
                                    f"({usage_pct * 100:.1f}% of {self.memory_monitor.max_memory_gb:.1f}GB). "
                                    "Waiting for memory..."
                                )
                                try:
                                    await asyncio.wait_for(
                                        self.memory_monitor.wait_for_memory_available(
                                            target_pct=self.memory_monitor.warn_threshold
                                        ),
                                        timeout=120.0,  # 2 minute timeout
                                    )
                                except TimeoutError:
                                    logger.warning(
                                        "Memory wait timed out after 120s, proceeding anyway"
                                    )

                            # Generate embeddings
                            vectors = None
                            if hasattr(self.database, "_embedding_function"):
                                vectors = self.database._embedding_function(contents)
                            elif hasattr(self.database, "_collection") and hasattr(
                                self.database._collection, "_embedding_function"
                            ):
                                vectors = self.database._collection._embedding_function(
                                    contents
                                )
                            elif hasattr(self.database, "embedding_function"):
                                vectors = (
                                    self.database.embedding_function.embed_documents(
                                        contents
                                    )
                                )
                            else:
                                logger.error(
                                    "Cannot access embedding function from database, skipping batch"
                                )
                                continue

                            if vectors is None:
                                raise ValueError("Failed to generate embeddings")

                            # Add to vectors table
                            chunks_with_vectors = []
                            for chunk, vec in zip(emb_batch, vectors, strict=True):
                                chunk_with_vec = {
                                    "chunk_id": chunk["chunk_id"],
                                    "vector": vec,
                                    "file_path": chunk["file_path"],
                                    "content": chunk["content"],
                                    "language": chunk["language"],
                                    "start_line": chunk["start_line"],
                                    "end_line": chunk["end_line"],
                                    "chunk_type": chunk["chunk_type"],
                                    "name": chunk["name"],
                                    "hierarchy_path": chunk["hierarchy_path"],
                                }
                                chunks_with_vectors.append(chunk_with_vec)

                            # Store to vectors.lance
                            model_name = self.get_embedding_model_name()
                            await self.vectors_backend.add_vectors(
                                chunks_with_vectors, model_version=model_name
                            )
                            chunks_embedded += len(emb_batch)

                            # Reset consecutive error counter on success
                            consecutive_errors = 0

                            # Show progress bar if tracker is available
                            # Note: We use chunks_created as an estimate of total (may update as producer runs)
                            if self.progress_tracker and chunks_created > 0:
                                self.progress_tracker.progress_bar_with_eta(
                                    current=chunks_embedded,
                                    total=chunks_created,
                                    prefix="Embedding chunks",
                                    start_time=embed_start_time,
                                )
                            else:
                                logger.info(
                                    f"Phase 2 progress: {chunks_embedded} chunks embedded"
                                )

                        except Exception as e:
                            consecutive_errors += 1
                            logger.error(
                                f"Failed to embed batch (consecutive errors: {consecutive_errors}): {e}"
                            )

                            # Fail after too many consecutive errors to avoid silent hangs
                            if consecutive_errors >= 3:
                                logger.error(
                                    f"Too many consecutive embedding errors ({consecutive_errors}), "
                                    "stopping consumer to prevent silent failure"
                                )
                                raise

                            continue

                # Update metrics
                embedding_metrics.item_count = chunks_embedded

            logger.info(f"✓ Phase 2 complete: {chunks_embedded} chunks embedded")

        # Run producer and consumer concurrently with pipeline parallelism
        logger.info(
            "Starting pipeline: Phase 1 (parsing) and Phase 2 (embedding) will overlap"
        )
        producer_task = asyncio.create_task(chunk_producer())
        consumer_task = asyncio.create_task(embed_consumer())

        # Wait for both to complete with proper error handling
        try:
            await asyncio.gather(producer_task, consumer_task)
        except Exception as e:
            logger.error(f"Pipeline task failed: {e}")

            # Cancel the other task if one fails
            for task in [producer_task, consumer_task]:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # Re-raise the original exception
            raise

        # Update metadata for backward compatibility
        if files_indexed > 0:
            metadata_dict = self.metadata.load()
            for file_path in files_to_index:
                try:
                    metadata_dict[str(file_path)] = os.path.getmtime(file_path)
                except OSError:
                    pass
            self.metadata.save(metadata_dict)

        # CLEANUP: Shutdown persistent ProcessPoolExecutor after indexing completes
        self.chunk_processor.close()

        return files_indexed, chunks_created, chunks_embedded

    def _detect_file_moves(
        self,
        current_files: list[Path],
        indexed_file_hashes: dict[str, str],
    ) -> tuple[list[tuple[str, str, str]], set[str]]:
        """Detect file moves/renames by matching content hashes.

        Compares the set of currently indexed paths against on-disk paths.
        When an indexed path is gone but a new path has the same SHA-256
        content hash, we treat that as a move rather than a delete+add.

        Only unambiguous 1-to-1 moves are handled.  If multiple indexed
        paths share the same hash, they are matched by sorted name order —
        this covers batch directory-rename scenarios while staying safe.

        Args:
            current_files: All currently-discoverable files on disk
            indexed_file_hashes: Mapping of rel_path → file_hash from the DB

        Returns:
            Tuple of:
            - List of (old_path, new_path, file_hash) for each detected move
            - Set of old_path strings that were moved (exclude from normal
              delete/re-process flow after caller reloads hashes)
        """
        # Reverse index: hash → set of paths currently in the DB
        hash_to_indexed: dict[str, set[str]] = {}
        for path, hash_val in indexed_file_hashes.items():
            hash_to_indexed.setdefault(hash_val, set()).add(path)

        # Build set of relative paths that exist on disk right now, and a
        # reverse map from hash → set of on-disk relative paths
        current_rel_paths: set[str] = set()
        current_hash_to_paths: dict[str, set[str]] = {}
        for file_path in current_files:
            try:
                rel_path = str(file_path.relative_to(self.project_root))
                current_rel_paths.add(rel_path)
                file_hash = compute_file_hash(file_path)
                current_hash_to_paths.setdefault(file_hash, set()).add(rel_path)
            except Exception:  # nosec B112  # noqa: B112
                continue

        # Find moves: indexed paths that are no longer on disk (orphaned) paired
        # with new on-disk paths that share the same hash and are not yet indexed
        moves: list[tuple[str, str, str]] = []
        moved_old_paths: set[str] = set()

        for hash_val, indexed_paths in hash_to_indexed.items():
            # Paths in DB that are no longer on disk at their original location
            orphaned = indexed_paths - current_rel_paths
            if not orphaned:
                continue

            # On-disk paths with the same hash that are not yet in the DB
            new_paths = current_hash_to_paths.get(hash_val, set()) - set(
                indexed_file_hashes.keys()
            )
            if not new_paths:
                continue

            if len(orphaned) == 1 and len(new_paths) == 1:
                # Unambiguous 1-to-1 move
                old_path = next(iter(orphaned))
                new_path = next(iter(new_paths))
                moves.append((old_path, new_path, hash_val))
                moved_old_paths.add(old_path)
            elif len(orphaned) == len(new_paths):
                # Same number of orphaned and new paths with matching hash
                # (e.g., directory rename).  Match by sorted name for stability.
                for old_p, new_p in zip(
                    sorted(orphaned), sorted(new_paths), strict=True
                ):
                    moves.append((old_p, new_p, hash_val))
                    moved_old_paths.add(old_p)

        return moves, moved_old_paths

    async def _phase1_chunk_files(
        self, files: list[Path], force: bool = False
    ) -> tuple[int, int]:
        """Phase 1: Parse and chunk files, store to chunks.lance.

        This phase is fast and durable - no expensive embedding generation.
        Incremental updates are supported via file_hash change detection.

        Args:
            files: Files to process
            force: If True, re-chunk even if unchanged

        Returns:
            Tuple of (files_processed, chunks_created)
        """
        files_processed = 0
        chunks_created = 0

        logger.info(
            f"📄 Phase 1: Chunking {len(files)} files (parsing and extracting code structure)..."
        )

        # Get metrics tracker
        metrics_tracker = get_metrics_tracker()

        # Track start time for progress bar ETA
        phase_start_time = time.time()

        with metrics_tracker.phase("parsing") as parsing_metrics:
            # OPTIMIZATION: Collect files that need deletion and batch delete upfront
            # This avoids O(n) delete_file_chunks() calls that each load the database
            files_to_delete = []
            files_to_process = []

            # OPTIMIZATION: Load all indexed file hashes ONCE for O(1) per-file lookup
            # This replaces 39K per-file database queries with a single scan
            indexed_file_hashes = {}
            if not force:
                logger.info("Loading indexed file hashes for change detection...")
                indexed_file_hashes = (
                    await self.chunks_backend.get_all_indexed_file_hashes()
                )
                logger.info(
                    f"Loaded {len(indexed_file_hashes)} indexed files for change detection"
                )

            # Detect file moves/renames — update metadata instead of re-chunking
            if not force:
                detected_moves, _moved_old_paths = self._detect_file_moves(
                    files, indexed_file_hashes
                )
                if detected_moves:
                    logger.info(
                        f"Detected {len(detected_moves)} file move(s), updating metadata..."
                    )
                    for old_path, new_path, _file_hash in detected_moves:
                        chunks_updated = await self.chunks_backend.update_file_path(
                            old_path, new_path
                        )
                        vectors_updated = await self.vectors_backend.update_file_path(
                            old_path, new_path
                        )
                        logger.info(
                            f"  Moved: {old_path} -> {new_path} "
                            f"({chunks_updated} chunks, {vectors_updated} vectors)"
                        )
                    # Reload so change detection sees updated paths
                    indexed_file_hashes = (
                        await self.chunks_backend.get_all_indexed_file_hashes()
                    )

            # Log start of hash computation phase
            if force:
                logger.info("Force mode enabled, skipping file hash computation...")
            else:
                logger.info(
                    f"Computing file hashes for change detection ({len(files)} files)..."
                )

            for idx, file_path in enumerate(files):
                try:
                    # Log progress every 1000 files
                    if idx > 0 and idx % 1000 == 0:
                        logger.info(f"Computing file hashes: {idx}/{len(files)}")

                    # Skip hash computation when force=True (all files will be reindexed)
                    if force:
                        file_hash = ""  # Empty hash when forcing full reindex
                    else:
                        # Compute current file hash for change detection
                        file_hash = compute_file_hash(file_path)

                    rel_path = str(file_path.relative_to(self.project_root))

                    # Check if file changed (skip if unchanged and not forcing)
                    # OPTIMIZATION: O(1) dict lookup instead of per-file database query
                    if not force:
                        stored_hash = indexed_file_hashes.get(rel_path)
                        if stored_hash is not None and stored_hash == file_hash:
                            logger.debug(f"Skipping unchanged file: {rel_path}")
                            continue

                    # Mark file for deletion and processing
                    files_to_delete.append(rel_path)
                    files_to_process.append((file_path, rel_path, file_hash))

                except Exception as e:
                    logger.error(f"Failed to check file {file_path}: {e}")
                    continue

            # Log completion of hash computation phase
            if not force:
                logger.info(
                    f"File hash computation complete: {len(files_to_process)} files changed, {len(files) - len(files_to_process)} unchanged"
                )

            # Batch delete old chunks for all files (skip if atomic rebuild is active)
            # WORKAROUND: Skip delete on macOS to avoid SIGBUS crash
            # LanceDB delete() triggers memory-mapped file compaction which conflicts
            # with PyTorch's memory-mapped model files on Apple Silicon.
            # Duplicate chunks will be handled by deduplication logic during query.
            if (
                not self._atomic_rebuild_active
                and files_to_delete
                and platform.system() != "Darwin"
            ):
                deleted_count = await self.chunks_backend.delete_files_batch(
                    files_to_delete
                )
                if deleted_count > 0:
                    logger.info(
                        f"Batch deleted {deleted_count} old chunks for {len(files_to_delete)} files"
                    )
            elif files_to_delete and platform.system() == "Darwin":
                logger.debug(
                    f"Skipping batch delete on macOS for {len(files_to_delete)} files "
                    "(defer cleanup to avoid SIGBUS)"
                )

            # Now process files for parsing and chunking
            backend_fatal_error: Exception | None = None
            for _idx, (file_path, rel_path, file_hash) in enumerate(files_to_process):
                # Check memory before processing each file (CRITICAL: not every 100!)
                is_ok, usage_pct, status = self.memory_monitor.check_memory_limit()
                if not is_ok:
                    logger.warning(
                        f"Memory limit exceeded during chunking "
                        f"({usage_pct * 100:.1f}% of {self.memory_monitor.max_memory_gb:.1f}GB), "
                        "waiting for memory to free up..."
                    )
                    await self.memory_monitor.wait_for_memory_available(
                        target_pct=self.memory_monitor.warn_threshold
                    )

                try:
                    # Parse file (runs in thread pool to avoid blocking event loop)
                    chunks = await self.chunk_processor.parse_file(file_path)

                    if not chunks:
                        logger.debug(f"No chunks extracted from {file_path}")
                        continue

                    # Build hierarchical relationships (CPU-bound, run in thread pool)
                    chunks_with_hierarchy = await asyncio.to_thread(
                        self.chunk_processor.build_chunk_hierarchy, chunks
                    )

                    # Convert CodeChunk objects to dicts for storage
                    chunk_dicts = []
                    for chunk in chunks_with_hierarchy:
                        chunk_dict = {
                            "chunk_id": chunk.chunk_id,
                            "file_path": rel_path,
                            "content": chunk.content,
                            "language": chunk.language,
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                            "start_char": 0,  # Not tracked in CodeChunk
                            "end_char": 0,  # Not tracked in CodeChunk
                            "chunk_type": chunk.chunk_type,
                            "name": chunk.function_name
                            or chunk.class_name
                            or "",  # Primary name
                            "parent_name": "",  # Could derive from parent_chunk_id if needed
                            "hierarchy_path": self._build_hierarchy_path(
                                chunk
                            ),  # e.g., "MyClass.my_method"
                            "docstring": chunk.docstring or "",
                            "signature": "",  # Not directly in CodeChunk, could build from params
                            "complexity": int(chunk.complexity_score),
                            "token_count": len(chunk.content.split()),  # Rough estimate
                            # Git blame metadata
                            "last_author": chunk.last_author or "",
                            "last_modified": chunk.last_modified or "",
                            "commit_hash": chunk.commit_hash or "",
                            # Code relationships (for KG)
                            "calls": chunk.calls or [],
                            "imports": [
                                json.dumps(imp) if isinstance(imp, dict) else imp
                                for imp in (chunk.imports or [])
                            ],
                            "inherits_from": chunk.inherits_from or [],
                        }
                        chunk_dicts.append(chunk_dict)

                    # Store chunks (without embeddings) to chunks.lance
                    if chunk_dicts:
                        count = await self.chunks_backend.add_chunks(
                            chunk_dicts, file_hash
                        )
                        chunks_created += count
                        files_processed += 1
                        logger.debug(f"Chunked {count} chunks from {rel_path}")

                        # Periodic GC to prevent Arrow buffer accumulation on Linux.
                        # LanceDB issue #2512: each append allocates Arrow buffers that
                        # CPython's reference counter does not release quickly enough,
                        # leading to RSS growth proportional to file count.
                        if files_processed % 1000 == 0:
                            gc.collect()
                            logger.debug(
                                "GC collect after %d files (Arrow buffer cleanup)",
                                files_processed,
                            )

                        # Show progress bar if tracker is available (update every file)
                        if self.progress_tracker:
                            self.progress_tracker.progress_bar_with_eta(
                                current=files_processed,
                                total=len(files_to_process),
                                prefix="Parsing files",
                                start_time=phase_start_time,
                            )

                except DatabaseError as e:
                    # Check if this is a backend-level failure (e.g., stale/corrupt table)
                    # that will affect every subsequent file — abort early instead of
                    # spamming identical errors for each remaining file.
                    err_lower = str(e).lower()
                    if "not found" in err_lower or "not initialized" in err_lower:
                        logger.error(
                            f"Backend error while storing chunks for {file_path}: {e}\n"
                            f"  Aborting Phase 1 to avoid repeated failures.\n"
                            f"  Try: mvs index --force  (to rebuild from scratch)"
                        )
                        backend_fatal_error = e
                        break
                    logger.error(f"Failed to chunk file {file_path}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Failed to chunk file {file_path}: {e}")
                    continue

            if backend_fatal_error is not None:
                logger.error(
                    f"Phase 1 aborted due to backend error: {backend_fatal_error}"
                )

            # Update metrics
            parsing_metrics.item_count = files_processed

        # Track chunking separately
        with metrics_tracker.phase("chunking") as chunking_metrics:
            chunking_metrics.item_count = chunks_created
            # Note: Duration will be minimal since actual work was done in parsing phase
            # This is just for tracking the chunk count

        # Log memory summary after Phase 1
        self.memory_monitor.log_memory_summary()
        logger.info(
            f"✓ Phase 1 complete: {files_processed} files processed, {chunks_created} chunks created"
        )
        return files_processed, chunks_created

    def _build_hierarchy_path(self, chunk: CodeChunk) -> str:
        """Build dotted hierarchy path (e.g., MyClass.my_method)."""
        parts = []
        if chunk.class_name:
            parts.append(chunk.class_name)
        if chunk.function_name:
            parts.append(chunk.function_name)
        return ".".join(parts) if parts else ""

    def _get_project_name(self, rel_path: str) -> str:
        """Get monorepo subproject name for a file path.

        Args:
            rel_path: Relative file path (e.g., 'packages/frontend/src/App.tsx')

        Returns:
            Subproject name or empty string for non-monorepo projects.
        """
        if not self.monorepo_detector.is_monorepo():
            return ""

        try:
            subprojects = self.monorepo_detector.detect_subprojects()
            file_path = Path(rel_path)
            for sp in subprojects:
                # Check if file is under this subproject's path
                try:
                    file_path.relative_to(sp.relative_path)
                    return sp.name
                except ValueError:
                    continue
            return ""  # File not in any subproject (root-level file)
        except Exception:
            return ""

    async def _phase2_embed_chunks(
        self, batch_size: int = 10000, checkpoint_interval: int = 50000
    ) -> tuple[int, int]:
        """Phase 2: Embed pending chunks, store to vectors.lance.

        This phase is resumable - can restart after crashes. Only embeds
        chunks that are in "pending" status in chunks.lance.

        Args:
            batch_size: Chunks per embedding batch
            checkpoint_interval: Chunks between checkpoint logs

        Returns:
            Tuple of (chunks_embedded, batches_processed)
        """
        chunks_embedded = 0
        batches_processed = 0
        batch_id = int(datetime.now().timestamp())

        logger.info(
            "🧠 Phase 2: Embedding pending chunks (GPU processing for semantic search)..."
        )

        # Get metrics tracker
        metrics_tracker = get_metrics_tracker()

        # Track start time for progress bar ETA
        embed_start_time = time.time()

        # Get total pending chunks count BEFORE starting (for accurate progress bar)
        total_pending_chunks = await self.chunks_backend.count_pending_chunks()
        logger.info(f"Found {total_pending_chunks:,} pending chunks to embed")

        with metrics_tracker.phase("embedding") as embedding_metrics:
            while True:
                # CRITICAL: Check memory BEFORE loading batch to prevent spikes
                usage_pct = self.memory_monitor.get_memory_usage_pct()

                # Proactively reduce batch size if memory is high
                while usage_pct > 0.90 and batch_size > 100:
                    batch_size = batch_size // 2
                    logger.warning(
                        f"⚠️  Memory at {usage_pct * 100:.1f}%, proactively reducing batch to {batch_size}"
                    )
                    usage_pct = self.memory_monitor.get_memory_usage_pct()

                # If still over 90%, wait for memory to free
                if usage_pct > 0.90:
                    logger.warning(
                        f"Memory usage at {usage_pct * 100:.1f}%, waiting for memory to drop..."
                    )
                    await self.memory_monitor.wait_for_memory_available(target_pct=0.80)

                # Get pending chunks from chunks.lance FIRST
                pending = await self.chunks_backend.get_pending_chunks(batch_size)
                if not pending:
                    logger.info("No more pending chunks to embed")
                    break

                # Check memory AFTER loading batch (when memory is at peak)
                is_ok, usage_pct, status = self.memory_monitor.check_memory_limit()

                if not is_ok:
                    # Memory limit exceeded after loading - apply backpressure
                    logger.warning(
                        f"Memory limit exceeded after loading batch "
                        f"({usage_pct * 100:.1f}% of {self.memory_monitor.max_memory_gb:.1f}GB), "
                        "waiting for memory to free up..."
                    )
                    await self.memory_monitor.wait_for_memory_available(
                        target_pct=self.memory_monitor.warn_threshold
                    )

                # Adjust batch size based on memory pressure
                adjusted_batch_size = self.memory_monitor.get_adjusted_batch_size(
                    batch_size, min_batch_size=100
                )
                if adjusted_batch_size != batch_size:
                    logger.info(
                        f"Adjusted embedding batch size: {batch_size} → {adjusted_batch_size} "
                        f"(memory usage: {usage_pct * 100:.1f}%)"
                    )
                    batch_size = adjusted_batch_size

                # Mark as processing (for crash recovery)
                chunk_ids = [c["chunk_id"] for c in pending]
                await self.chunks_backend.mark_chunks_processing(chunk_ids, batch_id)

                try:
                    # Generate embeddings using database's embedding function
                    contents = [c["content"] for c in pending]

                    # CRITICAL: Check memory before expensive embedding operation
                    is_ok, usage_pct, status = self.memory_monitor.check_memory_limit()
                    if not is_ok:
                        logger.error(
                            f"Memory limit exceeded before embedding "
                            f"({usage_pct * 100:.1f}% of {self.memory_monitor.max_memory_gb:.1f}GB). "
                            f"Reducing batch size and retrying..."
                        )

                        # Return chunks to pending state
                        await self.chunks_backend.mark_chunks_pending(chunk_ids)

                        # Wait for memory to free
                        await self.memory_monitor.wait_for_memory_available(
                            target_pct=self.memory_monitor.warn_threshold
                        )

                        # Dramatically reduce batch size
                        batch_size = max(100, batch_size // 4)
                        logger.info(
                            f"Reduced batch size to {batch_size} due to memory pressure"
                        )
                        continue  # Skip to next iteration with smaller batch

                    # Generate embeddings in batch
                    vectors = None

                    # Method 1: Check for _embedding_function (ChromaDB)
                    if hasattr(self.database, "_embedding_function"):
                        vectors = self.database._embedding_function(contents)
                    # Method 2: Check for _collection and its embedding function
                    elif hasattr(self.database, "_collection") and hasattr(
                        self.database._collection, "_embedding_function"
                    ):
                        vectors = self.database._collection._embedding_function(
                            contents
                        )
                    # Method 3: Use database.embedding_function (proper API)
                    elif hasattr(self.database, "embedding_function"):
                        vectors = self.database.embedding_function.embed_documents(
                            contents
                        )
                    else:
                        logger.error(
                            "Cannot access embedding function from database, "
                            "skipping batch embedding"
                        )
                        await self.chunks_backend.mark_chunks_complete(chunk_ids)
                        chunks_embedded += len(pending)
                        batches_processed += 1
                        continue

                    if vectors is None:
                        raise ValueError("Failed to generate embeddings")

                    # Add to vectors table with embeddings
                    chunks_with_vectors = []
                    for chunk, vec in zip(pending, vectors, strict=True):
                        # Derive function_name and class_name from chunk data
                        chunk_type = chunk.get("chunk_type", "")
                        chunk_name = chunk.get("name", "")
                        hierarchy = chunk.get("hierarchy_path", "")

                        # Determine function_name and class_name based on chunk_type
                        if chunk_type in ("function", "method"):
                            fn_name = chunk_name
                            cls_name = ""
                            # If hierarchy has a dot, the part before is the class
                            if "." in hierarchy:
                                parts = hierarchy.split(".")
                                cls_name = parts[0]
                                fn_name = parts[-1]
                        elif chunk_type == "class":
                            fn_name = ""
                            cls_name = chunk_name
                        else:
                            fn_name = ""
                            cls_name = ""

                        chunk_with_vec = {
                            "chunk_id": chunk["chunk_id"],
                            "vector": vec,
                            "file_path": chunk["file_path"],
                            "content": chunk["content"],
                            "language": chunk["language"],
                            "start_line": chunk["start_line"],
                            "end_line": chunk["end_line"],
                            "chunk_type": chunk["chunk_type"],
                            "name": chunk["name"],
                            "function_name": fn_name,
                            "class_name": cls_name,
                            "project_name": self._get_project_name(chunk["file_path"]),
                            "hierarchy_path": chunk["hierarchy_path"],
                        }
                        chunks_with_vectors.append(chunk_with_vec)

                    # Store vectors to vectors.lance
                    model_name = self.get_embedding_model_name()
                    await self.vectors_backend.add_vectors(
                        chunks_with_vectors, model_version=model_name
                    )

                    # Mark as complete in chunks.lance
                    await self.chunks_backend.mark_chunks_complete(chunk_ids)

                    chunks_embedded += len(pending)
                    batches_processed += 1

                    # Show progress bar if tracker is available
                    if self.progress_tracker and total_pending_chunks > 0:
                        self.progress_tracker.progress_bar_with_eta(
                            current=chunks_embedded,
                            total=total_pending_chunks,
                            prefix="Embedding chunks",
                            start_time=embed_start_time,
                        )

                    # Checkpoint logging with memory status
                    if chunks_embedded % checkpoint_interval == 0:
                        self.memory_monitor.log_memory_summary()
                        if not self.progress_tracker:
                            logger.info(
                                f"Checkpoint: {chunks_embedded} chunks embedded"
                            )

                    # Check memory after each batch and clear references
                    del pending, chunk_ids, contents, vectors, chunks_with_vectors

                except Exception as e:
                    logger.error(f"Embedding batch failed: {e}")
                    # Mark chunks as error in chunks.lance
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    await self.chunks_backend.mark_chunks_error(chunk_ids, error_msg)
                    # Continue with next batch instead of crashing
                    continue

            # Update metrics
            embedding_metrics.item_count = chunks_embedded

        # Log memory summary after Phase 2
        self.memory_monitor.log_memory_summary()
        logger.info(
            f"✓ Phase 2 complete: {chunks_embedded} chunks embedded in {batches_processed} batches"
        )
        return chunks_embedded, batches_processed

    async def re_embed_chunks(
        self, batch_size: int = 10000, checkpoint_interval: int = 50000
    ) -> tuple[int, int]:
        """Re-embed all chunks with current embedding model without re-parsing.

        This method:
        1. Reads all existing chunks from chunks backend
        2. Checks for dimension mismatch with current embedding model
        3. Drops and recreates vectors table if dimensions changed
        4. Resets all chunks to pending status
        5. Runs Phase 2 embedding with current model

        Useful for:
        - Upgrading embedding models (e.g., MiniLM 384D -> GraphCodeBERT 768D)
        - Re-embedding with better models without re-parsing files
        - Fixing corrupted embeddings

        Args:
            batch_size: Chunks per embedding batch
            checkpoint_interval: Chunks between checkpoint logs

        Returns:
            Tuple of (chunks_re_embedded, batches_processed)

        Raises:
            DatabaseNotInitializedError: If backends not initialized
        """
        from ..config.defaults import get_model_dimensions

        logger.info("🔄 Re-embedding all chunks with current embedding model...")

        # Get current embedding model dimensions
        if hasattr(self.database, "_embedding_function"):
            model_name = self.database._embedding_function.model_name
            expected_dim = get_model_dimensions(model_name)
            logger.info(f"Current embedding model: {model_name} ({expected_dim}D)")
        else:
            logger.error("Cannot access embedding model from database")
            raise ValueError("Cannot determine embedding model dimensions")

        # Check for dimension mismatch with existing vectors table
        if await self.vectors_backend.check_dimension_mismatch(expected_dim):
            logger.warning(
                f"Dimension mismatch detected! Recreating vectors table with {expected_dim}D..."
            )
            await self.vectors_backend.recreate_table_with_new_dimensions(expected_dim)
        else:
            logger.info(
                f"Vector dimensions match ({expected_dim}D), dropping existing vectors..."
            )
            # Drop vectors table to re-embed
            await self.vectors_backend.recreate_table_with_new_dimensions(expected_dim)

        # Reset all chunks to pending status
        logger.info("Resetting all chunks to pending status...")
        await self.chunks_backend.reset_all_to_pending()

        # Get total chunk count for progress
        total_chunks = await self.chunks_backend.count_chunks()
        logger.info(f"Found {total_chunks:,} chunks to re-embed")

        # Run Phase 2 embedding (reuses existing logic)
        chunks_embedded, batches_processed = await self._phase2_embed_chunks(
            batch_size=batch_size, checkpoint_interval=checkpoint_interval
        )

        logger.info(
            f"✓ Re-embedding complete: {chunks_embedded:,} chunks embedded with {model_name}"
        )
        return chunks_embedded, batches_processed

    async def _atomic_rebuild_databases(self, force: bool = False) -> bool:
        """Atomically rebuild databases when force is enabled.

        Strategy:
        1. Create new databases with .new suffix
        2. Build index into new databases
        3. On success: rename old to .old, rename new to final, delete .old
        4. On failure: delete .new, keep old database intact

        Args:
            force: If True, perform atomic rebuild

        Returns:
            True if atomic rebuild was performed, False otherwise
        """
        if not force:
            return False

        base_path = self.project_root / ".mcp-vector-search"

        # Database paths (only .new paths used in this method - others in _finalize)
        lance_new = base_path / "lance.new"
        kg_new = base_path / "knowledge_graph.new"
        chroma_new = base_path / "chroma.sqlite3.new"
        code_search_new = base_path / "code_search.lance.new"

        try:
            # Step 1: Rename existing databases to .new (build target)
            logger.info("🔄 Atomic rebuild: preparing new database directories...")

            # Remove any stale .new directories from previous interrupted rebuilds
            for new_path in [lance_new, kg_new, code_search_new]:
                if new_path.exists():
                    try:
                        shutil.rmtree(new_path)
                        logger.debug(f"Removed stale directory: {new_path}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to remove stale directory {new_path}: {e}. "
                            "Attempting to continue anyway."
                        )
            if chroma_new.exists():
                try:
                    chroma_new.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove stale file {chroma_new}: {e}")

            # Create new database directories
            lance_new.mkdir(parents=True, exist_ok=True)
            kg_new.mkdir(parents=True, exist_ok=True)

            # Update backend paths to point to .new databases
            self.chunks_backend = ChunksBackend(lance_new)
            self.vectors_backend = VectorsBackend(lance_new)

            # Update database path for LanceDB backend
            # This requires modifying the database's persist_directory
            # We'll handle this by creating a new database instance pointing to .new
            from .embeddings import create_embedding_function
            from .factory import create_database

            # Create new database instance for .new location
            # Pass config model to create_embedding_function (None = auto-select by device)
            model_name = self.config.embedding_model if self.config else None
            embedding_function, _ = create_embedding_function(model_name=model_name)
            new_db_path = base_path / "code_search.lance.new"
            self.database = create_database(
                persist_directory=new_db_path, embedding_function=embedding_function
            )

            logger.info("✓ New database directories prepared")
            # Set flag to skip delete operations (building into fresh database)
            self._atomic_rebuild_active = True
            return True

        except Exception as e:
            logger.error(f"Failed to prepare atomic rebuild: {e}")
            # Cleanup .new directories on failure
            for new_path in [lance_new, kg_new, code_search_new]:
                if new_path.exists():
                    shutil.rmtree(new_path, ignore_errors=True)
            if chroma_new.exists():
                chroma_new.unlink()
            self._atomic_rebuild_active = False
            return False

    async def _finalize_atomic_rebuild(self) -> None:
        """Finalize atomic rebuild by atomically switching databases.

        Called after successful indexing when force_reindex=True.
        Performs atomic rename operations to switch from .new to final.
        """
        base_path = self.project_root / ".mcp-vector-search"

        # Database paths
        lance_path = base_path / "lance"
        lance_new = base_path / "lance.new"
        lance_old = base_path / "lance.old"

        kg_path = base_path / "knowledge_graph"
        kg_new = base_path / "knowledge_graph.new"
        kg_old = base_path / "knowledge_graph.old"

        chroma_path = base_path / "chroma.sqlite3"
        chroma_new = base_path / "chroma.sqlite3.new"
        chroma_old = base_path / "chroma.sqlite3.old"

        code_search_path = base_path / "code_search.lance"
        code_search_new = base_path / "code_search.lance.new"
        code_search_old = base_path / "code_search.lance.old"

        try:
            logger.info("🔄 Atomic rebuild: finalizing database switch...")

            # Step 2: Atomic switch for lance directory
            if lance_new.exists():
                if lance_path.exists():
                    lance_path.rename(lance_old)
                lance_new.rename(lance_path)
                if lance_old.exists():
                    try:
                        shutil.rmtree(lance_old)
                        logger.debug(f"Removed old directory: {lance_old}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to remove old directory {lance_old}: {e}"
                        )

            # Step 3: Atomic switch for knowledge_graph directory
            if kg_new.exists():
                if kg_path.exists():
                    kg_path.rename(kg_old)
                kg_new.rename(kg_path)
                if kg_old.exists():
                    try:
                        shutil.rmtree(kg_old)
                        logger.debug(f"Removed old directory: {kg_old}")
                    except Exception as e:
                        logger.warning(f"Failed to remove old directory {kg_old}: {e}")

            # Step 4: Atomic switch for code_search.lance
            if code_search_new.exists():
                if code_search_path.exists():
                    code_search_path.rename(code_search_old)
                code_search_new.rename(code_search_path)
                if code_search_old.exists():
                    try:
                        shutil.rmtree(code_search_old)
                        logger.debug(f"Removed old directory: {code_search_old}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to remove old directory {code_search_old}: {e}"
                        )

            # Step 5: Handle chroma.sqlite3 (deprecated, but may exist)
            if chroma_new.exists():
                if chroma_path.exists():
                    chroma_path.rename(chroma_old)
                chroma_new.rename(chroma_path)
                if chroma_old.exists():
                    chroma_old.unlink()

            logger.info("✓ Atomic rebuild complete: databases switched successfully")
            # Reset flag after successful finalization
            self._atomic_rebuild_active = False

        except Exception as e:
            logger.error(f"Failed to finalize atomic rebuild: {e}")
            # Attempt to rollback
            try:
                logger.warning("Attempting rollback to old databases...")
                if lance_old.exists() and not lance_path.exists():
                    lance_old.rename(lance_path)
                if kg_old.exists() and not kg_path.exists():
                    kg_old.rename(kg_path)
                if code_search_old.exists() and not code_search_path.exists():
                    code_search_old.rename(code_search_path)
                if chroma_old.exists() and not chroma_path.exists():
                    chroma_old.rename(chroma_path)
                logger.info("✓ Rollback successful, old databases restored")
            except Exception as rollback_error:
                logger.error(f"Rollback failed: {rollback_error}")
            # Reset flag after rollback attempt
            self._atomic_rebuild_active = False
            raise

    async def chunk_files(self, fresh: bool = False) -> dict:
        """Chunk files independently without embedding.

        Discovers files, parses/chunks them, saves to chunks.lance,
        and builds the BM25 index. Can run independently from embed.

        Args:
            fresh: If True, clear existing chunks and re-chunk all files.
                   If False, only chunk new/changed files (incremental).

        Returns:
            Dict with stats: files_processed, chunks_created, files_skipped, errors
        """
        logger.info(
            f"Starting chunk_files (fresh={fresh}) for project: {self.project_root}"
        )

        # Handle atomic rebuild for fresh mode FIRST (replaces backend objects)
        if fresh:
            # Pre-initialize backends so _atomic_rebuild_databases can create new ones
            try:
                await self.chunks_backend.initialize()
                await self.vectors_backend.initialize()
            except DatabaseInitializationError as e:
                logger.error(
                    f"Cannot start chunking: {e}\n"
                    f"  Try: mvs index --force  (to rebuild from scratch)\n"
                    f"  Or:  mvs reset index --force  (to clear all data)"
                )
                return {
                    "files_processed": 0,
                    "chunks_created": 0,
                    "files_skipped": 0,
                    "errors": [str(e)],
                }
            atomic_rebuild_active = await self._atomic_rebuild_databases(True)
            self._atomic_rebuild_active = atomic_rebuild_active
        else:
            atomic_rebuild_active = False

        # Initialize backends (new backends after atomic rebuild, or original ones)
        if self.chunks_backend._db is None:
            try:
                await self.chunks_backend.initialize()
            except DatabaseInitializationError as e:
                logger.error(
                    f"Cannot start chunking: {e}\n"
                    f"  Try: mvs index --force  (to rebuild from scratch)\n"
                    f"  Or:  mvs reset index --force  (to clear all data)"
                )
                return {
                    "files_processed": 0,
                    "chunks_created": 0,
                    "files_skipped": 0,
                    "errors": [str(e)],
                }
        if self.vectors_backend._db is None:
            try:
                await self.vectors_backend.initialize()
            except DatabaseInitializationError as e:
                logger.error(
                    f"Cannot start chunking: {e}\n"
                    f"  Try: mvs index --force  (to rebuild from scratch)\n"
                    f"  Or:  mvs reset index --force  (to clear all data)"
                )
                return {
                    "files_processed": 0,
                    "chunks_created": 0,
                    "files_skipped": 0,
                    "errors": [str(e)],
                }

        # Run auto migrations
        await self._run_auto_migrations()

        # Apply auto-optimizations if enabled
        if self.auto_optimize:
            self.apply_auto_optimizations()

        # Clean up stale lock files
        cleanup_stale_locks(self.project_root)

        # Discover all indexable files
        all_files = self.file_discovery.find_indexable_files()

        # Track stats
        files_processed = 0
        chunks_created = 0
        files_skipped = 0
        errors = []

        if not all_files:
            logger.warning("No indexable files found")
            return {
                "files_processed": 0,
                "chunks_created": 0,
                "files_skipped": 0,
                "errors": [],
            }

        # Filter files for incremental mode
        files_to_index = all_files
        if not fresh:
            # Incremental: only process changed files
            logger.info(
                f"Incremental mode: checking {len(all_files)} files for changes..."
            )

            # OPTIMIZATION: Load all indexed file hashes ONCE for O(1) per-file lookup
            indexed_file_hashes = (
                await self.chunks_backend.get_all_indexed_file_hashes()
            )
            logger.info(
                f"Loaded {len(indexed_file_hashes)} indexed file hashes for change detection"
            )

            if not indexed_file_hashes and all_files:
                logger.warning(
                    f"No indexed file hashes found — all {len(all_files)} files will be "
                    "processed (first index or hash table unavailable)"
                )

            # Detect file moves/renames — update metadata instead of re-chunking
            detected_moves, _moved_old_paths = self._detect_file_moves(
                all_files, indexed_file_hashes
            )
            if detected_moves:
                logger.info(
                    f"Detected {len(detected_moves)} file move(s), updating metadata..."
                )
                for old_path, new_path, _file_hash in detected_moves:
                    chunks_updated = await self.chunks_backend.update_file_path(
                        old_path, new_path
                    )
                    vectors_updated = await self.vectors_backend.update_file_path(
                        old_path, new_path
                    )
                    logger.info(
                        f"  Moved: {old_path} -> {new_path} "
                        f"({chunks_updated} chunks, {vectors_updated} vectors)"
                    )
                # Reload so change detection sees updated paths
                indexed_file_hashes = (
                    await self.chunks_backend.get_all_indexed_file_hashes()
                )

            filtered_files = []
            files_checked = 0
            for f in all_files:
                try:
                    file_hash = compute_file_hash(f)
                    rel_path = str(f.relative_to(self.project_root))

                    # OPTIMIZATION: O(1) dict lookup instead of per-file database query
                    stored_hash = indexed_file_hashes.get(rel_path)
                    if stored_hash is not None and stored_hash == file_hash:
                        files_skipped += 1
                    else:
                        filtered_files.append(f)
                except Exception as e:
                    logger.warning(f"Error checking file {f}: {e}, will re-index")
                    filtered_files.append(f)
                    errors.append(str(e))

                files_checked += 1
                if files_checked % 1000 == 0:
                    logger.info(
                        f"Change detection progress: {files_checked}/{len(all_files)} files checked..."
                    )

            files_to_index = filtered_files
            logger.info(
                f"Change detection: {len(files_to_index)} changed, "
                f"{files_skipped} unchanged out of {len(all_files)} total files"
            )

        # Process files through Phase 1
        if files_to_index:
            indexed_count, created = await self._phase1_chunk_files(
                files_to_index, force=fresh
            )
            files_processed = indexed_count
            chunks_created = created

        # Update metadata after chunking
        if files_processed > 0:
            # Update directory index
            try:
                logger.debug("Updating directory index after chunking...")
                chunk_stats = {}
                for file_path in files_to_index:
                    try:
                        mtime = os.path.getmtime(file_path)
                        chunk_stats[str(file_path)] = {
                            "modified": mtime,
                            "chunks": 1,  # Placeholder
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
                errors.append(f"Directory index update failed: {e}")

        # Build BM25 index
        await self._build_bm25_index()

        # Finalize atomic rebuild if fresh and we processed files
        if fresh and atomic_rebuild_active and files_processed > 0:
            await self._finalize_atomic_rebuild()

            # CRITICAL: Re-initialize backends after finalization
            # After _finalize_atomic_rebuild(), the .new directories have been renamed
            # to final paths, but backend objects still point to old .new paths.
            # We must reinitialize backends to point to the correct final paths.
            base_path = self.project_root / ".mcp-vector-search"
            lance_path = base_path / "lance"

            # Create fresh backend instances pointing to final paths
            from .chunks_backend import ChunksBackend
            from .vectors_backend import VectorsBackend

            self.chunks_backend = ChunksBackend(lance_path)
            self.vectors_backend = VectorsBackend(lance_path)

            # Initialize new backends
            await self.chunks_backend.initialize()
            await self.vectors_backend.initialize()
            logger.debug("Backends re-initialized after atomic rebuild finalization")

        logger.info(
            f"✓ chunk_files complete: {files_processed} files processed, "
            f"{chunks_created} chunks created, {files_skipped} files skipped"
        )

        return {
            "files_processed": files_processed,
            "chunks_created": chunks_created,
            "files_skipped": files_skipped,
            "errors": errors,
        }

    async def embed_chunks(self, fresh: bool = False, batch_size: int = 512) -> dict:
        """Embed pending chunks independently without re-chunking.

        Reads chunks with embedding_status='pending' from chunks.lance,
        generates embeddings via GPU, and saves to vectors.lance.
        Can run independently from chunk_files.

        Args:
            fresh: If True, clear vectors table and reset all chunks to pending.
                   If False, only embed chunks that haven't been embedded yet.
            batch_size: Number of chunks per embedding batch.

        Returns:
            Dict with stats: chunks_embedded, chunks_skipped, batches_processed, errors, duration_seconds
        """
        from ..config.defaults import get_model_dimensions

        logger.info(
            f"Starting embed_chunks (fresh={fresh}) for project: {self.project_root}"
        )

        # Initialize backends (skip if already initialized)
        if self.chunks_backend._db is None:
            try:
                await self.chunks_backend.initialize()
            except DatabaseInitializationError as e:
                logger.error(
                    f"Cannot start embedding: {e}\n"
                    f"  Try: mvs index --force  (to rebuild from scratch)\n"
                    f"  Or:  mvs reset index --force  (to clear all data)"
                )
                return {
                    "chunks_embedded": 0,
                    "chunks_skipped": 0,
                    "batches_processed": 0,
                    "errors": [str(e)],
                    "duration_seconds": 0.0,
                }
        if self.vectors_backend._db is None:
            try:
                await self.vectors_backend.initialize()
            except DatabaseInitializationError as e:
                logger.error(
                    f"Cannot start embedding: {e}\n"
                    f"  Try: mvs index --force  (to rebuild from scratch)\n"
                    f"  Or:  mvs reset index --force  (to clear all data)"
                )
                return {
                    "chunks_embedded": 0,
                    "chunks_skipped": 0,
                    "batches_processed": 0,
                    "errors": [str(e)],
                    "duration_seconds": 0.0,
                }

        # Run auto migrations
        await self._run_auto_migrations()

        # Bug fix: Reset stale "processing" chunks from interrupted runs
        # This prevents chunks from being stuck in processing state indefinitely
        try:
            stale_count = await self.chunks_backend.cleanup_stale_processing(
                older_than_minutes=5
            )
            if stale_count > 0:
                logger.info(
                    f"Reset {stale_count} stale processing chunks from interrupted runs"
                )
        except Exception as e:
            logger.warning(f"Failed to reset stale processing chunks: {e}")

        # Apply auto-optimizations if enabled
        if self.auto_optimize:
            self.apply_auto_optimizations()

        # Track start time
        start_time = time.time()
        errors = []

        # Handle fresh mode: clear vectors and reset chunks
        if fresh:
            try:
                # Get current embedding model dimensions using robust model detection
                model_name = self.get_embedding_model_name()
                expected_dim = get_model_dimensions(model_name)
                logger.info(
                    f"Fresh mode: recreating vectors table with {expected_dim}D for model {model_name}"
                )

                # Recreate vectors table with correct dimensions
                await self.vectors_backend.recreate_table_with_new_dimensions(
                    expected_dim
                )

                # Reset all chunks to pending
                await self.chunks_backend.reset_all_to_pending()
                logger.info("Reset all chunks to pending status")
            except Exception as e:
                logger.error(f"Failed to prepare fresh embedding: {e}")
                errors.append(str(e))
                return {
                    "chunks_embedded": 0,
                    "chunks_skipped": 0,
                    "batches_processed": 0,
                    "errors": errors,
                    "duration_seconds": time.time() - start_time,
                }

        # Run Phase 2 embedding
        chunks_embedded, batches_processed = await self._phase2_embed_chunks(
            batch_size=batch_size, checkpoint_interval=50000
        )

        # Build ANN vector index after embedding completes.
        # This activates IVF_SQ approximate nearest-neighbor search for large
        # datasets.  Skipped automatically for small datasets (< 4,096 rows).
        # Non-fatal: search falls back to brute-force if index creation fails.
        await self.vectors_backend.rebuild_index()

        # Build BM25 index after embedding (vectors have changed)
        await self._build_bm25_index()

        duration = time.time() - start_time
        logger.info(
            f"✓ embed_chunks complete: {chunks_embedded} chunks embedded in "
            f"{batches_processed} batches ({duration:.1f}s)"
        )

        return {
            "chunks_embedded": chunks_embedded,
            "chunks_skipped": 0,  # We don't track skipped in current implementation
            "batches_processed": batches_processed,
            "errors": errors,
            "duration_seconds": duration,
        }

    async def chunk_and_embed(self, fresh: bool = False, batch_size: int = 512) -> dict:
        """Full index pipeline: chunk files then embed chunks.

        Convenience wrapper that runs both phases sequentially.
        Equivalent to running chunk_files() followed by embed_chunks().

        Args:
            fresh: If True, start from scratch (clear chunks + vectors).
            batch_size: Number of chunks per embedding batch.

        Returns:
            Combined stats dict from both phases.
        """
        logger.info(
            f"Starting chunk_and_embed (fresh={fresh}) for project: {self.project_root}"
        )

        # Run both phases
        chunk_result = await self.chunk_files(fresh=fresh)
        embed_result = await self.embed_chunks(fresh=fresh, batch_size=batch_size)

        # Combine results
        return {
            "files_processed": chunk_result["files_processed"],
            "chunks_created": chunk_result["chunks_created"],
            "files_skipped": chunk_result["files_skipped"],
            "chunks_embedded": embed_result["chunks_embedded"],
            "batches_processed": embed_result["batches_processed"],
            "duration_seconds": embed_result["duration_seconds"],
            "errors": chunk_result["errors"] + embed_result["errors"],
        }

    async def index_project(
        self,
        force_reindex: bool = False,
        show_progress: bool = True,
        skip_relationships: bool = False,
        phase: str = "all",
        metrics_json: bool = False,
        pipeline: bool = True,
    ) -> int:
        """Index all files in the project using two-phase architecture.

        Args:
            force_reindex: Whether to reindex existing files
            show_progress: Whether to show progress information
            skip_relationships: Skip computing relationships for visualization (faster, but visualize will be slower)
            phase: Which phase to run - "all", "chunk", or "embed"
                - "all": Run both phases (default, backward compatible)
                - "chunk": Only Phase 1 (parse and chunk, no embedding)
                - "embed": Only Phase 2 (embed pending chunks)
            metrics_json: Output metrics as JSON to stdout
            pipeline: Whether to use pipeline parallelism to overlap Phase 1 and Phase 2 (default: True)
                - When True: Phase 1 (parsing) and Phase 2 (embedding) run concurrently
                - When False: Sequential execution (fallback for debugging)

        Returns:
            Number of files indexed (for backward compatibility)

        Note:
            Two-phase architecture enables:
            - Fast Phase 1 for durable chunk storage
            - Resumable Phase 2 for embedding (can restart after crash)
            - Incremental updates (skip unchanged files)

            When force_reindex=True, uses atomic rebuild:
            - Builds into new database with .new suffix
            - Atomically switches on success
            - Keeps old database intact on failure

            When pipeline=True (default):
            - Phase 1 and Phase 2 overlap using producer-consumer pattern
            - Producer: Chunks files in batches, puts chunks on queue
            - Consumer: Embeds chunks from queue concurrently
            - Result: GPU doesn't sit idle during parsing phase (30-50% speedup)
        """
        logger.info(
            f"Starting indexing of project: {self.project_root} (phase: {phase})"
        )

        # Perform atomic rebuild if force is enabled
        atomic_rebuild_active = await self._atomic_rebuild_databases(force_reindex)

        # Initialize metrics tracker
        metrics_tracker = get_metrics_tracker()
        metrics_tracker.reset()

        # Initialize backends
        try:
            await self.chunks_backend.initialize()
        except DatabaseInitializationError as e:
            logger.error(
                f"Cannot start indexing: {e}\n"
                f"  Try: mvs index --force  (to rebuild from scratch)\n"
                f"  Or:  mvs reset index --force  (to clear all data)"
            )
            return 0
        try:
            await self.vectors_backend.initialize()
        except DatabaseInitializationError as e:
            logger.error(
                f"Cannot start indexing: {e}\n"
                f"  Try: mvs index --force  (to rebuild from scratch)\n"
                f"  Or:  mvs reset index --force  (to clear all data)"
            )
            return 0

        # Check and run pending migrations automatically
        await self._run_auto_migrations()

        # Apply auto-optimizations before indexing
        if self.auto_optimize:
            self.apply_auto_optimizations()

        # Clean up stale lock files from previous interrupted indexing runs
        cleanup_stale_locks(self.project_root)

        # Track indexed count for backward compatibility
        indexed_count = 0
        chunks_created = 0
        chunks_embedded = 0
        files_to_index = []  # Initialize to empty list for directory index updates

        # Decide whether to use pipeline or sequential execution
        use_pipeline = pipeline and phase == "all"

        if use_pipeline:
            # PIPELINE MODE: Overlap Phase 1 (chunking) and Phase 2 (embedding)
            logger.info(
                "🔄 Using pipeline parallelism to overlap parsing and embedding phases"
            )
            (
                indexed_count,
                chunks_created,
                chunks_embedded,
            ) = await self._index_with_pipeline(force_reindex)
            # Get files list for directory index updates
            files_to_index = self.file_discovery.find_indexable_files()
        else:
            # SEQUENTIAL MODE: Traditional two-phase execution
            # Phase 1: Chunk files (if requested)
            if phase in ("all", "chunk"):
                # Find all indexable files
                all_files = self.file_discovery.find_indexable_files()

                # Clean up stale metadata entries (files that no longer exist)
                if force_reindex:
                    # Clear all metadata on force reindex
                    logger.info("Force reindex: clearing all metadata entries")
                    self.metadata.save({})
                elif all_files:
                    # Clean up stale entries for incremental index
                    valid_files = {str(f) for f in all_files}
                    removed = self.metadata.cleanup_stale_entries(valid_files)
                    if removed > 0:
                        logger.info(
                            f"Removed {removed} stale metadata entries for deleted files"
                        )

                if not all_files:
                    logger.warning("No indexable files found")
                    if phase == "chunk":
                        return 0
                else:
                    # Filter files that need indexing
                    files_to_index = all_files
                    if not force_reindex:
                        # Use chunks_backend for change detection instead of metadata
                        logger.info(
                            f"Incremental change detection: checking {len(all_files)} files..."
                        )

                        # OPTIMIZATION: Load all indexed file hashes ONCE for O(1) per-file lookup
                        indexed_file_hashes = (
                            await self.chunks_backend.get_all_indexed_file_hashes()
                        )
                        logger.info(
                            f"Loaded {len(indexed_file_hashes)} indexed files for change detection"
                        )

                        # Detect file moves/renames — update metadata instead of re-chunking
                        detected_moves, _moved_old_paths = self._detect_file_moves(
                            all_files, indexed_file_hashes
                        )
                        if detected_moves:
                            logger.info(
                                f"Detected {len(detected_moves)} file move(s), updating metadata..."
                            )
                            for old_path, new_path, _file_hash in detected_moves:
                                chunks_updated = (
                                    await self.chunks_backend.update_file_path(
                                        old_path, new_path
                                    )
                                )
                                vectors_updated = (
                                    await self.vectors_backend.update_file_path(
                                        old_path, new_path
                                    )
                                )
                                logger.info(
                                    f"  Moved: {old_path} -> {new_path} "
                                    f"({chunks_updated} chunks, {vectors_updated} vectors)"
                                )
                            # Reload so change detection sees updated paths
                            indexed_file_hashes = (
                                await self.chunks_backend.get_all_indexed_file_hashes()
                            )

                        filtered_files = []
                        for idx, f in enumerate(all_files, start=1):
                            try:
                                file_hash = compute_file_hash(f)
                                rel_path = str(f.relative_to(self.project_root))

                                # OPTIMIZATION: O(1) dict lookup instead of per-file database query
                                stored_hash = indexed_file_hashes.get(rel_path)
                                if stored_hash is None or stored_hash != file_hash:
                                    filtered_files.append(f)
                            except Exception as e:
                                logger.warning(
                                    f"Error checking file {f}: {e}, will re-index"
                                )
                                filtered_files.append(f)

                            # Progress logging every 500 files
                            if idx % 500 == 0:
                                logger.info(
                                    f"Change detection progress: {idx}/{len(all_files)} files checked"
                                )

                        files_to_index = filtered_files
                        logger.info(
                            f"Incremental index: {len(files_to_index)} of {len(all_files)} files need updating"
                        )
                    else:
                        logger.info(
                            f"Force reindex: processing all {len(files_to_index)} files"
                        )

                    if files_to_index:
                        # Run Phase 1
                        indexed_count, chunks_created = await self._phase1_chunk_files(
                            files_to_index, force=force_reindex
                        )

                        # Update metadata for backward compatibility
                        if indexed_count > 0:
                            metadata_dict = self.metadata.load()
                            for file_path in files_to_index:
                                try:
                                    metadata_dict[str(file_path)] = os.path.getmtime(
                                        file_path
                                    )
                                except OSError:
                                    pass  # File might have been deleted during indexing
                            self.metadata.save(metadata_dict)
                    else:
                        logger.info("All files are up to date")
                        if phase == "chunk":
                            return 0

            # Phase 2: Embed chunks (if requested)
            if phase in ("all", "embed"):
                # Run Phase 2 on pending chunks
                chunks_embedded, _ = await self._phase2_embed_chunks(
                    batch_size=self.batch_size
                )
        if phase == "all":
            logger.info(
                f"✓ Two-phase indexing complete: {indexed_count} files, "
                f"{chunks_created} chunks created, {chunks_embedded} chunks embedded"
            )
        elif phase == "chunk":
            logger.info(
                f"✓ Phase 1 complete: {indexed_count} files, {chunks_created} chunks created"
            )
        elif phase == "embed":
            logger.info(f"✓ Phase 2 complete: {chunks_embedded} chunks embedded")

        # Update directory index (only if files were indexed)
        if indexed_count > 0:
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

        # Mark relationships for background computation (unless skipped)
        # Default behavior: skip blocking computation, mark for background processing
        # OPTIMIZATION: Skip this expensive operation during normal indexing
        # Relationships will be computed on-demand when visualization is requested
        if not skip_relationships and indexed_count > 0:
            logger.info(
                "Relationships will be computed on-demand during visualization (use 'mcp-vector-search index relationships' to pre-compute)"
            )

        # Save trend snapshot after successful indexing
        # OPTIMIZATION: Use stats-only computation to avoid expensive get_all_chunks()
        if indexed_count > 0:
            try:
                logger.info("Saving metrics snapshot for trend tracking...")
                # Get database stats (fast, no full scan)
                stats = await self.database.get_stats()
                # Compute metrics from stats only (no chunk loading)
                # Pass empty list for chunks - trend tracker will use stats-based computation
                metrics = self.trend_tracker.compute_metrics_from_stats(
                    stats.to_dict(), []
                )
                # Save snapshot (updates today's entry if exists)
                self.trend_tracker.save_snapshot(metrics)
                logger.info(
                    f"✓ Saved trend snapshot: {metrics['total_files']} files, "
                    f"{metrics['total_chunks']} chunks, health score {metrics['health_score']}"
                )
            except Exception as e:
                logger.warning(f"Failed to save trend snapshot: {e}")

        # Start background KG build if enabled and indexing completed successfully
        if (
            self._enable_background_kg
            and indexed_count > 0
            and not self._kg_build_task
            and phase in ("all", "embed")  # Only after embeddings are ready
        ):
            self._kg_build_task = asyncio.create_task(self._build_kg_background())
            logger.info(
                "🔗 Phase 3: Knowledge graph building in background (search available now)"
            )

        # Log metrics summary
        if indexed_count > 0:
            if metrics_json:
                # Output JSON metrics to stdout
                print(metrics_tracker.metrics.to_json())
            else:
                # Display formatted summary table
                metrics_tracker.log_summary()

        # Finalize atomic rebuild if active
        if atomic_rebuild_active and indexed_count > 0:
            await self._finalize_atomic_rebuild()

        # Phase 3: Build BM25 index for hybrid search
        await self._build_bm25_index()

        return indexed_count

    async def _parse_and_prepare_file(
        self, file_path: Path, force_reindex: bool = False, skip_delete: bool = False
    ) -> tuple[list[CodeChunk], dict[str, Any] | None]:
        """Parse file and prepare chunks with metrics (no database insertion).

        This method extracts the parsing and metric collection logic from index_file()
        to enable batch processing across multiple files.

        Args:
            file_path: Path to the file to parse
            force_reindex: Whether to force reindexing (always deletes existing chunks)
            skip_delete: If True, skip delete operation (used when building fresh database)

        Returns:
            Tuple of (chunks_with_hierarchy, chunk_metrics)

        Raises:
            ParsingError: If file parsing fails
        """
        # Check if file should be indexed
        if not self.file_discovery.should_index_file(file_path):
            return ([], None)

        # Skip delete on force reindex with fresh database OR when atomic rebuild is active
        # This optimization eliminates ~25k unnecessary database queries for large codebases
        # WORKAROUND: Skip delete on macOS to avoid SIGBUS crash
        if (
            not skip_delete
            and not self._atomic_rebuild_active
            and platform.system() != "Darwin"
        ):
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
        _ = self.metrics_collector.collect_metrics_for_chunks(
            chunks_with_hierarchy, file_path
        )

        return (chunks_with_hierarchy, None)

    async def _process_file_batch(
        self,
        file_paths: list[Path],
        force_reindex: bool = False,
        skip_delete: bool = False,
        metadata_dict: dict[str, float] | None = None,
    ) -> list[bool]:
        """Process a batch of files and accumulate chunks for batch embedding.

        This method processes multiple files in parallel (using multiprocessing for
        CPU-bound parsing) and then performs a single database insertion for all chunks,
        enabling efficient batch embedding generation.

        Args:
            file_paths: List of file paths to process
            force_reindex: Whether to force reindexing
            skip_delete: If True, skip delete operations (used when building fresh database)
            metadata_dict: Pre-loaded metadata dict (OPTIMIZATION: pass to avoid O(n) loads per batch)

        Returns:
            List of success flags for each file
        """
        # Initialize backends if not already initialized
        if self.chunks_backend._db is None:
            await self.chunks_backend.initialize()

        all_chunks: list[CodeChunk] = []
        file_to_chunks_map: dict[str, tuple[int, int]] = {}
        success_flags: list[bool] = []

        # Filter files that should be indexed and delete old chunks (unless building fresh)
        # Skip delete on force reindex with fresh database OR when atomic rebuild is active
        files_to_parse = []
        delete_tasks = []

        for file_path in file_paths:
            if not self.file_discovery.should_index_file(file_path):
                success_flags.append(True)  # Skipped file is not an error
                continue

            # Only schedule deletion if not building fresh database AND not in atomic rebuild
            # WORKAROUND: Skip delete on macOS to avoid SIGBUS crash (same reason as chunks_backend)
            if (
                not skip_delete
                and not self._atomic_rebuild_active
                and platform.system() != "Darwin"
            ):
                delete_task = asyncio.create_task(
                    self.database.delete_by_file(file_path)
                )
                delete_tasks.append(delete_task)
            files_to_parse.append(file_path)

        # Wait for all deletions to complete (if any were scheduled)
        if delete_tasks:
            await asyncio.gather(*delete_tasks, return_exceptions=True)

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

        # OPTIMIZATION: Load metadata once per batch instead of per file
        # This avoids O(n) json.load() calls that kill performance on large codebases (25k+ files)
        if metadata_dict is None:
            metadata_dict = self.metadata.load()

        # Accumulate chunks from all successfully parsed files
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
                _ = self.metrics_collector.collect_metrics_for_chunks(
                    chunks_with_hierarchy, file_path
                )

                # Accumulate chunks
                start_idx = len(all_chunks)
                all_chunks.extend(chunks_with_hierarchy)
                end_idx = len(all_chunks)
                file_to_chunks_map[str(file_path)] = (start_idx, end_idx)

                # Update metadata for successfully parsed file
                try:
                    metadata_dict[str(file_path)] = os.path.getmtime(file_path)
                except FileNotFoundError:
                    logger.warning(
                        f"Skipping metadata update for deleted file: {file_path}"
                    )
                success_flags.append(True)
            else:
                # Empty file is not an error
                try:
                    metadata_dict[str(file_path)] = os.path.getmtime(file_path)
                except FileNotFoundError:
                    logger.warning(
                        f"Skipping metadata update for deleted file: {file_path}"
                    )
                success_flags.append(True)

        # Single database insertion for entire batch using two-phase architecture
        if all_chunks:
            batch_start = time.perf_counter()
            logger.info(
                f"Batch inserting {len(all_chunks)} chunks from {len(file_paths)} files"
            )
            try:
                # Phase 1: Store chunks to chunks.lance (fast, durable)
                # Group chunks by file for proper file_hash tracking
                for file_path_str, (start_idx, end_idx) in file_to_chunks_map.items():
                    file_path = Path(file_path_str)
                    file_chunks = all_chunks[start_idx:end_idx]

                    # Compute file hash for change detection
                    file_hash = compute_file_hash(file_path)
                    rel_path = str(file_path.relative_to(self.project_root))

                    # Delete old chunks for this file (skip if atomic rebuild is active)
                    if not self._atomic_rebuild_active:
                        await self.chunks_backend.delete_file_chunks(rel_path)

                    # Convert CodeChunk objects to dicts for storage
                    chunk_dicts = []
                    for chunk in file_chunks:
                        chunk_dict = {
                            "chunk_id": chunk.chunk_id,
                            "file_path": rel_path,
                            "content": chunk.content,
                            "language": chunk.language,
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                            "start_char": 0,
                            "end_char": 0,
                            "chunk_type": chunk.chunk_type,
                            "name": chunk.function_name or chunk.class_name or "",
                            "parent_name": "",
                            "hierarchy_path": self._build_hierarchy_path(chunk),
                            "docstring": chunk.docstring or "",
                            "signature": "",
                            "complexity": int(chunk.complexity_score),
                            "token_count": len(chunk.content.split()),
                            # Code relationships (for KG)
                            "calls": chunk.calls or [],
                            "imports": [
                                json.dumps(imp) if isinstance(imp, dict) else imp
                                for imp in (chunk.imports or [])
                            ],
                            "inherits_from": chunk.inherits_from or [],
                        }
                        chunk_dicts.append(chunk_dict)

                    # Store to chunks.lance
                    if chunk_dicts:
                        await self.chunks_backend.add_chunks(chunk_dicts, file_hash)

                # Phase 2: Generate embeddings and store to vectors.lance
                if all_chunks:
                    # Extract content for embedding generation
                    contents = [chunk.content for chunk in all_chunks]

                    # Generate embeddings using database's embedding function
                    # Use __call__() which is the universal interface for all embedding functions
                    embeddings = self.database.embedding_function(contents)

                    # Prepare chunks with vectors for vectors_backend
                    chunks_with_vectors = []
                    for chunk, embedding in zip(all_chunks, embeddings, strict=True):
                        chunk_dict = {
                            "chunk_id": chunk.chunk_id or chunk.id,
                            "vector": embedding,
                            "file_path": str(chunk.file_path),
                            "content": chunk.content,
                            "language": chunk.language,
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                            "chunk_type": chunk.chunk_type,
                            "name": chunk.class_name or chunk.function_name or "",
                            "hierarchy_path": getattr(chunk, "hierarchy_path", ""),
                        }
                        chunks_with_vectors.append(chunk_dict)

                    # Store vectors to vectors.lance
                    model_name = self.get_embedding_model_name()
                    await self.vectors_backend.add_vectors(
                        chunks_with_vectors, model_version=model_name
                    )

                batch_elapsed = time.perf_counter() - batch_start
                logger.info(
                    f"Successfully indexed {len(all_chunks)} chunks from {sum(success_flags)} files "
                    f"in {batch_elapsed:.2f}s"
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
        skip_delete: bool = False,
    ) -> bool:
        """Index a single file.

        Args:
            file_path: Path to the file to index
            force_reindex: Whether to reindex if already indexed
            skip_delete: If True, skip delete operation (used when building fresh database)

        Returns:
            True if file was successfully indexed
        """
        # Initialize backends if not already initialized
        if self.chunks_backend._db is None:
            await self.chunks_backend.initialize()

        try:
            # Check if file should be indexed
            if not self.file_discovery.should_index_file(file_path):
                return False

            # Skip delete on force reindex with fresh database OR when atomic rebuild is active
            # WORKAROUND: Skip delete on macOS to avoid SIGBUS crash
            if (
                not skip_delete
                and not self._atomic_rebuild_active
                and platform.system() != "Darwin"
            ):
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
            _ = self.metrics_collector.collect_metrics_for_chunks(
                chunks_with_hierarchy, file_path
            )

            # Phase 1: Store chunks to chunks.lance (fast, durable)
            file_hash = compute_file_hash(file_path)
            rel_path = str(file_path.relative_to(self.project_root))

            # Delete old chunks for this file (skip if atomic rebuild is active)
            if not self._atomic_rebuild_active:
                await self.chunks_backend.delete_file_chunks(rel_path)

            # Convert CodeChunk objects to dicts for storage
            chunk_dicts = []
            for chunk in chunks_with_hierarchy:
                chunk_dict = {
                    "chunk_id": chunk.chunk_id,
                    "file_path": rel_path,
                    "content": chunk.content,
                    "language": chunk.language,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "start_char": 0,
                    "end_char": 0,
                    "chunk_type": chunk.chunk_type,
                    "name": chunk.function_name or chunk.class_name or "",
                    "parent_name": "",
                    "hierarchy_path": self._build_hierarchy_path(chunk),
                    "docstring": chunk.docstring or "",
                    "signature": "",
                    "complexity": int(chunk.complexity_score),
                    "token_count": len(chunk.content.split()),
                    # Code relationships (for KG)
                    "calls": chunk.calls or [],
                    "imports": [
                        json.dumps(imp) if isinstance(imp, dict) else imp
                        for imp in (chunk.imports or [])
                    ],
                    "inherits_from": chunk.inherits_from or [],
                }
                chunk_dicts.append(chunk_dict)

            # Store to chunks.lance
            if chunk_dicts:
                await self.chunks_backend.add_chunks(chunk_dicts, file_hash)

            # Phase 2: Generate embeddings and store to vectors.lance
            if chunks_with_hierarchy:
                # Extract content for embedding generation
                contents = [chunk.content for chunk in chunks_with_hierarchy]

                # Generate embeddings using database's embedding function
                # Use __call__() which is the universal interface for all embedding functions
                embeddings = self.database.embedding_function(contents)

                # Prepare chunks with vectors for vectors_backend
                chunks_with_vectors = []
                for chunk, embedding in zip(
                    chunks_with_hierarchy, embeddings, strict=True
                ):
                    chunk_dict = {
                        "chunk_id": chunk.chunk_id or chunk.id,
                        "vector": embedding,
                        "file_path": str(chunk.file_path),
                        "content": chunk.content,
                        "language": chunk.language,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "chunk_type": chunk.chunk_type,
                        "name": chunk.class_name or chunk.function_name or "",
                        "hierarchy_path": self._build_hierarchy_path(chunk),
                    }
                    chunks_with_vectors.append(chunk_dict)

                # Store vectors to vectors.lance
                model_name = self.get_embedding_model_name()
                await self.vectors_backend.add_vectors(
                    chunks_with_vectors, model_version=model_name
                )

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

        WORKAROUND: Skipped on macOS to avoid SIGBUS crash caused by memory conflict
        between PyTorch MPS memory-mapped model files and LanceDB delete operations.
        On macOS, stale chunks will remain until the next full reindex.

        Args:
            file_path: Path to the file to remove

        Returns:
            Number of chunks removed (0 on macOS as operation is skipped)
        """
        if platform.system() == "Darwin":
            logger.debug(
                f"Skipping remove_file on macOS for {file_path} to avoid SIGBUS crash "
                "(PyTorch MPS + LanceDB delete memory conflict; "
                "stale chunks will be cleaned up on next full reindex)"
            )
            return 0

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

    async def get_two_phase_status(self) -> dict[str, Any]:
        """Get indexing status for both phases.

        Returns:
            Dictionary with status for Phase 1 (chunks) and Phase 2 (vectors):
            {
                "phase1": {
                    "total_chunks": 1000,
                    "files_indexed": 50,
                    "languages": {"python": 30, "javascript": 20}
                },
                "phase2": {
                    "pending": 500,
                    "processing": 0,
                    "complete": 500,
                    "error": 0
                },
                "vectors": {
                    "total": 500,
                    "files": 25
                },
                "ready_for_search": True
            }
        """
        try:
            # Initialize backends if needed
            if self.chunks_backend._db is None:
                await self.chunks_backend.initialize()
            if self.vectors_backend._db is None:
                await self.vectors_backend.initialize()

            # Get stats from both backends
            chunks_stats = await self.chunks_backend.get_stats()
            vectors_stats = await self.vectors_backend.get_stats()

            return {
                "phase1": {
                    "total_chunks": chunks_stats.get("total", 0),
                    "files_indexed": chunks_stats.get("files", 0),
                    "languages": chunks_stats.get("languages", {}),
                },
                "phase2": {
                    "pending": chunks_stats.get("pending", 0),
                    "processing": chunks_stats.get("processing", 0),
                    "complete": chunks_stats.get("complete", 0),
                    "error": chunks_stats.get("error", 0),
                },
                "vectors": {
                    "total": vectors_stats.get("total", 0),
                    "files": vectors_stats.get("files", 0),
                    "chunk_types": vectors_stats.get("chunk_types", {}),
                },
                "ready_for_search": chunks_stats.get("complete", 0) > 0,
            }
        except Exception as e:
            logger.error(f"Failed to get two-phase status: {e}")
            return {
                "phase1": {"total_chunks": 0, "files_indexed": 0, "languages": {}},
                "phase2": {"pending": 0, "processing": 0, "complete": 0, "error": 0},
                "vectors": {"total": 0, "files": 0, "chunk_types": {}},
                "ready_for_search": False,
                "error": str(e),
            }

    async def get_indexing_stats(self, db_stats: IndexStats | None = None) -> dict:
        """Get statistics about the indexing process.

        Args:
            db_stats: Optional pre-fetched database stats to avoid duplicate queries (deprecated)

        Returns:
            Dictionary with indexing statistics

        Note:
            In two-phase architecture, uses chunks_backend.get_stats() to retrieve
            actual chunk counts from LanceDB. The old ChromaDB (database) is deprecated
            and may be empty during migration.
        """
        try:
            # Initialize chunks_backend if needed
            if self.chunks_backend._db is None:
                await self.chunks_backend.initialize()

            # Get stats from chunks_backend (Phase 1 storage)
            chunks_stats = await self.chunks_backend.get_stats()

            # Use chunks_backend stats for all file counts
            # This queries the actual LanceDB storage where chunks are stored
            return {
                "total_indexable_files": chunks_stats["files"],
                "indexed_files": chunks_stats["files"],
                "total_files": chunks_stats["files"],  # For backward compatibility
                "total_chunks": chunks_stats["total"],
                "languages": chunks_stats["languages"],
                "file_types": {},  # Not tracked in chunks_backend yet
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

    async def get_indexed_count(self) -> int:
        """Get count of files currently indexed in the database.

        Returns:
            Number of unique files indexed
        """
        try:
            # Initialize chunks_backend if needed
            if self.chunks_backend._db is None:
                await self.chunks_backend.initialize()

            # Get stats from chunks_backend (Phase 1 storage)
            chunks_stats = await self.chunks_backend.get_stats()
            return chunks_stats.get("files", 0)
        except Exception as e:
            logger.warning(f"Failed to get indexed count: {e}")
            return 0

    async def get_files_to_index(
        self,
        force_reindex: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
        file_limit: int | None = None,
    ) -> tuple[list[Path], list[Path]]:
        """Get all indexable files and those that need indexing.

        Args:
            force_reindex: Whether to force reindex of all files
            progress_callback: Optional callback(dirs_scanned, files_found) for discovery progress
            file_limit: Optional limit - stop scanning once this many files found

        Returns:
            Tuple of (all_indexable_files, files_to_index)
        """
        # Find all indexable files (with progress callback and optional limit)
        all_files = await self.file_discovery.find_indexable_files_async(
            progress_callback=progress_callback,
            file_limit=file_limit,
        )

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

    async def _build_bm25_index(self) -> None:
        """Build BM25 index from all chunks for keyword search.

        This is Phase 3 of indexing (after chunks are built).
        BM25 index enables hybrid search by combining keyword and semantic search.
        BM25 only needs text content, not embeddings, so reads from chunks.lance.
        """
        try:
            # Get all chunks from chunks.lance table
            # ChunksBackend has all chunk data including content (vectors not needed for BM25)
            if self.chunks_backend._db is None or self.chunks_backend._table is None:
                logger.warning(
                    "Chunks backend not initialized, skipping BM25 index build"
                )
                return

            logger.info("📚 Phase 3: Building BM25 index for keyword search...")

            # Query all records from chunks.lance table (to_pandas is most efficient for bulk reads)
            import asyncio

            df = await asyncio.to_thread(self.chunks_backend._table.to_pandas)

            if df.empty:
                logger.info("No chunks in chunks table, skipping BM25 index build")
                return

            # Convert DataFrame rows to dicts for BM25Backend
            # BM25Backend expects: chunk_id, content, name, file_path, chunk_type
            chunks_for_bm25 = []
            for _, row in df.iterrows():
                chunk_dict = {
                    "chunk_id": row["chunk_id"],
                    "content": row["content"],
                    "name": row.get("name", ""),  # chunks.lance has "name" field
                    "file_path": row["file_path"],
                    "chunk_type": row.get("chunk_type", "code"),
                }
                chunks_for_bm25.append(chunk_dict)

            logger.info(
                f"📚 Phase 3: Building BM25 index from {len(chunks_for_bm25)} chunks..."
            )

            # Build BM25 index
            bm25_backend = BM25Backend()
            bm25_backend.build_index(chunks_for_bm25)

            # Save to disk
            bm25_path = self.config.index_path / "bm25_index.pkl"
            bm25_backend.save(bm25_path)

            stats = bm25_backend.get_stats()
            logger.info(
                f"✓ Phase 3 complete: BM25 index built with {stats['chunk_count']} chunks "
                f"(avg doc length: {stats['avg_doc_length']:.1f} tokens)"
            )

        except Exception as e:
            # Non-fatal: BM25 failure shouldn't break indexing
            logger.warning(f"BM25 index building failed (non-fatal): {e}")
            logger.warning("Hybrid search will fall back to vector-only mode")
