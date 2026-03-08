"""Semantic indexer for MCP Vector Search with two-phase architecture.

Phase 1: Parse and chunk files, store to chunks.lance (fast, durable)
Phase 2: Embed chunks, store to vectors.lance (resumable, incremental)
"""

import asyncio
import gc
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from ..analysis.collectors.base import MetricCollector
from ..analysis.trends import TrendTracker
from ..config.settings import ProjectConfig
from ..parsers.registry import get_parser_registry
from ..utils.monorepo import MonorepoDetector
from .atomic_rebuild import AtomicRebuildManager
from .bm25_builder import build_bm25_index as _build_bm25_index_fn
from .chunk_dict import build_hierarchy_path as _build_hierarchy_path_fn
from .chunk_dict import chunk_to_storage_dict
from .chunk_processor import ChunkProcessor
from .chunks_backend import ChunksBackend, compute_file_hash
from .context_builder import build_contextual_text
from .database import VectorDatabase
from .directory_index import DirectoryIndex
from .embedding_runner import run_phase2_embedding
from .exceptions import (
    DatabaseError,
    DatabaseInitializationError,
    IndexingError,
    ParsingError,
)
from .file_discovery import FileDiscovery
from .file_move_detector import detect_file_moves as _detect_file_moves_fn
from .index_cleanup import (  # re-exported for backward compatibility
    _detect_filesystem_type,
    cleanup_stale_locks,
    cleanup_stale_progress,
    cleanup_stale_transactions,
)
from .index_cleanup import (
    run_auto_migrations as _run_auto_migrations_fn,
)
from .index_metadata import IndexMetadata
from .memory_monitor import MemoryMonitor
from .metrics import get_metrics_tracker
from .metrics_collector import IndexerMetricsCollector
from .models import (
    CodeChunk,
    HealthStatus,  # noqa: F401  # re-exported; avoids LanceDB in package __init__
    IndexResult,
    IndexStats,
    ProjectStatus,
)
from .relationships import RelationshipStore
from .resource_manager import calculate_optimal_workers
from .vectors_backend import VectorsBackend


@dataclass
class SemanticIndexerConfig:
    """Resolved configuration for SemanticIndexer.

    All environment variable reads happen in from_env() at the process
    boundary, not scattered through business logic.
    """

    index_path: Path | None = None
    file_batch_size: int = 512
    embed_batch_size: int = 0  # 0 = auto-detect via _detect_optimal_batch_size()
    max_workers: int | None = None  # None = auto via calculate_optimal_workers()
    num_producers: int = 4
    queue_depth: int | None = None  # None = num_producers * 4
    write_batch_size: int | None = None  # None = filesystem-auto
    max_memory_gb: float | None = None  # None = cgroup/system auto
    enable_background_kg: bool = False
    auto_optimize: bool = True
    two_pass: bool = (
        False  # If True, use sequential chunk-all-then-embed-all instead of pipeline
    )

    @classmethod
    def from_env(cls) -> "SemanticIndexerConfig":
        """Build a resolved config by reading all MCP_VECTOR_SEARCH_* env vars once."""
        import os as _os

        def _int_env(key: str, default: int) -> int:
            val = _os.environ.get(key)
            if val:
                try:
                    return int(val)
                except ValueError:
                    pass
            return default

        def _float_env(key: str) -> float | None:
            val = _os.environ.get(key)
            if val:
                try:
                    return float(val)
                except ValueError:
                    pass
            return None

        # FILE_BATCH_SIZE: MCP_VECTOR_SEARCH_FILE_BATCH_SIZE (also accept legacy MCP_VECTOR_SEARCH_BATCH_SIZE)
        file_batch_size = 512
        for key in (
            "MCP_VECTOR_SEARCH_FILE_BATCH_SIZE",
            "MCP_VECTOR_SEARCH_BATCH_SIZE",
        ):
            val = _os.environ.get(key)
            if val:
                try:
                    file_batch_size = int(val)
                    break
                except ValueError:
                    pass

        # embed_batch_size
        embed_bs = 0
        val = _os.environ.get("MCP_VECTOR_SEARCH_EMBED_BATCH_SIZE")
        if val:
            try:
                embed_bs = int(val)
            except ValueError:
                pass

        # index_path
        raw_index_path = _os.environ.get("INDEX_PATH")
        index_path = Path(raw_index_path) if raw_index_path else None

        # num_producers
        num_producers = _int_env("MCP_VECTOR_SEARCH_NUM_PRODUCERS", 4)

        # queue_depth — None means auto (num_producers * 4)
        queue_depth_val = _os.environ.get("MCP_VECTOR_SEARCH_QUEUE_DEPTH")
        queue_depth: int | None = None
        if queue_depth_val:
            try:
                queue_depth = int(queue_depth_val)
            except ValueError:
                pass

        # write_batch_size — None means filesystem-auto
        write_bs_val = _os.environ.get("MCP_VECTOR_SEARCH_WRITE_BATCH_SIZE")
        write_batch_size: int | None = None
        if write_bs_val:
            try:
                write_batch_size = int(write_bs_val)
            except ValueError:
                pass

        # max_memory_gb
        max_memory_gb = _float_env("MCP_VECTOR_SEARCH_MAX_MEMORY_GB")

        # max_workers
        max_workers_val = _os.environ.get("MCP_VECTOR_SEARCH_WORKERS")
        max_workers: int | None = None
        if max_workers_val:
            try:
                max_workers = int(max_workers_val)
            except ValueError:
                pass

        # enable_background_kg
        auto_kg_val = _os.environ.get("MCP_VECTOR_SEARCH_AUTO_KG", "false").lower()
        enable_background_kg = auto_kg_val in ("1", "true", "yes")

        # two_pass: sequential chunk-all then embed-all (avoids GPU starvation on slow EFS)
        two_pass = _os.environ.get("MCP_VECTOR_SEARCH_TWO_PASS", "").lower() in (
            "1",
            "true",
            "yes",
        )

        return cls(
            index_path=index_path,
            file_batch_size=file_batch_size,
            embed_batch_size=embed_bs,
            max_workers=max_workers,
            num_producers=num_producers,
            queue_depth=queue_depth,
            write_batch_size=write_batch_size,
            max_memory_gb=max_memory_gb,
            enable_background_kg=enable_background_kg,
            two_pass=two_pass,
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
        embed_batch_size: int = 0,
        debug: bool = False,
        collectors: list[MetricCollector] | None = None,
        use_multiprocessing: bool = True,
        auto_optimize: bool = True,
        ignore_patterns: set[str] | None = None,
        skip_blame: bool = True,
        progress_tracker: Any = None,
        index_path: str | None = None,
        indexer_config: "SemanticIndexerConfig | None" = None,
        chunks_backend: "ChunksBackend | None" = None,
        vectors_backend: "VectorsBackend | None" = None,
        memory_monitor: "MemoryMonitor | None" = None,
    ) -> None:
        """Initialize semantic indexer.

        Args:
            database: Vector database instance
            project_root: Project root directory (where source code lives)
            file_extensions: File extensions to index (deprecated, use config)
            config: Project configuration (preferred over file_extensions)
            max_workers: Maximum number of worker processes for parallel parsing (ignored if use_multiprocessing=False)
            batch_size: Number of files to process in each batch (default: 512, override with MCP_VECTOR_SEARCH_FILE_BATCH_SIZE)
            embed_batch_size: GPU embedding batch size (0=auto-detect from GPU/CPU). Override with
                MCP_VECTOR_SEARCH_EMBED_BATCH_SIZE. Larger values improve GPU throughput (512-4096 for CUDA).
            debug: Enable debug output for hierarchy building
            collectors: Metric collectors to run during indexing (defaults to all complexity collectors)
            use_multiprocessing: Enable multiprocess parallel parsing (default: True, disable for debugging)
            auto_optimize: Enable automatic optimization based on codebase profile (default: True)
            ignore_patterns: Additional patterns to ignore (merged with defaults, e.g., vendor patterns)
            skip_blame: Skip git blame tracking for faster indexing (default: False)
            progress_tracker: Optional ProgressTracker instance for displaying progress bars
            index_path: Optional separate directory for .mcp-vector-search/ index data.
                Falls back to INDEX_PATH env var, then project_root.
            indexer_config: Optional resolved config (from SemanticIndexerConfig.from_env()).
                When provided, env var reads in __init__ are skipped in favour of config values.
            chunks_backend: Injectable ChunksBackend (for testing / custom storage).
            vectors_backend: Injectable VectorsBackend (for testing / custom storage).
            memory_monitor: Injectable MemoryMonitor (for testing / custom limits).

        Environment Variables:
            MCP_VECTOR_SEARCH_FILE_BATCH_SIZE: Override file batch size (default: 512)
            MCP_VECTOR_SEARCH_EMBED_BATCH_SIZE: Override GPU embedding batch size (default: auto-detect)
            MCP_VECTOR_SEARCH_BATCH_SIZE: Legacy alias for MCP_VECTOR_SEARCH_FILE_BATCH_SIZE
            MCP_VECTOR_SEARCH_SKIP_BLAME: Skip git blame tracking (true/1/yes)
            INDEX_PATH: Separate directory for .mcp-vector-search/ index data
        """
        # Resolve config: use provided config or build from env vars (backward compat)
        _cfg = (
            indexer_config
            if indexer_config is not None
            else SemanticIndexerConfig.from_env()
        )

        self.database = database
        self.project_root = project_root

        # Resolve index_path: explicit parameter > config > INDEX_PATH env var > project_root
        if index_path:
            self.index_path = Path(index_path)
        elif _cfg.index_path is not None:
            self.index_path = _cfg.index_path
        else:
            self.index_path = self.project_root

        # Convenience: the .mcp-vector-search directory under index_path
        self._mcp_dir = self.index_path / ".mcp-vector-search"

        if self.index_path != self.project_root:
            logger.info(
                f"Using separate index path: {self.index_path} "
                f"(project root: {self.project_root})"
            )

        self.config = config
        self.auto_optimize = auto_optimize
        self._applied_optimizations: dict[str, Any] | None = None
        self.cancellation_flag: threading.Event | None = (
            None  # Set externally for cancellation support
        )
        self.progress_tracker = progress_tracker  # Optional progress bar display

        # Set batch size: explicit kwarg > config (from env) > default 512
        # PERFORMANCE: Larger batches reduce per-batch overhead.
        # 32K files @ 256/batch = 125 batches, @ 512/batch = 62 batches (50% reduction).
        self.batch_size = batch_size if batch_size is not None else _cfg.file_batch_size

        # Set embed_batch_size: explicit param (non-zero) > config (from env) > GPU auto-detect
        # _cfg.embed_batch_size == 0 means "not set via env var", so auto-detect applies.
        _resolved_embed = (
            embed_batch_size if embed_batch_size != 0 else _cfg.embed_batch_size
        )
        if _resolved_embed != 0:
            self.embed_batch_size = _resolved_embed
            logger.info(f"Using embed batch size: {self.embed_batch_size}")
        else:
            # Auto-detect from GPU/CPU capabilities
            from .embeddings import _detect_optimal_batch_size

            self.embed_batch_size = _detect_optimal_batch_size()
            logger.info(f"Auto-detected embed batch size: {self.embed_batch_size}")

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

        self.metadata = IndexMetadata(self.index_path)

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

        # Auto-configure workers: explicit kwarg > config (from env) > auto-detect
        if max_workers is None:
            max_workers = _cfg.max_workers

        if max_workers is None:
            # Auto-detect based on CPU cores and memory
            limits = calculate_optimal_workers(
                memory_per_worker_mb=800,  # Embedding models use more memory
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
        self.directory_index = DirectoryIndex(self._mcp_dir / "directory_index.json")
        # Load existing directory index
        self.directory_index.load()

        # Initialize relationship store for pre-computing visualization relationships
        self.relationship_store = RelationshipStore(self.index_path)

        # Initialize trend tracker for historical metrics
        self.trend_tracker = TrendTracker(self.index_path)

        # Initialize two-phase backends (injectable for testing / custom storage)
        # Both use same db_path directory for LanceDB
        lance_path = self._mcp_dir / "lance"
        self.chunks_backend = chunks_backend or ChunksBackend(lance_path)
        self.vectors_backend = vectors_backend or VectorsBackend(lance_path)

        # Background KG build tracking
        self._kg_build_task: asyncio.Task | None = None
        self._kg_build_status: str = "not_started"
        self._enable_background_kg: bool = _cfg.enable_background_kg

        # Track atomic rebuild state (fresh database doesn't need deletes)
        self._atomic_rebuild_active: bool = False

        # Store resolved config for use in runtime methods
        self._indexer_config = _cfg

        # Initialize memory monitor (injectable for testing / custom limits)
        self.memory_monitor = memory_monitor or MemoryMonitor(
            max_memory_gb=_cfg.max_memory_gb
        )

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

            kg_path = self._mcp_dir / "knowledge_graph"
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

            # Apply optimizations (only if batch size was not explicitly set via env/kwarg)
            # _indexer_config.file_batch_size == 512 means "default / not overridden"
            if self._indexer_config.file_batch_size == 512 and self.batch_size == 512:
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

        Delegates to :func:`core.index_cleanup.run_auto_migrations`.
        """
        await _run_auto_migrations_fn(self.project_root)

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

            # Detect file moves/renames — update metadata instead of re-chunking.
            # Also returns a file_hash_cache (Path → hash) so we skip double-hashing
            # during the change detection loop below.
            detected_moves, _moved_old_paths, file_hash_cache = self._detect_file_moves(
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

            # Change detection: compare on-disk hashes to indexed hashes.
            # Reuse file_hash_cache from _detect_file_moves to avoid re-hashing.
            filtered_files = []
            for f in all_files:
                try:
                    file_hash = file_hash_cache.get(f) or compute_file_hash(f)
                    rel_path = str(f.relative_to(self.project_root))

                    # OPTIMIZATION: O(1) dict lookup instead of per-file database query
                    stored_hash = indexed_file_hashes.get(rel_path)
                    if stored_hash is None or stored_hash != file_hash:
                        filtered_files.append(f)
                except Exception as e:
                    logger.warning(f"Error checking file {f}: {e}, will re-index")
                    filtered_files.append(f)

            files_to_index = filtered_files
            logger.info(
                f"Incremental index: {len(files_to_index)} of {len(all_files)} files need updating"
            )
        else:
            logger.info(f"Force reindex: processing all {len(files_to_index)} files")

        if not files_to_index:
            logger.info("All files are up to date")
            # Still build BM25 index if it doesn't exist.
            # Use self._mcp_dir (respects index_path) instead of config.index_path
            # so that a separate index_path is honoured here too.
            if self.config:
                bm25_path = self._mcp_dir / "bm25_index.pkl"
                if not bm25_path.exists():
                    logger.info(f"BM25 index not found at {bm25_path}, building now...")
                    await self._build_bm25_index()
                else:
                    logger.info(f"BM25 index already exists at {bm25_path}")
            return 0, 0, 0

        # PIPELINE IMPLEMENTATION: Producer-consumer pattern
        files_indexed = 0
        chunks_created = 0
        chunks_embedded = 0

        # Get metrics tracker
        metrics_tracker = get_metrics_tracker()

        # File batch size for parsing: use self.batch_size (configurable, not hardcoded)
        file_batch_size = self.batch_size

        # Number of concurrent producer tasks (each handles a slice of files_to_process)
        num_producers = self._indexer_config.num_producers

        # --- Prepare files_to_process and batch-delete old chunks BEFORE launching producers ---
        # This is done once in the main scope so all producers share the same list.
        # files_to_index was already filtered by the outer _index_with_pipeline scope
        # (get_all_indexed_file_hashes + _detect_file_moves + change detection loop).
        logger.info(
            f"📄 Phase 1: Chunking {len(files_to_index)} files (parsing and extracting code structure)..."
        )
        phase_start_time = time.time()

        files_to_delete = []
        files_to_process = []
        for file_path in files_to_index:
            try:
                if force_reindex:
                    file_hash = ""
                else:
                    # Reuse hash already computed by the outer scope
                    file_hash = file_hash_cache.get(file_path) or compute_file_hash(
                        file_path
                    )

                rel_path = str(file_path.relative_to(self.project_root))
                files_to_delete.append(rel_path)
                files_to_process.append((file_path, rel_path, file_hash))

            except Exception as e:
                logger.error(f"Failed to check file {file_path}: {e}")
                continue

        # Batch delete old chunks for changed files before re-chunking.
        # LanceDB delete() is metadata-only and does NOT trigger compact_files().
        # The macOS SIGBUS workaround is in _compact_table(), not here.
        if not self._atomic_rebuild_active and files_to_delete:
            deleted_count = await self.chunks_backend.delete_files_batch(
                files_to_delete
            )
            if deleted_count > 0:
                logger.info(
                    f"Batch deleted {deleted_count} old chunks for {len(files_to_delete)} files"
                )

        # Clamp num_producers to the actual number of files (avoid empty producers)
        effective_num_producers = max(1, min(num_producers, len(files_to_process)))
        if effective_num_producers < num_producers:
            logger.debug(
                f"Clamping producers from {num_producers} to {effective_num_producers} "
                f"(only {len(files_to_process)} files to process)"
            )

        # FIX 3 (HIGH): Auto-scale queue depth to num_producers * 4.
        # With depth=4 and 4 producers each producer gets ~1 slot, eliminating
        # pipelining benefit.  N*4 gives each producer 4 batches of lookahead.
        queue_maxsize = self._indexer_config.queue_depth or (
            effective_num_producers * 4
        )
        chunk_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_maxsize)

        # FIX 1 (CRITICAL RELIABILITY): Cancellation event so producers can detect
        # that the consumer has crashed and stop blocking on chunk_queue.put().
        pipeline_cancel = asyncio.Event()

        # FIX 6 (HIGH): LPT scheduling — sort files by descending size so large
        # files distribute evenly across producers (Longest Processing Time heuristic).
        try:
            files_to_process.sort(
                key=lambda x: x[0].stat().st_size if x[0].exists() else 0,
                reverse=True,
            )
        except OSError:
            pass  # Non-fatal: proceed with original order

        async def chunk_producer(producer_id: int, file_slice: list) -> None:
            """Producer: Parse and chunk a slice of files, put batches on queue.

            Args:
                producer_id: Index of this producer (0-based, for logging)
                file_slice: Subset of files_to_process this producer owns
            """
            nonlocal files_indexed, chunks_created

            with metrics_tracker.phase("parsing") as parsing_metrics:
                # Process files in batches and put on queue
                for batch_start in range(0, len(file_slice), file_batch_size):
                    # Check if pipeline was cancelled (consumer crashed)
                    if pipeline_cancel.is_set():
                        logger.warning(
                            "Producer %d: pipeline cancelled, stopping", producer_id
                        )
                        return

                    # Check if consumer is still alive
                    if consumer_task.done():
                        exc = consumer_task.exception()
                        logger.error(
                            f"Consumer died unexpectedly: {exc}. Stopping producer {producer_id}."
                        )
                        return

                    batch_end = min(batch_start + file_batch_size, len(file_slice))
                    batch = file_slice[batch_start:batch_end]

                    batch_chunks = []
                    batch_files_processed = 0

                    # FIX 2 (CRITICAL PERFORMANCE): Use process pool for batch parsing.
                    # parse_files_multiprocess() uses ProcessPoolExecutor giving 6-12x
                    # speedup on multi-core machines vs. serial per-file await.
                    batch_paths = [fp for fp, _rel, _hash in batch]
                    rel_paths = {fp: rel for fp, rel, _hash in batch}
                    file_hashes = {fp: fh for fp, _rel, fh in batch}

                    if self.use_multiprocessing and len(batch_paths) > 1:
                        parse_results = (
                            await self.chunk_processor.parse_files_multiprocess(
                                batch_paths
                            )
                        )
                    else:
                        # Fallback: parse sequentially for single-file batches or
                        # when multiprocessing is disabled
                        parse_results = []
                        for fp in batch_paths:
                            try:
                                chunks = await self.chunk_processor.parse_file(fp)
                                parse_results.append((fp, chunks, None))
                            except Exception as exc:
                                parse_results.append((fp, [], exc))

                    # FIX 8 (LOW): Throttle memory checks to every 10 files to
                    # avoid 200-500ms of psutil syscall overhead at 50K files.
                    for idx, (file_path, chunks, parse_error) in enumerate(
                        parse_results
                    ):
                        rel_path = rel_paths.get(file_path, str(file_path))
                        file_hash = file_hashes.get(file_path, "")

                        if idx % 10 == 0:
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

                        if parse_error:
                            logger.error(
                                f"Failed to chunk file {file_path}: {parse_error}"
                            )
                            continue

                        try:
                            if not chunks:
                                # Use TRACE to avoid cluttering progress displays
                                logger.trace(f"No chunks extracted from {file_path}")
                                continue

                            # Build hierarchical relationships (CPU-bound, run in thread pool)
                            chunks_with_hierarchy = await asyncio.to_thread(
                                self.chunk_processor.build_chunk_hierarchy, chunks
                            )

                            # Convert CodeChunk objects to dicts for storage
                            chunk_dicts = [
                                chunk_to_storage_dict(chunk, rel_path)
                                for chunk in chunks_with_hierarchy
                            ]

                            # Accumulate chunks for a single batched write after the
                            # full batch is parsed.  file_hash is injected here so
                            # add_chunks_batch() can deduplicate per-file.  This mirrors
                            # the two-pass path (see _flush_pending_chunks / add_chunks_batch).
                            # NOTE: the old per-file add_chunks() call was the primary cause
                            # of GPU starvation: 512 sequential EFS writes (25-100s) per
                            # batch while the GPU sat idle.
                            if chunk_dicts:
                                for cd in chunk_dicts:
                                    cd["file_hash"] = file_hash
                                batch_chunks.extend(chunk_dicts)
                                batch_files_processed += 1
                                logger.trace(
                                    f"Chunked {len(chunk_dicts)} chunks from {rel_path}"
                                )

                        except Exception as e:
                            logger.error(f"Failed to chunk file {file_path}: {e}")
                            continue

                    # Write all accumulated batch chunks to chunks.lance in ONE call.
                    # This replaces N per-file add_chunks() calls with a single
                    # add_chunks_batch() call per batch, eliminating the primary source
                    # of GPU starvation (N×EFS writes → 1×EFS write per batch).
                    if batch_chunks:
                        count = await self.chunks_backend.add_chunks_batch(batch_chunks)
                        chunks_created += count

                    files_indexed += batch_files_processed

                    # Put batch on queue for embedding.
                    # FIX 1 (CRITICAL RELIABILITY): Race put() against pipeline_cancel
                    # so producers don't hang forever if the consumer crashes and the
                    # queue is full.
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
                        put_task = asyncio.ensure_future(
                            chunk_queue.put(
                                {
                                    "chunks": batch_chunks,
                                    "batch_size": len(batch_chunks),
                                }
                            )
                        )
                        cancel_task = asyncio.ensure_future(pipeline_cancel.wait())
                        done, pending = await asyncio.wait(
                            [put_task, cancel_task],
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for t in pending:
                            t.cancel()
                            try:
                                await t
                            except asyncio.CancelledError:
                                pass
                        if pipeline_cancel.is_set():
                            logger.warning(
                                "Producer %d: pipeline cancelled, stopping", producer_id
                            )
                            return
                        # Yield to event loop to let consumer process the batch
                        await asyncio.sleep(0)

                # Update metrics for this producer's share
                parsing_metrics.item_count = files_indexed

            # Track chunking separately (each producer updates the shared metric)
            with metrics_tracker.phase("chunking") as chunking_metrics:
                chunking_metrics.item_count = chunks_created

            # Log this producer's completion
            self.memory_monitor.log_memory_summary()
            logger.info(
                f"✓ Producer {producer_id} complete: processed slice of {len(file_slice)} files"
            )

            # Each producer sends its own sentinel so the consumer knows when ALL are done
            await chunk_queue.put(None)

        async def embed_consumer():
            """Consumer: Take chunks from queue, embed, and store to vectors.lance."""
            nonlocal chunks_embedded

            logger.info(
                "🧠 Phase 2: Embedding pending chunks (GPU processing for semantic search)..."
            )

            embedding_batch_size = (
                self.embed_batch_size
            )  # GPU embedding batch size (separate from file batch size)
            consecutive_errors = 0  # Track consecutive errors to detect fatal issues
            # Track start time for ETA calculation
            embed_start_time = time.time()

            # Batched LanceDB writes: buffer chunks to reduce fragment count
            # Use configured write_batch_size if set; otherwise auto-detect from filesystem type.
            if self._indexer_config.write_batch_size is not None:
                write_batch_size = self._indexer_config.write_batch_size
            else:
                _fs_type = _detect_filesystem_type(self._mcp_dir / "lance")
                _fs_batch_defaults: dict[str, int] = {
                    "nfs": 16384,
                    "nvme": 1024,
                    "default": 4096,
                }
                write_batch_size = _fs_batch_defaults[_fs_type]
                if _fs_type != "default":
                    logger.info(
                        "Detected %s filesystem: write_batch_size=%d",
                        _fs_type.upper(),
                        write_batch_size,
                    )
            write_buffer: list[dict] = []

            # Count sentinels: one per producer — consumer stops after all producers finish
            sentinels_received = 0

            with metrics_tracker.phase("embedding") as embedding_metrics:
                try:
                    while sentinels_received < effective_num_producers:
                        # Get next batch from queue (blocks until available)
                        batch_data = await chunk_queue.get()

                        # Check for completion signal from a producer
                        if batch_data is None:
                            sentinels_received += 1
                            logger.debug(
                                f"Received sentinel {sentinels_received}/{effective_num_producers} from producers"
                            )
                            continue

                        chunks = batch_data["chunks"]

                        if not chunks:
                            continue

                        # Process chunks in embedding batches
                        for emb_batch_start in range(
                            0, len(chunks), embedding_batch_size
                        ):
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

                                # FIX 4 (HIGH): Wrap the synchronous embedding call in
                                # asyncio.to_thread so the event loop thread is not
                                # blocked for 100ms-2s per batch.  Progress display,
                                # memory monitoring, and cancellation can run during
                                # the embedding without this blocking them.
                                vectors = None
                                if hasattr(self.database, "_embedding_function"):
                                    vectors = await asyncio.to_thread(
                                        self.database._embedding_function, contents
                                    )
                                elif hasattr(self.database, "_collection") and hasattr(
                                    self.database._collection, "_embedding_function"
                                ):
                                    vectors = await asyncio.to_thread(
                                        self.database._collection._embedding_function,
                                        contents,
                                    )
                                elif hasattr(self.database, "embedding_function"):
                                    vectors = await asyncio.to_thread(
                                        self.database.embedding_function.embed_documents,
                                        contents,
                                    )
                                else:
                                    logger.error(
                                        "Cannot access embedding function from database, skipping batch"
                                    )
                                    continue

                                if vectors is None:
                                    raise ValueError("Failed to generate embeddings")

                                # Add to write buffer
                                for chunk, vec in zip(emb_batch, vectors, strict=True):
                                    write_buffer.append(
                                        {
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
                                    )
                                chunks_embedded += len(emb_batch)

                                # Flush write buffer when it reaches write_batch_size
                                # This reduces LanceDB fragment count and EFS I/O amplification
                                if len(write_buffer) >= write_batch_size:
                                    model_name = self.get_embedding_model_name()
                                    await self.vectors_backend.add_vectors(
                                        write_buffer, model_version=model_name
                                    )
                                    logger.debug(
                                        f"Flushed {len(write_buffer)} vectors to LanceDB"
                                    )
                                    write_buffer = []

                                # Reset consecutive error counter on success
                                consecutive_errors = 0

                                # Periodic GC to release Arrow/LanceDB buffers that
                                # CPython's reference-counter may not catch immediately.
                                # Runs every 10 000 embedded chunks to keep overhead low.
                                if (
                                    chunks_embedded % 10_000 == 0
                                    and chunks_embedded > 0
                                ):
                                    import gc

                                    gc.collect()
                                    logger.debug(
                                        "GC collect at %d embedded chunks (Arrow buffer cleanup)",
                                        chunks_embedded,
                                    )

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

                finally:
                    # FIX 1 (CRITICAL RELIABILITY): Signal producers that the
                    # consumer is exiting (whether normally or due to an error) so
                    # they stop blocking on chunk_queue.put().
                    pipeline_cancel.set()

                    # Always flush remaining write buffer even if an exception occurred
                    if write_buffer:
                        try:
                            model_name = self.get_embedding_model_name()
                            await self.vectors_backend.add_vectors(
                                write_buffer, model_version=model_name
                            )
                            logger.debug(
                                f"Flushed remaining {len(write_buffer)} vectors to LanceDB"
                            )
                        except Exception as flush_err:
                            # FIX 7 (MEDIUM): Mark lost chunk IDs for re-embedding on
                            # next run instead of silently losing vectors forever.
                            lost_ids = [
                                c["chunk_id"] for c in write_buffer if "chunk_id" in c
                            ]
                            logger.error(
                                "CRITICAL: Failed to flush %d vectors to LanceDB: %s",
                                len(lost_ids),
                                flush_err,
                            )
                            if lost_ids and hasattr(self, "chunks_backend"):
                                try:
                                    await self.chunks_backend.mark_chunks_error(
                                        lost_ids, str(flush_err)
                                    )
                                except Exception:
                                    pass
                        write_buffer = []

                # Update metrics
                embedding_metrics.item_count = chunks_embedded

            logger.info(f"✓ Phase 2 complete: {chunks_embedded} chunks embedded")

        # Split files_to_process into num_producers roughly equal slices (stride-based)
        # Stride-based (round-robin) distributes files evenly while keeping similar
        # file types together for better cache locality in the parser.
        producer_slices = [
            files_to_process[i::effective_num_producers]
            for i in range(effective_num_producers)
        ]
        logger.info(
            f"Starting pipeline: {effective_num_producers} producer(s) + 1 consumer "
            f"(Phase 1 parsing and Phase 2 embedding will overlap)"
        )

        # Run producer and consumer concurrently with pipeline parallelism
        # Consumer must be created FIRST so producer tasks can reference it
        consumer_task = asyncio.create_task(embed_consumer())

        # Create one producer task per slice
        producer_tasks = [
            asyncio.create_task(chunk_producer(i, producer_slices[i]))
            for i in range(effective_num_producers)
        ]

        # Wait for all tasks to complete with proper error handling.
        # FIX 5 (HIGH): Use return_exceptions=True so a second simultaneous
        # failure is not swallowed when the first exception is raised.
        all_tasks = producer_tasks + [consumer_task]
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        exceptions = [
            r
            for r in results
            if isinstance(r, BaseException)
            and not isinstance(r, asyncio.CancelledError)
        ]
        if exceptions:
            logger.error(f"Pipeline task failed: {exceptions[0]}")
            # Log any additional simultaneous failures
            for exc in exceptions[1:]:
                logger.error("Additional pipeline task failure: %s", exc)

            # Cancel all remaining tasks if one fails
            for task in all_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # Re-raise the first exception
            raise exceptions[0]

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
    ) -> tuple[list[tuple[str, str, str]], set[str], dict[Path, str]]:
        """Detect file moves/renames by matching content hashes.

        Delegates to :func:`core.file_move_detector.detect_file_moves`.
        """
        return _detect_file_moves_fn(
            project_root=self.project_root,
            current_files=current_files,
            indexed_file_hashes=indexed_file_hashes,
            progress_tracker=self.progress_tracker,
        )

    async def _phase1_chunk_files(
        self,
        files: list[Path],
        force: bool = False,
        file_hash_cache: dict[Path, str] | None = None,
    ) -> tuple[int, int]:
        """Phase 1: Parse and chunk files, store to chunks.lance.

        This phase is fast and durable - no expensive embedding generation.
        Incremental updates are supported via file_hash change detection.

        Args:
            files: Files to process
            force: If True, re-chunk even if unchanged
            file_hash_cache: Pre-computed file hashes from the caller.  When
                provided the caller has ALREADY performed change detection and
                ``files`` contains only files that need processing — skip the
                redundant ``get_all_indexed_file_hashes`` / ``_detect_file_moves``
                / change-detection loop and treat every entry in ``files`` as
                requiring processing.  When None (default) the full change
                detection is run (backward-compatible standalone behaviour).

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

            # When file_hash_cache is provided the caller already ran change
            # detection — every file in `files` needs processing.  Skip the
            # redundant get_all_indexed_file_hashes / _detect_file_moves /
            # per-file hash loop to avoid duplicate "Scanning files" bars and
            # wasted DB round-trips.
            caller_prefiltered = file_hash_cache is not None

            if caller_prefiltered:
                # Caller has already filtered; build files_to_process directly.
                _fhc: dict[Path, str] = file_hash_cache  # type: ignore[assignment]
                for file_path in files:
                    try:
                        file_hash = _fhc.get(file_path) or compute_file_hash(file_path)
                        rel_path = str(file_path.relative_to(self.project_root))
                        files_to_delete.append(rel_path)
                        files_to_process.append((file_path, rel_path, file_hash))
                    except Exception as e:
                        logger.error(f"Failed to check file {file_path}: {e}")
                        continue
            else:
                # Standalone call: run full change detection (backward-compatible).

                # OPTIMIZATION: Load all indexed file hashes ONCE for O(1) per-file lookup
                # This replaces 39K per-file database queries with a single scan
                indexed_file_hashes: dict[str, str] = {}
                _local_fhc: dict[Path, str] = {}
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
                    detected_moves, _moved_old_paths, _local_fhc = (
                        self._detect_file_moves(files, indexed_file_hashes)
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

                # Log start of hash computation phase
                if force:
                    logger.info("Force mode enabled, skipping file hash computation...")
                else:
                    logger.info(
                        f"Computing file hashes for change detection ({len(files)} files)..."
                    )

                for idx, file_path in enumerate(files):
                    try:
                        # Log progress every 1000 files (fallback when no progress tracker)
                        if idx > 0 and idx % 1000 == 0 and not self.progress_tracker:
                            logger.info(f"Computing file hashes: {idx}/{len(files)}")

                        # Skip hash computation when force=True (all files will be reindexed)
                        if force:
                            file_hash = ""  # Empty hash when forcing full reindex
                        else:
                            # Reuse hash from move-detection cache to avoid re-hashing
                            file_hash = _local_fhc.get(file_path) or compute_file_hash(
                                file_path
                            )

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

            # Batch delete old chunks for changed files before inserting new ones.
            # This prevents duplicate chunks from accumulating across re-index runs.
            # Note: LanceDB delete() is metadata-only (deletion vectors) and does NOT
            # trigger compact_files(). The SIGBUS workaround for macOS applies only to
            # compaction, which is separately guarded in _compact_table().
            if not self._atomic_rebuild_active and files_to_delete:
                deleted_count = await self.chunks_backend.delete_files_batch(
                    files_to_delete
                )
                if deleted_count > 0:
                    logger.info(
                        f"Batch deleted {deleted_count} old chunks for {len(files_to_delete)} files"
                    )

            # Now process files for parsing and chunking.
            # PERFORMANCE: Accumulate chunks across files and flush to LanceDB in
            # batches of write_batch_files files instead of one `add_chunks` call per
            # file.  This reduces LanceDB Arrow-allocation overhead from O(n_files) to
            # O(n_files / batch_size) — roughly 20x fewer write round-trips for a
            # 5,272-file repo with the default batch size of 256.
            write_batch_files = 256
            backend_fatal_error: Exception | None = None
            # Buffer accumulates (chunk_dict_with_hash, file_path) tuples.
            pending_chunks: list[dict] = []  # chunk dicts with file_hash pre-injected
            pending_files_count = 0  # number of source files in pending_chunks

            async def _flush_pending_chunks() -> int:
                """Write accumulated pending_chunks to LanceDB; return count written."""
                nonlocal pending_files_count
                if not pending_chunks:
                    return 0
                count = await self.chunks_backend.add_chunks_batch(pending_chunks)
                pending_chunks.clear()
                pending_files_count = 0
                return count

            for _idx, (file_path, rel_path, file_hash) in enumerate(files_to_process):
                # Check memory before processing each file (CRITICAL: not every 100!)
                is_ok, usage_pct, status = self.memory_monitor.check_memory_limit()
                if not is_ok:
                    logger.warning(
                        f"Memory limit exceeded during chunking "
                        f"({usage_pct * 100:.1f}% of {self.memory_monitor.max_memory_gb:.1f}GB), "
                        "waiting for memory to free up..."
                    )
                    # Flush before waiting so we don't hold large buffers in memory
                    await _flush_pending_chunks()
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

                    # Convert CodeChunk objects to dicts and inject file_hash for batching
                    file_chunk_count = 0
                    for chunk in chunks_with_hierarchy:
                        chunk_dict = chunk_to_storage_dict(chunk, rel_path, file_hash)
                        pending_chunks.append(chunk_dict)
                        file_chunk_count += 1

                    if file_chunk_count > 0:
                        files_processed += 1
                        chunks_created += file_chunk_count
                        pending_files_count += 1
                        logger.debug(
                            f"Parsed {file_chunk_count} chunks from {rel_path}"
                        )

                        # Flush when write-batch is full
                        if pending_files_count >= write_batch_files:
                            try:
                                await _flush_pending_chunks()
                            except DatabaseError as db_err:
                                err_lower = str(db_err).lower()
                                if (
                                    "not found" in err_lower
                                    or "not initialized" in err_lower
                                ):
                                    logger.error(
                                        f"Backend error flushing chunk batch: {db_err}\n"
                                        f"  Aborting Phase 1 to avoid repeated failures.\n"
                                        f"  Try: mvs index --force  (to rebuild from scratch)"
                                    )
                                    backend_fatal_error = db_err
                                    break
                                logger.error(f"Failed to flush chunk batch: {db_err}")

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
                    break

            # Flush any remaining chunks that didn't fill a full batch
            if pending_chunks and backend_fatal_error is None:
                if self.progress_tracker:
                    sys.stderr.write(
                        f"  Saving final batch ({len(pending_chunks):,} chunks)...\n"
                    )
                    sys.stderr.flush()
                try:
                    await _flush_pending_chunks()
                except DatabaseError as db_err:
                    logger.error(f"Failed to flush final chunk batch: {db_err}")

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
        return _build_hierarchy_path_fn(chunk)

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

        Thin delegate to :func:`~.embedding_runner.run_phase2_embedding`.

        Args:
            batch_size: Chunks per embedding batch
            checkpoint_interval: Chunks between checkpoint logs

        Returns:
            Tuple of (chunks_embedded, batches_processed)
        """
        return await run_phase2_embedding(
            chunks_backend=self.chunks_backend,
            vectors_backend=self.vectors_backend,
            database=self.database,
            memory_monitor=self.memory_monitor,
            progress_tracker=self.progress_tracker,
            get_model_name_fn=self.get_embedding_model_name,
            get_project_name_fn=self._get_project_name,
            batch_size=batch_size,
            checkpoint_interval=checkpoint_interval,
        )

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

        Thin delegate to :class:`~.atomic_rebuild.AtomicRebuildManager`.

        Args:
            force: If True, perform atomic rebuild

        Returns:
            True if atomic rebuild was performed, False otherwise
        """
        manager = AtomicRebuildManager(self._mcp_dir, self.config)
        active = await manager.rebuild(force=force)
        if active:
            self.chunks_backend = manager.chunks_backend
            self.vectors_backend = manager.vectors_backend
            self.database = manager.database
            self._atomic_rebuild_active = True
        else:
            self._atomic_rebuild_active = False
        # Store manager so _finalize_atomic_rebuild can reuse the same mcp_dir reference
        self._atomic_rebuild_manager = manager
        return active

    async def _finalize_atomic_rebuild(self) -> None:
        """Finalize atomic rebuild by atomically switching databases.

        Called after successful indexing when force_reindex=True.
        Performs atomic rename operations to switch from .new to final.

        Thin delegate to :class:`~.atomic_rebuild.AtomicRebuildManager`.
        """
        manager = getattr(self, "_atomic_rebuild_manager", None)
        if manager is None:
            # Fallback: construct a manager with the current mcp_dir
            manager = AtomicRebuildManager(self._mcp_dir, self.config)
        await manager.finalize()
        self._atomic_rebuild_active = manager.active

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

        # Auto-repair: rebuild index_metadata.json if missing but chunks exist
        if not self.metadata._metadata_file.exists():
            try:
                chunk_count = (
                    self.chunks_backend._table.count_rows()
                    if self.chunks_backend._table
                    else 0
                )
                if chunk_count > 0:
                    logger.info(
                        "Auto-rebuilding missing index_metadata.json from chunks database..."
                    )
                    if self.progress_tracker:
                        sys.stderr.write(
                            "  Rebuilding index metadata (auto-repair)...\n"
                        )
                        sys.stderr.flush()
                    await self._auto_rebuild_metadata()
            except Exception as e:
                logger.warning(
                    f"Auto-rebuild of index metadata failed (non-fatal): {e}"
                )

        # Apply auto-optimizations if enabled
        if self.auto_optimize:
            self.apply_auto_optimizations()

        # Clean up stale lock files
        cleanup_stale_locks(self.index_path)

        # Clean up stale LanceDB transaction files and stale progress.json
        # from any previous run that crashed mid-initialisation (non-fatal).
        cleanup_stale_transactions(self.index_path)
        cleanup_stale_progress(self.index_path)

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
            detected_moves, _moved_old_paths, file_hash_cache = self._detect_file_moves(
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
                    file_hash = file_hash_cache.get(f) or compute_file_hash(f)
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
                if files_checked % 1000 == 0 and not self.progress_tracker:
                    logger.info(
                        f"Change detection progress: {files_checked}/{len(all_files)} files checked..."
                    )

            files_to_index = filtered_files
            logger.info(
                f"Change detection: {len(files_to_index)} changed, "
                f"{files_skipped} unchanged out of {len(all_files)} total files"
            )

        # Process files through Phase 1.
        # Pass file_hash_cache so _phase1_chunk_files skips redundant change
        # detection (get_all_indexed_file_hashes / _detect_file_moves / hash
        # loop) — all of that was already done above.
        if files_to_index:
            _phase1_fhc: dict[Path, str] | None = file_hash_cache if not fresh else None
            indexed_count, created = await self._phase1_chunk_files(
                files_to_index, force=fresh, file_hash_cache=_phase1_fhc
            )
            files_processed = indexed_count
            chunks_created = created

        # Update metadata after chunking
        if files_processed > 0:
            if self.progress_tracker:
                sys.stderr.write("  Updating directory index...\n")
                sys.stderr.flush()
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

        # Build BM25 index — provide console feedback since this can take
        # 10-30 s for large projects (reads all chunks from LanceDB).
        if self.progress_tracker:
            sys.stderr.write("  Loading chunks for keyword index...\n")
            sys.stderr.flush()
        await self._build_bm25_index()

        # Finalize atomic rebuild if fresh and we processed files
        if fresh and atomic_rebuild_active and files_processed > 0:
            await self._finalize_atomic_rebuild()

            # CRITICAL: Re-initialize backends after finalization
            # After _finalize_atomic_rebuild(), the .new directories have been renamed
            # to final paths, but backend objects still point to old .new paths.
            # We must reinitialize backends to point to the correct final paths.
            base_path = self._mcp_dir
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
        if self.progress_tracker:
            sys.stderr.write("  Starting embedding phase...\n")
            sys.stderr.flush()
        chunks_embedded, batches_processed = await self._phase2_embed_chunks(
            batch_size=batch_size, checkpoint_interval=50000
        )

        # Build ANN vector index after embedding completes.
        # This activates IVF_SQ approximate nearest-neighbor search for large
        # datasets.  Skipped automatically for small datasets (< 4,096 rows).
        # Non-fatal: search falls back to brute-force if index creation fails.
        if self.progress_tracker:
            sys.stderr.write("  Building vector search index...\n")
            sys.stderr.flush()
        await self.vectors_backend.rebuild_index()

        # NOTE: BM25 index is NOT rebuilt here — it depends only on text
        # content (not embeddings) and was already built by chunk_files().
        # Rebuilding it again would be redundant for the chunk_and_embed path.

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
    ) -> "IndexResult":
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
            IndexResult subclassing int for backward compatibility.  The integer
            value is the number of files processed; additional attributes
            (.chunks_indexed, .duration_seconds, .errors, .status) carry the
            richer metadata introduced by issue #115.

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

        Raises:
            IndexingError: If an unexpected error occurs during indexing.
                           Note: ``DatabaseInitializationError`` is handled internally
                           (logged and returns 0) to allow graceful degradation.
        """
        try:
            return await self._index_project_impl(
                force_reindex=force_reindex,
                show_progress=show_progress,
                skip_relationships=skip_relationships,
                phase=phase,
                metrics_json=metrics_json,
                pipeline=pipeline,
            )
        except IndexingError:
            raise
        except Exception as e:
            raise IndexingError(
                f"Indexing failed for project {self.project_root}: {e}",
                context={"project_root": str(self.project_root), "phase": phase},
            ) from e

    async def _index_project_impl(
        self,
        force_reindex: bool = False,
        show_progress: bool = True,
        skip_relationships: bool = False,
        phase: str = "all",
        metrics_json: bool = False,
        pipeline: bool = True,
    ) -> int:
        """Internal implementation of index_project (no public exception wrapping)."""
        logger.info(
            f"Starting indexing of project: {self.project_root} (phase: {phase})"
        )

        _index_start = time.time()
        _index_errors: list[str] = []

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
            return IndexResult(
                files_processed=0,
                errors=[str(e)],
                status="error",
                duration_seconds=time.time() - _index_start,
            )
        try:
            await self.vectors_backend.initialize()
        except DatabaseInitializationError as e:
            logger.error(
                f"Cannot start indexing: {e}\n"
                f"  Try: mvs index --force  (to rebuild from scratch)\n"
                f"  Or:  mvs reset index --force  (to clear all data)"
            )
            return IndexResult(
                files_processed=0,
                errors=[str(e)],
                status="error",
                duration_seconds=time.time() - _index_start,
            )

        # Check and run pending migrations automatically
        await self._run_auto_migrations()

        # Apply auto-optimizations before indexing
        if self.auto_optimize:
            self.apply_auto_optimizations()

        # Clean up stale lock files from previous interrupted indexing runs
        cleanup_stale_locks(self.index_path)

        # Clean up stale LanceDB transaction files and stale progress.json
        # from any previous run that crashed mid-initialisation (non-fatal).
        cleanup_stale_transactions(self.index_path)
        cleanup_stale_progress(self.index_path)

        # Track indexed count for backward compatibility
        indexed_count = 0
        chunks_created = 0
        chunks_embedded = 0
        files_to_index = []  # Initialize to empty list for directory index updates

        # Decide whether to use pipeline or sequential execution.
        # two_pass forces sequential mode even when the caller requests pipeline.
        use_pipeline = pipeline and phase == "all" and not self._indexer_config.two_pass

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
                        return IndexResult(
                            files_processed=0,
                            duration_seconds=time.time() - _index_start,
                        )
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
                        detected_moves, _moved_old_paths, file_hash_cache = (
                            self._detect_file_moves(all_files, indexed_file_hashes)
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
                                file_hash = file_hash_cache.get(f) or compute_file_hash(
                                    f
                                )
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

                            if idx % 500 == 0 and not self.progress_tracker:
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
                            return IndexResult(
                                files_processed=0,
                                duration_seconds=time.time() - _index_start,
                            )

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

        # Persist embedding model info into index_metadata.json
        try:
            _model_name = self.get_embedding_model_name()
            await self._save_embedding_metadata(_model_name)
        except Exception as _meta_err:
            logger.warning(f"Failed to persist embedding metadata: {_meta_err}")

        _index_duration = time.time() - _index_start
        _result_status = (
            "error"
            if indexed_count == 0 and _index_errors
            else "partial"
            if _index_errors
            else "success"
        )
        return IndexResult(
            files_processed=indexed_count,
            chunks_indexed=chunks_created,
            duration_seconds=_index_duration,
            errors=_index_errors,
            status=_result_status,
        )

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

        # Skip delete on force reindex with fresh database OR when atomic rebuild is active.
        # This optimization eliminates ~25k unnecessary database queries for large codebases.
        # Note: LanceDB delete() is metadata-only and does NOT trigger compact_files().
        if not skip_delete and not self._atomic_rebuild_active:
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

            # Only schedule deletion if not building fresh database AND not in atomic rebuild.
            # LanceDB delete() is metadata-only and does NOT trigger compact_files().
            if not skip_delete and not self._atomic_rebuild_active:
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
                    chunk_dicts = [
                        chunk_to_storage_dict(chunk, rel_path) for chunk in file_chunks
                    ]

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

            # Skip delete on force reindex with fresh database OR when atomic rebuild is active.
            # LanceDB delete() is metadata-only and does NOT trigger compact_files().
            if not skip_delete and not self._atomic_rebuild_active:
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
            chunk_dicts = [
                chunk_to_storage_dict(chunk, rel_path)
                for chunk in chunks_with_hierarchy
            ]

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

        LanceDB delete() is metadata-only (creates deletion vectors) and does NOT
        trigger compact_files(). The macOS SIGBUS workaround applies only to
        compaction, which is separately guarded in _compact_table().

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

    async def _auto_rebuild_metadata(self) -> None:
        """Auto-rebuild index_metadata.json from chunks database.

        Called when metadata file is missing but chunks exist.
        Reconstructs file modification times by querying the chunks table
        for unique file paths and checking filesystem mtimes.
        """
        import os as _os

        if self.chunks_backend._table is None:
            return

        # Read unique file paths from chunks table (column projection for speed)
        scanner = self.chunks_backend._table.to_lance().scanner(columns=["file_path"])
        table = scanner.to_table()
        df = table.to_pandas()

        if df.empty:
            return

        file_paths = df["file_path"].unique().tolist()

        # Build metadata dict from filesystem mtimes
        metadata: dict[str, float] = {}
        for fp in file_paths:
            full_path = self.project_root / fp
            if full_path.exists():
                try:
                    metadata[fp] = _os.path.getmtime(full_path)
                except OSError:
                    pass

        # Save rebuilt metadata
        self.metadata.save(metadata)
        logger.info(
            f"Auto-rebuilt index_metadata.json: {len(metadata)} files "
            f"(from {len(file_paths)} in database)"
        )

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

    async def get_project_status(self) -> ProjectStatus:
        """Return a stable ProjectStatus aggregating stats from all backends.

        Queries both chunks_backend (Phase 1 - parsed chunks) and
        vectors_backend (Phase 2 - embedded vectors) so callers get a
        complete picture without having to know the two-phase internals.

        Returns:
            ProjectStatus dataclass with total_files, total_chunks,
            total_vectors, index_size_bytes, last_indexed, status, and a
            per-backend breakdown in ``backends``.

        The ``status`` field follows this convention:
        - "ready"   — vectors exist, search is available.
        - "empty"   — no chunks and no vectors (never indexed).
        - "indexing" — chunks exist but embedding is pending/in-progress.
        - "error"   — backends could not be queried.
        """
        import os

        try:
            if self.chunks_backend._db is None:
                await self.chunks_backend.initialize()
            if self.vectors_backend._db is None:
                await self.vectors_backend.initialize()

            chunks_stats = await self.chunks_backend.get_stats()
            vectors_stats = await self.vectors_backend.get_stats()

            total_chunks = chunks_stats.get("total", 0)
            total_vectors = vectors_stats.get("total", 0)
            total_files = chunks_stats.get("files", 0)

            # Compute on-disk size of the index directory
            index_size_bytes = 0
            if self.index_path.exists():
                for root, _dirs, files in os.walk(self.index_path):
                    for fname in files:
                        try:
                            index_size_bytes += os.path.getsize(
                                os.path.join(root, fname)
                            )
                        except OSError:
                            pass

            # Determine high-level status
            if total_vectors > 0:
                derived_status = "ready"
            elif total_chunks > 0:
                pending = chunks_stats.get("pending", 0)
                processing = chunks_stats.get("processing", 0)
                derived_status = "indexing" if (pending + processing) > 0 else "ready"
            else:
                derived_status = "empty"

            return ProjectStatus(
                total_files=total_files,
                total_chunks=total_chunks,
                total_vectors=total_vectors,
                index_size_bytes=index_size_bytes,
                last_indexed=None,  # Not persisted yet — reserved for future use
                status=derived_status,
                backends={
                    "chunks": chunks_stats,
                    "vectors": vectors_stats,
                },
            )

        except Exception as e:
            logger.error(f"Failed to get project status: {e}")
            return ProjectStatus(status="error")

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

        Delegates to :func:`core.bm25_builder.build_bm25_index`.
        """
        await _build_bm25_index_fn(
            chunks_backend=self.chunks_backend,
            mcp_dir=self._mcp_dir,
            progress_tracker=self.progress_tracker,
        )

    # ------------------------------------------------------------------
    # Public API: metadata and health
    # ------------------------------------------------------------------

    def get_index_metadata(self) -> dict[str, object]:
        """Return the persisted index metadata for this project.

        The dict includes embedding model name, vector dimensions, tool
        version, and ISO-8601 timestamps for index creation / last update.
        All values default to ``None`` when the index has not been built yet.

        Returns:
            dict with keys: ``embedding_model``, ``embedding_dimensions``,
            ``index_version``, ``created_at``, ``updated_at``.
        """
        return self.metadata.get_index_metadata()

    async def health_check(self) -> HealthStatus:
        """Perform a lightweight structural health check (< 100 ms).

        Checks performed (no search queries are executed):

        1. LanceDB directory is reachable.
        2. Determine which LanceDB tables exist on disk.
        3. Embedding function / model is attached to the database instance.
        4. Both ``chunks`` and ``vectors`` tables exist (index is non-empty).

        The overall ``status`` field is set as follows:

        * ``"healthy"``  – DB connected, model loaded, both tables present.
        * ``"degraded"`` – DB connected but missing tables or model not
          loaded yet (partial index, e.g. only Phase 1 complete).
        * ``"unhealthy"``– DB not reachable or a hard error occurred.

        Returns:
            HealthStatus dataclass instance.
        """
        health = HealthStatus()

        try:
            # 1. Check DB connection ----------------------------------------
            lance_path = self._mcp_dir / "lance"
            if lance_path.exists():
                import lancedb

                try:
                    db = await asyncio.to_thread(lancedb.connect, str(lance_path))
                    health.db_connected = True
                    health.details["db_path"] = str(lance_path)

                    # 2. List existing tables ----------------------------------
                    tables_response = await asyncio.to_thread(db.list_tables)
                    if hasattr(tables_response, "tables"):
                        health.tables_exist = list(tables_response.tables)
                    else:
                        health.tables_exist = list(tables_response)

                except Exception as db_err:
                    health.details["db_error"] = str(db_err)
            else:
                health.details["db_path"] = str(lance_path)
                health.details["db_missing"] = "lance directory does not exist"

            # 3. Check if embedding model is available -----------------------
            model_name: str | None = None
            try:
                model_name = self.get_embedding_model_name()
                if model_name and model_name != "unknown":
                    health.model_loaded = True
                    health.details["embedding_model"] = model_name
            except Exception as model_err:
                health.details["model_error"] = str(model_err)

            # 4. Structural validity -----------------------------------------
            has_chunks = "chunks" in health.tables_exist
            has_vectors = "vectors" in health.tables_exist
            health.index_valid = health.db_connected and has_chunks and has_vectors

            if has_chunks:
                health.details["chunks_table"] = "present"
            else:
                health.details["chunks_table"] = "missing"

            if has_vectors:
                health.details["vectors_table"] = "present"
            else:
                health.details["vectors_table"] = "missing"

            # 5. Determine overall status ------------------------------------
            if health.db_connected and health.model_loaded and health.index_valid:
                health.status = "healthy"
            elif health.db_connected and (has_chunks or health.model_loaded):
                health.status = "degraded"
            else:
                health.status = "unhealthy"

        except Exception as e:
            health.status = "unhealthy"
            health.details["error"] = str(e)
            logger.warning(f"health_check() encountered an error: {e}")

        return health

    async def _save_embedding_metadata(self, model_name: str) -> None:
        """Persist embedding model name and dimensions into index_metadata.json.

        Called internally at the end of a successful indexing run so that
        the metadata file reflects the model that produced the current index.

        Args:
            model_name: Fully-qualified model name used for embedding.
        """
        try:
            from ..config.defaults import get_model_dimensions

            try:
                dims = get_model_dimensions(model_name)
            except Exception:
                dims = None  # Unknown model – omit dimension

            file_mtimes = self.metadata.load()
            self.metadata.save(
                file_mtimes,
                embedding_model=model_name,
                embedding_dimensions=dims,
            )
            logger.debug(f"Saved embedding metadata: model={model_name}, dims={dims}")
        except Exception as e:
            logger.warning(f"Failed to save embedding metadata: {e}")
