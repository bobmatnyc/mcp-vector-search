"""Semantic indexer for MCP Vector Search with two-phase architecture.

Phase 1: Parse and chunk files, store to chunks.lance (fast, durable)
Phase 2: Embed chunks, store to vectors.lance (resumable, incremental)
"""

import asyncio
import os
import shutil
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
from .chunk_processor import ChunkProcessor
from .chunks_backend import ChunksBackend, compute_file_hash
from .database import VectorDatabase
from .directory_index import DirectoryIndex
from .exceptions import ParsingError
from .file_discovery import FileDiscovery
from .index_metadata import IndexMetadata
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
    ) -> None:
        """Initialize semantic indexer.

        Args:
            database: Vector database instance
            project_root: Project root directory
            file_extensions: File extensions to index (deprecated, use config)
            config: Project configuration (preferred over file_extensions)
            max_workers: Maximum number of worker processes for parallel parsing (ignored if use_multiprocessing=False)
            batch_size: Number of files to process in each batch (default: 32, override with MCP_VECTOR_SEARCH_BATCH_SIZE or auto-optimization)
            debug: Enable debug output for hierarchy building
            collectors: Metric collectors to run during indexing (defaults to all complexity collectors)
            use_multiprocessing: Enable multiprocess parallel parsing (default: True, disable for debugging)
            auto_optimize: Enable automatic optimization based on codebase profile (default: True)

        Environment Variables:
            MCP_VECTOR_SEARCH_BATCH_SIZE: Override batch size (default: 32)
        """
        self.database = database
        self.project_root = project_root
        self.config = config
        self.auto_optimize = auto_optimize
        self._applied_optimizations: dict[str, Any] | None = None

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
                        f"Invalid batch size value: {env_batch_size}, using default 256"
                    )
                    self.batch_size = 256
            else:
                # Increased default from 128 to 256 for better storage throughput
                # Larger batches = fewer LanceDB writes = less overhead
                # 256 files per batch is a good balance between memory and throughput
                self.batch_size = 256
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

        self.chunk_processor = ChunkProcessor(
            parser_registry=self.parser_registry,
            monorepo_detector=self.monorepo_detector,
            max_workers=max_workers,
            use_multiprocessing=use_multiprocessing,
            debug=debug,
            repo_root=project_root,  # Enable git blame tracking
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

        with metrics_tracker.phase("parsing") as parsing_metrics:
            # OPTIMIZATION: Collect files that need deletion and batch delete upfront
            # This avoids O(n) delete_file_chunks() calls that each load the database
            files_to_delete = []
            files_to_process = []

            for file_path in files:
                try:
                    # Compute current file hash
                    file_hash = compute_file_hash(file_path)
                    rel_path = str(file_path.relative_to(self.project_root))

                    # Check if file changed (skip if unchanged and not forcing)
                    if not force:
                        file_changed = await self.chunks_backend.file_changed(
                            rel_path, file_hash
                        )
                        if not file_changed:
                            logger.debug(f"Skipping unchanged file: {rel_path}")
                            continue

                    # Mark file for deletion and processing
                    files_to_delete.append(rel_path)
                    files_to_process.append((file_path, rel_path, file_hash))

                except Exception as e:
                    logger.error(f"Failed to check file {file_path}: {e}")
                    continue

            # Batch delete old chunks for all files (skip if atomic rebuild is active)
            if not self._atomic_rebuild_active and files_to_delete:
                deleted_count = await self.chunks_backend.delete_files_batch(
                    files_to_delete
                )
                if deleted_count > 0:
                    logger.info(
                        f"Batch deleted {deleted_count} old chunks for {len(files_to_delete)} files"
                    )

            # Now process files for parsing and chunking
            for file_path, rel_path, file_hash in files_to_process:
                try:
                    # Parse file
                    chunks = await self.chunk_processor.parse_file(file_path)

                    if not chunks:
                        logger.debug(f"No chunks extracted from {file_path}")
                        continue

                    # Build hierarchical relationships
                    chunks_with_hierarchy = self.chunk_processor.build_chunk_hierarchy(
                        chunks
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

                except Exception as e:
                    logger.error(f"Failed to chunk file {file_path}: {e}")
                    continue

            # Update metrics
            parsing_metrics.item_count = files_processed

        # Track chunking separately
        with metrics_tracker.phase("chunking") as chunking_metrics:
            chunking_metrics.item_count = chunks_created
            # Note: Duration will be minimal since actual work was done in parsing phase
            # This is just for tracking the chunk count

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

        with metrics_tracker.phase("embedding") as embedding_metrics:
            while True:
                # Get pending chunks from chunks.lance
                pending = await self.chunks_backend.get_pending_chunks(batch_size)
                if not pending:
                    logger.info("No more pending chunks to embed")
                    break

                # Mark as processing (for crash recovery)
                chunk_ids = [c["chunk_id"] for c in pending]
                await self.chunks_backend.mark_chunks_processing(chunk_ids, batch_id)

                try:
                    # Generate embeddings using database's embedding function
                    contents = [c["content"] for c in pending]

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

                    # Store vectors to vectors.lance
                    await self.vectors_backend.add_vectors(chunks_with_vectors)

                    # Mark as complete in chunks.lance
                    await self.chunks_backend.mark_chunks_complete(chunk_ids)

                    chunks_embedded += len(pending)
                    batches_processed += 1

                    # Checkpoint logging
                    if chunks_embedded % checkpoint_interval == 0:
                        logger.info(f"Checkpoint: {chunks_embedded} chunks embedded")

                except Exception as e:
                    logger.error(f"Embedding batch failed: {e}")
                    # Mark chunks as error in chunks.lance
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    await self.chunks_backend.mark_chunks_error(chunk_ids, error_msg)
                    # Continue with next batch instead of crashing
                    continue

            # Update metrics
            embedding_metrics.item_count = chunks_embedded

        logger.info(
            f"✓ Phase 2 complete: {chunks_embedded} chunks embedded in {batches_processed} batches"
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
                    shutil.rmtree(new_path, ignore_errors=True)
            if chroma_new.exists():
                chroma_new.unlink()

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
            embedding_function, _ = create_embedding_function(
                model_name=self.config.embedding_model if self.config else "default"
            )
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
                    shutil.rmtree(lance_old, ignore_errors=True)

            # Step 3: Atomic switch for knowledge_graph directory
            if kg_new.exists():
                if kg_path.exists():
                    kg_path.rename(kg_old)
                kg_new.rename(kg_path)
                if kg_old.exists():
                    shutil.rmtree(kg_old, ignore_errors=True)

            # Step 4: Atomic switch for code_search.lance
            if code_search_new.exists():
                if code_search_path.exists():
                    code_search_path.rename(code_search_old)
                code_search_new.rename(code_search_path)
                if code_search_old.exists():
                    shutil.rmtree(code_search_old, ignore_errors=True)

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

    async def index_project(
        self,
        force_reindex: bool = False,
        show_progress: bool = True,
        skip_relationships: bool = False,
        phase: str = "all",
        metrics_json: bool = False,
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
        """
        logger.info(
            f"Starting indexing of project: {self.project_root} (phase: {phase})"
        )

        # Track total timing for entire indexing process
        t_start_total = time.time()

        # Perform atomic rebuild if force is enabled
        atomic_rebuild_active = await self._atomic_rebuild_databases(force_reindex)

        # Initialize metrics tracker
        metrics_tracker = get_metrics_tracker()
        metrics_tracker.reset()

        # Initialize backends
        t_start = time.time()
        await self.chunks_backend.initialize()
        await self.vectors_backend.initialize()
        t_init = time.time()
        print(f"⏱️  TIMING: Backend initialization: {t_init - t_start:.2f}s", flush=True)

        # Check and run pending migrations automatically
        t_start = time.time()
        await self._run_auto_migrations()
        t_migrations = time.time()
        print(f"⏱️  TIMING: Migrations check: {t_migrations - t_start:.2f}s", flush=True)

        # Apply auto-optimizations before indexing
        t_start = time.time()
        if self.auto_optimize:
            self.apply_auto_optimizations()
        t_optimize = time.time()
        print(f"⏱️  TIMING: Auto-optimizations: {t_optimize - t_start:.2f}s", flush=True)

        # Clean up stale lock files from previous interrupted indexing runs
        cleanup_stale_locks(self.project_root)

        # Track indexed count for backward compatibility
        indexed_count = 0
        chunks_created = 0
        chunks_embedded = 0

        # Phase 1: Chunk files (if requested)
        if phase in ("all", "chunk"):
            # Find all indexable files
            t_start = time.time()
            all_files = self.file_discovery.find_indexable_files()
            t_scan = time.time()
            print(
                f"⏱️  TIMING: File scanning ({len(all_files)} files): {t_scan - t_start:.2f}s",
                flush=True,
            )

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
                t_start = time.time()
                files_to_index = all_files
                if not force_reindex:
                    # Use chunks_backend for change detection instead of metadata
                    filtered_files = []
                    for f in all_files:
                        try:
                            file_hash = compute_file_hash(f)
                            rel_path = str(f.relative_to(self.project_root))
                            if await self.chunks_backend.file_changed(
                                rel_path, file_hash
                            ):
                                filtered_files.append(f)
                        except Exception as e:
                            logger.warning(
                                f"Error checking file {f}: {e}, will re-index"
                            )
                            filtered_files.append(f)
                    files_to_index = filtered_files
                    t_filter = time.time()
                    logger.info(
                        f"Incremental index: {len(files_to_index)} of {len(all_files)} files need updating"
                    )
                    print(
                        f"⏱️  TIMING: File change detection: {t_filter - t_start:.2f}s",
                        flush=True,
                    )
                else:
                    logger.info(
                        f"Force reindex: processing all {len(files_to_index)} files"
                    )
                    t_filter = time.time()
                    print(
                        f"⏱️  TIMING: File filtering (force): {t_filter - t_start:.2f}s",
                        flush=True,
                    )

                if files_to_index:
                    # Run Phase 1
                    t_start = time.time()
                    indexed_count, chunks_created = await self._phase1_chunk_files(
                        files_to_index, force=force_reindex
                    )
                    t_phase1 = time.time()
                    print(
                        f"⏱️  TIMING: Phase 1 (parsing/chunking): {t_phase1 - t_start:.2f}s",
                        flush=True,
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
            t_start = time.time()
            chunks_embedded, _ = await self._phase2_embed_chunks(
                batch_size=self.batch_size
            )
            t_phase2 = time.time()
            print(
                f"⏱️  TIMING: Phase 2 (embedding): {t_phase2 - t_start:.2f}s", flush=True
            )

        # Log summary with total timing
        t_end_total = time.time()
        total_time = t_end_total - t_start_total
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

        print(f"\n⏱️  TIMING: TOTAL indexing time: {total_time:.2f}s\n", flush=True)

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

        return (chunks_with_hierarchy, chunk_metrics)

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
        all_metrics: dict[str, Any] = {}
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

                # Merge metrics
                if chunk_metrics:
                    all_metrics.update(chunk_metrics)

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
                    await self.vectors_backend.add_vectors(chunks_with_vectors)

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
                await self.vectors_backend.add_vectors(chunks_with_vectors)

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
        # TIMING: Track overall indexing time
        t_start_index = time.time()
        print(
            f"\n🚀 Starting indexing with timing instrumentation for {len(files_to_index)} files...\n",
            flush=True,
        )

        # Initialize backends if not already initialized
        t_start = time.time()
        if self.chunks_backend._db is None:
            await self.chunks_backend.initialize()
        if self.vectors_backend._db is None:
            await self.vectors_backend.initialize()
        t_init = time.time()
        print(f"⏱️  TIMING: Backend initialization: {t_init - t_start:.2f}s", flush=True)

        # Write version header to error log at start of indexing run
        self.metadata.write_indexing_run_header()

        # TIMING: Track time spent in different phases
        time_parsing_total = 0.0
        time_embedding_total = 0.0
        time_storage_total = 0.0

        # OPTIMIZATION: Load metadata once at start instead of per batch
        # This avoids O(n) json.load() calls that kill performance on large codebases (25k+ files)
        metadata_dict = self.metadata.load()

        # Process files in batches for better memory management and embedding efficiency
        batch_count = 0
        for i in range(0, len(files_to_index), self.batch_size):
            batch = files_to_index[i : i + self.batch_size]
            batch_count += 1

            # Accumulate chunks from all files in batch
            all_chunks: list[CodeChunk] = []
            all_metrics: dict[str, Any] = {}
            file_to_chunks_map: dict[str, tuple[int, int]] = {}
            file_results: dict[Path, tuple[int, bool]] = {}

            # Parse all files in parallel
            t_start = time.time()
            tasks = []
            for file_path in batch:
                task = asyncio.create_task(
                    self._parse_and_prepare_file(
                        file_path,
                        force_reindex,
                        skip_delete=self._atomic_rebuild_active,
                    )
                )
                tasks.append(task)

            parse_results = await asyncio.gather(*tasks, return_exceptions=True)
            time_parsing_total += time.time() - t_start

            # OPTIMIZATION: Cache mtimes for batch to avoid repeated os.path.getmtime() calls
            # Compute all mtimes upfront before processing results
            batch_mtimes: dict[str, float] = {}
            for file_path in batch:
                try:
                    batch_mtimes[str(file_path)] = os.path.getmtime(file_path)
                except FileNotFoundError:
                    pass  # File deleted during indexing

            # Accumulate chunks from successfully parsed files
            # metadata_dict is now loaded once above, not per batch
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

                    # Update metadata for successfully parsed file (use cached mtime)
                    file_path_str = str(file_path)
                    if file_path_str in batch_mtimes:
                        metadata_dict[file_path_str] = batch_mtimes[file_path_str]
                    else:
                        logger.warning(
                            f"Skipping metadata update for deleted file: {file_path}"
                        )
                    file_results[file_path] = (len(chunks), True)
                    logger.debug(f"Prepared {len(chunks)} chunks from {file_path}")
                else:
                    # Empty file is not an error (use cached mtime)
                    file_path_str = str(file_path)
                    if file_path_str in batch_mtimes:
                        metadata_dict[file_path_str] = batch_mtimes[file_path_str]
                    else:
                        logger.warning(
                            f"Skipping metadata update for deleted file: {file_path}"
                        )
                    file_results[file_path] = (0, True)

            # Single database insertion for entire batch using two-phase architecture
            if all_chunks:
                logger.info(
                    f"Batch inserting {len(all_chunks)} chunks from {len(batch)} files"
                )

                # OPTIMIZATION: Pre-compute file hashes BEFORE timing starts
                # This avoids double file I/O (already read during parsing)
                batch_file_hashes: dict[str, str] = {}
                for file_path_str in file_to_chunks_map.keys():
                    try:
                        batch_file_hashes[file_path_str] = compute_file_hash(
                            Path(file_path_str)
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to compute hash for {file_path_str}: {e}"
                        )
                        batch_file_hashes[file_path_str] = ""

                t_start_storage = time.time()
                try:
                    # Phase 1: Store chunks to chunks.lance (fast, durable)
                    # Accumulate all chunks from batch for single write
                    t_start = time.time()

                    # Delete old chunks for files being re-indexed (skip if atomic rebuild)
                    if not self._atomic_rebuild_active:
                        for file_path_str in file_to_chunks_map.keys():
                            rel_path = str(
                                Path(file_path_str).relative_to(self.project_root)
                            )
                            await self.chunks_backend.delete_file_chunks(rel_path)

                    # OPTIMIZATION: Pre-build all chunk dicts using simple loop
                    batch_chunk_dicts = []
                    for file_path_str, (
                        start_idx,
                        end_idx,
                    ) in file_to_chunks_map.items():
                        file_hash = batch_file_hashes.get(file_path_str, "")
                        rel_path = str(
                            Path(file_path_str).relative_to(self.project_root)
                        )
                        for chunk in all_chunks[start_idx:end_idx]:
                            batch_chunk_dicts.append(
                                {
                                    "chunk_id": chunk.chunk_id,
                                    "file_path": rel_path,
                                    "file_hash": file_hash,
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
                                    "hierarchy_path": (
                                        f"{chunk.class_name}.{chunk.function_name}"
                                        if chunk.class_name and chunk.function_name
                                        else (
                                            chunk.class_name
                                            or chunk.function_name
                                            or ""
                                        )
                                    ),
                                    "docstring": chunk.docstring or "",
                                    "signature": "",
                                    "complexity": int(chunk.complexity_score),
                                    "token_count": len(chunk.content) // 5,
                                }
                            )

                    # Single write for entire batch (use raw method to skip re-normalization)
                    if batch_chunk_dicts:
                        await self.chunks_backend.add_chunks_raw(batch_chunk_dicts)
                    time_storage_total += time.time() - t_start

                    # Phase 2: Generate embeddings and store to vectors.lance
                    # This enables search functionality
                    t_start = time.time()
                    if all_chunks:
                        # Extract content for embedding generation
                        contents = [chunk.content for chunk in all_chunks]

                        # Generate embeddings using database's embedding function
                        # Use __call__() which is the universal interface for all embedding functions
                        embeddings = self.database.embedding_function(contents)

                        # Prepare chunks with vectors for vectors_backend
                        chunks_with_vectors = []
                        for chunk, embedding in zip(
                            all_chunks, embeddings, strict=True
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
                                "hierarchy_path": (
                                    f"{chunk.class_name}.{chunk.function_name}"
                                    if chunk.class_name and chunk.function_name
                                    else chunk.class_name or chunk.function_name or ""
                                ),
                            }
                            chunks_with_vectors.append(chunk_dict)

                        # Store vectors to vectors.lance
                        await self.vectors_backend.add_vectors(chunks_with_vectors)
                    time_embedding_total += time.time() - t_start

                    logger.debug(
                        f"Successfully indexed {len(all_chunks)} chunks from batch"
                    )
                    if batch_count == 1:  # Print timing for first batch as sample
                        print(
                            f"⏱️  TIMING: First batch ({len(batch)} files, {len(all_chunks)} chunks):",
                            flush=True,
                        )
                        print(
                            f"  - Storage: {time.time() - t_start_storage:.2f}s total",
                            flush=True,
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

        # TIMING: Print final summary
        t_end_index = time.time()
        total_time = t_end_index - t_start_index
        print(f"\n⏱️  TIMING SUMMARY (total={total_time:.2f}s):", flush=True)
        print(
            f"  - Parsing: {time_parsing_total:.2f}s ({time_parsing_total / total_time * 100:.1f}%)",
            flush=True,
        )
        print(
            f"  - Embedding: {time_embedding_total:.2f}s ({time_embedding_total / total_time * 100:.1f}%)",
            flush=True,
        )
        print(
            f"  - Storage: {time_storage_total:.2f}s ({time_storage_total / total_time * 100:.1f}%)",
            flush=True,
        )
        print(
            f"  - Other overhead: {total_time - time_parsing_total - time_embedding_total - time_storage_total:.2f}s\n",
            flush=True,
        )
