"""Chunking runner for SemanticIndexer Phase 1.

Iterates over files, calls parsers to produce chunks, stores results to
chunks_backend, and tracks progress.

Extracted from indexer.py to reduce complexity.
"""

import asyncio
import gc
import sys
import time
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from .chunk_dict import chunk_to_storage_dict
from .chunks_backend import ChunksBackend, compute_file_hash
from .exceptions import DatabaseError
from .metrics import get_metrics_tracker
from .vectors_backend import VectorsBackend


async def _flush_pending_chunks(
    pending_chunks: list[dict],
    chunks_backend: ChunksBackend,
    pending_files_count_ref: list[int],
) -> int:
    """Write accumulated pending_chunks to LanceDB; return count written.

    Args:
        pending_chunks: Mutable list of chunk dicts to flush (cleared in-place).
        chunks_backend: Destination backend for chunk storage.
        pending_files_count_ref: Single-element list used as a mutable int ref;
            reset to 0 after flushing.

    Returns:
        Number of chunks written.
    """
    if not pending_chunks:
        return 0
    count = await chunks_backend.add_chunks_batch(pending_chunks)
    pending_chunks.clear()
    pending_files_count_ref[0] = 0
    return count


async def run_phase1_chunking(
    files: list[Path],
    chunks_backend: ChunksBackend,
    vectors_backend: VectorsBackend,
    project_root: Path,
    chunk_processor,
    memory_monitor,
    progress_tracker,
    atomic_rebuild_active: bool,
    detect_file_moves_fn: Callable[
        [list[Path], dict[str, str]],
        tuple[list[tuple[str, str, str]], set[str], dict[Path, str]],
    ],
    force: bool = False,
    file_hash_cache: dict[Path, str] | None = None,
) -> tuple[int, int]:
    """Phase 1: Parse and chunk files, store to chunks.lance.

    This phase is fast and durable — no expensive embedding generation.
    Incremental updates are supported via file_hash change detection.

    Args:
        files: Files to process.
        chunks_backend: Destination backend for chunk storage.
        vectors_backend: Backend used to update file paths on detected moves.
        project_root: Root of the project being indexed.
        chunk_processor: ChunkProcessor instance for parsing and hierarchy.
        memory_monitor: MemoryMonitor for throttling under memory pressure.
        progress_tracker: Optional tracker for progress-bar output to stderr.
        atomic_rebuild_active: When True, skip batch-deletion of old chunks
            (the rebuild manager handles table replacement instead).
        detect_file_moves_fn: Callable matching the signature of
            ``file_move_detector.detect_file_moves`` (passed explicitly to
            avoid importing project_root indirection from ``self``).
        force: If True, re-chunk every file even if unchanged.
        file_hash_cache: Pre-computed file hashes from the caller.  When
            provided the caller has ALREADY performed change detection and
            ``files`` contains only files that need processing — skip the
            redundant ``get_all_indexed_file_hashes`` / detect_file_moves /
            change-detection loop and treat every entry in ``files`` as
            requiring processing.  When None (default) the full change
            detection is run (backward-compatible standalone behaviour).

    Returns:
        Tuple of (files_processed, chunks_created).
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
        files_to_delete: list[str] = []
        files_to_process: list[tuple[Path, str, str]] = []

        # When file_hash_cache is provided the caller already ran change
        # detection — every file in `files` needs processing.  Skip the
        # redundant get_all_indexed_file_hashes / detect_file_moves /
        # per-file hash loop to avoid duplicate "Scanning files" bars and
        # wasted DB round-trips.
        caller_prefiltered = file_hash_cache is not None

        if caller_prefiltered:
            # Caller has already filtered; build files_to_process directly.
            _fhc: dict[Path, str] = file_hash_cache  # type: ignore[assignment]
            for file_path in files:
                try:
                    file_hash = _fhc.get(file_path) or compute_file_hash(file_path)
                    rel_path = str(file_path.relative_to(project_root))
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
                indexed_file_hashes = await chunks_backend.get_all_indexed_file_hashes()
                logger.info(
                    f"Loaded {len(indexed_file_hashes)} indexed files for change detection"
                )

            # Detect file moves/renames — update metadata instead of re-chunking
            if not force:
                detected_moves, _moved_old_paths, _local_fhc = detect_file_moves_fn(
                    files, indexed_file_hashes
                )
                if detected_moves:
                    logger.info(
                        f"Detected {len(detected_moves)} file move(s), updating metadata..."
                    )
                    for old_path, new_path, _file_hash in detected_moves:
                        chunks_updated = await chunks_backend.update_file_path(
                            old_path, new_path
                        )
                        vectors_updated = await vectors_backend.update_file_path(
                            old_path, new_path
                        )
                        logger.info(
                            f"  Moved: {old_path} -> {new_path} "
                            f"({chunks_updated} chunks, {vectors_updated} vectors)"
                        )
                    # Reload so change detection sees updated paths
                    indexed_file_hashes = (
                        await chunks_backend.get_all_indexed_file_hashes()
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
                    if idx > 0 and idx % 1000 == 0 and not progress_tracker:
                        logger.info(f"Computing file hashes: {idx}/{len(files)}")

                    # Skip hash computation when force=True (all files will be reindexed)
                    if force:
                        file_hash = ""  # Empty hash when forcing full reindex
                    else:
                        # Reuse hash from move-detection cache to avoid re-hashing
                        file_hash = _local_fhc.get(file_path) or compute_file_hash(
                            file_path
                        )

                    rel_path = str(file_path.relative_to(project_root))

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
                    f"File hash computation complete: {len(files_to_process)} files changed, "
                    f"{len(files) - len(files_to_process)} unchanged"
                )

        # Batch delete old chunks for changed files before inserting new ones.
        # This prevents duplicate chunks from accumulating across re-index runs.
        # Note: LanceDB delete() is metadata-only (deletion vectors) and does NOT
        # trigger compact_files(). The SIGBUS workaround for macOS applies only to
        # compaction, which is separately guarded in _compact_table().
        if not atomic_rebuild_active and files_to_delete:
            deleted_count = await chunks_backend.delete_files_batch(files_to_delete)
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
        # Buffer accumulates chunk dicts with file_hash pre-injected.
        pending_chunks: list[dict] = []
        # Single-element list used as a mutable int reference passed to helper.
        pending_files_count_ref = [0]

        for _idx, (file_path, rel_path, file_hash) in enumerate(files_to_process):
            # Check memory before processing each file (CRITICAL: not every 100!)
            is_ok, usage_pct, status = memory_monitor.check_memory_limit()
            if not is_ok:
                logger.warning(
                    f"Memory limit exceeded during chunking "
                    f"({usage_pct * 100:.1f}% of {memory_monitor.max_memory_gb:.1f}GB), "
                    "waiting for memory to free up..."
                )
                # Flush before waiting so we don't hold large buffers in memory
                await _flush_pending_chunks(
                    pending_chunks, chunks_backend, pending_files_count_ref
                )
                await memory_monitor.wait_for_memory_available(
                    target_pct=memory_monitor.warn_threshold
                )

            try:
                # Parse file (runs in thread pool to avoid blocking event loop)
                chunks = await chunk_processor.parse_file(file_path)

                if not chunks:
                    logger.debug(f"No chunks extracted from {file_path}")
                    continue

                # Build hierarchical relationships (CPU-bound, run in thread pool)
                chunks_with_hierarchy = await asyncio.to_thread(
                    chunk_processor.build_chunk_hierarchy, chunks
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
                    pending_files_count_ref[0] += 1
                    logger.debug(f"Parsed {file_chunk_count} chunks from {rel_path}")

                    # Flush when write-batch is full
                    if pending_files_count_ref[0] >= write_batch_files:
                        try:
                            await _flush_pending_chunks(
                                pending_chunks, chunks_backend, pending_files_count_ref
                            )
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
                    if progress_tracker:
                        progress_tracker.progress_bar_with_eta(
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
            if progress_tracker:
                sys.stderr.write(
                    f"  Saving final batch ({len(pending_chunks):,} chunks)...\n"
                )
                sys.stderr.flush()
            try:
                await _flush_pending_chunks(
                    pending_chunks, chunks_backend, pending_files_count_ref
                )
            except DatabaseError as db_err:
                logger.error(f"Failed to flush final chunk batch: {db_err}")

        if backend_fatal_error is not None:
            logger.error(f"Phase 1 aborted due to backend error: {backend_fatal_error}")

        # Update metrics
        parsing_metrics.item_count = files_processed

    # Track chunking separately
    with metrics_tracker.phase("chunking") as chunking_metrics:
        chunking_metrics.item_count = chunks_created
        # Note: Duration will be minimal since actual work was done in parsing phase
        # This is just for tracking the chunk count

    # Log memory summary after Phase 1
    memory_monitor.log_memory_summary()
    logger.info(
        f"✓ Phase 1 complete: {files_processed} files processed, {chunks_created} chunks created"
    )
    return files_processed, chunks_created
