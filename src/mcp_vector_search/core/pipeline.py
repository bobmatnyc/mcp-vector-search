"""Pipeline parallelism for semantic indexing.

Implements a producer-consumer pattern where file parsing (Phase 1) and
chunk embedding (Phase 2) overlap for a 30-50% indexing speedup.

This module was extracted from :class:`~.indexer.SemanticIndexer` so the
pipeline logic can be tested and reasoned about in isolation.  The public
entry-point is :class:`IndexPipeline`.
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Any

from loguru import logger

from .chunk_dict import chunk_to_storage_dict
from .chunks_backend import ChunksBackend
from .context_builder import build_contextual_text
from .database import VectorDatabase
from .index_cleanup import _detect_filesystem_type
from .metrics import get_metrics_tracker
from .vectors_backend import VectorsBackend


class IndexPipeline:
    """Async producer-consumer pipeline for indexing a set of files.

    Parsing (producers) and embedding (consumer) overlap so the GPU is
    kept busy while the CPU is still chunking the next batch of files.

    Usage::

        pipeline = IndexPipeline(
            files_to_process=files_to_process,
            files_to_index=files_to_index,
            chunks_backend=self.chunks_backend,
            vectors_backend=self.vectors_backend,
            database=self.database,
            chunk_processor=self.chunk_processor,
            memory_monitor=self.memory_monitor,
            progress_tracker=self.progress_tracker,
            project_root=self.project_root,
            mcp_dir=self._mcp_dir,
            indexer_config=self._indexer_config,
            metadata=self.metadata,
            batch_size=self.batch_size,
            embed_batch_size=self.embed_batch_size,
            use_multiprocessing=self.use_multiprocessing,
            atomic_rebuild_active=self._atomic_rebuild_active,
            get_embedding_model_name=self.get_embedding_model_name,
        )
        files_indexed, chunks_created, chunks_embedded = await pipeline.run()
    """

    def __init__(
        self,
        *,
        files_to_process: list[tuple[Path, str, str]],
        files_to_index: list[Path],
        chunks_backend: ChunksBackend,
        vectors_backend: VectorsBackend,
        database: VectorDatabase,
        chunk_processor: Any,
        memory_monitor: Any,
        progress_tracker: Any,
        project_root: Path,
        mcp_dir: Path,
        indexer_config: Any,
        metadata: Any,
        batch_size: int,
        embed_batch_size: int,
        use_multiprocessing: bool,
        atomic_rebuild_active: bool,
        get_embedding_model_name: Any,
    ) -> None:
        # --- Dependencies ---
        self.files_to_process = files_to_process
        self.files_to_index = files_to_index
        self.chunks_backend = chunks_backend
        self.vectors_backend = vectors_backend
        self.database = database
        self.chunk_processor = chunk_processor
        self.memory_monitor = memory_monitor
        self.progress_tracker = progress_tracker
        self.project_root = project_root
        self._mcp_dir = mcp_dir
        self._indexer_config = indexer_config
        self.metadata = metadata
        self.batch_size = batch_size
        self.embed_batch_size = embed_batch_size
        self.use_multiprocessing = use_multiprocessing
        self._atomic_rebuild_active = atomic_rebuild_active
        self.get_embedding_model_name = get_embedding_model_name

        # --- Shared mutable state (were nonlocal variables in the original closures) ---
        self.files_indexed: int = 0
        self.chunks_created: int = 0
        self.chunks_embedded: int = 0

        # --- Pipeline coordination primitives (set up in run()) ---
        self.pipeline_cancel: asyncio.Event | None = None
        self.consumer_task: asyncio.Task | None = None
        self.chunk_queue: asyncio.Queue | None = None

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    async def run(self) -> tuple[int, int, int]:
        """Execute the pipeline and return (files_indexed, chunks_created, chunks_embedded)."""
        files_to_process = self.files_to_process
        files_to_index = self.files_to_index

        num_producers = self._indexer_config.num_producers

        phase_start_time = time.time()

        # Clamp num_producers to the actual number of files (avoid empty producers)
        effective_num_producers = max(1, min(num_producers, len(files_to_process)))
        if effective_num_producers < num_producers:
            logger.debug(
                f"Clamping producers from {num_producers} to {effective_num_producers} "
                f"(only {len(files_to_process)} files to process)"
            )

        # FIX 3 (HIGH): Auto-scale queue depth to num_producers * 4.
        queue_maxsize = self._indexer_config.queue_depth or (
            effective_num_producers * 4
        )
        self.chunk_queue = asyncio.Queue(maxsize=queue_maxsize)
        self._effective_num_producers = effective_num_producers

        # FIX 1 (CRITICAL RELIABILITY): Cancellation event so producers can detect
        # that the consumer has crashed and stop blocking on chunk_queue.put().
        self.pipeline_cancel = asyncio.Event()

        # FIX 6 (HIGH): LPT scheduling — sort files by descending size so large
        # files distribute evenly across producers.
        try:
            files_to_process.sort(
                key=lambda x: x[0].stat().st_size if x[0].exists() else 0,
                reverse=True,
            )
        except OSError:
            pass  # Non-fatal: proceed with original order

        # Split files_to_process into num_producers roughly equal slices (stride-based)
        producer_slices = [
            files_to_process[i::effective_num_producers]
            for i in range(effective_num_producers)
        ]
        logger.info(
            f"Starting pipeline: {effective_num_producers} producer(s) + 1 consumer "
            f"(Phase 1 parsing and Phase 2 embedding will overlap)"
        )

        # Consumer must be created FIRST so producer tasks can reference it via self.consumer_task
        self.consumer_task = asyncio.create_task(self._embed_consumer())

        # Create one producer task per slice
        producer_tasks = [
            asyncio.create_task(
                self._chunk_producer(
                    i, producer_slices[i], effective_num_producers, phase_start_time
                )
            )
            for i in range(effective_num_producers)
        ]

        # FIX 5 (HIGH): Use return_exceptions=True so a second simultaneous
        # failure is not swallowed when the first exception is raised.
        all_tasks = producer_tasks + [self.consumer_task]
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        exceptions = [
            r
            for r in results
            if isinstance(r, BaseException)
            and not isinstance(r, asyncio.CancelledError)
        ]
        if exceptions:
            logger.error(f"Pipeline task failed: {exceptions[0]}")
            for exc in exceptions[1:]:
                logger.error("Additional pipeline task failure: %s", exc)

            for task in all_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            raise exceptions[0]

        # Update metadata for backward compatibility
        if self.files_indexed > 0:
            metadata_dict = self.metadata.load()
            for file_path in files_to_index:
                try:
                    metadata_dict[str(file_path)] = os.path.getmtime(file_path)
                except OSError:
                    pass
            self.metadata.save(metadata_dict)

        # CLEANUP: Shutdown persistent ProcessPoolExecutor after indexing completes
        self.chunk_processor.close()

        return self.files_indexed, self.chunks_created, self.chunks_embedded

    # ------------------------------------------------------------------
    # Producer
    # ------------------------------------------------------------------

    async def _chunk_producer(
        self,
        producer_id: int,
        file_slice: list[tuple[Path, str, str]],
        effective_num_producers: int,
        phase_start_time: float,
    ) -> None:
        """Parse and chunk a slice of files, put batches on the queue.

        Args:
            producer_id: Index of this producer (0-based, for logging)
            file_slice: Subset of files_to_process this producer owns
            effective_num_producers: Total number of active producers (for sentinel logic)
            phase_start_time: Wall-clock start time of Phase 1 (for ETA display)
        """
        metrics_tracker = get_metrics_tracker()
        file_batch_size = self.batch_size

        with metrics_tracker.phase("parsing") as parsing_metrics:
            for batch_start in range(0, len(file_slice), file_batch_size):
                # Check if pipeline was cancelled (consumer crashed)
                if self.pipeline_cancel.is_set():
                    logger.warning(
                        "Producer %d: pipeline cancelled, stopping", producer_id
                    )
                    return

                # Check if consumer is still alive
                if self.consumer_task.done():
                    exc = self.consumer_task.exception()
                    logger.error(
                        f"Consumer died unexpectedly: {exc}. Stopping producer {producer_id}."
                    )
                    return

                batch_end = min(batch_start + file_batch_size, len(file_slice))
                batch = file_slice[batch_start:batch_end]

                batch_chunks: list[dict] = []
                batch_files_processed = 0

                # FIX 2 (CRITICAL PERFORMANCE): Use process pool for batch parsing.
                batch_paths = [fp for fp, _rel, _hash in batch]
                rel_paths = {fp: rel for fp, rel, _hash in batch}
                file_hashes = {fp: fh for fp, _rel, fh in batch}

                if self.use_multiprocessing and len(batch_paths) > 1:
                    parse_results = await self.chunk_processor.parse_files_multiprocess(
                        batch_paths
                    )
                else:
                    parse_results = []
                    for fp in batch_paths:
                        try:
                            chunks = await self.chunk_processor.parse_file(fp)
                            parse_results.append((fp, chunks, None))
                        except Exception as exc:
                            parse_results.append((fp, [], exc))

                # FIX 8 (LOW): Throttle memory checks to every 10 files.
                for idx, (file_path, chunks, parse_error) in enumerate(parse_results):
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
                        logger.error(f"Failed to chunk file {file_path}: {parse_error}")
                        continue

                    try:
                        if not chunks:
                            logger.trace(f"No chunks extracted from {file_path}")
                            continue

                        # Build hierarchical relationships (CPU-bound, run in thread pool)
                        chunks_with_hierarchy = await asyncio.to_thread(
                            self.chunk_processor.build_chunk_hierarchy, chunks
                        )

                        chunk_dicts = [
                            chunk_to_storage_dict(chunk, rel_path)
                            for chunk in chunks_with_hierarchy
                        ]

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
                if batch_chunks:
                    count = await self.chunks_backend.add_chunks_batch(batch_chunks)
                    self.chunks_created += count

                self.files_indexed += batch_files_processed

                # Put batch on queue for embedding.
                # FIX 1 (CRITICAL RELIABILITY): Race put() against pipeline_cancel.
                if batch_chunks:
                    if self.progress_tracker:
                        self.progress_tracker.progress_bar_with_eta(
                            current=self.files_indexed,
                            total=len(self.files_to_process),
                            prefix="Parsing files",
                            start_time=phase_start_time,
                        )
                    else:
                        logger.info(
                            f"Phase 1 progress: {self.files_indexed}/{len(self.files_to_process)} files, "
                            f"{self.chunks_created} chunks | Queuing {len(batch_chunks)} chunks for embedding"
                        )
                    put_task = asyncio.ensure_future(
                        self.chunk_queue.put(
                            {
                                "chunks": batch_chunks,
                                "batch_size": len(batch_chunks),
                            }
                        )
                    )
                    cancel_task = asyncio.ensure_future(self.pipeline_cancel.wait())
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
                    if self.pipeline_cancel.is_set():
                        logger.warning(
                            "Producer %d: pipeline cancelled, stopping", producer_id
                        )
                        return
                    # Yield to event loop to let consumer process the batch
                    await asyncio.sleep(0)

            # Update metrics for this producer's share
            parsing_metrics.item_count = self.files_indexed

        # Track chunking separately (each producer updates the shared metric)
        with metrics_tracker.phase("chunking") as chunking_metrics:
            chunking_metrics.item_count = self.chunks_created

        self.memory_monitor.log_memory_summary()
        logger.info(
            f"Producer {producer_id} complete: processed slice of {len(file_slice)} files"
        )

        # Each producer sends its own sentinel so the consumer knows when ALL are done
        await self.chunk_queue.put(None)

    # ------------------------------------------------------------------
    # Consumer
    # ------------------------------------------------------------------

    async def _embed_consumer(self) -> None:
        """Take chunks from the queue, embed, and store to vectors.lance."""
        metrics_tracker = get_metrics_tracker()

        logger.info(
            "Phase 2: Embedding pending chunks (GPU processing for semantic search)..."
        )

        embedding_batch_size = self.embed_batch_size
        consecutive_errors = 0
        embed_start_time = time.time()

        # Batched LanceDB writes: buffer chunks to reduce fragment count
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

        # Set by run() before this task is created; counts how many producer sentinels to expect.
        effective_num_producers = self._effective_num_producers
        sentinels_received = 0

        with metrics_tracker.phase("embedding") as embedding_metrics:
            try:
                while sentinels_received < effective_num_producers:
                    batch_data = await self.chunk_queue.get()

                    if batch_data is None:
                        sentinels_received += 1
                        logger.debug(
                            f"Received sentinel {sentinels_received}/{effective_num_producers} from producers"
                        )
                        continue

                    chunks = batch_data["chunks"]

                    if not chunks:
                        continue

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
                                f"  Memory at {usage_pct * 100:.1f}%, proactively reducing batch to {embedding_batch_size}"
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
                                    timeout=120.0,
                                )
                            except TimeoutError:
                                logger.warning(
                                    "Memory wait timed out after 120s, proceeding anyway"
                                )

                        try:
                            # Generate embeddings using context-enriched text.
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
                                        timeout=120.0,
                                    )
                                except TimeoutError:
                                    logger.warning(
                                        "Memory wait timed out after 120s, proceeding anyway"
                                    )

                            # FIX 4 (HIGH): Wrap the synchronous embedding call in
                            # asyncio.to_thread so the event loop thread is not blocked.
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
                            self.chunks_embedded += len(emb_batch)

                            # Flush write buffer when it reaches write_batch_size
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

                            # Periodic GC to release Arrow/LanceDB buffers
                            if (
                                self.chunks_embedded % 10_000 == 0
                                and self.chunks_embedded > 0
                            ):
                                import gc

                                gc.collect()
                                logger.debug(
                                    "GC collect at %d embedded chunks (Arrow buffer cleanup)",
                                    self.chunks_embedded,
                                )

                            # Show progress bar if tracker is available
                            if self.progress_tracker and self.chunks_created > 0:
                                self.progress_tracker.progress_bar_with_eta(
                                    current=self.chunks_embedded,
                                    total=self.chunks_created,
                                    prefix="Embedding chunks",
                                    start_time=embed_start_time,
                                )
                            else:
                                logger.info(
                                    f"Phase 2 progress: {self.chunks_embedded} chunks embedded"
                                )

                        except Exception as e:
                            consecutive_errors += 1
                            logger.error(
                                f"Failed to embed batch (consecutive errors: {consecutive_errors}): {e}"
                            )

                            if consecutive_errors >= 3:
                                logger.error(
                                    f"Too many consecutive embedding errors ({consecutive_errors}), "
                                    "stopping consumer to prevent silent failure"
                                )
                                raise

                            continue

            finally:
                # FIX 1 (CRITICAL RELIABILITY): Signal producers that the
                # consumer is exiting so they stop blocking on chunk_queue.put().
                self.pipeline_cancel.set()

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
                        # FIX 7 (MEDIUM): Mark lost chunk IDs for re-embedding on next run.
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
            embedding_metrics.item_count = self.chunks_embedded

        logger.info(f"Phase 2 complete: {self.chunks_embedded} chunks embedded")
