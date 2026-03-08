"""Embedding runner for SemanticIndexer Phase 2.

Reads pending chunks from chunks_backend, generates embeddings via the
database embedding function, writes vectors to vectors_backend.

Extracted from indexer.py to reduce complexity.
"""

import sys
import time
from collections.abc import Callable
from datetime import datetime

from loguru import logger

from .chunks_backend import ChunksBackend
from .metrics import get_metrics_tracker
from .vectors_backend import VectorsBackend


async def run_phase2_embedding(
    chunks_backend: ChunksBackend,
    vectors_backend: VectorsBackend,
    database,
    memory_monitor,
    progress_tracker,
    get_model_name_fn: Callable[[], str],
    get_project_name_fn: Callable[[str], str],
    batch_size: int = 10000,
    checkpoint_interval: int = 50000,
) -> tuple[int, int]:
    """Embed all pending chunks and write vectors to vectors_backend.

    This phase is resumable — only embeds chunks in "pending" status.

    Args:
        chunks_backend: Source of pending chunks; updated to mark progress.
        vectors_backend: Destination for embedded vectors.
        database: Provides the embedding function (multiple attribute layouts
            supported: _embedding_function, _collection._embedding_function,
            or embedding_function).
        memory_monitor: Monitors and throttles based on RAM/GPU usage.
        progress_tracker: Optional tracker for progress-bar output to stderr.
        get_model_name_fn: Zero-arg callable returning the current model name.
        get_project_name_fn: Callable(rel_path) -> subproject name string.
        batch_size: Chunks per embedding batch (adapted dynamically under
            memory pressure).
        checkpoint_interval: Chunks between checkpoint log lines.

    Returns:
        Tuple of (chunks_embedded, batches_processed).
    """
    chunks_embedded = 0
    batches_processed = 0
    batch_id = int(datetime.now().timestamp())

    logger.info(
        "Phase 2: Embedding pending chunks (GPU processing for semantic search)..."
    )

    metrics_tracker = get_metrics_tracker()
    embed_start_time = time.time()

    total_pending_chunks = await chunks_backend.count_pending_chunks()
    logger.info(f"Found {total_pending_chunks:,} pending chunks to embed")

    if total_pending_chunks == 0:
        logger.info("No pending chunks — skipping embedding phase")
        return 0, 0

    if progress_tracker:
        sys.stderr.write(
            f"  Embedding {total_pending_chunks:,} chunks"
            f" (batch size: {batch_size})...\n"
        )
        sys.stderr.flush()

    with metrics_tracker.phase("embedding") as embedding_metrics:
        while True:
            # Check memory BEFORE loading batch to prevent spikes
            usage_pct = memory_monitor.get_memory_usage_pct()

            # Proactively reduce batch size if memory is high
            while usage_pct > 0.90 and batch_size > 100:
                batch_size = batch_size // 2
                logger.warning(
                    f"Memory at {usage_pct * 100:.1f}%, proactively reducing batch to {batch_size}"
                )
                usage_pct = memory_monitor.get_memory_usage_pct()

            if usage_pct > 0.90:
                logger.warning(
                    f"Memory usage at {usage_pct * 100:.1f}%, waiting for memory to drop..."
                )
                await memory_monitor.wait_for_memory_available(target_pct=0.80)

            pending = await chunks_backend.get_pending_chunks(batch_size)
            if not pending:
                logger.info("No more pending chunks to embed")
                break

            # Check memory AFTER loading batch (when memory is at peak)
            is_ok, usage_pct, status = memory_monitor.check_memory_limit()

            if not is_ok:
                logger.warning(
                    f"Memory limit exceeded after loading batch "
                    f"({usage_pct * 100:.1f}% of {memory_monitor.max_memory_gb:.1f}GB), "
                    "waiting for memory to free up..."
                )
                await memory_monitor.wait_for_memory_available(
                    target_pct=memory_monitor.warn_threshold
                )

            # Adjust batch size based on memory pressure
            adjusted_batch_size = memory_monitor.get_adjusted_batch_size(
                batch_size, min_batch_size=100
            )
            if adjusted_batch_size != batch_size:
                logger.info(
                    f"Adjusted embedding batch size: {batch_size} -> {adjusted_batch_size} "
                    f"(memory usage: {usage_pct * 100:.1f}%)"
                )
                batch_size = adjusted_batch_size

            # Mark as processing (for crash recovery)
            chunk_ids = [c["chunk_id"] for c in pending]
            await chunks_backend.mark_chunks_processing(chunk_ids, batch_id)

            try:
                contents = [c["content"] for c in pending]

                # Check memory before expensive embedding operation
                is_ok, usage_pct, status = memory_monitor.check_memory_limit()
                if not is_ok:
                    logger.error(
                        f"Memory limit exceeded before embedding "
                        f"({usage_pct * 100:.1f}% of {memory_monitor.max_memory_gb:.1f}GB). "
                        "Reducing batch size and retrying..."
                    )
                    await chunks_backend.mark_chunks_pending(chunk_ids)
                    await memory_monitor.wait_for_memory_available(
                        target_pct=memory_monitor.warn_threshold
                    )
                    batch_size = max(100, batch_size // 4)
                    logger.info(
                        f"Reduced batch size to {batch_size} due to memory pressure"
                    )
                    continue

                # Generate embeddings — support three database embedding layouts
                vectors = None

                if hasattr(database, "_embedding_function"):
                    vectors = database._embedding_function(contents)
                elif hasattr(database, "_collection") and hasattr(
                    database._collection, "_embedding_function"
                ):
                    vectors = database._collection._embedding_function(contents)
                elif hasattr(database, "embedding_function"):
                    vectors = database.embedding_function.embed_documents(contents)
                else:
                    logger.error(
                        "Cannot access embedding function from database, "
                        "skipping batch embedding"
                    )
                    await chunks_backend.mark_chunks_complete(chunk_ids)
                    chunks_embedded += len(pending)
                    batches_processed += 1
                    continue

                if vectors is None:
                    raise ValueError("Failed to generate embeddings")

                # Build records with vectors
                chunks_with_vectors = []
                for chunk, vec in zip(pending, vectors, strict=True):
                    chunk_type = chunk.get("chunk_type", "")
                    chunk_name = chunk.get("name", "")
                    hierarchy = chunk.get("hierarchy_path", "")

                    if chunk_type in ("function", "method"):
                        fn_name = chunk_name
                        cls_name = ""
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
                        "project_name": get_project_name_fn(chunk["file_path"]),
                        "hierarchy_path": chunk["hierarchy_path"],
                    }
                    chunks_with_vectors.append(chunk_with_vec)

                model_name = get_model_name_fn()
                await vectors_backend.add_vectors(
                    chunks_with_vectors, model_version=model_name
                )

                await chunks_backend.mark_chunks_complete(chunk_ids)

                chunks_embedded += len(pending)
                batches_processed += 1

                if progress_tracker and total_pending_chunks > 0:
                    progress_tracker.progress_bar_with_eta(
                        current=chunks_embedded,
                        total=total_pending_chunks,
                        prefix="Embedding chunks",
                        start_time=embed_start_time,
                    )

                if chunks_embedded % checkpoint_interval == 0:
                    memory_monitor.log_memory_summary()
                    if not progress_tracker:
                        logger.info(f"Checkpoint: {chunks_embedded} chunks embedded")

                del pending, chunk_ids, contents, vectors, chunks_with_vectors

            except Exception as e:
                logger.error(f"Embedding batch failed: {e}")
                error_msg = f"{type(e).__name__}: {str(e)}"
                await chunks_backend.mark_chunks_error(chunk_ids, error_msg)
                continue

        embedding_metrics.item_count = chunks_embedded

    memory_monitor.log_memory_summary()
    logger.info(
        f"Phase 2 complete: {chunks_embedded} chunks embedded in {batches_processed} batches"
    )
    return chunks_embedded, batches_processed
