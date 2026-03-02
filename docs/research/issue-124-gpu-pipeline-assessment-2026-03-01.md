# Issue #124 Assessment: GPU Embedding Pipeline Starvation

**Date**: 2026-03-01
**Issue**: #124 — GPU embedding pipeline 1000x slower than theoretical: batch starvation and queue starvation on large indexes
**Branch**: main

---

## Executive Summary

Of the 4 requested changes in issue #124, **2 are fully implemented**, **1 is partially implemented with a critical gap**, and **1 is not implemented**. The recent commit series addressed the most impactful items (CUDA-aware batch sizes, queue depth), but the parallel producers feature documented in `docs/performance/performance-optimizations-summary.md` was **never wired into the live `_index_with_pipeline` code path**, and LanceDB write batching is entirely absent.

---

## Change-by-Change Assessment

---

### 1. Configurable Embedding Batch Size (--embed-batch-size, default 512+)

**Status: PARTIAL — env var exists, CLI flag for embed batch size does NOT**

#### What Is Implemented

- `MCP_VECTOR_SEARCH_BATCH_SIZE` env var is read in `embeddings.py::_detect_optimal_batch_size()` (line 187–194) and applied to `SentenceTransformer.encode()` calls.
- `MCP_VECTOR_SEARCH_FILE_BATCH_SIZE` env var controls **file** batch size (number of files per parsing batch), read in `indexer.py` lines 163–183 with a 512 default.
- `BatchEmbeddingProcessor.__init__` accepts a `batch_size` parameter (line 664–681 of `embeddings.py`) and falls back to `_detect_optimal_batch_size()` if `None`.
- The `index embed` subcommand (`cli/commands/index.py` line 2169–2204) has `--batch-size` with default 1000, but that controls **chunks per embed phase sweep**, not the embedding model's internal batch size.
- `auto_batch_size()` in `index.py` (line 41) auto-tunes file batch size (32–1024) based on RAM/CPU.

#### What Is Missing

- **No `--embed-batch-size` CLI flag on the main `index` command.** The top-level `index` command (`index.py` line 306–313) has only `--batch-size` which maps to `file_batch_size` (files per parse batch). There is no separate CLI knob for the embedding model's internal batch size.
- The embedding model's internal batch size is set in `embed_consumer()` at line 807: `embedding_batch_size = self.batch_size`, where `self.batch_size` is the **file** batch size (512 default). These two concerns are conflated.
- To expose a dedicated embed batch size flag, a `--embed-batch-size` option must be added to the `index` command in `cli/commands/index.py`, threaded through `run_indexing()`, and passed to `SemanticIndexer` (which then passes it separately to the embed consumer, distinct from file batch size).

#### Files to Change

- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/index.py` — add `--embed-batch-size` typer option; thread into `run_indexing()` call.
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/indexer.py` — add `embed_batch_size` parameter to `SemanticIndexer.__init__`; use it inside `embed_consumer()` at line 807 instead of `self.batch_size`.

---

### 2. Decoupled Chunker and Embedder Queues (async queue, embedder consumes in large batches)

**Status: PARTIAL — single producer/consumer exists but parallel producers are NOT wired in**

#### What Is Implemented

- A single-producer / single-consumer pipeline exists in `_index_with_pipeline()` (`indexer.py` lines 485–1017).
- `chunk_queue: asyncio.Queue = asyncio.Queue(maxsize=10)` — queue buffer was already increased from 2 to 10 (line 601).
- `asyncio.to_thread()` is used for embedding in `BatchEmbeddingProcessor` (confirmed in `embeddings.py` lines 744, 870).
- The consumer (`embed_consumer`, line 799) loops until it receives `None` sentinel.

#### What Is Missing — Critical Gap

The `docs/performance/performance-optimizations-summary.md` documents 4 parallel producers using `MCP_VECTOR_SEARCH_NUM_PRODUCERS` env var, but **this implementation does not exist in the current `_index_with_pipeline()` code**. The live code has exactly one `producer_task` (line 983) and one `consumer_task` (line 984). Searching for `NUM_PRODUCERS` across the entire `src/` tree returns zero matches.

The single `chunk_producer()` (line 612) processes files sequentially with a hardcoded `file_batch_size = 256` (line 610) — this hardcoded value is separate from and ignored by `self.batch_size`, meaning the queue starvation described in #124 for large indexes is only partially mitigated.

Additionally, the consumer accumulates full queue batches before embedding rather than using a dedicated accumulator loop — if file batches from the producer are smaller than the GPU-optimal embed batch size, the embedder runs under-filled batches.

#### Files to Change

- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/indexer.py`:
  - Line 610: `file_batch_size = 256` should use `self.batch_size` instead of a hardcoded 256 (or a separate env var).
  - Lines 612–797: Replace single `chunk_producer()` with N parallel producers; read `MCP_VECTOR_SEARCH_NUM_PRODUCERS` (defaulting to 4); split `files_to_process` into N ranges; use N sentinels in the consumer.
  - Lines 813–815: Consumer should accumulate across multiple `chunk_queue.get()` calls to fill a full GPU batch before calling the embedder, rather than embedding whatever single batch arrives.

---

### 3. GPU-Aware Batch Size (CUDA detection, larger defaults)

**Status: YES — fully implemented**

#### Evidence

- `_detect_device()` in `embeddings.py` (lines 104–163) detects MPS (Apple Silicon) > CUDA > CPU in priority order.
- `_detect_optimal_batch_size()` in `embeddings.py` (lines 166–264) returns:
  - Apple Silicon: 256–512 based on unified RAM (64GB+ → 512, 32GB+ → 384, <32GB → 256)
  - CUDA (NVIDIA): 512 for 8GB+ VRAM, 256 for 4–8GB, 128 for <4GB
  - CPU: 128
- `MCP_VECTOR_SEARCH_DEVICE` env var can override device selection.
- `MCP_VECTOR_SEARCH_BATCH_SIZE` env var can override the auto-detected batch size.
- Both `CodeBERTEmbeddingFunction._generate_embeddings()` (lines 579, 612) and `BatchEmbeddingProcessor.__init__()` (line 680) call `_detect_optimal_batch_size()`.
- The CUDA path in `__call__()` (line 544–546) bypasses the thread pool to avoid CUDA context issues.

**No remaining work needed for this item.**

---

### 4. Batched LanceDB Writes (fewer, larger transactions)

**Status: NO — not implemented**

#### What Currently Exists

- `VectorsBackend.add_vectors()` (`vectors_backend.py` lines 469–650) calls `self._table.add(pa_table, mode="append")` **once per embedding batch** (lines 602 and 636).
- The embedding batch size used in `embed_consumer` is `self.batch_size` (default 512). Each batch triggers one LanceDB `.add()` call.
- For a 32K-file codebase producing ~200K chunks, that is ~390 separate LanceDB write transactions at 512-chunk batches.
- Periodic compaction (`_compact_table()`) runs every 500 appends, but only on Linux (explicitly skipped on macOS to avoid SIGBUS from MPS + compaction memory conflict).

#### What Is Missing

- No write accumulator exists — each `embed_consumer` iteration immediately calls `add_vectors()` and thereby `.add()`.
- LanceDB creates one data fragment file per `.add()` call. Fewer, larger writes reduce fragment count and avoid the 100K-row compaction limit guard at line 680–685 (which skips compaction entirely for large tables, leaving fragments unmerged).
- A `write_batch_size` parameter (e.g., default 4096) with env var `MCP_VECTOR_SEARCH_WRITE_BATCH_SIZE` should buffer embedded vectors and flush when the buffer exceeds the threshold.

#### Files to Change

- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/indexer.py`:
  - `embed_consumer()` (around line 930–938): instead of calling `self.vectors_backend.add_vectors(chunks_with_vectors, ...)` immediately, accumulate into a `write_buffer` list. Flush to LanceDB when `len(write_buffer) >= write_batch_size` (e.g., 4096) or when the sentinel is received.
  - Add `write_batch_size` parameter to `SemanticIndexer.__init__()` with env var `MCP_VECTOR_SEARCH_WRITE_BATCH_SIZE`.

- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/vectors_backend.py`:
  - No structural changes needed; `add_vectors()` already accepts a list of arbitrary size. Batching is purely a caller-side concern.

---

## Summary Table

| Change | Status | Key Files |
|--------|--------|-----------|
| 1. --embed-batch-size CLI flag | PARTIAL | `cli/commands/index.py`, `core/indexer.py` |
| 2. Decoupled chunker/embedder queues (N producers) | PARTIAL | `core/indexer.py` lines 610, 612–797, 983 |
| 3. GPU-aware batch size (CUDA detection) | YES (complete) | `core/embeddings.py` lines 104–264 |
| 4. Batched LanceDB writes | NO | `core/indexer.py` embed_consumer(), `core/vectors_backend.py` |

---

## Specific Remaining Work

### High Priority (directly causes GPU starvation)

1. **Wire parallel producers**: `indexer.py` — replace single `chunk_producer` task with N parallel producers keyed off `MCP_VECTOR_SEARCH_NUM_PRODUCERS` env var (documented but not implemented in live code).
2. **Fix hardcoded file_batch_size=256**: `indexer.py` line 610 — use `self.batch_size` so the file-level batch size respects env var overrides.

### Medium Priority (reduces I/O overhead)

3. **Accumulate LanceDB writes**: `indexer.py` `embed_consumer()` — buffer embedded vectors into `write_buffer` and flush in larger batches (4096+) to reduce fragment creation.

### Low Priority (ergonomics)

4. **Add --embed-batch-size CLI flag**: Separate the embedding model's internal batch size from the file batch size in the CLI to avoid conflation.
