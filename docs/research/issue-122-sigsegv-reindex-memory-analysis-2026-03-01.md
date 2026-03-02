# Issue #122: SIGSEGV During Reindex on GPU Instance (g4dn.xlarge)

**Date:** 2026-03-01
**Investigator:** Research Agent
**Files Analyzed:**
- `/src/mcp_vector_search/core/indexer.py`
- `/src/mcp_vector_search/core/chunk_processor.py`
- `/src/mcp_vector_search/core/embeddings.py`
- `/src/mcp_vector_search/core/memory_monitor.py`
- `/src/mcp_vector_search/core/resource_manager.py`
- `/src/mcp_vector_search/core/chunks_backend.py`
- `/src/mcp_vector_search/cli/commands/index.py`

---

## Incident Summary

- **Crash:** SIGSEGV after ~2h wall clock, only ~2m 12s CPU time used
- **RAM:** 13.2 GB / 14 GB container limit consumed at time of crash
- **Lance directory:** timestamps did NOT update during a 17h run — writes had not yet started
- **Scale:** 232,047 chunks / ~17 GB on disk / 151 repositories
- **Instance:** AWS g4dn.xlarge (1x T4 GPU, 16 GB RAM)

---

## Root Cause Analysis

### Primary Cause: Memory Monitor Cap Mismatch (Critical Bug)

**File:** `src/mcp_vector_search/core/memory_monitor.py`, lines 50-52

```python
# Default to 25GB if not specified
if max_memory_gb is None:
    max_memory_gb = 25.0
```

The `MemoryMonitor` defaults to a 25 GB cap, but the g4dn.xlarge container has a **14 GB RAM limit**. This means:

- The monitor measures `rss / 25GB` as its usage percentage
- At 13.2 GB RSS, the monitor reports `13.2 / 25 = 52.8%` — far below its 80% warn threshold and 90% critical threshold
- The monitor sees no pressure and takes no action while the process is actually at **94% of the real container limit**
- The native crash (SIGSEGV) occurs when PyTorch/LanceDB/Arrow native code tries to allocate memory and the OS kills the process with SIGKILL or returns OOM to native code, which manifests as SIGSEGV in the native library

**This is the primary bug.** The memory safety net has the wrong cap and is entirely ineffective for a 14 GB container.

### Secondary Cause: Phase 1 Completes Before Phase 2 Writes Begin

**Evidence:** Lance directory timestamps did not update during a 17h run.

**File:** `src/mcp_vector_search/core/indexer.py`, lines 598-984

The pipeline uses a producer-consumer architecture:
- **Producer** (`chunk_producer`): Parses all 232,047 chunks from 151 repos, stores them to `chunks.lance`, puts batches on the queue
- **Consumer** (`embed_consumer`): Reads from the queue, generates embeddings, writes to `vectors.lance`

The queue has `maxsize=10` entries (line 601), and each entry contains a full file batch of 256 files' worth of chunks (line 610: `file_batch_size = 256`). With 232K chunks across 151 repos, the producer can accumulate **all chunk data in memory** before the consumer makes meaningful progress on embedding:

1. Producer parses 256 files → stores to `chunks.lance` → puts on queue → immediately starts next batch
2. Queue fills to 10 items (each ~2,560+ chunks of content)
3. Producer blocks waiting for queue space
4. Meanwhile, all 232K chunk dictionaries are also held in memory across the queue entries

The chunk dictionaries are large: each contains `content`, `docstring`, `hierarchy_path`, `calls`, `imports` etc. At 232K chunks averaging ~500 bytes of content each, that is ~116 MB minimum for raw content strings alone — likely several GB with all metadata.

**The Lance write timestamps not updating during 17h** strongly suggests the crash happened in Phase 1 (producer), before Phase 2 (consumer) wrote any vectors. This aligns with memory exhaustion during the chunking phase.

### Tertiary Cause: No `torch.cuda.empty_cache()` After Embedding Batches

**File:** `src/mcp_vector_search/core/indexer.py`, lines 860-942 (pipeline consumer)

```python
# Generate embeddings
vectors = self.database._embedding_function(contents)
# ... add to vectors table ...
chunks_embedded += len(emb_batch)
# (no torch.cuda.empty_cache() call)
```

In `embeddings.py`, `_generate_embeddings()` uses `model.encode()` which allocates GPU tensors. On CUDA, the PyTorch CUDA allocator holds memory in a pool and does not release it back to the OS between batches. Over a 2-hour run with 232K chunks at batch size 512, CUDA can accumulate a large fragmented allocation pool in VRAM. When VRAM is exhausted, PyTorch may fall back to unified memory (pinned host memory), which directly competes with the 14 GB RAM limit.

The T4 GPU has 16 GB VRAM. With sentence-transformers all-MiniLM-L6-v2, the model itself is ~90 MB. Batch size 512 at 768-dim float32 requires ~1.5 MB per batch. But PyTorch's CUDA memory allocator retains freed blocks, so VRAM grows monotonically unless `torch.cuda.empty_cache()` is called.

### Quaternary Cause: EFS + LanceDB Compaction Under Memory Pressure

**File:** `src/mcp_vector_search/core/chunks_backend.py`, line 31

```python
# DataFusion (used by LanceDB) has a recursive descent parser that stack-overflows
# when processing thousands of OR clauses. Using IN with bounded batch size keeps
# the parse tree shallow and prevents SEGV (signal 139) on Linux.
DELETE_BATCH_LIMIT = 500
```

The codebase already documents a LanceDB SEGV issue with DataFusion. Under memory pressure, LanceDB's Arrow/Rust native code can SEGV when internal allocations fail or when EFS (network filesystem) writes experience latency that causes buffer bloat in the Rust async runtime.

### Issue with `auto_batch_size()` on GPU Instances

**File:** `src/mcp_vector_search/cli/commands/index.py`, lines 41-79

```python
def auto_batch_size() -> int:
    cpu_count = os.cpu_count() or 4   # g4dn.xlarge: 4 vCPU
    available_ram_gb = _get_available_ram_gb()
    base_batch = cpu_count * 16        # = 64
    # With 14GB RAM available at startup: batch = 64, rounded to 64
```

`auto_batch_size()` calculates the **file** batch size (files per parsing batch), not the embedding batch size. It reads available RAM at startup (when the process is fresh), so it sees the full ~14 GB and returns a relatively large file batch. It does not account for the memory consumed by the embedding model itself (~400 MB for MiniLM loaded onto GPU), LanceDB write buffers, or Arrow/PyArrow in-process data.

The **embedding** batch size (`MCP_VECTOR_SEARCH_BATCH_SIZE`) defaults to 512 for GPUs with 8+ GB VRAM (in `embeddings.py`, line 241-244). This is appropriate for GPU throughput but, when combined with `max_concurrent=16` parallel embedding threads (`embed_batches_parallel`, line 727), can create 16 × 512 = 8,192 simultaneous chunks in embedding tensors at peak.

---

## Specific Code Locations

| # | File | Lines | Issue |
|---|------|--------|-------|
| 1 | `core/memory_monitor.py` | 50-52 | Default 25 GB cap — wrong for 14 GB container |
| 2 | `core/memory_monitor.py` | 37-48 | Does not auto-detect actual system/container memory limit |
| 3 | `core/indexer.py` | 288-289 | `MemoryMonitor()` created with no cap argument — always uses default |
| 4 | `core/indexer.py` | 601 | `asyncio.Queue(maxsize=10)` — can buffer 10 × 256 files = 2,560 chunk dicts |
| 5 | `core/indexer.py` | 860-942 | No `torch.cuda.empty_cache()` after embedding batches |
| 6 | `core/indexer.py` | 1412-1416 | `gc.collect()` only every 1,000 files in Phase 1 sequential path — not in pipeline |
| 7 | `core/embeddings.py` | 722-727 | `max_concurrent=16` default — 16 concurrent embedding threads |
| 8 | `core/embeddings.py` | 544-546 | CUDA path runs directly on main thread (no thread-pool timeout) |

---

## Recommended Fixes

### Fix 1: Auto-detect Container Memory Limit (Critical)

**File:** `core/memory_monitor.py`

Replace the hardcoded 25 GB default with auto-detection of the real container memory limit:

```python
def _detect_container_memory_limit_gb() -> float:
    """Detect actual container/cgroup memory limit.

    Reads from cgroup v2 memory limit first, then cgroup v1,
    then falls back to total system RAM via psutil.
    """
    # cgroup v2 (modern Docker/Kubernetes)
    cgroup_v2 = Path("/sys/fs/cgroup/memory.max")
    if cgroup_v2.exists():
        try:
            raw = cgroup_v2.read_text().strip()
            if raw != "max":  # "max" means no cgroup limit
                limit_bytes = int(raw)
                return limit_bytes / (1024 ** 3)
        except (ValueError, OSError):
            pass

    # cgroup v1 (older Docker)
    cgroup_v1 = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if cgroup_v1.exists():
        try:
            limit_bytes = int(cgroup_v1.read_text().strip())
            # 9223372036854771712 == no limit (2^63 - 4096) on v1
            if limit_bytes < (1 << 62):
                return limit_bytes / (1024 ** 3)
        except (ValueError, OSError):
            pass

    # Fallback: total physical RAM
    return psutil.virtual_memory().total / (1024 ** 3)


class MemoryMonitor:
    def __init__(self, max_memory_gb=None, ...):
        if max_memory_gb is None:
            max_memory_gb = _detect_container_memory_limit_gb()
            logger.info(f"Auto-detected memory limit: {max_memory_gb:.1f}GB")
        # Leave a safety margin: use 85% of detected limit
        # This reserves headroom for native library allocations (PyTorch, Arrow)
        # that may not be tracked in Python RSS
        max_memory_gb = max_memory_gb * 0.85
        ...
```

**Why 85%:** PyTorch CUDA unified memory, Arrow off-heap buffers, and LanceDB Rust allocations are not always counted in the Python process RSS. The 85% margin prevents native OOM from within headroom that psutil cannot see.

### Fix 2: Add `torch.cuda.empty_cache()` After Embedding Batches (Important)

**File:** `core/indexer.py` (embed_consumer, around line 938)
**File:** `core/embeddings.py` (_generate_embeddings, around line 620)

In the embedding generation path for CUDA:

```python
def _generate_embeddings(self, input: list[str]) -> list[list[float]]:
    # ... (existing code) ...
    embeddings = self.model.encode(input, ...)
    result = embeddings.tolist()
    del embeddings  # free the numpy array
    if self.device == "cuda":
        import torch
        torch.cuda.empty_cache()  # Return CUDA cache to pool
    return result
```

In the consumer after writing vectors:

```python
# After: await self.vectors_backend.add_vectors(...)
chunks_embedded += len(emb_batch)

# Explicit CUDA cache flush every N batches to prevent VRAM fragmentation
if self.device == "cuda" and chunks_embedded % 5000 == 0:
    import torch
    torch.cuda.empty_cache()
    logger.debug("Flushed CUDA cache (periodic)")
```

### Fix 3: Reduce Queue Buffer Size Under Memory Pressure (Important)

**File:** `core/indexer.py`, line 601

The queue `maxsize=10` allows up to 10 × 256-file batches to pile up in memory simultaneously. On a 14 GB machine with 232K chunks, this is a significant fraction of available RAM.

```python
# Dynamically size queue based on available memory
# Each queue entry ~ file_batch_size * avg_chunk_size_bytes
# Default 256 files * ~4KB avg chunk = ~1MB per queue entry
# With 10 entries = ~10MB (small), but also includes chunk metadata
# The real cost is the chunk dicts: 256 files * ~10 chunks * ~2KB = ~5MB per entry

# Reduce to 4 entries to limit peak memory from the queue
chunk_queue: asyncio.Queue = asyncio.Queue(maxsize=4)
```

Alternatively, make `maxsize` configurable via environment variable:

```python
queue_maxsize = int(os.environ.get("MCP_VECTOR_SEARCH_QUEUE_DEPTH", "4"))
chunk_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_maxsize)
```

### Fix 4: Add Explicit GC in Pipeline Consumer (Moderate)

**File:** `core/indexer.py`, embed_consumer (around line 938)

The Phase 1 sequential path already does `gc.collect()` every 1,000 files, but the pipeline consumer has no equivalent. Add:

```python
# After storing vectors and clearing chunk references:
del emb_batch, contents, vectors, chunks_with_vectors

# Periodic GC to release Arrow buffers (same reasoning as Phase 1)
if chunks_embedded % 10000 == 0:
    import gc
    gc.collect()
    logger.debug(f"GC collect after {chunks_embedded} embeddings (Arrow buffer cleanup)")
```

### Fix 5: Memory-Aware OOM Signal Handler (Moderate)

**File:** `core/indexer.py` or CLI entry point

Instead of crashing with SIGSEGV (which happens inside native code where Python cannot intercept), install a `SIGUSR1` or periodic memory watchdog that raises a graceful Python exception before the native crash occurs:

```python
import signal, os, psutil, threading

def _install_memory_watchdog(limit_gb: float, check_interval_sec: float = 10.0):
    """Background thread that raises MemoryError before OOM SIGSEGV."""
    limit_bytes = limit_gb * (1024 ** 3)
    process = psutil.Process()

    def _watch():
        while True:
            try:
                rss = process.memory_info().rss
                if rss > limit_bytes * 0.95:  # 95% of limit
                    logger.critical(
                        f"MEMORY WATCHDOG: RSS {rss / (1024**3):.2f}GB >= "
                        f"95% of {limit_gb:.1f}GB limit. "
                        "Triggering graceful shutdown to prevent SIGSEGV."
                    )
                    os.kill(os.getpid(), signal.SIGTERM)  # graceful shutdown
                    break
            except Exception:
                pass
            time.sleep(check_interval_sec)

    t = threading.Thread(target=_watch, daemon=True, name="memory-watchdog")
    t.start()
    return t
```

This gives the Python OOM handler a chance to checkpoint progress before native code crashes.

### Fix 6: Writes Should Start Before Chunking Completes (Architecture)

**File:** `core/indexer.py`, `_index_with_pipeline()`

The Lance directory timestamp issue confirms that Phase 2 (vector writes) had not started by the time of the crash, despite the pipeline design. This happens because:

1. Phase 1 (producer) is sequential across all 151 repos
2. Phase 1 takes the full 2h+ just to parse and chunk files
3. Phase 2 (consumer) never gets meaningful work before the crash

**The pipeline architecture is correct in principle** — Phase 1 and Phase 2 DO overlap. But Phase 1 on 151 repos × ~1,500 files each is slow enough that the queue fills quickly and the consumer cannot keep up. The crash happens before Phase 1 finishes.

**Recommendation:** For very large codebases (>100K chunks), Phase 1 should complete and flush chunks to `chunks.lance` first, then Phase 2 should run separately. This is already supported via `phase="chunk"` and `phase="embed"` arguments. For the reindex command, use `pipeline=False` or split into two commands:

```bash
# Phase 1 only (fast, durable)
mvs index --phase chunk

# Phase 2 only (resumable, can restart after crash)
mvs index --phase embed
```

Add documentation warning users with >100K chunks to use separate phases.

---

## Environment Variables for Immediate Mitigation

Until fixes are deployed, set these env vars on the g4dn.xlarge:

```bash
# Tell memory monitor about the real container limit
export MCP_VECTOR_SEARCH_MAX_MEMORY_GB=11.0   # 11/14GB = ~78% safety margin

# Reduce embedding batch size (less VRAM pressure per batch)
export MCP_VECTOR_SEARCH_BATCH_SIZE=128

# Reduce max concurrent embedding threads
export MCP_VECTOR_SEARCH_MAX_CONCURRENT=4

# Reduce file batch size (smaller queue entries)
export MCP_VECTOR_SEARCH_FILE_BATCH_SIZE=64

# Run phases separately to avoid concurrent memory peaks
# Step 1: Parse only
mvs index --phase chunk

# Step 2: Embed only (resumable if it crashes)
mvs index --phase embed
```

---

## Memory Budget Breakdown for g4dn.xlarge (14 GB container)

| Component | Estimated Memory |
|-----------|-----------------|
| Python interpreter + imports | ~500 MB |
| MiniLM model weights (CPU side) | ~90 MB |
| MiniLM on GPU (VRAM) | ~90 MB |
| PyTorch CUDA allocator overhead | ~500 MB - 2 GB |
| LanceDB chunks.lance write buffers | ~200 MB - 1 GB |
| LanceDB vectors.lance write buffers | ~200 MB - 1 GB |
| Arrow / PyArrow in-process tables | ~500 MB - 2 GB |
| Queue contents (10 × 256 file batches) | ~50 MB - 500 MB |
| All 232K chunk content strings in flight | ~500 MB - 2 GB |
| ProcessPoolExecutor (fork workers) | ~200 MB per worker |
| **Total estimate at peak** | **~8 GB - 12+ GB** |

The estimate range overlaps the 14 GB container limit, confirming OOM is plausible. The lack of `gc.collect()` and `torch.cuda.empty_cache()` means peak is reached and held, not released.

---

## Summary: Priority Order

1. **CRITICAL — Fix `MemoryMonitor` default cap** (`memory_monitor.py`): Auto-detect cgroup limit; use 85% of detected limit. This is the safety net that never fired.

2. **CRITICAL — Set `MCP_VECTOR_SEARCH_MAX_MEMORY_GB` immediately** as a workaround env var (11 GB for 14 GB container).

3. **HIGH — Separate phases for large codebases**: Use `--phase chunk` then `--phase embed` for >100K chunk jobs. Phase 2 is resumable so a crash there loses no Phase 1 work.

4. **HIGH — Add `torch.cuda.empty_cache()`** after every embedding batch in `_generate_embeddings()`. Prevents VRAM from leaking into pinned/unified memory.

5. **MODERATE — Reduce queue depth** from `maxsize=10` to `maxsize=4` or make it configurable. Reduces peak RAM from buffered chunk dicts.

6. **MODERATE — Add `gc.collect()`** in the pipeline consumer every 10K embedded chunks. Currently only in the sequential Phase 1 path.

7. **LOW — Memory watchdog thread** that sends SIGTERM before SIGSEGV when RSS approaches limit. Allows checkpoint/graceful shutdown.
