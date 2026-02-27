# Parsing Phase Slowdown Root Cause Analysis

**Date:** 2026-02-27
**Symptom:** `mvs index` showing "Parsing files" at ~1.7 files/second — 5,296 files estimated at 2h 52m 38s

---

## Executive Summary

The "Parsing files" progress bar is driven by `_phase1_chunk_files()` which processes files **one at a time** — parse, then immediately `await add_chunks()` (a synchronous LanceDB write) — with no parallelism. The bottleneck is the **per-file LanceDB write** (`self._table.add(...)` called inside `async def add_chunks()`), not the tree-sitter parsing itself. Additionally, the `reindex.py` path (which `mvs index` uses via `chunk_and_embed`) does NOT pass `skip_blame`, leaving it at the constructor default of `False`, meaning git blame subprocess calls are potentially occurring per file.

---

## 1. Where Is "Parsing files" Shown?

**Two code paths both show "Parsing files":**

### Path A: `chunk_and_embed()` → `chunk_files()` → `_phase1_chunk_files()`

This is the path taken by `mvs index` (via `index_cmd.py` → `reindex.py` → `_run_reindex()` → `indexer.chunk_and_embed()`).

- `chunk_and_embed()`: `indexer.py:2408` — calls `chunk_files()` then `embed_chunks()` sequentially
- `chunk_files()`: `indexer.py:1994` — calls `_phase1_chunk_files()` at line 2215
- `_phase1_chunk_files()`: `indexer.py:1202–1496` — **main bottleneck**

**Progress bar shown at:** `indexer.py:1450–1456`
```python
if self.progress_tracker:
    self.progress_tracker.progress_bar_with_eta(
        current=files_processed,
        total=len(files_to_process),
        prefix="Parsing files",   # <-- this label
        start_time=phase_start_time,
    )
```

The progress counter increments only AFTER a successful `add_chunks()` write (line 1401: `files_processed += 1`). This means the ETA is measuring the full parse + write cycle, not just text parsing.

### Path B: `index_project()` → `chunk_producer()` inner function

The pipeline path (used by `index_project()` at line 2456) also shows "Parsing files" at `indexer.py:845–850`. However, `mvs index` does NOT use `index_project()` — it uses `chunk_and_embed()` which calls `_phase1_chunk_files()` directly.

---

## 2. What Each File Iteration Does

### `_phase1_chunk_files()` per-file loop (lines 1365–1475):

For **each file** in `files_to_process`, the loop does:

1. **Memory check** (lines 1366–1376): `self.memory_monitor.check_memory_limit()` — may `await wait_for_memory_available()` if over threshold (potential blocking `await`)

2. **`await self.chunk_processor.parse_file(file_path)`** (line 1380):
   - `parse_file()` is `chunk_processor.py:290–303`
   - It calls `await asyncio.to_thread(self.parse_file_sync, file_path)` — offloads to thread pool
   - `parse_file_sync()` calls `parser.parse_file_sync(file_path)` (tree-sitter AST)
   - Then calls `self._enrich_chunks_with_blame()` **if `git_blame_cache` is not None**

3. **`await asyncio.to_thread(build_chunk_hierarchy, chunks)`** (line 1353): another thread pool hop

4. **Dict conversion loop** (lines 1357–1393): Python dict building for each chunk — O(chunks_per_file)

5. **`await self.chunks_backend.add_chunks(chunk_dicts, file_hash)`** (line 1397):
   - This is `chunks_backend.py:241–349`
   - Normalizes chunks in pure Python (another O(chunks_per_file) loop)
   - Then calls `self._table.add(normalized_chunks, mode="append")` — **SYNCHRONOUS LanceDB write, not wrapped in asyncio.to_thread**
   - Every 500 appends, calls `self._compact_table()` (but skipped on macOS/Darwin)

6. **Progress bar update** (lines 1416–1422)

**Key finding:** The LanceDB write at `chunks_backend.py:335` is a synchronous blocking call (`self._table.add(...)`) that runs on the event loop thread. This is called once per file, not batched across files.

---

## 3. Are There `await` Calls That Block?

Yes, several:

| Location | `await` call | Blocking risk |
|---|---|---|
| `indexer.py:1336–1341` | `await self.memory_monitor.wait_for_memory_available()` | Can block indefinitely if memory is tight |
| `indexer.py:1346` | `await self.chunk_processor.parse_file(file_path)` | Waits for thread pool — one file at a time, no parallelism |
| `indexer.py:1353` | `await asyncio.to_thread(build_chunk_hierarchy, chunks)` | Waits for thread pool — sequential with parse |
| `indexer.py:1397` | `await self.chunks_backend.add_chunks(chunk_dicts, file_hash)` | **Synchronous LanceDB write on event loop** — no `asyncio.to_thread` wrapping |

The critical observation: **all three awaits are chained sequentially for each file**. There is no concurrency between files in `_phase1_chunk_files()`.

---

## 4. Git Blame Status

### Constructor default: `skip_blame=False`

`indexer.py:95`:
```python
skip_blame: bool = False,
```

### `reindex.py` does NOT pass `skip_blame`:

`reindex.py:163–169`:
```python
indexer = SemanticIndexer(
    database=database,
    project_root=project_root,
    config=config,
    batch_size=batch_size,
    progress_tracker=progress_tracker,
)
```

No `skip_blame` argument — defaults to `False`.

### `index.py` (the `run_indexing` function) DOES handle it:

`index.py:694`: `skip_blame: bool = True` — the signature default is `True` in `run_indexing`.
`index.py:989–998`: passes `skip_blame=skip_blame` to `SemanticIndexer`.

**BUT:** `mvs index` via `index_cmd.py` goes through `reindex_main()` in `reindex.py`, NOT `run_indexing()` in `index.py`. So the `skip_blame=True` default in `run_indexing` is irrelevant to this path.

### Git blame cost when enabled:

`chunk_processor.py:276–278`:
```python
if self.git_blame_cache:
    self._enrich_chunks_with_blame(file_path, valid_chunks)
```

`_enrich_chunks_with_blame()` calls `git_blame_cache.get_blame_for_range()` per chunk.

On **first access per file**, `_populate_file_blame()` runs `subprocess.run(["git", "blame", "--porcelain", rel_path])` with a 10-second timeout (`git_blame.py:105–117`). This is a **blocking subprocess call inside `asyncio.to_thread`**. For 5,296 files, this is 5,296 `git blame` subprocess invocations.

**Verdict on v3.0.27 fix:** The `skip_blame=True` default only helps when using `run_indexing()` (invoked by some CLI paths). The `reindex.py` path used by `mvs index` does NOT get this fix — it always uses `skip_blame=False` (blame enabled).

---

## 5. Serial LanceDB Write Per File

**Yes — confirmed serial, per-file LanceDB write.**

`chunks_backend.py:335`:
```python
self._table.add(normalized_chunks, mode="append")
```

This is called inside `async def add_chunks()` at `chunks_backend.py:241`. It is a synchronous LanceDB call with **no `asyncio.to_thread` wrapping**. It runs directly on the event loop thread, blocking it until LanceDB finishes the write.

There IS a `add_chunks_batch()` method at `chunks_backend.py:351` that could write multiple files' chunks at once. It is **not used** in `_phase1_chunk_files()`.

**LanceDB append behavior:** Each call to `self._table.add()` creates a new Lance fragment (a separate file on disk). With 5,296 files at average ~5 chunks each = ~26,480 fragments before any compaction. Fragment accumulation degrades write performance over time as LanceDB must track more files.

---

## 6. Embedding Model Called Per-File?

**No.** During the "Parsing files" phase (Phase 1 / `_phase1_chunk_files`), there is **no embedding**. Chunks are stored with `embedding_status='pending'` and embedded later in Phase 2 (`embed_chunks()` / `_phase2_embed_chunks()`). The slow parsing phase is pure text parsing + LanceDB writes.

---

## 7. `_phase1_chunk_files()` vs `_phase2_embed_chunks()` — Separate or Interleaved?

### Via `chunk_and_embed()` (the `mvs index` path):

```python
chunk_result = await self.chunk_files(fresh=fresh)        # Phase 1 complete
embed_result = await self.embed_chunks(fresh=fresh, ...)  # Phase 2 starts after
```

**Strictly sequential.** Phase 1 fully completes before Phase 2 starts.

### Via `index_project()` with `pipeline=True`:

`indexer.py:1062–1063`:
```python
producer_task = asyncio.create_task(chunk_producer())   # Phase 1
consumer_task = asyncio.create_task(embed_consumer())   # Phase 2
await asyncio.gather(producer_task, consumer_task)      # overlapped
```

This uses a `chunk_queue` (maxsize=10) to overlap parsing and embedding. **But `mvs index` does not reach this code path.** It goes through `reindex.py` → `chunk_and_embed()`, not `index_project()`.

---

## 8. Root Cause Summary

The 1.7 files/second rate during "Parsing files" is caused by a combination of factors, ranked by likely impact:

### Factor 1: Serial, unbatched LanceDB writes (HIGH IMPACT)
- Location: `chunks_backend.py:335`, called from `indexer.py:1397`
- Each file does one synchronous `self._table.add()` on the event loop
- Creates one new Lance fragment per file = fragment accumulation
- Should use `add_chunks_batch()` accumulating across multiple files, then write once per batch

### Factor 2: Git blame subprocess per file (HIGH IMPACT — if enabled)
- Location: `git_blame.py:105–117`, triggered from `chunk_processor.py:277`
- `reindex.py` does not pass `skip_blame=True` — uses constructor default `False`
- 5,296 `git blame --porcelain` subprocess calls, each up to 10 seconds timeout
- Fix: `reindex.py` must pass `skip_blame=True` to `SemanticIndexer`

### Factor 3: No file-level parallelism in `_phase1_chunk_files()` (MEDIUM IMPACT)
- Location: `indexer.py:1365` — sequential `for` loop, one file at a time
- `parse_file()` uses `asyncio.to_thread` but only one file is in-flight at a time
- The pipeline path (`index_project()`) has a producer-consumer to overlap with embedding, but `chunk_and_embed()` is fully sequential
- Fix: Process N files concurrently using `asyncio.gather()` or semaphore-limited tasks

### Factor 4: Double thread-pool round-trip per file (LOW-MEDIUM IMPACT)
- Location: `indexer.py:1346` + `indexer.py:1353`
- Two separate `asyncio.to_thread` calls per file (parse, then hierarchy build)
- Could be merged into one combined synchronous function dispatched to thread pool

### Factor 5: Fragment accumulation in LanceDB (LATENT, GROWS OVER TIME)
- First-run (5,296 new files) creates 5,296 fragments in `chunks.lance`
- Compaction is skipped on macOS (`_compact_table` returns early at `chunks_backend.py:531`)
- Write latency increases as fragment count grows: first file is fast, 5000th file is slower

---

## 9. Why `chunk_producer()` (the fast path) Is Not Used

`mvs index` routes through:
```
index_cmd.py:index_main()
  → reindex.py:reindex_main()
    → _run_reindex()
      → indexer.chunk_and_embed()       # NOT index_project()
        → chunk_files()
          → _phase1_chunk_files()       # serial, no pipeline
        → embed_chunks()               # phase 2 separate
```

The pipelined `chunk_producer()`/`embed_consumer()` code (lines 600–1096) lives inside `index_project()` (line 2456) and is never called by this route.

The `chunk_producer()` path DOES use the `file_batch_size = 256` batch grouping (line 598) but still calls `add_chunks()` per-file within each batch (line 826). It would have the same per-file write problem.

---

## 10. Verification: The Duetto/CTO Project Context

7,439 total files with 5,296 needing update suggests this is likely a **first-time index** (no existing index) or a **force reindex**. In either case:

- `indexed_file_hashes` will be empty → all files go into `files_to_process`
- 5,296 sequential parse+write cycles at ~1.7 files/sec = ~3,115 seconds (52 minutes)
- The ETA shown (2h 52m) suggests the rate is already degrading mid-run, consistent with growing LanceDB fragment count

---

## Recommended Fixes

1. **Fix `reindex.py` to pass `skip_blame=True`** (or read from env). This may halve the time if blame is the dominant cost.

2. **Batch LanceDB writes across files**: Accumulate chunk dicts for N files (e.g., 100–256), then call `add_chunks_batch()` once. This reduces fragment creation from 5,296 to ~21 fragments.

3. **Parallelize parsing**: Use `asyncio.Semaphore` to parse K files concurrently (K = CPU cores), then batch their chunks into a single write.

4. **Route `mvs index` through `index_project(pipeline=True)`**: The existing pipelined producer-consumer code would at minimum overlap parsing with embedding.

5. **Wrap the LanceDB write in `asyncio.to_thread`**: At minimum, prevent blocking the event loop during `self._table.add()`.

---

## Key File References

| File | Lines | Description |
|---|---|---|
| `src/mcp_vector_search/cli/commands/reindex.py` | 163–169 | Missing `skip_blame=True` |
| `src/mcp_vector_search/core/indexer.py` | 1202–1496 | `_phase1_chunk_files()` — serial loop |
| `src/mcp_vector_search/core/indexer.py` | 1346 | `await parse_file()` — one at a time |
| `src/mcp_vector_search/core/indexer.py` | 1397 | `await add_chunks()` — per-file write |
| `src/mcp_vector_search/core/indexer.py` | 1450–1456 | "Parsing files" progress bar |
| `src/mcp_vector_search/core/indexer.py` | 2390–2420 | `chunk_and_embed()` — sequential phases |
| `src/mcp_vector_search/core/chunks_backend.py` | 335 | `self._table.add()` — sync, no batching |
| `src/mcp_vector_search/core/chunks_backend.py` | 351 | `add_chunks_batch()` — exists but unused here |
| `src/mcp_vector_search/core/chunks_backend.py` | 518–570 | `_compact_table()` — skipped on macOS |
| `src/mcp_vector_search/core/chunk_processor.py` | 276–278 | Git blame enrichment call |
| `src/mcp_vector_search/core/git_blame.py` | 105–117 | `subprocess.run(git blame)` — per file |
