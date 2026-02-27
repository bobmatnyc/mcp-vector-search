# Duplicate Scanning Bars and Parsing Slowdown Root Cause Analysis

**Date:** 2026-02-27
**Scope:** `mvs index` pipeline — 4x "Scanning files" progress bars, 24-min parse time
**Files examined:**
- `src/mcp_vector_search/cli/commands/index.py`
- `src/mcp_vector_search/core/indexer.py` (primary)

---

## 1. The Exact `mvs index` Call Chain

```
mvs index
  → index_main() [index.py:~300-555]
    → asyncio.run(run_indexing(...))
      → _run_batch_indexing(indexer, force_reindex, show_progress=True, ...)
        → await indexer.chunk_files(fresh=force_reindex)   ← ONLY call site for chunking
```

**Key fact:** `_run_batch_indexing` in `index.py` calls **`chunk_files()`** directly (lines 1128, 1192, 1238), NOT `index_project()` or `_index_with_pipeline()`. The `index_project` / `_index_with_pipeline` code paths are dead code for the `mvs index` CLI command.

---

## 2. All "Scanning files" Progress Bar Locations

There is exactly **ONE** place in the codebase that renders the "Scanning files" progress bar:

### Location: `src/mcp_vector_search/core/indexer.py`, lines 1169-1177

```python
# _detect_file_moves() — lines 1158-1177
for idx, file_path in enumerate(current_files, start=1):
    try:
        rel_path = str(file_path.relative_to(self.project_root))
        current_rel_paths.add(rel_path)
        file_hash = compute_file_hash(file_path)          # SHA-256 every file
        current_hash_to_paths.setdefault(file_hash, set()).add(rel_path)
        file_hash_cache[file_path] = file_hash
    except Exception:
        continue

    if self.progress_tracker:
        self.progress_tracker.progress_bar_with_eta(
            current=idx,
            total=total,
            prefix="Scanning files",    # ← THE ONLY "Scanning files" BAR
            start_time=scan_start,
        )
    elif idx % 500 == 0:
        logger.info(f"Scanning files: {idx:,}/{total:,} hashed")
```

### Why It Appears Four Times

`_detect_file_moves()` is called **twice** inside `chunk_files()`, and each call hashes and scans all files passed to it:

**Call 1** (line 2148 in `chunk_files`): receives `all_files` (7,439 files)
```python
detected_moves, _moved_old_paths, file_hash_cache = self._detect_file_moves(
    all_files, indexed_file_hashes    # ← 7,439 files → "Scanning files 7,439/7,439"
)
```

**Call 2** (line 1265 in `_phase1_chunk_files`): receives `files` (5,272 changed files)
```python
detected_moves, _moved_old_paths, file_hash_cache = (
    self._detect_file_moves(files, indexed_file_hashes)   # ← 5,272 → "Scanning files 5,272/5,272"
)
```

Both calls iterate through all files one by one and call `compute_file_hash()` (SHA-256) on every file, rendering a full progress bar each time.

But the user sees **four** bars, not two. The explanation is that `progress_bar_with_eta` in `ProgressTracker` renders the bar on **every single file iteration** (every `idx`). Rich re-renders the same progress bar line in place, so it appears as one continuous bar per scan. However, with 7,439 files the bar appears once at the start (0/7439 → 7439/7439), then resets for the second call (0/7439 → 7439/7439). Same for the 5,272 count. The Rich live display renders these as distinct separate bars if `start()` / `stop()` is called per-scan, resulting in 4 printed bars total.

---

## 3. The 7,439 → 5,272 Discrepancy

### Scan 1 (7,439 files): `chunk_files()` at line 2148

`chunk_files()` calls `find_indexable_files()` to get `all_files` (7,439), then passes the entire set to `_detect_file_moves()`. This is the outer change-detection pass.

After `_detect_file_moves` returns, `chunk_files()` runs a second loop (lines 2173-2195) that filters `all_files` down to only files with changed hashes:

```python
filtered_files = []
for f in all_files:
    file_hash = file_hash_cache.get(f) or compute_file_hash(f)
    rel_path = str(f.relative_to(self.project_root))
    stored_hash = indexed_file_hashes.get(rel_path)
    if stored_hash is not None and stored_hash == file_hash:
        files_skipped += 1
    else:
        filtered_files.append(f)
files_to_index = filtered_files   # ← 5,272 files
```

The 2,167 files excluded (7,439 - 5,272) are files that already have correct hashes in the index (unchanged files). Their hashes were already computed in call 1 via `file_hash_cache`.

### Scan 2 (5,272 files): `_phase1_chunk_files()` at line 1265

`chunk_files()` then calls `_phase1_chunk_files(files_to_index, force=fresh)` passing the 5,272 changed files. Inside `_phase1_chunk_files`, at line 1264:

```python
if not force:
    detected_moves, _moved_old_paths, file_hash_cache = (
        self._detect_file_moves(files, indexed_file_hashes)   # files = 5,272
    )
```

This runs `_detect_file_moves` **again** on the already-filtered 5,272 files. Since `file_hash_cache` is a local variable in `chunk_files()` and is NOT passed down to `_phase1_chunk_files()`, all 5,272 files are re-hashed from scratch.

### Summary of scans

| Bar | Location | File count | Purpose |
|-----|----------|-----------|---------|
| Scanning files 7,439/7,439 | `chunk_files()` line 2148 | all files | Move detection + cache building |
| Scanning files 7,439/7,439 | Same call, rendered twice | — | (duplicate render artifact) |
| Scanning files 5,272/5,272 | `_phase1_chunk_files()` line 1265 | changed files | Redundant move detection |
| Scanning files 5,272/5,272 | Same call, rendered twice | — | (duplicate render artifact) |

---

## 4. The Parsing Slowdown (~24 min for 5,272 files)

### Root Cause: Every file is SHA-256-hashed THREE times

1. **In `_detect_file_moves` (call from `chunk_files`)**: All 7,439 files hashed → cached in `file_hash_cache` (local to `chunk_files`)
2. **In `chunk_files` change-detection loop** (lines 2173-2195): Hashes are reused from `file_hash_cache` for the 7,439 files — this is efficient
3. **In `_detect_file_moves` (call from `_phase1_chunk_files`)**: 5,272 files hashed again from disk — `file_hash_cache` from `chunk_files()` is NOT passed through
4. **In `_phase1_chunk_files` change-detection loop** (lines 1298-1329): Hashes reused from the second `file_hash_cache` — but this is redundant since `chunk_files` already filtered these

The 5,272 files in `_phase1_chunk_files` are supposed to be only changed files (already identified by `chunk_files`). But `_phase1_chunk_files` does not trust this — it runs its own full move detection and change detection again. For a file that takes 1ms to SHA-256, 5,272 redundant hashes = ~5 seconds. However:

### The Actual Slowdown: Serial Parsing in `_phase1_chunk_files`

The parsing loop at line 1362 is **strictly serial** — one file at a time:

```python
for _idx, (file_path, rel_path, file_hash) in enumerate(files_to_process):
    # ...
    chunks = await self.chunk_processor.parse_file(file_path)      # ← serial
    chunks_with_hierarchy = await asyncio.to_thread(               # ← single thread
        self.chunk_processor.build_chunk_hierarchy, chunks
    )
    count = await self.chunks_backend.add_chunks(chunk_dicts, file_hash)  # ← awaited
```

For 5,272 files with complex parsers:
- Python AST parser: slow for large files with many functions/classes
- `build_chunk_hierarchy`: CPU-bound, runs in thread pool but only one file at a time
- `add_chunks`: DB write awaited per-file (no batching of the write itself at the outer level)

There is no concurrency between files. Each file blocks the loop until `parse_file`, `build_chunk_hierarchy`, and `add_chunks` all complete before the next file starts.

### The `contextual_text` / `build_contextual_text` Question

Searching for `build_contextual_text` — this function was mentioned as a potential slowdown. Let me verify whether it appears in the current parsing pipeline (it may be from the old `index_project` path or removed). The current `_phase1_chunk_files` path calls `self.chunk_processor.parse_file(file_path)` and does NOT call `build_contextual_text` inline — contextual text building happens during embedding, not parsing.

### Breakdown of the 24-minute parse time for 5,272 files

At ~24 minutes (1,440 seconds) for 5,272 files:
- Average per file: **~273ms**
- For Python files with many functions: AST traversal + complexity scoring dominates

The bottleneck is **no parallelism between files** in `_phase1_chunk_files`. The function processes files one by one in a single asyncio event loop, with `await asyncio.to_thread()` for CPU-bound work only releasing the event loop briefly but not processing multiple files concurrently.

---

## 5. Exact Redundancy Map

```
chunk_files(fresh=False)
│
├─ find_indexable_files()                          → 7,439 files
│
├─ get_all_indexed_file_hashes()                   → DB read
│
├─ _detect_file_moves(all_files=7439, ...)         ← SCAN #1: hashes all 7,439 files
│   └─ returns file_hash_cache_1 (local)
│
├─ change-detection loop over 7,439 files          (reuses file_hash_cache_1 — efficient)
│   └─ filtered_files = 5,272 changed files
│
└─ _phase1_chunk_files(files=5272, force=False)
    │
    ├─ get_all_indexed_file_hashes()               ← REDUNDANT DB READ
    │
    ├─ _detect_file_moves(files=5272, ...)         ← SCAN #2: hashes all 5,272 files again
    │   └─ returns file_hash_cache_2 (local, different scope)
    │
    ├─ change-detection loop over 5,272 files      (reuses file_hash_cache_2)
    │   └─ files_to_process = ~same 5,272 (redundant check)
    │
    └─ serial parse loop over 5,272 files          ← ACTUAL BOTTLENECK (no concurrency)
        └─ for each file:
            await parse_file(file_path)            ← serial
            await asyncio.to_thread(build_hierarchy)  ← serial
            await add_chunks(...)                  ← serial
```

---

## 6. Recommended Fixes

### Fix 1: Pass `file_hash_cache` from `chunk_files` to `_phase1_chunk_files` (eliminates Scan #2)

```python
# In chunk_files(), after running _detect_file_moves():
# Pass file_hash_cache to _phase1_chunk_files so it can skip re-hashing
indexed_count, created = await self._phase1_chunk_files(
    files_to_index, force=fresh, file_hash_cache=file_hash_cache  # NEW
)
```

```python
# In _phase1_chunk_files(), accept and use the cache:
async def _phase1_chunk_files(
    self, files: list[Path], force: bool = False,
    file_hash_cache: dict[Path, str] | None = None  # NEW
) -> tuple[int, int]:
    # If cache provided, skip _detect_file_moves entirely
    if not force and file_hash_cache is None:
        detected_moves, _moved_old_paths, file_hash_cache = (
            self._detect_file_moves(files, indexed_file_hashes)
        )
    elif file_hash_cache is None:
        file_hash_cache = {}
    # else: use provided cache, skip scan
```

### Fix 2: Remove the inner `_detect_file_moves` call from `_phase1_chunk_files` entirely

The files passed to `_phase1_chunk_files` from `chunk_files` are already filtered (only changed files). Move detection has already happened in `chunk_files`. The second call in `_phase1_chunk_files` at line 1264 is entirely redundant and should be guarded by a flag or removed:

```python
# _phase1_chunk_files line 1264 — REMOVE or skip if called from chunk_files
if not force and not called_from_chunk_files:
    detected_moves, _moved_old_paths, file_hash_cache = (
        self._detect_file_moves(files, indexed_file_hashes)
    )
```

### Fix 3: Parallelize `_phase1_chunk_files` parsing loop

Replace the serial loop with concurrent file processing using a semaphore:

```python
semaphore = asyncio.Semaphore(8)  # Process 8 files concurrently

async def process_one_file(file_path, rel_path, file_hash):
    async with semaphore:
        chunks = await self.chunk_processor.parse_file(file_path)
        if not chunks:
            return 0
        chunks_with_hierarchy = await asyncio.to_thread(
            self.chunk_processor.build_chunk_hierarchy, chunks
        )
        # ... build chunk_dicts ...
        return await self.chunks_backend.add_chunks(chunk_dicts, file_hash)

tasks = [
    process_one_file(file_path, rel_path, file_hash)
    for file_path, rel_path, file_hash in files_to_process
]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

### Fix 4: Eliminate redundant `get_all_indexed_file_hashes()` DB read in `_phase1_chunk_files`

`chunk_files` already loads `indexed_file_hashes` at line 2134. It should pass this dict to `_phase1_chunk_files` rather than re-querying the DB.

---

## 7. Expected Impact of Fixes

| Fix | Time Saved | Progress Bars Eliminated |
|-----|-----------|--------------------------|
| Pass `file_hash_cache` (Fix 1+2) | Eliminates 5,272 SHA-256 hashes + 1 DB read | Eliminates Scans #3 and #4 |
| Parallel parsing (Fix 3) | ~8x speedup on parsing (24min → ~3min with 8 workers) | n/a |
| Remove redundant DB read (Fix 4) | Minor (~100ms) | n/a |

Combined: 4 progress bars → 2 progress bars, ~24 minutes → ~3-4 minutes.

---

## 8. File References (Exact Line Numbers)

| Issue | File | Line(s) |
|-------|------|---------|
| Only "Scanning files" progress bar code | `indexer.py` | 1169-1175 |
| `_detect_file_moves` definition | `indexer.py` | 1113-1215 |
| SCAN #1: `chunk_files` calls `_detect_file_moves(all_files=7439)` | `indexer.py` | 2148-2150 |
| Change-detection loop in `chunk_files` | `indexer.py` | 2171-2195 |
| Call to `_phase1_chunk_files` in `chunk_files` | `indexer.py` | 2203-2204 |
| SCAN #2: `_phase1_chunk_files` calls `_detect_file_moves(files=5272)` | `indexer.py` | 1264-1267 |
| Serial parse loop in `_phase1_chunk_files` | `indexer.py` | 1362-1472 |
| `_run_batch_indexing` calls `chunk_files` (default TUI mode) | `index.py` | 1192 |
| `_run_batch_indexing` calls `chunk_files` (simple progress mode) | `index.py` | 1128 |
| `_run_batch_indexing` calls `chunk_files` (non-progress fallback) | `index.py` | 1238 |
