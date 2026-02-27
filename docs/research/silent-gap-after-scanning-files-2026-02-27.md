# Silent Gap After "Scanning files" Progress Bar — Root Cause Analysis

**Date**: 2026-02-27
**Project**: mcp-vector-search
**Scope**: `mvs index` on a 7,439-file project

---

## Executive Summary

After the "Scanning files" progress bar reaches 100%, there is a silent processing gap before parsing begins. The gap consists of **three serial blocking operations** on the full 7,439-file list with no user-visible progress. The dominant cost is a double-hashing of all files: `_detect_file_moves()` hashes every file once, then `chunk_producer()` inside `_index_with_pipeline` hashes them all again (unless the cache is reused). Combined with a pandas `groupby` over potentially 100K+ LanceDB rows and a silent file-classification loop, the total gap for 7,439 files is estimated at **15–45 seconds** on a cold SSD.

---

## Exact Code Path: Scanning Completion → Parsing Start

### Entry Point: `mvs index`

```
mvs index
  → index_cmd.py: _run_all_phases()
  → reindex.py: reindex_main() / _run_reindex()
  → indexer.py: indexer.chunk_and_embed(fresh=False)
  → indexer.py: chunk_files(fresh=False)          ← Phase 1 entry
```

### Inside `chunk_files()` (indexer.py line 2025)

```
chunk_files(fresh=False)
  1. _atomic_rebuild_databases(False)              → fast, no-op for incremental
  2. chunks_backend.initialize()                   → opens LanceDB, fast
  3. vectors_backend.initialize()                  → opens LanceDB, fast
  4. _run_auto_migrations()                        → checks migration table, fast
  5. apply_auto_optimizations()                    → CodebaseProfiler sampling, fast
  6. cleanup_stale_locks(self.index_path)          → filesystem stat, fast
  7. file_discovery.find_indexable_files()         → os.walk, fast (cached after first call)
  8. metadata.cleanup_stale_entries(valid_files)   → JSON dict diff, fast
  9. chunks_backend.get_all_indexed_file_hashes()  ← BOTTLENECK 1
 10. _detect_file_moves(all_files, indexed_hashes) ← BOTTLENECK 2  ("Scanning files" bar)
 11. [if moves detected] reload indexed_hashes     ← conditional reload
 12. filtered_files loop over all_files            ← BOTTLENECK 3  (silent)
 13. _phase1_chunk_files(filtered_files)           → "Parsing files" bar begins HERE
```

The "Scanning files" progress bar is rendered inside `_detect_file_moves()` (indexer.py lines 1169–1175). It finishes when the hash loop exits at line 1180. Steps 11 and 12 follow immediately with no progress output.

---

## The Three Silent-Gap Operations

### Bottleneck 1: `get_all_indexed_file_hashes()` — LanceDB full scan + pandas groupby

**Location**: `chunks_backend.py` lines 609–620
**Code**:
```python
scanner = self._table.to_lance().scanner(columns=["file_path", "file_hash"])
result = scanner.to_table()          # full column scan of all chunks
df = result.to_pandas()              # Arrow → pandas copy
file_hashes = df.groupby("file_path")["file_hash"].first().to_dict()
```

**What it does**: Scans every row in `chunks.lance` for two columns, copies to pandas, then groups to deduplicate per-file hashes.

**Scale at 7,439 files**: Assuming ~10 chunks per file = ~74,000 rows. At ~500 bytes/row for those two string columns that is ~37 MB of Arrow data copied through pandas.

**Estimated time**: 1–4 seconds (LanceDB scan) + 1–3 seconds (Arrow→pandas copy + groupby) = **2–7 seconds total**. No progress indicator is shown.

### Bottleneck 2: `_detect_file_moves()` — SHA-256 hash of all 7,439 files

**Location**: `indexer.py` lines 1158–1175
**Code**:
```python
for idx, file_path in enumerate(current_files, start=1):
    file_hash = compute_file_hash(file_path)   # SHA-256 of file bytes
    ...
    if self.progress_tracker:
        self.progress_tracker.progress_bar_with_eta(current=idx, total=total,
            prefix="Scanning files", ...)
```

**What it does**: Reads and SHA-256 hashes every file on disk. This IS the "Scanning files" bar the user sees. The bar completes at line 1180.

**Scale**: 7,439 files. Reading ~50 KB average × 7,439 = ~372 MB of file I/O. SHA-256 throughput on modern hardware is ~1 GB/s, so this takes **5–15 seconds** for a cold page cache (disk-bound) or **1–3 seconds** warm.

After the bar finishes, `_detect_file_moves()` still iterates `hash_to_indexed` to find orphaned paths. This is a pure dict operation — fast, O(n) where n = indexed paths.

### Bottleneck 3: Change detection loop — re-hash + O(1) dict lookup, no progress bar

**Location**: `chunk_files()` lines 2171–2193 (called from `chunk_and_embed` path) or `_index_with_pipeline()` lines 584–596

**Code** (pipeline path, indexer.py line 584):
```python
filtered_files = []
for f in all_files:
    file_hash = file_hash_cache.get(f) or compute_file_hash(f)  # cache miss = re-hash!
    rel_path = str(f.relative_to(self.project_root))
    stored_hash = indexed_file_hashes.get(rel_path)
    if stored_hash is None or stored_hash != file_hash:
        filtered_files.append(f)
```

**Critical observation**: The `file_hash_cache` populated by `_detect_file_moves` IS reused here via `file_hash_cache.get(f)`. If every file is a cache hit (which it will be after a full `_detect_file_moves` run), this loop is purely a dict lookup + string path construction for 7,439 files — **fast (~0.1 seconds)**.

However, inside `chunk_producer()` (the pipeline path, lines 697–718), `_detect_file_moves` is called **a second time** for the same `files_to_index` list. This results in **double-hashing** when `_index_with_pipeline` is used:

```
_index_with_pipeline():
  ├── get_all_indexed_file_hashes()          # LanceDB scan
  ├── _detect_file_moves(all_files, ...)     # hashes all 7,439 files → "Scanning files" bar
  ├── [change detection loop]                # uses file_hash_cache, fast
  └── chunk_producer():                      # coroutine started as task
        ├── get_all_indexed_file_hashes()    # SECOND LanceDB scan  ← SILENT
        ├── _detect_file_moves(files_to_index, ...)  # SECOND hash of changed files ← SILENT
        └── [hash loop again]                # SILENT
```

The consumer task starts immediately, but the producer begins with two more expensive operations before any parsing happens.

**No progress indicator** exists for the second `get_all_indexed_file_hashes()` call or the second `_detect_file_moves` call inside `chunk_producer()`.

---

## Timeline of the Silent Gap (7,439 files, incremental, warm cache)

| Step | Operation | Location | Duration estimate | Has progress bar? |
|------|-----------|----------|-------------------|-------------------|
| 1 | `get_all_indexed_file_hashes()` — LanceDB full scan | `chunk_files()` line 2134 | 2–7 s | No |
| 2 | `_detect_file_moves()` — hash 7,439 files | `chunk_files()` line 2148 | 1–15 s | YES — "Scanning files" |
| 3 | Change detection loop — dict lookups | `chunk_files()` lines 2171–2193 | 0.1–0.5 s | No |
| 4 | `_phase1_chunk_files()` → starts "Parsing files" bar | line 2203 | — | YES |

**For the `chunk_and_embed` path specifically** (`reindex.py` → `chunk_files()`), Steps 1–3 cover the gap.

**For the `_index_with_pipeline` path** (if called), there is an additional silent gap inside `chunk_producer()` at lines 650–695:

| Step | Operation | Duration estimate | Has progress bar? |
|------|-----------|-------------------|-------------------|
| 5 | `get_all_indexed_file_hashes()` inside producer | 2–7 s | No |
| 6 | `_detect_file_moves()` inside producer | 0.5–5 s (only changed files) | No |
| 7 | Hash loop inside producer | 0.1 s | No |

**Total silent gap estimate**: **5–30 seconds** (dominated by LanceDB scan and file hashing).

---

## Root Cause Summary

1. **Double `get_all_indexed_file_hashes()` call** (Bottleneck 1 + Step 5): The LanceDB full scan is executed twice in the pipeline path — once in `_index_with_pipeline()` and once inside `chunk_producer()`. Each call reads and groups all chunk rows. No caching between them.

2. **Double `_detect_file_moves()` call** (Bottleneck 2 + Step 6): In the pipeline path, `_detect_file_moves` is called in the outer function and again inside `chunk_producer()`. The first call produces `file_hash_cache` which is passed through `files_to_index` but the second call discards it and rehashes.

3. **No progress indicator** on the LanceDB scan (Step 1 and Step 5). The pandas groupby over 74K+ rows is completely silent.

4. **No progress indicator** on the silent change-detection loop (Step 3), though it is fast.

---

## Recommended Fixes — Priority Order

### Fix 1 (CRITICAL): Eliminate duplicate `get_all_indexed_file_hashes()` inside `chunk_producer()`

**File**: `indexer.py`, `_index_with_pipeline()` lines 649–658
**Problem**: `chunk_producer()` calls `get_all_indexed_file_hashes()` and `_detect_file_moves()` even though the outer `_index_with_pipeline()` already ran them and computed `files_to_index`.
**Fix**: Remove both calls from inside `chunk_producer()`. The outer function already filtered `files_to_index` — pass it directly. The producer only needs to iterate `files_to_index`, not re-detect changes.

```python
# BEFORE (chunk_producer lines 649-695):
indexed_file_hashes = {}
if not force_reindex:
    indexed_file_hashes = await self.chunks_backend.get_all_indexed_file_hashes()
    detected_moves, _, file_hash_cache = self._detect_file_moves(...)
    # ... more hash loops ...
    files_to_process = [...]

# AFTER: files_to_index is already filtered by outer function.
# Just iterate it directly.
files_to_process = [(f, str(f.relative_to(self.project_root)), "") for f in files_to_index]
```

**Estimated savings**: 3–12 seconds eliminated entirely.

### Fix 2 (HIGH): Add a progress indicator for `get_all_indexed_file_hashes()`

**File**: `chunks_backend.py` line 611
**Problem**: The LanceDB full scan is the first silent 2–7 second operation the user sees after setup.
**Fix**: Print a brief status message before and after the scan.

```python
async def get_all_indexed_file_hashes(self) -> dict[str, str]:
    if self._table is None:
        return {}
    try:
        logger.info("Loading indexed file hashes...")   # visible in console if logger shows INFO
        scanner = self._table.to_lance().scanner(columns=["file_path", "file_hash"])
        result = scanner.to_table()
        ...
        logger.info(f"Loaded {len(file_hashes)} file hashes in {elapsed:.1f}s")
```

Or surface this through `progress_tracker` if one is available (requires passing it as argument or using a module-level accessor). A one-line `console.print("[dim]Loading change index...[/dim]")` before the call in `chunk_files()` would also work.

**Estimated time**: 1 hour. Immediate UX improvement.

### Fix 3 (HIGH): Replace pandas groupby with pure PyArrow aggregation

**File**: `chunks_backend.py` lines 617–620
**Problem**: `result.to_pandas()` copies Arrow memory into pandas unnecessarily. For 74K rows this is wasteful.
**Fix**: Use PyArrow's native groupby:

```python
# BEFORE:
df = result.to_pandas()
file_hashes = df.groupby("file_path")["file_hash"].first().to_dict()

# AFTER (pure PyArrow, no pandas dependency for this operation):
import pyarrow.compute as pc
grouped = result.group_by("file_path").aggregate([("file_hash", "first")])
file_hashes = dict(zip(
    grouped["file_path"].to_pylist(),
    grouped["file_hash_first"].to_pylist(),
))
```

**Estimated savings**: 0.5–2 seconds (eliminates Arrow→pandas copy overhead). Also reduces peak memory by ~2×.

### Fix 4 (MEDIUM): Cache `get_all_indexed_file_hashes()` result for the session

**File**: `chunks_backend.py`
**Problem**: The result is valid for the entire indexing session unless moves are detected (which trigger a reload anyway).
**Fix**: Add an in-memory cache on `ChunksBackend` with a dirty flag.

```python
class ChunksBackend:
    def __init__(self, ...):
        self._file_hash_cache: dict[str, str] | None = None

    async def get_all_indexed_file_hashes(self) -> dict[str, str]:
        if self._file_hash_cache is not None:
            return self._file_hash_cache
        ...  # compute as before
        self._file_hash_cache = file_hashes
        return file_hashes

    def invalidate_hash_cache(self) -> None:
        self._file_hash_cache = None
```

Call `invalidate_hash_cache()` after any `update_file_path()` or `delete_files_batch()` call. This eliminates the second scan in the pipeline path and any future redundant calls.

**Estimated savings**: Removes the second 2–7 second scan entirely.

### Fix 5 (MEDIUM): Add progress bar for the change detection loop

**File**: `indexer.py` lines 2171–2193 (in `chunk_files()`), lines 584–596 (in `_index_with_pipeline()`)
**Problem**: The file classification loop has no visual feedback. For 7,439 files it is fast (~0.1 s) but for projects with 50K+ files it becomes noticeable.
**Fix**: Add progress_tracker update every 1,000 files, similar to how `_detect_file_moves` does it:

```python
for idx, f in enumerate(all_files, start=1):
    ...
    if self.progress_tracker and idx % 500 == 0:
        self.progress_tracker.progress_bar_with_eta(
            current=idx, total=len(all_files),
            prefix="Checking changes", start_time=loop_start,
        )
```

### Fix 6 (LOW): Pass `file_hash_cache` from outer scope into `chunk_producer()`

If Fix 1 (removing duplicate calls) is not done immediately, at minimum ensure the `file_hash_cache` from `_index_with_pipeline()` outer scope is available inside `chunk_producer()`. Currently it IS captured as a closure variable (`file_hash_cache` at line 707 does use `file_hash_cache.get(file_path)`) — but `_detect_file_moves` is still called a second time to rebuild it. The closure already has the first cache available; remove the second call.

---

## Code Locations Quick Reference

| File | Lines | Issue |
|------|-------|-------|
| `indexer.py` | 559 | First `_detect_file_moves()` in `_index_with_pipeline()` — has "Scanning files" bar |
| `indexer.py` | 584–596 | Change detection loop — no progress bar, uses hash cache |
| `indexer.py` | 649–695 | `chunk_producer()` — duplicate LanceDB scan + duplicate `_detect_file_moves()` |
| `indexer.py` | 2134–2149 | `chunk_files()` — first LanceDB scan, no progress shown before it |
| `indexer.py` | 2171–2193 | `chunk_files()` change detection loop — no progress bar |
| `chunks_backend.py` | 609–620 | `get_all_indexed_file_hashes()` — silent LanceDB scan + pandas groupby |

---

## Verification Commands

To measure the gap empirically:

```bash
# Add timing around the key operations (temporary debug logging):
MCP_VECTOR_SEARCH_LOG_LEVEL=DEBUG mvs index 2>&1 | grep -E "Scanning files|Loading indexed|Loaded [0-9]+ indexed|Incremental change|Incremental index|Phase 1|Chunking"
```

Expected log sequence (reveals gap timing):
```
INFO  Loading indexed file hashes for change detection...
INFO  Loaded N indexed files for change detection           ← end of LanceDB scan
INFO  Scanning files: ...                                  ← _detect_file_moves starts
INFO  Scanned 7,439 files in X.Xs                         ← "Scanning files" bar complete
# [SILENT GAP: change detection loop]
INFO  Incremental index: M of 7,439 files need updating   ← gap ends
INFO  Phase 1: Chunking M files                           ← "Parsing files" bar starts
```
