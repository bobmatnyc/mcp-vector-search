# Incremental Indexing Analysis: Why `mvs index` Shows 17,225 Files

**Date:** 2026-02-25
**Project:** mcp-vector-search
**Question:** Why does `mvs index` process all 17,225 files in "incremental" mode?

---

## Executive Summary

**This is NOT a bug.** The 17,225 files shown in `Parsing files... 0% 31/17,225` is the count of **files that have actually changed** and need to be processed, not the total file count. However, there is a significant **misleading UX issue**: the "Incremental reindex: processing only changes" message is correct, but the progress bar appears before change detection completes (in the pipeline path), making it look like all 17,225 files are being scanned unnecessarily.

**Key finding:** The progress bar `total=len(files_to_process)` shows only changed files. If 17,225 files show as needing reprocessing, it means the `get_all_indexed_file_hashes()` scan returned empty results (fresh index, corrupted index, or path mismatch), causing all files to be treated as new/changed.

---

## 1. Code Path: `mvs index` → What Actually Runs

### Full call chain:

```
mvs index
  → index_cmd.py::index_main()           [index_cmd_app callback]
    → _run_all_phases(force=False)
      → reindex.py::reindex_main()        [via MockContext]
        → _run_reindex(project_root, fresh=False)
          → indexer.chunk_and_embed(fresh=False)
            → indexer.chunk_files(fresh=False)
              → indexer._phase1_chunk_files(files_to_index)  ← PROGRESS BAR HERE
            → indexer.embed_chunks(fresh=False)
```

### The "Incremental reindex: processing only changes" message source

```python
# reindex.py line 142
if fresh:
    print_warning("Full reindex: clearing all data and rebuilding from scratch")
else:
    print_info("Incremental reindex: processing only changes")
```

This message is **always** printed when `fresh=False` (the default), regardless of whether change detection actually finds anything. It's a label, not a post-detection result.

---

## 2. The Two-Stage Change Detection

### Stage 1: File Discovery (returns ALL indexable files)

```python
# chunk_files() in indexer.py line 1796
all_files = self.file_discovery.find_indexable_files()
# This returns ALL indexable files - 17,225 in this case
```

### Stage 2: Hash-based filtering (the actual change detection)

```python
# indexer.py lines 1821-1845
indexed_file_hashes = await self.chunks_backend.get_all_indexed_file_hashes()

filtered_files = []
for f in all_files:
    file_hash = compute_file_hash(f)           # SHA-256 of file content
    rel_path = str(f.relative_to(self.project_root))
    stored_hash = indexed_file_hashes.get(rel_path)
    if stored_hash is not None and stored_hash == file_hash:
        files_skipped += 1                     # SKIP - unchanged
    else:
        filtered_files.append(f)               # QUEUE - changed or new

files_to_index = filtered_files                # Only changed files!
```

### Stage 3: Progress bar shows ONLY changed files

```python
# indexer.py _phase1_chunk_files() line 1197
self.progress_tracker.progress_bar_with_eta(
    current=files_processed,
    total=len(files_to_process),   # ← len(files_to_process) = only CHANGED files
    prefix="Parsing files",
    start_time=phase_start_time,
)
```

**The `total=len(files_to_process)` is the count of files that NEED reprocessing, not the total file count.**

---

## 3. What the "31/17,225" Count Represents

The number **17,225 is the count of changed (or new) files**, not the total project files. The "31" is how many have been processed so far.

If 17,225 out of potentially ~20,000 total files appear as "changed", the most likely root cause is one of these:

### Root Cause A: Empty or unreachable `chunks_backend` (most likely)

```python
# chunks_backend.py line 534-535
async def get_all_indexed_file_hashes(self) -> dict[str, str]:
    if self._table is None:
        return {}   # ← Returns EMPTY dict if table not initialized
```

If `self._table is None`, ALL files fail the `stored_hash is None` check and are treated as new/changed:

```python
stored_hash = indexed_file_hashes.get(rel_path)  # None (empty dict)
if stored_hash is not None and stored_hash == file_hash:
    files_skipped += 1
else:
    filtered_files.append(f)  # ← ALL files land here
```

### Root Cause B: Path mismatch in stored hashes

The stored `file_path` in `chunks.lance` is a **relative path** (e.g., `src/foo/bar.py`), and the lookup uses:

```python
rel_path = str(f.relative_to(self.project_root))
stored_hash = indexed_file_hashes.get(rel_path)
```

If `project_root` differs between indexing runs (e.g., symlinks, different absolute paths), the relative paths won't match, causing all files to be treated as new.

### Root Cause C: Schema mismatch or LanceDB corruption

```python
# chunks_backend.py line 555-557
except Exception as e:
    logger.warning(f"Failed to get all indexed file hashes: {e}")
    return {}  # ← Silent failure returns empty dict
```

Any exception during the LanceDB scan silently returns `{}`, causing all files to be treated as new.

### Root Cause D: Double change detection in pipeline mode (inefficiency)

In the `_index_with_pipeline` path (used by `index_project()` with `pipeline=True`), change detection happens **twice**:

```python
# First pass - outer function _index_with_pipeline (line 496-526)
indexed_file_hashes = await self.chunks_backend.get_all_indexed_file_hashes()
filtered_files = [f for f in all_files if hash_changed(f)]
files_to_index = filtered_files  # Already filtered

# Second pass - inside chunk_producer (line 574-623)
indexed_file_hashes = await self.chunks_backend.get_all_indexed_file_hashes()  # LOADED AGAIN
for file_path in files_to_index:
    file_hash = compute_file_hash(file_path)
    if stored_hash == file_hash:
        continue  # Skip unchanged (but files_to_index is already filtered!)
```

This is wasteful but not a bug - the second pass catches edge cases where files change between the two checks.

---

## 4. The `chunk_and_embed` vs `index_project` / `_index_with_pipeline` Distinction

### `mvs index` (via `reindex_main`) uses `chunk_and_embed`

```python
# reindex.py line 176
result = await indexer.chunk_and_embed(fresh=fresh, batch_size=batch_size)
```

`chunk_and_embed` calls `chunk_files()` (sequential, not pipeline), which:
1. Discovers all files
2. Runs change detection (loading hashes from chunks.lance)
3. Passes only changed files to `_phase1_chunk_files()`
4. Progress bar shows `total=len(files_to_process)` (changed files only)

### `mvs index chunk` (via `index.py run_indexing`) uses `chunk_files` directly

```python
# index.py line 1192
result = await indexer.chunk_files(fresh=force_reindex)
```

Same behavior as above.

### `index_project()` with `pipeline=True` uses `_index_with_pipeline`

This is a more complex pipeline mode not called by `mvs index` default path.

---

## 5. Is the Progress Bar Accurate?

**Yes and no:**

- `total=len(files_to_process)` = number of **changed** files needing processing ✓
- `current=files_processed` = number successfully chunked so far ✓
- If `files_to_process` has 17,225 entries, that IS the number being processed

**The potentially misleading part:** The "Incremental reindex: processing only changes" banner is printed BEFORE change detection runs. So users see:

```
ℹ Incremental reindex: processing only changes
[change detection scans 17,225 files - no progress shown]
Parsing files... 0% 31/17,225  ← this is ONLY changed files
```

There is NO progress bar or status output during the change detection phase itself (the `compute_file_hash` loop). For 17,225 files, this silent phase takes significant time and makes it appear the system is hung.

---

## 6. Summary: Bug vs Expected Behavior

| Question | Answer |
|----------|--------|
| Is "Parsing files... X/17,225" scanning all 17,225 files for change detection? | **No** - it's processing 17,225 changed files |
| Does change detection scan all 17,225 files? | **Yes, silently** (no progress bar during hash computation) |
| Is the 17,225 the total file count? | **No** - it's the count of files that need reprocessing |
| Is the progress bar misleading? | **Yes** - it starts after silent change detection, making users think it's scanning all files |
| Is incremental detection working? | **Depends** - if 17,225 out of ~N total files are "changed", the index may be empty/corrupted |
| Where is the bug? | **UX**: no progress bar during change detection; possible silent `get_all_indexed_file_hashes()` failure |

---

## 7. Recommended Fixes

### Fix 1: Add progress output during change detection phase

```python
# In chunk_files() and _phase1_chunk_files()
logger.info(f"Incremental mode: checking {len(all_files)} files for changes...")
# ADD: console.print(f"[dim]Scanning {len(all_files)} files for changes...[/dim]")
for f in all_files:
    file_hash = compute_file_hash(f)
    # ... existing logic ...

logger.info(f"Found {len(files_to_index)} changed files to index")
# ADD: console.print(f"[green]Found {len(files_to_index)} changed files[/green]")
```

### Fix 2: Log when `get_all_indexed_file_hashes` returns empty on a non-fresh index

```python
async def get_all_indexed_file_hashes(self) -> dict[str, str]:
    if self._table is None:
        logger.warning("chunks table not initialized - treating all files as new")
        return {}
    # ...
    if len(file_hashes) == 0:
        logger.warning("No indexed file hashes found - all files will be treated as new/changed")
    return file_hashes
```

### Fix 3: Distinguish "all files are new" from "all files changed"

In the `chunk_files()` method, after change detection:
```python
if len(indexed_file_hashes) == 0 and not fresh:
    logger.warning("No existing index found - treating all files as new (first-time index)")
else:
    logger.info(f"Found {len(files_to_index)} changed files to index ({files_skipped} unchanged)")
```

---

## 8. Specific Code Locations

| What | File | Line |
|------|------|------|
| "Incremental reindex" message | `src/mcp_vector_search/cli/commands/reindex.py` | 142 |
| Change detection loop | `src/mcp_vector_search/core/indexer.py` | 1829-1844 |
| Progress bar `total=len(files_to_process)` | `src/mcp_vector_search/core/indexer.py` | 1197 |
| `get_all_indexed_file_hashes()` | `src/mcp_vector_search/core/chunks_backend.py` | 524-557 |
| Silent failure → empty dict | `src/mcp_vector_search/core/chunks_backend.py` | 555-557 |
| Double hash load in pipeline | `src/mcp_vector_search/core/indexer.py` | 496, 574 |
| `compute_file_hash()` | `src/mcp_vector_search/core/chunks_backend.py` | 27-40 |
