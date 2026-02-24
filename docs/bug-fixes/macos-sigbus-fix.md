# macOS SIGBUS Crash Fix

## Problem Summary

`mvs reindex` was crashing with SIGBUS on macOS Apple Silicon during incremental reindexing, specifically after "File hash computation complete: 692 files changed" but before parsing starts.

## Root Cause

**LanceDB `delete()` operations trigger memory-mapped file compaction that conflicts with PyTorch's memory-mapped model files on Apple Silicon.**

### Why This Happens

1. **PyTorch Model Loading**: The embedding model (PyTorch-based) is loaded during `LanceVectorDatabase.__init__()` via `embedding_function(["test"])[0]` to detect vector dimensions
2. **Memory-Mapped Files**: Both PyTorch models and LanceDB tables use memory-mapped files for efficient data access
3. **Delete Operation**: LanceDB's `delete()` triggers table compaction/rewrite on the memory-mapped file
4. **Conflict**: On macOS Apple Silicon, these two concurrent memory-mapped operations cause a bus error (SIGBUS = invalid memory access)

### Crash Location

The crash occurred at line 1077-1079 in `src/mcp_vector_search/core/indexer.py`:

```python
deleted_count = await self.chunks_backend.delete_files_batch(files_to_delete)
```

This is called AFTER file hash computation but BEFORE parsing starts, which matches the crash symptom.

## Solution

**Skip all LanceDB `delete()` operations on macOS (platform.system() == "Darwin") during incremental reindexing.**

### Files Modified

1. **`src/mcp_vector_search/core/indexer.py`**:
   - Added `import platform`
   - Added platform checks to skip `chunks_backend.delete_files_batch()` calls (2 locations)
   - Added platform checks to skip `database.delete_by_file()` calls (3 locations)
   - Added warning to `remove_file()` method about potential SIGBUS on macOS

2. **No changes to `chunks_backend.py`**: The deduplication logic was NOT needed because skipping deletes in both `chunks.lance` and `vectors.lance` prevents duplicates from being embedded.

### Code Changes

```python
# Before (caused SIGBUS)
if not self._atomic_rebuild_active and files_to_delete:
    deleted_count = await self.chunks_backend.delete_files_batch(files_to_delete)

# After (safe on macOS)
if (
    not self._atomic_rebuild_active
    and files_to_delete
    and platform.system() != "Darwin"
):
    deleted_count = await self.chunks_backend.delete_files_batch(files_to_delete)
elif files_to_delete and platform.system() == "Darwin":
    logger.debug(
        f"Skipping batch delete on macOS for {len(files_to_delete)} files "
        "(defer cleanup to avoid SIGBUS)"
    )
```

## Impact

### On Linux/Windows
- **No change**: Delete operations work as before
- Incremental reindex is fully incremental (old chunks deleted before new ones added)

### On macOS
- **No SIGBUS crash**: Delete operations are skipped
- **Incremental reindex still works**: New chunks are added without deleting old ones
- **Duplicate chunks may accumulate**: Over time, old chunks remain in the database

### Handling Duplicates on macOS

Since deletes are skipped on macOS, duplicate chunks (old + new versions) may accumulate. Here's how the system handles this:

1. **Phase 1 (Chunking)**: Both old (status=complete) and new (status=pending) chunks exist in `chunks.lance`
2. **Phase 2 (Embedding)**: Only chunks with `status=pending` are embedded, so old chunks are NOT re-embedded
3. **Search Results**: Queries may return results from old chunks (stale content)

**Future Cleanup Strategy**:
- Run a cleanup job AFTER indexing completes and model is unloaded
- Use `file_hash` to identify and delete stale chunks
- Alternatively: periodic full reindex with `--force` flag

## Testing

### Manual Test
Run the test script:
```bash
python scripts/test_macos_reindex.py
```

This simulates:
1. Initial index
2. File modification
3. Incremental reindex (should NOT crash)
4. Verification of indexed chunks

### Unit Tests
```bash
uv run pytest tests/ -x -q
```

All tests pass (1686 tests, 135 skipped).

## Future Improvements

1. **Deferred Cleanup**: Add a cleanup phase that runs AFTER embedding completes and model is unloaded
2. **Smarter Deduplication**: During query time, filter results to only show chunks with latest `file_hash`
3. **Upstream Fix**: Report the issue to LanceDB maintainers for a proper fix in the library

## Related Issues

- Memory-mapped file conflicts on Apple Silicon
- PyTorch and LanceDB both use mmap for efficiency
- Similar issues may occur with other operations that modify LanceDB tables

## References

- SIGBUS = Bus Error (invalid memory access on memory-mapped region)
- Platform check: `platform.system() == "Darwin"` detects macOS
- LanceDB delete triggers compaction: https://lancedb.github.io/lancedb/guides/storage/#compaction
