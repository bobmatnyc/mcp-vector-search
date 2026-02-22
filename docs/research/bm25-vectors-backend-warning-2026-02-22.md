# Investigation: "Vectors backend not initialized" Warning During BM25 Index Build

**Date**: 2026-02-22
**Investigator**: Research Agent
**Status**: Root cause identified, fix recommended

---

## Executive Summary

The warning "Vectors backend not initialized, skipping BM25 index build" occurs during `mvs reindex -f` because the BM25 index building happens **before** the vectors table is created in atomic rebuild mode. The vectors table is only created during Phase 2 (embedding), but BM25 index build expects it to exist after Phase 1 (chunking).

**Impact**: BM25 index is not built during fresh reindex, degrading search quality by removing keyword search capabilities.

---

## Root Cause Analysis

### 1. Code Location

**File**: `src/mcp_vector_search/core/indexer.py`
**Line**: 3437 (warning emission)
**Method**: `_build_bm25_index()`

```python
async def _build_bm25_index(self) -> None:
    """Build BM25 index from all chunks for keyword search."""
    try:
        # Get all chunks from vectors.lance table
        if self.vectors_backend._db is None or self.vectors_backend._table is None:
            logger.warning(
                "Vectors backend not initialized, skipping BM25 index build"  # Line 3437
            )
            return
```

### 2. Call Stack Leading to Warning

```
chunk_files(fresh=True)                    # Line 1685
  ├─ _atomic_rebuild_databases()           # Line 1705 (creates NEW backends)
  ├─ _phase1_chunk_files()                 # Line 1767 (chunks → chunks.lance)
  └─ _build_bm25_index()                   # Line 1801 ⚠️ PROBLEM HERE
       └─ Check: vectors_backend._table exists? → NO (not created yet)
```

### 3. Why Vectors Backend is Not Initialized

During atomic rebuild (`fresh=True`), the workflow is:

**Phase 1: Chunking** (in `chunk_files()`)
1. `_atomic_rebuild_databases()` creates **new backend instances** pointing to `.new` directories:
   ```python
   # Line 1551-1552
   self.chunks_backend = ChunksBackend(lance_new)
   self.vectors_backend = VectorsBackend(lance_new)
   ```

2. `_phase1_chunk_files()` writes chunks to `chunks.lance` table

3. **`_build_bm25_index()` is called** (line 1801) - but it expects data in `vectors.lance` table

**Problem**: The `vectors.lance` table is **only created in Phase 2** (embedding phase), which hasn't run yet!

**Phase 2: Embedding** (separate method `embed_chunks()`)
- Reads chunks from `chunks.lance`
- Generates embeddings
- Writes to `vectors.lance` table ← **This is when vectors table is created**

### 4. Why the Check Fails

```python
# Line 3435 in _build_bm25_index()
if self.vectors_backend._db is None or self.vectors_backend._table is None:
    logger.warning("Vectors backend not initialized, skipping BM25 index build")
    return
```

**During fresh reindex**:
- `vectors_backend._db` is **not None** (initialized at line 1714)
- `vectors_backend._table` **IS None** because:
  - Table doesn't exist yet in `.new` directory (fresh rebuild)
  - VectorsBackend.initialize() sets `_table = None` when table doesn't exist (line 276)

```python
# vectors_backend.py, line 274-277
else:
    # Table will be created on first add_vectors with schema
    self._table = None
    logger.debug("Vectors table will be created on first write")
```

---

## Additional Call Sites

BM25 index building is attempted in **4 locations**:

1. **Line 523**: `chunk_files()` - when BM25 index file missing (incremental mode)
2. **Line 1801**: `chunk_files()` - after Phase 1 completes ⚠️ **PROBLEMATIC**
3. **Line 1904**: `embed_chunks()` - after Phase 2 completes ✅ **CORRECT**
4. **Line 2234**: `index_project()` - after full index (pipeline mode) ✅ **CORRECT**

---

## Why This Design is Problematic

### BM25 Should Use Chunks Table, Not Vectors Table

The `_build_bm25_index()` method currently reads from **`vectors.lance`**:

```python
# Line 3446
df = await asyncio.to_thread(self.vectors_backend._table.to_pandas)
```

**Problem**: BM25 is a **keyword search algorithm** that only needs:
- `chunk_id`
- `content` (text)
- `name`
- `file_path`
- `chunk_type`

These fields are **all available in `chunks.lance`** (Phase 1), and don't require embeddings.

**Current behavior**: BM25 index build is tied to vectors table (Phase 2), creating unnecessary dependency.

---

## Recommended Fix

### Option 1: Read BM25 Data from Chunks Table (Preferred)

**Change `_build_bm25_index()` to read from `chunks.lance` instead of `vectors.lance`**:

```python
async def _build_bm25_index(self) -> None:
    """Build BM25 index from all chunks for keyword search."""
    try:
        # BM25 only needs text content, not embeddings
        # Read from chunks.lance (Phase 1) instead of vectors.lance (Phase 2)
        if self.chunks_backend._db is None or self.chunks_backend._table is None:
            logger.warning(
                "Chunks backend not initialized, skipping BM25 index build"
            )
            return

        logger.info("📚 Phase 3: Building BM25 index for keyword search...")

        # Query all records from chunks.lance table
        import asyncio
        df = await asyncio.to_thread(self.chunks_backend._table.to_pandas)

        # ... rest of method unchanged ...
```

**Benefits**:
- BM25 index can be built after Phase 1 (chunking) completes
- No dependency on Phase 2 (embedding) completing first
- Works correctly in atomic rebuild mode
- Faster: don't wait for embedding to complete before keyword search is available
- More logical: keyword search doesn't need vector embeddings

**Required Changes**:
1. Update line 3435 to check `chunks_backend` instead of `vectors_backend`
2. Update line 3446 to read from `self.chunks_backend._table`
3. Update field extraction (line 3456-3469) if chunk schema differs

### Option 2: Move BM25 Build After Finalization (Workaround)

Move `_build_bm25_index()` call to **after** atomic rebuild finalization:

```python
# In chunk_files() method, line 1801
# Build BM25 index
# REMOVED: await self._build_bm25_index()  # Too early - vectors table not created yet

# Finalize atomic rebuild if fresh and we processed files
if fresh and atomic_rebuild_active and files_processed > 0:
    await self._finalize_atomic_rebuild()

    # Re-initialize backends after finalization
    # ... existing code ...

    # NOW build BM25 index (after backends re-initialized)
    await self._build_bm25_index()  # MOVED HERE
```

**Benefits**:
- Minimal code change
- Preserves current data source (vectors table)

**Drawbacks**:
- Still requires Phase 2 (embedding) to complete
- BM25 build happens **after** finalization, delaying search availability
- Doesn't fix the architectural coupling issue

---

## Testing Recommendations

### Test Case 1: Fresh Reindex with BM25 Build
```bash
# Delete existing index
rm -rf .mcp-vector-search/

# Run fresh reindex (force mode)
uv run mvs reindex -f

# Verify BM25 index was built
ls -la .mcp-vector-search/bm25_index.pkl

# Test keyword search
uv run mvs search "function calculate"
```

**Expected**: No warning, BM25 index created, keyword search works

### Test Case 2: Incremental Reindex
```bash
# Run incremental reindex
uv run mvs reindex

# Verify BM25 index exists
ls -la .mcp-vector-search/bm25_index.pkl
```

**Expected**: BM25 index created if missing, no warning

### Test Case 3: Two-Phase Workflow
```bash
# Run Phase 1 only (chunking)
uv run mvs index --phase chunk

# Verify BM25 index was built after Phase 1
ls -la .mcp-vector-search/bm25_index.pkl

# Run Phase 2 (embedding)
uv run mvs index --phase embed

# Test hybrid search (semantic + keyword)
uv run mvs search "authentication logic"
```

**Expected**: BM25 index available after Phase 1, before Phase 2

---

## Implementation Priority

**Priority**: HIGH

**Rationale**:
- Degrades search quality (keyword search disabled during fresh reindex)
- Affects all users running `mvs reindex -f` or `mvs index` on new projects
- Fix is straightforward (change data source from vectors to chunks table)
- Improves architecture by removing unnecessary coupling

**Estimated Effort**: 2-4 hours
- Code change: 30 minutes
- Testing: 1-2 hours (verify all call sites work)
- Documentation: 30 minutes (update architecture docs)
- Code review: 1 hour

---

## Related Files

- `src/mcp_vector_search/core/indexer.py` (main logic)
- `src/mcp_vector_search/core/vectors_backend.py` (vectors table)
- `src/mcp_vector_search/core/chunks_backend.py` (chunks table)
- `src/mcp_vector_search/core/bm25_backend.py` (BM25 index)

---

## References

- **Atomic Rebuild Implementation**: Lines 1527-1683 in `indexer.py`
- **BM25 Index Build**: Lines 3426-3495 in `indexer.py`
- **VectorsBackend Initialization**: Lines 236-283 in `vectors_backend.py`
- **ChunksBackend Schema**: Lines 38-56 in `chunks_backend.py`

---

## Conclusion

The warning occurs because BM25 index building attempts to read from the `vectors.lance` table (Phase 2 output) immediately after Phase 1 (chunking) completes, before Phase 2 (embedding) has run. The fix is to change BM25 index building to read from the `chunks.lance` table instead, which contains all necessary data and is available after Phase 1. This removes the architectural coupling between keyword search and embedding, improving both correctness and performance.
