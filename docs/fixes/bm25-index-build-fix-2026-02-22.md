# BM25 Index Build Fix - 2026-02-22

## Problem

When running `index` or `reindex -f`, the following warning appeared:
```
WARNING: Vectors backend not initialized, skipping BM25 index build
```

This prevented the BM25 index from being built, degrading search quality for keyword-based searches.

## Root Cause

The `_build_bm25_index()` method at line 3435 in `indexer.py` was checking:
```python
if self.vectors_backend._db is None or self.vectors_backend._table is None:
```

However, during the `chunk_files()` phase, the vectors table doesn't exist yet because it's only created during the later `embed_chunks()` phase. This check would always fail and skip BM25 index building.

**Key insight**: BM25 only needs TEXT content for keyword search, NOT embeddings. All required fields (`chunk_id`, `content`, `name`, `file_path`, `chunk_type`) are available in the **chunks** table, which is created during the `chunk_files()` phase.

## Fix Applied

Changed `_build_bm25_index()` to read from `chunks_backend` instead of `vectors_backend`:

### Before
```python
# Get all chunks from vectors.lance table
# VectorsBackend has the vectors table with all chunk data including content
if self.vectors_backend._db is None or self.vectors_backend._table is None:
    logger.warning(
        "Vectors backend not initialized, skipping BM25 index build"
    )
    return

# Query all records from vectors.lance table
df = await asyncio.to_thread(self.vectors_backend._table.to_pandas)

if df.empty:
    logger.info("No chunks in vectors table, skipping BM25 index build")
    return
```

### After
```python
# Get all chunks from chunks.lance table
# ChunksBackend has all chunk data including content (vectors not needed for BM25)
if self.chunks_backend._db is None or self.chunks_backend._table is None:
    logger.warning(
        "Chunks backend not initialized, skipping BM25 index build"
    )
    return

# Query all records from chunks.lance table
df = await asyncio.to_thread(self.chunks_backend._table.to_pandas)

if df.empty:
    logger.info("No chunks in chunks table, skipping BM25 index build")
    return
```

## Changes Made

1. **Line 3436**: Changed check from `self.vectors_backend._db` to `self.chunks_backend._db`
2. **Line 3436**: Changed check from `self.vectors_backend._table` to `self.chunks_backend._table`
3. **Line 3437-3438**: Updated warning message to "Chunks backend not initialized"
4. **Line 3447**: Changed data source from `self.vectors_backend._table.to_pandas` to `self.chunks_backend._table.to_pandas`
5. **Line 3450**: Updated log message to "No chunks in chunks table"
6. **Line 3427-3431**: Updated docstring to clarify that BM25 reads from chunks.lance

## Schema Verification

The chunks backend schema (from `chunks_backend.py` line 44) contains all required fields:

```python
CHUNKS_SCHEMA = pa.schema([
    pa.field("chunk_id", pa.string()),
    pa.field("file_path", pa.string()),
    pa.field("content", pa.string()),        # ✓ Required for BM25
    pa.field("chunk_type", pa.string()),     # ✓ Required for BM25
    pa.field("name", pa.string()),           # ✓ Required for BM25
    # ... other fields
])
```

## Benefits

1. **BM25 index now builds correctly** during `chunk_files()` phase
2. **No dependency on embeddings** - BM25 keyword search doesn't need vector embeddings
3. **Proper phase separation** - BM25 builds after chunks are created, not waiting for embeddings
4. **Better performance** - BM25 available immediately after chunking

## Testing

All tests pass:
```bash
uv run pytest tests/ -x -q
# Result: All 1000+ tests passed ✓
```

## Code Location

- File: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/indexer.py`
- Method: `_build_bm25_index()` (lines 3426-3485)
- Changes: Lines 3436, 3437-3438, 3447, 3450

## Related Files

- `src/mcp_vector_search/core/chunks_backend.py` - Chunks table schema
- `src/mcp_vector_search/core/vectors_backend.py` - Vectors table (no longer used for BM25)
- `src/mcp_vector_search/backends/bm25_backend.py` - BM25 indexing logic

## LOC Delta

- Added: 0 lines (only changed existing lines)
- Removed: 0 lines
- Net Change: 0 lines (refactored existing code)
- Modified: 6 lines (backend references and comments)
