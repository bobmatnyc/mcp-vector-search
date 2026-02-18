# Root Cause Analysis: Empty Search Index

**Date:** 2025-02-18
**Issue:** `mcp-vector-search search` returns no results after indexing
**Symptom:** 16,590 chunks indexed but searches fail with "No results found"

---

## Executive Summary

The search index is empty because **Phase 2 (embedding generation) is writing to the wrong database table**. The indexer is still writing embeddings to `code_search.lance` (old architecture), but the search engine now looks for vectors in `vectors.lance` (new two-phase architecture). This architectural mismatch causes `code_search.lance` to never be created, leaving search with no vectors to query.

---

## Investigation Findings

### 1. Database State Analysis

```
.mcp-vector-search/lance/
├── chunks.lance/          ✓ EXISTS (16,590 rows)
│   └── No vector column   ✓ EXPECTED (Phase 1: parsed chunks only)
├── vectors.lance/         ✗ MISSING (should have 16,590 vectors)
└── code_search.lance/     ✗ MISSING (old architecture, shouldn't exist)
```

**Key Observation:**
- `chunks.lance` has all parsed chunks with `embedding_status = "pending"`
- Neither `vectors.lance` nor `code_search.lance` exists
- Search expects `vectors.lance` but indexer writes to `code_search.lance`

### 2. Architecture Mismatch

#### **Phase 1 (Parsing) - Working Correctly**
```python
# src/mcp_vector_search/core/indexer.py:516
await self.chunks_backend.add_chunks(chunk_dicts, file_hash)
# ✓ Writes to chunks.lance successfully
```

#### **Phase 2 (Embedding) - Writing to Wrong Table**
```python
# src/mcp_vector_search/core/indexer.py:1360
await self.database.add_chunks(all_chunks, metrics=all_metrics)
# ✗ PROBLEM: self.database writes to code_search.lance (old architecture)
```

The indexer calls `self.database.add_chunks()` which maps to:
- **File:** `lancedb_backend.py`
- **Method:** `LanceVectorDatabase.add_chunks()`
- **Target Table:** `code_search.lance`

But the search engine now uses:
- **File:** `vectors_backend.py`
- **Class:** `VectorsBackend`
- **Target Table:** `vectors.lance`

### 3. Data Flow Trace

**Current (Broken) Flow:**
```
Indexing Phase:
  chunks_backend.add_chunks()  → chunks.lance ✓
  database.add_chunks()        → code_search.lance ✗ (wrong table)

Search Phase:
  vectors_backend.search()     → vectors.lance ✗ (empty)
```

**Expected (Fixed) Flow:**
```
Indexing Phase:
  chunks_backend.add_chunks()  → chunks.lance ✓
  vectors_backend.add_vectors() → vectors.lance ✓

Search Phase:
  vectors_backend.search()     → vectors.lance ✓
```

### 4. Evidence from Code

#### Indexer Initialization (Line 221-222)
```python
self.chunks_backend = ChunksBackend(index_path)
self.vectors_backend = VectorsBackend(index_path)  # Created but NOT used
```

The `vectors_backend` is instantiated but **never called** during Phase 2.

#### Search Engine Detection (search.py:494-525)
```python
def _check_vectors_backend(self) -> None:
    """Check if VectorsBackend is available for two-phase architecture."""
    vectors_path = index_path / "lance" / "vectors.lance"
    if vectors_path.exists() and vectors_path.is_dir():
        vectors_backend = VectorsBackend(index_path / "lance")
        self._vectors_backend = vectors_backend
```

Search correctly looks for `vectors.lance`, but indexing never creates it.

#### Phase 2 Embedding Code (indexer.py:580-658)
```python
# PROBLEM: This code uses vectors_backend correctly in _embed_phase2
await self.vectors_backend.add_vectors(chunks_with_vectors)

# BUT: The main index_batch_parallel method bypasses this and uses:
await self.database.add_chunks(all_chunks, metrics=all_metrics)
```

There are **two code paths**:
1. **Correct path:** `_embed_phase2()` → uses `vectors_backend.add_vectors()` ✓
2. **Broken path:** `index_batch_parallel()` → uses `database.add_chunks()` ✗

The broken path is the default for full indexing.

---

## Root Cause

**The indexer has a two-phase architecture with `chunks_backend` and `vectors_backend`, but Phase 2 is still using the old monolithic `self.database` (LanceVectorDatabase) which writes to `code_search.lance` instead of `vectors.lance`.**

Specific problem locations:
- **Line 1360:** `await self.database.add_chunks(all_chunks, metrics=all_metrics)`
- **Line 1485:** `await self.database.add_chunks(chunks_with_hierarchy, metrics=chunk_metrics)`
- **Line 1985:** `await self.database.add_chunks(all_chunks, metrics=all_metrics)`

All three locations should use `vectors_backend.add_vectors()` instead.

---

## Fix Recommendation

### Option 1: Use Correct Backend (Preferred)

Replace all `await self.database.add_chunks()` calls with:

```python
# Phase 2: Generate embeddings and store to vectors.lance
# Convert CodeChunk objects to dicts with vectors
chunks_with_vectors = []
for chunk in all_chunks:
    # Generate embedding
    embedding = self.database.embedding_function([chunk.content])[0]

    chunk_with_vec = {
        "chunk_id": chunk.chunk_id,
        "vector": embedding,
        "file_path": str(chunk.file_path),
        "content": chunk.content,
        "language": chunk.language,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "chunk_type": chunk.chunk_type,
        "name": chunk.function_name or chunk.class_name or "",
        "hierarchy_path": "",  # Can be constructed from chunk metadata
    }
    chunks_with_vectors.append(chunk_with_vec)

# Store to vectors.lance
await self.vectors_backend.add_vectors(chunks_with_vectors)
```

### Option 2: Simplify Architecture (Alternative)

If two-phase architecture is not needed:
1. Remove `vectors_backend` entirely
2. Keep using `database.add_chunks()` to write to `code_search.lance`
3. Update search engine to use `code_search.lance` instead of `vectors.lance`

This reverts to the previous monolithic design.

---

## Impact Analysis

### Why This Wasn't Caught
1. **No integration test** verifying search after indexing
2. **Status reporting** only checks `chunks.lance` count, not search readiness
3. **Silent failure:** Indexing "succeeds" but writes to unused table

### Affected Functionality
- ✗ All semantic search queries (returns empty)
- ✓ File parsing and chunking (works correctly)
- ✓ Status reporting (reports 16,590 chunks)
- ✗ Vector database queries (no vectors to query)

---

## Recommended Actions

### Immediate Fix (1 hour)
1. **Update indexer.py lines 1360, 1485, 1985:**
   - Replace `await self.database.add_chunks()` with `await self.vectors_backend.add_vectors()`
   - Add embedding generation before vector storage
   - Ensure chunk-to-vector format conversion

### Verification (15 minutes)
1. Delete `.mcp-vector-search/lance/` directory
2. Run `mcp-vector-search index`
3. Verify `vectors.lance/` directory exists
4. Run `mcp-vector-search search "test query"`
5. Confirm results are returned

### Long-term Improvements
1. Add integration test: index → search → verify results
2. Add health check: verify both chunks and vectors exist
3. Update status command to report vector table separately
4. Add migration script to convert existing indexes

---

## Files Requiring Changes

### Primary Fix
- **src/mcp_vector_search/core/indexer.py**
  - Lines 1360, 1485, 1985
  - Replace `database.add_chunks()` with `vectors_backend.add_vectors()`

### Testing
- **tests/test_indexer.py** (if exists)
  - Add integration test for full index → search flow

### Documentation
- **README.md** (if affected)
  - Update architecture description if two-phase design is emphasized

---

## Conclusion

The root cause is an **incomplete migration from monolithic `code_search.lance` architecture to two-phase `chunks.lance` + `vectors.lance` architecture**. Phase 1 (parsing) correctly uses the new `chunks_backend`, but Phase 2 (embedding) still uses the old `database` object, causing embeddings to be written to a table that search no longer queries.

**Fix:** Update Phase 2 embedding code to use `vectors_backend.add_vectors()` instead of `database.add_chunks()`.

**Priority:** Critical - all search functionality is broken until this is fixed.
