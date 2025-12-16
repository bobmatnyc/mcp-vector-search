# Batch Embedding Implementation Analysis for Issue #59

**Date:** 2025-12-16
**Issue:** #59 - Implement batch embedding generation across files
**Status:** Analysis Complete
**Working Directory:** /Users/masa/Projects/mcp-vector-search

---

## Executive Summary

The current indexer processes files sequentially and generates embeddings per-file during `database.add_chunks()` calls. This creates a significant bottleneck as embeddings are generated in small batches (per-file) rather than leveraging the existing `BatchEmbeddingProcessor` (batch_size=32) across multiple files.

**Key Finding:** The bottleneck is NOT in the embedding generation itself (which already uses batching via `BatchEmbeddingProcessor`), but in the **per-file database insertion pattern** that prevents batching chunks across multiple files.

---

## Current Implementation Analysis

### 1. Indexing Flow (indexer.py)

**File Processing:**
```python
# Line 331-347: index_project() batches files
for i in range(0, len(files_to_index), self.batch_size):  # batch_size default=10
    batch = files_to_index[i : i + self.batch_size]
    batch_results = await self._process_file_batch(batch, force_reindex)
```

**Problem:** `_process_file_batch()` processes files in parallel but calls `index_file()` for each file individually:

```python
# Line 442-472: _process_file_batch()
async def _process_file_batch(self, file_paths: list[Path], force_reindex: bool = False) -> list[bool]:
    tasks = []
    for file_path in file_paths:
        task = asyncio.create_task(self._index_file_safe(file_path, force_reindex))
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    # ...
```

**Per-File Database Insertion:**
```python
# Line 624: index_file() - Called once per file
await self.database.add_chunks(chunks_with_hierarchy, metrics=chunk_metrics)
```

**Result:** Each file's chunks are sent to the database independently, preventing cross-file batching of embeddings.

---

### 2. Database Add Chunks Flow (database.py)

**ChromaVectorDatabase.add_chunks()** (Line 347-436):

```python
async def add_chunks(self, chunks: list[CodeChunk], metrics: dict[str, Any] | None = None) -> None:
    # Prepare data
    documents = []
    metadatas = []
    ids = []

    for chunk in chunks:
        documents.append(chunk.content)  # Original content
        metadatas.append(metadata)
        ids.append(chunk.id)

    # Single ChromaDB call with all chunks from ONE file
    self._collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
```

**Embedding Generation in ChromaDB:**
When `collection.add()` is called, ChromaDB internally:
1. Receives the list of `documents`
2. Calls the `embedding_function` (provided at collection creation)
3. The `embedding_function` is `CodeBERTEmbeddingFunction.__call__()`

**Actual Batching Happens Here:**
```python
# embeddings.py Line 175-182: CodeBERTEmbeddingFunction.__call__()
def __call__(self, input: list[str]) -> list[list[float]]:
    try:
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()
    except Exception as e:
        raise EmbeddingError(f"Failed to generate embeddings: {e}") from e
```

**SentenceTransformer.encode()** internally batches the input, typically using batch_size=32.

---

### 3. BatchEmbeddingProcessor (embeddings.py)

**Current Implementation** (Line 185-273):
- Has `batch_size=32` parameter
- Implements `process_batch()` with cache support
- **NOT CURRENTLY USED** by the indexer

```python
class BatchEmbeddingProcessor:
    def __init__(
        self,
        embedding_function: CodeBERTEmbeddingFunction,
        cache: EmbeddingCache | None = None,
        batch_size: int = 32,
    ):
        self.embedding_function = embedding_function
        self.cache = cache
        self.batch_size = batch_size

    async def process_batch(self, contents: list[str]) -> list[list[float]]:
        # Check cache
        # Generate embeddings in batches
        for i in range(0, len(uncached_contents), self.batch_size):
            batch = uncached_contents[i : i + self.batch_size]
            batch_embeddings = self.embedding_function(batch)
            new_embeddings.extend(batch_embeddings)
```

**Key Insight:** This class exists but is bypassed because embeddings are generated implicitly by ChromaDB during `collection.add()`.

---

## Current Bottleneck Identification

### The Real Problem: Per-File Database Calls

**Current Pattern:**
```
File 1 (5 chunks)  → database.add_chunks(5 chunks)  → ChromaDB.add() → embedding(5 texts)
File 2 (8 chunks)  → database.add_chunks(8 chunks)  → ChromaDB.add() → embedding(8 texts)
File 3 (3 chunks)  → database.add_chunks(3 chunks)  → ChromaDB.add() → embedding(3 texts)
File 4 (12 chunks) → database.add_chunks(12 chunks) → ChromaDB.add() → embedding(12 texts)
```

**Issues:**
1. **Small Batch Sizes:** Each file has variable chunk counts (typically 5-15 chunks per file)
2. **Overhead:** Multiple ChromaDB `add()` calls with transaction overhead
3. **Suboptimal GPU Utilization:** Small batches don't fully utilize GPU/CPU vectorization
4. **Cache Fragmentation:** Caching happens per-file, missing cross-file duplicate detection

**Optimal Pattern:**
```
Files 1-10 (78 chunks total) → database.add_chunks(78 chunks) → ChromaDB.add() → embedding(78 texts in batches of 32)
```

---

## Proposed Solution

### Approach: Accumulate Chunks Across Files Before Database Insertion

**Goal:** Collect chunks from multiple files (e.g., 5-10 files) before calling `database.add_chunks()` once.

### Implementation Strategy

#### Option 1: Batch-Level Chunk Accumulation (RECOMMENDED)

**Modify:** `SemanticIndexer._process_file_batch()` and `index_files_with_progress()`

**Changes:**

```python
# indexer.py - NEW METHOD
async def _process_file_batch_with_chunk_accumulation(
    self,
    file_paths: list[Path],
    force_reindex: bool = False
) -> list[bool]:
    """Process files and accumulate chunks for batch embedding."""

    all_chunks = []
    all_metrics = {}
    file_to_chunks_map = {}  # Track which chunks belong to which file
    success_flags = []

    # Parse all files first (keep existing parallel parsing)
    parse_tasks = []
    for file_path in file_paths:
        task = asyncio.create_task(self._parse_and_prepare_file(file_path, force_reindex))
        parse_tasks.append(task)

    parse_results = await asyncio.gather(*parse_tasks, return_exceptions=True)

    # Accumulate chunks from all successfully parsed files
    for i, (file_path, result) in enumerate(zip(file_paths, parse_results)):
        if isinstance(result, Exception):
            success_flags.append(False)
            continue

        chunks, metrics = result
        if chunks:
            start_idx = len(all_chunks)
            all_chunks.extend(chunks)
            end_idx = len(all_chunks)
            file_to_chunks_map[str(file_path)] = (start_idx, end_idx)

            # Merge metrics
            if metrics:
                all_metrics.update(metrics)

            success_flags.append(True)
        else:
            success_flags.append(True)  # Empty file is not an error

    # Single database insertion with ALL chunks from batch
    if all_chunks:
        logger.info(f"Batch inserting {len(all_chunks)} chunks from {len(file_paths)} files")
        await self.database.add_chunks(all_chunks, metrics=all_metrics)

    return success_flags


async def _parse_and_prepare_file(
    self,
    file_path: Path,
    force_reindex: bool = False
) -> tuple[list[CodeChunk], dict[str, Any] | None]:
    """Parse file and prepare chunks with metrics (no database insertion)."""

    # Delete existing chunks for this file
    await self.database.delete_by_file(file_path)

    # Parse file
    chunks = await self._parse_file(file_path)
    if not chunks:
        return ([], None)

    # Build hierarchy
    chunks_with_hierarchy = self._build_chunk_hierarchy(chunks)

    # Collect metrics
    chunk_metrics = None
    if self.collectors:
        try:
            source_code = file_path.read_bytes()
            language = EXTENSION_TO_LANGUAGE.get(file_path.suffix.lower(), "unknown")

            chunk_metrics = {}
            for chunk in chunks_with_hierarchy:
                metrics = self._collect_metrics(chunk, source_code, language)
                if metrics:
                    chunk_metrics[chunk.chunk_id] = metrics.to_metadata()
        except Exception as e:
            logger.warning(f"Failed to collect metrics for {file_path}: {e}")
            chunk_metrics = None

    return (chunks_with_hierarchy, chunk_metrics)
```

**Modify index_project():**
```python
# Line 331-347: Replace _process_file_batch with new method
batch_results = await self._process_file_batch_with_chunk_accumulation(
    batch, force_reindex
)
```

#### Option 2: Async Queue with Dynamic Batching (ADVANCED)

**Concept:** Use an async queue to accumulate chunks dynamically as files are parsed, with a separate consumer that batches chunks for database insertion.

**Pros:**
- Better memory management (chunks flow through queue)
- Dynamic batch sizing based on chunk accumulation rate
- Can adapt to varying file sizes

**Cons:**
- More complex implementation
- Harder to track file-to-chunk mapping
- Potential for queue overflow with large projects

**Recommendation:** Start with Option 1 (simpler, sufficient for most cases)

---

## Implementation Plan

### Phase 1: Refactor Chunk Collection (Breaking Change)

**Files to Modify:**
1. `src/mcp_vector_search/core/indexer.py`
   - Add `_process_file_batch_with_chunk_accumulation()`
   - Add `_parse_and_prepare_file()`
   - Modify `index_project()` to use new batch method
   - Modify `index_files_with_progress()` to use new pattern

**Key Changes:**
- **Remove:** Per-file `database.add_chunks()` calls in `index_file()`
- **Add:** Batch-level chunk accumulation and single `add_chunks()` call
- **Preserve:** File-to-chunk mapping for error handling and metadata updates

**Code Locations:**
- Line 442-472: `_process_file_batch()` - Replace entire method
- Line 554-636: `index_file()` - Split into parse + prepare (no DB insert)
- Line 1046-1149: `index_files_with_progress()` - Apply same pattern

### Phase 2: Maintain Chunk-to-File Mapping

**Challenge:** After batching, we need to track which chunks came from which file for:
1. Error handling (if a chunk fails, mark that file as failed)
2. Metadata updates (update file modification times after success)
3. Progress reporting (report chunks per file)

**Solution:** Use a mapping dictionary:
```python
file_to_chunks_map = {
    "/path/to/file1.py": (0, 5),     # Chunks 0-5 from file1
    "/path/to/file2.py": (5, 13),    # Chunks 5-13 from file2
    "/path/to/file3.py": (13, 16),   # Chunks 13-16 from file3
}
```

### Phase 3: Optimize Batch Size

**Research Findings:**
- SentenceTransformer optimal batch size: 32-64 items
- GPU memory permitting, larger batches (128-256) can be faster
- File batch size (default=10) may need adjustment

**Recommendations:**
1. Keep file batch size at 10 (good balance)
2. This typically yields 50-150 chunks per batch (10 files × 5-15 chunks/file)
3. SentenceTransformer will sub-batch internally at batch_size=32

**Configuration:**
- Add `chunk_batch_size` parameter to indexer (default=100-200 chunks)
- Dynamic batch sizing: accumulate until reaching chunk threshold OR file count threshold

---

## Risk Assessment

### Low Risk ✅
- Embeddings are still generated the same way (via ChromaDB's embedding_function)
- No changes to embedding model or generation logic
- Database `add_chunks()` already handles lists of chunks

### Medium Risk ⚠️
- **Memory Usage:** Accumulating 10 files worth of chunks before insertion
  - **Mitigation:** Monitor memory, keep file batch size reasonable (10-20 files max)
  - **Impact:** ~1-2MB per batch (manageable)

- **Error Handling Complexity:** Need to track which file each chunk came from
  - **Mitigation:** Use file_to_chunks_map for precise error attribution
  - **Impact:** More complex error reporting logic

- **Transaction Semantics:** Single `add_chunks()` call is atomic
  - **Current:** File-level atomicity (file succeeds or fails)
  - **New:** Batch-level atomicity (batch succeeds or fails)
  - **Mitigation:** Keep existing per-file `delete_by_file()` before parsing
  - **Impact:** If batch fails, some files may have been deleted but not re-indexed (recovery: reindex)

### High Risk 🔴
- **None Identified** - This is a performance optimization, not a semantic change

---

## Performance Impact Estimate

### Current Performance (Per-File Pattern)
- 100 files × 10 chunks/file = 1,000 chunks
- 100 database `add_chunks()` calls
- Embedding batches: highly variable (5-15 chunks per batch)
- ChromaDB overhead: 100 transactions

### Optimized Performance (Batch Pattern)
- 100 files in 10 batches (10 files/batch)
- 10 database `add_chunks()` calls
- Embedding batches: consistent (~100 chunks per batch, sub-batched at 32)
- ChromaDB overhead: 10 transactions

**Expected Speedup:**
- **3-5x** reduction in database transaction overhead
- **2-3x** improvement in embedding generation (better GPU/CPU utilization)
- **Overall:** 2-4x faster indexing for typical projects

**Benchmark Targets:**
- Small project (100 files): 10-15 seconds → 3-5 seconds
- Medium project (1,000 files): 2-3 minutes → 40-60 seconds
- Large project (10,000 files): 20-30 minutes → 6-10 minutes

---

## Next Steps

### 1. Implementation (1-2 days)
- [ ] Create feature branch: `feature/59-batch-embedding`
- [ ] Implement `_parse_and_prepare_file()` method
- [ ] Implement `_process_file_batch_with_chunk_accumulation()` method
- [ ] Update `index_project()` to use new batch method
- [ ] Update `index_files_with_progress()` similarly
- [ ] Maintain file-to-chunk mapping for error handling

### 2. Testing (1 day)
- [ ] Unit tests for new methods
- [ ] Integration tests with small/medium/large test projects
- [ ] Memory usage profiling
- [ ] Performance benchmarks (before/after comparison)

### 3. Documentation (0.5 days)
- [ ] Update CHANGELOG.md
- [ ] Document new batch_size implications
- [ ] Add performance tuning guide

### 4. Validation (0.5 days)
- [ ] Test with mcp-vector-search codebase (dogfooding)
- [ ] Monitor memory usage and error handling
- [ ] Verify embeddings are identical (hash comparison)

---

## Code Snippets

### Example: Simplified Batch Implementation

```python
async def _process_file_batch_with_chunk_accumulation(
    self,
    file_paths: list[Path],
    force_reindex: bool = False
) -> list[bool]:
    """Process files and accumulate chunks for batch embedding."""

    all_chunks = []
    all_metrics = {}
    success_flags = []

    # Parse all files in parallel
    for file_path in file_paths:
        try:
            # Parse file (no DB insertion)
            chunks, metrics = await self._parse_and_prepare_file(file_path, force_reindex)

            if chunks:
                all_chunks.extend(chunks)
                if metrics:
                    all_metrics.update(metrics)

            success_flags.append(True)
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            success_flags.append(False)

    # Single database insertion for entire batch
    if all_chunks:
        logger.info(f"Batch inserting {len(all_chunks)} chunks from {len(file_paths)} files")
        await self.database.add_chunks(all_chunks, metrics=all_metrics)

    return success_flags
```

---

## References

- **Issue:** #59 - Implement batch embedding generation across files
- **Files Analyzed:**
  - `src/mcp_vector_search/core/indexer.py` (1278 lines)
  - `src/mcp_vector_search/core/database.py` (1527 lines)
  - `src/mcp_vector_search/core/embeddings.py` (319 lines)

- **Key Classes:**
  - `SemanticIndexer` - Main indexing orchestrator
  - `ChromaVectorDatabase` - Database abstraction with ChromaDB backend
  - `BatchEmbeddingProcessor` - Batch embedding processor (currently unused)
  - `CodeBERTEmbeddingFunction` - Embedding function wrapper for ChromaDB

---

**Analysis Complete**
**Recommendation:** Proceed with Option 1 (Batch-Level Chunk Accumulation) for immediate 2-4x performance improvement with manageable complexity.
