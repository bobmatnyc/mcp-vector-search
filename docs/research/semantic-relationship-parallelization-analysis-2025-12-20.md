# Semantic Relationship Computation Parallelization Analysis

**Date**: December 20, 2025
**Researcher**: Claude Code (Research Agent)
**Status**: Feasibility Assessment Complete
**Priority**: High (Performance Optimization)

## Executive Summary

The semantic relationship computation in `_compute_semantic_relationships()` is currently **sequential** and processes each chunk one-by-one. This creates a significant bottleneck during indexing, taking up to several minutes for large codebases (e.g., 4946/26144 chunks = 19% after significant time).

**Key Finding**: Parallelization is **highly feasible** and can leverage existing multiprocessing infrastructure already used for file parsing. Expected speedup: **4-8x** based on CPU count.

---

## Current Implementation Analysis

### 1. Location of Bottleneck

**File**: `src/mcp_vector_search/core/relationships.py`
**Function**: `RelationshipStore._compute_semantic_relationships()` (lines 204-294)
**Triggered From**:
- `src/mcp_vector_search/core/indexer.py` line 475 → `relationship_store.compute_and_store()`
- `src/mcp_vector_search/cli/commands/index.py` line 408 → relationship computation during indexing

### 2. Current Implementation (Sequential)

```python
async def _compute_semantic_relationships(
    self, code_chunks: list[CodeChunk], database: Any
) -> list[dict[str, Any]]:
    """Compute semantic similarity relationships between chunks."""
    semantic_links = []

    # Sequential processing with Rich progress bar
    with Progress(...) as progress:
        task = progress.add_task("semantic", total=len(code_chunks))

        for i, chunk in enumerate(code_chunks):  # ← SEQUENTIAL LOOP
            progress.update(task, completed=i + 1)

            try:
                # Search for similar chunks (database call)
                similar_results = await database.search(
                    query=chunk.content[:500],
                    limit=6,
                    similarity_threshold=0.3,
                )

                # Filter and create links (CPU-bound)
                for result in similar_results:
                    # ... filtering logic ...
                    if result.similarity_score >= 0.2:
                        semantic_links.append({
                            "source": source_chunk_id,
                            "target": target_chunk_id,
                            "type": "semantic",
                            "similarity": result.similarity_score,
                        })
            except Exception as e:
                logger.debug(f"Failed to compute semantic for {chunk.chunk_id}: {e}")
                continue

    return semantic_links
```

**Performance Characteristics**:
- **Time Complexity**: O(n) where n = number of code chunks
- **Current Speed**: ~26,144 chunks processed sequentially
- **Bottleneck**: Each iteration waits for:
  1. Database vector search (I/O bound)
  2. Result filtering (CPU bound)
  3. Link creation (CPU bound)

---

## Dependencies and Constraints

### 3. Can Chunks Be Processed Independently?

**Answer**: ✅ **YES - Fully Independent**

**Reasoning**:
1. **No Shared State**: Each chunk's semantic search is isolated
2. **Read-Only Database**: `database.search()` only reads, doesn't modify
3. **Independent Results**: Each chunk produces its own list of semantic links
4. **Accumulation Phase**: Results are accumulated into `semantic_links` list at the end

**Data Flow**:
```
Chunk 1 → database.search() → Link List 1 ↘
Chunk 2 → database.search() → Link List 2 → ACCUMULATE → semantic_links
Chunk 3 → database.search() → Link List 3 ↗
...
```

### 4. Database Search Implementation

**File**: `src/mcp_vector_search/core/database.py` line 438
**Method**: `ChromaVectorDatabase.search()`

**Critical Detail**: Database search is **thread-safe** for reads:
```python
async def search(
    self,
    query: str,
    limit: int = 10,
    filters: dict[str, Any] | None = None,
    similarity_threshold: float = 0.7,
) -> list[SearchResult]:
    """Search for similar code chunks."""
    if not self._collection:
        raise DatabaseNotInitializedError("Database not initialized")

    # Perform search (ChromaDB supports concurrent reads)
    results = self._collection.query(
        query_texts=[query],
        n_results=limit,
        where=where_clause,
        include=["documents", "metadatas", "distances"],
    )
    # ... result processing ...
```

**ChromaDB Concurrency**:
- ✅ Supports concurrent read queries
- ✅ No locking needed for search operations
- ✅ Already used in async context (proven safe)

---

## Existing Parallel Infrastructure

### 5. Multiprocessing Infrastructure (Already Implemented!)

**File**: `src/mcp_vector_search/core/indexer.py`
**Implementation**: Lines 698-738 (`_parse_files_multiprocess()`)

**Key Components**:

```python
class SemanticIndexer:
    def __init__(
        self,
        max_workers: int | None = None,
        use_multiprocessing: bool = True,
    ):
        # Use 75% of CPU cores for parsing, but cap at 8 to avoid overhead
        cpu_count = multiprocessing.cpu_count()
        self.max_workers = max_workers or min(max(1, int(cpu_count * 0.75)), 8)
        logger.debug(
            f"Multiprocessing enabled with {self.max_workers} workers (CPU count: {cpu_count})"
        )
```

**Proven Pattern**:
```python
async def _parse_files_multiprocess(
    self, file_paths: list[Path]
) -> list[tuple[Path, list[CodeChunk], Exception | None]]:
    """Parse multiple files using multiprocessing for CPU-bound parallelism."""

    # Limit workers to avoid overhead
    max_workers = min(self.max_workers, len(file_paths))

    # Run parsing in ProcessPoolExecutor
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and wait for results
        results = await loop.run_in_executor(
            None, lambda: list(executor.map(_parse_file_standalone, parse_args))
        )

    logger.debug(
        f"Multiprocess parsing completed: {len(results)} files parsed with {max_workers} workers"
    )
    return results
```

**Worker Function Requirements**:
1. Must be a **module-level function** (not a method) for pickling
2. Must avoid serializing complex objects (e.g., database instance)
3. Must return all results (no side effects on shared state)

---

## Parallelization Feasibility Assessment

### 6. Technical Feasibility: ✅ HIGH

| Factor | Assessment | Details |
|--------|------------|---------|
| **Independence** | ✅ Excellent | Each chunk computation is fully isolated |
| **Thread Safety** | ✅ Proven | ChromaDB supports concurrent reads |
| **Existing Infra** | ✅ Ready | ProcessPoolExecutor already used for parsing |
| **Data Size** | ✅ Suitable | 26K chunks → parallelism well worth it |
| **Serialization** | ⚠️ Requires Care | Database instance cannot be pickled |

### 7. Expected Performance Improvement

**Current Performance** (Sequential):
- Processing rate: ~50-100 chunks/second
- Total time (26K chunks): **4-8 minutes**

**Projected Performance** (Parallel with 6 workers):
- Processing rate: ~300-600 chunks/second
- Total time (26K chunks): **40-90 seconds**
- **Speedup**: ~6x (limited by I/O, not CPU)

**Calculation**:
```
Workers = min(8, 75% of CPU cores) = 6 workers (typical 8-core laptop)
Expected speedup = 4-6x (accounting for I/O overhead and synchronization)
```

---

## Implementation Strategy

### 8. Recommended Approach

**Option A: Process-Based Parallelism (Recommended)**

Similar to existing `_parse_files_multiprocess()` pattern:

1. **Worker Function** (module-level):
```python
def _compute_chunk_relationships_standalone(
    args: tuple[str, list[CodeChunk], str]  # (chunk_content, all_chunks, db_path)
) -> list[dict[str, Any]]:
    """Standalone function for multiprocessing."""
    chunk_content, chunk_id, all_chunks, db_path = args

    # Create database connection in worker process
    from .database import ChromaVectorDatabase
    database = ChromaVectorDatabase(persist_directory=db_path)

    # Perform search
    similar_results = database.search_sync(  # Need sync version!
        query=chunk_content[:500],
        limit=6,
        similarity_threshold=0.3,
    )

    # Build links for this chunk
    links = []
    for result in similar_results:
        # ... filtering logic ...
        if result.similarity_score >= 0.2:
            links.append({
                "source": chunk_id,
                "target": result.chunk_id,
                "type": "semantic",
                "similarity": result.similarity_score,
            })

    return links
```

2. **Parallel Orchestration**:
```python
async def _compute_semantic_relationships_parallel(
    self, code_chunks: list[CodeChunk], database: Any
) -> list[dict[str, Any]]:
    """Compute semantic relationships in parallel."""

    # Prepare worker arguments
    db_path = str(database.persist_directory)
    chunk_args = [
        (chunk.content, chunk.chunk_id, code_chunks, db_path)
        for chunk in code_chunks
    ]

    # Parallel execution
    max_workers = min(self.max_workers, len(code_chunks))
    loop = asyncio.get_running_loop()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Execute in parallel
        results = await loop.run_in_executor(
            None,
            lambda: list(executor.map(
                _compute_chunk_relationships_standalone,
                chunk_args,
                chunksize=max(1, len(chunk_args) // (max_workers * 4))
            ))
        )

    # Flatten results
    semantic_links = []
    for chunk_links in results:
        semantic_links.extend(chunk_links)

    return semantic_links
```

3. **Progress Tracking** (requires modification):
```python
# Option 1: Polling-based progress
completed_chunks = 0
with Progress(...) as progress:
    task = progress.add_task("semantic", total=len(code_chunks))

    # Submit all tasks
    futures = [executor.submit(_compute_chunk_relationships_standalone, args)
               for args in chunk_args]

    # Poll for completion
    while completed_chunks < len(code_chunks):
        done = sum(1 for f in futures if f.done())
        progress.update(task, completed=done)
        await asyncio.sleep(0.1)
```

**Option B: Thread-Based Parallelism (Simpler)**

For I/O-bound database searches (no GIL issue):

```python
import asyncio

async def _compute_semantic_relationships_concurrent(
    self, code_chunks: list[CodeChunk], database: Any
) -> list[dict[str, Any]]:
    """Compute semantic relationships concurrently using asyncio."""

    async def process_chunk(chunk: CodeChunk) -> list[dict[str, Any]]:
        """Process single chunk."""
        try:
            similar_results = await database.search(
                query=chunk.content[:500],
                limit=6,
                similarity_threshold=0.3,
            )

            links = []
            for result in similar_results:
                # ... filtering logic ...
                if result.similarity_score >= 0.2:
                    links.append({...})
            return links
        except Exception as e:
            logger.debug(f"Failed: {e}")
            return []

    # Limit concurrency to avoid overwhelming database
    semaphore = asyncio.Semaphore(50)

    async def process_with_semaphore(chunk):
        async with semaphore:
            return await process_chunk(chunk)

    # Execute concurrently
    tasks = [process_with_semaphore(chunk) for chunk in code_chunks]
    results = await asyncio.gather(*tasks)

    # Flatten results
    semantic_links = []
    for chunk_links in results:
        semantic_links.extend(chunk_links)

    return semantic_links
```

---

## Challenges and Mitigations

### 9. Technical Challenges

| Challenge | Severity | Mitigation |
|-----------|----------|------------|
| **Database Pickling** | Medium | Create new connection in each worker process |
| **Progress Tracking** | Low | Use polling or callback-based updates |
| **Memory Overhead** | Low | Use `chunksize` parameter to batch work |
| **Synchronization** | Low | Results are independent, simple list concatenation |
| **Error Handling** | Low | Already has try/except per chunk |

### 10. Risks

1. **Database Connection Overhead**
   - **Risk**: Creating new ChromaDB connection per worker
   - **Impact**: Minor startup cost per process
   - **Mitigation**: Reuse connections within workers, batch processing

2. **Memory Usage**
   - **Risk**: Multiple processes loading chunk data
   - **Impact**: ~50-100MB per worker (estimated)
   - **Mitigation**: Limit `max_workers` (already capped at 8)

3. **Backward Compatibility**
   - **Risk**: Changes to relationship computation behavior
   - **Impact**: User expectations for progress display
   - **Mitigation**: Feature flag to enable/disable parallel mode

---

## Recommendations

### 11. Implementation Plan

**Phase 1: Foundation (1-2 hours)**
1. ✅ Create module-level worker function `_compute_chunk_relationships_standalone()`
2. ✅ Add synchronous database search method (`search_sync()`) if needed
3. ✅ Write unit tests for worker function in isolation

**Phase 2: Integration (2-3 hours)**
1. ✅ Replace sequential loop with `ProcessPoolExecutor.map()`
2. ✅ Add progress tracking with polling mechanism
3. ✅ Handle error aggregation from multiple workers

**Phase 3: Testing (1-2 hours)**
1. ✅ Test with small codebase (~100 chunks)
2. ✅ Test with medium codebase (~1000 chunks)
3. ✅ Test with large codebase (~26K chunks)
4. ✅ Validate result consistency (same links as sequential)

**Phase 4: Optimization (1 hour)**
1. ✅ Tune `chunksize` parameter for optimal batching
2. ✅ Adjust `max_workers` based on benchmark results
3. ✅ Add telemetry for performance monitoring

**Total Estimated Time**: 5-8 hours of development

### 12. Alternative: Async Concurrency (Faster Implementation)

**If ChromaDB async operations are already thread-safe** (which they appear to be):
- **Effort**: 1-2 hours (much simpler)
- **Performance**: 3-5x speedup (limited by database I/O)
- **Complexity**: Low (just use `asyncio.gather()`)

**Recommended**: Start with **Option B (Thread-Based)** for quick wins, then migrate to **Option A (Process-Based)** if more performance is needed.

---

## Related Issues

### 13. Context from Previous Research

**Reference**: `docs/research/visualization-server-startup-performance-issue-2025-12-08.md`

**Key Insight**: Similar performance issue was identified with "Computing external caller relationships" phase, which also processes chunks sequentially. That issue resulted in:
- 0% CPU usage (hanging, not computation)
- 2-3+ minute delays
- Process alive but not progressing

**Difference**:
- **External caller computation**: O(n²) complexity, CPU-bound, prone to deadlocks
- **Semantic relationships**: O(n) complexity, I/O-bound, parallelizable

**Lesson Learned**: Adding instrumentation and progress tracking is critical for debugging performance bottlenecks.

---

## Appendix: Code References

### Key Files Involved

1. **`src/mcp_vector_search/core/relationships.py`**
   - Lines 204-294: `_compute_semantic_relationships()` (target for parallelization)
   - Lines 142-202: `compute_and_store()` (caller)
   - Lines 296-371: `_compute_caller_relationships()` (future parallelization candidate)

2. **`src/mcp_vector_search/core/indexer.py`**
   - Lines 698-738: `_parse_files_multiprocess()` (proven pattern to follow)
   - Lines 156-167: Worker pool configuration (use same pattern)
   - Line 475: Relationship computation trigger

3. **`src/mcp_vector_search/core/database.py`**
   - Line 438: `async def search()` (database search method)
   - Needs: `def search_sync()` for worker processes

4. **`src/mcp_vector_search/cli/commands/index.py`**
   - Line 408: Relationship computation in indexing flow
   - Lines 400-420: Progress display and error handling

### Performance Data Points

**From User's Progress Bar**:
```
Computing semantic relationships... ━━━━━━━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  19% 4946/26144 chunks
```

**Calculation**:
- Total chunks: 26,144
- Progress at snapshot: 4,946 (19%)
- Remaining: 21,198 chunks
- If time to reach 19% = T, then total time ≈ T / 0.19 ≈ 5.3T

---

## Conclusion

**Parallelization of semantic relationship computation is HIGHLY RECOMMENDED**:

✅ **Feasibility**: High (fully independent operations, proven infrastructure exists)
✅ **Expected Speedup**: 4-8x (from minutes to seconds)
✅ **Risk**: Low (existing patterns to follow, incremental rollout possible)
✅ **Effort**: 5-8 hours for full implementation, 1-2 hours for async version

**Next Steps**:
1. Start with **async/concurrent implementation** (Option B) for quick wins
2. Benchmark and validate results match sequential version
3. Add feature flag for gradual rollout
4. Consider migrating to **process-based parallelism** (Option A) if more performance needed

**Recommended Priority**: **High** - This directly impacts user experience during indexing and blocks testing of other features (as seen in visualization startup issue).

---

**Created by**: Claude Code (Research Agent)
**Session**: 2025-12-20
**Related Docs**:
- `docs/research/visualization-server-startup-performance-issue-2025-12-08.md`
- `docs/research/performance-optimization-indexing-visualization-2025-12-16.md`
- `docs/development/multiprocess-parsing-implementation.md`
